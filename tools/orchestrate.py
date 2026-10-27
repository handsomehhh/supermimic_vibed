"""Run the three velocity-interface experiments one after another on all GPUs, with a monitor that
stops each run once it is done, diverged, or no longer improving.

  exp1  motion + velocity interfaces on the LAFAN1 locomotion clips (code check; one 8-GPU run)
  exp2  both interfaces on BONES-SEED (sonic_filtered) locomotion + LAFAN1 locomotion
  exp3  velocity interface on exp2's library if exp2 succeeded, else exp1's; motion interface on all of
        motion_data_used_g1

exp2 and exp3 run as one single-GPU training per GPU (seeds 0..NGPU-1, the paper's 1-GPU setting): data
parallel training on 8 GPUs diverged on BONES-SEED at every learning rate tried, single-GPU runs did not
(logs/diag, logs/diag2). Each run has its own evaluation watcher on its GPU.

Every POLL seconds the monitor reads train_log.csv / eval.csv of each run and writes STOP (the trainer
saves a checkpoint and exits) when
  - the run diverged: losses NaN/inf for 3 consecutive log rows, the simulator reward of the motion envs
    below half of its best for 3 consecutive log rows, or the latest evaluation's motion return 0.3 below
    its best (the trainer itself rolls back to snapshots first),
  - wall time exceeds max_hours,
  - after min_iters, none of the eval metrics improved for `patience` evaluations (converged),
  - after nogo_iters, motion fail rate and velocity fall rate are both still > 0.9 (no progress).
A run succeeds if its best evaluation meets the experiment's criteria; an experiment succeeds if at least
half of its runs do. If more than half of the runs crash, the chain stops; rerunning the orchestrator
resumes unfinished runs from their last.pt. Status: logs/orchestrator.json.

  nohup python tools/orchestrate.py > logs/orchestrator.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
NGPU = int(os.environ.get("NGPU", 8))
POLL = 300
PACKS = os.path.join(REPO, "data", "packs")
LAFAN = os.path.join(PACKS, "lafan_loco.pt")
SONIC = os.path.join(PACKS, "sonic_loco_p1.pt")  # --plausible: 68 clips with velocity glitches removed
ALL = os.path.join(PACKS, "all_g1.pt")  # --plausible: 4.4% of clips with impossible velocities removed
ALL_RAW = os.path.join(PACKS, "all_g1_raw.pt")  # every clip of motion_data_used_g1 that loads
STATUS = os.path.join(REPO, "logs", "orchestrator.json")

CRITERIA = {  # best evaluation must satisfy all
    "exp1": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.12), "vel_med_err": ("<=", 0.35), "vel_fall_rate": ("<=", 0.10)},
    "exp2": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
    "exp3": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
    "exp3_full": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
    "exp4": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
    "exp4_resume": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
}
STOP_RULES = {"min_iters": 20_000, "patience": 6, "nogo_iters": 30_000, "max_hours": 30.0, "max_iters": 100_000}
# stabilizer for the BPTT policy update (the in-trainer rollback is on by default)
TRAIN_ARGS = ["--pi_gn_skip", "5"]
IMPROVE = {"norm_return": +0.005, "fail_rate": -0.01, "vel_med_err": -0.01, "vel_fall_rate": -0.01}

status = json.load(open(STATUS)) if os.path.exists(STATUS) else {}


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_status():
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    json.dump(status, open(STATUS + ".tmp", "w"), indent=1)
    os.replace(STATUS + ".tmp", STATUS)


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def meets(row: dict, crit: dict) -> bool:
    for k, (op, thr) in crit.items():
        v = num(row.get(k))
        if math.isnan(v) or not (v >= thr if op == ">=" else v <= thr):
            return False
    return True


def best_row(rows: list[dict], crit: dict) -> dict | None:
    """Latest passing evaluation if any, else the one with the highest motion return minus velocity error."""
    passing = [r for r in rows if meets(r, crit)]
    if passing:
        return passing[-1]
    if not rows:
        return None
    return max(rows, key=lambda r: num(r["norm_return"]) - (num(r.get("vel_med_err")) if r.get("vel_med_err") else 0.0))


def plateaued(rows: list[dict], patience: int) -> bool:
    """No metric improved on its best-so-far during the last `patience` evaluations."""
    if len(rows) <= patience:
        return False
    head, tail = rows[:-patience], rows[-patience:]
    for k, d in IMPROVE.items():
        vals = [num(r.get(k)) for r in head if not math.isnan(num(r.get(k)))]
        new = [num(r.get(k)) for r in tail if not math.isnan(num(r.get(k)))]
        if not vals or not new:
            continue
        best = max(vals) if d > 0 else min(vals)
        if (max(new) - best >= d) if d > 0 else (min(new) - best <= d):
            return False
    return True


def stop_reason(tl: list[dict], ev: list[dict], hours: float) -> str | None:
    last = ev[-1] if ev else {}
    bad = [r for r in tl[-3:] if any(not math.isfinite(num(r[k])) for k in ("wm_loss", "pi_loss"))]
    sim = [num(r["sim_rew"]) for r in tl if num(r["iteration"]) >= 1000]
    nr = [num(r["norm_return"]) for r in ev]
    if len(tl) >= 3 and len(bad) == 3:
        return "diverged (NaN/inf losses)"
    if len(sim) > 10 and max(sim) > 0 and all(x < 0.5 * max(sim) for x in sim[-3:]):
        return "diverged (simulator reward collapsed)"
    if len(nr) >= 2 and max(nr) - nr[-1] > 0.3:
        return "diverged (evaluation return collapsed)"
    if hours > STOP_RULES["max_hours"]:
        return "max wall time"
    if ev and int(num(last["iteration"])) >= STOP_RULES["min_iters"] and plateaued(ev, STOP_RULES["patience"]):
        return "converged (no eval improvement)"
    if ev and int(num(last["iteration"])) >= STOP_RULES["nogo_iters"] and num(last["fail_rate"]) > 0.9 and num(last.get("vel_fall_rate", 1)) > 0.9:
        return "no progress"
    return None


class Run:
    """One training process (one or all GPUs) plus its evaluation watcher."""

    def __init__(self, name: str, log_dir: str, gpus: list[int], motion_packs: list[str], loco_packs: list[str], seed: int,
                 extra: list[str] = (), init_ckpt: str | None = None):  # fmt: skip
        self.name, self.log_dir, self.gpus = name, log_dir, gpus
        os.makedirs(log_dir, exist_ok=True)
        args = ["supermimic/train.py", "--log_dir", log_dir, "--motion_packs", *motion_packs, "--loco_packs", *loco_packs,
                "--iterations", str(STOP_RULES["max_iters"]), "--seed", str(seed), *TRAIN_ARGS, *extra]  # fmt: skip
        if os.path.exists(os.path.join(log_dir, "last.pt")):
            args += ["--resume", os.path.join(log_dir, "last.pt")]
        elif init_ckpt:  # continue another experiment's run (weights, optimizers, normalizers, iteration)
            args += ["--resume", init_ckpt]
        if os.path.exists(os.path.join(log_dir, "STOP")):
            os.remove(os.path.join(log_dir, "STOP"))
        cmd = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}", *args] if len(gpus) > 1 else [PY, *args]
        self.env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "PYTHONUNBUFFERED": "1"}
        self.train = subprocess.Popen(cmd, cwd=REPO, env=self.env, stdout=open(os.path.join(log_dir, "stdout.log"), "a"),
                                      stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        self.start_eval()
        self.t0, self.reason, self.exit_reason = time.time(), None, None
        log(f"{name}: started train pid {self.train.pid} on GPUs {gpus}: {' '.join(cmd)}")

    def start_eval(self):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": str(self.gpus[-1])}
        self.evaler = subprocess.Popen([PY, "supermimic/evaluate.py", "--run_dir", self.log_dir, "--watch"], cwd=REPO, env=env,
                                       stdout=open(os.path.join(self.log_dir, "eval_stdout.log"), "a"), stderr=subprocess.STDOUT,
                                       start_new_session=True)  # fmt: skip

    def poll(self) -> dict:
        """Apply the stop rules; returns a status summary."""
        rc = self.train.poll()
        tl = read_csv(os.path.join(self.log_dir, "train_log.csv"))
        ev = read_csv(os.path.join(self.log_dir, "eval.csv"))
        if rc is None and self.evaler.poll() is not None:
            log(f"{self.name}: eval watcher exited ({self.evaler.returncode}); restarting it")
            self.start_eval()
        if rc is None and self.reason is None:
            self.reason = stop_reason(tl, ev, (time.time() - self.t0) / 3600)
            if self.reason:
                log(f"{self.name}: stopping at it {tl[-1]['iteration'] if tl else 0}: {self.reason}")
                open(os.path.join(self.log_dir, "STOP"), "w").write(self.reason + "\n")
        if rc is not None and self.exit_reason is None:
            self.exit_reason = self.reason or ("finished" if rc == 0 else f"crashed (exit {rc})")
        last = ev[-1] if ev else {}
        return {"iteration": int(num(tl[-1]["iteration"])) if tl else 0, "it_per_s": num(tl[-1]["it_per_s"]) if tl else None,
                "rollbacks": int(num(tl[-1].get("rollbacks", 0))) if tl and tl[-1].get("rollbacks") else 0, "n_evals": len(ev),
                "last_eval": {k: last.get(k) for k in ("iteration", "norm_return", "fail_rate", "vel_med_err", "vel_fall_rate")},
                "running": rc is None}  # fmt: skip

    def finish(self, crit: dict) -> dict:
        """Let the watcher score the final checkpoint, stop it, and judge the run."""
        ckpts = [int(f[5:11]) for f in os.listdir(self.log_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
        final, t = max(ckpts, default=0), time.time()
        while final and time.time() - t < 1800 and final not in [int(num(r["iteration"])) for r in read_csv(os.path.join(self.log_dir, "eval.csv"))]:
            time.sleep(30)
        try:
            os.killpg(self.evaler.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        best = best_row(read_csv(os.path.join(self.log_dir, "eval.csv")), crit)
        reason = self.exit_reason
        if reason.startswith(("crashed", "diverged")):
            state = reason.split(" ")[0]
        else:
            state = "success" if best is not None and meets(best, crit) else "failed"
        return {"state": state, "reason": reason, "best_eval": best, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_experiment(name: str, motion_packs: list[str], loco_packs: list[str], per_gpu: bool = True, extra: list[str] = (),
                   init_from: str | None = None) -> dict:  # fmt: skip
    """per_gpu: one single-GPU run per GPU (seed = GPU index); else one run on all GPUs.
    init_from: experiment whose run with the same seed initializes each run (from its last.pt)."""
    st = status.setdefault(name, {})
    if st.get("state") in ("success", "failed"):
        log(f"{name}: already {st['state']}, skipping")
        return st
    crit = CRITERIA[name]
    specs = [(f"{name}/s{g}", [g], g) for g in range(NGPU)] if per_gpu else [(name, list(range(NGPU)), 0)]
    runs_st = st.setdefault("runs", {})
    init = lambda seed: os.path.join(REPO, "logs", init_from, f"s{seed}", "last.pt") if init_from else None  # noqa: E731
    runs = [Run(n, os.path.join(REPO, "logs", n), gpus, motion_packs, loco_packs, seed, extra, init(seed)) for n, gpus, seed in specs
            if runs_st.get(n, {}).get("state") not in ("success", "failed", "diverged")]  # fmt: skip
    st.update(state="running", started=st.get("started", time.strftime("%Y-%m-%d %H:%M:%S")), motion_packs=motion_packs,
              loco_packs=loco_packs, criteria=crit)  # fmt: skip
    save_status()
    active = list(runs)
    replaced = set()  # GPUs that already got a replacement run
    while active:
        time.sleep(POLL)
        for r in list(active):
            info = r.poll()
            runs_st.setdefault(r.name, {}).update(info, state="running")
            if not info["running"]:
                runs_st[r.name].update(r.finish(crit))
                log(f"{r.name}: {runs_st[r.name]['state']} ({runs_st[r.name]['reason']}); best eval {runs_st[r.name]['best_eval']}")
                active.remove(r)
                # keep the GPU busy: a run that ended before min_iters gets one replacement seed while others still train
                g = r.gpus[0]
                if per_gpu and active and g not in replaced and info["iteration"] < STOP_RULES["min_iters"]:
                    replaced.add(g)
                    n = f"{name}/s{g + NGPU}"
                    active.append(Run(n, os.path.join(REPO, "logs", n), [g], motion_packs, loco_packs, g + NGPU, extra, init(g)))
        save_status()
        log(f"{name}: " + " | ".join(f"{n.split('/')[-1]} it {s.get('iteration')} rb {s.get('rollbacks')} "
                                     f"eval {s.get('last_eval', {}).get('norm_return')}/{s.get('last_eval', {}).get('vel_med_err')} {s.get('state')}"
                                     for n, s in sorted(runs_st.items())))  # fmt: skip
    states = [s["state"] for s in runs_st.values()]
    n_ok = states.count("success")
    if states.count("crashed") * 2 > len(states):
        st["state"] = "crashed"
    else:
        st["state"] = "success" if n_ok * 2 >= len(states) else "failed"
    st.update(n_success=n_ok, n_runs=len(states), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_status()
    log(f"{name}: {st['state']} ({n_ok}/{len(states)} runs meet the criteria)")
    return st


def wait_for(path: str):
    while not os.path.exists(path):
        log(f"waiting for {path}")
        time.sleep(POLL)


def main():
    s1 = run_experiment("exp1", [LAFAN], [LAFAN], per_gpu=False)
    if s1["state"] != "success":
        log(f"exp1 {s1['state']}: did not pass the code check; stopping the chain")
        return
    wait_for(SONIC)
    s2 = run_experiment("exp2", [SONIC, LAFAN], [SONIC, LAFAN])
    if s2["state"] == "crashed":
        log("exp2 crashed; stopping the chain (fix, then rerun the orchestrator)")
        return
    loco = [SONIC, LAFAN] if s2["state"] == "success" else [LAFAN]
    status["exp3_loco_choice"] = "exp2" if s2["state"] == "success" else "exp1"
    save_status()
    wait_for(ALL)
    run_experiment("exp3", [ALL], loco)
    log("all experiments done")


if __name__ == "__main__":
    if sys.argv[1:] == ["exp4"]:  # exp3 with 100% uniformly sampled velocity commands: from scratch, then resumed from exp3
        loco = [SONIC, LAFAN] if status.get("exp3_loco_choice") == "exp2" else [LAFAN]
        run_experiment("exp4", [ALL], loco, extra=["--ref_cmd_prob", "0"])
        run_experiment("exp4_resume", [ALL], loco, extra=["--ref_cmd_prob", "0"], init_from="exp3")
        log("exp4 done")
    elif sys.argv[1:] == ["exp3_full"]:  # exp3 with the unfiltered motion library, velocity library as chosen for exp3
        loco = [SONIC, LAFAN] if status.get("exp3_loco_choice") == "exp2" else [LAFAN]
        run_experiment("exp3_full", [ALL_RAW], loco)
        log("exp3_full done")
    else:
        main()
