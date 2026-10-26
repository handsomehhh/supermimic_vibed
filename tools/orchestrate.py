"""Run the three velocity-interface experiments one after another on all GPUs, with a monitor that
stops a run once it is done, diverged, or no longer improving.

  exp1  motion + velocity interfaces on the LAFAN1 locomotion clips (code check)
  exp2  both interfaces on BONES-SEED (sonic_filtered) locomotion + LAFAN1 locomotion
  exp3  velocity interface on exp2's library if exp2 succeeded, else exp1's; motion interface on all of
        motion_data_used_g1

Every POLL seconds the monitor reads train_log.csv / eval.csv of the running experiment and writes
STOP (the trainer saves a checkpoint and exits) when
  - the trainer crashed, or losses are NaN/inf for 3 consecutive log rows (diverged),
  - wall time exceeds max_hours,
  - after min_iters, none of the eval metrics improved for `patience` evaluations (converged),
  - after nogo_iters, motion fail rate and velocity fall rate are both still > 0.9 (no progress).
A run succeeds if its best evaluation meets the experiment's criteria. A crash stops the chain; rerunning
the orchestrator resumes the unfinished experiment from its last.pt. Status: logs/orchestrator.json.

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
TORCHRUN = os.path.join(os.path.dirname(PY), "torchrun")
NGPU = int(os.environ.get("NGPU", 8))
POLL = 300
PACKS = os.path.join(REPO, "data", "packs")
LAFAN = os.path.join(PACKS, "lafan_loco.pt")
SONIC = os.path.join(PACKS, "sonic_loco.pt")
ALL = os.path.join(PACKS, "all_g1.pt")
STATUS = os.path.join(REPO, "logs", "orchestrator.json")

CRITERIA = {  # best evaluation must satisfy all
    "exp1": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.12), "vel_med_err": ("<=", 0.35), "vel_fall_rate": ("<=", 0.10)},
    "exp2": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
    "exp3": {"norm_return": (">=", 0.75), "fail_rate": ("<=", 0.05), "vel_med_err": ("<=", 0.30), "vel_fall_rate": ("<=", 0.05)},
}
STOP_RULES = {"min_iters": 20_000, "patience": 6, "nogo_iters": 30_000, "max_hours": 30.0, "max_iters": 100_000}
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


def run(name: str, motion_packs: list[str], loco_packs: list[str]) -> dict:
    log_dir = os.path.join(REPO, "logs", name)
    os.makedirs(log_dir, exist_ok=True)
    st = status.setdefault(name, {})
    if st.get("state") in ("success", "failed"):
        log(f"{name}: already {st['state']}, skipping")
        return st
    resume = os.path.join(log_dir, "last.pt")
    cmd = [TORCHRUN, "--standalone", f"--nproc_per_node={NGPU}", "supermimic/train.py", "--log_dir", log_dir,
           "--motion_packs", *motion_packs, "--loco_packs", *loco_packs, "--iterations", str(STOP_RULES["max_iters"])]  # fmt: skip
    if os.path.exists(resume):
        cmd += ["--resume", resume]
    if os.path.exists(os.path.join(log_dir, "STOP")):
        os.remove(os.path.join(log_dir, "STOP"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(NGPU))), "PYTHONUNBUFFERED": "1"}
    train = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(os.path.join(log_dir, "stdout.log"), "a"), stderr=subprocess.STDOUT,
                             start_new_session=True)  # fmt: skip
    ev_env = {**env, "CUDA_VISIBLE_DEVICES": str(NGPU - 1)}
    evaler = subprocess.Popen([PY, "supermimic/evaluate.py", "--run_dir", log_dir, "--watch"], cwd=REPO, env=ev_env,
                              stdout=open(os.path.join(log_dir, "eval_stdout.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
    st.update(state="running", started=st.get("started", time.strftime("%Y-%m-%d %H:%M:%S")), motion_packs=motion_packs, loco_packs=loco_packs,
              train_pid=train.pid, eval_pid=evaler.pid)  # fmt: skip
    save_status()
    log(f"{name}: started train pid {train.pid}, eval pid {evaler.pid}: {' '.join(cmd)}")
    t0 = time.time()
    reason = None
    while True:
        time.sleep(POLL)
        rc = train.poll()
        tl = read_csv(os.path.join(log_dir, "train_log.csv"))
        ev = read_csv(os.path.join(log_dir, "eval.csv"))
        it = int(num(tl[-1]["iteration"])) if tl else 0
        last = ev[-1] if ev else {}
        st.update(iteration=it, wall_h=round((time.time() - t0) / 3600, 3), n_evals=len(ev), last_eval=last,
                  it_per_s=num(tl[-1]["it_per_s"]) if tl else None)  # fmt: skip
        if evaler.poll() is not None and rc is None:
            log(f"{name}: eval watcher exited ({evaler.returncode}); restarting it")
            evaler = subprocess.Popen([PY, "supermimic/evaluate.py", "--run_dir", log_dir, "--watch"], cwd=REPO, env=ev_env,
                                      stdout=open(os.path.join(log_dir, "eval_stdout.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        if rc is not None:
            reason = reason or ("finished" if rc == 0 else f"crashed (exit {rc})")
            break
        if reason is None:
            bad = [r for r in tl[-3:] if any(not math.isfinite(num(r[k])) for k in ("wm_loss", "pi_loss"))]
            if len(tl) >= 3 and len(bad) == 3:
                reason = "diverged (NaN/inf losses)"
            elif (time.time() - t0) / 3600 > STOP_RULES["max_hours"]:
                reason = "max wall time"
            elif ev and int(num(last["iteration"])) >= STOP_RULES["min_iters"] and plateaued(ev, STOP_RULES["patience"]):
                reason = "converged (no eval improvement)"
            elif ev and int(num(last["iteration"])) >= STOP_RULES["nogo_iters"] and num(last["fail_rate"]) > 0.9 and num(last.get("vel_fall_rate", 1)) > 0.9:
                reason = "no progress"
            if reason:
                log(f"{name}: stopping at it {it}: {reason}")
                open(os.path.join(log_dir, "STOP"), "w").write(reason + "\n")
        save_status()
        log(f"{name}: it {it}, {st['it_per_s']} it/s, evals {len(ev)}, last eval {({k: last.get(k) for k in ('iteration', 'norm_return', 'fail_rate', 'vel_med_err', 'vel_fall_rate')})}")
    # let the watcher score the final checkpoint, then stop it
    final = max([int(f[5:11]) for f in os.listdir(log_dir) if f.startswith("ckpt_") and f.endswith(".pt")], default=0)
    t_wait = time.time()
    while final and time.time() - t_wait < 1800 and final not in [int(num(r["iteration"])) for r in read_csv(os.path.join(log_dir, "eval.csv"))]:
        time.sleep(30)
    os.killpg(evaler.pid, signal.SIGTERM)
    ev = read_csv(os.path.join(log_dir, "eval.csv"))
    best = best_row(ev, CRITERIA[name])
    ok = best is not None and meets(best, CRITERIA[name]) and not reason.startswith(("crashed", "diverged"))
    state = "crashed" if reason.startswith("crashed") else "success" if ok else "failed"
    st.update(state=state, reason=reason, best_eval=best, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
              criteria=CRITERIA[name])  # fmt: skip
    save_status()
    log(f"{name}: {st['state']} ({reason}); best eval {best}")
    return st


def wait_for(path: str):
    while not os.path.exists(path):
        log(f"waiting for {path}")
        time.sleep(POLL)


def main():
    s1 = run("exp1", [LAFAN], [LAFAN])
    if s1["state"] != "success":
        log(f"exp1 {s1['state']}: did not pass the code check; stopping the chain")
        return
    wait_for(SONIC)
    s2 = run("exp2", [SONIC, LAFAN], [SONIC, LAFAN])
    if s2["state"] == "crashed":
        log("exp2 crashed; stopping the chain (rerun the orchestrator to resume)")
        return
    loco = [SONIC, LAFAN] if s2["state"] == "success" else [LAFAN]
    status["exp3_loco_choice"] = "exp2" if s2["state"] == "success" else "exp1"
    save_status()
    wait_for(ALL)
    run("exp3", [ALL], loco)
    log("all experiments done")


if __name__ == "__main__":
    main()
