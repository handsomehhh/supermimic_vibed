"""CPU-only CSV run monitor. Write a reasoned STOP request; never kill processes."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import time


def read_rows(path):
    """Ignore an unfinished trailing write; completed malformed rows are errors."""
    if not path.exists():
        return [], []
    text = path.read_text()
    if not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    rows, errors = [], []
    for line, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
        if None in row or any(v is None or v == "" for v in row.values()):
            errors.append(f"{path.name}:{line}: malformed completed row")
            continue
        parsed = {}
        for key, value in row.items():
            if key in ("terrain", "map_noise", "depth_noise", "checkpoint", "policy_type"):
                parsed[key] = value
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                errors.append(f"{path.name}:{line}: nonnumeric {key}={value}")
                continue
            if not math.isfinite(parsed[key]):
                errors.append(f"{path.name}:{line}: nonfinite {key}={value}")
        if all(math.isfinite(v) for v in parsed.values() if isinstance(v, float)):
            rows.append(parsed)
    return rows, errors


def last_improvement(points, delta):
    best, improved = -math.inf, None
    for iteration, value in sorted(points):
        if value > best + delta:
            best, improved = value, iteration
    return improved


def assess(run, args, now=None):
    now = time.time() if now is None else now
    train, train_errors = read_rows(run / "train_log.csv")
    evaluation, eval_errors = read_rows(run / "eval_loco.csv")
    reasons = train_errors + eval_errors
    # a resume from an older checkpoint rewrites iterations: keep the latest row written for each iteration
    train = sorted({r["iteration"]: r for r in train if "iteration" in r}.values(), key=lambda r: r["iteration"])
    result = {"run": str(run), "iteration": 0, "status": "waiting", "reasons": reasons}
    if train:
        latest = train[-1]
        iteration = latest["iteration"]
        logged_seconds = latest.get("wall_time", latest.get("wall_h", 0) * 3600)
        # Include time since the last log flush, to catch stalled training runs.
        hours = (logged_seconds + max(0, now - (run / "train_log.csv").stat().st_mtime)) / 3600
        cutoff = iteration - args.blowup_window
        # sim_blowups is cumulative and resumes from the checkpoint: the first logged row is the baseline
        blowups, prior = 0, train[0].get("sim_blowups", 0)
        for row in train:
            current = row.get("sim_blowups", prior)
            if row["iteration"] > cutoff:
                blowups += max(0, current - prior)  # a decrease is a restart from a checkpoint, not new blow-ups
            prior = current
        if args.max_blowups > 0 and blowups >= args.max_blowups:
            reasons.append(f"blowups {blowups:g} >= {args.max_blowups} in last {args.blowup_window} iterations")
        if args.max_hours > 0 and hours >= args.max_hours:
            reasons.append(f"elapsed {hours:.3f} h >= max_hours {args.max_hours:g}")
        levels = [(r["iteration"], r["terrain_level"]) for r in train if "terrain_level" in r]
        groups = {}
        for row in evaluation:
            if row.get("depth_noise", row.get("map_noise", "on")) != args.eval_map_noise or not {"iteration", "success"} <= row.keys():
                continue
            group = groups.setdefault(row["iteration"], {})
            # Last row wins if a writer accidentally duplicated a terrain/level.
            group[(row.get("terrain"), row.get("level"))] = row
        success = [(it, sum(r["success"] * r.get("trials", 1) for r in group.values()) /
                    max(1, sum(r.get("trials", 1) for r in group.values()))) for it, group in groups.items()]
        last_level = last_improvement(levels, args.level_delta)
        last_eval = last_improvement(success, args.success_delta)
        eval_iteration = max(groups, default=None)
        # Missing/stale eval data cannot prove that eval success has plateaued.
        plateau = (iteration >= args.min_iterations and last_level is not None and last_eval is not None
                   and iteration - last_level >= args.patience
                   and eval_iteration - last_eval >= args.patience
                   and 0 <= iteration - eval_iteration <= args.eval_max_lag)
        if plateau:
            reasons.append(f"plateau: level last improved at {last_level:g}, eval at {last_eval:g}; patience {args.patience}")
        result.update(iteration=int(iteration), hours=hours, recent_blowups=blowups,
                      terrain_level=latest.get("terrain_level"), last_level_improvement=last_level,
                      eval_iteration=eval_iteration, last_eval_improvement=last_eval,
                      eval_success=max(success)[1] if success else None, status="running")
    stop = run / "STOP"
    if reasons and not stop.exists():
        try:
            with stop.open("x") as f:
                f.write("monitor_loco " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "\n".join(reasons) + "\n")
        except FileExistsError:
            pass
    if stop.exists():
        result.update(status="stop_requested", stop_reason=stop.read_text().strip())
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", type=Path, nargs="*", help="Explicit run directories (default: logs/loco/*/)")
    p.add_argument("--root", type=Path, default=Path("logs/loco"))
    p.add_argument("--output", type=Path, default=Path("logs/loco/monitor.json"))
    p.add_argument("--poll_seconds", type=float, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max_blowups", type=int, default=1000, help="Window count; 0 disables")
    p.add_argument("--blowup_window", type=int, default=1000)
    p.add_argument("--patience", type=int, default=3000)
    p.add_argument("--min_iterations", type=int, default=4000)
    p.add_argument("--level_delta", type=float, default=.01)
    p.add_argument("--success_delta", type=float, default=.01)
    p.add_argument("--eval_max_lag", type=int, default=1000)
    p.add_argument("--eval_map_noise", "--eval_depth_noise", choices=("on", "off"), default="on")
    p.add_argument("--max_hours", type=float, default=24, help="0 disables")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.poll_seconds, args.blowup_window, args.patience) <= 0 or min(
            args.min_iterations, args.level_delta, args.success_delta, args.max_hours, args.max_blowups, args.eval_max_lag) < 0:
        p.error("Periods must be positive and thresholds nonnegative")
    while True:
        runs = args.dirs or sorted(d for d in args.root.glob("*") if d.is_dir())
        statuses = []
        for run in runs:
            if not run.is_dir():
                status = {"run": str(run), "status": "missing", "iteration": 0}
            else:
                try:
                    status = assess(run, args)
                except (OSError, csv.Error) as error:
                    status = {"run": str(run), "status": "read_error", "iteration": 0, "error": str(error)}
            statuses.append(status)
            print(f"{run}: {status['status']} it={status['iteration']} "
                  f"level={status.get('terrain_level')} success={status.get('eval_success')} "
                  f"{' | '.join(status.get('reasons', []))}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps({"updated_at": time.time(), "runs": statuses}, indent=2, allow_nan=False))
        temp.replace(args.output)
        if args.once:
            break
        deadline = time.monotonic() + args.poll_seconds
        while time.monotonic() < deadline:
            time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
