"""Pack G1 motions into one file for MotionLib: root position, root quaternion (wxyz), 29 joints in
Unitree SDK order, all at 50 Hz.

Sources:
  --csv_dir DIR    LAFAN1 G1 retarget CSVs (30 fps; root pos, root quat xyzw, 29 joints in SDK order)
  --npz_root DIR   BeyondMimic-style npz (fps, joint_pos/body_* in Isaac Lab breadth-first order), recursive
  --in_pack FILE   an existing pack (to re-filter it without reading the sources again)
Filters:
  --filter loco    walking/jogging/running clips by name (default scheme) and an upright check
  --glob PATTERN   only file names matching PATTERN (fnmatch, e.g. 'walk*')
  --plausible      drop clips with physically implausible velocities (root jumps, joint flips):
                   root speed > 10 m/s, root angular velocity > 20 rad/s or joint velocity > 30 rad/s

  python tools/pack_motions.py --csv_dir data/lafan1_g1_loco --out data/packs/lafan_loco.pt
  python tools/pack_motions.py --npz_root .../sonic_filtered --filter loco --out data/packs/sonic_loco.pt
"""

from __future__ import annotations

import argparse
import fnmatch
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supermimic.robot import G1_JOINT_NAMES

FPS = 50
# joint order of the npz files (Isaac Lab breadth-first order of the G1 USD), verified by FK against body_pos_w
BFS_JOINTS = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee", "left_shoulder_pitch",
    "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch", "left_wrist_yaw", "right_wrist_yaw",
]  # fmt: skip
BFS_TO_SDK = [BFS_JOINTS.index(n.removesuffix("_joint")) for n in G1_JOINT_NAMES]

LOCO_WORDS = {"walk", "walking", "jog", "jogging", "run", "running", "sprint", "sprinting"}
LOCO_FIRST = LOCO_WORDS | {"neutral", "turn"}
# props, injuries, dances, emotions, non-standing postures and stylizations
LOCO_EXCLUDE = {
    "injured", "inj", "painful", "crate", "box", "hand", "hands", "handed", "dog", "leash", "baby", "diaper", "smoke",
    "phone", "dance", "dancing", "dancecard", "vouge", "crouch", "crawl", "kneeling", "stair", "stairs", "ladder",
    "chair", "sit", "sitting", "drunk", "zombie", "limp", "limping", "tired", "carry", "carrying", "push", "pull",
    "bag", "cup", "drink", "drinking", "obstacle", "avoid", "door", "street", "wall", "item", "tray", "umbrella",
    "exaggerated", "impro", "sexy", "old", "heavy", "light", "lift", "grab", "wave", "watering", "swear", "bump",
    "relax", "stealth", "sneak", "sneaking", "cartoon", "robot", "military", "march", "proud", "sad", "happy",
    "angry", "scared", "tiptoe", "stomp", "skip", "skipping", "hop", "jump", "fall", "stumble", "trip", "look",
    "looking", "talk", "talking", "texting", "pointing", "clap", "strafing", "medium", "small", "big",
}  # fmt: skip


def base_name(path: str) -> str:
    return re.sub(r"__A\d+(_M)?$", "", os.path.splitext(os.path.basename(path))[0])


def is_loco_name(path: str) -> bool:
    toks = [t for t in re.split(r"[_\d]+", base_name(path).lower()) if t]
    return bool(toks) and toks[0] in LOCO_FIRST and bool(LOCO_WORDS & set(toks)) and not (LOCO_EXCLUDE & set(toks))


def resample(p, quat, q, src_fps):
    """Linear (slerp for the root rotation) resampling to 50 Hz."""
    if abs(src_fps - FPS) < 1e-6:
        return p, quat, q
    T = len(p)
    n = int(np.floor((T - 1) / src_fps * FPS)) + 1
    t = np.arange(n) / FPS * src_fps
    i0 = np.minimum(np.floor(t).astype(int), T - 2)
    a = np.clip(t - i0, 0, 1)[:, None]
    q0, q1 = quat[i0], quat[i0 + 1]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    dot = np.clip((q0 * q1).sum(-1, keepdims=True), -1, 1)
    th = np.arccos(dot)
    s = np.sin(th)
    small = s < 1e-6
    w0 = np.where(small, 1 - a, np.sin((1 - a) * th) / np.where(small, 1, s))
    w1 = np.where(small, a, np.sin(a * th) / np.where(small, 1, s))
    qq = w0 * q0 + w1 * q1
    qq /= np.linalg.norm(qq, axis=-1, keepdims=True)
    return p[i0] * (1 - a) + p[i0 + 1] * a, qq, q[i0] * (1 - a) + q[i0 + 1] * a


def load(path: str):
    try:
        if path.endswith(".csv"):
            raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
            p, quat, q, fps = raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:36], 30.0
        else:
            z = np.load(path)
            fps = float(np.asarray(z["fps"]).reshape(-1)[0])
            p, quat, q = z["body_pos_w"][:, 0].astype(np.float64), z["body_quat_w"][:, 0].astype(np.float64), z["joint_pos"][:, BFS_TO_SDK].astype(np.float64)
        p, quat, q = resample(p, quat, q, fps)
        if len(p) < 25 or not (np.isfinite(p).all() and np.isfinite(quat).all() and np.isfinite(q).all()):
            return None
        return os.path.splitext(os.path.basename(path))[0], p.astype(np.float32), quat.astype(np.float32), q.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e!r}", flush=True)
        return None


def upright(p: np.ndarray, quat: np.ndarray) -> bool:
    """Pelvis stays at standing height and roughly upright for the whole clip."""
    w, x, y, z = quat.T
    r22 = 1 - 2 * (x * x + y * y)
    return bool(p[:, 2].min() > 0.55 and r22.min() > 0.7)


def plausible(p: np.ndarray, quat: np.ndarray, q: np.ndarray) -> bool:
    dt = 1.0 / FPS
    v = np.linalg.norm(np.diff(p, axis=0), axis=-1) / dt
    dot = np.clip(np.abs((quat[1:] * quat[:-1]).sum(-1)), 0, 1)
    w = 2 * np.arccos(dot) / dt
    qd = np.abs(np.diff(q, axis=0)).max(-1) / dt
    return bool(v.max() <= 10.0 and w.max() <= 20.0 and qd.max() <= 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", action="append", default=[])
    ap.add_argument("--npz_root", action="append", default=[])
    ap.add_argument("--filter", choices=["none", "loco"], default="none")
    ap.add_argument("--in_pack", action="append", default=[])
    ap.add_argument("--plausible", action="store_true")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    for d in args.csv_dir:
        files += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv"))
    for r in args.npz_root:
        for dp, _, fn in sorted(os.walk(r)):
            files += sorted(os.path.join(dp, f) for f in fn if f.endswith(".npz"))
    files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), args.glob)]
    if args.filter == "loco":
        files = [f for f in files if f.endswith(".csv") or is_loco_name(f)]
    print(f"{len(files)} files", flush=True)
    with mp.Pool(args.workers) as pool:
        clips = [c for c in pool.imap(load, files, chunksize=16) if c is not None]
    for path in args.in_pack:
        d = torch.load(path, weights_only=False)
        ends = np.cumsum(d["lengths"])
        for n, a, b in zip(d["names"], ends - d["lengths"], ends):
            clips.append((n, d["root_p"][a:b], d["root_quat"][a:b], d["q"][a:b]))
    n_load = len(clips)
    if args.filter == "loco":
        clips = [c for c in clips if upright(c[1], c[2])]
    if args.plausible:
        clips = [c for c in clips if plausible(c[1], c[2], c[3])]
    lens = np.array([len(c[1]) for c in clips], dtype=np.int64)
    pack = {
        "names": [c[0] for c in clips], "lengths": lens,
        "root_p": np.concatenate([c[1] for c in clips]), "root_quat": np.concatenate([c[2] for c in clips]),
        "q": np.concatenate([c[3] for c in clips]), "sources": args.csv_dir + args.npz_root + args.in_pack,
        "filter": args.filter + ("+plausible" if args.plausible else ""),
    }  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(pack, args.out, pickle_protocol=4)  # packs exceed 4 GiB
    print(f"loaded {n_load}, kept {len(clips)} clips, {lens.sum()} frames = {lens.sum() / FPS / 3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
