"""Reference motion library at 50 Hz, with FK of the tracked bodies.

Clips come from packs written by ``tools/pack_motions.py`` (root position, root quaternion wxyz and
29 joint angles in Unitree SDK order, already at 50 Hz). Each clip is tagged with the goal
interfaces that may use it: ``in_motion`` (motion-tracking interface) and ``in_loco`` (velocity
interface). Small libraries cache the tracked-body FK; large ones compute it on demand.
"""

from __future__ import annotations

import numpy as np
import torch

from .kinematics import G1Kinematics
from .rotations import mat_to_6d, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
CMD_WINDOW = 25  # centered window of the reference velocity command (paper App. G.1)
BODY_CACHE_MAX_FRAMES = 20_000_000


def load_packs(paths: list[str]) -> dict:
    """Concatenate packs; returns numpy arrays plus per-clip names/lengths/source index."""
    names, lens, src, root_p, root_quat, q = [], [], [], [], [], []
    for i, p in enumerate(paths):
        d = torch.load(p, weights_only=False)
        names.extend(d["names"]), lens.append(np.asarray(d["lengths"], dtype=np.int64)), src.append(np.full(len(d["lengths"]), i))
        root_p.append(d["root_p"]), root_quat.append(d["root_quat"]), q.append(d["q"])
    return {"names": names, "lengths": np.concatenate(lens), "source": np.concatenate(src),
            "root_p": np.concatenate(root_p), "root_quat": np.concatenate(root_quat), "q": np.concatenate(q)}  # fmt: skip


class MotionLib:
    def __init__(self, packs: dict, kin: G1Kinematics, device: str, in_motion: np.ndarray, in_loco: np.ndarray):
        self.device, self.kin = device, kin
        dev = device
        self.names = list(packs["names"])
        lens = torch.as_tensor(packs["lengths"], dtype=torch.long)
        self.num_clips = len(lens)
        self.clip_len = lens.to(dev)
        self.clip_start = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(lens, 0)[:-1])).to(dev)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(lens.sum())
        self.clip_id = torch.repeat_interleave(torch.arange(self.num_clips), lens).to(dev)
        self.in_motion = torch.as_tensor(in_motion, dtype=torch.bool, device=dev)
        self.in_loco = torch.as_tensor(in_loco, dtype=torch.bool, device=dev)

        self.root_p = torch.as_tensor(packs["root_p"], dtype=torch.float32, device=dev)
        self.root_R = quat_to_mat(torch.as_tensor(packs["root_quat"], dtype=torch.float32, device=dev))
        self.q = torch.as_tensor(packs["q"], dtype=torch.float32, device=dev)
        # finite differences inside each clip (central, one-sided at the clip ends)
        i = torch.arange(self.num_frames, device=dev)
        cs, ce = self.clip_start[self.clip_id], self.clip_end[self.clip_id]
        prv, nxt = torch.maximum(i - 1, cs), torch.minimum(i + 1, ce)
        dt = ((nxt - prv).float() * CONTROL_DT).clamp(min=CONTROL_DT).unsqueeze(-1)
        self.root_v = (self.root_p[nxt] - self.root_p[prv]) / dt
        self.qd = (self.q[nxt] - self.q[prv]) / dt
        self.root_w = so3_log(self.root_R[nxt] @ self.root_R[prv].transpose(-1, -2)) / dt
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)

        # reference velocity command (vx, vy in the reference heading frame, world yaw rate)
        h = CMD_WINDOW // 2
        lo, hi = torch.maximum(i - h, cs), torch.minimum(i + h, ce)
        cum = torch.cat((torch.zeros(1, 3, device=dev), torch.cumsum(torch.cat((self.root_v[:, :2], self.root_w[:, 2:]), -1).double(), 0).float()))
        mean = (cum[hi + 1] - cum[lo]) / (hi - lo + 1).float().unsqueeze(-1)
        Hm = yaw_mat(self.root_R)
        vh = (Hm.transpose(-1, -2) @ torch.cat((mean[:, :2], torch.zeros_like(mean[:, :1])), -1).unsqueeze(-1)).squeeze(-1)
        self.ref_cmd = torch.cat((vh[:, :2], mean[:, 2:]), -1)

        self.cached_bodies = self.num_frames <= BODY_CACHE_MAX_FRAMES
        if self.cached_bodies:
            out = [[], [], [], []]
            with torch.no_grad():
                for s in range(0, self.num_frames, 262144):
                    for o, x in zip(out, self._fk(torch.arange(s, min(s + 262144, self.num_frames), device=dev))):
                        o.append(x)
            self.body_p, self.body_R, self.body_v, self.body_w = (torch.cat(o) for o in out)

        # 1 s bins inside each clip
        nb = (lens + BIN_FRAMES - 1) // BIN_FRAMES
        self.num_bins = int(nb.sum())
        self.bin_clip = torch.repeat_interleave(torch.arange(self.num_clips), nb).to(dev)
        first_bin = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(nb, 0)[:-1])).to(dev)
        local = torch.arange(self.num_bins, device=dev) - first_bin[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local * BIN_FRAMES
        self.frame_bin = first_bin[self.clip_id] + (i - cs) // BIN_FRAMES
        self.bin_motion = self.in_motion[self.bin_clip]
        self.bin_loco = self.in_loco[self.bin_clip]

    @staticmethod
    def from_packs(motion_packs: list[str], loco_packs: list[str], kin: G1Kinematics, device: str) -> MotionLib:
        """Union of the packs; a pack listed for both interfaces is loaded once."""
        paths = list(dict.fromkeys(motion_packs + loco_packs))
        packs = load_packs(paths)
        src = packs["source"]
        in_motion = np.isin(src, [paths.index(p) for p in motion_packs])
        in_loco = np.isin(src, [paths.index(p) for p in loco_packs])
        return MotionLib(packs, kin, device, in_motion, in_loco)

    def _fk(self, f: torch.Tensor):
        return self.kin.forward(self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f])

    def bodies(self, f: torch.Tensor):
        """Tracked-body world positions, rotations, linear and angular velocities of frames f."""
        if self.cached_bodies:
            return self.body_p[f], self.body_R[f], self.body_v[f], self.body_w[f]
        return self._fk(f)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]
