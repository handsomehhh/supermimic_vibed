"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "cmd": (3, torch.float32),
    "cmd_ref": ((), torch.bool),
    "valid": ((), torch.bool),  # False for transitions in which the simulator blew up
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, cmd_dim: int = 3, terrain_vel: int = 0, cam_dim: int = 0,
                 frames: int = 10, image: tuple[int, int] = (36, 64)):
        """terrain_vel > 0: the last `terrain_vel` envs are velocity worlds on terrain; also store per transition the
        terrain info (stairs flag, level), camera parameters, whether a new depth frame arrived (tick) and, for those
        envs only, the latest depth frame (uint8)."""
        self.T, self.N, self.device = T, N, device
        self.fields = {**FIELDS, "cmd": (cmd_dim, torch.float32)}
        if terrain_vel:
            self.fields.update(info=(2, torch.float32), cam=(cam_dim, torch.float32), tick=((), torch.bool))
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.terrain_vel, self.vel_lo, self.frames = terrain_vel, N - terrain_vel, frames
        if terrain_vel:
            self.depth = torch.zeros((T, terrain_vel, *image), dtype=torch.uint8, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        if self.terrain_vel:
            self.depth[self.ptr] = tr["depth"]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done", "valid")}
        # transition j is used if no episode ended at transitions 0..j-1 and transitions 0..j are valid
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        mask = mask * torch.cumprod(out["valid"].float(), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, env_lo: int = 0, env_hi: int | None = None, depth_window: int = 0):
        """Start points for imagination (from envs [env_lo, env_hi)) plus their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(env_lo, self.N if env_hi is None else env_hi, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        keys = ("s", "a_prev", "xi", "f", "cmd", "cmd_ref", "valid") + (("info", "cam", "ep_step") if self.terrain_vel else ())
        start = {k: self.data[k][row, env] for k in keys}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        if depth_window:
            start["depth"] = self._depth_stack(age, env, ep_step, depth_window)
        return start, hist_d

    def _depth_stack(self, age, env, ep_step, W: int):
        """The `frames` most recent depth frames (uint8, oldest first) the policy saw at the sampled states: the
        frames stored at the last `frames` tick steps of a W-step window (steps before the episode start repeat
        its first frame, as the simulator does). W = 20 covers 10 frames at 30 Hz against 50 Hz control."""
        back = torch.arange(W - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        rows = self._row(age.unsqueeze(-1) - back)  # (B, W) oldest first
        e = env.unsqueeze(-1).expand_as(rows)
        tick = self.data["tick"][rows, e]
        # a repeated (clamped) step counts once: keep only the newest copy of each distinct row
        dup = torch.cat((rows[:, 1:] == rows[:, :-1], torch.zeros_like(rows[:, :1], dtype=torch.bool)), -1)
        tick = tick & ~dup
        rank = torch.flip(torch.cumsum(torch.flip(tick.long(), [-1]), -1), [-1])  # newest tick has rank 1
        slot = self.frames - rank  # 0 .. frames-1 for the last `frames` ticks
        take = tick & (slot >= 0)
        frames = self.depth[rows, e - self.vel_lo]  # (B, W, H, W')
        out = frames[:, :1].expand(-1, self.frames, -1, -1).clone()  # fallback: the oldest frame in the window
        b_idx, w_idx = take.nonzero(as_tuple=True)
        out[b_idx, slot[b_idx, w_idx]] = frames[b_idx, w_idx]
        # fewer ticks than frames (episode start): pad the front with the oldest taken frame, like the simulator
        n_taken = take.sum(-1)
        first = (self.frames - n_taken).clamp(max=self.frames - 1)
        pad = torch.arange(self.frames, device=self.device).unsqueeze(0) < first.unsqueeze(-1)
        oldest = out[torch.arange(len(out), device=self.device), first]
        out = torch.where(pad[..., None, None], oldest.unsqueeze(1), out)
        return out
