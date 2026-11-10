"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call

from .mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from .rotations import so3_exp, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    terrain_dim > 0: the input also carries privileged terrain features (height scan around the pelvis,
    sole-pad clearances; terrain_mdp.terrain_features), and the height feature is taken above the ground."""

    def __init__(self, hidden=(2048, 2048, 2048), terrain_dim: int = 0):
        super().__init__()
        self.terrain_dim = terrain_dim
        self.in_dim = INV_DIM + NUM_J + XI_DIM + terrain_dim
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n, terr=None):
        """terr: (ground height under the pelvis (B,), terrain features (B, terrain_dim)) when terrain_dim > 0."""
        if not self.terrain_dim:
            return torch.cat((invariant_features(S), a, xi_n), -1)
        gz, feat = terr
        return torch.cat((invariant_features(S, gz), a, xi_n, feat), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, terr=None):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, terr)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class DepthCNN(nn.Module):
    """Per-frame encoder of a normalized 36 x 64 depth image (values in [-1, 1]) to `out` features."""

    def __init__(self, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),  # 18 x 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),  # 9 x 16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),  # 5 x 8
            nn.Flatten(), nn.Linear(32 * 5 * 8, out), nn.ELU())  # fmt: skip

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        lead = img.shape[:-2]
        y = run_mlp(self.net, img.reshape(-1, 1, *img.shape[-2:]))
        return y.reshape(*lead, -1)


class Policy(nn.Module):
    """Per-interface goal encoders and a shared decoder (paper Eq. A.1): a = dec(enc_i(g), o_{t-9:t}).

    ``enc``: motion encoder on K future reference frames (+ reference-relative root orientation);
    ``enc_vel``: velocity encoder on the normalized command and the observation history. The latent
    z in R^32 is bounded by tanh; the policy is deterministic.
    """

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, cmd_dim: int = 3,
                 depth_frames: int = 0, depth_feat: int = 64, depth_latent: int = 64):
        super().__init__()
        self.hist = hist
        self.depth_frames = depth_frames
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        # velocity encoder input: [command, observation history, depth latent]; the depth latent comes last so
        # a flat-ground checkpoint's first layer is a prefix of this one (warm start with zero new columns)
        self.enc_vel = mlp(cmd_dim + obs_dim * hist + (depth_latent if depth_frames else 0), [256, 256], z_dim)
        if depth_frames:
            self.depth_cnn = DepthCNN(depth_feat)
            self.depth_mlp = mlp(depth_frames * depth_feat, [256], depth_latent)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def z_motion(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))

    def depth_features(self, img: torch.Tensor) -> torch.Tensor:
        """(..., 36, 64) frames -> (..., depth_feat); frames are encoded once and stacked by the caller."""
        return self.depth_cnn(img)

    def z_vel(self, obs_hist: torch.Tensor, cmd_n: torch.Tensor, dfeat: torch.Tensor | None = None) -> torch.Tensor:
        """dfeat: (B, depth_frames, depth_feat) per-frame features of the last frames, oldest first."""
        o = self.obs_norm(obs_hist).flatten(-2)
        x = torch.cat((cmd_n, o), -1)
        if self.depth_frames:
            x = torch.cat((x, run_mlp(self.depth_mlp, dfeat.flatten(-2))), -1)
        return torch.tanh(run_mlp(self.enc_vel, x))

    def decode(self, z: torch.Tensor, obs_hist: torch.Tensor, frozen: bool = False) -> torch.Tensor:
        """frozen: stop-gradient on the decoder weights (gradients still reach z)."""
        x = torch.cat((z, self.obs_norm(obs_hist).flatten(-2)), -1)
        if not frozen:
            return run_mlp(self.dec, x)
        params = {k: v.detach() for k, v in self.dec.named_parameters()}
        return run_mlp(lambda y: functional_call(self.dec, params, (y,)), x)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Motion-tracking policy. obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        return self.decode(self.z_motion(goal), obs_hist)

    def act_vel(self, obs_hist: torch.Tensor, cmd_n: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        """depth: (B, depth_frames, 36, 64) frame stack (oldest first) for depth policies."""
        dfeat = self.depth_features(depth) if self.depth_frames else None
        return self.decode(self.z_vel(obs_hist, cmd_n, dfeat), obs_hist)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
