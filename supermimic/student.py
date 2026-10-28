"""Deployable student model: measured proprioception history and onboard map only."""
from copy import deepcopy

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization


class StudentNormalization(EmpiricalNormalization):
    """Normalize proprioception; preserve metric heights and binary validity.

    Per-cell empirical normalization is ill-conditioned for consistently blind
    cells: making them valid at deployment would amplify heights by 100x.
    """
    def __init__(self, size, history_dim):
        super().__init__(size)
        self.history_dim = history_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = (x[..., :self.history_dim] - self._mean[..., :self.history_dim]) / (self._std[..., :self.history_dim] + self.eps)
        return torch.cat((history, x[..., self.history_dim:]), -1)


class ConvMapEncoder(nn.Module):
    def __init__(self, shape, map_latent):
        super().__init__()
        self.nx, self.ny = shape
        self.map_latent = map_latent
        self.layers = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(), nn.Flatten(),
            nn.Linear(32 * ((self.nx + 1) // 2) * ((self.ny + 1) // 2), map_latent), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        shape[-1] = self.map_latent
        # ElevationMap: all heights then all masks, x outer and y inner.
        return self.layers(x.reshape(-1, 2, self.nx, self.ny)).reshape(shape)


class StudentModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, map_latent=64,
                 map_encoder="mlp", map_shape=None, **kwargs):
        if obs_groups[obs_set] != ["policy_history", "elevation_map"]:
            raise ValueError("Student inputs must be policy_history, elevation_map in that order")
        self.history_dim = obs["policy_history"].shape[-1]
        self.map_dim = obs["elevation_map"].shape[-1]
        self.map_latent = map_latent
        self.map_encoder_type = map_encoder
        if map_encoder not in ("mlp", "cnn"):
            raise ValueError("map_encoder must be mlp or cnn")
        if map_latent < 1:
            raise ValueError("map_latent must be positive")
        if map_encoder == "cnn" and (map_shape is None or len(map_shape) != 2 or
                                     any(not isinstance(n, int) or n < 1 for n in map_shape) or
                                     2 * map_shape[0] * map_shape[1] != self.map_dim):
            raise ValueError("CNN map_shape must be (nx, ny) with 2 * nx * ny == elevation_map size")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_normalization:
            self.obs_normalizer = StudentNormalization(self.obs_dim, self.history_dim)
        if map_encoder == "cnn":
            self.map_encoder = ConvMapEncoder(map_shape, map_latent)
        else:
            self.map_encoder = nn.Sequential(nn.Linear(self.map_dim, 128), nn.ELU(),
                                             nn.Linear(128, map_latent), nn.ELU())

    def _get_latent_dim(self):
        return self.history_dim + self.map_latent

    def get_latent(self, obs, masks=None, hidden_state=None):
        x = super().get_latent(obs, masks, hidden_state)
        return torch.cat((x[..., :self.history_dim], self.map_encoder(x[..., self.history_dim:])), -1)

    def as_jit(self):
        return ExportStudent(self)

    def as_onnx(self, verbose=False):
        return ExportStudent(self)


class ExportStudent(nn.Module):
    """Stateless interface: caller maintains history and map refreshes."""
    def __init__(self, actor):
        super().__init__()
        self.history_dim = actor.history_dim
        self.input_size = actor.obs_dim
        self.obs_normalizer = deepcopy(actor.obs_normalizer)
        self.map_encoder = deepcopy(actor.map_encoder)
        self.mlp = deepcopy(actor.mlp)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(observations)
        latent = torch.cat((x[..., :self.history_dim], self.map_encoder(x[..., self.history_dim:])), -1)
        return self.mlp(latent)

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self):
        return ["observations"]

    @property
    def output_names(self):
        return ["actions"]


def is_student(ck):
    return ck.get("runner_cfg", {}).get("policy_type") == "student"


def load_student(ck, device="cpu"):
    """Load just the actor, without physics, teacher, optimizer or GPU allocation."""
    cfg = ck["env_cfg"]
    from .elevation_map import ElevationCfg
    shape = ElevationCfg(**cfg["elevation"]).shape
    obs = TensorDict({"policy_history": torch.zeros(1, 96 * cfg["history_length"], device=device),
                      "elevation_map": torch.zeros(1, 2 * shape[0] * shape[1], device=device)}, [1])
    name = "student" if "student_state_dict" in ck else "actor"
    model_cfg = deepcopy(ck["runner_cfg"][name])
    model_cfg.pop("class_name")
    if model_cfg.get("map_encoder", "mlp") == "cnn":
        if tuple(model_cfg.setdefault("map_shape", shape)) != shape:
            raise ValueError("Student map_shape does not match checkpoint elevation config")
    actor = StudentModel(obs, {name: ["policy_history", "elevation_map"]}, name, 29, **model_cfg).to(device)
    actor.load_state_dict(ck[name + "_state_dict"])
    return actor.eval()
