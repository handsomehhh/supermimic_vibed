"""Tiled terrain in meters: height rows run along +y, columns along +x."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .rotations import yaw_mat

TILE_TYPES = ("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes")


@dataclass
class TerrainCfg:
    tile_size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    num_cols: int = 20
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    border_width: float = 5.0
    proportions: tuple[float, ...] = (0.10, 0.20, 0.15, 0.15, 0.15, 0.15, 0.10)
    platform_size: float = 1.2


@dataclass
class Terrain:
    heights: np.ndarray
    origins: np.ndarray
    tile_types: np.ndarray
    resolution: float
    offset: tuple[float, float]

    @property
    def extent(self) -> tuple[float, float]:
        return ((self.heights.shape[1] - 1) * self.resolution, (self.heights.shape[0] - 1) * self.resolution)


def _smooth(h: np.ndarray) -> np.ndarray:
    # Three-sample box filter rounds stair/block corners without removing their flat treads.
    a = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    a = (a[:-2] + a[1:-1] + a[2:]) / 3
    a = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    return (a[:, :-2] + a[:, 1:-1] + a[:, 2:]) / 3


def _tile(cfg: TerrainCfg, kind: int, difficulty: float, rng: np.random.Generator) -> np.ndarray:
    sx, sy = cfg.tile_size
    r = cfg.horizontal_scale
    nx, ny = round(sx / r), round(sy / r)
    x, y = np.meshgrid(np.arange(nx + 1) * r, np.arange(ny + 1) * r)
    edge = np.minimum(np.minimum(x, sx - x), np.minimum(y, sy - y))
    depth = np.clip(edge - 0.25, 0, (min(sx, sy) - cfg.platform_size) / 2 - 0.25)
    h = np.zeros_like(x)
    if kind == 1:
        # Uniform random control heights at >= 20 cm spacing, interpolated to the map grid.
        stride = max(1, int(np.ceil(0.20 / r)))
        cx, cy = np.arange(0, nx + stride, stride), np.arange(0, ny + stride, stride)
        noise = rng.uniform(-0.08 * difficulty, 0.08 * difficulty, (len(cy), len(cx)))
        a = np.stack([np.interp(np.arange(nx + 1), cx, row) for row in noise])
        h = np.stack([np.interp(np.arange(ny + 1), cy, col) for col in a.T], axis=1)
        h *= np.clip((edge - 0.15) / 0.25, 0, 1)
        center = h[ny // 2:(ny + 1) // 2 + 1, nx // 2:(nx + 1) // 2 + 1].mean()
        blend = np.clip((cfg.platform_size / 2 + 0.2 - np.abs(x - sx / 2)) / 0.2, 0, 1)
        blend *= np.clip((cfg.platform_size / 2 + 0.2 - np.abs(y - sy / 2)) / 0.2, 0, 1)
        h += blend * (center - h)
    elif kind in (2, 3):
        h = depth * np.tan(np.deg2rad(25 * difficulty)) * (1 if kind == 2 else -1)
    elif kind in (4, 5):
        h = np.floor((depth + 1e-8) / 0.30) * (0.05 + 0.15 * difficulty) * (1 if kind == 4 else -1)
        h = _smooth(h)
    elif kind == 6:
        # Disjoint rectangular blocks; the center block also provides a standing platform.
        for bx in np.arange(1.0, sx - 0.5, 1.75):
            for by in np.arange(1.0, sy - 0.5, 1.75):
                if abs(bx - sx / 2) < 1.5 and abs(by - sy / 2) < 1.5:
                    continue
                wx, wy = rng.uniform(0.35, 0.65, 2)
                block = (np.abs(x - bx) <= wx) & (np.abs(y - by) <= wy)
                h[block] = rng.uniform(0.05, 0.05 + 0.20 * difficulty)
        center = (np.abs(x - sx / 2) <= cfg.platform_size / 2) & (np.abs(y - sy / 2) <= cfg.platform_size / 2)
        h[center] = 0.05 + 0.20 * difficulty
        h[edge < 0.15] = 0
        h = _smooth(h)
    h = np.rint(h / cfg.vertical_scale) * cfg.vertical_scale
    h[[0, -1], :], h[:, [0, -1]] = 0, 0
    return h.astype(np.float32)


def generate(cfg: TerrainCfg, seed: int = 0) -> Terrain:
    """Rows increase difficulty along +x; columns choose types along +y.

    Proportions are normalized and stratified across columns (the same type per column
    at every difficulty). Heights are quantized in meters by ``vertical_scale``.
    Each tile has a level center platform for spawning; roughness blends into it over 20 cm.
    Default stair/block edges are rounded over three samples to keep the bilinear vs
    triangle surface error below 1 cm; custom scales should be checked with test_terrain.
    """
    r = cfg.horizontal_scale
    if not np.isfinite([*cfg.tile_size, r, cfg.vertical_scale, cfg.border_width, cfg.platform_size]).all():
        raise ValueError("Terrain dimensions and scales must be finite")
    if r <= 0 or cfg.vertical_scale <= 0 or cfg.border_width < 0 or cfg.platform_size <= 0:
        raise ValueError("Scales and platform size must be positive; border must be nonnegative")
    if any(not isinstance(n, int) or n < 1 for n in (cfg.num_rows, cfg.num_cols)):
        raise ValueError("num_rows and num_cols must be positive integers")
    if min(cfg.tile_size) <= cfg.platform_size + 0.5:
        raise ValueError("Tiles must fit the center platform plus a 0.25 m margin on each side")
    cells = np.asarray((*cfg.tile_size, cfg.border_width)) / r
    if not np.allclose(cells, np.rint(cells), rtol=0, atol=1e-6):
        raise ValueError("Tile size and border width must be multiples of horizontal_scale")
    weights = np.asarray(cfg.proportions, dtype=float)
    if weights.shape != (len(TILE_TYPES),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"proportions must contain {len(TILE_TYPES)} nonnegative weights with positive sum")
    nx, ny, border = np.rint(cells).astype(int)
    heights = np.zeros((cfg.num_cols * ny + 2 * border + 1, cfg.num_rows * nx + 2 * border + 1), np.float32)
    kinds = np.searchsorted(np.cumsum(weights / weights.sum()), (np.arange(cfg.num_cols) + 0.5) / cfg.num_cols)
    tile_types = np.broadcast_to(kinds, (cfg.num_rows, cfg.num_cols)).copy()
    origins = np.empty((cfg.num_rows, cfg.num_cols, 3), np.float32)
    rng = np.random.default_rng(seed)
    for i in range(cfg.num_rows):
        for j in range(cfg.num_cols):
            tile = _tile(cfg, int(kinds[j]), i / max(cfg.num_rows - 1, 1), rng)
            ix, iy = border + i * nx, border + j * ny
            heights[iy:iy + ny + 1, ix:ix + nx + 1] = tile
            z = tile[ny // 2:(ny + 1) // 2 + 1, nx // 2:(nx + 1) // 2 + 1].mean()
            origins[i, j] = ((i + 0.5) * cfg.tile_size[0], (j + 0.5) * cfg.tile_size[1], z)
    return Terrain(heights, origins, tile_types, r, (-cfg.border_width, -cfg.border_width))


def add_hfield(xml: str, terrain: Terrain) -> str:
    """Replace floor with an allocated hfield; call write_hfield after compilation.

    Filling the compiled buffer avoids encoding millions of elevations into XML.
    The geom origin is at the map center and minimum height; size[3] extends below it.
    """
    h = terrain.heights
    if h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all() or terrain.resolution <= 0:
        raise ValueError("Expected a finite height matrix of at least 2 x 2 and positive resolution")
    sx, sy = terrain.extent
    lo, hi = float(h.min()), float(h.max())
    root = ET.fromstring(xml)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "hfield", name="terrain", nrow=str(h.shape[0]), ncol=str(h.shape[1]),
                  size=f"{sx / 2} {sy / 2} {max(hi - lo, 1e-6)} 0.1")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name") == "floor":
            world.remove(geom)
    ET.SubElement(world, "geom", name="floor", type="hfield", hfield="terrain", contype="0", conaffinity="1", condim="3",
                  pos=f"{terrain.offset[0] + sx / 2} {terrain.offset[1] + sy / 2} {lo}")
    return ET.tostring(root, encoding="unicode")


def write_hfield(model: mujoco.MjModel, terrain: Terrain) -> None:
    """Fill normalized compiled data in row-major (+y, +x) order, before mjw.put_model.

    Unlike inline XML elevation, mjModel.hfield_data is bottom-to-top: no y flip here.
    """
    hf = model.hfield("terrain")
    h = terrain.heights
    if (int(hf.nrow[0]), int(hf.ncol[0])) != h.shape:
        raise ValueError("Heightfield allocation does not match terrain")
    hf.data[:] = (h - h.min()) / hf.size[2]


class HeightSampler:
    def __init__(self, terrain: Terrain, device: str | torch.device = "cuda:0"):
        self.heights = torch.tensor(terrain.heights, dtype=torch.float32, device=device)
        self.offset = torch.tensor(terrain.offset, dtype=torch.float32, device=device)
        self.resolution = terrain.resolution
        self.grid = torch.cartesian_prod(torch.linspace(-0.8, 0.8, 17, device=device), torch.linspace(-0.5, 0.5, 11, device=device))

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear world height for (..., 2), clamped to the map's inclusive bounds."""
        # non-finite queries (e.g. body positions of a blown-up world before rollback) must not index out of range
        uv = torch.nan_to_num((xy - self.offset) / self.resolution, nan=0.0, posinf=0.0, neginf=0.0)
        x = uv[..., 0].clamp(0, self.heights.shape[1] - 1)
        y = uv[..., 1].clamp(0, self.heights.shape[0] - 1)
        ix = x.long().clamp(max=self.heights.shape[1] - 2)
        iy = y.long().clamp(max=self.heights.shape[0] - 2)
        fx, fy = x - ix, y - iy
        h = self.heights
        return ((1 - fy) * ((1 - fx) * h[iy, ix] + fx * h[iy, ix + 1])
                + fy * ((1 - fx) * h[iy + 1, ix] + fx * h[iy + 1, ix + 1]))

    def scan(self, p: torch.Tensor, R: torch.Tensor, grid: torch.Tensor | None = None) -> torch.Tensor:
        """Root-relative p_z - h on an (n, 2) heading-frame grid, returning (B, n)."""
        grid = self.grid if grid is None else grid
        xy = p[:, None, :2] + grid @ yaw_mat(R)[:, :2, :2].transpose(-1, -2)
        return p[:, 2:3] - self.sample(xy)
