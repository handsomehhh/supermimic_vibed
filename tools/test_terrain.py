"""Check terrain sampling, G1 contacts in MuJoCo-Warp, and CUDA graph throughput.

  /home/lxz/envs/sm_local/bin/python tools/test_terrain.py

Without CUDA the sampling checks still run, then exit 2 with GPU checks marked NOT RUN.
Use --cpu-only for sampling/conventions, or --warp-cpu for supplemental Warp CPU contacts.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from supermimic.robot import default_joint_pos_list, sim_model
from supermimic.rotations import axis_angle_to_mat
from supermimic.terrain import TILE_TYPES, HeightSampler, Terrain, TerrainCfg, add_hfield, generate, write_hfield


def surface_model(terrain: Terrain) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(add_hfield("<mujoco><worldbody/></mujoco>", terrain))
    write_hfield(m, terrain)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def ray_heights(m: mujoco.MjModel, d: mujoco.MjData, xy: np.ndarray) -> np.ndarray:
    top = float(m.geom("floor").pos[2] + m.hfield("terrain").size[2] + 10)
    heights = []
    geomid = np.empty(1, np.int32)
    for x, y in xy:
        dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0., 0., -1.]), None, True, -1, geomid)
        assert dist >= 0 and geomid[0] == m.geom("floor").id, (x, y, dist, geomid)
        heights.append(top - dist)
    return np.asarray(heights)


def triangle_heights(terrain: Terrain, xy: np.ndarray) -> np.ndarray:
    uv = (xy - terrain.offset) / terrain.resolution
    uv = np.clip(uv, 0, np.array(terrain.heights.shape[::-1]) - 1)
    ij = np.minimum(uv.astype(int), np.array(terrain.heights.shape[::-1]) - 2)
    x, y = ij.T
    fx, fy = (uv - ij).T
    h = terrain.heights
    return np.where(fy >= fx, (1 - fy) * h[y, x] + (fy - fx) * h[y + 1, x] + fx * h[y + 1, x + 1],
                    (1 - fx) * h[y, x] + (fx - fy) * h[y, x + 1] + fy * h[y + 1, x + 1])


def convention_checks(device: str) -> None:
    inline = mujoco.MjModel.from_xml_string('<mujoco><asset><hfield name="h" nrow="2" ncol="3" size="1 1 1 .1" elevation="0 1 2 3 4 5"/></asset></mujoco>')
    np.testing.assert_allclose(inline.hfield_data.reshape(2, 3), np.array([[3, 4, 5], [0, 1, 2]]) / 5, atol=3e-8)
    x, y = np.meshgrid(np.arange(7) * 0.2, np.arange(4) * 0.2)
    h = (-0.4 + 0.3 * x + 0.6 * y + 0.025 * x * y).astype(np.float32)
    terrain = Terrain(h, np.empty((0, 0, 3)), np.empty((0, 0), int), 0.2, (2.3, -1.7))
    sampler = HeightSampler(terrain, device)
    m, d = surface_model(terrain)
    base_start = np.array([2.5, -1.5, float(h.min()) - 0.2])
    base_dist = mujoco.mj_ray(m, d, base_start, np.array([0., 0., 1.]), None, True, -1, np.empty(1, np.int32))
    assert abs(base_dist - 0.1) < 1e-7
    xy = np.stack((x.ravel(), y.ravel()), -1) + terrain.offset
    np.testing.assert_allclose(ray_heights(m, d, xy), h.ravel(), atol=2e-7)
    rng = np.random.default_rng(7)
    uv = rng.uniform(0, terrain.extent, (128, 2))
    xy = uv + terrain.offset
    tensor = torch.tensor(xy, dtype=torch.float32, device=device)
    expected = -0.4 + 0.3 * uv[:, 0] + 0.6 * uv[:, 1] + 0.025 * uv[:, 0] * uv[:, 1]
    np.testing.assert_allclose(sampler.sample(tensor).cpu(), expected, atol=4e-7)
    np.testing.assert_allclose(ray_heights(m, d, xy), triangle_heights(terrain, xy), atol=2e-7)
    clamped = torch.tensor([[-100., -100.], [100., 100.]], device=device)
    np.testing.assert_allclose(sampler.sample(clamped).cpu(), h[[0, -1], [0, -1]], atol=1e-7)
    assert sampler.sample(tensor[0]).shape == ()
    assert sampler.sample(tensor.reshape(4, 8, 4, 2)).shape == (4, 8, 4)
    assert sampler.sample(tensor[:0]).shape == (0,)

    p = torch.tensor([[2.7, -1.4, 1.2], [3.0, -1.3, 0.8]], device=device)
    yaw = torch.tensor([0.7, -1.1], device=device)
    Rz = axis_angle_to_mat(torch.tensor([0., 0., 1.], device=device), yaw)
    Ry = axis_angle_to_mat(torch.tensor([0., 1., 0.], device=device), torch.tensor([0.3, -0.4], device=device))
    Rx = axis_angle_to_mat(torch.tensor([1., 0., 0.], device=device), torch.tensor([-0.2, 0.5], device=device))
    scan = sampler.scan(p, Rz @ Ry @ Rx)
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    gx, gy = sampler.grid.T
    xy_scan = torch.stack((p[:, 0:1] + c * gx - s * gy, p[:, 1:2] + s * gx + c * gy), -1)
    torch.testing.assert_close(scan, p[:, 2:3] - sampler.sample(xy_scan), atol=5e-7, rtol=0)
    assert scan.shape == (2, 187)
    torch.testing.assert_close(sampler.scan(p, Rz, sampler.grid[:5]), scan[:, :5], atol=5e-7, rtol=0)
    for z in (-0.7, 0., 0.4):
        flat = Terrain(np.full((3, 4), z, np.float32), terrain.origins, terrain.tile_types, 0.2, terrain.offset)
        fm, fd = surface_model(flat)
        np.testing.assert_allclose(ray_heights(fm, fd, np.array([[2.5, -1.5]])), z, atol=1e-7)
    print("conventions: XML y-flip, asymmetric 4 x 7 field, negative/constant heights, 10 cm base, bounds, bilinear lookup, yaw-only 187-point scan PASS")


def sampler_checks(terrain: Terrain, device: str, seed: int, n: int) -> None:
    h = terrain.heights
    cross = np.diff(np.diff(h, axis=0), axis=1)
    bound = float(np.abs(cross).max() / 4)
    assert bound <= 0.01, f"Bilinear/triangle error bound {bound:.6f} m exceeds 1 cm"
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, terrain.extent, (n, 2)) + terrain.offset
    iy, ix = np.unravel_index(np.abs(cross).argmax(), cross.shape)
    worst = (np.array([[ix, iy]]) + 0.5) * terrain.resolution + terrain.offset
    xy = np.concatenate((xy, terrain.origins[..., :2].reshape(-1, 2), worst))
    # Use precisely the same float32 positions in both torch and CPU ray queries.
    xy = xy.astype(np.float32)
    sampler = HeightSampler(terrain, device)
    sampled = sampler.sample(torch.tensor(xy, device=device)).cpu().numpy()
    m, d = surface_model(terrain)
    rays = ray_heights(m, d, xy)
    error = np.abs(sampled - rays)
    exact_error = np.abs(triangle_heights(terrain, xy) - rays).max()
    np.testing.assert_allclose(m.hfield_data.reshape(h.shape) * m.hfield_size[0, 2] + m.geom("floor").pos[2], h, atol=5e-7)
    assert error.max() < 0.01, f"Sampler/ray max error {error.max():.6f} m"
    assert exact_error < 2e-6, f"Triangle/ray max error {exact_error} m"
    center_heights = sampler.sample(torch.tensor(terrain.origins[..., :2], device=device)).cpu()
    np.testing.assert_allclose(center_heights, terrain.origins[..., 2], atol=2e-6)
    print(f"sampler vs mj_ray: {len(xy)} xy, max {error.max():.8f} m, mean {error.mean():.8f} m, p99 {np.quantile(error, .99):.8f} m")
    print(f"triangle vs mj_ray: max {exact_error:.3e} m; all-cell bilinear/triangle bound {bound:.8f} m")


def preview(terrain: Terrain, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 11))
    ox, oy = terrain.offset
    sx, sy = terrain.extent
    im = ax.imshow(terrain.heights, origin="lower", extent=(ox, ox + sx, oy, oy + sy), cmap="terrain", interpolation="nearest")
    ax.scatter(*terrain.origins[..., :2].reshape(-1, 2).T, s=3, c="black")
    ax.set(xlabel="x (m): increasing difficulty →", ylabel="y (m): terrain type", title="G1 terrain (tile centers in black)")
    fig.colorbar(im, ax=ax, label="Height (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"preview: {path}")


def collision_pairs(m: mujoco.MjModel) -> None:
    from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE
    from mujoco_warp._src.types import GeomType

    active = (m.geom_bodyid > 0) & (m.geom_contype != 0)
    missing = []
    for kind in np.unique(m.geom_type[active]):
        pair = (GeomType.HFIELD, GeomType(int(kind)))
        supported = pair in MJ_COLLISION_TABLE
        name = mujoco.mjtGeom(int(kind)).name
        print(f"Warp dispatch hfield / {name}: {'supported' if supported else 'MISSING'}, {sum(m.geom_type[active] == kind)} G1 geoms")
        if not supported:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing hfield collision pair(s): {missing}; smallest candidate fix: explicit sphere/capsule proxies on affected links. Geometry was not changed.")


def standing_poses(m: mujoco.MjModel, terrain: Terrain | None, origins: np.ndarray) -> np.ndarray:
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = 0, (1, 0, 0, 0), default_joint_pos_list()
    mujoco.mj_kinematics(m, d)
    feet = [m.body(s + "_ankle_roll_link").id for s in ("left", "right")]
    geoms = np.flatnonzero(np.isin(m.geom_bodyid, feet) & (m.geom_contype != 0))
    bottom = d.geom_xpos[geoms].copy()
    for i, g in enumerate(geoms):
        size = m.geom_size[g]
        extent = size[0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else abs(d.geom_xmat[g].reshape(3, 3)[2]) @ size
        bottom[i, 2] -= extent
    xy = origins[:, None, :2] + bottom[None, :, :2]
    heights = np.zeros(xy.shape[:-1]) if terrain is None else HeightSampler(terrain, "cpu").sample(torch.tensor(xy)).numpy()
    poses = np.tile(d.qpos, (len(origins), 1))
    poses[:, :2] = origins[:, :2]
    poses[:, 2] = (heights - bottom[:, 2]).max(-1) + 0.002
    return poses


class WarpSim:
    def __init__(self, m: mujoco.MjModel, poses: np.ndarray, device: str):
        self.device = device
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(m)
            self.data = mjw.make_data(m, nworld=len(poses), nconmax=128, njmax=600)
        self.qpos, self.qvel, self.ctrl = (wp.to_torch(a) for a in (self.data.qpos, self.data.qvel, self.data.ctrl))
        self.poses = torch.tensor(poses, device=device, dtype=self.qpos.dtype)
        self.reset()
        if device == "cpu":
            return
        with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
            for _ in range(4):
                mjw.step(self.model, self.data)
        self.graph = capture.graph
        with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
            mjw.step(self.model, self.data)
        self.substep_graph = capture.graph
        self.reset()

    def reset(self) -> None:
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model, self.data)
        self.qpos.copy_(self.poses)
        self.ctrl[:] = self.poses[:, 7:]

    def step(self, substep: bool = False) -> None:
        if self.device == "cpu":
            with wp.ScopedDevice(self.device):
                for _ in range(1 if substep else 4):
                    mjw.step(self.model, self.data)
        else:
            wp.capture_launch(self.substep_graph if substep else self.graph)

    def check(self) -> None:
        assert torch.isfinite(self.qpos).all() and torch.isfinite(self.qvel).all(), "Nonfinite Warp state"
        root_speed, joint_speed = self.qvel[:, :6].abs().amax(-1), self.qvel[:, 6:].abs().amax(-1)
        bad = (root_speed >= 50) | (joint_speed >= 100)
        assert not bad.any(), f"Warp velocity blow-up in worlds {bad.nonzero().flatten().tolist()}: root {root_speed.max():.3f}, joints {joint_speed.max():.3f}"
        assert int(self.data.nacon.numpy()[0]) <= self.data.naconmax, "Warp contact capacity exceeded"
        assert int(self.data.nefc.numpy().max()) <= self.data.njmax, "Warp constraint capacity exceeded"


def collision_points(m: mujoco.MjModel, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    points, geomids = [], []
    for g in np.flatnonzero((m.geom_bodyid > 0) & (m.geom_contype != 0)):
        kind, size = m.geom_type[g], m.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            a = m.mesh_vert[start:start + count]
            graph = m.mesh_graphadr[mesh]
            if graph >= 0:
                nv = m.mesh_graph[graph]
                a = a[m.mesh_graph[graph + 2 + nv:graph + 2 + 2 * nv]]
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            theta, phi = np.meshgrid(np.linspace(0, np.pi, 17), np.arange(32) * 2 * np.pi / 32)
            a = size[0] * np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), -1).reshape(-1, 3)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            angle = np.arange(64) * 2 * np.pi / 64
            a = np.array([(size[0] * np.cos(t), size[0] * np.sin(t), z) for z in (-size[1], size[1]) for t in angle])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            theta, phi = np.meshgrid(np.linspace(0, np.pi, 17), np.arange(32) * 2 * np.pi / 32)
            a = size[0] * np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), -1).reshape(-1, 3)
            a[:, 2] += np.where(a[:, 2] >= 0, size[1], -size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            a = np.array([(x, y, z) for x in (-size[0], size[0]) for y in (-size[1], size[1]) for z in (-size[2], size[2])])
        else:
            raise ValueError(f"Add clearance probes for geom {g}: {mujoco.mjtGeom(int(kind)).name}")
        points.append(a)
        geomids.extend([g] * len(a))
    return torch.tensor(np.concatenate(points), device=device, dtype=torch.float32), torch.tensor(geomids, device=device)


def contact_checks(m: mujoco.MjModel, terrain: Terrain, device: str) -> None:
    rows = np.unique([0, len(terrain.origins) // 2, len(terrain.origins) - 1])
    cols = [int(np.flatnonzero(terrain.tile_types[0] == k)[0]) for k in range(len(TILE_TYPES))]
    origins = terrain.origins[rows[:, None], cols].reshape(-1, 3)
    kinds = terrain.tile_types[rows[:, None], cols].ravel()
    sim = WarpSim(m, standing_poses(m, terrain, origins), device)
    points, geomids = collision_points(m, device)
    sampler = HeightSampler(terrain, device)
    lowest = np.full(len(origins), np.inf)
    surface_clearance = np.full(len(origins), np.inf)
    min_contact = np.inf
    max_contacts, max_constraints = 0, 0
    ray_m, ray_d = surface_model(terrain)
    spheres = torch.tensor(np.flatnonzero((m.geom_bodyid > 0) & (m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)), device=device)
    radii = torch.tensor(m.geom_size[spheres.cpu().numpy(), 0], device=device, dtype=torch.float32)
    if device != "cpu":
        with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
            mjw.kinematics(sim.model, sim.data)
        kinematics_graph = capture.graph
    print(f"Warp contact backend: {device}; testing {len(origins)} worlds at difficulty rows {rows.tolist()}", flush=True)
    failure = None
    for step in range(201):
        if step:
            sim.step(substep=True)
        try:
            sim.check()
        except AssertionError as exc:
            failure = exc
            print(f"FAIL at substep {step}: {exc}", flush=True)
            break
        if device == "cpu":
            with wp.ScopedDevice(device):
                mjw.kinematics(sim.model, sim.data)
        else:
            wp.capture_launch(kinematics_graph)
        gp = wp.to_torch(sim.data.geom_xpos)
        gR = wp.to_torch(sim.data.geom_xmat).reshape(len(origins), m.ngeom, 3, 3)
        world = gp[:, geomids] + (gR[:, geomids] @ points[None, :, :, None]).squeeze(-1)
        sphere_bottom = gp[:, spheres].clone()
        sphere_bottom[:, :, 2] -= radii
        world = torch.cat((world, sphere_bottom), dim=1)
        idx = world[:, :, 2].argmin(-1)
        low = world[torch.arange(len(origins), device=device), idx].cpu().numpy()
        lowest = np.minimum(lowest, low[:, 2] - ray_heights(ray_m, ray_d, low[:, :2]))
        clearance = world[:, :, 2] - sampler.sample(world[:, :, :2])
        surface_clearance = np.minimum(surface_clearance, clearance.amin(-1).cpu().numpy())
        nc = int(sim.data.nacon.numpy()[0])
        if nc:
            min_contact = min(min_contact, float(wp.to_torch(sim.data.contact.dist)[:nc].min()))
        max_contacts = max(max_contacts, nc)
        max_constraints = max(max_constraints, int(sim.data.nefc.numpy().max()))
    finite = torch.isfinite(sim.qpos).all(-1) & torch.isfinite(sim.qvel).all(-1)
    for kind, name in enumerate(TILE_TYPES):
        selected = kinds == kind
        status = "PASS" if finite.cpu().numpy()[selected].all() else "FAIL"
        print(f"Warp contact {name:11s}: lowest point - ray surface {lowest[selected].min():+.6f} m, min probe clearance {surface_clearance[selected].min():+.6f} m, finite {status} ({selected.sum()} worlds)")
    print(f"Warp contacts: {step * .005:.3f} s / {step} substeps, {len(origins)} worlds, {len(points)} surface probes/robot; min contact dist {min_contact:+.6f} m, peak contacts {max_contacts}, peak constraints/world {max_constraints}")
    if failure is not None:
        raise failure
    assert max_contacts > 0, "No terrain contacts detected"
    assert lowest.min() > -0.02, "Lowest robot point penetrated terrain by >= 2 cm"
    assert surface_clearance.min() > -0.02, "A collision surface probe penetrated terrain by >= 2 cm"
    assert min_contact > -0.02, "Warp contact penetration >= 2 cm"
    del sim
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()


def benchmark(m: mujoco.MjModel, terrain: Terrain | None, origins: np.ndarray, args: argparse.Namespace) -> float:
    base = standing_poses(m, terrain, origins)
    poses = base[np.arange(args.bench_envs) % len(base)]
    sim = WarpSim(m, poses, args.device)
    for _ in range(10):
        wp.capture_launch(sim.graph)
    torch.cuda.synchronize()
    rates = []
    for _ in range(args.bench_repeats):
        sim.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.bench_steps):
            wp.capture_launch(sim.graph)
        torch.cuda.synchronize()
        rates.append(args.bench_envs * args.bench_steps / (time.perf_counter() - start))
        sim.check()
    rate = float(np.median(rates))
    label = "terrain" if terrain is not None else "plane"
    print(f"throughput {label}: {rate:.0f} world control steps/s, {rate / args.bench_envs:.2f} batched control steps/s at {args.bench_envs} worlds; trials {[round(v) for v in rates]}")
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collision", choices=("mesh", "proxy"), default="mesh")
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--warp-cpu", action="store_true")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--bench-envs", type=int, default=4096)
    parser.add_argument("--bench-steps", type=int, default=50)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--preview", default="/tmp/terrain_preview.png")
    args = parser.parse_args()
    cuda = not (args.cpu_only or args.warp_cpu) and torch.cuda.is_available()
    device = args.device if cuda else "cpu"
    if cuda:
        from supermimic.loco_env import setup_device
        setup_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False  # sub-micrometer sampler assertions
    elif args.warp_cpu:
        wp.init()
        torch.set_num_threads(4)
    print(f"versions: torch {torch.__version__}, mujoco {mujoco.__version__}, mujoco_warp {mjw.__version__}, warp {wp.__version__}; sampler device {device}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    cfg = TerrainCfg()
    terrain = generate(cfg, args.seed)
    np.testing.assert_array_equal(terrain.heights, generate(cfg, args.seed).heights)
    border = round(cfg.border_width / cfg.horizontal_scale)
    assert not np.any(terrain.heights[:border]) and not np.any(terrain.heights[-border:])
    assert not np.any(terrain.heights[:, :border]) and not np.any(terrain.heights[:, -border:])
    print(f"terrain: {terrain.heights.shape}, extent {terrain.extent} m, height [{terrain.heights.min():.3f}, {terrain.heights.max():.3f}] m, seed {args.seed}; reproducibility and flat border PASS")
    print(f"tile counts: {dict(zip(TILE_TYPES, np.bincount(terrain.tile_types.ravel()).tolist()))}")
    preview(terrain, args.preview)
    convention_checks(device)
    sampler_checks(terrain, device, args.seed, args.samples)
    m = sim_model(terrain=terrain, collision=args.collision)
    assert m.geom("floor").type[0] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert m.geom("floor").contype[0] == 0 and m.geom("floor").conaffinity[0] == 1
    collision_pairs(m)
    if not cuda:
        if args.warp_cpu:
            contact_checks(m, terrain, "cpu")
        print("NOT RUN: GPU sampler, MuJoCo-Warp GPU G1 contact sanity, 4096-world terrain/plane CUDA graph throughput (CUDA unavailable or explicit CPU mode).")
        return 0 if args.cpu_only or args.warp_cpu else 2
    contact_checks(m, terrain, device)
    origins = terrain.origins.reshape(-1, 3)
    flat_rate = benchmark(sim_model(collision=args.collision), None, origins, args)
    terrain_rate = benchmark(m, terrain, origins, args)
    print(f"terrain / plane throughput: {terrain_rate / flat_rate:.3f}; simulation only, fixed default targets, implicitfast, 4 x 5 ms, one graph/control step, nconmax=128, njmax=600")
    print("ALL TERRAIN CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
