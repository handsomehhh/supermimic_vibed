"""Configuration of the terrain velocity task (shared by the MuJoCo-Warp simulator and the CPU player)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .head_camera import HeadCameraCfg
from .terrain import TerrainCfg, mjlab_rough_cfg
from .terrain_mdp import TerrainRewardCfg


@dataclass
class TerrainSetup:
    # physics map at 5 cm cells (finer cells overflow MuJoCo-Warp's 50-contact cap per geom-heightfield pair and
    # triple the blow-ups); the camera renders the same terrain from a 2 cm copy (camera_scale) so risers look
    # near-vertical (~82 deg) instead of the physics map's 5 cm ramp (~72 deg); step tops coincide
    terrain: TerrainCfg = field(default_factory=lambda: mjlab_rough_cfg(map_origin=(60.0, -85.0), base_height=1.5, horizontal_scale=0.05))
    camera_scale: float | None = 0.02
    seed: int = 0
    camera: HeadCameraCfg = field(default_factory=HeadCameraCfg)
    reward: TerrainRewardCfg = field(default_factory=TerrainRewardCfg)
    max_init_level: int = 5
    episode_steps: int = 1000  # 20 s
    cmd_x: tuple[float, float] = (-0.8, 1.0)
    cmd_y: tuple[float, float] = (-0.7, 0.7)
    cmd_yaw: tuple[float, float] = (-0.7, 0.7)
    forward_x: tuple[float, float] = (0.3, 1.0)
    resample_s: tuple[float, float] = (3.0, 8.0)
    standing: float = 0.1
    heading: float = 0.3
    forward: float = 0.2
    heading_stiffness: float = 0.5
    spawn_xy: float = 0.5
    curriculum: bool = True
    bounds_margin: float = 1.0


# named variants (train.py --terrain_variant; evaluation and the player read it from the checkpoint's args)
#  v1        exp8: mjlab legacy rewards, mjlab velocity-slot depth noise
#  stair     + stair swing clearance with lookahead (tracking scandot_stair_cfg), stronger edge (all levels) and
#            foothold penalties; depth realism: 0-60 ms latency, 5% dropped frames, 20% invalid pixels at depth jumps
#  stair_mem stair + 10 frames 2 apart (0.6 s of visual memory instead of 0.3 s: the ground within ~0.37 m of the
#            pelvis is outside the head camera's view, so a riser leaves the image ~0.4 s before the foot reaches it)
#  stair2    exp10a: stair with feet_edge on stair tiles only at weight -1 (exp9's -3 on every tile made stepping
#            onto any slope or step costly; exp9a stood still ~1 m into 60% of stairs-down and 50% of slope-down
#            trials, without a single fall), default foothold weight, swing clearance on the leading half of the foot
#  realism   exp10b (control): v1 rewards + stair's depth realism only
#  shin      exp10c: realism + a light shin clearance term only. exp10a stopped at the stair top in 20-35% of the
#            down trials although stopping scores lower in the true reward (6.99 vs 7.89 per step): the swing
#            clearance cost rises as the foot crosses the edge, a barrier on the only path down that the analytic
#            gradients of the imagined rollouts do not get over; the shin term is zero until a shin point is within
#            3 cm of the step below it
TERRAIN_VARIANTS = ("v1", "stair", "stair_mem", "stair2", "realism", "shin")


def make_setup(variant: str = "v1", seed: int = 0) -> TerrainSetup:
    from dataclasses import replace

    st = TerrainSetup(seed=seed)
    if variant == "v1":
        return st
    if variant not in TERRAIN_VARIANTS:
        raise ValueError(f"unknown terrain variant {variant!r}")
    st.camera = replace(st.camera, latency_steps=(0, 3), frame_drop=0.05, edge_dropout=0.2)
    if variant == "realism":
        return st
    if variant == "shin":
        st.reward = replace(st.reward, w_shin_clear=-0.5)
        return st
    if variant == "stair2":
        st.reward = replace(st.reward, w_swing_clear=-3.0, swing_front_only=True, w_edge=-1.0, edge_level=-1, edge_stairs_only=True)
        return st
    st.reward = replace(st.reward, w_swing_clear=-3.0, w_edge=-3.0, edge_level=-1, w_foothold=-2.0)
    if variant == "stair_mem":
        st.camera = replace(st.camera, stride=2)
    return st
