"""Opt-in stair reward math and world-frame contact forces (no policy inputs)."""
from __future__ import annotations

import torch
import warp as wp
from mujoco_warp._src.types import vec5


def clearance_cost(clearance, velocity_xy, contact, command, target: float):
    """mjlab feet_clearance, additionally gated to swing feet; heights at the sole."""
    active = command[:, :2].norm(dim=-1) + command[:, 2].abs() > .05
    return ((clearance - target).abs() * velocity_xy.norm(dim=-1) * ~contact).sum(-1) * active


def stumble_cost(force, ratio: float = 3.0):
    """mjlab MoRE feet_stumble: any foot with |Fxy| > ratio * |Fz|."""
    return (force[..., :2].norm(dim=-1) > ratio * force[..., 2].abs()).any(-1).float()


def nearest_edge_heading(xy, center, half_size, yaw):
    """Nearest rectangular tile edge; ties favor the robot's current heading."""
    directions = xy.new_tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    distance = half_size.repeat(2) - (xy - center) @ directions.T
    nearest = distance <= distance.amin(-1, keepdim=True) + 1e-5
    facing = torch.stack((yaw.cos(), yaw.sin()), -1) @ directions.T
    edge = facing.masked_fill(~nearest, -2).argmax(-1)
    return torch.atan2(directions[edge, 1], directions[edge, 0])


@wp.kernel
def body_contact_vectors(nacon: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
                         address: wp.array2d[int], force: wp.array2d[float], bodies: wp.array[int],
                         frame: wp.array[wp.mat33], friction: wp.array[vec5],
                         out: wp.array3d[float]):
    """Decode condim=3 pyramidal forces, rotate to world, average four substeps.

    The contact frame normal points from geom 0 to geom 1. The old scalar normal
    accumulator stays untouched, including its use for contact/termination rewards.
    Read all four addresses: the GPU builder does not guarantee contiguous rows.
    """
    i = wp.tid()
    if i < nacon[0]:
        w = world[i]
        pyramid = wp.vec4(0.0)
        for k in range(4):
            a = address[i, k]
            if a >= 0 and a < force.shape[1]:
                pyramid[k] = force[w, a]
        local = wp.vec3(pyramid[0] + pyramid[1] + pyramid[2] + pyramid[3],
                        (pyramid[0] - pyramid[1]) * friction[i][0],
                        (pyramid[2] - pyramid[3]) * friction[i][1])
        world_force = wp.transpose(frame[i]) * local
        for k in range(2):
            g = geom[i][k]
            if g >= 0:
                b = bodies[g]
                if b > 0:
                    sign = float(2 * k - 1)
                    for axis in range(3):
                        wp.atomic_add(out, w, b, axis, sign * world_force[axis] * 0.25)
