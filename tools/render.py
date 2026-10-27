"""Render recorded rollouts (tools/record.py) to an MP4 with MuJoCo (EGL, no display needed).

Motion tracking: the simulated robot solid, the reference a translucent blue ghost placed at the robot's
xy (as the tracking reward compares it). Velocity commands: green arrow commanded, orange measured.
Usage: python tools/render.py rec/exp3.npz out.mp4 --episodes walk dance "command tour" --seconds 12
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import two_robot_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("out")
    ap.add_argument("--episodes", nargs="*", default=None, help="episodes whose name contains one of these (default: all)")
    ap.add_argument("--seconds", type=float, default=12.0, help="per episode")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its own xy")
    a = ap.parse_args()

    d = dict(np.load(a.rec))
    dt = float(d.get("dt", 0.02))
    eps = [("trk", c, str(n)) for c, n in enumerate(d["trk_names"])] + [("vel", e, str(n)) for e, n in enumerate(d.get("vel_names", []))]
    if a.episodes:
        eps = [e for e in eps if any(s in e[2] for s in a.episodes)]
    model = mujoco.MjModel.from_xml_string(two_robot_xml(a.width, a.height))
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = max(int(round(1 / (dt * a.fps))), 1)
    for kind, c, name in eps:
        p = kind + "_"
        T = int(d["trk_len"][c]) if kind == "trk" else d["vel_root_p"].shape[0]
        for i in range(0, min(T, int(a.seconds / dt)), stride):
            q = data.qpos
            q[0:3], q[3:7], q[7:36] = d[p + "root_p"][i, c], d[p + "root_quat"][i, c], d[p + "q"][i, c]
            if kind == "trk":
                q[36:39], q[39:43], q[43:72] = d["trk_ref_root_p"][i, c], d["trk_ref_root_quat"][i, c], d["trk_ref_q"][i, c]
                if not a.global_ghost:
                    q[36:38] = q[0:2]
                text = [f"{d['label']}   motion tracking: {name}", f"t = {float(d['trk_t'][i, c]):4.1f} s   clip return {float(d['trk_return'][c]):.3f}"]
            else:
                q[36:39] = (0, 0, -10)
                cmd, meas = d["vel_cmd"][i, c], d["vel_meas"][i, c]
                text = [f"{d['label']}   {name}: {d['vel_label'][i, c]}",
                        f"command  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  yaw {cmd[2]:+.2f}     measured  vx {meas[0]:+.2f}  vy {meas[1]:+.2f}  yaw {meas[2]:+.2f}"]  # fmt: skip
            if not d[p + "alive"][i, c]:
                text[1] += "   [fell / terminated]"
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [q[0], q[1], 0.75 if kind == "trk" else 0.5]
            cam.elevation = -12.0 if kind == "trk" else -32.0  # look down on the velocity arrows
            renderer.update_scene(data, camera=cam, scene_option=opt)
            if kind == "vel":
                w, x, y, z = q[3:7]
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                H = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                base = np.array([q[0], q[1], 0.02])
                for v, rgba in ((cmd, (0.15, 0.75, 0.35, 0.9)), (meas, (1.0, 0.55, 0.1, 0.9))):
                    vec = np.r_[H @ v[:2], 0.0]
                    scn = renderer.scene
                    if np.linalg.norm(vec) < 1e-3 or scn.ngeom >= scn.maxgeom:
                        continue
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.04, base, base + vec)
                    scn.ngeom += 1
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            for row, txt in enumerate(text):
                xy = (16, 12 + 34 * row)
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
