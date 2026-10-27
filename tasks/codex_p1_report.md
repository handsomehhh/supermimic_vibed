Implemented the terrain module, optional robot-model integration, and validation/benchmark tool. CPU sampling and MuJoCo-Warp CPU standing checks pass. **The requested GPU validation and 4096-world throughput measurements remain blocked:** this session exposes no NVIDIA device nodes, `nvidia-smi` cannot communicate with the driver, torch reports CUDA unavailable / zero devices, and Warp reports CUDA error 100. No GPU results are inferred from CPU results.

Changes are limited to `supermimic/terrain.py`, `supermimic/robot.py`, `tools/test_terrain.py`, and this report. No dependencies were added, robot collision geometry and `TrackingSim` were not changed, and nothing was committed.

`TerrainCfg` defaults to 10 difficulty rows × 20 columns of 8 × 8 m tiles, 0.05 m horizontal resolution, 0.005 m height quantization, and a 5 m flat border. Seven types cover flat, uniform random roughness, positive/negative pyramid slopes, positive/negative pyramid stairs, and rectangular blocks. Difficulty scales linearly from row 0 to row 9. Maximum settings are ±0.08 m roughness, approximately 25° slopes, 0.20 m stair rise with nominal 0.30 m tread, and 0.25 m blocks. Roughness uses random control heights spaced at least 0.20 m apart. Stair/block edges have a three-sample box filter, retaining flat treads while bounding triangle-versus-bilinear error. Each tile has a 1.2 m level center for standing; roughness blends into that patch over 0.20 m.

`generate(cfg, seed)` returns float32 meter heights, tile-center origins with sampled z, type IDs, resolution, and lower-left offset. Difficulty rows run along +x, tile columns along +y. The default seed-42 map is **3401 × 1801 samples**, covers **90 × 170 m** including borders, and spans **−2 to +2 m**. Type counts are 20 flat, 40 rough, 30 each slope/stair type, and 20 boxes. The top-down preview is saved at `/tmp/terrain_preview.png`.

`HeightSampler.sample` performs batched clamped bilinear lookup. `scan` applies root yaw only and returns `p_z - h`; its default grid is 17 × 11 points over 1.6 × 1.0 m. Both operate as vectorized torch operations on the chosen device. Construction defaults to CUDA; CPU execution was used for the available checks.

`sim_model(timestep, terrain=None)` preserves the original path. With terrain, `add_hfield` allocates the asset and replaces the floor; `write_hfield` fills the compiled height buffer before `mjw.put_model`. This two-stage API avoids serializing millions of elevations into XML. Floor collision masks remain `contype=0`, `conaffinity=1`, `condim=3`.

The verified hfield convention is: compiled data is row-major, rows increase world y, columns increase world x, and the first sample is the map's lower-left corner. Inline XML elevations are vertically flipped by the compiler; direct `hfield_data` writes must not be flipped. The asset's horizontal half-extents are `(W−1)r/2` and `(H−1)r/2`. The geom is placed at the map center and minimum height. Data is `(h−min(h))/height_range`, `size[2]` is that range, and `size[3]=0.1` extends 10 cm below the minimum. Constant maps use zero elevation data and a positive dummy range. Tests verify asymmetric rectangular maps, negative and constant heights, boundary clamping, XML row reversal, and the base with an upward ray. This agrees with the [MuJoCo hfield reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-hfield).

MuJoCo's surface is triangular, so bilinear sampling is approximate between grid vertices. The per-cell maximum discrepancy is one quarter of the absolute mixed height difference. The test checks this bound across every cell and includes the worst cell's midpoint in its ray comparisons; it does not rely only on random smooth locations. Custom scales/configurations can exceed the 1 cm bound and need revalidation.

Measured results using `/home/lxz/envs/sm_local/bin/python` (torch 2.9.0+cu128, MuJoCo 3.8.0, MuJoCo-Warp 3.8.0, Warp 1.12.0):

| Check | Result |
|---|---|
| Sampler vs CPU `mj_ray`, 5,000 random xy + 200 centers + worst-cell midpoint | Max 0.00875601 m; mean 0.00004370 m; p99 0.00104644 m |
| All-cell bilinear/triangle discrepancy bound | 0.00875000 m |
| Exact triangle interpolation vs CPU `mj_ray` | Max 1.192e−7 m |
| Analytic bilinear lookup; yaw-only scan with nonzero roll/pitch; arbitrary batch shapes | Pass |
| Same-seed generation and flat borders | Pass |
| All seven single-type configurations, odd-sized rectangular tiles, eight invalid configurations | Pass |
| Original `HEAD` model vs both implicit/explicit `terrain=None`, timesteps 0.005 and 0.0025 | Byte-identical serialized models, 75,078,727 bytes each |
| `py_compile` and `git diff --check` | Pass |

The supplemental **MuJoCo-Warp CPU backend**, not CPU MuJoCo dynamics, ran 21 G1 worlds: all seven types at difficulty rows 0, 5, and 9, using default joint targets for 200 × 5 ms substeps. Every substep checked finite state, the existing simulator's velocity bounds, contact/constraint capacity, the lowest collision point against a CPU terrain ray, and 32,724 surface probes per robot against sampled heights. All worlds remained finite and within velocity limits.

| Tile type | Worst lowest point minus local ray height | Worst surface-probe clearance |
|---|---:|---:|
| Flat | −0.002978 m | −0.002978 m |
| Rough | −0.002978 m | −0.002978 m |
| Slope up | −0.002978 m | −0.002978 m |
| Slope down | −0.002978 m | −0.002978 m |
| Stairs up | −0.002978 m | −0.002978 m |
| Stairs down | −0.002979 m | −0.002979 m |
| Boxes | −0.002978 m | −0.002978 m |

Worst reported contact distance was −0.002978 m. Peak total contacts were 294; peak constraints per world were 56, below capacities 128 contacts/world and 600 constraints/world. The final run had no hfield overflow warnings. These are standing-platform contact checks, not demonstrations of walking on ramps, stair risers, or rough bumps.

The installed Warp dispatch table supports every required G1 pair: **hfield–sphere (8 geoms), hfield–cylinder (4), and hfield–mesh (24)**. No missing pair was found. The asset includes cylinders in addition to the collision types listed in the task.

An additional limitation was observed before adding the rough tile's level center patch: an open-loop stand on uneven roughness fell, emitted `height field collision overflow, number of collisions >= 50`, and exceeded the joint-velocity bound at substep 151 (0.755 s), reaching 120.013 rad/s in one world. Contacts involved hfield–mesh on the right knee/hip links; a hfield–sphere contact on `right_ankle_roll_link` reported an anomalous −1.835482 m distance. This does not establish that overflow caused the fall or identify the cause of that sphere distance. Warp's hfield convex kernel has a 50-contact-per-pair cap; increasing global `nconmax` does not remove it. A candidate small geometry change for the mesh overflow is explicit narrower capsule/sphere collision proxies on affected leg links, validated separately. Splitting the map into smaller hfields could also be investigated for the deep-prism contact anomaly. Neither change was applied. Falls and contacts outside the center platforms remain an open robustness issue, including on GPU.

Throughput is **unmeasured** for both terrain and plane. The tool implements the requested GPU benchmark with 4096 worlds, implicitfast, four 5 ms substeps, one CUDA graph per control step, `nconmax=128`, and `njmax=600`. It warms up, resets outside timed regions, and reports median world-control-steps/s and batched-control-steps/s over three 50-step trials. It measures simulator stepping with fixed targets, excluding policy/reward/reset overhead.

Commands and logs:

```bash
# Full required check; exits 2 in this session after CPU checks because CUDA is absent.
/home/lxz/envs/sm_local/bin/python tools/test_terrain.py

# Sampling/conventions only; passed.
/home/lxz/envs/sm_local/bin/python tools/test_terrain.py --cpu-only

# Supplemental Warp CPU contact checks; final run passed (exit 0).
MPLCONFIGDIR=/home/lxz/.cache/matplotlib /home/lxz/envs/sm_local/bin/python -u tools/test_terrain.py --warp-cpu
```

Final CPU results are in `/tmp/terrain_warp_cpu_final.log`; the initial failure diagnosis is in `/tmp/terrain_contact_debug.log`. Re-run the default command in a session exposing the RTX 4090 to complete GPU sampling, GPU contact validation, and both throughput measurements. The tool explicitly marks these checks NOT RUN when CUDA is absent and does not report the full suite as passing.
