Implemented the terrain module, sampler, integration, and test tool. Wrote the [report](tasks/codex_p1_report.md) and [preview](/tmp/terrain_preview.png).

CPU checks passed: maximum sampling error **8.756 mm**, standing penetration **under 3 mm**. GPU validation and throughput remain blocked because CUDA is unavailable. The report documents the Warp collision limitation found.