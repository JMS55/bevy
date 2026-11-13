---
title: "Solari: NEE sample counts are now local light counts"
pull_requests: []
---

Solari now samples directional lights apart from local (point, spot, rect, and emissive mesh) lights, with one directional light sample on top of the local light samples. The sample count settings on `SolariLighting` were renamed to say what they count:

- `primary_di_samples` is now `primary_local_light_samples`.
- `secondary_di_samples` is now `secondary_local_light_samples`.
- `world_cache_direct_light_sample_count` is now `world_cache_local_light_sample_count`.
