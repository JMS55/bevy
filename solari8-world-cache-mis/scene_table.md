# World cache MIS: full tables (solari8-world-cache-mis @ 053a5c407, lab PC RTX 5080)

Raw 640x360, 256 frames, ReSTIR off, off = before, on = after, alternating x3 in one process per scene. References: Solari's pathtracer on this base. Reference-only emitter mask. relvar = per-pixel temporal variance / (ref^2 + (0.1 mean ref)^2). fireflies/frame: pixels > 20x their 9x9 local mean in single frames (8 dumped frames). band: 2-16 px around the reference's emitter pixels.

## Zero-Day

| variant | bright | relvar_t999 | relvar | p99.99 | max | fireflies/frame | band t999 | band relvar |
|---|---|---|---|---|---|---|---|---|
| off | -14.97% | 14.1 | 78.7 | 6.24e+04 | 8.23e+06 | 577.6 | 17.6 | 37.9 |
| on | -15.34% | 12.9 | 19.9 | 1.36e+04 | 3.94e+04 | 554.5 | 17.1 | 27.1 |
| off2 | -16.77% | 10.6 | 27.8 | 2.56e+04 | 1.13e+06 | 535.2 | 14.8 | 32.8 |
| on2 | -15.75% | 11.9 | 19.7 | 1.41e+04 | 5.88e+04 | 556.5 | 15 | 26.2 |
| off3 | -15.78% | 11.6 | 31.1 | 1.78e+04 | 1.42e+06 | 549.8 | 16.4 | 27.7 |
| on3 | -15.52% | 10.9 | 18.1 | 1.44e+04 | 5.32e+04 | 550.6 | 14.2 | 22.8 |

## indoor (view 0)

| variant | bright | relvar_t999 | relvar | p99.99 | max | fireflies/frame | band t999 | band relvar |
|---|---|---|---|---|---|---|---|---|
| off | -4.41% | 5.43 | 5.93 | 1.04e+03 | 7.31e+03 | 673.4 | 5.16 | 5.32 |
| on | -4.35% | 5.24 | 5.32 | 87 | 4.59e+03 | 682.2 | 5.17 | 5.23 |
| off2 | -3.51% | 6.16 | 10.1 | 6.73e+03 | 2.2e+05 | 691.2 | 5.44 | 5.63 |
| on2 | -4.08% | 5.26 | 5.31 | 90.4 | 289 | 671.9 | 5.23 | 5.3 |
| off3 | -4.43% | 5.49 | 5.97 | 1.26e+03 | 1.1e+04 | 692.8 | 5.17 | 5.23 |
| on3 | -4.28% | 5.24 | 5.3 | 95.6 | 854 | 665.8 | 5.19 | 5.25 |

## Bistro

| variant | bright | relvar_t999 | relvar | p99.99 | max | fireflies/frame | band t999 | band relvar |
|---|---|---|---|---|---|---|---|---|
| off | -1.17% | 2.2 | 2.33 | 260 | 1.08e+03 | 781.6 | 1.4 | 1.58 |
| on | -1.09% | 2.21 | 2.32 | 208 | 778 | 793.0 | 1.41 | 1.59 |
| off2 | -1.13% | 2.2 | 2.32 | 186 | 3.38e+03 | 775.1 | 1.4 | 1.54 |
| on2 | -1.13% | 2.19 | 2.29 | 183 | 699 | 803.2 | 1.4 | 1.6 |
| off3 | -1.14% | 2.2 | 2.35 | 185 | 8.68e+03 | 795.0 | 1.39 | 1.65 |
| on3 | -1.17% | 2.21 | 2.33 | 214 | 3.17e+03 | 782.4 | 1.42 | 1.62 |

## many_lights

| variant | bright | relvar_t999 | relvar | p99.99 | max | fireflies/frame | band t999 | band relvar |
|---|---|---|---|---|---|---|---|---|
| off | -3.13% | 6.69 | 7.52 | 1.37e+03 | 1.02e+04 | 444.1 | 21.9 | 23.5 |
| on | -2.78% | 6.68 | 7.49 | 1.34e+03 | 1.09e+04 | 442.5 | 21.7 | 23.2 |
| off2 | -2.99% | 6.7 | 7.64 | 1.6e+03 | 1.27e+04 | 443.8 | 21.9 | 23.6 |
| on2 | -2.78% | 6.73 | 7.62 | 1.6e+03 | 6.37e+03 | 449.9 | 22.1 | 23.7 |
| off3 | -2.78% | 6.78 | 7.72 | 1.41e+03 | 7.69e+03 | 450.1 | 22.2 | 24 |
| on3 | -2.64% | 6.75 | 7.65 | 1.63e+03 | 1.05e+04 | 444.2 | 22.2 | 23.8 |

## default scene (pica pica)

| variant | bright | relvar_t999 | relvar | p99.99 | max | fireflies/frame | band t999 | band relvar |
|---|---|---|---|---|---|---|---|---|
| off | -0.63% | 4.2 | 10.3 | 3.94e+03 | 4.26e+05 | 486.0 | 21.4 | 53.6 |
| on | -0.66% | 4.08 | 5.85 | 1.56e+03 | 1.36e+05 | 484.1 | 9.43 | 73.6 |
| off2 | -0.72% | 4.11 | 5.52 | 2.27e+03 | 4.14e+04 | 496.1 | 8.79 | 20.6 |
| on2 | -0.47% | 4.16 | 6.66 | 2.73e+03 | 1.3e+05 | 507.6 | 11.6 | 46.8 |
| off3 | -0.46% | 4.14 | 5.63 | 2.26e+03 | 6.27e+04 | 486.4 | 10.2 | 14.1 |
| on3 | -0.63% | 4.12 | 8.19 | 2.28e+03 | 3.2e+05 | 495.1 | 7.22 | 49 |

## After DLSS-RR (1080p, 180 frames, alternating x3; relvar vs the 640 reference upscaled)

```
zd off   bright   -7.9%  t999 0.000353  untrimmed 0.0154  p99.99 5.41  max 6.35e+03
zd on    bright   -7.1%  t999 0.000302  untrimmed 0.00382  p99.99 1.74  max 1.54e+03
zd off2  bright   -8.3%  t999 0.000302  untrimmed 0.00136  p99.99 1.31  max 120
zd on2   bright   -8.0%  t999 0.000304  untrimmed 0.00112  p99.99 1.05  max 117
zd off3  bright   -7.8%  t999 0.000302  untrimmed 0.000985  p99.99 1.05  max 66.8
zd on3   bright   -6.6%  t999 0.000317  untrimmed 0.00118  p99.99 1.21  max 110
i0 off   bright   -2.9%  t999 0.000111  untrimmed 0.000121  p99.99 0.0175  max 0.141
i0 on    bright   -3.7%  t999 9.06e-05  untrimmed 0.000101  p99.99 0.0192  max 0.156
i0 off2  bright   -3.6%  t999 0.0001  untrimmed 0.00011  p99.99 0.0156  max 0.146
i0 on2   bright   -3.6%  t999 8.96e-05  untrimmed 9.93e-05  p99.99 0.0169  max 0.192
i0 off3  bright   -3.5%  t999 0.000107  untrimmed 0.000117  p99.99 0.0174  max 0.137
i0 on3   bright   -3.8%  t999 9.43e-05  untrimmed 0.000104  p99.99 0.0164  max 0.142
```

## Timing (1080p, DLSS-RR, ReSTIR off, 4 interleaved cycles, settle 2 s / measure 3 s; Solari = world_cache + lighting + presample_light_tiles, ms)

```
== zd
off      total med 1.760 fast 1.759 | world_cache med 0.181 fast 0.179 | lighting med 1.523 | reps total 1.759 1.771 1.761 1.759 | wc 0.182 0.179 0.182 0.180 (frames 483)
on       total med 1.760 fast 1.759 | world_cache med 0.181 fast 0.180 | lighting med 1.523 | reps total 1.759 1.761 1.760 1.760 | wc 0.180 0.182 0.182 0.181 (frames 490)
== i0
off      total med 0.694 fast 0.694 | world_cache med 0.120 fast 0.120 | lighting med 0.549 | reps total 0.733 0.694 0.694 0.694 | wc 0.120 0.120 0.120 0.120 (frames 850)
on       total med 0.694 fast 0.694 | world_cache med 0.120 fast 0.120 | lighting med 0.549 | reps total 0.694 0.694 0.694 0.694 | wc 0.120 0.120 0.120 0.120 (frames 865)
== bi
off      total med 3.003 fast 2.975 | world_cache med 0.230 fast 0.212 | lighting med 2.713 | reps total 3.153 2.975 3.006 3.001 | wc 0.217 0.212 0.243 0.243 (frames 469)
on       total med 2.997 fast 2.976 | world_cache med 0.215 fast 0.215 | lighting med 2.724 | reps total 2.976 3.191 3.017 2.978 | wc 0.215 0.216 0.238 0.215 (frames 479)
== ml
off      total med 0.379 fast 0.379 | world_cache med 0.119 fast 0.118 | lighting med 0.231 | reps total 0.379 0.379 0.379 0.380 | wc 0.118 0.119 0.119 0.119 (frames 864)
on       total med 0.380 fast 0.379 | world_cache med 0.119 fast 0.119 | lighting med 0.231 | reps total 0.379 0.380 0.379 0.380 | wc 0.119 0.119 0.119 0.119 (frames 853)
== pica
off      total med 0.562 fast 0.558 | world_cache med 0.122 fast 0.121 | lighting med 0.409 | reps total 0.558 0.577 0.558 0.566 | wc 0.122 0.121 0.122 0.121 (frames 873)
on       total med 0.558 fast 0.558 | world_cache med 0.122 fast 0.122 | lighting med 0.406 | reps total 0.558 0.558 0.558 0.558 | wc 0.122 0.122 0.122 0.122 (frames 880)
```
