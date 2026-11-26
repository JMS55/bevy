# Scene set: legacy cache vs energy fix (lab PC RTX 5080, 640x360, raw 1 spp, 256 frames, ReSTIR off)

Measured on exp/energy-loss (before the PR's base). Columns from noise.py: brightness vs Solari's pathtracer (reference-only emitter mask), relvar_t999, untrimmed relvar, p99/max of per-pixel relative variance, black-pixel fraction, single-frame p99/mean and CV. base = legacy cache, def = albedo + cache MIS, b_on = + last-bounce cache read (= the PR).

## Zero-Day

| variant | bright | relvar_t999 | relvar (untrimmed) | p99 | max | zero px % | frame p99/mean | frame CV |
|---|---|---|---|---|---|---|---|---|
| base | -10.4% | 43 | 3.01e+03 | 545 | 2.38e+08 | 27.5% | 14.28 | 10.67 |
| base2 | -17.2% | 10.8 | 26.6 | 219 | 1.18e+06 | 27.5% | 13.17 | 8.99 |
| def | -11.4% | 10.6 | 17.6 | 214 | 9.61e+04 | 27.3% | 13.65 | 15.40 |
| def2 | -10.6% | 11.6 | 18.2 | 244 | 3.93e+04 | 27.5% | 14.11 | 4.88 |
| b_on | +0.1% | 18.2 | 26.7 | 417 | 9.03e+04 | 24.4% | 15.30 | 5.62 |

## indoor-view-0

| variant | bright | relvar_t999 | relvar (untrimmed) | p99 | max | zero px % | frame p99/mean | frame CV |
|---|---|---|---|---|---|---|---|---|
| base | -3.5% | 6.49 | 15.2 | 33.7 | 2.29e+05 | 5.7% | 14.59 | 3.22 |
| base2 | -4.0% | 5.77 | 7.21 | 22.9 | 2.09e+04 | 5.7% | 14.40 | 2.76 |
| def | +0.0% | 5.24 | 5.29 | 17.8 | 949 | 5.7% | 14.45 | 2.59 |
| def2 | -0.0% | 5.24 | 5.3 | 17.7 | 404 | 5.7% | 14.87 | 2.58 |
| b_on | -0.1% | 5.24 | 5.31 | 17.8 | 3.61e+03 | 5.8% | 14.59 | 2.62 |

## indoor-view-1

| variant | bright | relvar_t999 | relvar (untrimmed) | p99 | max | zero px % | frame p99/mean | frame CV |
|---|---|---|---|---|---|---|---|---|
| base | -3.0% | 7.29 | 10 | 42.1 | 2.18e+04 | 4.8% | 17.36 | 3.40 |
| base2 | -3.5% | 6.71 | 7.2 | 40.4 | 2.75e+03 | 4.9% | 18.52 | 4.37 |
| def | +0.6% | 5.88 | 6.07 | 21.4 | 1.21e+04 | 4.9% | 16.77 | 3.12 |
| def2 | +0.7% | 5.87 | 6 | 21 | 670 | 4.9% | 17.80 | 3.22 |
| b_on | +1.0% | 5.89 | 6.03 | 21.4 | 677 | 4.7% | 17.76 | 2.73 |

## many_lights

| variant | bright | relvar_t999 | relvar (untrimmed) | p99 | max | zero px % | frame p99/mean | frame CV |
|---|---|---|---|---|---|---|---|---|
| base | -2.7% | 6.89 | 7.82 | 117 | 9.21e+03 | 88.3% | 11.58 | 14.54 |
| base2 | -2.6% | 6.96 | 7.86 | 119 | 5.58e+03 | 88.3% | 11.22 | 14.43 |
| def | -2.6% | 6.9 | 7.89 | 119 | 6.09e+03 | 88.4% | 12.10 | 13.51 |
| def2 | -2.9% | 6.86 | 7.73 | 116 | 4.51e+03 | 88.2% | 12.20 | 14.06 |
| b_on | -2.9% | 6.9 | 7.8 | 117 | 1.02e+04 | 88.3% | 11.63 | 13.68 |

## Bistro

| variant | bright | relvar_t999 | relvar (untrimmed) | p99 | max | zero px % | frame p99/mean | frame CV |
|---|---|---|---|---|---|---|---|---|
| base | -1.3% | 2.21 | 2.32 | 18.2 | 3.83e+03 | 25.5% | 9.53 | 90.73 |
| base2 | -1.3% | 2.2 | 2.3 | 18 | 665 | 25.5% | 9.54 | 89.24 |
| def | +0.4% | 2.38 | 2.48 | 18.6 | 487 | 24.7% | 9.58 | 86.68 |
| def2 | +0.4% | 2.38 | 2.5 | 18.5 | 6.43e+03 | 24.8% | 9.60 | 90.81 |
| b_on | +0.4% | 2.38 | 2.48 | 18.8 | 698 | 24.7% | 9.63 | 86.13 |
