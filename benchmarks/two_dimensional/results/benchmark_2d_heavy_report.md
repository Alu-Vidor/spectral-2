# 2D SPFDE Benchmark Report

## Problem

Equation: `\epsilon (D_x^\alpha u + D_y^\alpha u) + u = f` on the interior tensor grid of `(0,1) x (0,1)`.

Manufactured solution: `u_{exact}(x,y) = (1 - \Psi(x))(1 - \Psi(y))`, with `\Psi(z) = E_\alpha(-z^\alpha / \epsilon)`.

Forcing: `f(x,y) = 1 - \Psi(x)\Psi(y)`.

## Configuration

- `alpha = 0.5`
- `epsilon = 1.0e-04`
- `FEPG-DEMM n_basis in [3, 5, 7]`
- `L1 FDM n_nodes in [50, 100, 150]`
- `dense_points = 181`

## Executive Summary

- Maximum observed peak RAM for dense 2D L1 FDM: `4746.32 MB`.
- Maximum observed peak RAM for 2D FEPG-DEMM: `0.45 MB`.
- Largest observed RAM ratio `RAM(FDM) / RAM(FEPG)` over the sweep: `1.130e+04`.
- Largest observed runtime ratio `time(FDM) / time(FEPG)` over the sweep: `5.879e+02`.

## Result Table

| method | requested | actual | matrix size | max error | cpu time (s) | peak RAM (MB) | note |
|---|---:|---:|---:|---:|---:|---:|---|
| 2D FEPG-DEMM | 3 | 3 | 9 x 9 | 1.22125e-15 | 1.59287e-02 | 0.42 | - |
| 2D FEPG-DEMM | 5 | 5 | 25 x 25 | 8.88178e-16 | 1.62327e-02 | 0.43 | - |
| 2D FEPG-DEMM | 7 | 7 | 49 x 49 | 9.99201e-16 | 1.54595e-02 | 0.45 | - |
| 2D L1 FDM | 50 | 50 | 2500 x 2500 | 8.03555e-04 | 2.48658e-01 | 143.09 | - |
| 2D L1 FDM | 100 | 100 | 10000 x 10000 | 1.12952e-03 | 4.18398e+00 | 2288.97 | - |
| 2D L1 FDM | 150 | 120 | 14400 x 14400 | 1.23584e-03 | 9.08901e+00 | 4746.32 | requested 150, auto-reduced to 120 by RAM guard |

Raw CSV: [benchmark_2d_heavy_results.csv](benchmark_2d_heavy_results.csv)

## 3D Surface Plot

![Surface plot](benchmark_2d_heavy_surface.png)

## Boundary-Layer Corner Zoom

![Corner zoom](benchmark_2d_heavy_corner_zoom.png)

## Boundary Cuts Near x = 0 and y = 0

![Boundary cuts](benchmark_2d_heavy_boundary_cuts.png)

## Performance Metrics

![Metrics](benchmark_2d_heavy_metrics.png)

