# Benchmarks

This directory is split by method family:

- `spectral/` contains benchmarks only for the spectral `FEPG-DEMM` solver.
- `alpha_shishkin_l1/` contains the benchmark only for the proposed `Alpha-Shishkin L1` method on the same canonical test problem as the spectral benchmark.
- `comparison/` contains the direct comparison between the spectral solver and the proposed `Alpha-Shishkin L1` method.

Run them from the repository root:

```bash
python -m benchmarks.spectral.benchmark_spectral
python -m benchmarks.alpha_shishkin_l1.benchmark_alpha_shishkin_l1
python -m benchmarks.comparison.benchmark_spectral_vs_alpha_shishkin_l1
```

Each script writes its outputs into its own `results/` subdirectory.
