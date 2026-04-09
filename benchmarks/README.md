# Benchmarks

This directory is split by method family:

- `spectral/` contains benchmarks only for the spectral `FEPG-DEMM` solver.
- `alpha_shishkin_l1/` contains benchmarks only for the proposed `Alpha-Shishkin L1` method, together with the uniform `L1` reference.
- `aeml_vpinn/` contains benchmarks for the asymptotically enriched variational `AEML-vPINN` method, together with the uniform `L1` reference.
- `two_dimensional/` contains the tensor-product 2D benchmark.

All 1D benchmark suites run the same two problems:

- `canonical`: the constant-coefficient problem from the article
- `objective_manufactured`: `u(x)=E_alpha(-x^alpha / epsilon) + x^2`, which keeps the boundary layer but is no longer matched exactly by the spectral singular corrector

Run them from the repository root:

```bash
python -m benchmarks.spectral.benchmark_spectral
python -m benchmarks.alpha_shishkin_l1.benchmark_alpha_shishkin_l1
python -m benchmarks.aeml_vpinn.benchmark_aeml_vpinn
python -m benchmarks.two_dimensional.benchmark_2d
```

Each script writes its outputs into its own `results/` subdirectory. For the 1D suites, files are prefixed by the problem key, for example `canonical_...` and `objective_manufactured_...`.
