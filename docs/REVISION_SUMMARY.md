# Revision summary

This note records the code-facing changes prepared for the manuscript revision.
The manuscript PDF and LaTeX sources remain outside this code-only repository.

## New 2D Monte Carlo benchmark

Added `configs/ns2d_vorticity_monte_carlo.yaml` to run a repeated-noise
periodic vorticity benchmark:

- problem: 2D periodic vorticity equation,
  `omega_t = -u omega_x - v omega_y + nu Delta omega`;
- grid: `64 x 64 x 101`, `T = 1.0`, `nu = 0.001`;
- repeats: 30 seeds per noise level;
- noise levels: `sigma = 0.02` and `sigma = 0.05`;
- methods: `strong_stlsq_ns2d` and `weak_stlsq_ns2d`;
- outputs: CSV diagnostics only, with arrays and figures disabled.

The revision run produced the following aggregate results:

| Method | sigma | Median `E_xi` | IQR `E_xi` | Median F1 | Exact support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong STLSQ | 0.02 | 2.65e-1 | 3.44e-3 | 0.364 | 0/30 |
| Weak STLSQ | 0.02 | 5.90e-4 | 3.42e-4 | 1.000 | 30/30 |
| Strong STLSQ | 0.05 | 6.87e-1 | 7.51e-3 | 0.364 | 0/30 |
| Weak STLSQ | 0.05 | 9.73e-4 | 4.76e-4 | 1.000 | 17/30 |

## Scope clarification

The manuscript revision narrows the claims to the validated contribution:
noise-robust weak sparse identification for synthetic convective-diffusive
benchmarks and a periodic 2D vorticity stress test. Tensor-basis closure,
Hamiltonian/symplectic consistency, pressure projection, and full
velocity-pressure Navier-Stokes constraints are documented as future-work or
conceptual components, not as numerically validated claims in this repository.

## Reproducibility

Run the new benchmark with:

```bash
python scripts/run_experiment.py --config configs/ns2d_vorticity_monte_carlo.yaml
```

The generated `results/` directory is intentionally ignored by git. Permanent
large numerical outputs should be distributed as supplementary data or release
assets.
