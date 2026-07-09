# weak-sindy-cfd-discovery

Reproducible Python experiments for noise-robust PDE model discovery with
weak variational SINDy, physically admissible sparse libraries, and
convective-diffusive benchmark systems.

This repository contains the code supporting the article:

> Computational Model Discovery for Noisy Convective-Diffusive Systems:
> Weak Variational SINDy with Physically Constrained Sparse Libraries

The article studies equation discovery as a computational inverse problem:
instead of differentiating noisy measurements directly, the weak formulation
transfers derivatives onto compact analytic test functions before sparse
regression. The validation focuses on viscous Burgers dynamics, repeated-noise
Monte Carlo tests, competitive sparse-regression baselines, non-iid observation
noise, enlarged candidate libraries, rollout diagnostics, computational cost,
and a periodic two-dimensional Navier-Stokes vorticity benchmark.

This is a code-only repository. Manuscript sources, journal templates,
compiled PDFs, and LaTeX submission folders are intentionally excluded.

## Scope

The implemented evidence is designed to test the numerical mechanism behind
weak variational identification for noisy convective-diffusive systems. It does
not claim to provide a full turbulence-closure model or a boundary-aware
velocity-pressure Navier-Stokes discovery solver.

The repository currently supports:

- 1D viscous Burgers equation discovery with strong and weak SINDy.
- Monte Carlo robustness over repeated Gaussian noise realizations.
- Strong-form baselines with smoothing, spectral filtering, ridge-STLSQ, SR3,
  and ElasticNet.
- Weak STLSQ, weak SR3, and weak stability selection.
- Realistic noise stress tests, including correlated, heteroscedastic,
  impulsive, and missing-data perturbations.
- Enlarged-library support recovery diagnostics.
- Declared positive-diffusion projection diagnostics for admissibility checks.
- Out-of-sample rollout and runtime/cost diagnostics.
- Periodic 2D vorticity identification as a controlled incompressible-flow
  stress test.

## Repository Layout

```text
src/weak_sindy_robust/   Python package
scripts/                 command-line experiment and aggregation scripts
configs/                 reproducible YAML experiment configurations
tests/                   pytest regression tests
docs/                    implementation notes
results/                 generated outputs, ignored by git except .gitkeep
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Quick Start

Run the baseline reproducibility experiment:

```bash
python scripts/run_experiment.py --config configs/burgers_reproduce.yaml
```

Run the main robustness and validation experiments:

```bash
python scripts/run_experiment.py --config configs/burgers_monte_carlo.yaml
python scripts/run_experiment.py --config configs/burgers_baselines.yaml
python scripts/run_experiment.py --config configs/burgers_noise_realistic.yaml
python scripts/run_experiment.py --config configs/burgers_ood_rollout.yaml
python scripts/run_experiment.py --config configs/burgers_large_library.yaml
python scripts/run_experiment.py --config configs/burgers_cost_scaling.yaml
python scripts/run_experiment.py --config configs/ns2d_vorticity_minimal.yaml
python scripts/run_experiment.py --config configs/ns2d_vorticity_monte_carlo.yaml
```

Outputs are written to `results/<experiment_name>/` and include resolved
configuration files, run manifests, coefficient tables, support tables, metrics,
runtime diagnostics, rollouts, and generated figures when `matplotlib` is
available.

## Experiment Map

| Configuration | Purpose |
| --- | --- |
| `burgers_reproduce.yaml` | Reproduce the core 1D Burgers strong/weak comparison. |
| `burgers_monte_carlo.yaml` | Estimate median error, IQR, and support recovery across repeated noise realizations. |
| `burgers_baselines.yaml` | Compare weak SINDy against strong-form preprocessing and regularized sparse baselines. |
| `burgers_noise_realistic.yaml` | Probe correlated, heteroscedastic, impulsive, and missing-data noise. |
| `burgers_ood_rollout.yaml` | Test whether identified coefficients remain predictive beyond the training horizon. |
| `burgers_large_library.yaml` | Test false-support attraction under an enlarged candidate library. |
| `burgers_cost_scaling.yaml` | Measure assembly, fit, rollout, and memory cost for different weak test counts. |
| `ns2d_vorticity_minimal.yaml` | Run the periodic 2D vorticity support-recovery stress test. |
| `ns2d_vorticity_monte_carlo.yaml` | Run 30 repeated noisy 2D vorticity trials at `sigma=0.02` and `sigma=0.05`. |

## Main Methods

- Strong-form STLSQ, temporal smoothing, spectral filtering, ridge-STLSQ, SR3,
  and ElasticNet baselines.
- Weak STLSQ, weak SR3, and weak stability selection using compact cosine test
  functions.
- Declared positive-diffusion projection diagnostics for admissibility checks.
- Enlarged-library support tests for Burgers dynamics.
- Periodic two-dimensional vorticity identification with strong and weak
  operators.
- A lightweight 2D vorticity Monte Carlo configuration that stores CSV
  diagnostics without large array outputs.

## Tests

```bash
python -m pytest
```

The test suite covers Fourier derivatives, Burgers integration, sparse
regression, weak assembly, metrics, and reproducibility.

## Reproducibility Notes

- The default rollout policy uses raw identified coefficients.
- Positive-diffusion projection variants are reported explicitly and are not
  silent rollout repairs.
- Random seeds and resolved configurations are stored with each run.
- Generated results and figures are reproducible from the scripts and configs.
- Generated outputs are ignored by git; keep permanent numerical artifacts in
  a release, archive, or supplementary data package.

## Citation

If this repository supports your work, please cite the associated manuscript
once publication metadata are available.
