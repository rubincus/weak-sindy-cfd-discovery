"""Experiment runner controlled by external configuration files."""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from weak_sindy_robust.aggregation import aggregate_results
from weak_sindy_robust.config import DEFAULT_CONFIG, deep_update, save_config
from weak_sindy_robust.grids import Dataset1D, Dataset2D
from weak_sindy_robust.io import environment_report, safe_slug, save_npz, write_csv, write_json
from weak_sindy_robust.libraries import library_from_config
from weak_sindy_robust.metrics import (
    coefficient_error,
    condition_number_subset,
    energy_violation_rate,
    rmse,
    support_from_coefficients,
    support_scores,
)
from weak_sindy_robust.noise import MissingDataMask, fill_missing_nearest_mean, noise_model_from_config
from weak_sindy_robust.plotting import make_standard_figures
from weak_sindy_robust.problems import (
    Burgers1DSpec,
    NS2DVorticitySpec,
    burgers_spec_from_config,
    large_library_true_coefficients,
    ns2d_spec_from_config,
)
from weak_sindy_robust.regressors import fit_regressor
from weak_sindy_robust.rollouts import rollout_burgers
from weak_sindy_robust.solvers import BurgersRK4Solver, NS2DVorticityRK4Solver
from weak_sindy_robust.strong_assembly import assemble_strong_system
from weak_sindy_robust.test_functions import test_function_config_from_method
from weak_sindy_robust.utils import timed, utc_now_iso
from weak_sindy_robust.weak_assembly import assemble_weak_system


class ExperimentRunner:
    """Run configured SINDy experiments and write standardized artifacts."""

    def __init__(self, config: dict, base_dir: str | Path | None = None):
        """Create a runner from a config dictionary."""
        self.config = deep_update(DEFAULT_CONFIG, config)
        self.base_dir = Path(base_dir or ".").resolve()
        self.output_dir = self._resolve_output_dir()
        self.coefficient_rows: list[dict[str, Any]] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.support_rows: list[dict[str, Any]] = []
        self.residual_rows: list[dict[str, Any]] = []
        self.runtime_rows: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {
            "experiment": self.config.get("experiment", {}).get("name"),
            "started_at_utc": utc_now_iso(),
            "status": "running",
        }

    def run(self) -> None:
        """Execute the configured experiment end to end."""
        self.prepare_output_dir()
        self.save_resolved_config()
        self.save_environment()

        problem = self.build_problem()
        clean = self.solve_clean(problem)
        self.save_clean_solution(clean)
        library = library_from_config(self.config)

        for noise_case in self.iter_noise_cases():
            dataset = self.apply_noise(clean, noise_case)
            self.save_noisy_case(dataset, noise_case)
            for method_cfg in self.config["identification"]["methods"]:
                self.run_single_identification(problem, clean, dataset, noise_case, method_cfg, library)

        self.write_tables()
        self.aggregate()
        if self.config.get("experiment", {}).get("save_figures", True):
            make_standard_figures(self.output_dir)
        self.manifest["finished_at_utc"] = utc_now_iso()
        self.manifest["status"] = "complete"
        write_json(self.output_dir / "run_manifest.json", self.manifest)

    def prepare_output_dir(self) -> None:
        """Create the output directory layout."""
        for subdir in [
            self.output_dir,
            self.output_dir / "noisy_cases",
            self.output_dir / "rollouts",
            self.output_dir / "summaries",
            self.output_dir / "figures",
        ]:
            subdir.mkdir(parents=True, exist_ok=True)

    def save_resolved_config(self) -> None:
        """Write the resolved configuration to the output directory."""
        save_config(self.config, self.output_dir / "config_resolved.yaml")

    def save_environment(self) -> None:
        """Write runtime package and platform information."""
        (self.output_dir / "environment.txt").write_text(environment_report(), encoding="utf-8")

    def build_problem(self) -> Burgers1DSpec | NS2DVorticitySpec:
        """Build the configured problem."""
        name = self.config.get("problem", {}).get("name", "burgers_1d")
        if name == "ns2d_vorticity_minimal":
            return ns2d_spec_from_config(self.config)
        if name != "burgers_1d":
            raise NotImplementedError(f"Unsupported problem: {name}")
        return burgers_spec_from_config(self.config)

    def solve_clean(self, problem: Burgers1DSpec | NS2DVorticitySpec) -> Dataset1D | Dataset2D:
        """Solve the clean reference problem."""
        if isinstance(problem, NS2DVorticitySpec):
            return NS2DVorticityRK4Solver(problem).solve()
        return BurgersRK4Solver(problem).solve()

    def save_clean_solution(self, data: Dataset1D | Dataset2D) -> None:
        """Write the clean solution arrays."""
        if not self.config.get("experiment", {}).get("save_arrays", True):
            return
        arrays = {"U_clean": data.U_clean, "x": data.x, "t": data.t, "metadata": data.metadata}
        if isinstance(data, Dataset2D):
            arrays["y"] = data.y
        save_npz(self.output_dir / "clean_solution.npz", **arrays)

    def iter_noise_cases(self) -> list[dict[str, Any]]:
        """Return noise cases, expanding Monte Carlo sweeps when requested."""
        noise_cfg = self.config.get("noise", {})
        mc_cfg = self.config.get("monte_carlo", {})
        if mc_cfg.get("enabled", False):
            sweep = dict(noise_cfg.get("sweep", {"type": "gaussian_iid", "sigmas": [0.05]}))
            sigmas = sweep.pop("sigmas", [sweep.get("sigma", 0.05)])
            n_repeats = int(mc_cfg.get("n_repeats", 30))
            seed_start = int(mc_cfg.get("seed_start", 1000))
            cases = []
            for repeat in range(n_repeats):
                seed = seed_start + repeat
                for sigma in sigmas:
                    case = dict(sweep)
                    case["sigma"] = float(sigma)
                    case["seed"] = int(seed)
                    cases.append(case)
            return cases
        return [dict(case) for case in noise_cfg.get("cases", [])]

    def apply_noise(self, clean: Dataset1D | Dataset2D, noise_case: dict[str, Any]) -> Dataset1D | Dataset2D:
        """Apply one configured noise case to a clean dataset."""
        seed = int(noise_case.get("seed", self.config.get("experiment", {}).get("seed_master", 0)))
        rng = np.random.default_rng(seed)
        model = noise_model_from_config(noise_case)
        metadata = dict(clean.metadata)
        metadata.update({"noise": dict(noise_case), "noise_seed": seed})
        if isinstance(model, MissingDataMask):
            masked, mask = model.apply(clean.U_clean, rng)
            U_noisy = fill_missing_nearest_mean(masked)
            metadata["missing_observation_fraction"] = float(1.0 - np.mean(mask))
        else:
            U_noisy = model.apply(clean.U_clean, rng)
        if isinstance(clean, Dataset2D):
            data = Dataset2D(U_clean=clean.U_clean, U_noisy=U_noisy, x=clean.x, y=clean.y, t=clean.t, metadata=metadata)
        else:
            data = Dataset1D(U_clean=clean.U_clean, U_noisy=U_noisy, x=clean.x, t=clean.t, metadata=metadata)
        data.validate()
        return data

    def save_noisy_case(self, data: Dataset1D | Dataset2D, noise_case: dict[str, Any]) -> None:
        """Write one noisy observation field."""
        if not self.config.get("experiment", {}).get("save_arrays", True):
            return
        noise_type = noise_case.get("type", "noise")
        sigma = noise_case.get("sigma", "na")
        seed = noise_case.get("seed", "na")
        name = f"noise_{safe_slug(noise_type)}_sigma_{safe_slug(sigma)}_seed_{safe_slug(seed)}.npz"
        arrays = {"U_noisy": data.U_noisy, "x": data.x, "t": data.t, "metadata": data.metadata}
        if isinstance(data, Dataset2D):
            arrays["y"] = data.y
        save_npz(self.output_dir / "noisy_cases" / name, **arrays)

    def run_single_identification(
        self,
        problem: Burgers1DSpec | NS2DVorticitySpec,
        clean: Dataset1D | Dataset2D,
        dataset: Dataset1D | Dataset2D,
        noise_case: dict[str, Any],
        method_cfg: dict[str, Any],
        library,
    ) -> None:
        """Assemble, fit, roll out and record one method/noise result."""
        method_name = method_cfg.get("name", "method")
        formulation = method_cfg.get("formulation") or ("weak" if method_name.startswith("weak") else "strong")
        U_obs = dataset.observed

        tracemalloc.start()
        with timed() as assembly_timer:
            if formulation == "weak":
                test_cfg = test_function_config_from_method(method_cfg)
                system = assemble_weak_system(U_obs, problem.grid, library, test_cfg)
            elif formulation == "strong":
                system = assemble_strong_system(U_obs, problem.grid, library, method_cfg)
            else:
                raise ValueError(f"Unknown formulation: {formulation}")

        with timed() as fit_timer:
            result = fit_regressor(system.A, system.b, system.feature_names, method_cfg)

        with timed() as rollout_timer:
            rollout_result = None
            rollout_cfg = self.config.get("rollout", {})
            if rollout_cfg.get("enabled", True) and isinstance(problem, Burgers1DSpec):
                T_rollout = rollout_cfg.get("T_rollout", problem.T)
                reference = clean.U_clean if abs(float(T_rollout) - problem.T) < 1.0e-14 else None
                rollout_result = rollout_burgers(
                    result.xi,
                    system.feature_names,
                    problem,
                    reference=reference,
                    T_rollout=float(T_rollout),
                    diffusion_policy=rollout_cfg.get("diffusion_policy", "raw"),
                    blowup_threshold=float(rollout_cfg.get("blowup_threshold", 50.0)),
                )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        true_xi, true_support = self.true_coefficients_and_support(library, problem)
        metric_cfg = self.config.get("metrics", {})
        selected_support = support_from_coefficients(
            result.xi,
            threshold_mode=metric_cfg.get("support_threshold_mode", "relative"),
            threshold_relative=float(metric_cfg.get("support_threshold_relative", 1.0e-4)),
            threshold_absolute=float(metric_cfg.get("support_threshold_absolute", 1.0e-10)),
        )
        support_metrics = support_scores(selected_support, true_support)
        rollout_metrics = self._rollout_metrics(rollout_result, problem)
        constraints_active = result.constraints.active if result.constraints else False

        identity = self._identity_row(method_cfg, noise_case, system, library)
        metrics_row = {
            **identity,
            "E_xi": coefficient_error(result.xi, true_xi),
            "residual_relative": result.residual_relative,
            **rollout_metrics,
            **support_metrics,
            "condition_true_support": condition_number_subset(system.A, true_support),
            "condition_identified_support": condition_number_subset(system.A, selected_support),
            "runtime_assembly_sec": assembly_timer.elapsed_sec,
            "runtime_fit_sec": fit_timer.elapsed_sec,
            "runtime_rollout_sec": rollout_timer.elapsed_sec,
            "memory_peak_mb": peak_bytes / (1024.0**2),
            "diffusion_policy": self.config.get("rollout", {}).get("diffusion_policy", "raw"),
            "constraints_active": bool(constraints_active),
        }
        self.metric_rows.append(metrics_row)
        self.residual_rows.append({**identity, "residual_relative": result.residual_relative})
        self.runtime_rows.append(
            {
                **identity,
                "runtime_assembly_sec": assembly_timer.elapsed_sec,
                "runtime_fit_sec": fit_timer.elapsed_sec,
                "runtime_rollout_sec": rollout_timer.elapsed_sec,
                "memory_peak_mb": peak_bytes / (1024.0**2),
            }
        )
        self.coefficient_rows.append(self._coefficient_row(identity, system.feature_names, result.xi, true_xi))
        self.support_rows.extend(
            self._support_rows(identity, system.feature_names, selected_support, true_support, result.xi, true_xi)
        )
        self.save_rollout(method_name, noise_case, rollout_result)

    def true_coefficients_and_support(
        self,
        library,
        problem: Burgers1DSpec | NS2DVorticitySpec,
    ) -> tuple[np.ndarray, set[int]]:
        """Return true coefficients and support matching a library."""
        cfg = self.config.get("library", {})
        if cfg.get("true_coefficients") and len(cfg.get("true_coefficients", [])) == len(library.names):
            xi_true = np.array(cfg["true_coefficients"], dtype=float)
        elif library.name == "burgers_large":
            xi_true = large_library_true_coefficients(problem.nu)
        elif isinstance(problem, NS2DVorticitySpec):
            xi_true = problem.true_coefficients
        else:
            xi_true = problem.true_coefficients

        if cfg.get("true_support") and len(cfg.get("true_support", [])) > 0:
            true_support = set(int(i) for i in cfg["true_support"])
        elif library.name == "burgers_large":
            true_support = {5, 7}
        elif isinstance(problem, NS2DVorticitySpec):
            true_support = problem.true_support
        else:
            true_support = problem.true_support
        return xi_true, true_support

    def save_rollout(self, method_name: str, noise_case: dict[str, Any], rollout_result) -> None:
        """Write a rollout NPZ file when available."""
        if rollout_result is None or not self.config.get("experiment", {}).get("save_arrays", True):
            return
        name = (
            f"rollout_{safe_slug(method_name)}_{safe_slug(noise_case.get('type', 'noise'))}"
            f"_sigma_{safe_slug(noise_case.get('sigma', 'na'))}_seed_{safe_slug(noise_case.get('seed', 'na'))}.npz"
        )
        arrays = {
            "U_model": rollout_result.U,
            "x": rollout_result.grid.x,
            "t": rollout_result.grid.t,
            "metadata": rollout_result.metadata,
        }
        if rollout_result.reference is not None:
            arrays["U_reference"] = rollout_result.reference
        save_npz(self.output_dir / "rollouts" / name, **arrays)

    def write_tables(self) -> None:
        """Write standardized CSV tables."""
        write_csv(self.output_dir / "coefficients.csv", self.coefficient_rows)
        write_csv(self.output_dir / "metrics.csv", self.metric_rows)
        write_csv(self.output_dir / "supports.csv", self.support_rows)
        write_csv(self.output_dir / "residuals.csv", self.residual_rows)
        write_csv(self.output_dir / "runtimes.csv", self.runtime_rows)

    def aggregate(self) -> None:
        """Write summary CSV files."""
        aggregate_results(self.output_dir)

    def _resolve_output_dir(self) -> Path:
        """Resolve output_dir relative to the project base directory."""
        output = Path(self.config.get("experiment", {}).get("output_dir", "results/run"))
        if not output.is_absolute():
            output = self.base_dir / output
        return output

    def _identity_row(self, method_cfg: dict, noise_case: dict, system, library) -> dict[str, Any]:
        """Return common identifying columns for result tables."""
        tf = method_cfg.get("test_functions", {})
        return {
            "experiment": self.config.get("experiment", {}).get("name"),
            "problem": self.config.get("problem", {}).get("name", "burgers_1d"),
            "method": method_cfg.get("name"),
            "formulation": method_cfg.get("formulation") or system.metadata.get("formulation"),
            "regressor": method_cfg.get("regressor", "stlsq"),
            "noise_type": noise_case.get("type", "none"),
            "sigma": noise_case.get("sigma", 0.0),
            "seed": noise_case.get("seed", ""),
            "lambda": method_cfg.get("lambda", method_cfg.get("lam", "")),
            "ax": tf.get("ax", system.metadata.get("ax", "")),
            "at": tf.get("at", system.metadata.get("at", "")),
            "ay": tf.get("ay", system.metadata.get("ay", "")),
            "nxc": tf.get("nxc", system.metadata.get("nxc", "")),
            "nyc": tf.get("nyc", system.metadata.get("nyc", "")),
            "ntc": tf.get("ntc", system.metadata.get("ntc", "")),
            "n_test_functions": system.metadata.get("n_test_functions", 0),
            "library_name": library.name,
            "n_features": len(system.feature_names),
        }

    def _coefficient_row(
        self,
        identity: dict[str, Any],
        feature_names: list[str],
        xi: np.ndarray,
        xi_true: np.ndarray,
    ) -> dict[str, Any]:
        """Return a wide coefficient row."""
        row = dict(identity)
        for name, coeff, true_coeff in zip(feature_names, xi, xi_true):
            slug = safe_slug(name)
            row[f"coef_{slug}"] = float(coeff)
            row[f"true_{slug}"] = float(true_coeff)
        return row

    def _support_rows(
        self,
        identity: dict[str, Any],
        feature_names: list[str],
        selected_support: set[int],
        true_support: set[int],
        xi: np.ndarray,
        xi_true: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return long-form support rows."""
        rows = []
        for idx, name in enumerate(feature_names):
            rows.append(
                {
                    **identity,
                    "term_index": idx,
                    "term": name,
                    "selected": idx in selected_support,
                    "true_active": idx in true_support,
                    "coefficient": float(xi[idx]),
                    "true_coefficient": float(xi_true[idx]),
                }
            )
        return rows

    def _rollout_metrics(self, rollout_result, problem: Burgers1DSpec) -> dict[str, Any]:
        """Return rollout RMSE and energy metrics."""
        if rollout_result is None:
            return {
                "rollout_rmse_train": float("nan"),
                "rollout_rmse_extrap": float("nan"),
                "energy_violation_rate": float("nan"),
                "stable_rollout": False,
            }
        U = rollout_result.U
        ref = rollout_result.reference
        train_rmse = rollout_result.rmse
        extrap_rmse = float("nan")
        if ref is not None and ref.shape == U.shape and np.isfinite(U).all():
            train_len = min(problem.Nt, U.shape[0])
            train_rmse = rmse(U[:train_len], ref[:train_len])
            if U.shape[0] > train_len:
                extrap_rmse = rmse(U[train_len - 1 :], ref[train_len - 1 :])
        if np.isfinite(U).all():
            evr = energy_violation_rate(U, rollout_result.grid.dx)
        else:
            evr = float("nan")
        return {
            "rollout_rmse_train": train_rmse,
            "rollout_rmse_extrap": extrap_rmse,
            "energy_violation_rate": evr,
            "stable_rollout": bool(rollout_result.stable),
        }
