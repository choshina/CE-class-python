from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

try:
    import cma
except ImportError:
    cma = None

from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp
from ceclass.utils.stl_eval import max_rob0_vmap, min_rob0_vmap


@dataclass
class SynthResult:
    """Result of a parameter synthesis run."""
    satisfied: bool                        # True if some trace's robustness < 0 was found
    obj_best: float                        # Best objective value (-max robustness over traces)
    params_best: Optional[dict[str, float]] = None  # Best parameter values
    num_evals: int = 0
    time_spent: float = 0.0


class ParamSynthesis:
    """
    CMA-ES parameter synthesis with GPU-batched robustness evaluation.

    Searches for temporal parameter values (interval boundaries) that make
    at least one input trace a counterexample for the formula (robustness < 0).

    The objective minimizes −max_i(rob(NOT φ, σ_i)) over traces σ_i. A solution
    is "satisfied" when the best objective is < 0, meaning max_i(rob(NOT φ, σ_i)) > 0,
    i.e., some trace σ_i satisfies NOT φ (violates φ). This matches the paper's
    one-trace-at-a-time semantics via batch union.

    Port of MyParamSynthProblem.m with stlcgpp replacing Breach.
    """

    def __init__(
        self,
        formula: STLNode,
        traces: torch.Tensor,
        param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        device: Optional[torch.device] = None,
        dt: float = 1.0,
        max_time: float = 60.0,
        max_evals: int = 500,
        pop_size: Optional[int] = None,
        eval_devices: Optional[Sequence[torch.device]] = None,
    ):
        self.formula = formula
        self.traces = traces          # (num_traces, timesteps, dims)
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.device = device
        self.dt = dt
        self.max_time = max_time
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.eval_devices = eval_devices

        # Compute initial guess and bounds
        self.lb = np.array([param_bounds[p][0] for p in param_names])
        self.ub = np.array([param_bounds[p][1] for p in param_names])
        self.x0 = (self.lb + self.ub) / 2.0
        self.sigma0 = np.mean((self.ub - self.lb) / 4.0)

    def solve(self) -> SynthResult:
        """
        Run CMA-ES to find params where negated formula has robustness < 0.

        For single-parameter problems, falls back to scipy's minimize_scalar
        since CMA-ES requires dimension >= 2.
        """
        if cma is None:
            raise ImportError("cma package required. Install with: pip install cma")

        neg_formula = STLNode.negate(self.formula)
        start_time = time.time()
        num_evals = 0
        n_params = len(self.param_names)

        if n_params == 1:
            return self._solve_1d(neg_formula)

        opts = {
            'bounds': [self.lb.tolist(), self.ub.tolist()],
            'maxfevals': self.max_evals,
            'timeout': self.max_time,
            'verbose': -9,  # Suppress output
        }
        if self.pop_size is not None:
            opts['popsize'] = self.pop_size

        sigma0 = float(self.sigma0) if self.sigma0 > 0 else 1.0
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), sigma0, opts)

        while not es.stop():
            if time.time() - start_time >= self.max_time:
                break

            candidates = es.ask()
            fitnesses = self._batch_evaluate(candidates, neg_formula)
            num_evals += len(candidates)
            es.tell(candidates, fitnesses)

            # Early termination: found satisfying params
            if es.result.fbest < 0:
                break

        elapsed = time.time() - start_time
        best_x = es.result.xbest
        best_params = dict(zip(self.param_names, best_x)) if best_x is not None else None

        return SynthResult(
            satisfied=es.result.fbest < 0,
            obj_best=es.result.fbest,
            params_best=best_params,
            num_evals=num_evals,
            time_spent=elapsed,
        )

    def _solve_1d(self, neg_formula: STLNode) -> SynthResult:
        """Solve single-parameter synthesis using grid search + refinement."""
        start_time = time.time()
        lb, ub = float(self.lb[0]), float(self.ub[0])
        best_obj = float('inf')
        best_x = None
        num_evals = 0

        # Grid search with 20 points
        n_grid = min(20, self.max_evals)
        for val in np.linspace(lb, ub, n_grid):
            params = {self.param_names[0]: float(val)}
            try:
                max_rob = max_rob0_vmap(
                    lambda d: to_stlcgpp(neg_formula, params, d, self.dt),
                    self.traces,
                    self.device,
                    eval_devices=self.eval_devices,
                )
                obj = -max_rob
            except Exception:
                obj = 1e9
            num_evals += 1

            if obj < best_obj:
                best_obj = obj
                best_x = float(val)

            if best_obj < 0:
                break

            if time.time() - start_time >= self.max_time:
                break

        elapsed = time.time() - start_time
        best_params = {self.param_names[0]: best_x} if best_x is not None else None

        return SynthResult(
            satisfied=best_obj < 0,
            obj_best=best_obj,
            params_best=best_params,
            num_evals=num_evals,
            time_spent=elapsed,
        )

    def _batch_evaluate(self, candidates: list, neg_formula: STLNode) -> list[float]:
        """
        Evaluate all CMA-ES candidates. Each candidate is a parameter vector.

        For each candidate, compute robustness of NOT(φ) across ALL traces on GPU.
        The objective is −max_i(rob(NOT φ, σ_i)): we want to find params where
        the best (most-violated) trace has rob(NOT φ) > 0, i.e., some trace violates φ.
        """
        fitnesses = []
        for candidate in candidates:
            params = dict(zip(self.param_names, candidate))
            try:
                max_rob = max_rob0_vmap(
                    lambda d: to_stlcgpp(neg_formula, params, d, self.dt),
                    self.traces,
                    self.device,
                    eval_devices=self.eval_devices,
                )
                fitnesses.append(-max_rob)  # Minimize -max_rob to find any violating trace
            except Exception:
                fitnesses.append(1e9)  # Invalid params → large penalty
        return fitnesses

    def evaluate_direct(self, formula: STLNode) -> float:
        """
        Direct robustness evaluation (no parameters to search).
        Returns min robustness of phi at t=0 across all traces (most-violated trace).
        Negative means some trace violates the formula.
        """
        return min_rob0_vmap(
            lambda d: to_stlcgpp(formula, {}, d, self.dt),
            self.traces,
            self.device,
            eval_devices=self.eval_devices,
        )
