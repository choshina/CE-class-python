from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp
from ceclass.lattice.phi_graph import PhiGraph
from ceclass.lattice.phi_node import PhiNode
from ceclass.lattice.parser import Parser
from ceclass.synthesis.param_synth import ParamSynthesis, SynthResult
from ceclass.utils.stl_eval import min_rob0_vmap


@dataclass
class ClassificationResult:
    """Results of a classification run."""
    num_classes: int         # Total refined formulas in lattice
    num_covered: int         # Formulas with identified counterexamples
    time_split: float        # Time for parsing/lattice construction
    time_class: float        # Time for classification loop
    time_total: float        # Total time
    num_synth_calls: int     # Number of synthesis/robustness evaluations
    covered_nodes: list[PhiNode] = field(default_factory=list)


class BaseClassifier(ABC):
    """
    Abstract base class for counterexample classification strategies.

    All 5 strategies share:
    - The same initialization (parse formula → build lattice)
    - The same node testing logic (negate formula → CMA-ES → check robustness)
    - Different node selection and pruning strategies (implemented in solve())
    """

    def __init__(
        self,
        formula: STLNode,
        k: list,
        traces: torch.Tensor,
        device: Optional[torch.device] = None,
        dt: float = 1.0,
        max_time_per_node: float = 60.0,
        max_evals_per_node: int = 500,
        eval_devices: Optional[Sequence[torch.device]] = None,
    ):
        """
        Args:
            formula: Root STL formula to classify counterexamples for.
            k: Hierarchy depth config (nested list).
            traces: Falsifying traces, shape (num_traces, timesteps, dims).
            device: Torch device for GPU computation.
            dt: Timestep duration.
            max_time_per_node: Max CMA-ES time per node.
            max_evals_per_node: Max CMA-ES evaluations per node.
            eval_devices: Robustness vmap devices (``None`` → use both CUDA GPUs
                when available, else ``device``). Pass ``(device,)`` for single GPU.
        """
        self.traces = traces
        self.device = device
        self.dt = dt
        self.max_time_per_node = max_time_per_node
        self.max_evals_per_node = max_evals_per_node
        self.eval_devices = eval_devices

        # Parse formula into refinement lattice
        t_start = time.time()
        self.parser = Parser(formula, k)
        self.graph = self.parser.parse()
        self.time_split = time.time() - t_start

        self.num_classes = len(self.graph.nodes)
        self._num_synth_calls = 0

    @abstractmethod
    def solve(self) -> ClassificationResult:
        """Run the classification algorithm. Subclasses implement their strategy."""
        ...

    def _test_node(self, node: PhiNode) -> tuple[bool, Optional[SynthResult]]:
        """
        Test a single node: negate its formula, run synthesis, check robustness.

        Returns:
            (satisfied, synth_result): satisfied=True if counterexample exists.
        """
        self._num_synth_calls += 1
        param_names = node.formula.get_param_names()
        param_bounds = self.parser.get_param_bounds_for_node(node)

        if not param_names:
            # No parametric intervals — direct robustness check.
            # A node is "satisfied" (covered) if any input trace violates the formula
            # (robustness of phi < 0 for at least one trace). This matches the paper's
            # semantics: each counterexample trace is tested individually and a node is
            # covered if ANY trace falsifies it.
            #
            # IMPORTANT: torch.vmap(stl_formula)(traces) returns shape (num_traces, T)
            # where rob[:, 0] is the global robustness at t=0.  Using rob.min() would
            # pick up the -1e9 out-of-bounds sentinel stlcgpp emits for time windows
            # that extend past the trace, causing false positives for any formula whose
            # interval reaches the trace boundary.
            try:
                min_rob = min_rob0_vmap(
                    lambda d: to_stlcgpp(node.formula, {}, d, self.dt),
                    self.traces,
                    self.device,
                    eval_devices=self.eval_devices,
                )
                result = SynthResult(
                    satisfied=min_rob < 0,
                    obj_best=min_rob,
                    num_evals=1,
                )
                return min_rob < 0, result
            except Exception:
                return False, SynthResult(satisfied=False, obj_best=1e9)

        # CMA-ES parameter synthesis
        synth = ParamSynthesis(
            formula=node.formula,
            traces=self.traces,
            param_names=param_names,
            param_bounds=param_bounds,
            device=self.device,
            dt=self.dt,
            max_time=self.max_time_per_node,
            max_evals=self.max_evals_per_node,
            eval_devices=self.eval_devices,
        )
        result = synth.solve()
        return result.satisfied, result

    def _build_result(self, time_class: float) -> ClassificationResult:
        """Build the final classification result."""
        covered = self.graph.get_covered_nodes()
        return ClassificationResult(
            num_classes=self.num_classes,
            num_covered=len(covered),
            time_split=self.time_split,
            time_class=time_class,
            time_total=self.time_split + time_class,
            num_synth_calls=self._num_synth_calls,
            covered_nodes=covered,
        )
