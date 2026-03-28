# CEClass

Counterexample classification against Signal Temporal Logic (STL) specifications.

Given falsifying traces of a system, CEClass classifies *why* they fail by decomposing the STL spec into a lattice of refined sub-formulas, then systematically testing which refinements have counterexamples.

Python port of the [original MATLAB tool](https://github.com/choshina/CEClass) from the paper "Counterexample Classification against Signal Temporal Logic Specifications" (FM 2026). Uses [stlcg++](https://github.com/StanfordASL/stlcg) for GPU-accelerated STL robustness computation instead of Breach/Simulink.

## Installation

Requires Python >= 3.8 with PyTorch and stlcg++.

```bash
pip install torch stlcgpp cma numpy scipy
```

## Quick Start

```python
import torch
from ceclass.formula.stl_node import STLNode
from ceclass.strategies.long_bs import LongBSClassifier

# Define specification: alw_[0,30]((speed < 90) and (RPM < 4000))
speed = STLNode.predicate("speed", "<", 90.0, signal_index=0, node_id="speed_lt_90")
rpm = STLNode.predicate("RPM", "<", 4000.0, signal_index=1, node_id="RPM_lt_4000")
spec = STLNode.always_node(
    STLNode.and_node(speed, rpm, node_id="speed_and_RPM"),
    interval=(0, 30), node_id="alw_0_30"
)

# Hierarchy depth: k=2 splits for the always operator
k = [2, [1, [1], [1]]]

# Load or generate falsifying traces (num_traces, timesteps, signal_dims)
traces = torch.randn(10, 31, 2)

# Run classification
classifier = LongBSClassifier(formula=spec, k=k, traces=traces, dt=1.0)
result = classifier.solve()

print(f"Refined formulas: {result.num_classes}")
print(f"Covered (have counterexamples): {result.num_covered}")
print(f"Synthesis calls: {result.num_synth_calls}")
```

## Classification Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| **LongBS** (proposed) | `LongBSClassifier` | Binary search on longest path. O(log n) node tests. |
| BFS | `BFSClassifier` | Queue-based BFS from maxima with descendant pruning. |
| AlwMid | `AlwMidClassifier` | Test midpoint of longest path, bidirectional elimination. |
| BSRandom | `BSRandomClassifier` | Test midpoint of random path, bidirectional elimination. |
| NoPrune | `NoPruneClassifier` | Exhaustive baseline, tests all nodes. |

All strategies are in `ceclass.strategies` and share the same interface.

## How It Works

1. **Parse** the STL formula into a refinement lattice using hierarchy depth `k`. Temporal operators (always/eventually) are split into `k` sub-intervals with symbolic boundaries.

2. **Build** a DAG (PhiGraph) where edges represent logical implication between refined formulas. Transitive reduction gives immediate edges.

3. **Classify** by iteratively selecting lattice nodes, negating their formula, and running CMA-ES parameter synthesis to find interval boundary values that make the negated formula satisfiable (robustness < 0). The graph is pruned based on results:
   - **Satisfied** (counterexample exists): eliminate the node and all weaker ancestors.
   - **Failed** (no counterexample): eliminate the node and all stronger descendants.

4. **Output**: which refined formulas are covered (have counterexamples), classifying the failure modes.

## Package Structure

```
ceclass/
├── formula/
│   ├── stl_node.py      # STL formula tree (introspectable, for parsing)
│   └── converter.py      # STLNode → stlcg++ formula (for GPU robustness)
├── lattice/
│   ├── parser.py          # Formula → refinement lattice generator
│   ├── phi_node.py        # Node in the lattice
│   └── phi_graph.py       # DAG with pruning operations
├── strategies/
│   ├── base.py            # Shared classification logic
│   ├── long_bs.py         # Binary search on longest path (proposed)
│   ├── bfs.py             # BFS from maxima
│   ├── no_prune.py        # Exhaustive baseline
│   ├── alw_mid.py         # Midpoint of longest path
│   └── bs_random.py       # Midpoint of random path
├── synthesis/
│   └── param_synth.py     # CMA-ES with GPU-batched robustness
├── utils/
│   └── data.py            # Load traces from .mat / .npy / tensors
└── examples/
    └── autotrans.py       # Autotrans benchmark reproduction
```

## Parallelization

Compared to the sequential MATLAB original:

- **Trace-level**: `torch.vmap(formula)(traces)` evaluates robustness across all traces in a single GPU pass.
- **CMA-ES population**: Each candidate parameter set is evaluated with GPU-batched robustness.
- **1D optimization**: Falls back to grid search when CMA-ES is inapplicable (single parameter).

## Building Formulas

```python
from ceclass.formula.stl_node import STLNode

# Predicates (leaf nodes)
p = STLNode.predicate("speed", "<", 90.0, signal_index=0, node_id="p1")
q = STLNode.predicate("RPM", ">", 2000.0, signal_index=1, node_id="q1")

# Boolean operators
phi = STLNode.and_node(p, q, node_id="p_and_q")
phi = STLNode.or_node(p, q, node_id="p_or_q")
phi = STLNode.not_node(p, node_id="not_p")

# Temporal operators (interval in continuous time, converted to timesteps via dt)
phi = STLNode.always_node(p, interval=(0, 30), node_id="alw_p")
phi = STLNode.eventually_node(p, interval=(5, 20), node_id="ev_p")
```

## Hierarchy Depth `k`

The `k` parameter controls refinement granularity as a nested list matching the formula structure:

```python
# alw_[0,30]((speed < 90) and (RPM < 4000))
# k structure: [k_always, [k_and, [k_speed], [k_rpm]]]

k = [2, [1, [1], [1]]]  # Split always into 2 segments, no splitting for predicates
k = [3, [1, [1], [1]]]  # Split always into 3 segments (more refined lattice)
```

Higher `k` values produce larger lattices with finer-grained classification but require more synthesis calls.

## Example

```bash
# Run with synthetic data
python -m ceclass.examples.autotrans --strategy long_bs --k 2

# Run with real trace data
python -m ceclass.examples.autotrans --data test/data/AT1.mat --strategy long_bs --k 2 --device cuda


python -m ceclass.examples.benchmark --device cuda --max-time 5 --max-evals 50 --output ./test_bench_cpu.csv 2>&1

```