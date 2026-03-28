# CEClass Visualisation Results

Classification results for the TACAS paper experiments:
> *Counterexample Classification for Cyber-Physical Systems* (arXiv 2601.13743)

---

## What the figures show

Each PNG is a **Hasse diagram of the refinement lattice** for one benchmark/k/strategy
combination.  The lattice is a directed acyclic graph where:

- **Each node** is a *refined sub-formula* of the original STL specification.
  A node φ′ is *above* φ if φ′ ⇒ φ (φ′ is *stronger* — it describes a more
  specific failure mode).  The top node is the original spec; the bottom is
  `TRUE` (trivially satisfied, never a useful classification).

- **A directed edge** A → B means A is a direct implication of B (A is
  strictly stronger than B).

### Node colours

| Colour | Meaning |
|--------|---------|
| **Green** (covered) | A counterexample witness was found: at least one trace violates this sub-formula. CMA-ES found parameters (if the interval bounds are symbolic) such that the robustness is negative. |
| **Red** (pruned) | The strategy determined no trace can violate this sub-formula, so all sub-formulas beneath it were eliminated without testing. |
| **Grey** (active / untested) | The node was never reached by a pruning strategy. |

### What "covered" means intuitively

If node φ′ is covered, it means the 30 pre-generated falsifying traces contain
at least one trace that characteristically exhibits the failure pattern described
by φ′.  A larger number of covered nodes means the classification found more
*distinct* failure modes at the chosen temporal granularity k.

The one node that is almost never covered is **TRUE** — since `TRUE` is always
satisfied by every trace, its robustness is always non-negative.

---

## Benchmarks

| ID | Spec | Signals |
|----|------|---------|
| **AT1** | `alw_[0,30]( speed < 90 ∧ RPM < 4000 )` | speed (mph), RPM |
| **AT2** | `alw_[0,30]( brake < 250 ∨ ev_[0,5](speed < 30 ∧ RPM < 2000) )` | speed, RPM, brake |
| **AT3** | `alw_[0,30]( speed < 100 )` | speed (mph) |
| **AT5** | `ev_[0,30]( speed > 70 ∧ RPM > 3800 )` | speed, RPM |
| **AFC1** | `ev_[0,40]( alw_[0,10]( AF_err ∈ (−0.05, 0.05) ) )` | AF − AF_ref |

All traces come from 30 Simulink simulations of the Automatic Transmission
(`Autotrans_shift`) or Abstract Fuel Control (`AbstractFuelControl_M1`) models,
generated with the Breach falsification tool.  dt = 0.01 s throughout.

---

## Hierarchy depth k

The parameter k controls how finely the time intervals of temporal operators are
split in the refinement lattice.

| k | Interpretation |
|---|----------------|
| 1 | No splits — lattice contains only the original formula and its predicate-weakened variants |
| 2 | One split point t₂ is introduced, dividing each temporal interval into two sub-intervals |
| 3 | Two split points t₂, t₃ → three sub-intervals |
| 4 | Three split points t₂, t₃, t₄ → four sub-intervals |

Higher k yields a *larger* lattice (more refined formulas) and finer-grained
classification at the cost of more synthesis calls.

**Lattice sizes at k = 1…4:**

| Benchmark | k=1 | k=2 | k=3 | k=4 |
|-----------|-----|-----|-----|-----|
| AT1 | 4 | 16 | 64 | 208 |
| AT2 | 4 | 16 | 64 | 208 |
| AT3 | 2 | 4 | 8 | 12 |
| AT5 | 4 | 10 | 28 | 82 |
| AFC1 | 4 | 10 | 28 | 82 |

---

## Classification strategies

### `no_prune` (NoPrune / exhaustive)
Tests every node in the lattice regardless of prior results.  Gives an upper
bound on coverage but uses the maximum number of synthesis calls.

### `alw_mid` (AlwMid / midpoint)
At each step, selects the *midpoint* of the currently longest active path in
the lattice.  If a node is covered, all nodes above it (stronger) are also
marked covered without re-testing.  If not covered, all nodes below it
(weaker) are pruned.  Significantly fewer synthesis calls at large k.

---

## File naming

```
{BENCH}_k{K}_{STRATEGY}.png          — Hasse diagram
{BENCH}_k{K}_{STRATEGY}_covered.txt  — list of covered sub-formulas with
                                        best robustness value and eval count
overview.png                          — multi-panel summary figure
```

### Label notation in figures

Interval bounds that are symbolic (optimised by CMA-ES) are written as
subscripted split points: **t₂**, **t₃**, **t₄**.  Their numeric values are
found during synthesis and reported in the corresponding `_covered.txt` file.

For example, in AT3 k=2:

```
alw_[0,t₂](speed < 100)   — "always from 0 to some t₂ ∈ (0,30)"
alw_[t₂,30](speed < 100)  — "always from t₂ to 30"
```

If `alw_[t₂,30]` is covered but `alw_[0,t₂]` is not, the failure is
characteristically concentrated in the *second half* of the mission.

---

## Experiment summary (all k, all benchmarks)

Results from `results/summary.csv` — re-run after applying the `rob[:,0]`
robustness fix (uses t=0 global robustness instead of min over the full temporal
signal, avoiding stlcgpp's out-of-bounds −1e9 sentinel).

| Bench | k | Strategy | Classes | Covered | Time (s) |
|-------|---|----------|---------|---------|----------|
| AT1 | 1 | no_prune  | 4   | 3   | 0.21  |
| AT1 | 1 | alw_mid   | 4   | 3   | 0.04  |
| AT1 | 2 | no_prune  | 16  | 15  | 0.33  |
| AT1 | 2 | alw_mid   | 16  | 15  | 0.25  |
| AT1 | 3 | no_prune  | 64  | 63  | 9.26  |
| AT1 | 3 | alw_mid   | 64  | 63  | 3.62  |
| AT1 | 4 | no_prune  | 208 | 207 | 47.6  |
| AT1 | 4 | alw_mid   | 208 | 207 | 10.0  |
| AT2 | 1 | no_prune  | 4   | 3   | 0.06  |
| AT2 | 2 | no_prune  | 16  | 15  | 0.44  |
| AT2 | 3 | no_prune  | 64  | 63  | 14.3  |
| AT2 | 4 | no_prune  | 208 | 207 | 79.0  |
| AT3 | 1 | no_prune  | 2   | 1   | 0.01  |
| AT3 | 2 | no_prune  | 4   | 3   | 0.11  |
| AT3 | 3 | no_prune  | 8   | 7   | 0.80  |
| AT3 | 4 | no_prune  | 12  | 10  | 1.65  |
| AT5 | 1 | no_prune  | 4   | 3   | 0.03  |
| AT5 | 2 | no_prune  | 10  | 9   | 0.14  |
| AT5 | 3 | no_prune  | 28  | 27  | 4.14  |
| AT5 | 4 | no_prune  | 82  | 81  | 19.0  |
| AFC1 | 1 | no_prune | 4   | 2   | 0.16  |
| AFC1 | 2 | no_prune | 10  | 7   | 5.60  |
| AFC1 | 3 | no_prune | 28  | 24  | 29.0  |
| AFC1 | 4 | no_prune | 82  | 71  | 168.0 |

> **Note on AFC1 coverage**: The fix reduced AFC1 covered counts vs. the buggy
> version (which incorrectly reported 3/4, 9/10, 27/28, 81/82).  The previous
> bug made *all* temporal formulas appear covered because stlcgpp's −1e9
> sentinel (out-of-bounds `Always`) dominated `rob.min()`, giving every node a
> negative robustness.  The corrected counts reflect genuine falsifiability.
> AFC1 CMA-ES results also have higher variance across runs due to the
> stochastic optimiser.

All experiments run on an NVIDIA RTX 5090 using the `spoc` conda environment
with PyTorch 2.10.0+cu128 and `stlcgpp` for GPU-accelerated robustness
evaluation.  Max CMA-ES time per node: 20 s.
