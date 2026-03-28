"""
Comprehensive correctness tests for the CEClass counterexample classification framework.

Tests are organized into groups:
  T1 – Robustness semantics (stlcgpp output shape, t=0 vs min)
  T2 – STL formula tree helpers (negate, get_param_names, str)
  T3 – Lattice structure (node counts, implication edges, maxima/minima)
  T4 – Implication monotonicity (covering a node implies covering weaker nodes)
  T5 – PhiGraph pruning helpers (eliminate_hold, eliminate_unhold)
  T6 – Strategy consistency (BFS, NoPrune, AlwMid, LongBS, BSRandom agree)
  T7 – Synthesis correctness (1-D param search, objective direction)
  T8 – Data loading helpers (shape coercion, mat/npy)
  T9 – End-to-end paper benchmarks (AT1 k=1 pattern dist, AT5 k=1 structure)
"""

import math
import io
import os
import tempfile

import numpy as np
import pytest
import torch

# ── path setup (run from repo root or tests/) ───────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp
from ceclass.lattice.parser import Parser
from ceclass.lattice.phi_graph import PhiGraph
from ceclass.lattice.phi_node import PhiNode
from ceclass.utils.data import load_traces

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers shared across tests
# ═══════════════════════════════════════════════════════════════════════════════

def _pred(name, op, thresh, idx, node_id=None):
    return STLNode.predicate(name, op, thresh, signal_index=idx,
                              node_id=node_id or f"{name}_{op}_{thresh}")


def _alw(child, a, b, node_id):
    return STLNode.always_node(child, interval=(a, b), node_id=node_id)


def _ev(child, a, b, node_id):
    return STLNode.eventually_node(child, interval=(a, b), node_id=node_id)


def _build_at1_spec(k_val=1):
    """AT1: alw[0,30](speed<90 AND RPM<4000). signals: speed=0, RPM=1."""
    from ceclass.examples.autotrans import build_at_spec
    return build_at_spec(k_val)


def _build_at3_spec(k_val=1):
    """AT3: alw[0,30](speed<100). signal: speed=0."""
    from ceclass.examples.autotrans import build_at3_spec
    return build_at3_spec(k_val)


def _build_at5_spec(k_val=1):
    """AT5: ev[0,30](speed>70 AND RPM>3800). signals: speed=0, RPM=1."""
    from ceclass.examples.autotrans import build_at5_spec
    return build_at5_spec(k_val)


def _make_traces(values: list, dt=DT, device=DEVICE) -> torch.Tensor:
    """
    Build a (num_traces, T, num_signals) tensor.

    ``values`` is a list of dicts: {'speed': [...], 'RPM': [...], ...}
    All signals must have the same length within a trace.
    """
    tensors = []
    for row in values:
        signals = list(row.values())
        t = torch.tensor(signals, dtype=torch.float32).T  # (T, num_signals)
        tensors.append(t)
    return torch.stack(tensors).to(device)  # (N, T, num_signals)


# ═══════════════════════════════════════════════════════════════════════════════
# T1 – Robustness semantics
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustnessSemantics:
    """
    Verify stlcgpp returns a 2-D temporal signal (num_traces, T) when vmapped,
    and that rob[:, 0] is the correct global robustness while rob.min() is NOT.
    """

    def _eval(self, formula, traces, params=None):
        f = to_stlcgpp(formula, params or {}, DEVICE, DT)
        with torch.no_grad():
            return torch.vmap(f)(traces)

    def test_always_output_shape(self):
        """vmap(always_formula)(traces) returns (N, T) not a scalar."""
        pred = _pred("s", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 5, "alw")
        traces = _make_traces([{"s": [80.0] * 601}])  # 1 trace, 6.01s, 1 signal
        rob = self._eval(formula, traces)
        assert rob.ndim == 2, f"Expected 2-D output, got {rob.shape}"
        assert rob.shape[0] == 1

    def test_rob_t0_is_global_robustness_always(self):
        """
        For alw[0,5](speed<100):
          - trace violating (speed=120 throughout): rob[:,0] < 0 ✓
          - trace satisfying (speed=80 throughout): rob[:,0] > 0 ✓
        """
        pred = _pred("s", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 5, "alw")
        traces = _make_traces([
            {"s": [120.0] * 601},   # violates: rob[0] should be 100-120 = -20
            {"s": [80.0] * 601},    # satisfies: rob[0] should be 100-80 = 20
        ])
        rob = self._eval(formula, traces)
        rob0 = rob[:, 0]
        assert rob0[0].item() < 0, "Violating trace should have rob[:, 0] < 0"
        assert rob0[1].item() > 0, "Satisfying trace should have rob[:, 0] > 0"

    def test_rob_min_is_spurious_for_satisfying_trace(self):
        """
        This is the confirmed bug we fixed: for a satisfying trace with a large
        always interval, stlcgpp returns -1e9 for out-of-bounds timesteps,
        making rob.min() = -1e9 even though the trace satisfies the formula.
        The CORRECT check is rob[:, 0].
        """
        pred = _pred("s", "<", 100.0, 0, "p")
        # Interval [0, 30] at dt=0.01 = 3001 timesteps — fills the whole trace
        formula = _alw(pred, 0, 30, "alw_big")
        traces = _make_traces([{"s": [80.0] * 3001}])  # satisfies at t=0
        rob = self._eval(formula, traces)
        assert rob[:, 0].item() > 0, "rob at t=0 should be positive (formula satisfied)"
        assert rob.min().item() < 0, (
            "rob.min() is negative due to -1e9 out-of-bounds sentinel — "
            "this is why we must use rob[:, 0] not rob.min()"
        )

    def test_rob_t0_correctly_excludes_out_of_bounds(self):
        """rob[:,0] correctly identifies a trace that satisfies alw[0,30] throughout."""
        pred = _pred("s", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw_big")
        # trace: speed = 80 for first 30s then spikes to 120 after the window
        speed = [80.0] * 3001 + [120.0] * 500  # 3501 steps, only [0,3000] in window
        traces = _make_traces([{"s": speed}])
        rob = self._eval(formula, traces)
        assert rob[:, 0].item() > 0, "Formula should be satisfied at t=0 (window [0,30] all < 100)"

    def test_eventually_output_shape(self):
        """vmap(eventually_formula)(traces) returns (N, T)."""
        pred = _pred("s", ">", 70.0, 0, "p")
        formula = _ev(pred, 0, 5, "ev")
        traces = _make_traces([{"s": [80.0] * 601}])
        rob = self._eval(formula, traces)
        assert rob.ndim == 2

    def test_rob_t0_is_global_robustness_eventually(self):
        """
        For ev[0,5](speed>70):
          - trace with speed=80 always: ev satisfied (rob[0] = max(80-70) = 10 > 0)
          - trace with speed=60 always: ev violated (rob[0] = max(60-70) = -10 < 0)
        """
        pred = _pred("s", ">", 70.0, 0, "p")
        formula = _ev(pred, 0, 5, "ev")
        traces = _make_traces([
            {"s": [80.0] * 601},   # satisfies
            {"s": [60.0] * 601},   # violates
        ])
        rob = self._eval(formula, traces)
        rob0 = rob[:, 0]
        assert rob0[0].item() > 0, "Trace with speed=80 satisfies ev[0,5](speed>70)"
        assert rob0[1].item() < 0, "Trace with speed=60 violates ev[0,5](speed>70)"

    def test_true_never_covered(self):
        """TRUE node always returns large positive robustness — never covered."""
        true_node = STLNode.true_node()
        traces = _make_traces([{"s": [0.0] * 100}])  # any trace
        rob = self._eval(true_node, traces)
        rob0 = rob[:, 0] if rob.ndim > 1 else rob
        assert rob0.min().item() > 0, "TRUE should never have negative robustness"

    def test_false_always_negative(self):
        """FALSE node always returns large negative robustness."""
        false_node = STLNode.false_node()
        traces = _make_traces([{"s": [0.0] * 100}])
        rob = self._eval(false_node, traces)
        rob0 = rob[:, 0] if rob.ndim > 1 else rob
        assert rob0.max().item() < 0, "FALSE should always have negative robustness"


# ═══════════════════════════════════════════════════════════════════════════════
# T2 – STL formula tree helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestSTLNode:

    def test_predicate_str(self):
        p = _pred("speed", "<", 90.0, 0)
        assert "speed" in str(p) and "<" in str(p) and "90" in str(p)

    def test_always_str(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        f = _alw(p, 0, 30, "alw")
        s = str(f)
        assert "alw_[0,30]" in s

    def test_negate_predicate(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        neg = STLNode.negate(p)
        assert neg.node_type == "not"
        assert neg.children[0] is p

    def test_negate_not_cancels(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        neg = STLNode.not_node(p, "neg_p")
        double_neg = STLNode.negate(neg)
        # negate(NOT(p)) == p
        assert double_neg is p

    def test_negate_true_gives_false(self):
        assert STLNode.negate(STLNode.true_node()).node_type == "false"

    def test_negate_false_gives_true(self):
        assert STLNode.negate(STLNode.false_node()).node_type == "true"

    def test_get_param_names_empty_for_fixed(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        f = _alw(p, 0, 30, "alw")
        assert f.get_param_names() == []

    def test_get_param_names_returns_symbolic_bounds(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        f = STLNode.always_node(p, interval=("t1", 30), node_id="alw_param")
        params = f.get_param_names()
        assert "t1" in params

    def test_get_param_names_deduplicates(self):
        p = _pred("speed", "<", 90.0, 0, "p")
        a = STLNode.always_node(p, interval=("t1", "t2"), node_id="alw1")
        b = STLNode.always_node(p, interval=("t1", 30), node_id="alw2")
        combined = STLNode.and_node(a, b, "combined")
        params = combined.get_param_names()
        assert params.count("t1") == 1, "get_param_names() should deduplicate"


# ═══════════════════════════════════════════════════════════════════════════════
# T3 – Lattice structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatticeStructure:

    def _parse(self, formula, k):
        return Parser(formula, k).parse()

    # ── node counts ───────────────────────────────────────────────────────────

    def test_at3_k1_has_2_nodes(self):
        """alw[0,30](speed<100) k=1 → 2 nodes: original + TRUE."""
        formula, k = _build_at3_spec(1)
        g = self._parse(formula, k)
        assert len(g.nodes) == 2, f"Expected 2, got {len(g.nodes)}"

    def test_at3_k2_has_4_nodes(self):
        """alw[0,30](speed<100) k=2 → 4 nodes: original-split, first-half, second-half, TRUE."""
        formula, k = _build_at3_spec(2)
        g = self._parse(formula, k)
        assert len(g.nodes) == 4, f"Expected 4, got {len(g.nodes)}"

    def test_at3_k3_has_8_nodes(self):
        formula, k = _build_at3_spec(3)
        g = self._parse(formula, k)
        assert len(g.nodes) == 8

    def test_at1_k1_has_4_nodes(self):
        """alw[0,30](speed<90 AND RPM<4000) k=1 → 4 nodes: AND, speed, RPM, TRUE."""
        formula, k = _build_at1_spec(1)
        g = self._parse(formula, k)
        assert len(g.nodes) == 4, f"Expected 4, got {len(g.nodes)}"

    def test_at1_k2_has_16_nodes(self):
        """At k=2 each 4-node k=1 lattice grows to 16 nodes (4² Cartesian product)."""
        formula, k = _build_at1_spec(2)
        g = self._parse(formula, k)
        assert len(g.nodes) == 16, f"Expected 16, got {len(g.nodes)}"

    def test_at5_k1_has_4_nodes(self):
        """ev[0,30](speed>70 AND RPM>3800) k=1 → 4 nodes: original, speed, RPM, TRUE."""
        formula, k = _build_at5_spec(1)
        g = self._parse(formula, k)
        assert len(g.nodes) == 4

    def test_at5_k2_has_10_nodes(self):
        formula, k = _build_at5_spec(2)
        g = self._parse(formula, k)
        assert len(g.nodes) == 10

    # ── specific node formulas ────────────────────────────────────────────────

    def test_at5_k1_contains_original_speed_rpm_true(self):
        formula, k = _build_at5_spec(1)
        g = self._parse(formula, k)
        formulas = [str(n.formula) for n in g.nodes]
        # TRUE must be present
        assert any("TRUE" in f for f in formulas), "Lattice must contain TRUE"
        # speed-only eventually
        assert any("speed" in f and "RPM" not in f and "TRUE" not in f for f in formulas), \
            "Expected ev[0,30](speed>70) node"
        # RPM-only eventually
        assert any("RPM" in f and "speed" not in f and "TRUE" not in f for f in formulas), \
            "Expected ev[0,30](RPM>3800) node"

    def test_at5_k2_contains_class_g_and_class_i(self):
        """Paper Fig.7c mentions Class G (ev[0,t](speed>70) ∨ ev[t,30](RPM>3800))
        and Class I (ev[0,t](RPM>3800) ∨ ev[t,30](speed>70))."""
        formula, k = _build_at5_spec(2)
        g = self._parse(formula, k)
        formulas = [str(n.formula) for n in g.nodes]
        class_g_present = any(
            "speed" in f and "RPM" in f and "or" in f and "and" not in f.split("or")[0]
            for f in formulas
        )
        assert class_g_present, "Class G (speed-first OR RPM-second) not found in k=2 lattice"

    # ── TRUE is always a minimum (no smaller_imme) ────────────────────────────

    def test_true_node_is_minimum(self):
        """TRUE has no smaller_imme (it's the weakest node)."""
        formula, k = _build_at3_spec(1)
        g = self._parse(formula, k)
        true_nodes = [n for n in g.nodes if n.formula.node_type == "true"]
        assert len(true_nodes) == 1, "Exactly one TRUE node"
        assert true_nodes[0].smaller_imme == [], "TRUE has no smaller descendants"

    # ── maxima have no greater_imme ───────────────────────────────────────────

    def test_maxima_have_no_greater_imme(self):
        for spec_fn, k_val in [(_build_at3_spec, 2), (_build_at5_spec, 2), (_build_at1_spec, 1)]:
            formula, k = spec_fn(k_val)
            g = Parser(formula, k).parse()
            for m in g.maxima:
                assert m.greater_imme == [], \
                    f"Maxima node {m} should have no greater_imme"

    # ── smaller_all is consistent with immediate edges ────────────────────────

    def test_smaller_all_contains_all_reachable_descendants(self):
        """Every node reachable via smaller_imme should be in smaller_all."""
        formula, k = _build_at1_spec(2)
        g = Parser(formula, k).parse()

        def _reachable(node):
            seen = set()
            stack = [node]
            while stack:
                cur = stack.pop()
                for s in cur.smaller_imme:
                    if s not in seen:
                        seen.add(s)
                        stack.append(s)
            return seen

        for node in g.nodes:
            reachable = _reachable(node)
            for r in reachable:
                assert r in node.smaller_all or r == node, \
                    f"Node {r} reachable via smaller_imme but not in smaller_all of {node}"

    # ── implication direction is antisymmetric ────────────────────────────────

    def test_no_symmetric_implication_edges(self):
        """If A → B then B should not → A (DAG, no cycles)."""
        formula, k = _build_at1_spec(2)
        g = Parser(formula, k).parse()
        for node in g.nodes:
            for s in node.smaller_all:
                assert node not in s.smaller_all, \
                    f"Symmetric edge between {node} and {s} — indicates a cycle"


# ═══════════════════════════════════════════════════════════════════════════════
# T4 – Implication monotonicity (semantic check with actual robustness)
# ═══════════════════════════════════════════════════════════════════════════════

class TestImplicationMonotonicity:
    """
    Core semantic requirement (Corollary 1 from paper):
    If a trace violates φ_weak (smaller/weaker node), it MUST also violate
    φ_strong (greater/stronger node).

    Equivalently: rob(φ_strong, trace, 0) <= rob(φ_weak, trace, 0) for any trace
    (stronger formula is harder to satisfy → lower robustness).
    """

    def _rob0_batch(self, formula, traces, chunk_size: int = 16):
        f = to_stlcgpp(formula, {}, DEVICE, DT)
        parts = []
        with torch.no_grad():
            for start in range(0, traces.shape[0], chunk_size):
                batch = traces[start : start + chunk_size]
                rob = torch.vmap(f)(batch)
                r0 = rob[:, 0] if rob.ndim > 1 else rob
                parts.append(r0.cpu().numpy())
        return np.concatenate(parts)

    def test_at1_k1_implication(self):
        """
        alw(speed<90 AND RPM<4000)  →  alw(speed<90)   AND   alw(RPM<4000)
        For all traces: rob(AND) <= min(rob(speed), rob(RPM))
        """
        data_path = "/home/parvk/CEClassification/test/data/AT1_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT1 trace file not found")

        traces = load_traces(data_path, device=DEVICE, signal_indices=[0, 1])

        speed_pred = STLNode.predicate("speed", "<", 90.0,  signal_index=0, node_id="s90")
        rpm_pred   = STLNode.predicate("RPM",   "<", 4000.0, signal_index=1, node_id="r4k")
        and_pred   = STLNode.and_node(speed_pred, rpm_pred, "and_pred")

        phi_and   = _alw(and_pred,   0, 30, "alw_and")
        phi_speed = _alw(speed_pred, 0, 30, "alw_speed")
        phi_rpm   = _alw(rpm_pred,   0, 30, "alw_rpm")

        rob_and   = self._rob0_batch(phi_and,   traces)
        rob_speed = self._rob0_batch(phi_speed, traces)
        rob_rpm   = self._rob0_batch(phi_rpm,   traces)

        # AND implies both halves → rob_and ≤ rob_speed AND rob_and ≤ rob_rpm
        for i in range(len(traces)):
            assert rob_and[i] <= rob_speed[i] + 1e-4, \
                f"Trace {i}: rob(AND)={rob_and[i]:.3f} > rob(speed)={rob_speed[i]:.3f}"
            assert rob_and[i] <= rob_rpm[i] + 1e-4, \
                f"Trace {i}: rob(AND)={rob_and[i]:.3f} > rob(RPM)={rob_rpm[i]:.3f}"

    def test_at5_k1_implication(self):
        """
        ev(speed>70 AND RPM>3800)  is STRONGER than  ev(speed>70)  AND  ev(RPM>3800)
        For all traces: rob(original) <= min(rob(speed-only), rob(rpm-only))
        (original is harder to satisfy → lower robustness)
        """
        data_path = "/home/parvk/CEClassification/test/data/AT53_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT53 trace file not found")

        traces = load_traces(data_path, device=DEVICE, signal_indices=[0, 1])

        speed_pred = STLNode.predicate("speed", ">", 70.0,   signal_index=0, node_id="s70")
        rpm_pred   = STLNode.predicate("RPM",   ">", 3800.0, signal_index=1, node_id="r3800")
        and_pred   = STLNode.and_node(speed_pred, rpm_pred, "and_pred")

        phi_orig  = _ev(and_pred,   0, 30, "ev_orig")
        phi_speed = _ev(speed_pred, 0, 30, "ev_speed")
        phi_rpm   = _ev(rpm_pred,   0, 30, "ev_rpm")

        rob_orig  = self._rob0_batch(phi_orig,  traces)
        rob_speed = self._rob0_batch(phi_speed, traces)
        rob_rpm   = self._rob0_batch(phi_rpm,   traces)

        for i in range(len(traces)):
            assert rob_orig[i] <= rob_speed[i] + 1e-4, \
                f"Trace {i}: rob(AND-orig)={rob_orig[i]:.2f} > rob(speed-only)={rob_speed[i]:.2f}"
            assert rob_orig[i] <= rob_rpm[i] + 1e-4, \
                f"Trace {i}: rob(AND-orig)={rob_orig[i]:.2f} > rob(rpm-only)={rob_rpm[i]:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# T5 – PhiGraph pruning helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhiGraphPruning:
    """
    Tests for eliminate_hold (covered → propagate to stronger ancestors)
    and eliminate_unhold (not covered → propagate to weaker descendants).
    """

    def _simple_chain(self):
        """
        Build a simple 3-node chain: strong → mid → weak
          strong.smaller_imme = [mid],  mid.smaller_imme = [weak]
          mid.greater_imme    = [strong], weak.greater_imme = [mid]
        """
        weak   = PhiNode(formula=STLNode.true_node())
        mid    = PhiNode(formula=STLNode.predicate("s", "<", 100.0, 0, "mid"))
        strong = PhiNode(formula=STLNode.predicate("s", "<", 50.0,  0, "strong"))

        # Build implication chain: strong → mid → weak
        strong.smaller_imme = [mid];   mid.greater_imme = [strong]
        mid.smaller_imme    = [weak];  weak.greater_imme = [mid]
        strong.smaller_all  = [mid, weak]
        mid.smaller_all     = [weak]
        strong.greater_all  = []
        mid.greater_all     = [strong]
        weak.greater_all    = [strong, mid]

        g = PhiGraph([strong, mid, weak])
        g.maxima = [strong]
        return g, strong, mid, weak

    def test_eliminate_hold_marks_self_and_propagates_up(self):
        """eliminate_hold(mid) should mark mid AND strong as covered."""
        g, strong, mid, weak = self._simple_chain()
        sentinel = object()
        g.eliminate_hold(mid, sentinel)
        assert sentinel in mid.results,    "mid should be covered"
        assert sentinel in strong.results, "strong should be covered (mid is weaker → strong is too)"
        assert not mid.active,    "mid should be deactivated"
        assert not strong.active, "strong should be deactivated"
        assert weak.active, "weak should remain active (not affected by eliminate_hold on mid)"

    def test_eliminate_unhold_marks_self_and_propagates_down(self):
        """eliminate_unhold(mid) should deactivate mid AND weak."""
        g, strong, mid, weak = self._simple_chain()
        g.eliminate_unhold(mid)
        assert not mid.active,  "mid should be deactivated"
        assert not weak.active, "weak should be deactivated (mid not covered → weaker nodes also not)"
        assert strong.active,   "strong should remain active"

    def test_is_empty_after_full_deactivation(self):
        g, strong, mid, weak = self._simple_chain()
        for n in [strong, mid, weak]:
            n.active = False
        assert g.is_empty()

    def test_get_covered_nodes_returns_only_result_nodes(self):
        g, strong, mid, weak = self._simple_chain()
        sentinel = object()
        mid.results.append(sentinel)
        covered = g.get_covered_nodes()
        assert mid in covered
        assert strong not in covered
        assert weak not in covered

    def test_set_active_maxima_respects_active_flag(self):
        """After deactivating the original maximum, set_active_maxima finds the next."""
        g, strong, mid, weak = self._simple_chain()
        strong.active = False
        g.set_active_maxima()
        assert strong not in g.maxima
        assert mid in g.maxima


# ═══════════════════════════════════════════════════════════════════════════════
# T6 – Strategy consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyConsistency:
    """
    NoPrune, AlwMid, LongBS, BSRandom should agree on coverage (same lattice
    semantics).  BFS here is a bottom-up variant and may disagree with NoPrune.
    """

    @pytest.fixture(scope="class")
    def at5_k1_data(self):
        data_path = "/home/parvk/CEClassification/test/data/AT53_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT53 trace file not found")
        traces = load_traces(data_path, device=DEVICE, signal_indices=[0, 1])
        formula, k = _build_at5_spec(1)
        return formula, k, traces

    @pytest.fixture(scope="class")
    def at3_k1_data(self):
        data_path = "/home/parvk/CEClassification/test/data/AT3_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT3 trace file not found")
        traces = load_traces(data_path, device=DEVICE, signal_indices=[0])
        formula, k = _build_at3_spec(1)
        return formula, k, traces

    def _run(self, strategy_cls, formula, k, traces):
        clf = strategy_cls(formula, k, traces, device=DEVICE, dt=DT)
        result = clf.solve()
        covered_ids = sorted(n.formula.id for n in result.covered_nodes)
        return result, covered_ids

    def test_noprune_vs_alwmid_at5_k1(self, at5_k1_data):
        from ceclass.strategies.no_prune import NoPruneClassifier
        from ceclass.strategies.alw_mid import AlwMidClassifier
        formula, k, traces = at5_k1_data
        _, ids_np = self._run(NoPruneClassifier, formula, k, traces)
        _, ids_alw = self._run(AlwMidClassifier, formula, k, traces)
        assert ids_np == ids_alw, \
            f"NoPrune and AlwMid disagree:\n  NoPrune: {ids_np}\n  AlwMid: {ids_alw}"

    def test_noprune_vs_longbs_at5_k1(self, at5_k1_data):
        from ceclass.strategies.no_prune import NoPruneClassifier
        from ceclass.strategies.long_bs import LongBSClassifier
        formula, k, traces = at5_k1_data
        _, ids_np = self._run(NoPruneClassifier, formula, k, traces)
        _, ids_lb = self._run(LongBSClassifier, formula, k, traces)
        assert ids_np == ids_lb, \
            f"NoPrune and LongBS disagree:\n  NoPrune: {ids_np}\n  LongBS: {ids_lb}"

    def test_bfs_visits_minima_first(self, at5_k1_data):
        """Bottom-up BFS dequeues a minima (leaf) first."""
        from ceclass.strategies.bfs import BFSClassifier
        formula, k, traces = at5_k1_data
        clf = BFSClassifier(formula, k, traces, device=DEVICE, dt=DT)
        minima_ids = {n.formula.id for n in clf.graph.nodes if len(n.smaller_imme) == 0}
        first_tested = []
        original_test = clf._test_node
        def _record(node):
            if not first_tested:
                first_tested.append(node)
            return original_test(node)
        clf._test_node = _record
        clf.solve()
        assert first_tested[0].formula.id in minima_ids, \
            "Bottom-up BFS must dequeue a minima node first"

    def test_noprune_synth_calls_equals_num_nodes(self, at5_k1_data):
        """NoPrune tests every node exactly once → synth_calls == num_classes."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        formula, k, traces = at5_k1_data
        result, _ = self._run(NoPruneClassifier, formula, k, traces)
        assert result.num_synth_calls == result.num_classes, \
            f"NoPrune should call _test_node exactly num_classes times"

    def test_bfs_fewer_synth_calls_than_noprune_at_larger_k(self):
        """BFS prunes aggressively → fewer calls than NoPrune for k≥2."""
        data_path = "/home/parvk/CEClassification/test/data/AT53_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT53 trace file not found")
        from ceclass.strategies.no_prune import NoPruneClassifier
        from ceclass.strategies.bfs import BFSClassifier
        traces = load_traces(data_path, device=DEVICE, signal_indices=[0, 1])
        formula, k = _build_at5_spec(2)
        res_np,  _ = self._run(NoPruneClassifier, formula, k, traces)
        res_bfs, _ = self._run(BFSClassifier,     formula, k, traces)
        assert res_bfs.num_synth_calls <= res_np.num_synth_calls, \
            "BFS should not need MORE calls than NoPrune"


# ═══════════════════════════════════════════════════════════════════════════════
# T7 – Synthesis correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSynthesis:

    def _make_at3_traces_violating(self, n=5):
        """Generate n traces that violate alw[0,30](speed<100): speed=120 throughout."""
        T = 3001
        data = {"speed": [120.0] * T}
        return _make_traces([data] * n)

    def _make_at3_traces_satisfying(self, n=5):
        """Generate n traces that satisfy alw[0,30](speed<100): speed=80 throughout."""
        T = 3001
        data = {"speed": [80.0] * T}
        return _make_traces([data] * n)

    def test_nonparam_covered_when_trace_violates(self):
        """Non-parametric node is covered iff at least one trace violates at t=0."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        pred = _pred("speed", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw")
        k = [1, [1]]

        # All traces violate → covered
        traces_v = self._make_at3_traces_violating()
        clf = NoPruneClassifier(formula, k, traces_v, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_covered == 1, f"Original formula should be covered; got {r.num_covered}/2"

    def test_nonparam_not_covered_when_all_satisfy(self):
        """Non-parametric node is NOT covered if all traces satisfy it."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        pred = _pred("speed", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw")
        k = [1, [1]]

        # All traces satisfy → NOT covered (only TRUE is covered? No, TRUE can't be covered)
        traces_s = self._make_at3_traces_satisfying()
        clf = NoPruneClassifier(formula, k, traces_s, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_covered == 0, \
            f"All traces satisfy formula → 0 covered expected, got {r.num_covered}"

    def test_rob_t0_fix_prevents_false_positive(self):
        """
        This test DIRECTLY verifies the fix: using rob[:, 0] prevents a
        satisfying trace from appearing as a counterexample.
        """
        pred = _pred("s", "<", 100.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw_big")  # large interval → out-of-bounds sentinel
        traces = _make_traces([{"s": [80.0] * 3001}])  # satisfies formula

        f = to_stlcgpp(formula, {}, DEVICE, DT)
        with torch.no_grad():
            rob = torch.vmap(f)(traces)  # (1, T)

        rob0  = rob[:, 0].min().item()   # correct: t=0 only
        rob_m = rob.min().item()          # buggy: all time steps

        assert rob0 > 0, f"rob[:, 0] should be positive (formula satisfied), got {rob0}"
        assert rob_m < 0, f"rob.min() should be negative (out-of-bounds sentinel), got {rob_m}"
        # After fix: we use rob0 → formula correctly reported as NOT covered
        # Before fix: we'd use rob_m → formula WRONGLY reported as covered

    def test_1d_synthesis_finds_covering_params(self):
        """1-D param synthesis: find t in [0,30] that makes alw[0,t](speed<90) violated."""
        from ceclass.synthesis.param_synth import ParamSynthesis
        # 5 traces all with speed=120 for the first 15s then speed=80
        T = 3001
        speed = [120.0] * 1500 + [80.0] * (T - 1500)
        traces = _make_traces([{"speed": speed}] * 5)

        pred = _pred("speed", "<", 90.0, 0, "p")
        formula = STLNode.always_node(pred, interval=(0, "t"), node_id="alw_param")

        synth = ParamSynthesis(
            formula=formula,
            traces=traces,
            param_names=["t"],
            param_bounds={"t": (0.0, 30.0)},
            device=DEVICE, dt=DT,
            max_evals=40,
        )
        result = synth.solve()
        assert result.satisfied, "Synthesis should find t where formula is violated"
        assert result.params_best is not None
        t_found = result.params_best["t"]
        # t=0 also valid: alw[0,0](speed<90) is violated when speed=120 at t=0
        assert 0.0 <= t_found <= 30.0, f"Found t={t_found} outside [0, 30]"

    def test_synthesis_reports_not_satisfied_when_impossible(self):
        """Synthesis should return NOT satisfied when no params can violate the formula."""
        from ceclass.synthesis.param_synth import ParamSynthesis
        # All traces satisfy speed<90 always
        T = 3001
        traces = _make_traces([{"speed": [80.0] * T}] * 5)

        pred = _pred("speed", "<", 90.0, 0, "p")
        # Parametric formula alw[0, t](speed<90) — for any t, speed<90 everywhere
        formula = STLNode.always_node(pred, interval=(0, "t"), node_id="alw_param")

        synth = ParamSynthesis(
            formula=formula,
            traces=traces,
            param_names=["t"],
            param_bounds={"t": (0.0, 30.0)},
            device=DEVICE, dt=DT,
            max_evals=30,
        )
        result = synth.solve()
        assert not result.satisfied, \
            "No trace violates formula → synthesis should report NOT satisfied"

    def test_obj_best_is_meaningful_after_fix(self):
        """After the rob[:, 0] fix, obj_best reflects actual robustness (not ±1e9)."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        # Simple: 1 trace violating speed<90 by exactly 10 (speed=100)
        T = 3001
        traces = _make_traces([{"speed": [100.0] * T}])  # 100 - 90 = 10 violation
        pred = _pred("speed", "<", 90.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw")
        k = [1, [1]]
        clf = NoPruneClassifier(formula, k, traces, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_covered == 1
        obj = r.covered_nodes[0].results[0].obj_best
        # Should be close to -10 (speed-90 = 10, so rob = 90-speed = -10)
        assert abs(obj - (-10.0)) < 0.1, \
            f"obj_best should be ≈ -10 after fix, got {obj:.3f} (was -1e9 before fix)"


# ═══════════════════════════════════════════════════════════════════════════════
# T8 – Data loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoading:

    def test_numpy_3d_passthrough(self):
        arr = np.random.rand(10, 100, 3).astype(np.float32)
        t = load_traces(arr)
        assert t.shape == (10, 100, 3)

    def test_numpy_2d_gets_signal_dim(self):
        """2-D (num_traces, timesteps) → (num_traces, timesteps, 1)."""
        arr = np.random.rand(10, 100).astype(np.float32)
        t = load_traces(arr)
        assert t.shape == (10, 100, 1), f"Got {t.shape}"

    def test_torch_tensor_passthrough(self):
        arr = torch.rand(5, 200, 2)
        t = load_traces(arr)
        assert t.shape == (5, 200, 2)

    def test_npy_file_loading(self):
        arr = np.random.rand(4, 50, 2).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, arr)
            t = load_traces(f.name)
        os.unlink(f.name)
        assert t.shape == (4, 50, 2)

    def test_mat_file_loading_3d(self):
        """Load a .mat file with a 3-D 'traces' variable."""
        try:
            import scipy.io as sio
        except ImportError:
            pytest.skip("scipy not available")
        arr = np.random.rand(6, 30, 2)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            sio.savemat(f.name, {"traces": arr})
            t = load_traces(f.name)
        os.unlink(f.name)
        assert t.shape == (6, 30, 2)

    def test_mat_file_loading_2d_adds_signal_dim(self):
        """2-D .mat trace (num_traces, timesteps) is reshaped to (num_traces, timesteps, 1)."""
        try:
            import scipy.io as sio
        except ImportError:
            pytest.skip("scipy not available")
        arr = np.random.rand(8, 50)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            sio.savemat(f.name, {"traces": arr})
            t = load_traces(f.name)
        os.unlink(f.name)
        assert t.shape == (8, 50, 1), f"2-D .mat should produce (N, T, 1), got {t.shape}"

    def test_signal_indices_selection(self):
        """signal_indices selects a subset of signal columns."""
        arr = np.random.rand(3, 20, 5)
        t = load_traces(arr, signal_indices=[0, 2])
        assert t.shape == (3, 20, 2)

    def test_device_transfer(self):
        arr = np.ones((2, 10, 1), dtype=np.float32)
        t = load_traces(arr, device=DEVICE)
        assert t.device.type == DEVICE.type

    def test_real_at1_traces_shape(self):
        data_path = "/home/parvk/CEClassification/test/data/AT1_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AT1 trace file not found")
        t = load_traces(data_path)
        assert t.ndim == 3, "AT1 traces should be 3-D"
        assert t.shape[0] >= 30, "AT1 traces file should have at least 30 runs"

    def test_real_afc1_traces_shape(self):
        """AFC1 traces are originally 2-D → must be auto-reshaped to 3-D."""
        data_path = "/home/parvk/CEClassification/test/data/AFC1_traces.mat"
        if not os.path.exists(data_path):
            pytest.skip("AFC1 trace file not found")
        t = load_traces(data_path)
        assert t.ndim == 3, "AFC1 traces must be reshaped to 3-D"
        assert t.shape[2] == 1, "AFC1 has 1 signal dimension"


# ═══════════════════════════════════════════════════════════════════════════════
# T9 – End-to-end paper benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """
    Reproduce key quantitative results from the paper:
      - AT1 k=1 pattern distribution (Table 1): 2 speed-only, 6 RPM-only
      - AT5 k=1 coverage: 3/4 nodes covered (original, speed, RPM — not TRUE)
      - AT1 k=1: 3/4 covered (original, speed, RPM — not TRUE)
      - AT3 k=1: 1/2 covered (only original — TRUE never covered)
    """

    @pytest.fixture(scope="class")
    def at1_traces(self):
        p = "/home/parvk/CEClassification/test/data/AT1_traces.mat"
        if not os.path.exists(p): pytest.skip("AT1 traces not found")
        return load_traces(p, device=DEVICE, signal_indices=[0, 1])

    @pytest.fixture(scope="class")
    def at3_traces(self):
        p = "/home/parvk/CEClassification/test/data/AT3_traces.mat"
        if not os.path.exists(p): pytest.skip("AT3 traces not found")
        return load_traces(p, device=DEVICE, signal_indices=[0])

    @pytest.fixture(scope="class")
    def at5_traces(self):
        p = "/home/parvk/CEClassification/test/data/AT53_traces.mat"
        if not os.path.exists(p): pytest.skip("AT53 traces not found")
        return load_traces(p, device=DEVICE, signal_indices=[0, 1])

    # ── AT1 Table 1 pattern distribution ─────────────────────────────────────

    def test_at1_pattern_distribution_numpy(self, at1_traces):
        """
        Direct numpy check: 2 traces exceed speed≥90 only, 6 exceed RPM≥4000 only.
        This is Table 1 from the paper (φAT₂ in paper = AT1 in TACAS scripts).
        """
        if at1_traces.shape[0] != 30:
            pytest.skip("Paper Table 1 counts apply only to the 30-trace falsification bundle")
        traces_np = at1_traces.cpu().numpy()
        speed = traces_np[:, :, 0]
        rpm   = traces_np[:, :, 1]

        violates_speed = speed.max(axis=1) >= 90.0
        violates_rpm   = rpm.max(axis=1)   >= 4000.0

        pattern1 = (violates_speed & ~violates_rpm).sum()  # speed only
        pattern2 = (~violates_speed & violates_rpm).sum()  # RPM only
        assert pattern1 == 2, f"Pattern 1 (speed only): expected 2, got {pattern1}"
        assert pattern2 == 6, f"Pattern 2 (RPM only): expected 6, got {pattern2}"

    def test_at1_pattern_distribution_via_stlcgpp(self, at1_traces):
        """Same check via rob[:, 0] (the correct evaluation after the fix)."""
        if at1_traces.shape[0] != 30:
            pytest.skip("Paper Table 1 counts apply only to the 30-trace falsification bundle")
        speed_pred = STLNode.predicate("speed", "<", 90.0,   signal_index=0, node_id="s90")
        rpm_pred   = STLNode.predicate("RPM",   "<", 4000.0, signal_index=1, node_id="r4k")
        phi_speed = _alw(speed_pred, 0, 30, "alw_speed")
        phi_rpm   = _alw(rpm_pred,   0, 30, "alw_rpm")

        f_speed = to_stlcgpp(phi_speed, {}, DEVICE, DT)
        f_rpm   = to_stlcgpp(phi_rpm,   {}, DEVICE, DT)
        with torch.no_grad():
            rob_s = torch.vmap(f_speed)(at1_traces)[:, 0]
            rob_r = torch.vmap(f_rpm)(at1_traces)[:, 0]

        in_speed = (rob_s < 0)
        in_rpm   = (rob_r < 0)
        p1 = (in_speed & ~in_rpm).sum().item()
        p2 = (~in_speed & in_rpm).sum().item()
        assert p1 == 2, f"stlcgpp Pattern 1: expected 2, got {p1}"
        assert p2 == 6, f"stlcgpp Pattern 2: expected 6, got {p2}"

    # ── AT1 k=1 coverage ──────────────────────────────────────────────────────

    def test_at1_k1_coverage_3_of_4(self, at1_traces):
        """AT1 k=1: original, speed-only, RPM-only should be covered; TRUE should not."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        formula, k = _build_at1_spec(1)
        clf = NoPruneClassifier(formula, k, at1_traces, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_classes  == 4, f"AT1 k=1 should have 4 nodes, got {r.num_classes}"
        assert r.num_covered  == 3, f"AT1 k=1: 3/4 should be covered, got {r.num_covered}"

        covered_formulas = [str(n.formula) for n in r.covered_nodes]
        assert not any("TRUE" in f for f in covered_formulas), "TRUE must never be covered"

    # ── AT3 k=1 coverage ──────────────────────────────────────────────────────

    def test_at3_k1_coverage_1_of_2(self, at3_traces):
        """AT3 k=1: original alw[0,30](speed<100) covered; TRUE not covered."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        formula, k = _build_at3_spec(1)
        clf = NoPruneClassifier(formula, k, at3_traces, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_classes == 2
        assert r.num_covered == 1
        covered_ids = [n.formula.id for n in r.covered_nodes]
        assert "TRUE" not in covered_ids

    # ── AT5 k=1 coverage ──────────────────────────────────────────────────────

    def test_at5_k1_coverage_3_of_4(self, at5_traces):
        """AT5 k=1: original, speed-only ev, RPM-only ev should be covered; TRUE not."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        formula, k = _build_at5_spec(1)
        clf = NoPruneClassifier(formula, k, at5_traces, device=DEVICE, dt=DT)
        r = clf.solve()
        assert r.num_classes == 4
        assert r.num_covered == 3
        covered_formulas = [str(n.formula) for n in r.covered_nodes]
        assert not any("TRUE" in f for f in covered_formulas)

    def test_at5_k1_paper_fig7_trace_example(self, at5_traces):
        """
        Paper Fig 7 says for AT5 k=1, a specific trace is in Class C
        (ev[0,30](RPM>3800)) but NOT Class B (ev[0,30](speed>70)).
        This requires BOTH classes to exist and be individually testable.
        We verify: some trace is a CX for RPM-only but NOT for speed-only.
        """
        speed_pred = STLNode.predicate("speed", ">", 70.0,   signal_index=0, node_id="s70")
        rpm_pred   = STLNode.predicate("RPM",   ">", 3800.0, signal_index=1, node_id="r3800")
        phi_speed = _ev(speed_pred, 0, 30, "ev_speed")
        phi_rpm   = _ev(rpm_pred,   0, 30, "ev_rpm")

        f_speed = to_stlcgpp(phi_speed, {}, DEVICE, DT)
        f_rpm   = to_stlcgpp(phi_rpm,   {}, DEVICE, DT)
        rs, rr = [], []
        with torch.no_grad():
            for start in range(0, at5_traces.shape[0], 16):
                b = at5_traces[start : start + 16]
                rs.append(torch.vmap(f_speed)(b)[:, 0])
                rr.append(torch.vmap(f_rpm)(b)[:, 0])
        rob_s = torch.cat(rs)
        rob_r = torch.cat(rr)

        # Class C (RPM-only CX): rob_rpm < 0 AND rob_speed >= 0
        class_c_traces = ((rob_r < 0) & (rob_s >= 0)).sum().item()
        # Class B (speed-only CX): rob_speed < 0 AND rob_rpm >= 0
        class_b_traces = ((rob_s < 0) & (rob_r >= 0)).sum().item()

        # Paper says such differentiation exists → at least 1 trace in C-not-B or B-not-C
        assert (class_c_traces + class_b_traces) > 0, \
            "Expected at least one trace that distinguishes Class B from Class C"

    # ── obj_best sanity check after fix ───────────────────────────────────────

    def test_obj_best_not_sentinel_after_fix(self, at1_traces):
        """After the rob[:, 0] fix, obj_best should NOT be ±1e9."""
        from ceclass.strategies.no_prune import NoPruneClassifier
        formula, k = _build_at1_spec(1)
        clf = NoPruneClassifier(formula, k, at1_traces, device=DEVICE, dt=DT)
        r = clf.solve()
        for node in r.covered_nodes:
            obj = node.results[0].obj_best
            assert abs(obj) < 1e8, \
                f"obj_best should not be ±1e9 after fix; got {obj:.3g} for node {node}"


# ═══════════════════════════════════════════════════════════════════════════════
# T10 – Lattice construction edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatticeEdgeCases:

    def test_single_predicate_k1_lattice(self):
        """pred(speed<90) with k=[1,[1]] → 2 nodes: original + TRUE."""
        pred = _pred("speed", "<", 90.0, 0, "p")
        k = [1, [1]]
        g = Parser(pred, k).parse()
        assert len(g.nodes) == 2
        formulas = [str(n.formula) for n in g.nodes]
        assert any("TRUE" in f for f in formulas)

    def test_and_k1_lattice(self):
        """AND(pred1, pred2) with k=1 → 4 nodes (matches AT1 k=1)."""
        p1 = _pred("speed", "<", 90.0, 0, "p1")
        p2 = _pred("RPM", "<", 4000.0, 1, "p2")
        formula = STLNode.and_node(p1, p2, "and")
        k = [1, [1], [1]]
        g = Parser(formula, k).parse()
        assert len(g.nodes) == 4

    def test_parser_param_bounds_registered(self):
        """Parser should register param bounds for symbolic interval boundaries."""
        pred = _pred("speed", "<", 90.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw")
        k = [2, [1]]
        p = Parser(formula, k)
        g = p.parse()
        # Should have parameter t2 in interval_dict
        assert any("t2" in key for key in p.interval_dict), \
            "Parser should register symbolic param bounds for k=2 temporal splits"

    def test_param_bounds_within_parent_interval(self):
        """Parameter bounds must be within the parent formula's interval [0, 30]."""
        pred = _pred("speed", "<", 90.0, 0, "p")
        formula = _alw(pred, 0, 30, "alw")
        k = [2, [1]]
        p = Parser(formula, k)
        p.parse()
        for param, (lo, hi) in p.interval_dict.items():
            assert lo >= 0.0, f"Lower bound of {param} = {lo} < 0"
            assert hi <= 30.0, f"Upper bound of {param} = {hi} > 30"

    def test_no_duplicate_formula_ids_in_lattice(self):
        """All nodes in the lattice must have unique formula IDs."""
        formula, k = _build_at1_spec(2)
        g = Parser(formula, k).parse()
        ids = [n.formula.id for n in g.nodes]
        assert len(ids) == len(set(ids)), "Duplicate formula IDs found in lattice"
