"""
Example: Counterexample classification for Automatic Transmission model.

Reproduces the experiments from the CEClass paper using the AT1 benchmark.
Specification: alw_[0,30]((speed < 90) and (RPM < 4000))

Usage:
    python -m ceclass.examples.autotrans --data test/data/AT1.mat --k 2 --strategy long_bs
"""
from __future__ import annotations
import argparse
import time

import torch

from ceclass.formula.stl_node import STLNode
from ceclass.strategies.bfs import BFSClassifier
from ceclass.strategies.no_prune import NoPruneClassifier
from ceclass.strategies.alw_mid import AlwMidClassifier
from ceclass.strategies.bs_random import BSRandomClassifier
from ceclass.strategies.long_bs import LongBSClassifier
from ceclass.utils.data import load_traces


STRATEGIES = {
    'bfs': BFSClassifier,
    'no_prune': NoPruneClassifier,
    'alw_mid': AlwMidClassifier,
    'bs_random': BSRandomClassifier,
    'long_bs': LongBSClassifier,
}

def build_at_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the AT specification: alw_[0,30]((speed < 90) and (RPM < 4000))

    Returns (formula, k) where k is the hierarchy depth configuration.
    """
    # Signal mapping: speed=column 0, RPM=column 1
    speed_pred = STLNode.predicate("speed", "<", 90.0, signal_index=0, node_id="speed_lt_90")
    rpm_pred = STLNode.predicate("RPM", "<", 4000.0, signal_index=1, node_id="RPM_lt_4000")

    and_node = STLNode.and_node(speed_pred, rpm_pred, node_id="speed_and_RPM")
    formula = STLNode.always_node(and_node, interval=(0, 30), node_id="alw_0_30")

    # Hierarchy depth: k_val splits for always, 1 for each predicate child
    k = [k_val, [1, [1], [1]]]

    return formula, k


def build_reach_avoid_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the 2D reach-avoid specification:
        Phi  = phi1 AND phi2
        phi1 = ev[0,10]( R1 AND ev[20,30]( alw[0,10](R4) ) )
        phi2 = alw[0,60]( NOT R3 )

    Signals: x = column 0, y = column 1.
    Regions:
        R1: 0<x<5,  0<y<5   (start)
        R3: 8<x<13, 8<y<13  (unsafe)
        R4: 15<x<20,15<y<20 (goal)

    k structure mirrors the formula tree:
      - AND nodes:      k[0] unused, k[1]/k[2] for children
      - temporal nodes: k[0] = num splits, k[1] for child
      - predicate:      k = [1]
    nary_and([p1,p2,p3,p4]) produces AND(AND(AND(p1,p2),p3),p4),
    so each 4-pred region needs 3 levels of AND nesting in k.
    """
    def _region(x_lo, x_hi, y_lo, y_hi, label):
        return STLNode.nary_and([
            STLNode.predicate("x", ">", x_lo, signal_index=0, node_id=f"{label}_x_gt_{x_lo}"),
            STLNode.predicate("x", "<", x_hi, signal_index=0, node_id=f"{label}_x_lt_{x_hi}"),
            STLNode.predicate("y", ">", y_lo, signal_index=1, node_id=f"{label}_y_gt_{y_lo}"),
            STLNode.predicate("y", "<", y_hi, signal_index=1, node_id=f"{label}_y_lt_{y_hi}"),
        ], node_id=label)

    R1 = _region(0, 5,  0, 5,  "R1")
    R3 = _region(8, 13, 8, 13, "R3")

    not_R3  = STLNode.not_node(R3, node_id="not_R3")
    phi1    = STLNode.eventually_node(R1,     interval=(0, 10), node_id="phi1")
    phi2    = STLNode.always_node(not_R3,     interval=(0, 60), node_id="phi2")
    formula = STLNode.and_node(phi1, phi2,    node_id="Phi")

    # k for a 4-pred nary_and: AND(AND(AND(p1,p2), p3), p4)
    p   = [1]
    k_R = [1, [1, [1, p, p], p], p]

    # Flat formula — no nested temporal ops, so k_val applies directly.
    # Lattice size: (16^k_val) for phi1 × (16^k_val) for phi2 × AND combo.
    # k=1 → 256 nodes (fast), k=2 → 65k nodes (slow), so default k=1.
    k_not_R3 = [1, k_R]
    k_phi1   = [k_val, k_R]
    k_phi2   = [k_val, k_not_R3]
    k        = [1, k_phi1, k_phi2]

    return formula, k


def build_reach_avoid_r4_spec(k_val: int = 1) -> tuple[STLNode, list]:
    """
    Build the 2D reach-avoid-goal specification:
        Phi  = phi1 AND phi2
        phi1 = ev[0,10]( R1 AND ev[20,30](R4) )
        phi2 = alw[0,60]( NOT R3 )

    Signals: x = column 0, y = column 1.
    Regions:
        R1: 0<x<5,  0<y<5   (start)
        R3: 8<x<13, 8<y<13  (unsafe)
        R4: 15<x<20,15<y<20 (goal)
    """
    def _region(x_lo, x_hi, y_lo, y_hi, label):
        return STLNode.nary_and([
            STLNode.predicate("x", ">", x_lo, signal_index=0, node_id=f"{label}_x_gt_{x_lo}"),
            STLNode.predicate("x", "<", x_hi, signal_index=0, node_id=f"{label}_x_lt_{x_hi}"),
            STLNode.predicate("y", ">", y_lo, signal_index=1, node_id=f"{label}_y_gt_{y_lo}"),
            STLNode.predicate("y", "<", y_hi, signal_index=1, node_id=f"{label}_y_lt_{y_hi}"),
        ], node_id=label)

    R1 = _region(0,  5,  0,  5,  "R1")
    R3 = _region(8,  13, 8,  13, "R3")
    R4 = _region(15, 20, 15, 20, "R4")

    not_R3    = STLNode.not_node(R3, node_id="not_R3")
    ev_R4     = STLNode.eventually_node(R4, interval=(20, 30), node_id="ev_20_30_R4")
    R1_and_ev = STLNode.and_node(R1, ev_R4, node_id="R1_and_ev_R4")
    phi1      = STLNode.eventually_node(R1_and_ev, interval=(0, 10), node_id="phi1")
    phi2      = STLNode.always_node(not_R3, interval=(0, 60), node_id="phi2")
    formula   = STLNode.and_node(phi1, phi2, node_id="Phi")

    p        = [1]
    k_R      = [1, [1, [1, p, p], p], p]   # 4-pred nary_and
    k_not_R3 = [1, k_R]
    k_ev_R4  = [k_val, k_R]
    k_R1_and = [1, k_R, k_ev_R4]
    k_phi1   = [k_val, k_R1_and]
    k_phi2   = [k_val, k_not_R3]
    k        = [1, k_phi1, k_phi2]

    return formula, k


def build_at2_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the AT2 specification:
        alw_[0,30]((brake < 250) OR ev_[0,5](speed < 30 AND RPM < 2000))

    Signal mapping: speed=0, RPM=1, brake=2
    k structure mirrors MATLAB: {k, {0, {0}, {1, {0, {0}, {0}}}}}
    """
    speed_pred = STLNode.predicate("speed", "<", 30.0, signal_index=0, node_id="speed_lt_30")
    rpm_pred   = STLNode.predicate("RPM",   "<", 2000.0, signal_index=1, node_id="RPM_lt_2000")
    brake_pred = STLNode.predicate("brake", "<", 250.0, signal_index=2, node_id="brake_lt_250")

    and_node = STLNode.and_node(speed_pred, rpm_pred, node_id="speed_and_RPM_slow")
    ev_node  = STLNode.eventually_node(and_node, interval=(0, 5), node_id="ev_0_5_slow")
    or_node  = STLNode.or_node(brake_pred, ev_node, node_id="brake_or_ev")
    formula  = STLNode.always_node(or_node, interval=(0, 30), node_id="alw_0_30_AT2")

    # k: always k_val, OR([1], ev with 1 split, AND([1],[1]))
    k = [k_val, [1, [1], [1, [1, [1], [1]]]]]

    return formula, k


def build_at3_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the AT3 specification: alw_[0,30](speed < 100)

    Signal mapping: speed=0
    k structure mirrors MATLAB: {k, {0}}
    """
    speed_pred = STLNode.predicate("speed", "<", 100.0, signal_index=0, node_id="speed_lt_100")
    formula    = STLNode.always_node(speed_pred, interval=(0, 30), node_id="alw_0_30_AT3")

    k = [k_val, [1]]

    return formula, k


def build_at5_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the AT5 (AT53) specification: ev_[0,30](speed > 70 AND RPM > 3800)

    Signal mapping: speed=0, RPM=1
    k structure mirrors MATLAB: {k, {0, {0}, {0}}}
    """
    speed_pred = STLNode.predicate("speed", ">", 70.0,  signal_index=0, node_id="speed_gt_70")
    rpm_pred   = STLNode.predicate("RPM",   ">", 3800.0, signal_index=1, node_id="RPM_gt_3800")

    and_node = STLNode.and_node(speed_pred, rpm_pred, node_id="speed_and_RPM_fast")
    formula  = STLNode.eventually_node(and_node, interval=(0, 30), node_id="ev_0_30_AT5")

    k = [k_val, [1, [1], [1]]]

    return formula, k


def build_afc_spec(k_val: int = 2) -> tuple[STLNode, list]:
    """
    Build the AFC1 specification:
        ev_[0,40](alw_[0,10]((AF_err > -0.05) AND (AF_err < 0.05)))

    where AF_err = AF - AFref is signal column 0.
    k structure mirrors MATLAB: {k_ev, {1, {0, {0}, {0}}}}
    The always inside the eventually always uses 1 temporal split (fixed).
    """
    af_lower = STLNode.predicate("AF_err", ">", -0.05, signal_index=0, node_id="AF_err_gt_neg005")
    af_upper = STLNode.predicate("AF_err", "<",  0.05, signal_index=0, node_id="AF_err_lt_005")

    and_node    = STLNode.and_node(af_lower, af_upper, node_id="AF_in_range")
    always_node = STLNode.always_node(and_node, interval=(0, 10), node_id="alw_0_10_AF")
    formula     = STLNode.eventually_node(always_node, interval=(0, 40), node_id="ev_0_40")

    # always inside ev uses 1 split (not k_val), matching MATLAB {k_ev, {1, ...}}
    k = [k_val, [1, [1, [1], [1]]]]

    return formula, k


SPEC_BUILDERS = {
    'at1':            build_at_spec,
    'at':             build_at_spec,          # alias
    'at2':            build_at2_spec,
    'at3':            build_at3_spec,
    'at5':            build_at5_spec,
    'afc':            build_afc_spec,
    'afc1':           build_afc_spec,         # alias
    'reach_avoid':    build_reach_avoid_spec,
    'reach_avoid_r4': build_reach_avoid_r4_spec,
}


def run_classification(
    traces: torch.Tensor,
    formula: STLNode,
    k: list,
    strategy_name: str = 'long_bs',
    device=None,
    dt: float = 1.0,
    max_time_per_node: float = 60.0,
    eval_devices=None,
):
    """Run classification and print results. Returns (result, classifier)."""
    strategy_cls = STRATEGIES[strategy_name]

    print(f"Strategy: {strategy_name}")
    print(f"Formula: {formula}")
    print(f"Traces shape: {traces.shape}")
    print(f"Device: {device}")
    if eval_devices is not None:
        print(f"Eval devices (robustness vmap): {eval_devices}")
    print("-" * 60)

    classifier = strategy_cls(
        formula=formula,
        k=k,
        traces=traces,
        device=device,
        dt=dt,
        max_time_per_node=max_time_per_node,
        eval_devices=eval_devices,
    )

    print(f"Lattice: {classifier.num_classes} refined formulas")
    print(f"Parse time: {classifier.time_split:.3f}s")
    print("-" * 60)

    result = classifier.solve()

    print(f"\nResults:")
    print(f"  Classes (total):     {result.num_classes}")
    print(f"  Classes (covered):   {result.num_covered}")
    print(f"  Parse time:          {result.time_split:.3f}s")
    print(f"  Classification time: {result.time_class:.3f}s")
    print(f"  Total time:          {result.time_total:.3f}s")
    print(f"  Synthesis calls:     {result.num_synth_calls}")

    return result, classifier


def main():
    parser = argparse.ArgumentParser(description="CEClass Autotrans Example")
    parser.add_argument("--data", type=str, help="Path to trace data (.mat/.npy)")
    parser.add_argument("--k", type=int, default=2, help="Hierarchy depth")
    parser.add_argument("--strategy", type=str, default="long_bs", choices=STRATEGIES.keys())
    parser.add_argument("--spec", type=str, default="at1",
                        choices=list(SPEC_BUILDERS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--plot-lattice", type=str, default=None,
                        help="Save lattice Hasse diagram to this path (e.g. lattice.png)")
    parser.add_argument("--plot-landscape", type=str, default=None,
                        help="Save robustness landscape for each parametric covered node to this prefix (e.g. landscape)")
    args = parser.parse_args()

    device = torch.device(args.device)

    formula, k = SPEC_BUILDERS[args.spec](args.k)

    if args.data:
        traces = load_traces(args.data, device=device)
    else:
        # Generate synthetic traces for testing
        print("No data provided, generating synthetic traces...")
        num_traces = 30
        timesteps = 50
        traces = torch.randn(num_traces, timesteps, 2, device=device)
        # Make speed oscillate around 90, RPM around 4000
        traces[:, :, 0] = 80 + 20 * torch.rand(num_traces, timesteps, device=device)
        traces[:, :, 1] = 3500 + 1000 * torch.rand(num_traces, timesteps, device=device)

    result, classifier = run_classification(
        traces=traces,
        formula=formula,
        k=k,
        strategy_name=args.strategy,
        device=device,
        dt=args.dt,
        max_time_per_node=args.max_time,
    )

    if args.plot_lattice or args.plot_landscape:
        from ceclass.viz import plot_lattice, plot_landscape

    if args.plot_lattice:
        fmt = args.plot_lattice.rsplit('.', 1)[-1] if '.' in args.plot_lattice else 'png'
        save = args.plot_lattice.rsplit('.', 1)[0] if '.' in args.plot_lattice else args.plot_lattice
        plot_lattice(classifier.graph, save_path=save, format=fmt,
                     title=f'{args.spec.upper()} k={args.k} ({args.strategy})')
        print(f"\nLattice diagram saved to: {args.plot_lattice}")

    if args.plot_landscape:
        count = 0
        for node in result.covered_nodes:
            pnames = node.formula.get_param_names()
            if not pnames:
                continue
            pbounds = classifier.parser.get_param_bounds_for_node(node)
            best_params = None
            if node.results:
                synth_result = node.results[0]
                if hasattr(synth_result, 'params_best'):
                    best_params = synth_result.params_best
            save = f"{args.plot_landscape}_{count}.png"
            plot_landscape(
                node.formula, traces, pnames, pbounds,
                device=device, dt=args.dt, grid_resolution=30,
                best_params=best_params, save_path=save, verbose=True,
                title=str(node.formula),
            )
            print(f"Landscape saved to: {save}")
            count += 1
        if count == 0:
            print("\nNo parametric covered nodes to plot landscapes for.")


if __name__ == "__main__":
    main()
