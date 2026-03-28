"""
Run CEClass counterexample classification for the robot navigation example.

Formula:
    Phi = ev[0,25]( goal1 AND ev[0,25](goal2) ) AND alw[0,50]( NOT danger )

Regions (2D: signal 0=x, signal 1=y):
    goal1:  1<x<5,  1<y<5
    goal2:  15<x<19, 15<y<19
    danger: 8<x<12,  8<y<12

Usage:
    python -m examples.robot.run_classification --k 2 --strategy alw_mid
    python -m examples.robot.run_classification --k 3 --strategy no_prune
"""
import argparse
import torch
import numpy as np

from ceclass.formula.stl_node import STLNode
from ceclass.examples.autotrans import run_classification, STRATEGIES
from examples.robot.gen_counterexamples import build_formula


def build_k(k_val: int):
    """
    Build the nested k list matching the formula tree.

    Formula tree:
        Phi (AND)
        ├── phi1: ev[0,25](...)             -- temporal, k splits
        │   └── AND
        │       ├── goal1 (nary_and of 4 preds)
        │       └── ev[0,25](...)           -- temporal, k splits
        │           └── goal2 (nary_and of 4 preds)
        └── phi2: alw[0,50](...)            -- temporal, k splits
            └── NOT
                └── danger (nary_and of 4 preds)
    """
    p = [1]
    k_R = [1, [1, [1, p, p], p], p]  # 4-pred nary_and

    k_ev_goal2 = [k_val, k_R]                  # ev[0,25](goal2)
    k_g1_and   = [1, k_R, k_ev_goal2]          # goal1 AND ev(goal2)
    k_phi1     = [k_val, k_g1_and]              # ev[0,25](goal1 AND ...)
    k_phi2     = [k_val, [1, k_R]]              # alw[0,50](NOT danger)
    return [1, k_phi1, k_phi2]                  # Phi = AND(phi1, phi2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="alw_mid", choices=STRATEGIES.keys())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--traces", type=str, default="examples/robot/counterexamples.npy")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    traces_np = np.load(args.traces)
    traces = torch.tensor(traces_np, dtype=torch.float32, device=device)
    print(f"Loaded traces: {traces.shape}")

    formula = build_formula()
    k = build_k(args.k)
    print(f"Formula: {formula}")
    print(f"k={args.k}  strategy={args.strategy}\n")

    result, classifier = run_classification(
        traces=traces,
        formula=formula,
        k=k,
        strategy_name=args.strategy,
        device=device,
        dt=1.0,
        max_time_per_node=args.max_time,
    )


if __name__ == "__main__":
    main()
