"""
Run CEClass counterexample classification for the robot navigation example.

Formula:
    Phi = ev[0,25]( goal1 AND ev[0,25](goal2) ) AND alw[0,50]( NOT danger )

Traces are 3D signals: (d_goal1, d_goal2, d_danger) where >0 means inside region.

Usage:
    python -m examples.robot.run_classification --k 2 --strategy alw_mid
    python -m examples.robot.run_classification --k 2 --strategy no_prune
"""
import argparse
import torch
import numpy as np

from ceclass.examples.autotrans import run_classification, STRATEGIES
from examples.robot.gen_counterexamples import build_formula, build_k


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
