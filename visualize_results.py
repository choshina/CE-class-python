"""
Visualization script for CEClass paper experiments.

Re-runs selected benchmarks, saves:
  - Lattice Hasse diagram (PNG, color-coded: green=covered, red=pruned)
  - Text summary of covered formulas
  - Combined summary figure

Usage:
    python visualize_results.py                    # all interesting cases
    python visualize_results.py --bench AT1 --k 3  # single case
    python visualize_results.py --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

from ceclass.examples.autotrans import (
    build_at_spec, build_at2_spec, build_at3_spec, build_at5_spec, build_afc_spec,
    run_classification, STRATEGIES,
)
from ceclass.utils.data import load_traces
from ceclass.viz import plot_lattice

DATA_DIR = Path("/home/parvk/CEClassification/test/data")
OUT_DIR  = Path("viz_results")

SPEC_BUILDERS = {
    "AT1":  (build_at_spec,  "AT1_traces.mat",  0.01),
    "AT2":  (build_at2_spec, "AT2_traces.mat",  0.01),
    "AT3":  (build_at3_spec, "AT3_traces.mat",  0.01),
    "AT5":  (build_at5_spec, "AT53_traces.mat", 0.01),
    "AFC1": (build_afc_spec, "AFC1_traces.mat", 0.01),
}

# Cases to visualize: (bench, k, strategy) — NoPrune + AlwMid per paper TACAS *MainBS.m
DEFAULT_CASES = [
    ("AT1",  3, "no_prune"),
    ("AT1",  3, "alw_mid"),
    ("AT1",  4, "no_prune"),
    ("AT1",  4, "alw_mid"),
    ("AT2",  3, "no_prune"),
    ("AT2",  3, "alw_mid"),
    ("AT3",  4, "no_prune"),
    ("AT3",  4, "alw_mid"),
    ("AT5",  4, "no_prune"),
    ("AT5",  4, "alw_mid"),
    ("AFC1", 3, "no_prune"),
    ("AFC1", 3, "alw_mid"),
]


def run_and_visualize(
    bench: str,
    k_val: int,
    strategy: str,
    device: torch.device,
    max_time: float = 20.0,
    eval_devices=None,
) -> None:
    builder, trace_file, dt = SPEC_BUILDERS[bench]
    formula, k = builder(k_val)
    traces = load_traces(DATA_DIR / trace_file, device=device)

    print(f"\n[{bench} k={k_val} {strategy}]")
    result, classifier = run_classification(
        traces=traces, formula=formula, k=k,
        strategy_name=strategy, device=device, dt=dt,
        max_time_per_node=max_time,
        eval_devices=eval_devices,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{bench}_k{k_val}_{strategy}"

    # 1. Lattice diagram
    lattice_path = str(OUT_DIR / tag)
    plot_lattice(
        classifier.graph,
        save_path=lattice_path,
        format="png",
        title=f"{bench}  k={k_val}  [{strategy}]  "
              f"{result.num_covered}/{result.num_classes} covered  "
              f"{result.time_total:.1f}s",
    )
    print(f"  Lattice saved → {lattice_path}.png")

    # 2. Covered formulas text file
    txt_path = OUT_DIR / f"{tag}_covered.txt"
    with open(txt_path, "w") as f:
        f.write(f"Benchmark : {bench}\n")
        f.write(f"k         : {k_val}\n")
        f.write(f"Strategy  : {strategy}\n")
        f.write(f"Covered   : {result.num_covered} / {result.num_classes}\n")
        f.write(f"Time      : {result.time_total:.3f}s\n")
        f.write(f"Synth calls: {result.num_synth_calls}\n")
        f.write("\n--- Covered formulas ---\n")
        for node in sorted(result.covered_nodes, key=lambda n: str(n.formula)):
            f.write(f"  {node.formula}\n")
            if node.results:
                r = node.results[0]
                f.write(f"    obj_best={r.obj_best:.4f}  evals={r.num_evals}\n")
    print(f"  Covered formulas → {txt_path}")


def make_comparison_figure(cases: list[tuple]) -> None:
    """Build a multi-panel figure with all lattice PNGs."""
    imgs = []
    for bench, k_val, strategy in cases:
        p = OUT_DIR / f"{bench}_k{k_val}_{strategy}.png"
        if p.exists():
            imgs.append((str(p), f"{bench} k={k_val} [{strategy}]"))

    if not imgs:
        return

    ncols = 2
    nrows = (len(imgs) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 10 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (img_path, title) in zip(axes, imgs):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.suptitle("CEClass — Refinement Lattice Classification Results", fontsize=16, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nOverview figure saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=None, choices=list(SPEC_BUILDERS.keys()))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=list(STRATEGIES.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Robustness vmap on --device only (no dual-GPU trace split)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    eval_devices = (device,) if args.single_gpu else None

    if args.bench:
        strategies = [args.strategy] if args.strategy else ["no_prune", "alw_mid"]
        k_vals = [args.k] if args.k else [1, 2, 3, 4]
        cases = [(args.bench, k, s) for k in k_vals for s in strategies]
    else:
        cases = DEFAULT_CASES

    for bench, k_val, strategy in cases:
        try:
            run_and_visualize(bench, k_val, strategy, device, args.max_time, eval_devices)
        except Exception as e:
            print(f"  ERROR: {bench} k={k_val} {strategy}: {e}")
            import traceback; traceback.print_exc()

    make_comparison_figure(cases)


if __name__ == "__main__":
    main()
