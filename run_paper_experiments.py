"""
Paper experiment runner for CEClass TACAS benchmarks.

Reproduces the experiments from:
  "Counterexample Classification for Cyber-Physical Systems" (2601.13743v1)

Benchmarks: AT1, AT2, AT3, AT5 (AT53), AFC1
Strategies:  NoPrune + AlwMid on every benchmark (matches *MainBS.m / AlwMid in TACAS)
k values:    1, 2, 3, 4

Traces are loaded from --data-dir (default: CEClassification/test/data/) and
processed in batch mode (all traces at once via stlcgpp vectorisation), which is
equivalent to the per-trace MATLAB loop in terms of coverage semantics.
Use --max-traces N to use only the first N rows (e.g. 30 for ablation).

Usage:
    python run_paper_experiments.py                       # all benchmarks
    python run_paper_experiments.py --bench AT1           # single benchmark
    python run_paper_experiments.py --bench AT1 --k 2     # single k
    python run_paper_experiments.py --strategy alw_mid    # override strategy
    python run_paper_experiments.py --device cpu          # force CPU
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from ceclass.examples.autotrans import (
    build_at_spec,
    build_at2_spec,
    build_at3_spec,
    build_at5_spec,
    build_afc_spec,
    run_classification,
    STRATEGIES,
)
from ceclass.utils.data import load_traces

DATA_DIR = Path("/home/parvk/CEClassification/test/data")

# ---------------------------------------------------------------------------
# Benchmark definitions (matching TACAS scripts exactly)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    name: str
    trace_file: str
    spec_builder: callable
    dt: float
    strategies: list[str]


BENCHMARKS = {
    "AT1": BenchmarkConfig(
        name="AT1",
        trace_file="AT1_traces.mat",
        spec_builder=build_at_spec,
        dt=0.01,
        strategies=["no_prune", "alw_mid"],
    ),
    "AT2": BenchmarkConfig(
        name="AT2",
        trace_file="AT2_traces.mat",
        spec_builder=build_at2_spec,
        dt=0.01,
        strategies=["no_prune", "alw_mid"],
    ),
    "AT3": BenchmarkConfig(
        name="AT3",
        trace_file="AT3_traces.mat",
        spec_builder=build_at3_spec,
        dt=0.01,
        strategies=["no_prune", "alw_mid"],
    ),
    "AT5": BenchmarkConfig(
        name="AT5",
        trace_file="AT53_traces.mat",
        spec_builder=build_at5_spec,
        dt=0.01,
        strategies=["no_prune", "alw_mid"],
    ),
    "AFC1": BenchmarkConfig(
        name="AFC1",
        trace_file="AFC1_traces.mat",
        spec_builder=build_afc_spec,
        dt=0.01,
        strategies=["no_prune", "alw_mid"],
    ),
}

K_VALUES = [1, 2, 3, 4]


def run_benchmark(
    bench: BenchmarkConfig,
    k_val: int,
    strategy_name: str,
    device: torch.device,
    max_time: float = 20.0,
    output_dir: Optional[Path] = None,
    data_dir: Path = DATA_DIR,
    max_traces: Optional[int] = None,
    eval_devices: Optional[Sequence[torch.device]] = None,
) -> dict:
    trace_path = data_dir / bench.trace_file
    print(f"\n{'='*70}")
    print(f"  {bench.name}  k={k_val}  strategy={strategy_name}")
    print(f"{'='*70}")

    traces = load_traces(trace_path, device=device)
    n_loaded = traces.shape[0]
    if max_traces is not None:
        traces = traces[: max_traces]
    num_traces = traces.shape[0]
    print(f"  Traces: {tuple(traces.shape)}  (loaded {n_loaded}, using {num_traces})  device={device}")

    formula, k = bench.spec_builder(k_val)

    result, _ = run_classification(
        traces=traces,
        formula=formula,
        k=k,
        strategy_name=strategy_name,
        device=device,
        dt=bench.dt,
        max_time_per_node=max_time,
        eval_devices=eval_devices,
    )

    row = {
        "bench":       bench.name,
        "k":           k_val,
        "strategy":    strategy_name,
        "num_traces":  num_traces,
        "num_classes": result.num_classes,
        "num_covered": result.num_covered,
        "time_split":  round(result.time_split, 4),
        "time_class":  round(result.time_class, 4),
        "time_total":  round(result.time_total, 4),
        "num_synth":   result.num_synth_calls,
    }

    if output_dir is not None:
        prefix = "BS_" if strategy_name == "alw_mid" else ""
        csv_path = output_dir / f"{prefix}{bench.name}_{k_val}.csv"
        _write_csv(csv_path, row)
        print(f"  Saved: {csv_path}")

    return row


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=";")
        writer.writeheader()
        writer.writerow(row)


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Bench':<6} {'k':>2} {'Strategy':<10} {'Classes':>8} {'Covered':>8} {'SplitT':>8} {'ClassT':>8} {'TotalT':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['bench']:<6} {r['k']:>2} {r['strategy']:<10} "
            f"{r['num_classes']:>8} {r['num_covered']:>8} "
            f"{r['time_split']:>8.2f} {r['time_class']:>8.2f} {r['time_total']:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="CEClass paper experiment runner")
    parser.add_argument("--bench", type=str, default=None,
                        choices=list(BENCHMARKS.keys()),
                        help="Run only this benchmark (default: all)")
    parser.add_argument("--k", type=int, default=None,
                        help="Run only this k value (default: 1,2,3,4)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help="Override strategy (default: per-benchmark default)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--max-time", type=float, default=20.0,
                        help="Max CMA-ES seconds per node (default: 20)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save CSV results (default: results/)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing *_traces.mat files",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N traces from each file (default: all)",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Robustness vmap on primary --device only (no cuda:0+cuda:1 split)",
    )
    parser.add_argument(
        "--eval-devices",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated devices for vmap, e.g. cuda:0,cuda:1 (overrides default)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        print("  CUDA not available, falling back to CPU.")

    benches = [BENCHMARKS[args.bench]] if args.bench else list(BENCHMARKS.values())
    k_vals  = [args.k] if args.k else K_VALUES
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    eval_devices: Optional[Tuple[torch.device, ...]] = None
    if args.single_gpu:
        eval_devices = (device,)
    elif args.eval_devices:
        eval_devices = tuple(
            torch.device(x.strip()) for x in args.eval_devices.split(",") if x.strip()
        )

    rows = []
    wall_start = time.time()

    for bench in benches:
        for k_val in k_vals:
            strategies = [args.strategy] if args.strategy else bench.strategies
            for strat in strategies:
                try:
                    row = run_benchmark(
                        bench=bench,
                        k_val=k_val,
                        strategy_name=strat,
                        device=device,
                        max_time=args.max_time,
                        output_dir=output_dir,
                        data_dir=data_dir,
                        max_traces=args.max_traces,
                        eval_devices=eval_devices,
                    )
                    rows.append(row)
                except Exception as exc:
                    print(f"  ERROR: {bench.name} k={k_val} {strat}: {exc}")
                    import traceback
                    traceback.print_exc()

    print_summary(rows)
    print(f"\nTotal wall time: {time.time() - wall_start:.1f}s")

    # Write consolidated summary CSV
    if rows:
        summary_path = output_dir / "summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
