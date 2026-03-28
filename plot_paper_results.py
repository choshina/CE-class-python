#!/usr/bin/env python3
"""
Publication-style figures from CEClass paper experiment CSV.

Outputs (default directory: paper_figures/):
  1. time_comparison.png       — Per-benchmark subplots: NoPrune vs AlwMid time (Fig. 4 style)
  2. speedup_vs_k.png          — Speedup ratio (NoPrune / AlwMid) vs k
  3. synth_calls_comparison.png — Synthesis calls saved: all benchmarks
  4. coverage_comparison.png   — Coverage fraction: NoPrune vs AlwMid side-by-side
  5. summary_table.png         — Comprehensive table with both strategies

Usage:
  python plot_paper_results.py
  python plot_paper_results.py --csv results_100/summary.csv --out paper_figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BENCH_ORDER = ["AT1", "AT2", "AT3", "AT5", "AFC1", "Robot"]
BENCH_LABELS = {
    "AT1": r"$\varphi_1^{AT}$",
    "AT2": r"$\varphi_2^{AT}$",
    "AT3": r"$\varphi_3^{AT}$",
    "AT5": r"$\varphi_5^{AT}$",
    "AFC1": r"$\varphi_1^{AFC}$",
    "Robot": r"$\varphi^{Rob}$",
}
K_ORDER = [1, 2, 3, 4]

NP_COLOR = "#2E7D32"
AM_COLOR = "#1565C0"
NP_LABEL = "NoPrune (Baseline)"
AM_LABEL = "AlwMid (Binary Search)"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            rows.append(row)
    return rows


def get_val(rows: list[dict], bench: str, k: int, strategy: str, key: str) -> float | None:
    for r in rows:
        if r["bench"] == bench and int(r["k"]) == k and r["strategy"] == strategy:
            return float(r[key])
    return None


def get_int(rows: list[dict], bench: str, k: int, strategy: str, key: str) -> int | None:
    for r in rows:
        if r["bench"] == bench and int(r["k"]) == k and r["strategy"] == strategy:
            return int(r[key])
    return None


# ── Figure 1: Per-benchmark time comparison (paper Fig. 4 style) ─────────

def plot_time_comparison(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(21, 4), dpi=150, sharey=False)

    x = np.arange(len(K_ORDER))
    w = 0.35

    for ax, bench in zip(axes, BENCH_ORDER):
        np_times = []
        am_times = []
        for k in K_ORDER:
            np_t = get_val(rows, bench, k, "no_prune", "time_class")
            am_t = get_val(rows, bench, k, "alw_mid", "time_class")
            np_times.append(np_t if np_t is not None else 0)
            am_times.append(am_t if am_t is not None else 0)

        bars_np = ax.bar(x - w / 2, np_times, w, label=NP_LABEL, color=NP_COLOR, alpha=0.85)
        bars_am = ax.bar(x + w / 2, am_times, w, label=AM_LABEL, color=AM_COLOR, alpha=0.85)

        for bar, val in zip(bars_np, np_times):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, color=NP_COLOR)
        for bar, val in zip(bars_am, am_times):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, color=AM_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_ORDER])
        ax.set_xlabel("$k$", fontsize=11)
        ax.set_title(f"{bench}  ({BENCH_LABELS[bench]})", fontsize=11, fontweight="bold")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.grid(True, axis="y", alpha=0.3, which="both")

    axes[0].set_ylabel("Classification time (s, log scale)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.08), frameon=True, edgecolor="gray")
    fig.suptitle("Classification Time: NoPrune (Baseline) vs AlwMid (Binary Search)",
                 fontsize=14, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 2: Speedup ratio vs k ─────────────────────────────────────────

def plot_speedup_vs_k(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    cmap = plt.cm.Set1(np.linspace(0, 0.6, len(BENCH_ORDER)))
    markers = ["o", "s", "D", "^", "v", "P"]

    for i, bench in enumerate(BENCH_ORDER):
        ks_plot, speedups = [], []
        for k in K_ORDER:
            np_t = get_val(rows, bench, k, "no_prune", "time_class")
            am_t = get_val(rows, bench, k, "alw_mid", "time_class")
            if np_t is not None and am_t is not None and am_t > 0:
                ks_plot.append(k)
                speedups.append(np_t / am_t)
        if ks_plot:
            ax.plot(ks_plot, speedups, f"{markers[i]}-", linewidth=2.2, markersize=8,
                    label=f"{bench} ({BENCH_LABELS[bench]})", color=cmap[i])

    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Hierarchy depth $k$", fontsize=12)
    ax.set_ylabel("Speedup (NoPrune time / AlwMid time)", fontsize=12)
    ax.set_title("Speedup from Binary Search Strategy vs Hierarchy Depth",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(K_ORDER)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 3: Synthesis calls comparison (all benchmarks) ─────────────────

def plot_synth_calls(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(21, 4), dpi=150, sharey=False)

    x = np.arange(len(K_ORDER))
    w = 0.35

    for ax, bench in zip(axes, BENCH_ORDER):
        np_synth = [get_int(rows, bench, k, "no_prune", "num_synth") or 0 for k in K_ORDER]
        am_synth = [get_int(rows, bench, k, "alw_mid", "num_synth") or 0 for k in K_ORDER]

        ax.bar(x - w / 2, np_synth, w, label=NP_LABEL, color=NP_COLOR, alpha=0.85)
        ax.bar(x + w / 2, am_synth, w, label=AM_LABEL, color=AM_COLOR, alpha=0.85)

        for j, k in enumerate(K_ORDER):
            total = np_synth[j]
            saved = np_synth[j] - am_synth[j]
            if total > 0 and saved > 0:
                pct = 100 * saved / total
                ax.text(x[j], max(np_synth[j], am_synth[j]) * 1.02,
                        f"−{pct:.0f}%", ha="center", va="bottom",
                        fontsize=7, color="#B71C1C", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_ORDER])
        ax.set_xlabel("$k$", fontsize=11)
        ax.set_title(f"{bench}  ({BENCH_LABELS[bench]})", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Synthesis calls (membership queries)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.08), frameon=True, edgecolor="gray")
    fig.suptitle("Membership Queries: NoPrune Checks Every Node vs AlwMid Prunes via Partial Order",
                 fontsize=13, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 4: Coverage comparison ─────────────────────────────────────────

def plot_coverage_comparison(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(21, 3.8), dpi=150, sharey=True)

    x = np.arange(len(K_ORDER))
    w = 0.35

    for ax, bench in zip(axes, BENCH_ORDER):
        np_cov, am_cov = [], []
        for k in K_ORDER:
            nc_np = get_int(rows, bench, k, "no_prune", "num_covered")
            cl_np = get_int(rows, bench, k, "no_prune", "num_classes")
            nc_am = get_int(rows, bench, k, "alw_mid", "num_covered")
            cl_am = get_int(rows, bench, k, "alw_mid", "num_classes")
            np_cov.append(nc_np / cl_np if nc_np and cl_np else 0)
            am_cov.append(nc_am / cl_am if nc_am and cl_am else 0)

        ax.bar(x - w / 2, np_cov, w, label=NP_LABEL, color=NP_COLOR, alpha=0.85)
        ax.bar(x + w / 2, am_cov, w, label=AM_LABEL, color=AM_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_ORDER])
        ax.set_xlabel("$k$", fontsize=11)
        ax.set_title(f"{bench}  ({BENCH_LABELS[bench]})", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Coverage (covered / classes)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.08), frameon=True, edgecolor="gray")
    fig.suptitle("Coverage Is Preserved: AlwMid Achieves Same Coverage as NoPrune",
                 fontsize=13, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 5: Comprehensive summary table ────────────────────────────────

def plot_summary_table(rows: list[dict], out: Path) -> None:
    header = ["Bench", "$k$", "Classes",
              "Covered\n(NP)", "Covered\n(AM)",
              "Time NP\n(s)", "Time AM\n(s)", "Speedup",
              "Synth\nNP", "Synth\nAM", "Synth\nSaved"]
    cells = []
    cell_colors = []

    for bench in BENCH_ORDER:
        for k in K_ORDER:
            cl = get_int(rows, bench, k, "no_prune", "num_classes")
            nc_np = get_int(rows, bench, k, "no_prune", "num_covered")
            nc_am = get_int(rows, bench, k, "alw_mid", "num_covered")
            t_np = get_val(rows, bench, k, "no_prune", "time_class")
            t_am = get_val(rows, bench, k, "alw_mid", "time_class")
            s_np = get_int(rows, bench, k, "no_prune", "num_synth")
            s_am = get_int(rows, bench, k, "alw_mid", "num_synth")

            speedup = t_np / t_am if t_np and t_am and t_am > 0 else 0
            synth_saved = f"{100 * (1 - s_am / s_np):.0f}%" if s_np and s_am else "—"

            row = [
                bench, str(k), str(cl or "—"),
                f"{nc_np}/{cl}" if nc_np is not None else "—",
                f"{nc_am}/{cl}" if nc_am is not None else "—",
                f"{t_np:.2f}" if t_np is not None else "—",
                f"{t_am:.2f}" if t_am is not None else "—",
                f"{speedup:.1f}×" if speedup > 0 else "—",
                str(s_np) if s_np is not None else "—",
                str(s_am) if s_am is not None else "—",
                synth_saved,
            ]
            cells.append(row)

            base = "#F5F5F5" if BENCH_ORDER.index(bench) % 2 == 0 else "#FFFFFF"
            cell_colors.append([base] * len(header))

    nrows = len(cells)
    fig_h = 1.5 + nrows * 0.38
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=cells,
        colLabels=header,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("#BDBDBD")

    ax.set_title("Classification Results Summary",
                 fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot paper figures from summary CSV")
    ap.add_argument("--csv", type=Path, default=Path("results_100/summary.csv"))
    ap.add_argument("--out", type=Path, default=Path("paper_figures"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })

    plot_time_comparison(rows, args.out / "time_comparison.png")
    plot_speedup_vs_k(rows, args.out / "speedup_vs_k.png")
    plot_synth_calls(rows, args.out / "synth_calls_comparison.png")
    plot_coverage_comparison(rows, args.out / "coverage_comparison.png")
    plot_summary_table(rows, args.out / "summary_table.png")


if __name__ == "__main__":
    main()
