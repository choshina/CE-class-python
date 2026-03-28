"""
Plot the 75 counterexamples in 2D (x, y) with region bounds — one panel per type.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

traces = np.load("counterexamples.npy")             # (75, 70, 2)
robs   = np.load("counterexample_robustness.npy")   # (75,)

REGIONS = {
    "R1\n(start)":  (0,  5,  0,  5,  "#2ecc71"),
    "R3\n(unsafe)": (8,  13, 8,  13, "#e74c3c"),
    "R4\n(goal)":   (15, 20, 15, 20, "#3498db"),
}

TYPES = [
    (slice(0,  25), "#e67e22", "Type A",
     "violates ♢[0,10](R1∧…)\nnever reaches R1 in [0,10]"),
    (slice(25, 50), "#8e44ad", "Type B",
     "violates inner ♢[20,30](R4)\nreaches R1 but never reaches R4"),
    (slice(50, 75), "#2980b9", "Type C",
     "violates □[0,60](¬R3)\nreaches R1 & R4 but enters R3"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

def draw_regions(ax):
    for rname, (xl, xh, yl, yh, fc) in REGIONS.items():
        rect = mpatches.FancyBboxPatch(
            (xl, yl), xh - xl, yh - yl,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor=fc, facecolor=fc, alpha=0.25, zorder=2)
        ax.add_patch(rect)
        ax.text((xl + xh) / 2, (yl + yh) / 2, rname,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=fc, zorder=5)

for ax, (sl, color, tag, desc) in zip(axes, TYPES):
    draw_regions(ax)
    sub     = traces[sl]
    sub_rob = robs[sl]
    for i in range(len(sub)):
        x, y = sub[i, :, 0], sub[i, :, 1]
        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.8, zorder=3)
        ax.plot(x[0],  y[0],  "o", color=color, markersize=3, alpha=0.8, zorder=4)
        ax.plot(x[-1], y[-1], "s", color=color, markersize=3, alpha=0.8, zorder=4)

    ax.set_title(f"{tag} — {desc}\nρ ∈ [{sub_rob.min():.2f}, {sub_rob.max():.2f}]",
                 fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 22)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

legend_elements = [
    mpatches.Patch(facecolor="#2ecc71", alpha=0.4, label="R1: 0<x<5, 0<y<5  (start)"),
    mpatches.Patch(facecolor="#e74c3c", alpha=0.4, label="R3: 8<x<13, 8<y<13  (unsafe)"),
    mpatches.Patch(facecolor="#3498db", alpha=0.4, label="R4: 15<x<20, 15<y<20  (goal)"),
    Line2D([0],[0], color="#e67e22", lw=1.5, label="Type A"),
    Line2D([0],[0], color="#8e44ad", lw=1.5, label="Type B"),
    Line2D([0],[0], color="#2980b9", lw=1.5, label="Type C"),
    Line2D([0],[0], marker="o", color="gray", lw=0, markersize=5, label="Start"),
    Line2D([0],[0], marker="s", color="gray", lw=0, markersize=5, label="End"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.05))

fig.suptitle(
    "Φ = ♢[0,10](R1 ∧ ♢[20,30](R4)) ∧ □[0,60](¬R3)\n"
    "75 counterexamples — all ρ < 0",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("counterexamples_plot.png", dpi=150, bbox_inches="tight")
print("Saved counterexamples_plot.png")
