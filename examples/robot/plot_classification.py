"""
Run CEClass and plot per-trace most-specific covered nodes:
  - each trace assigned to the strongest covered formula it satisfies
  - one subplot per unique class
  - formula shown as title
"""
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from ceclass.examples.autotrans import build_reach_avoid_r4_spec
from ceclass.strategies.bfs import BFSClassifier
from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT       = 1.0
K        = 1
TRACES_F = "counterexamples.npy"

REGIONS = {
    "R1 (start)":  (0,  5,  0,  5,  "#2ecc71"),
    "R3 (unsafe)": (8,  13, 8,  13, "#e74c3c"),
    "R4 (goal)":   (15, 20, 15, 20, "#3498db"),
}
COLORS = ["#e67e22", "#8e44ad", "#2980b9", "#27ae60", "#c0392b",
          "#f39c12", "#1abc9c", "#e91e63"]

# ── Run classification ───────────────────────────────────────────────────────
traces_np = np.load(TRACES_F)
traces    = torch.tensor(traces_np, dtype=torch.float32, device=DEVICE)

formula, k = build_reach_avoid_r4_spec(K)
clf = BFSClassifier(formula=formula, k=k, traces=traces, device=DEVICE, dt=DT)
result = clf.solve()

covered = [n for n in clf.graph.get_covered_nodes()
           if n.formula.node_type != "true"]
print(f"Total covered nodes (excl. TRUE): {len(covered)}")

# ── Evaluate membership for every covered node ───────────────────────────────
def get_member_mask(node_formula: STLNode) -> np.ndarray:
    try:
        stl = to_stlcgpp(node_formula, params={}, device=DEVICE, dt=DT)
        with torch.no_grad():
            rob = torch.vmap(stl)(traces)
        return (rob[:, 0] > 0).cpu().numpy()
    except Exception as e:
        print(f"  Warning: could not evaluate {node_formula}: {e}")
        return np.zeros(len(traces), dtype=bool)

print("Evaluating membership for all covered nodes...")
masks = {n.formula.id: get_member_mask(n.formula) for n in covered}

# ── Per-trace assignment: most specific covered node ─────────────────────────
# A node is "most specific for trace i" if:
#   - trace i satisfies it
#   - no stronger (greater_imme) covered node also satisfies trace i
covered_ids = {n.formula.id for n in covered}

def most_specific_for_trace(trace_idx):
    satisfying = [n for n in covered if masks[n.formula.id][trace_idx]]
    if not satisfying:
        return None
    # Keep nodes with no stronger covered ancestor also satisfied by this trace
    satisfying_ids = {n.formula.id for n in satisfying}
    frontier = [
        n for n in satisfying
        if not any(g.formula.id in satisfying_ids for g in n.greater_imme)
    ]
    return frontier[0] if frontier else satisfying[0]

assignments = {}   # node_id -> (node, list of trace indices)
unclassified = []
for i in range(len(traces_np)):
    node = most_specific_for_trace(i)
    if node is None:
        unclassified.append(i)
        continue
    nid = node.formula.id
    if nid not in assignments:
        assignments[nid] = (node, [])
    assignments[nid][1].append(i)

classes = list(assignments.values())
print(f"\nClasses found: {len(classes)}")
for i, (node, idxs) in enumerate(classes):
    print(f"  [{i+1}] {len(idxs)} traces — {node.formula}")
if unclassified:
    print(f"  Unclassified: {len(unclassified)} traces")

# ── Plot ─────────────────────────────────────────────────────────────────────
ncols = len(classes)
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                         sharex=True, sharey=True)
if ncols == 1:
    axes = [axes]

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

all_member_indices = set(i for _, idxs in classes for i in idxs)

for ax, (node, idxs), color in zip(axes, classes, COLORS):
    draw_regions(ax)

    member_set = set(idxs)
    for i, tr in enumerate(traces_np):
        if i in member_set:
            ax.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.6,
                    linewidth=1.0, zorder=4)
            ax.plot(tr[0, 0],  tr[0, 1],  "o", color=color,
                    markersize=4, alpha=0.8, zorder=5)
            ax.plot(tr[-1, 0], tr[-1, 1], "s", color=color,
                    markersize=4, alpha=0.8, zorder=5)
        else:
            ax.plot(tr[:, 0], tr[:, 1], color="#cccccc", alpha=0.25,
                    linewidth=0.6, zorder=3)

    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 22)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(
        f"Class {COLORS.index(color)+1}  ({len(idxs)} traces)\n{node.formula}",
        fontsize=8, wrap=True
    )

# Shared legend
legend_elements = [
    mpatches.Patch(facecolor="#2ecc71", alpha=0.4, label="R1: 0<x<5, 0<y<5"),
    mpatches.Patch(facecolor="#e74c3c", alpha=0.4, label="R3: 8<x<13, 8<y<13 (unsafe)"),
    mpatches.Patch(facecolor="#3498db", alpha=0.4, label="R4: 15<x<20, 15<y<20 (goal)"),
    Line2D([0], [0], marker="o", color="gray", lw=0, markersize=5, label="Start"),
    Line2D([0], [0], marker="s", color="gray", lw=0, markersize=5, label="End"),
    Line2D([0], [0], color="#cccccc", lw=1.5, label="Other traces"),
]
for i, color in enumerate(COLORS[:ncols]):
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f"Class {i+1}"))

fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    "CEClass — Per-Trace Most Specific Covered Class\n"
    "Φ = ♢[0,10](R1 ∧ ♢[20,30](R4)) ∧ □[0,60](¬R3)   |   k=1",
    fontsize=11
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("classification_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved: classification_plot.png")

# ── Save to MATLAB ────────────────────────────────────────────────────────────
class_labels = np.zeros(len(traces_np), dtype=int)   # 0 = unclassified
class_formulas = {}
for cls_idx, (node, idxs) in enumerate(classes, start=1):
    for i in idxs:
        class_labels[i] = cls_idx
    class_formulas[f"class{cls_idx}_formula"] = str(node.formula)

scipy.io.savemat("classification_results.mat", {
    "traces":       traces_np,
    "class_labels": class_labels,
    **{f"class{i+1}_formula": np.array([str(node.formula)], dtype=object)
       for i, (node, _) in enumerate(classes)},
})
print("Saved: classification_results.mat  (vars: traces, class_labels, class1_formula, ...)")
