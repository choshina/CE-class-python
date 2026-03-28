"""
Generate 75 counterexamples for the robot navigation specification:

    Phi = ev[0,25]( goal1 AND ev[0,25](goal2) ) AND alw[0,50]( NOT danger )

2D workspace but traces are augmented with region-membership signals:
    signal 0: d_goal1  = min(x-1, 5-x, y-1, 5-y)    (>0 iff in goal1)
    signal 1: d_goal2  = min(x-15, 19-x, y-15, 19-y) (>0 iff in goal2)
    signal 2: d_danger = min(x-8, 12-x, y-8, 12-y)   (>0 iff in danger)

This makes each region an atomic predicate, keeping lattice size tractable.

Counterexample types (25 each):
    Type A -- never reach goal1 in [0,25]
    Type B -- reach goal1 but never reach goal2
    Type C -- reach goal1 and goal2 but pass through danger
"""
import argparse
import torch
import numpy as np

from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp

T  = 51
dt = 1.0
N  = 75

GOAL1  = (1, 5, 1, 5)
GOAL2  = (15, 19, 15, 19)
DANGER = (8, 12, 8, 12)


def build_formula():
    """ev[0,25]( goal1 AND ev[0,25](goal2) ) AND alw[0,50]( NOT danger )"""
    goal1  = STLNode.predicate("d_goal1",  ">", 0.0, signal_index=0, node_id="goal1")
    goal2  = STLNode.predicate("d_goal2",  ">", 0.0, signal_index=1, node_id="goal2")
    danger = STLNode.predicate("d_danger", ">", 0.0, signal_index=2, node_id="danger")

    ev_goal2     = STLNode.eventually_node(goal2, interval=(0, 25), node_id="ev_goal2")
    g1_and_ev_g2 = STLNode.and_node(goal1, ev_goal2, node_id="goal1_and_ev_goal2")
    phi1         = STLNode.eventually_node(g1_and_ev_g2, interval=(0, 25), node_id="phi1")
    phi2         = STLNode.always_node(
        STLNode.not_node(danger, node_id="not_danger"),
        interval=(0, 50), node_id="phi2",
    )
    return STLNode.and_node(phi1, phi2, node_id="Phi")


def build_k(k_val: int):
    """
    Formula tree:
        Phi (AND)
        ├── phi1: ev[0,25](...)           -- k splits
        │   └── AND
        │       ├── goal1                 -- atomic pred
        │       └── ev[0,25](goal2)       -- k splits
        │           └── goal2             -- atomic pred
        └── phi2: alw[0,50](NOT danger)   -- k splits
            └── NOT
                └── danger                -- atomic pred
    """
    p = [1]
    k_ev_goal2 = [k_val, p]               # ev[0,25](goal2)
    k_g1_and   = [1, p, k_ev_goal2]       # goal1 AND ev(goal2)
    k_phi1     = [k_val, k_g1_and]         # ev[0,25](...)
    k_phi2     = [k_val, [1, p]]           # alw[0,50](NOT danger)
    return [1, k_phi1, k_phi2]             # AND(phi1, phi2)


def _region_dist(xy, box):
    """Signed distance: >0 inside the box."""
    x_lo, x_hi, y_lo, y_hi = box
    return np.minimum(
        np.minimum(xy[:, 0] - x_lo, x_hi - xy[:, 0]),
        np.minimum(xy[:, 1] - y_lo, y_hi - xy[:, 1]),
    )


def xy_to_signals(xy_traces):
    """(N, T, 2) raw xy -> (N, T, 3) region-distance signals."""
    N, T, _ = xy_traces.shape
    sigs = np.zeros((N, T, 3))
    for i in range(N):
        sigs[i, :, 0] = _region_dist(xy_traces[i], GOAL1)
        sigs[i, :, 1] = _region_dist(xy_traces[i], GOAL2)
        sigs[i, :, 2] = _region_dist(xy_traces[i], DANGER)
    return sigs


def smooth_trace(start_xy, waypoints, T):
    trace = np.zeros((T, 2))
    pts = [(0, start_xy[0], start_xy[1])] + waypoints
    for i in range(len(pts) - 1):
        t0, x0, y0 = pts[i]
        t1, x1, y1 = pts[i + 1]
        t0, t1 = int(t0), int(t1)
        if t0 == t1:
            continue
        for t in range(t0, min(t1, T)):
            alpha = (t - t0) / (t1 - t0)
            trace[t, 0] = x0 + alpha * (x1 - x0)
            trace[t, 1] = y0 + alpha * (y1 - y0)
    last_t, last_x, last_y = pts[-1]
    for t in range(min(int(last_t), T), T):
        trace[t, 0] = last_x
        trace[t, 1] = last_y
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="examples/robot")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    Phi = build_formula()
    print(f"Formula: {Phi}\n")
    formula_stlcg = to_stlcgpp(Phi, params={}, device=device, dt=dt)

    xy_list = []

    # Type A: never reach goal1 — wander in neutral zone [6,7.5]
    print("Generating Type A (never reach goal1)...")
    for _ in range(25):
        trace = np.zeros((T, 2))
        bx, by = rng.uniform(6.0, 7.5), rng.uniform(6.0, 7.5)
        noise = rng.standard_normal((T, 2)) * 0.3
        trace[:, 0] = np.clip(bx + noise[:, 0], 5.5, 7.9)
        trace[:, 1] = np.clip(by + noise[:, 1], 5.5, 7.9)
        xy_list.append(trace)

    # Type B: reach goal1, but never reach goal2 — get stuck
    print("Generating Type B (reach goal1, never goal2)...")
    for _ in range(25):
        sx, sy = rng.uniform(1.5, 4.5), rng.uniform(1.5, 4.5)
        mx, my = rng.uniform(5.5, 7.5), rng.uniform(5.5, 7.5)
        trace = smooth_trace(
            (sx, sy),
            [(12, mx, my),
             (T - 1, mx + rng.uniform(-0.5, 0.5), my + rng.uniform(-0.5, 0.5))],
            T,
        )
        xy_list.append(trace)

    # Type C: reach goal1 and goal2 but pass through danger
    print("Generating Type C (reach both goals, enter danger)...")
    for _ in range(25):
        sx, sy = rng.uniform(1.5, 4.5), rng.uniform(1.5, 4.5)
        t_d = rng.integers(10, 18)
        dx, dy = rng.uniform(8.5, 11.5), rng.uniform(8.5, 11.5)
        t_g2 = rng.integers(22, 35)
        gx, gy = rng.uniform(15.5, 18.5), rng.uniform(15.5, 18.5)
        trace = smooth_trace(
            (sx, sy),
            [(t_d, dx, dy), (t_g2, gx, gy), (T - 1, gx, gy)],
            T,
        )
        xy_list.append(trace)

    xy_traces = np.stack(xy_list, axis=0)       # (75, T, 2) raw positions
    sig_traces = xy_to_signals(xy_traces)        # (75, T, 3) region distances
    traces_t = torch.tensor(sig_traces, dtype=torch.float32, device=device)

    print(f"\nXY shape: {xy_traces.shape}  Signal shape: {sig_traces.shape}")
    print("Computing robustness...")

    with torch.no_grad():
        rob_all = torch.vmap(formula_stlcg)(traces_t)

    rob_cpu = rob_all[:, 0].cpu().numpy()
    n_neg = (rob_cpu < 0).sum()
    print(f"  Min: {rob_cpu.min():.4f}  Max: {rob_cpu.max():.4f}  Negative: {n_neg}/{N}")

    out = args.out_dir
    np.save(f"{out}/counterexamples_xy.npy", xy_traces)
    np.save(f"{out}/counterexamples.npy", sig_traces)
    print(f"\nSaved: {out}/counterexamples_xy.npy  (raw positions, shape {xy_traces.shape})")
    print(f"Saved: {out}/counterexamples.npy  (region signals, shape {sig_traces.shape})")


if __name__ == "__main__":
    main()
