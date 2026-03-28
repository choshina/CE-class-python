"""
Generate 75 counterexamples for the robot navigation specification:

    Phi = ev[0,25]( goal1 AND ev[0,25](goal2) ) AND alw[0,50]( NOT danger )

2D state space: signal 0 = x, signal 1 = y

Region definitions:
    goal1:  x in (1, 5)  AND y in (1, 5)    -- first waypoint
    goal2:  x in (15,19) AND y in (15,19)   -- second waypoint
    danger: x in (8, 12) AND y in (8, 12)   -- obstacle (between goal1 and goal2)

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


def _region(x_lo, x_hi, y_lo, y_hi, label):
    return STLNode.nary_and([
        STLNode.predicate("x", ">", x_lo, signal_index=0, node_id=f"{label}_x_gt_{x_lo}"),
        STLNode.predicate("x", "<", x_hi, signal_index=0, node_id=f"{label}_x_lt_{x_hi}"),
        STLNode.predicate("y", ">", y_lo, signal_index=1, node_id=f"{label}_y_gt_{y_lo}"),
        STLNode.predicate("y", "<", y_hi, signal_index=1, node_id=f"{label}_y_lt_{y_hi}"),
    ], node_id=label)


def build_formula():
    goal1  = _region(*GOAL1,  "goal1")
    goal2  = _region(*GOAL2,  "goal2")
    danger = _region(*DANGER, "danger")

    ev_goal2     = STLNode.eventually_node(goal2, interval=(0, 25), node_id="ev_goal2")
    g1_and_ev_g2 = STLNode.and_node(goal1, ev_goal2, node_id="goal1_and_ev_goal2")
    phi1         = STLNode.eventually_node(g1_and_ev_g2, interval=(0, 25), node_id="phi1")
    phi2         = STLNode.always_node(
        STLNode.not_node(danger, node_id="not_danger"),
        interval=(0, 50), node_id="phi2",
    )
    return STLNode.and_node(phi1, phi2, node_id="Phi")


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
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    Phi = build_formula()
    print(f"Formula: {Phi}\n")
    formula_stlcg = to_stlcgpp(Phi, params={}, device=device, dt=dt)

    traces_list = []

    # Type A: never reach goal1 — wander in neutral zone [6,7.5]
    print("Generating Type A (never reach goal1)...")
    for _ in range(25):
        trace = np.zeros((T, 2))
        bx, by = rng.uniform(6.0, 7.5), rng.uniform(6.0, 7.5)
        noise = rng.standard_normal((T, 2)) * 0.3
        trace[:, 0] = np.clip(bx + noise[:, 0], 5.5, 7.9)
        trace[:, 1] = np.clip(by + noise[:, 1], 5.5, 7.9)
        traces_list.append(trace)

    # Type B: reach goal1, but never reach goal2 — get stuck in middle
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
        traces_list.append(trace)

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
        traces_list.append(trace)

    traces_np = np.stack(traces_list, axis=0)
    traces_t = torch.tensor(traces_np, dtype=torch.float32, device=device)

    print(f"\nTraces shape: {traces_t.shape}")
    print("Computing robustness...")

    with torch.no_grad():
        rob_all = torch.vmap(formula_stlcg)(traces_t)

    rob_cpu = rob_all[:, 0].cpu().numpy()
    n_neg = (rob_cpu < 0).sum()
    print(f"  Min: {rob_cpu.min():.4f}  Max: {rob_cpu.max():.4f}  Negative: {n_neg}/{N}")

    out = args.out_dir
    np.save(f"{out}/counterexamples.npy", traces_np)
    print(f"\nSaved: {out}/counterexamples.npy  (shape {traces_np.shape})")


if __name__ == "__main__":
    main()
