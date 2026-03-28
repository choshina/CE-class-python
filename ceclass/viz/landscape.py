from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from ceclass.formula.stl_node import STLNode
from ceclass.formula.converter import to_stlcgpp

if TYPE_CHECKING:
    import matplotlib.figure
    from ceclass.synthesis.param_synth import ParamSynthesis, SynthResult


def _evaluate_objective(
    formula: STLNode,
    traces: torch.Tensor,
    param_names: list[str],
    param_values: list[float],
    device: Optional[torch.device],
    dt: float,
) -> float:
    """Evaluate synthesis objective at a single parameter point.

    Returns -min_robustness (negative means the negated formula is satisfied,
    i.e. the original formula is violated).
    """
    params = dict(zip(param_names, param_values))
    neg_formula = STLNode.negate(formula)
    try:
        stl_formula = to_stlcgpp(neg_formula, params, device, dt)
        with torch.no_grad():
            rob = torch.vmap(stl_formula)(traces)
            min_rob = rob.min().item()
        return -min_rob
    except Exception:
        return float('nan')


def plot_landscape(
    formula: STLNode,
    traces: torch.Tensor,
    param_names: list[str],
    param_bounds: dict[str, tuple[float, float]],
    device: Optional[torch.device] = None,
    dt: float = 1.0,
    grid_resolution: int = 50,
    best_params: Optional[dict[str, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    verbose: bool = False,
) -> matplotlib.figure.Figure:
    """
    Plot robustness landscape over parameter space.

    Evaluates the synthesis objective on a dense grid and produces:
      - 1D: line plot with satisfaction boundary
      - 2D: filled contour plot with zero contour
      - 3+D: 2D slice through first two params (others fixed at best/midpoint)

    Args:
        formula: The STLNode formula (not negated).
        traces: Falsifying traces, shape (num_traces, timesteps, dims).
        param_names: Symbolic parameter names in evaluation order.
        param_bounds: Maps each param name to (lower, upper).
        device: Torch device.
        dt: Timestep duration.
        grid_resolution: Number of grid points per dimension.
        best_params: If given, mark the best point on the plot.
        save_path: If given, save figure to this path.
        show: If True, call plt.show().
        title: Optional plot title.
        verbose: If True, print progress during evaluation.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n_params = len(param_names)

    if n_params == 1:
        fig = _plot_1d(
            formula, traces, param_names, param_bounds,
            device, dt, grid_resolution, best_params, title, verbose,
        )
    elif n_params == 2:
        fig = _plot_2d(
            formula, traces, param_names, param_bounds,
            device, dt, grid_resolution, best_params, title, verbose,
        )
    else:
        fig = _plot_2d_slice(
            formula, traces, param_names, param_bounds,
            device, dt, grid_resolution, best_params, title, verbose,
        )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def _plot_1d(formula, traces, param_names, param_bounds,
             device, dt, grid_resolution, best_params, title, verbose):
    import matplotlib.pyplot as plt

    name = param_names[0]
    lb, ub = param_bounds[name]
    x_vals = np.linspace(lb, ub, grid_resolution)
    obj_vals = np.empty(grid_resolution)

    for i, x in enumerate(x_vals):
        if verbose and i % 10 == 0:
            print(f'\rEvaluating landscape: {i + 1}/{grid_resolution}', end='', flush=True)
        obj_vals[i] = _evaluate_objective(formula, traces, param_names, [x], device, dt)
    if verbose:
        print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, obj_vals, 'b-', linewidth=2, label='objective')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='satisfaction boundary')

    satisfied_mask = obj_vals < 0
    if satisfied_mask.any():
        ax.fill_between(
            x_vals, obj_vals, 0, where=satisfied_mask,
            alpha=0.15, color='green', label='satisfied region',
        )

    if best_params and name in best_params:
        bx = best_params[name]
        by = _evaluate_objective(formula, traces, param_names, [bx], device, dt)
        ax.plot(bx, by, 'r*', markersize=15, zorder=5, label=f'best ({bx:.3f})')

    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel('objective (\u2212min robustness)', fontsize=12)
    ax.set_title(title or f'Robustness Landscape: {name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_2d(formula, traces, param_names, param_bounds,
             device, dt, grid_resolution, best_params, title, verbose,
             fixed_params=None):
    import matplotlib.pyplot as plt

    n1, n2 = param_names[0], param_names[1]
    lb1, ub1 = param_bounds[n1]
    lb2, ub2 = param_bounds[n2]

    x = np.linspace(lb1, ub1, grid_resolution)
    y = np.linspace(lb2, ub2, grid_resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)

    # Build full param names list (first two vary, rest are fixed)
    all_param_names = param_names if fixed_params is None else list(param_names) + list(fixed_params.keys())
    fixed_vals = list(fixed_params.values()) if fixed_params else []

    total = grid_resolution * grid_resolution
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            idx = i * grid_resolution + j
            if verbose and idx % 100 == 0:
                print(f'\rEvaluating landscape: {idx + 1}/{total}', end='', flush=True)
            vals = [X[i, j], Y[i, j]] + fixed_vals
            Z[i, j] = _evaluate_objective(formula, traces, all_param_names, vals, device, dt)
    if verbose:
        print()

    fig, ax = plt.subplots(figsize=(8, 7))

    vmin, vmax = np.nanmin(Z), np.nanmax(Z)
    levels = np.linspace(vmin, vmax, 25)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlGn_r')
    plt.colorbar(cf, ax=ax, label='objective (\u2212min robustness)')

    if vmin < 0 < vmax:
        ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2, linestyles='--')

    if best_params and n1 in best_params and n2 in best_params:
        ax.plot(
            best_params[n1], best_params[n2], 'r*', markersize=15, zorder=5,
            label=f'best ({best_params[n1]:.2f}, {best_params[n2]:.2f})',
        )
        ax.legend()

    ax.set_xlabel(n1, fontsize=12)
    ax.set_ylabel(n2, fontsize=12)
    ax.set_title(title or f'Robustness Landscape: {n1} vs {n2}', fontsize=14)
    fig.tight_layout()
    return fig


def _plot_2d_slice(formula, traces, param_names, param_bounds,
                   device, dt, grid_resolution, best_params, title, verbose):
    """For 3+ params: fix extra params at best/midpoint and plot first two."""
    fixed = {}
    for name in param_names[2:]:
        if best_params and name in best_params:
            fixed[name] = best_params[name]
        else:
            lb, ub = param_bounds[name]
            fixed[name] = (lb + ub) / 2.0

    suffix_parts = [f'{name}={fixed[name]:.2f}' for name in param_names[2:]]
    slice_title = title or f'Robustness Landscape (slice: {", ".join(suffix_parts)})'

    return _plot_2d(
        formula, traces, param_names[:2], param_bounds,
        device, dt, grid_resolution, best_params, slice_title, verbose,
        fixed_params=fixed,
    )


def plot_landscape_from_synth(
    synth: ParamSynthesis,
    result: Optional[SynthResult] = None,
    grid_resolution: int = 50,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    verbose: bool = False,
) -> matplotlib.figure.Figure:
    """Convenience wrapper that extracts parameters from a ParamSynthesis instance.

    Args:
        synth: The ParamSynthesis instance (provides formula, traces, params, etc.).
        result: If given, marks the best point found during synthesis.
        grid_resolution: Number of grid points per dimension.
        save_path: If given, save figure to this path.
        show: If True, call plt.show().
        title: Optional plot title.
        verbose: If True, print progress.

    Returns:
        The matplotlib Figure.
    """
    best_params = result.params_best if result is not None else None
    return plot_landscape(
        formula=synth.formula,
        traces=synth.traces,
        param_names=synth.param_names,
        param_bounds=synth.param_bounds,
        device=synth.device,
        dt=synth.dt,
        grid_resolution=grid_resolution,
        best_params=best_params,
        save_path=save_path,
        show=show,
        title=title,
        verbose=verbose,
    )
