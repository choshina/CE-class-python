from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import graphviz as gv

from ceclass.lattice.phi_graph import PhiGraph


# ── Label formatting helpers ────────────────────────────────────────────────

def _shorten_params(formula: str) -> str:
    """
    Replace long internal parameter IDs with short Unicode subscript symbols.

    Parser names like  alw_0_30____t2  →  t₂
                       ev_0_40____t3   →  t₃
    These appear inside interval brackets: [alw_0_30____t2, alw_0_30____t3]
    → [t₂, t₃]
    """
    _SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

    def _replace(m: re.Match) -> str:
        idx = m.group(1)
        return 't' + idx.translate(_SUB)

    # Match any  <word>____t<digits>  (the internal param ID format)
    return re.sub(r'\w+____t(\d+)', _replace, formula)


def _wrap_label(formula: str, line_width: int = 32) -> str:
    """
    Insert graphviz line-break markers (\\n) so no line exceeds *line_width*
    characters.  Breaks preferentially at binary operators (and / or) or
    after closing parentheses, then falls back to hard wrap.
    """
    if len(formula) <= line_width:
        return formula

    lines: list[str] = []
    remaining = formula

    while len(remaining) > line_width:
        # Try to break at ' and ' or ' or ' within the target window
        window = remaining[:line_width + 8]   # look a bit ahead
        cut = -1
        for sep in (' and ', ' or '):
            pos = window.rfind(sep, 0, line_width + len(sep))
            if pos != -1 and pos > cut:
                cut = pos + len(sep) - 1   # break *after* the keyword

        if cut == -1:
            # Fall back: break at last space in window
            cut = window.rfind(' ', 0, line_width)

        if cut <= 0:
            # Hard break
            cut = line_width

        lines.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        lines.append(remaining)

    return '\\n'.join(lines)


def _format_label(raw: str, line_width: int = 32) -> str:
    """Shorten param IDs then wrap to *line_width* chars per line."""
    return _wrap_label(_shorten_params(raw), line_width)


# ── Main plot function ───────────────────────────────────────────────────────

def plot_lattice(
    graph: PhiGraph,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    line_width: int = 36,
    format: str = 'png',
    dpi: str = '150',
) -> gv.Digraph:
    """
    Render the PhiGraph refinement lattice as a Hasse diagram.

    Nodes are colored by classification status:
      - Green: covered (counterexample found)
      - Red: pruned (inactive, no counterexample)
      - Gray: unexplored (still active)

    Args:
        graph: The PhiGraph to visualize.
        save_path: If given, render to this file path (without extension).
        show: If True, open the rendered image in the default viewer.
        title: Optional title displayed above the diagram.
        line_width: Max characters per line in node labels before wrapping.
        format: Output format ('png', 'pdf', 'svg').
        dpi: Resolution for raster formats.

    Returns:
        The graphviz.Digraph object for further customization.
    """
    import graphviz

    data = graph.to_dict()

    dot = graphviz.Digraph(
        comment='PhiGraph Hasse Diagram',
        engine='dot',
        format=format,
    )
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.8', dpi=dpi)
    if title:
        dot.attr(label=title, labelloc='t', fontsize='14')

    for node in data['nodes']:
        if node['has_results']:
            fillcolor, color, fontcolor = '#C8E6C9', '#2E7D32', '#1B5E20'
        elif not node['active']:
            fillcolor, color, fontcolor = '#FFCDD2', '#C62828', '#B71C1C'
        else:
            fillcolor, color, fontcolor = '#E0E0E0', '#616161', '#212121'

        raw   = node['formula']
        label = _format_label(raw, line_width)

        dot.node(
            str(node['id']),
            label=label,
            tooltip=raw,          # full text on hover (SVG/web)
            shape='box',
            style='filled,rounded',
            fillcolor=fillcolor,
            color=color,
            fontcolor=fontcolor,
            fontsize='10',
            margin='0.12,0.08',   # tighter padding so boxes stay compact
        )

    for src, dst in data['edges']:
        dot.edge(str(src), str(dst))

    # Legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='dashed', color='gray')
        legend.node(
            '_leg_covered', 'Covered', shape='box', style='filled,rounded',
            fillcolor='#C8E6C9', color='#2E7D32', fontcolor='#1B5E20', fontsize='9',
        )
        legend.node(
            '_leg_pruned', 'Pruned', shape='box', style='filled,rounded',
            fillcolor='#FFCDD2', color='#C62828', fontcolor='#B71C1C', fontsize='9',
        )
        legend.node(
            '_leg_active', 'Active', shape='box', style='filled,rounded',
            fillcolor='#E0E0E0', color='#616161', fontcolor='#212121', fontsize='9',
        )
        legend.edge('_leg_covered', '_leg_pruned', style='invis')
        legend.edge('_leg_pruned', '_leg_active', style='invis')

    if save_path:
        dot.render(save_path, cleanup=True)
    if show:
        dot.view()

    return dot
