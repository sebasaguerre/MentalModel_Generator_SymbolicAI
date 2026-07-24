"""
thesis_plots.py — Publication-quality transparent PNGs for the thesis.

Outputs (saved in plots_thesis/):
  structure_epoch003.png  – Raw MDP structure at epoch  3  (grid layout)
  structure_epoch005.png  – Raw MDP structure at epoch  5  (grid layout)
  structure_epoch010.png  – Raw MDP structure at epoch 10  (grid layout)
  model_final.png         – Compressed mental model after full training
  macro_mapping.png       – Two-panel: compressed model + GridWorld coloured
                            by macro-state (colours match across panels)
"""

import copy, os, textwrap, random
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from playground import GridWorld
from mentalmodel import SymbolicMM

# ── Output ─────────────────────────────────────────────────────────────────────
PLOT_DIR = "plots_thesis"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Mirrors run_model.py exactly ───────────────────────────────────────────────
TOTAL_EPOCHS    = 350
GRID_SIZE       = 10
N_PITS          = 4
K               = 2
SEED            = 8
ZONE_RADIOUS    = 2
SNAPSHOT_EPOCHS = [1, 3, 5, 10]
plt.rcParams.update({"font.family": "DejaVu Sans"})


# ── Labelling (must match run_model.py) ────────────────────────────────────────
def env_labelling(self, s, action, next_s, reward, done):
    if done:
        self.labels[next_s].add("Goal" if reward > 0 else "TS")
    else:
        self.labels[s].add("NTS")
        self.labels[next_s].add("NTS")


# ── Colour helpers ─────────────────────────────────────────────────────────────
_SEMANTIC = {
    "Goal":      "#0a8c38",   # deep, vivid green
    "TS":        "#c50000",   # vivid red
    "GoalZone":  "#b8ead0",   # pale mint
    "DeathZone": "#f7c8c8",   # pale rose
}
_NTS_COL = "#a8c8e8"


def semantic_color(labels):
    for key, col in _SEMANTIC.items():
        if key in labels:
            return col
    return _NTS_COL


def make_palette(n):
    """n visually distinct RGBA colours, interleaved across three tab-20 maps."""
    raw = []
    for cm_name in ("tab20", "tab20b", "tab20c"):
        cm = plt.colormaps[cm_name]
        raw.extend(cm(i / 20) for i in range(20))
    return (raw[::2] + raw[1::2])[:n]  # separate paired entries for max contrast


# ── Edge iterator (works for both simple & multi-edge relations) ───────────────
def _iter_edges(relations):
    for s, nbrs in relations.items():
        if isinstance(nbrs, dict):
            for _, nxt in nbrs.items():
                for ns in nxt:
                    yield s, ns
        else:
            for ns in nbrs:
                yield s, ns


# ── Snapshot of struct state ───────────────────────────────────────────────────
def take_snapshot(struct):
    return {
        "states":    set(struct.states),
        "relations": copy.deepcopy(dict(struct.relations)),
        "labels":    {s: set(v) for s, v in struct.labels.items()},
    }


# ── Graph layout ───────────────────────────────────────────────────────────────
def _layout(G, min_dist=0.45):
    try:
        return nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pass
    n = G.number_of_nodes()
    pos = nx.spring_layout(
        G,
        k=5.5 / max(n ** 0.38, 1),
        iterations=600,
        seed=42,
        scale=3.0,
    )
    # Post-process: iteratively push overlapping nodes apart
    nodes = list(pos.keys())
    for _ in range(200):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                xa, ya = pos[a];  xb, yb = pos[b]
                dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                if dist < min_dist:
                    moved = True
                    push = (min_dist - dist) / 2 + 0.02
                    if dist < 1e-9:
                        dx, dy = push, push
                    else:
                        dx = (xa - xb) / dist * push
                        dy = (ya - yb) / dist * push
                    pos[a] = (xa + dx, ya + dy)
                    pos[b] = (xb - dx, yb - dy)
        if not moved:
            break
    return pos


def _abst_to_nx(abst):
    G = nx.DiGraph()
    G.add_nodes_from(abst.states)
    seen = set()
    for s, ns in _iter_edges(abst.relations):
        if (s, ns) not in seen:
            seen.add((s, ns))
            G.add_edge(s, ns)
    return G


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Structure snapshots  (grid-layout, transparent)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_structure(snap, epoch, env, ax, show_labels=False):
    g = env.grid_size
    ax.set_facecolor("none")
    ax.set_xlim(-.5, g - .5)
    ax.set_ylim(-.5, g - .5)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # cell backgrounds
    for r in range(g):
        for c in range(g):
            if   (r, c) in env.death_pits: fc, a = "#e74c3c", 0.12
            elif (r, c) == env.goal_pos:   fc, a = "#27ae60", 0.12
            else:                           fc, a = "#f4f6f7", 0.55
            ax.add_patch(plt.Rectangle(
                (c - .5, r - .5), 1, 1, color=fc, alpha=a, zorder=1, lw=0))

    # grid lines
    for i in range(g + 1):
        ax.axhline(i - .5, color="#c8cdd0", lw=0.35, zorder=2)
        ax.axvline(i - .5, color="#c8cdd0", lw=0.35, zorder=2)

    # transitions (one arrow per unique (s, ns) pair, no self-loops)
    seen_pairs = set()
    for s, ns in _iter_edges(snap["relations"]):
        if s != ns and (s, ns) not in seen_pairs:
            seen_pairs.add((s, ns))
            ax.annotate(
                "", xy=(ns[1], ns[0]), xytext=(s[1], s[0]),
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6",
                                lw=0.55, shrinkA=12, shrinkB=12,
                                mutation_scale=9),
                zorder=3)

    # state circles (drawn last so they sit on top of arrows)
    for s in snap["states"]:
        fc = semantic_color(snap["labels"].get(s, set()))
        ax.add_patch(plt.Circle(
            (s[1], s[0]), 0.28, facecolor=fc, zorder=4,
            linewidth=1.1, edgecolor="#2c3e50", alpha=0.95))

    if show_labels:
        for s in snap["states"]:
            lbls = sorted(snap["labels"].get(s, set()))
            if lbls:
                ax.text(s[1], s[0] - 0.38, "\n".join(lbls),
                        ha="center", va="top", fontsize=4.8,
                        color="#2c3e50", style="italic", zorder=6)

    # special markers
    for dp in env.death_pits:
        ax.text(dp[1], dp[0], "✕", ha="center", va="center",
                fontsize=8, color="#922b21", fontweight="bold", zorder=5)
    gr, gc = env.goal_pos
    ax.text(gc, gr, "★", ha="center", va="center",
            fontsize=11, color="#1a5276", zorder=5)

    ax.set_title(
        f"Epoch {epoch}  ·  {len(snap['states'])} / {g*g} states explored",
        fontsize=11, fontweight="bold", color="#2c3e50", pad=6)


def _draw_structure_as_graph(snap, epoch, ax, node_size=480, font_size=6.5,
                             fit_to_states=False, show_labels=False):
    """
    Draw the raw structure as a graph using actual grid coordinates as node positions.
    Grid states have unique (row, col) positions so nodes never overlap.
    fit_to_states=True  — axes tight around discovered states (for standalone plots).
    fit_to_states=False — axes cover the full grid (for side-by-side with env panel).
    """
    G = nx.DiGraph()
    G.add_nodes_from(snap["states"])

    seen = set()
    for s, ns in _iter_edges(snap["relations"]):
        if s != ns and (s, ns) not in seen:   # skip self-loops
            seen.add((s, ns))
            G.add_edge(s, ns)

    # Actual grid coordinates: x = col, y = row — unique per state, no overlap
    pos = {s: (s[1], s[0]) for s in G.nodes()}

    node_colors = [semantic_color(snap["labels"].get(s, set())) for s in G.nodes()]
    border_colors = [
        "#145a32" if "Goal" in snap["labels"].get(s, set()) else
        "#7b241c" if "TS"   in snap["labels"].get(s, set()) else
        "#2c3e50"
        for s in G.nodes()
    ]
    border_widths = [
        2.5 if any(l in snap["labels"].get(s, set()) for l in ("Goal", "TS")) else 1.2
        for s in G.nodes()
    ]
    node_labels = {s: f"{s[0]},{s[1]}" for s in G.nodes()}

    n = G.number_of_nodes()
    ax.set_facecolor("none")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        linewidths=border_widths,
        edgecolors=border_colors,
        alpha=0.92)
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=font_size,
        font_color="#1c2833", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#7f8c8d", alpha=0.50,
        arrows=True, arrowsize=9,
        width=0.85,
        connectionstyle="arc3,rad=0.18",   # arc to separate bi-directional edges
        node_size=node_size,
        min_source_margin=11, min_target_margin=11)

    if show_labels:
        for s, (x, y) in pos.items():
            lbls = sorted(snap["labels"].get(s, set()))
            if lbls:
                ax.text(x, y - 0.38, "\n".join(lbls),
                        ha="center", va="top", fontsize=4.8,
                        color="#2c3e50", style="italic", zorder=6)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if fit_to_states and snap["states"]:
        xs = [s[1] for s in snap["states"]]
        ys = [s[0] for s in snap["states"]]
        pad = 1.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        # no forced equal aspect — let matplotlib pick the best fit
    else:
        ax.set_xlim(-0.8, GRID_SIZE - 0.2)
        ax.set_ylim(-0.8, GRID_SIZE - 0.2)
        ax.set_aspect("equal")

    ax.set_title(
        f"Underlying structure — epoch {epoch}  ·  {n} states  ·  {G.number_of_edges()} transitions",
        fontsize=11, fontweight="bold", color="#2c3e50", pad=6)


def save_structure_snapshots(snapshots, env):
    grid_legend = [
        mpatches.Patch(color="#27ae60", alpha=0.75, label="Goal"),
        mpatches.Patch(color="#e74c3c", alpha=0.50, label="Death pit"),
        mpatches.Patch(color=_NTS_COL,  alpha=0.95, label="Visited state"),
        Line2D([], [], color="#95a5a6", lw=1.1, label="Transition"),
    ]
    graph_legend = [
        mpatches.Patch(color=c, label=l, alpha=0.85)
        for l, c in _SEMANTIC.items()
    ] + [mpatches.Patch(color=_NTS_COL, label="Neutral state")]

    for epoch, snap in snapshots.items():
        # ── Combined: grid exploration + structure graph side by side ──────────
        fig, (ax_grid, ax_graph) = plt.subplots(
            1, 2, figsize=(15, 7),
            gridspec_kw={"wspace": 0.06})
        fig.patch.set_alpha(0.0)

        _draw_structure(snap, epoch, env, ax_grid)
        ax_grid.legend(handles=grid_legend, fontsize=8, loc="upper left",
                       framealpha=0.38, edgecolor="#bdc3c7")

        _draw_structure_as_graph(snap, epoch, ax_graph)
        # no legend on structure graph panel

        path = os.path.join(PLOT_DIR, f"structure_epoch{epoch:03d}.png")
        fig.savefig(path, dpi=180, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

        # ── Standalone: structure graph only (tight around discovered states) ──
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.patch.set_alpha(0.0)
        _draw_structure_as_graph(snap, epoch, ax2, node_size=550, font_size=7.0,
                                 fit_to_states=True)
        # no legend
        path2 = os.path.join(PLOT_DIR, f"structure_graph_epoch{epoch:03d}.png")
        fig2.savefig(path2, dpi=180, transparent=True, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {path2}")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Compressed mental model  (semantic colours, transparent)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_model_graph(G, pos, abst, ax, node_colors, node_size=1100):
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        linewidths=1.8,
        edgecolors="#2c3e50",
        alpha=0.92)

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7.5,
        font_color="#1c2833",
        font_weight="bold")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#7f8c8d",
        alpha=0.55,
        arrows=True,
        arrowsize=13,
        width=1.1,
        connectionstyle="arc3,rad=0.10",
        node_size=node_size,
        min_source_margin=15,
        min_target_margin=15)

    # small label text under each node (abbreviated)
    y_vals = [y for _, y in pos.values()]
    y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1
    offset = y_range * 0.045

    for s, (x, y) in pos.items():
        lbls = sorted(abst.labels.get(s, set()))
        short = [l for l in lbls if l not in ("NTS",)]   # skip uninformative
        if short:
            ax.text(x, y - offset, "\n".join(short),
                    ha="center", va="top",
                    fontsize=5.2, color="#4a4a4a", style="italic",
                    transform=ax.transData)


def save_compressed_model(model, abst, filename="model_final.png"):
    G   = _abst_to_nx(abst)
    pos = _layout(G)

    node_colors = [semantic_color(abst.labels.get(s, set())) for s in G.nodes()]

    fig, ax = plt.subplots(figsize=(20, 13))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    _draw_model_graph(G, pos, abst, ax, node_colors, node_size=1100)

    # only include NTS entry if at least one node carries that colour
    has_nts = any(
        all(lbl not in abst.labels.get(s, set()) for lbl in _SEMANTIC)
        for s in abst.states
    )
    legend = [mpatches.Patch(color=c, label=l, alpha=0.88)
              for l, c in _SEMANTIC.items()]
    if has_nts:
        legend.append(mpatches.Patch(color=_NTS_COL, label="Neutral state/NTS"))
    ax.legend(handles=legend, loc="lower right", framealpha=0.38,
              fontsize=9.5, edgecolor="#bdc3c7", title="State type",
              title_fontsize=9)

    n_str = len(model.struct.states)
    n_abs = len(abst.states)
    ax.set_title(
        f"{n_str} ground states  →  {n_abs} macro-states"
        f"   (k-bisimulation,  k = {K},  zone radius = {ZONE_RADIOUS})",
        fontsize=13, fontweight="bold", color="#1c2833", pad=14)
    ax.axis("off")

    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=180, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Macro-state colour mapping  (model + GridWorld, side-by-side)
# ══════════════════════════════════════════════════════════════════════════════

def save_macro_mapping(model, abst, env, filename="macro_mapping.png"):
    macro_states = sorted(abst.states)
    palette      = make_palette(len(macro_states))
    color_map    = dict(zip(macro_states, palette))
    mapping      = abst.map           # (r,c) -> macro_state name

    G   = _abst_to_nx(abst)
    pos = _layout(G)

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_alpha(0.0)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.04)
    ax_m = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs[1])

    # ── Left: compressed model, unique colour per macro-state ─────────────────
    ax_m.set_facecolor("none")

    node_colors = [color_map[s] for s in G.nodes()]
    # terminal nodes get a stronger border colour
    border_colors = [
        "#145a32" if "Goal" in abst.labels.get(s, set()) else
        "#7b241c" if "TS"   in abst.labels.get(s, set()) else
        "#2c3e50"
        for s in G.nodes()
    ]
    border_widths = [
        3.0 if any(l in abst.labels.get(s, set()) for l in ("Goal", "TS")) else 1.5
        for s in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G, pos, ax=ax_m,
        node_color=node_colors,
        node_size=1200,
        linewidths=border_widths,
        edgecolors=border_colors,
        alpha=0.90)
    nx.draw_networkx_labels(
        G, pos, ax=ax_m,
        font_size=7,
        font_color="#1c2833",
        font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax_m,
        edge_color="#95a5a6",
        alpha=0.50,
        arrows=True,
        arrowsize=11,
        width=0.9,
        connectionstyle="arc3,rad=0.10",
        node_size=1200,
        min_source_margin=15,
        min_target_margin=15)

    ax_m.set_title(
        "Compressed Mental Model\n(unique colour per macro-state)",
        fontsize=12, fontweight="bold", color="#1c2833", pad=10)
    ax_m.axis("off")

    # ── Right: GridWorld cells coloured by macro-state ─────────────────────────
    g = env.grid_size
    ax_g.set_facecolor("none")
    ax_g.set_xlim(-.5, g - .5)
    ax_g.set_ylim(-.5, g - .5)
    ax_g.set_aspect("equal")
    for sp in ax_g.spines.values():
        sp.set_visible(False)
    ax_g.set_xticks(range(g))
    ax_g.set_yticks(range(g))
    ax_g.tick_params(labelsize=7, length=0, colors="#7f8c8d")

    for r in range(g):
        for c in range(g):
            s = (r, c)
            if s in mapping:
                fc, alpha = color_map[mapping[s]], 0.82
            else:
                fc, alpha = "#f0f3f4", 0.45
            ax_g.add_patch(plt.Rectangle(
                (c - .5, r - .5), 1, 1, color=fc, alpha=alpha, zorder=1, lw=0))
            if s in mapping:
                ax_g.text(c, r, mapping[s],
                          ha="center", va="center",
                          fontsize=5.8, color="#1c2833",
                          fontweight="bold", zorder=3)

    # grid lines
    for i in range(g + 1):
        ax_g.axhline(i - .5, color="#bdc3c7", lw=0.55, zorder=2)
        ax_g.axvline(i - .5, color="#bdc3c7", lw=0.55, zorder=2)

    # special markers (drawn after rectangles)
    for dp in env.death_pits:
        ax_g.text(dp[1], dp[0], "✕", ha="center", va="center",
                  fontsize=12, color="#641e16", fontweight="bold", zorder=4)
    gr, gc = env.goal_pos
    ax_g.text(gc, gr, "★", ha="center", va="center",
              fontsize=14, color="#1a5276", zorder=4)

    ax_g.set_title(
        f"{g}×{g} GridWorld  —  cells coloured by macro-state",
        fontsize=12, fontweight="bold", color="#1c2833", pad=10)

    # shared legend: one entry per macro-state (small swatches)
    n = len(macro_states)
    if n <= 30:
        handles = [mpatches.Patch(color=color_map[ms], label=ms, alpha=0.85)
                   for ms in macro_states]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(n, 15), fontsize=6.5,
                   framealpha=0.35, edgecolor="#bdc3c7",
                   title="Macro-states", title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.01))

    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=180, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Entailment witnesses  (formula text + highlighted model, side-by-side)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_formula_lines(formula):
    """Split a top-level conjunction into one display line per conjunct."""
    parts = [repr(c) for c in formula.conjuncts] if hasattr(formula, "conjuncts") else [repr(formula)]
    lines = []
    for i, p in enumerate(parts):
        prefix = "    " if i == 0 else "∧   "
        wrapped = textwrap.wrap(p, width=62) or [p]
        lines.append(prefix + wrapped[0])
        for cont in wrapped[1:]:
            lines.append("        " + cont)
    return lines


def save_entailment_plot(model, abst, filename="entailment_plot.png", k=2, seed=42):
    """
    Two-panel figure:
      Left  — entailment statements  SMM, s ⊨ φ  for two chosen macro-states
      Right — compressed mental model with those states highlighted
    """
    from context_gen import Extractor, Generator

    # Choose 2 non-terminal states (have outgoing edges) for interesting formulas
    candidates = sorted(s for s in abst.states if abst.relations.get(s))
    if len(candidates) < 2:
        candidates = sorted(abst.states)
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(2, len(candidates)))

    extractor = Extractor(abst)
    generator = Generator(button_up=True)
    formulas = {s: generator.generate_formula(extractor.extract_labels(s, k), k)
                for s in chosen}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 13))
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.65], wspace=0.04)
    ax_text  = fig.add_subplot(gs[0])
    ax_model = fig.add_subplot(gs[1])

    # ── Left: entailment text ─────────────────────────────────────────────────
    ax_text.set_facecolor("none")
    ax_text.axis("off")

    y = 0.97
    ax_text.text(0.04, y, "Entailment witnesses",
                 fontsize=15, fontweight="bold", color="#1c2833",
                 va="top", transform=ax_text.transAxes)
    y -= 0.07

    LINE_H   = 0.052   # height of one formula line
    HDR_H    = 0.065   # height of the "SMM, s ⊨" header line
    SEP_H    = 0.055   # gap + separator between the two blocks

    for idx, state in enumerate(chosen):
        if idx > 0:
            sep_y = y - 0.01
            ax_text.plot([0.03, 0.97], [sep_y, sep_y],
                         color="#bdc3c7", lw=0.9,
                         transform=ax_text.transAxes)
            y = sep_y - SEP_H

        # Header: SMM, s ⊨
        ax_text.text(0.04, y,
                     f"SMM ,  {state}  ⊨",
                     fontsize=12.5, fontweight="bold", color="#154360",
                     va="top", transform=ax_text.transAxes,
                     fontfamily="DejaVu Sans")
        y -= HDR_H

        # Formula conjuncts, one per line
        for line in _fmt_formula_lines(formulas[state]):
            ax_text.text(0.06, y, line,
                         fontsize=9, color="#2c3e50",
                         va="top", transform=ax_text.transAxes,
                         fontfamily="DejaVu Sans Mono")
            y -= LINE_H

    # ── Right: model graph ────────────────────────────────────────────────────
    G   = _abst_to_nx(abst)
    pos = _layout(G)

    node_colors   = [semantic_color(abst.labels.get(s, set())) for s in G.nodes()]
    border_colors = []
    border_widths = []
    for s in G.nodes():
        if s in chosen:
            border_colors.append("#154360")
            border_widths.append(4.5)
        elif "Goal" in abst.labels.get(s, set()):
            border_colors.append("#145a32")
            border_widths.append(2.5)
        elif "TS" in abst.labels.get(s, set()):
            border_colors.append("#7b241c")
            border_widths.append(2.5)
        else:
            border_colors.append("#2c3e50")
            border_widths.append(1.8)

    ax_model.set_facecolor("none")
    nx.draw_networkx_nodes(G, pos, ax=ax_model,
                           node_color=node_colors, node_size=1100,
                           linewidths=border_widths, edgecolors=border_colors,
                           alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax_model,
                            font_size=7.5, font_color="#1c2833", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax_model,
                           edge_color="#7f8c8d", alpha=0.55,
                           arrows=True, arrowsize=13, width=1.1,
                           connectionstyle="arc3,rad=0.10",
                           node_size=1100,
                           min_source_margin=15, min_target_margin=15)

    # italic label text under each node
    y_vals  = [yy for _, yy in pos.values()]
    y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1
    offset  = y_range * 0.045
    for s, (x, yy) in pos.items():
        short = sorted(l for l in abst.labels.get(s, set()) if l != "NTS")
        if short:
            ax_model.text(x, yy - offset, "\n".join(short),
                          ha="center", va="top",
                          fontsize=5.2, color="#4a4a4a", style="italic")

    # Legend
    has_nts = any(all(lbl not in abst.labels.get(s, set()) for lbl in _SEMANTIC)
                  for s in abst.states)
    legend = [mpatches.Patch(color=c, label=l, alpha=0.88) for l, c in _SEMANTIC.items()]
    if has_nts:
        legend.append(mpatches.Patch(color=_NTS_COL, label="Neutral / NTS"))
    legend.append(Line2D([], [], color="none", marker="o",
                         markeredgecolor="#154360", markeredgewidth=3.0,
                         markersize=10, label="Witness state"))
    ax_model.legend(handles=legend, loc="lower right", framealpha=0.38,
                    fontsize=9.5, edgecolor="#bdc3c7",
                    title="State type", title_fontsize=9)

    n_str = len(model.struct.states)
    n_abs = len(abst.states)
    ax_model.set_title(
        f"{n_str} ground states  →  {n_abs} macro-states"
        f"   (k-bisimulation,  k = {K},  zone radius = {ZONE_RADIOUS})",
        fontsize=12, fontweight="bold", color="#1c2833", pad=12)
    ax_model.axis("off")

    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=180, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    env = GridWorld(GRID_SIZE, N_PITS, seed=SEED)
    model = SymbolicMM(
        num_act=len(env.actions),
        labelling_function=env_labelling,
        complex_labels=True,
        multi_edges=True,
        zone_radious=ZONE_RADIOUS,
    )

    snapshots = {}
    snapshot_set = set(SNAPSHOT_EPOCHS)

    for i in range(1, TOTAL_EPOCHS + 1):
        env.reset()
        episode_xp = []
        while True:
            action = np.random.choice(len(env.actions))
            xp = env.step(action)
            episode_xp.append(xp)
            if xp[-1]:
                break
        model.update_structure(episode_xp)
        if i in snapshot_set:
            snapshots[i] = take_snapshot(model.struct)

    model.generate_model(k=K)
    n_str = len(model.struct.states)
    n_abs = len(model.abst.states)
    print(f"\nTraining done — {n_str} ground states → {n_abs} macro-states\n")

    save_structure_snapshots(snapshots, env)
    save_compressed_model(model, model.abst)
    save_macro_mapping(model, model.abst, env)
    save_entailment_plot(model, model.abst)

    print(f"\nAll plots saved to: {os.path.abspath(PLOT_DIR)}")


def regen_epoch1_plots():
    """Re-save only the epoch-1 structure plots with state labels. Fast — 1 episode."""
    env = GridWorld(GRID_SIZE, N_PITS, seed=SEED)
    model = SymbolicMM(
        num_act=len(env.actions),
        labelling_function=env_labelling,
        complex_labels=True,
        multi_edges=True,
        zone_radious=ZONE_RADIOUS,
    )
    env.reset()
    episode_xp = []
    while True:
        action = np.random.choice(len(env.actions))
        xp = env.step(action)
        episode_xp.append(xp)
        if xp[-1]:
            break
    model.update_structure(episode_xp)
    snap = take_snapshot(model.struct)

    grid_legend = [
        mpatches.Patch(color="#27ae60", alpha=0.75, label="Goal"),
        mpatches.Patch(color="#e74c3c", alpha=0.50, label="Death pit"),
        mpatches.Patch(color=_NTS_COL,  alpha=0.95, label="Visited state"),
        Line2D([], [], color="#95a5a6", lw=1.1, label="Transition"),
    ]

    # Combined: grid exploration + structure graph
    fig, (ax_grid, ax_graph) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"wspace": 0.06})
    fig.patch.set_alpha(0.0)
    _draw_structure(snap, 1, env, ax_grid, show_labels=True)
    ax_grid.legend(handles=grid_legend, fontsize=8, loc="upper left",
                   framealpha=0.38, edgecolor="#bdc3c7")
    _draw_structure_as_graph(snap, 1, ax_graph, show_labels=True)
    path = os.path.join(PLOT_DIR, "structure_epoch001.png")
    fig.savefig(path, dpi=180, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Standalone structure graph
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.patch.set_alpha(0.0)
    _draw_structure_as_graph(snap, 1, ax2, node_size=550, font_size=7.0,
                             fit_to_states=True, show_labels=True)
    path2 = os.path.join(PLOT_DIR, "structure_graph_epoch001.png")
    fig2.savefig(path2, dpi=180, transparent=True, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()