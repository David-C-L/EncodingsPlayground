#!/usr/bin/env python3
"""Visualize AutoSIS / OracleSIS / OpenZL operator graphs from an
explore_best_encoding JSON export.

The JSON export (written by EncodingsPlayground/Benchmarks/explore_best_encoding.cpp,
schema defined in Source/benchmark/OperatorGraphJson.hpp) captures, for a single
benchmark run:
  - AutoSIS / OracleSIS(random) / OracleSIS(consecutive): a flat partition of the
    64-bit integer into segments, each with a chosen encoding, sample-byte cost,
    and the full ranked list of alternative encodings considered for that
    bit-range (from the exhaustive EncodingGrid search).
  - OpenZL: the true stream/codec DAG reconstructed via ZL_ReflectionCtx, with
    per-stream byte sizes and each stream's share of the total compressed size.
  - Per-segment OpenZL (OracleSIS plans only): OpenZL applied directly to each
    OracleSIS split's own bit-range data, so a segment's chosen SubIntSplit
    encoding can be compared head-to-head against what OpenZL alone achieves
    on that exact same data.

Each graph (AutoSIS track, OracleSIS-random track, OracleSIS-consecutive track,
OpenZL DAG, OpenZL linearised pipeline, per-segment OracleSIS-vs-OpenZL bar
charts, one linearised OpenZL graph per OracleSIS segment, summary bars) is
drawn in its own figure/window so it can be inspected at full resolution,
rather than crammed into shared subplots. The full-dataset OpenZL graph is
rendered twice: once preserving its real branching structure (openzl_graph),
and once flattened into a single left-to-right chain (openzl_graph_linear)
for a simpler "pipeline stage by stage" reading. Both include a stage-role
key and per-node input/output byte labels.

Usage examples:
    # Interactive view: one window per graph, with hover tooltips
    python plot_operator_graph.py --json operator_graphs/twitter_snowflake.json

    # Save one PNG per graph into a directory instead of showing windows
    python plot_operator_graph.py --json operator_graphs/twitter_snowflake.json \
        --outdir operator_graph_pngs/

    # Use a Graphviz "dot" layout for the OpenZL DAG instead of the default
    # layered layout (dot requires pygraphviz + Graphviz installed; falls
    # back to the layered layout if unavailable)
    python plot_operator_graph.py --json operator_graphs/twitter_snowflake.json --layout dot

Dependencies: matplotlib, numpy, networkx, mplcursors (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch, Rectangle

try:
    import mplcursors
except ImportError:  # pragma: no cover
    mplcursors = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_export(json_path: Path) -> dict[str, Any]:
    with json_path.open("r") as f:
        data = json.load(f)
    if data.get("schemaVersion") != 1:
        print(f"Warning: unexpected schemaVersion {data.get('schemaVersion')!r} "
              f"(expected 1) - fields may not match what this script expects.")
    return data


# ---------------------------------------------------------------------------
# Shared coloring, so the same encoding renders identically across the
# AutoSIS / OracleSIS(random) / OracleSIS(consecutive) tracks.
# ---------------------------------------------------------------------------

def collect_all_encoding_names(export: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for plan_key in ("autoSis", "oracleRandom", "oracleConsecutive", "oracleMerged"):
        for seg in export["bitRangePlans"].get(plan_key, {}).get("segments", []):
            names.append(seg["encoding"])
            for alt in seg["alternatives"]:
                names.append(alt["encoding"])
    return names


def encoding_color_map(names: list[str]) -> dict[str, tuple]:
    unique = sorted(set(names))
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(unique)}


# ---------------------------------------------------------------------------
# Bit-range tracks (AutoSIS / OracleSIS)
# ---------------------------------------------------------------------------

def plot_bit_range_track(
    ax: plt.Axes,
    plan: dict[str, Any],
    color_map: dict[str, tuple],
    title: str,
) -> list[tuple[Rectangle, dict[str, Any]]]:
    """Draws one horizontal track of segments spanning bits [0, 63]. Each
    segment boundary gets its own tick + vertical guide line labeled with the
    exact bit index, since boundaries differ per plan and aren't just multiples
    of a fixed step. Returns a list of (patch, segment_dict) pairs for
    hover-tooltip wiring."""
    segs_sorted = sorted(plan["segments"], key=lambda s: s["bitStart"])

    artists: list[tuple[Rectangle, dict[str, Any]]] = []
    for seg in segs_sorted:
        color = color_map.get(seg["encoding"], (0.6, 0.6, 0.6, 1.0))
        rect = Rectangle(
            (seg["bitStart"], 0.0), seg["width"], 1.0,
            facecolor=color, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
        label = seg["encoding"].replace("Encoding", "")
        range_label = f"[{seg['bitStart']}..{seg['bitEnd']}]"
        if seg["width"] >= 3:
            ax.text(
                seg["bitStart"] + seg["width"] / 2.0, 0.62, label,
                ha="center", va="center", fontsize=7,
                rotation=90 if seg["width"] < 8 else 0, clip_on=True,
            )
            ax.text(
                seg["bitStart"] + seg["width"] / 2.0, 0.30, range_label,
                ha="center", va="center", fontsize=6, color="black",
                rotation=90 if seg["width"] < 8 else 0, clip_on=True,
            )
        if seg.get("blockFpeStats") is not None:
            ax.plot(seg["bitStart"] + seg["width"] / 2.0, 1.07, marker="v",
                     color="black", markersize=5, clip_on=False)
        artists.append((rect, seg))

    # Explicit boundary ticks: every segment's bitStart, plus the final bitEnd+1.
    boundaries = [s["bitStart"] for s in segs_sorted]
    if segs_sorted:
        boundaries.append(segs_sorted[-1]["bitEnd"] + 1)
    for b in boundaries:
        ax.axvline(b, color="black", linewidth=0.8, alpha=0.7, ymin=0, ymax=1)

    ax.set_xlim(0, 64)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks(boundaries)
    ax.set_xticklabels([str(b) for b in boundaries], fontsize=8,
                        rotation=90 if len(boundaries) > 16 else 0)
    ax.set_xlabel("bit index (0 = LSB)")
    ax.set_title(title, fontsize=11, loc="left")

    total_cost = plan.get("totalCost")
    if total_cost is not None:
        ax.text(1.0, 1.08, f"totalCost≈{total_cost:,.0f} bits",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    # Color -> encoding key, restricted to encodings actually used in this plan
    # (color_map itself is shared/stable across all three tracks).
    used_encodings = sorted({seg["encoding"] for seg in segs_sorted})
    handles = [
        Patch(facecolor=color_map.get(enc, (0.6, 0.6, 0.6, 1.0)), edgecolor="black",
              label=enc.replace("Encoding", ""))
        for enc in used_encodings
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32),
        ncol=min(len(handles), 5), fontsize=7, frameon=False, title="Encoding",
        title_fontsize=8,
    )
    return artists


# ---------------------------------------------------------------------------
# OpenZL operator graph
# ---------------------------------------------------------------------------

# Synthetic node ids for the explicit source/sink added by build_openzl_digraph
# below -- unlikely to collide with a real OpenZL codec name.
SOURCE_NODE = "__openzl_source__"
SINK_NODE = "__openzl_sink__"


def build_openzl_digraph(openzl: dict[str, Any], include_io: bool = True) -> nx.DiGraph:
    """Aggregates the per-invocation codec/stream DAG (which can have
    thousands of instances — e.g. huffman_v2 invoked once per block) down to
    one node per unique codec *name*, mirroring the aggregation
    printOpenZLAnalysis already performs for the C++ stdout table. Full
    per-instance detail (byte sizes, individual stream ids) remains available
    in the raw JSON for anyone who wants it; this aggregation is purely for
    keeping the rendered graph small and readable (and avoids needing scipy
    for large-graph layouts, which networkx requires above ~500 nodes).

    include_io=True (default) also inserts two synthetic nodes -- SOURCE_NODE
    ("Original Input", the frame's uncompressed source stream(s)) and
    SINK_NODE ("Final Output", the compressed bytes actually written) -- each
    wired to every codec that consumes/produces a stream with no producer/
    consumer codec of its own (`kind="io"` on these two nodes, vs. `"codec"`
    on the rest, so callers can draw them differently). This gives every
    rendered graph an explicit, unambiguous start and end instead of leaving
    the reader to infer them from whichever codec happens to sit at an edge
    of the DAG."""
    codec_by_id = {c["id"]: c for c in openzl["codecs"]}
    stream_by_id = {s["id"]: s for s in openzl["streams"]}

    agg: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for c in openzl["codecs"]:
        name = c["name"]
        if name not in agg:
            agg[name] = {
                "kind": "codec", "name": name, "count": 0, "isStandard": c["isStandard"],
                "totalIn": 0, "totalOut": 0, "totalHeader": 0,
            }
            order.append(name)
        a = agg[name]
        a["count"] += 1
        a["totalHeader"] += c["headerSize"]
        a["totalIn"] += sum(stream_by_id[sid]["contentSize"] for sid in c["inputStreamIds"])
        a["totalOut"] += sum(stream_by_id[sid]["contentSize"] for sid in c["outputStreamIds"])

    g = nx.DiGraph()
    for name in order:
        g.add_node(name, **agg[name])

    edge_weight: dict[tuple[str, str], int] = {}
    for s in openzl["streams"]:
        pid, cid = s["producerCodecId"], s["consumerCodecId"]
        if pid is None or cid is None:
            continue
        key = (codec_by_id[pid]["name"], codec_by_id[cid]["name"])
        edge_weight[key] = edge_weight.get(key, 0) + 1
    for (src, dst), w in edge_weight.items():
        g.add_edge(src, dst, weight=w)

    if include_io:
        total_in, total_out = _graph_io_totals(openzl)
        g.add_node(SOURCE_NODE, kind="io", name="Original Input",
                   totalIn=0, totalOut=total_in, count=1, isStandard=True, totalHeader=0)
        g.add_node(SINK_NODE, kind="io", name="Final Output",
                   totalIn=total_out, totalOut=0, count=1, isStandard=True, totalHeader=0)
        for s in openzl["streams"]:
            if s["producerCodecId"] is None and s["consumerCodecId"] is not None:
                g.add_edge(SOURCE_NODE, codec_by_id[s["consumerCodecId"]]["name"])
            if s["consumerCodecId"] is None and s["producerCodecId"] is not None:
                g.add_edge(codec_by_id[s["producerCodecId"]]["name"], SINK_NODE)

    _split_zstd_node(g, openzl)

    return g


def _split_zstd_node(g: nx.DiGraph, openzl: dict[str, Any]) -> None:
    """zstd is a single opaque leaf as far as OpenZL's own model goes (one
    ZSTD_compress2 call) -- there's no real sub-DAG to reconstruct for what it
    does internally. explore_best_encoding.cpp's OpenZLGraphJson::
    zstdLzOnlyBytes carries an ESTIMATE obtained from zstd's OWN real
    behavior (not a substitute algorithm): what this exact zstd call would
    have produced with literals forced to be stored raw instead of Huffman-
    coded, at the same compression level (see zstdCompressLiteralsRaw in the
    C++). That splits zstd's real work into two genuine, separately-
    measured stages -- LZ77 match-finding (+ FSE-coded sequences) and Huffman
    coding of literals -- so this replaces the single "zstd" node in `g` with
    two nodes wired in series, preserving all of zstd's real predecessor/
    successor edges. Mutates `g` in place; a no-op if there's no zstd node or
    no estimate available for it."""
    if "zstd" not in g.nodes or openzl.get("zstdLzOnlyBytes") is None:
        return

    zstd_attrs = g.nodes["zstd"]
    lz_only_bytes = openzl["zstdLzOnlyBytes"]
    predecessors = list(g.predecessors("zstd"))
    successors = list(g.successors("zstd"))

    lz_id, huffman_id = "zstd (LZ matching, est.)", "zstd (Huffman entropy, est.)"
    g.add_node(
        lz_id, kind="codec", name=lz_id, count=zstd_attrs["count"],
        isStandard=zstd_attrs["isStandard"], totalIn=zstd_attrs["totalIn"],
        totalOut=lz_only_bytes, totalHeader=0, estimated=True,
    )
    g.add_node(
        huffman_id, kind="codec", name=huffman_id, count=zstd_attrs["count"],
        isStandard=zstd_attrs["isStandard"], totalIn=lz_only_bytes,
        totalOut=zstd_attrs["totalOut"], totalHeader=zstd_attrs["totalHeader"], estimated=True,
    )
    g.add_edge(lz_id, huffman_id)
    for p in predecessors:
        g.add_edge(p, lz_id, **g.edges[p, "zstd"])
    for s in successors:
        g.add_edge(huffman_id, s, **g.edges["zstd", s])

    g.remove_node("zstd")


def _topological_layers(g: nx.DiGraph) -> dict[str, int]:
    """Assigns each node a layer index consistent with edge direction
    (producer codec -> consumer codec, which empirically matches OpenZL's
    real *encode*-time execution order — see plot_openzl_linear's docstring).

    The aggregated codec graph can have cycles: collapsing per-invocation
    codecs down to one node per codec *name* (see build_openzl_digraph) means
    a name genuinely invoked at two different real depths (e.g. a shared
    conversion codec reused on two branches) can end up with edges pointing
    both "forward" and "backward" between the same two names, even though no
    single real invocation is ever part of an actual cycle.

    An earlier version handled this via nx.condensation(), collapsing each
    cycle into one node before laying out generations — correct, but it
    forces every node in a cycle onto the exact same layer, which then
    renders as edges running sideways within the (now top-to-bottom) layout
    below, looking like unrelated crossings instead of a top-to-bottom flow.
    Instead, this greedily removes the lowest-occurrence-count edge of each
    cycle (a minimum-feedback-arc-set heuristic) from a working copy until it
    is acyclic, then layers THAT with a plain topological sort. Removed edges
    are skipped only for this layering computation — callers still draw every
    real edge from the original graph, so a removed edge simply renders as a
    (rare) upward-pointing arrow instead of a same-layer sideways one,
    honestly showing it as the aggregation artifact it is rather than hiding
    it or letting it look like a stray horizontal crossing."""
    acyclic = g.copy()
    while True:
        try:
            cycle_edges = nx.find_cycle(acyclic)
        except nx.NetworkXNoCycle:
            break
        u, v = min(
            ((a, b) for a, b, *_ in cycle_edges),
            key=lambda e: acyclic.edges[e].get("weight", 1),
        )
        acyclic.remove_edge(u, v)

    layer_of: dict[str, int] = {}
    for layer, gen in enumerate(nx.topological_generations(acyclic)):
        for node in gen:
            layer_of[node] = layer
    return layer_of


def _layered_positions(
    g: nx.DiGraph, layer_of: dict[str, int], vertical: bool = True,
) -> dict[str, tuple[float, float]]:
    """Layered layout: nodes are placed by topological layer along one axis,
    spread within a layer along the other. vertical=True (default) puts the
    layer index on the y-axis, negated so layer 0 (source) sits at the top
    and the last layer (sink) at the bottom, with same-layer nodes spread
    horizontally -- reads like a standard top-to-bottom dataflow diagram and
    makes the DAG's start/end unambiguous by position alone, before the
    explicit source/sink node shapes are even drawn. vertical=False keeps the
    original left-to-right orientation (layer -> x), used by the linearised
    pipeline view where layer already maps 1:1 to horizontal draw order.

    Takes a precomputed layer_of (from _topological_layers, possibly after
    _duplicate_backward_targets has added duplicate nodes) rather than
    computing it itself, so callers that already needed the layer dict for
    duplication don't pay for it twice."""
    by_layer: dict[int, list[str]] = {}
    for node, layer in layer_of.items():
        by_layer.setdefault(layer, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes_in_layer in by_layer.items():
        nodes_in_layer.sort()
        n = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            spread = float(i) - (n - 1) / 2.0
            pos[node] = (spread, -float(layer)) if vertical else (float(layer), spread)
    return pos


def _linear_order(g: nx.DiGraph, layer_of: dict[str, int]) -> list[str]:
    """A single sequential ordering of all nodes consistent with their
    topological layer (ties broken alphabetically), for the linearised
    pipeline view. Takes a precomputed layer_of, same reasoning as
    _layered_positions above."""
    by_layer: dict[int, list[str]] = {}
    for node, layer in layer_of.items():
        by_layer.setdefault(layer, []).append(node)
    order: list[str] = []
    for layer in sorted(by_layer):
        order.extend(sorted(by_layer[layer]))
    return order


def _duplicate_backward_targets(
    g: nx.DiGraph, layer_of: dict[str, int],
) -> tuple[nx.DiGraph, dict[str, int]]:
    """Every edge u->v with layer_of[u] >= layer_of[v] points backward (or
    sideways) in the layered layout -- these are exactly the aggregation
    artifacts _topological_layers' docstring describes (a codec name reused
    at two real depths, collapsed into one node). Drawing them as-is renders
    a confusing upward-pointing arrow in an otherwise top-to-bottom diagram.

    Instead, for DISPLAY PURPOSES ONLY, redirect each such edge to a fresh
    duplicate copy of its target node, placed exactly one layer below the
    edge's source -- so every drawn edge points strictly downward, at the
    cost of a codec name occasionally appearing more than once in the
    picture (each occurrence keeps the same stats/attributes, tagged with
    `dup_of`/`occurrence` so labeling and tooltips can call out that it's a
    repeat rather than a second, distinct codec). Returns a NEW graph and a
    matching layer dict; the original g/layer_of are untouched, since other
    consumers (role legend, edge-weight counts) want the true, undeduplicated
    structure."""
    g2 = g.copy()
    layer2 = dict(layer_of)
    dup_counts: dict[str, int] = {}

    for u, v in list(g.edges()):
        if layer2[u] < layer2[v]:
            continue
        dup_counts[v] = dup_counts.get(v, 0) + 1
        dup_id = f"{v}__dup{dup_counts[v]}"
        attrs = dict(g2.nodes[v])
        attrs["dup_of"] = v
        attrs["occurrence"] = dup_counts[v] + 1  # the original node is occurrence 1
        g2.add_node(dup_id, **attrs)
        layer2[dup_id] = layer2[u] + 1
        edge_attrs = dict(g2.edges[u, v])
        g2.remove_edge(u, v)
        g2.add_edge(u, dup_id, **edge_attrs)

    return g2, layer2


def _prepare_display_graph(
    g: nx.DiGraph, vertical: bool,
) -> tuple[nx.DiGraph, dict[str, int], dict[str, tuple[float, float]]]:
    """One-stop shop for turning the true aggregated codec graph into what
    actually gets drawn: computes layers, duplicates backward-edge targets
    (see _duplicate_backward_targets) so every drawn edge points the same
    direction, then lays out positions on the resulting (possibly larger)
    graph. Shared by plot_openzl_graph, plot_openzl_linear, and
    render_openzl_graph_figures' figure-size planning pass, so all three
    agree on the exact same node set."""
    layer_of = _topological_layers(g)
    g_draw, layer_draw = _duplicate_backward_targets(g, layer_of)
    pos = _layered_positions(g_draw, layer_draw, vertical=vertical)
    return g_draw, layer_draw, pos


# Keyword -> (short description, category, random-access note), checked in
# order (most specific first) against a space-normalized, lowercased codec
# name. Used to build the "role" key so a reader unfamiliar with OpenZL's
# internal codec names can still follow the pipeline's intent. Heuristic,
# mirrors the substring-matching approach explore_best_encoding.cpp's
# printOpenZLAnalysis already uses for its "Strategy detected" line.
#
# category is "Transform" (pure restructuring/remapping -- reversible,
# doesn't itself rely on match-finding or variable-length codes) or
# "Encoding" (an actual compressor: entropy coding and/or LZ match-finding).
# The random-access note flags whether decoding a SINGLE element at this
# stage needs the whole stream (or an unbounded prefix) rather than O(1)/
# bounded-cost access -- this project's own SubIntSplit/FOR encoders exist
# specifically because of this distinction (see e.g. the bounded-frame PREV
# reference policy work: plain whole-stream delta breaks access, but a
# bounded frame keeps it to O(frameSize)). "delta" here is OpenZL's own
# unbounded whole-stream delta node, so it's marked as breaking access.
_CODEC_ROLE_RULES: list[tuple[str, str, str, str]] = [
    ("zstd (lz matching", "Estimated: zstd's own LZ77 match-finding + FSE-coded "
     "sequences (literals forced uncompressed to isolate this stage)",
     "Encoding", "breaks random access (LZ back-references)"),
    ("zstd (huffman entropy", "Estimated: zstd's own Huffman coding of literals "
     "(gap between literals-raw and the real zstd output)",
     "Encoding", "breaks random access (entropy coding)"),
    ("range pack", "Range-packs values into a smaller numeric range",
     "Transform", "preserves random access"),
    ("field lz", "Field-level LZ match-finding (dedups repeated patterns)",
     "Encoding", "breaks random access (LZ back-references)"),
    ("transpose", "Byte-transposition (struct-of-arrays layout)",
     "Transform", "preserves random access"),
    ("splitn", "Splits one stream into N parallel sub-streams",
     "Transform", "preserves random access"),
    ("quantize offsets", "Quantizes LZ match offsets for entropy coding",
     "Transform", "preserves random access"),
    ("quantize lengths", "Quantizes LZ match lengths for entropy coding",
     "Transform", "preserves random access"),
    ("huffman", "Huffman entropy coding",
     "Encoding", "breaks random access (entropy coding)"),
    ("fse ncount", "FSE normalized-count table (entropy coding metadata)",
     "Encoding", "breaks random access (entropy coding)"),
    ("fse", "FSE / tANS entropy coding",
     "Encoding", "breaks random access (entropy coding)"),
    ("delta", "Delta coding (successive differences)",
     "Transform", "breaks random access (unbounded prefix sum)"),
    ("zstd", "General-purpose Zstandard compression",
     "Encoding", "breaks random access (LZ + entropy coding)"),
    ("convert num to struct", "Type conversion: numeric → fixed-width struct",
     "Transform", "preserves random access"),
    ("convert struct to serial", "Type conversion: struct → serialized bytes",
     "Transform", "preserves random access"),
    ("convert serial to num", "Type conversion: serialized bytes → numeric",
     "Transform", "preserves random access"),
    ("convert num to serial", "Type conversion: numeric → serialized bytes",
     "Transform", "preserves random access"),
    ("convert", "Type conversion (no compression)",
     "Transform", "preserves random access"),
]


def _codec_role(name: str) -> str:
    key = name.lower().replace("_", " ")
    for pattern, role, _category, _note in _CODEC_ROLE_RULES:
        if pattern in key:
            return role
    return "Transform stage (role not recognized)"


def _codec_classification(name: str) -> tuple[str, str]:
    """Returns (category, random-access note) -- see _CODEC_ROLE_RULES."""
    key = name.lower().replace("_", " ")
    for pattern, _role, category, note in _CODEC_ROLE_RULES:
        if pattern in key:
            return category, note
    return "Transform", "random-access impact unknown"


def _codec_classification_short(name: str) -> tuple[str, str]:
    """Compact (category, RA badge) for on-node text, where the full
    sentence from _codec_classification would be too wide/tall to fit
    without crowding neighbors -- the full note is still available in the
    hover tooltip and the bottom stage-key legend."""
    category, note = _codec_classification(name)
    ra_badge = "RA ok" if note.startswith("preserves") else "breaks RA"
    return category, ra_badge


def _format_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:,.0f}{unit}" if unit == "B" else f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}GB"


# All codec/io nodes are drawn at this same fixed point-area (see
# _draw_codec_nodes' docstring for why it's no longer data-volume-scaled).
# io nodes (_draw_io_nodes) are drawn slightly larger to stand out as the
# DAG's anchors.
_CODEC_NODE_SIZE = 2400.0
_IO_NODE_SIZE = 2700.0


def _draw_codec_nodes(
    ax: plt.Axes, g: nx.DiGraph, nodelist: list[str], pos: dict[str, tuple[float, float]],
    *, stagger_labels: bool = False,
) -> tuple[Any, list[float]]:
    """Draws codec nodes, all the SAME size (see _CODEC_NODE_SIZE below) and
    colored by compression ratio, labeling each with its short name plus the
    actual in/out byte counts flowing through it — so the data volumes are
    visible directly on the plot, not just in the hover tooltip. Shared by
    both the branching and linearised OpenZL graph views.

    Node size used to scale with bytes processed, but with real graphs
    spanning many orders of magnitude (a handful of input bytes up to tens of
    megabytes), that made all but the single largest node shrink toward the
    minimum size and become hard to read/click -- a uniform size keeps every
    node equally legible; the actual byte volumes are still fully available
    via the in/out text under each node and the hover tooltip.

    stagger_labels=True (used by the linearised view, where every node sits on
    the same y=0 line) alternates each node's name+io-byte text block above/
    below the node by draw order, so adjacent nodes' text doesn't collide;
    stagger_labels=False (used by the branching view, where nodes already
    occupy distinct y positions) draws the name centered on the node as usual.
    Returns (node_collection, sizes)."""
    sizes = [_CODEC_NODE_SIZE] * len(nodelist)
    ratios = [g.nodes[n]["totalOut"] / max(1, g.nodes[n]["totalIn"]) for n in nodelist]
    coll = nx.draw_networkx_nodes(
        g, pos, ax=ax, nodelist=nodelist, node_shape="o",
        node_size=sizes, node_color=ratios, cmap="RdYlGn_r", vmin=0, vmax=1,
        edgecolors="black", linewidths=1.2,
    )

    # Derived from each node's own "name" attribute rather than its node id --
    # duplicate nodes added by _duplicate_backward_targets have ids like
    # "foo__dup1" but carry the ORIGINAL codec's "name" attribute, so this
    # gives them the same base label plus an explicit "(occurrence N)" marker
    # instead of leaking the internal "__dup1" suffix into the picture.
    def _short_label(n: str) -> str:
        base = g.nodes[n]["name"].removeprefix("zl.private.").removeprefix("zl.")
        occurrence = g.nodes[n].get("occurrence", 1)
        return base if occurrence <= 1 else f"{base} (occurrence {occurrence})"

    short_labels = {n: _short_label(n) for n in nodelist}

    def _io_text(n: str) -> str:
        category, ra_badge = _codec_classification_short(g.nodes[n]["name"])
        return (f"in {_format_bytes(g.nodes[n]['totalIn'])}  →  "
                f"out {_format_bytes(g.nodes[n]['totalOut'])}\n"
                f"{category} · {ra_badge}")

    if not stagger_labels:
        nx.draw_networkx_labels(
            g, pos, ax=ax, labels=short_labels, font_size=11, font_weight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.92, pad=2.2, linewidth=0.6),
        )
        for n in nodelist:
            x, y = pos[n]
            ax.annotate(
                _io_text(n), (x, y), xytext=(0, -22), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="black", fontweight="bold",
                clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            )
    else:
        count = len(nodelist)
        # The first/last codec node sits immediately next to the Original
        # Input/Final Output io node in x-position (see build_openzl_digraph:
        # SOURCE has no incoming edges so always sorts first, SINK no
        # outgoing so always sorts last), and _draw_io_nodes always places
        # ITS OWN byte-count text below itself -- so both ends need "up" to
        # avoid landing on top of the io node's text. But a simple 2-periodic
        # alternation can't have BOTH ends be "up" unless count is odd (it's
        # a parity constraint, not a bug to patch at the boundary) -- forcing
        # only the boundary just relocates the inevitable same-side clash to
        # wherever the forced/natural parities happen to collide, which can
        # land right next to the already-crowded Sink end. Instead, alternate
        # from BOTH ends inward ("up","down","up",... from the front and the
        # same from the back); the one unavoidable same-side pair (when count
        # is even) then falls near the MIDDLE of the sequence instead, away
        # from both io nodes and away from whichever end happens to be
        # busiest with converging edges.
        half = (count + 1) // 2
        up_by_index = [False] * count
        for i in range(half):
            up_by_index[i] = (i % 2 == 0)
        for i in range(count - 1, half - 1, -1):
            up_by_index[i] = ((count - 1 - i) % 2 == 0)
        for i, n in enumerate(nodelist):
            x, y = pos[n]
            up = up_by_index[i]
            va = "bottom" if up else "top"
            name_dy = 18 if up else -18
            io_dy = 38 if up else -38
            ax.annotate(
                short_labels[n], (x, y), xytext=(0, name_dy), textcoords="offset points",
                ha="center", va=va, fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.95, pad=2.0),
            )
            ax.annotate(
                _io_text(n), (x, y), xytext=(0, io_dy), textcoords="offset points",
                ha="center", va=va, fontsize=8.5, color="black", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
            )
    return coll, sizes


def _draw_io_nodes(
    ax: plt.Axes, g: nx.DiGraph, nodelist: list[str], pos: dict[str, tuple[float, float]],
) -> tuple[Any, list[float]]:
    """Draws the synthetic source/sink nodes added by build_openzl_digraph
    (SOURCE_NODE/SINK_NODE) as squares, visually distinct from the round,
    compression-ratio-colored codec nodes -- so a reader can tell at a glance
    where the original, uncompressed data enters the graph and where the
    final compressed bytes leave it, independent of whichever codec happens
    to sit at either end of the pipeline. Returns (node_collection, sizes),
    matching _draw_codec_nodes' return shape so callers can combine both into
    one node_size lookup for edge drawing."""
    sizes = [_IO_NODE_SIZE] * len(nodelist)
    colors = ["#AED6F1" if n == SOURCE_NODE else "#F5B7B1" for n in nodelist]
    coll = nx.draw_networkx_nodes(
        g, pos, ax=ax, nodelist=nodelist, node_shape="s",
        node_size=sizes, node_color=colors, edgecolors="black", linewidths=1.6,
    )
    labels = {n: g.nodes[n]["name"] for n in nodelist}
    nx.draw_networkx_labels(
        g, pos, ax=ax, labels=labels, font_size=11, font_weight="bold",
    )
    for n in nodelist:
        x, y = pos[n]
        attrs = g.nodes[n]
        bytes_val = attrs["totalOut"] if n == SOURCE_NODE else attrs["totalIn"]
        ax.annotate(
            _format_bytes(bytes_val), (x, y), xytext=(0, -22), textcoords="offset points",
            ha="center", va="top", fontsize=9, color="black", fontweight="bold", clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )
    return coll, sizes


def _add_role_legend(fig: plt.Figure, g: nx.DiGraph, nodelist: list[str]) -> None:
    """Adds a text key at the bottom of the figure explaining what each stage
    (by its short displayed name) actually does, so the graph is readable
    without prior OpenZL codec-naming knowledge. Keyed off each node's "name"
    attribute (not its id), so a codec duplicated by _duplicate_backward_targets
    to avoid a backward-pointing arrow (see build_openzl_digraph) still gets
    exactly one entry here rather than one per occurrence."""
    short_labels = sorted({
        g.nodes[n]["name"].removeprefix("zl.private.").removeprefix("zl.") for n in nodelist
    })
    if not short_labels:
        return
    width = max(len(lbl) for lbl in short_labels)
    lines = []
    for lbl in short_labels:
        category, ra_note = _codec_classification(lbl)
        lines.append(f"{lbl.ljust(width)}  [{category:<9s} {ra_note:<38s}]  {_codec_role(lbl)}")
    text = "Stage key:\n" + "\n".join(lines)
    fig.text(
        0.01, 0.01, text, fontsize=6.5, va="bottom", ha="left", family="monospace",
        bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round", pad=0.6),
    )


def _add_compression_colorbar(fig: plt.Figure, ax: plt.Axes, coll: Any) -> None:
    cbar = fig.colorbar(coll, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("compression ratio (output bytes / input bytes)\n"
                    "green = compresses well, red = little/no compression", fontsize=7)
    cbar.ax.tick_params(labelsize=7)


def _graph_io_totals(openzl: dict[str, Any]) -> tuple[float, float]:
    """Total bytes flowing into the whole graph (summed over streams with no
    producer codec, i.e. the original source stream(s)) vs. the final
    compressed size — the same "end-to-end ratio" printOpenZLAnalysis prints
    in explore_best_encoding.cpp's stdout output."""
    total_in = sum(s["contentSize"] for s in openzl["streams"] if s["producerCodecId"] is None)
    total_out = float(openzl["compressedBytes"])
    return float(total_in), total_out


def _io_totals_suffix(openzl: dict[str, Any]) -> str:
    total_in, total_out = _graph_io_totals(openzl)
    ratio = (total_out / total_in) if total_in else 0.0
    return (f"Total: in {_format_bytes(total_in)} → out {_format_bytes(total_out)}  "
            f"(ratio {ratio:.3f}x)")


def _fit_view_with_edges(
    ax: plt.Axes, g: nx.DiGraph, pos: dict[str, tuple[float, float]], rad: float,
) -> None:
    """Sets xlim/ylim to fit both node positions and curved edges. Edges are
    drawn via matplotlib FancyArrowPatch (connectionstyle="arc3,rad=..."),
    which respects clip_on and gets silently cut off at the axes boundary —
    unlike nx.draw_networkx_nodes' PathCollection, patches don't participate
    in ax.autoscale(), so limits fit purely from node positions (as this
    function used to do) clip any edge whose arc bows out far enough. arc3's
    control point is offset by `rad * dist` perpendicular to the chord, and a
    quadratic Bezier's max deviation from the chord is half that — so
    `abs(rad) * dist` is a safe (~2x) upper bound on the sagitta."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not xs or not ys:
        return

    max_sag = 0.0
    for u, v in g.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        max_sag = max(max_sag, abs(rad) * dist)

    xpad = (max(xs) - min(xs)) * 0.18 + 0.5 + max_sag
    ypad = (max(ys) - min(ys)) * 0.18 + 0.5 + max_sag
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)


def plot_openzl_graph(
    ax: plt.Axes, fig: plt.Figure, openzl: dict[str, Any], layout: str = "layered"
) -> list[tuple[Any, list[str], nx.DiGraph]]:
    """Draws the aggregated OpenZL codec DAG (see build_openzl_digraph), with
    its 2D branching structure preserved (a stage can have multiple
    predecessors/successors), top-to-bottom (source at top, sink at bottom)
    so the pipeline's depth uses the figure's vertical space rather than being
    squeezed left-to-right, and with the synthetic Original Input / Final
    Output nodes drawn as squares (see _draw_io_nodes) so the DAG's start and
    end are visually unambiguous regardless of which codec sits at either
    edge. Returns a list of (node_collection, nodelist, graph) triples for
    hover-tooltip wiring — each triple corresponds to one PathCollection
    artist returned by nx.draw_networkx_nodes, with nodelist giving the draw
    order so a hovered point's `sel.index` can be mapped back to a node id.

    Under the default "layered" layout, edges that would otherwise point
    backward/sideways (aggregation artifacts -- see _topological_layers) are
    resolved by _duplicate_backward_targets, so every codec name potentially
    appears more than once in the picture; the title's stage count still
    reports the TRUE unique-name count from before duplication."""
    g = build_openzl_digraph(openzl)
    true_codec_count = sum(1 for n in g.nodes() if g.nodes[n].get("kind") == "codec")

    pos = None
    g_render = g
    if layout == "dot":
        try:
            pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        except Exception as e:  # pygraphviz/graphviz not installed, or layout failed
            print(f"--layout dot unavailable ({e}); falling back to layered layout")
    elif layout == "spring":
        k = 2.5 / max(1.0, len(g) ** 0.5)
        pos = nx.spring_layout(g, seed=0, k=k, iterations=300)
    if pos is None:
        g_render, _, pos = _prepare_display_graph(g, vertical=True)
        # Modest horizontal spread within a layer, generous vertical gaps
        # between layers — dense clustering along the pipeline's depth is the
        # main legibility problem this layout is meant to solve.
        pos = {n: (x * 1.6, y * 2.0) for n, (x, y) in pos.items()}

    codec_nodes = [n for n in g_render.nodes() if g_render.nodes[n].get("kind") == "codec"]
    io_nodes = [n for n in g_render.nodes() if g_render.nodes[n].get("kind") == "io"]
    groups: list[tuple[Any, list[str], nx.DiGraph]] = []
    size_by_node: dict[str, float] = {}
    coll = None
    if codec_nodes:
        coll, codec_sizes = _draw_codec_nodes(ax, g_render, codec_nodes, pos)
        size_by_node.update(zip(codec_nodes, codec_sizes))
        if coll is not None:
            groups.append((coll, codec_nodes, g_render))
    if io_nodes:
        io_coll, io_sizes = _draw_io_nodes(ax, g_render, io_nodes, pos)
        size_by_node.update(zip(io_nodes, io_sizes))
        groups.append((io_coll, io_nodes, g_render))

    edge_rad = 0.08
    nx.draw_networkx_edges(
        g_render, pos, ax=ax, arrows=True, arrowsize=10, width=0.8, alpha=0.55,
        node_size=[size_by_node.get(n, 300) for n in g_render.nodes()],
        connectionstyle=f"arc3,rad={edge_rad}",
    )

    ax.set_title(
        f'OpenZL graph: "{openzl["selectedGraph"]}"  '
        f"({true_codec_count} unique codec stages)\n"
        f"{_io_totals_suffix(openzl)}",
        fontsize=10,
    )
    ax.axis("off")

    # Explicitly fit view to node positions AND edge curvature (see
    # _fit_view_with_edges) with generous padding so labels, io-byte subtext,
    # and arcs near the border aren't clipped.
    _fit_view_with_edges(ax, g_render, pos, edge_rad)

    if coll is not None:
        _add_compression_colorbar(fig, ax, coll)
    _add_role_legend(fig, g_render, codec_nodes)

    return groups


def plot_openzl_linear(
    ax: plt.Axes, fig: plt.Figure, openzl: dict[str, Any],
) -> list[tuple[Any, list[str], nx.DiGraph]]:
    """Draws the aggregated OpenZL codec DAG (see build_openzl_digraph) as a
    single left-to-right sequential chain instead of a branching layout — a
    "linearised" view that reads like the printOpenZLAnalysis stdout pipeline
    table (#1, #2, #3, ...), useful when plot_openzl_graph's 2D branching
    structure is more detail than needed.

    Ordering note: nodes are placed by topological layer over the aggregated
    graph's producer→consumer edges (via _linear_order), NOT by the raw
    first-occurrence order in the JSON's `codecs` list. The latter reflects
    ZL_ReflectionCtx's *decode*-order enumeration (reflection works by
    decompressing the frame), which runs in the opposite direction from the
    real *encode*-time pipeline order that printOpenZLAnalysis reports —
    empirically, the producer→consumer edge direction matches encode order,
    so a topological sort over those edges is what actually linearises
    correctly. The synthetic Original Input / Final Output nodes (see
    build_openzl_digraph) sort to the very first/last position by construction
    (source has no incoming edges, sink no outgoing ones), so they act as
    explicit square end-caps on the chain without any special-casing here.

    Same as plot_openzl_graph, backward/sideways edges (aggregation
    artifacts) are resolved via _duplicate_backward_targets before ordering,
    so a codec name can appear more than once in the chain; the title's
    stage count reports the TRUE unique-name count from before duplication."""
    g = build_openzl_digraph(openzl)
    true_codec_count = sum(1 for n in g.nodes() if g.nodes[n].get("kind") == "codec")
    g_render, layer_draw, _ = _prepare_display_graph(g, vertical=False)
    order = _linear_order(g_render, layer_draw)
    # Wide horizontal spacing so adjacent (often large) node circles and their
    # staggered text blocks don't collide.
    pos = {n: (float(i) * 5.0, 0.0) for i, n in enumerate(order)}

    codec_order = [n for n in order if g_render.nodes[n].get("kind") == "codec"]
    io_order = [n for n in order if g_render.nodes[n].get("kind") == "io"]

    coll: Any = None
    size_by_node: dict[str, float] = {}
    groups: list[tuple[Any, list[str], nx.DiGraph]] = []
    if codec_order:
        coll, codec_sizes = _draw_codec_nodes(ax, g_render, codec_order, pos, stagger_labels=True)
        size_by_node.update(zip(codec_order, codec_sizes))
        if coll is not None:
            groups.append((coll, codec_order, g_render))
    if io_order:
        io_coll, io_sizes = _draw_io_nodes(ax, g_render, io_order, pos)
        size_by_node.update(zip(io_order, io_sizes))
        groups.append((io_coll, io_order, g_render))

    edge_rad = 0.2
    nx.draw_networkx_edges(
        g_render, pos, ax=ax, arrows=True, arrowsize=10, width=0.9, alpha=0.5,
        node_size=[size_by_node.get(n, 300) for n in g_render.nodes()],
        connectionstyle=f"arc3,rad={edge_rad}",
    )

    ax.set_title(
        f'OpenZL graph (linearised): "{openzl["selectedGraph"]}"  '
        f"({true_codec_count} unique codec stages, left→right = pipeline order)\n"
        f"{_io_totals_suffix(openzl)}",
        fontsize=10,
    )
    ax.axis("off")

    _fit_view_with_edges(ax, g_render, pos, edge_rad)

    if coll is not None:
        _add_compression_colorbar(fig, ax, coll)
    _add_role_legend(fig, g_render, codec_order)

    return groups


# ---------------------------------------------------------------------------
# OracleSIS segment vs per-segment OpenZL comparison
# ---------------------------------------------------------------------------

def _oracle_bytes_for_segment(s: dict[str, Any]) -> int:
    """The oracle's chosen encoding size to compare against per-segment
    OpenZL: prefer fullDatasetBytes (the same codec re-encoded over the full
    dataset, set alongside openZlBytes/openZlOnOracleBytes) so the comparison
    is apples-to-apples at the same scale; falls back to the sample-scale
    sampleBytes only if fullDatasetBytes wasn't recorded (e.g. OpenZL/codec
    unavailable for that segment)."""
    full = s.get("fullDatasetBytes")
    return full if full is not None else s["sampleBytes"]


def plot_oracle_vs_openzl_segments(ax: plt.Axes, plan: dict[str, Any], plan_label: str) -> None:
    """Grouped bar chart comparing, for each OracleSIS segment, three bytes
    counts — all at full-dataset scale: (1) its chosen SubIntSplit encoding,
    (2) OpenZL applied directly to that same bit-range's raw data, and (3)
    OpenZL applied as a *second* pass on top of the already SubIntSplit-
    encoded bytes (double-compression) — see attachOpenZLToSegments /
    attachOpenZLOnOracleBytes in explore_best_encoding.cpp (populated for
    OracleSIS plans only, not AutoSIS). A star marks whichever of the three
    wins per segment."""
    segs = sorted(
        (s for s in plan["segments"] if s.get("openZlBytes") is not None),
        key=lambda s: s["bitStart"],
    )
    if not segs:
        ax.text(0.5, 0.5, "No per-segment OpenZL data in this export", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    has_double = any(s.get("openZlOnOracleBytes") is not None for s in segs)

    labels = [f"[{s['bitStart']}..{s['bitEnd']}]\n{s['encoding'].replace('Encoding', '')}" for s in segs]
    oracle_bytes = [_oracle_bytes_for_segment(s) for s in segs]
    openzl_bytes = [s["openZlBytes"] for s in segs]
    double_bytes = [s.get("openZlOnOracleBytes") for s in segs]

    x = np.arange(len(segs))
    width = 0.26 if has_double else 0.38
    series: list[tuple[Any, list[int], float]] = []

    b1 = ax.bar(x - width, oracle_bytes, width, label="OracleSIS (chosen encoding)",
                color="#4C72B0", edgecolor="black") if has_double else \
         ax.bar(x - width / 2, oracle_bytes, width, label="OracleSIS (chosen encoding)",
                color="#4C72B0", edgecolor="black")
    series.append((b1, oracle_bytes, -width if has_double else -width / 2))

    b2 = ax.bar(x, openzl_bytes, width, label="OpenZL (direct, same bit range)",
                color="#DD8452", edgecolor="black") if has_double else \
         ax.bar(x + width / 2, openzl_bytes, width, label="OpenZL (direct, same bit range)",
                color="#DD8452", edgecolor="black")
    series.append((b2, openzl_bytes, 0.0 if has_double else width / 2))

    all_vals = list(oracle_bytes) + list(openzl_bytes)
    if has_double:
        # Segments missing this field (OpenZL failed on that segment) get a
        # zero-height placeholder bar so the x-positions stay aligned.
        double_plot_vals = [v if v is not None else 0 for v in double_bytes]
        b3 = ax.bar(x + width, double_plot_vals, width, label="OracleSIS → OpenZL (double-compress)",
                    color="#55A868", edgecolor="black")
        series.append((b3, double_plot_vals, width))
        all_vals += [v for v in double_bytes if v is not None]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("bytes (full dataset)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"OracleSIS vs OpenZL per segment — {plan_label}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    for bars, vals, _ in series:
        for rect, val in zip(bars, vals):
            if val is None:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{val:,}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

    max_val = max(all_vals) if all_vals else 1
    star_dy = max_val * 0.03
    for i, s in enumerate(segs):
        candidates = [(_oracle_bytes_for_segment(s), -width if has_double else -width / 2)]
        candidates.append((s["openZlBytes"], 0.0 if has_double else width / 2))
        if s.get("openZlOnOracleBytes") is not None:
            candidates.append((s["openZlOnOracleBytes"], width))
        best_val, best_dx = min(candidates, key=lambda c: c[0])
        ax.plot(x[i] + best_dx, best_val + star_dy, marker="*", color="darkviolet",
                markersize=11, zorder=5, clip_on=False)


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------

def _segment_openzl_totals(plan: dict[str, Any]) -> tuple[int | None, int | None]:
    """Sums openZlBytes (OpenZL applied directly to each segment's raw data,
    full dataset scale) and openZlOnOracleBytes (OpenZL applied on top of each
    segment's already SubIntSplit-encoded bytes, also full dataset scale)
    across all of a plan's segments. Returns None for either component where
    no segment has that field (e.g. OpenZL unavailable, or an AutoSIS plan,
    which never gets this data)."""
    direct_vals = [s["openZlBytes"] for s in plan["segments"]
                   if s.get("openZlBytes") is not None]
    double_vals = [s["openZlOnOracleBytes"] for s in plan["segments"]
                   if s.get("openZlOnOracleBytes") is not None]
    return (sum(direct_vals) if direct_vals else None,
            sum(double_vals) if double_vals else None)


def plot_summary_panel(ax: plt.Axes, export: dict[str, Any]) -> None:
    summary = export["summary"]
    dataset_size = export["dataset"]["datasetSize"]
    plans = export["bitRangePlans"]

    # (label, bits/elem, raw bytes, kind) — all bars are full-dataset scale
    # (per-segment OpenZL runs over the full dataset, not a sample), so all
    # use the same bpe formula; `kind` only distinguishes how a bar was
    # produced, for the hatching below:
    #   "plain"             a single whole-dataset encode (AutoSIS/Oracle/flat OpenZL)
    #   "whole_plan_double" a plan's own full output fed through OpenZL as ONE
    #                       second pass (tryOpenZLCompressFull in
    #                       explore_best_encoding.cpp) — sees the WHOLE
    #                       already-encoded buffer at once, so can exploit
    #                       cross-segment structure a per-segment pass can't.
    #   "segment_sum"       independently OpenZL-compressing each bit-range
    #                       segment and summing the results (older, per-segment
    #                       attachOpenZLToSegments/attachOpenZLOnOracleBytes
    #                       path) — no cross-segment structure visible to OpenZL.
    bars: list[tuple[str, float, int, str]] = [
        ("AutoSIS", summary["autoSisBpe"], summary["autoSisBytes"], "plain"),
        ("Oracle\n(random)", summary["oracleRandomBpe"], summary["oracleRandomBytes"], "plain"),
    ]
    if summary.get("oracleConsecBpe") is not None:
        bars.append(("Oracle\n(consec)", summary["oracleConsecBpe"], summary["oracleConsecBytes"], "plain"))
    if summary.get("oracleMergedBpe") is not None:
        bars.append(("Oracle\n(merged)", summary["oracleMergedBpe"], summary["oracleMergedBytes"], "plain"))
    if summary.get("openZlBpe") is not None:
        bars.append(("OpenZL\n(full)", summary["openZlBpe"], summary["openZlBytes"], "plain"))

    # Whole-plan double-compression: each plan's own already-encoded output,
    # fed through OpenZL as a single second pass over the combined buffer.
    whole_plan_double_specs = [
        ("AutoSIS→OpenZL\n(plan)", "autoSisThenOpenZlBpe", "autoSisThenOpenZlBytes"),
        ("Oracle(random)→OpenZL\n(plan)", "oracleRandomThenOpenZlBpe", "oracleRandomThenOpenZlBytes"),
        ("Oracle(consec)→OpenZL\n(plan)", "oracleConsecThenOpenZlBpe", "oracleConsecThenOpenZlBytes"),
        ("Oracle(merged)→OpenZL\n(plan)", "oracleMergedThenOpenZlBpe", "oracleMergedThenOpenZlBytes"),
    ]
    for label_text, bpe_key, bytes_key in whole_plan_double_specs:
        if summary.get(bpe_key) is not None:
            bars.append((label_text, summary[bpe_key], summary[bytes_key], "whole_plan_double"))

    def bpe_from_bytes(total_bytes: int) -> float:
        return total_bytes * 8.0 / dataset_size if dataset_size else 0.0

    for plan_key, plan_label in (("oracleRandom", "random"), ("oracleConsecutive", "consec"),
                                 ("oracleMerged", "merged")):
        direct_total, double_total = _segment_openzl_totals(plans[plan_key])
        if direct_total is not None:
            bars.append((f"OpenZL direct\n(segs, {plan_label})",
                         bpe_from_bytes(direct_total), direct_total, "segment_sum"))
        if double_total is not None:
            bars.append((f"Oracle→OpenZL\n(segs, {plan_label})",
                         bpe_from_bytes(double_total), double_total, "segment_sum"))

    labels = [b[0] for b in bars]
    bpe_values = [b[1] for b in bars]
    byte_values = [b[2] for b in bars]
    bar_kinds = [b[3] for b in bars]

    x = np.arange(len(labels))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, max(len(labels), 2)))
    rects = ax.bar(x, bpe_values, color=colors[: len(labels)], edgecolor="black")
    # Hatch by kind: "plain" bars are unhatched; "whole_plan_double" and
    # "segment_sum" bars get distinct hatches so the two different
    # double-compression methodologies aren't visually conflated.
    hatch_by_kind = {"plain": None, "whole_plan_double": "xx", "segment_sum": "//"}
    for rect, kind in zip(rects, bar_kinds):
        hatch = hatch_by_kind.get(kind)
        if hatch:
            rect.set_hatch(hatch)
    ax.set_xticks(x)
    # Rotate once there are enough bars that horizontal labels would overlap
    # (older ~9-bar exports still render unrotated, matching prior behavior).
    rotate = len(labels) > 9
    ax.set_xticklabels(labels, fontsize=7.5, rotation=25 if rotate else 0,
                        ha="right" if rotate else "center")
    ax.set_ylabel("bits / element")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Summary (AutoSIS efficiency={summary['efficiencyPct']:.1f}% of oracle, "
        f"segments matching={summary['segmentsMatching']}/{summary['segmentsTotal']})\n"
        f"(xx = plan's whole output fed through OpenZL as one pass; "
        f"// = per-segment OpenZL results summed independently; "
        f"all bars full-dataset scale)",
        fontsize=9,
    )
    for rect, bytes_val in zip(rects, byte_values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"{bytes_val:,}B", ha="center", va="bottom", fontsize=7, rotation=90)


# ---------------------------------------------------------------------------
# Hover tooltips
# ---------------------------------------------------------------------------

def _format_segment(seg: dict[str, Any]) -> str:
    lines = [
        f"[{seg['bitStart']}..{seg['bitEnd']}] width={seg['width']}",
        f"encoding: {seg['encoding']}",
        f"reorderer: {seg['reorderer']}",
        f"sampleBytes: {seg['sampleBytes']:,}",
    ]
    if seg.get("estimatedCostBits") is not None:
        lines.append(f"estimatedCostBits: {seg['estimatedCostBits']:,.1f}")
    if seg.get("fullDatasetBytes") is not None:
        lines.append(f"fullDatasetBytes (oracle encoding): {seg['fullDatasetBytes']:,}")
    if seg.get("openZlBytes") is not None:
        lines.append(f"openZlBytes (direct, full dataset): {seg['openZlBytes']:,}")
    if seg.get("openZlOnOracleBytes") is not None:
        lines.append(f"openZlOnOracleBytes (double-compress): {seg['openZlOnOracleBytes']:,}")
    bfpe = seg.get("blockFpeStats")
    if bfpe is not None:
        lines.append(
            f"BlockFPE tiers: avgNumTiers={bfpe['avgNumTiers']:.2f} "
            f"avgTagBits={bfpe['avgTagBitWidth']:.2f} "
            f"avgFallback={bfpe['avgFallbackFraction']:.1%} "
            f"(blockSize={bfpe['blockSize']}, numBlocks={bfpe['numBlocks']})"
        )
    lines.append("alternatives considered:")
    for alt in seg["alternatives"][:6]:
        lines.append(f"  #{alt['rank']} {alt['encoding']}: {alt['sampleBytes']:,} B")
    if len(seg["alternatives"]) > 6:
        lines.append(f"  ... and {len(seg['alternatives']) - 6} more")
    return "\n".join(lines)


def _format_openzl_node(attrs: dict[str, Any]) -> str:
    if attrs.get("kind") == "io":
        bytes_val = attrs["totalIn"] or attrs["totalOut"]
        return f"{attrs['name']}\n{bytes_val:,} bytes"
    ratio = attrs["totalOut"] / max(1, attrs["totalIn"])
    category, ra_note = _codec_classification(attrs["name"])
    lines = [
        f"Codec: {attrs['name']}",
        f"{category} -- {ra_note}",
        f"{'standard' if attrs['isStandard'] else 'custom'} codec, "
        f"{attrs['count']:,} invocation(s)",
        f"in={attrs['totalIn']:,} B  out={attrs['totalOut']:,} B  ratio={ratio:.3f}x",
        f"header total={attrs['totalHeader']:,} B",
    ]
    if attrs.get("estimated"):
        lines.append(
            "(estimated -- see _split_zstd_node: zstd is a single opaque "
            "leaf codec, this stage's split is inferred from a separate "
            "libzstd measurement, not reported by OpenZL itself)"
        )
    if attrs.get("occurrence", 1) > 1:
        lines.append(
            f"(occurrence {attrs['occurrence']} -- same codec drawn again here "
            f"to avoid a backward-pointing arrow; see _duplicate_backward_targets)"
        )
    return "\n".join(lines)


def attach_segment_tooltips(bitrange_artists: list[tuple[Rectangle, dict[str, Any]]]) -> Any:
    if not bitrange_artists:
        return None
    rect_lookup = {id(rect): seg for rect, seg in bitrange_artists}
    cursor = mplcursors.cursor([rect for rect, _ in bitrange_artists], hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        sel.annotation.set_text(_format_segment(rect_lookup[id(sel.artist)]))
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.95)

    return cursor


def attach_openzl_tooltips(openzl_node_groups: list[tuple[Any, list[str], nx.DiGraph]]) -> Any:
    if not openzl_node_groups:
        return None
    collection_lookup = {id(coll): (nodelist, g) for coll, nodelist, g in openzl_node_groups}
    cursor = mplcursors.cursor([coll for coll, _, _ in openzl_node_groups], hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        nodelist, g = collection_lookup[id(sel.artist)]
        node_id = nodelist[sel.index]
        sel.annotation.set_text(_format_openzl_node(g.nodes[node_id]))
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.95)

    return cursor


def render_openzl_graph_figures(
    openzl: dict[str, Any],
    name_prefix: str,
    title: str,
    layout: str,
    finish: Any,
    cursors: list[Any],
) -> None:
    """Renders both OpenZL graph views (branching + linearised) for a single
    OpenZLGraphJson payload, saving/showing them under
    f"{name_prefix}_graph" / f"{name_prefix}_graph_linear". Factored out of
    main() so the flat OpenZL baseline and each of the whole-plan
    double-compression passes (AutoSIS/OracleSIS x3 -> OpenZL) can all reuse
    the exact same sizing/rendering logic instead of duplicating it once per
    call site."""
    # fig.tight_layout() doesn't know about the bottom role-key text (it's
    # placed via fig.text, outside the managed subplot area), so reserve
    # margin explicitly instead. Figure size scales with the number of
    # topological layers/max layer width so node labels have physical
    # room — matplotlib auto-fits axes limits to the data, so scaling
    # node *positions* alone (without also growing the figure) would be a
    # no-op for label crowding; only inches-on-paper actually helps.
    # plot_openzl_graph now lays layers out top-to-bottom (see
    # _layered_positions(vertical=True)), so it's num_layers (pipeline depth)
    # that needs vertical room and max_layer_width (fan-out) that needs
    # horizontal room — the opposite of the old left-to-right sizing.
    # Uses the same _prepare_display_graph pass plot_openzl_graph itself runs
    # (including backward-edge node duplication), so the size plan matches
    # the node count actually drawn.
    zl_digraph_preview = build_openzl_digraph(openzl)
    _, layer_of_preview, _ = _prepare_display_graph(zl_digraph_preview, vertical=True)
    num_layers = len(set(layer_of_preview.values())) or 1
    by_layer_preview: dict[int, int] = {}
    for layer in layer_of_preview.values():
        by_layer_preview[layer] = by_layer_preview.get(layer, 0) + 1
    max_layer_width = max(by_layer_preview.values(), default=1)
    # Inches-per-layer/-per-sibling. NOTE: since _fit_view_with_edges auto-
    # fits the axes to whatever data range the nodes occupy, it's THIS figsize
    # (physical inches) -- not the pos-spacing multipliers in plot_openzl_graph
    # -- that actually controls how large fixed-point-size nodes look relative
    # to the canvas: growing pos-spacing alone is close to a no-op under
    # auto-fit (everything just gets zoomed out proportionally), so an earlier
    # pass that grew BOTH ended up with a much bigger canvas but the same (or
    # worse, via the padding term) visual density -- an inflated, mostly-empty
    # image you had to zoom into. These multipliers are deliberately modest.
    fig_h = max(11.0, num_layers * 2.1)
    fig_w = max(12.0, max_layer_width * 2.5)
    # Force a landscape canvas regardless of how deep the pipeline itself
    # wants to make it -- the DAG's own layout stays top-to-bottom (unaffected
    # by this), only the saved image's outer proportions change; the extra
    # width becomes margin/room for the bottom role-key text and left/right
    # padding rather than a tall, narrow strip.
    fig_w = max(fig_w, fig_h * 1.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    zl_groups = plot_openzl_graph(ax, fig, openzl, layout=layout)
    fig.suptitle(title, fontsize=11)
    # Reserve a fixed number of INCHES (not a fixed fraction) for the suptitle
    # + the axes' own two-line title, and for the bottom role-key text --
    # fig_h now grows with pipeline depth, and a fixed fraction would waste
    # ever-more absolute space as it grows, since the title/legend's own
    # physical size doesn't scale with it.
    fig.subplots_adjust(top=1.0 - 1.5 / fig_h, bottom=2.3 / fig_h)
    if mplcursors is not None:
        cursors.append(attach_openzl_tooltips(zl_groups))
    finish(fig, f"{name_prefix}_graph")

    num_stages = len(build_openzl_digraph(openzl))
    fig, ax = plt.subplots(figsize=(max(14.0, num_stages * 4.0), 7.5))
    zl_lin_groups = plot_openzl_linear(ax, fig, openzl)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.88, bottom=0.34)
    if mplcursors is not None:
        cursors.append(attach_openzl_tooltips(zl_lin_groups))
    finish(fig, f"{name_prefix}_graph_linear")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize AutoSIS/OracleSIS/OpenZL operator graphs from an "
                     "explore_best_encoding JSON export. Each graph is its own "
                     "figure/window (or its own PNG with --outdir).")
    parser.add_argument("--json", required=True, type=Path,
                         help="Path to the operator-graph JSON export")
    parser.add_argument("--outdir", type=Path, default=None,
                         help="Directory to save one PNG per graph, instead of "
                              "showing interactive windows")
    parser.add_argument("--layout", choices=("layered", "spring", "dot"), default="layered",
                         help="OpenZL graph layout engine: layered (default, SCC-condensation "
                              "left-to-right flow), spring (force-directed), or dot "
                              "(needs pygraphviz+Graphviz)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for --outdir output")
    args = parser.parse_args()

    export = load_export(args.json)
    color_map = encoding_color_map(collect_all_encoding_names(export))
    plans = export["bitRangePlans"]
    label = export["dataset"]["label"]

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    cursors: list[Any] = []  # keep references alive so hover callbacks stay connected

    def finish(fig: plt.Figure, name: str) -> None:
        if args.outdir:
            out_path = args.outdir / f"{name}.png"
            fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)

    bitrange_specs = [
        ("autosis_bitrange", plans["autoSis"], f"AutoSIS (cost-model) — {label}"),
        ("oraclesis_random_bitrange", plans["oracleRandom"], f"OracleSIS (random sample) — {label}"),
        ("oraclesis_consecutive_bitrange", plans["oracleConsecutive"],
         f"OracleSIS (consecutive sample) — {label}"),
        ("oraclesis_merged_bitrange", plans["oracleMerged"],
         f"OracleSIS (merged sampling) — {label}"),
    ]
    for name, plan, title in bitrange_specs:
        fig, ax = plt.subplots(figsize=(15, 4.2))
        artists = plot_bit_range_track(ax, plan, color_map, title)
        fig.tight_layout()
        if mplcursors is not None:
            cursors.append(attach_segment_tooltips(artists))
        finish(fig, name)

    if export.get("openZl") is not None:
        render_openzl_graph_figures(
            export["openZl"], "openzl", label, args.layout, finish, cursors)
    else:
        print("OpenZL data unavailable in this export — skipping openzl_graph(s).")

    # Whole-plan double-compression graphs: the codec DAG OpenZL chose when
    # compressing each plan's OWN already-encoded full-dataset output as a
    # single buffer (tryOpenZLCompressFull in explore_best_encoding.cpp) --
    # distinct from the per-segment openZlGraph/openZlOnOracleGraph fields
    # rendered further below, which are per-segment, not whole-plan, graphs.
    double_compress_graph_specs = [
        ("autoSisThenOpenZl", "autosis_then_openzl", f"AutoSIS → OpenZL (whole plan) — {label}"),
        ("oracleRandomThenOpenZl", "oraclesis_random_then_openzl",
         f"OracleSIS (random sample) → OpenZL (whole plan) — {label}"),
        ("oracleConsecThenOpenZl", "oraclesis_consecutive_then_openzl",
         f"OracleSIS (consecutive sample) → OpenZL (whole plan) — {label}"),
        ("oracleMergedThenOpenZl", "oraclesis_merged_then_openzl",
         f"OracleSIS (merged sampling) → OpenZL (whole plan) — {label}"),
    ]
    for json_key, name_prefix, title in double_compress_graph_specs:
        graph_data = export.get(json_key)
        if graph_data is None:
            print(f"{json_key} unavailable in this export — skipping {name_prefix}_graph(s).")
            continue
        render_openzl_graph_figures(graph_data, name_prefix, title, args.layout, finish, cursors)

    # Per-segment OpenZL: explore_best_encoding.cpp applies OpenZL two ways to
    # each OracleSIS segment (AutoSIS is not included) — see
    # attachOpenZLToSegments (direct, on the segment's raw data) and
    # attachOpenZLOnOracleBytes (double-compression, on top of the segment's
    # already SubIntSplit-encoded bytes). For each plan with this data
    # present, draw one comparison bar chart plus one linearised OpenZL graph
    # per segment per variant.
    oracle_specs = [
        ("oraclesis_random", plans["oracleRandom"], "OracleSIS (random sample)"),
        ("oraclesis_consecutive", plans["oracleConsecutive"], "OracleSIS (consecutive sample)"),
        ("oraclesis_merged", plans["oracleMerged"], "OracleSIS (merged sampling)"),
    ]
    graph_variants = [
        ("openZlGraph", "direct", "OpenZL applied directly to this segment's raw data"),
        ("openZlOnOracleGraph", "on_oracle_bytes",
         "OpenZL applied on top of this segment's already SubIntSplit-encoded bytes"),
    ]
    for plan_key, plan, plan_label in oracle_specs:
        segs_with_ozl = [s for s in plan["segments"] if s.get("openZlBytes") is not None]
        if not segs_with_ozl:
            continue

        fig, ax = plt.subplots(figsize=(max(10.0, len(segs_with_ozl) * 2.2), 6.0))
        plot_oracle_vs_openzl_segments(ax, plan, f"{plan_label} — {label}")
        fig.tight_layout()
        finish(fig, f"{plan_key}_vs_openzl")

        for seg in sorted(segs_with_ozl, key=lambda s: s["bitStart"]):
            for graph_field, name_suffix, variant_desc in graph_variants:
                seg_graph = seg.get(graph_field)
                if seg_graph is None:
                    continue
                num_stages = len(build_openzl_digraph(seg_graph))
                fig, ax = plt.subplots(figsize=(max(10.0, num_stages * 4.0), 6.0))
                seg_groups = plot_openzl_linear(ax, fig, seg_graph)
                fig.suptitle(
                    f"{label} — {plan_label}: segment [{seg['bitStart']}..{seg['bitEnd']}] "
                    f"({seg['encoding']})\n{variant_desc}",
                    fontsize=10,
                )
                fig.subplots_adjust(top=0.83, bottom=0.34)
                if mplcursors is not None:
                    cursors.append(attach_openzl_tooltips(seg_groups))
                finish(fig, f"openzl_linear_{plan_key}_bits{seg['bitStart']}-{seg['bitEnd']}_{name_suffix}")

    # Bar count now varies with how much double-compression data the export
    # contains (up to ~15 with all whole-plan + per-segment-sum bars present),
    # so size the figure generously rather than assuming the older ~9-bar count.
    fig, ax = plt.subplots(figsize=(18, 7.0))
    plot_summary_panel(ax, export)
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    finish(fig, "summary")

    if mplcursors is None:
        print("mplcursors not installed - hover tooltips disabled (pip install mplcursors)")

    if not args.outdir:
        plt.show()


if __name__ == "__main__":
    main()
