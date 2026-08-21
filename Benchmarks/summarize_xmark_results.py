#!/usr/bin/env python3
"""Summarize the XMark pre/post-id SubIntSplit sweep into markdown tables.

Reads the per-driver result CSVs (bench_compression, bench_encode,
bench_decode_bulk, bench_decode_range, bench_decode_gather, bench_decode_point,
bench_ablation), restricted to datasets whose name starts with "XMarkPrePost",
and writes one markdown report: one table per (driver, dataset) plus a final
"best codec per metric per dataset" summary. Within each table the best value
among non-Raw, non-`_Prof` codecs is bolded.

Each driver sweeps more axes than fit in a compact table (cache_state, access
pattern, sigma, ...). The per-codec tables collapse those by taking the median
metric value over cache_state=="hot" cells (dropping cold-all rows, which the
raw CSV still has) -- payload/compression-ratio columns are constant across
that collapse since they don't depend on cache state or access pattern, so the
median is exact for them and a representative summary for timing columns.

Also reads bench_openzl_graph's per-step codec-pipeline CSV, if present, and
renders OpenZL's internal codec-DAG selection per dataset -- this is what
OpenZL actually does internally to win so decisively on bench_compression,
and the gap between it and every registered codec's own technique set.
"""

import argparse
from pathlib import Path

import pandas as pd

# metric -> True if larger is better (everything else assumed smaller-is-better)
HIGHER_IS_BETTER = {
    "compression_ratio", "decode_MBps", "decode_Meps", "elem_Meps",
    "sel_elem_Meps", "span_elem_Meps", "encode_Meps", "encode_MBps",
    "useful_MBps", "input_MBps",
}

# driver -> metric columns to report, in table-column order
DRIVER_METRICS = {
    "bench_compression": ["compression_ratio", "payload_bytes"],
    "bench_encode": ["encode_ns", "selection_ns", "compression_ratio"],
    "bench_decode_bulk": ["time_ns", "decode_MBps"],
    "bench_decode_range": ["time_ns", "elem_Meps"],
    "bench_decode_gather": ["time_ns", "sel_elem_Meps"],
    "bench_decode_point": ["ns_per_probe"],
}

AUTOSIS_DEFAULT_SAMPLE_NOTE = (
    "AutoSIS_LSB/AutoSIS_MSB use the registry's *default* 100,000-sample "
    "AutoSIS config. `Benchmarks/drivers/FINDINGS.md` documents this default "
    "collapsing to a degenerate single full-width section on near-unique "
    "high-cardinality ids (measured on Twitter Snowflake: 64.13 bits/element, "
    "worse than Raw) due to an open entropy-cost-model defect at that sample "
    "size. XMark's `pre`/`post` fields are similarly high-cardinality across "
    "16-32M rows, so a compression_ratio for AutoSIS_LSB/MSB at or worse than "
    "Raw's is very likely this same known issue, not a real finding about SIS "
    "on tree ids -- treat the bench_ablation numbers below (10,000-sample "
    "fresh DP runs) as the more trustworthy signal for what SIS actually wants "
    "to do here."
)

AUTOSIS_ACCESS_LATENCY_NOTE = (
    "AutoSIS_LSB/AutoSIS_MSB are excluded from the bench_decode_gather and "
    "bench_decode_point tables below (and only appear in bench_compression/"
    "bench_encode/bench_decode_bulk/bench_decode_range, all of which use few, "
    "large accesses). A gather cell for AutoSIS on this data ran for 3.5+ "
    "hours at 99.9% CPU before being killed: its degenerate single "
    "full-width section (see the note above) reports `FastSkip=true`, but "
    "each individual small range/point access is apparently still very "
    "expensive on this near-unique data -- a real, access-pattern-dependent "
    "performance defect, not merely the known compression-collapse one. "
    "bulk/range access (one or a few large reads) did not trigger it."
)

FOR_VALIDATION_NOTE = (
    "The FOR baseline (registered this session) failed round-trip "
    "validation once, on XMarkPrePostFull at N=200,000, inside "
    "bench_compression (`materializeAll mismatch at row 1`) -- correctly "
    "caught and excluded by `--validate`. The same (encoder, dataset, N) "
    "pair validated successfully afterward in bench_encode, "
    "bench_decode_bulk and bench_decode_range. This looks like a real, "
    "not-yet-diagnosed edge case (possibly nondeterministic) rather than a "
    "consistent break; flagging it here rather than debugging it in this "
    "session."
)

ABLATION_SCOPE_NOTE = (
    "This ablation run only completed one of its four planned "
    "(reorderer, dataset) combinations -- `reorderer=none` on "
    "`XMarkPrePostElements`, full 29-rung `raw_upward_through_ra` curve, "
    "203 rows -- before the session's time budget ran out. `reorderer=bwt512`, "
    "the `XMarkPrePostFull` dataset, and the (cheaper, 2-rung) "
    "`ra_vs_sequential_whole` ladder were not reached and are a natural "
    "follow-up. The completed curve still answers the central question: "
    "cumulatively admitting every codec (Raw up through all RA-capable "
    "codecs, then all sequential ones, 29 rungs) never once dropped "
    "`sis_fast_skip` from `1` (true) -- SubIntSplit's DP never sacrificed "
    "random access on this data even with the entire codec universe "
    "available, while still reaching 6.54x compression at rung 21 "
    "(admitting `CascadingFORPrevFrequencyPartitionEncoding`)."
)


def load(results_dir: Path, driver: str) -> pd.DataFrame | None:
    path = results_dir / f"xmark_{driver}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["dataset"].str.startswith("XMarkPrePost")].copy()
    if "skipped" in df.columns:
        df = df[df["skipped"] != 1]
    df = df[~df["encoding"].str.endswith("_Prof")]
    return df


def collapse(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if "cache_state" in df.columns:
        df = df[df["cache_state"] == "hot"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(group_cols)[metric_cols].median(numeric_only=True)
    counts = df.groupby(group_cols).size().rename("n_cells")
    return agg.join(counts).reset_index()


def fmt(value, metric: str) -> str:
    if pd.isna(value):
        return "n/a"
    if metric in ("payload_bytes",):
        return f"{value:,.0f}"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def markdown_table(df: pd.DataFrame, name_col: str, metric_cols: list[str],
                    extra_cols: list[str] | None = None) -> str:
    extra_cols = extra_cols or []
    header = [name_col] + extra_cols + metric_cols + ["n_cells"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    winners = {}
    for metric in metric_cols:
        eligible = df[df[name_col] != "Raw"]
        if eligible.empty or eligible[metric].isna().all():
            continue
        idx = (eligible[metric].idxmax() if metric in HIGHER_IS_BETTER
               else eligible[metric].idxmin())
        winners[metric] = idx

    for idx, row in df.iterrows():
        cells = [str(row[name_col])] + [str(row[c]) for c in extra_cols]
        for metric in metric_cols:
            val = fmt(row.get(metric), metric) if metric in row else "n/a"
            if winners.get(metric) == idx:
                val = f"**{val}**"
            cells.append(val)
        cells.append(str(int(row["n_cells"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_ablation_tables(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Returns (markdown, best-rows) for the RA-vs-all-codecs ablation."""
    # SubStreamReordererType::None serializes to an empty CSV field, which
    # pandas parses as NaN -- and groupby drops NaN keys, silently emptying
    # every "no reorderer" row. "none" makes it a real group key.
    df = df.copy()
    df["reorderer"] = df["reorderer"].fillna("none")
    group_cols = ["dataset", "ladder", "reorderer", "rung_index", "rung_name"]
    metric_cols = ["allowed_count", "payload_bytes", "compression_ratio",
                    "rel_bytes_vs_rung0", "sis_fast_skip", "admitted_codec"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    collapsed = collapse(df, group_cols, [c for c in metric_cols if c not in
                                           ("admitted_codec",)])
    # admitted_codec / rung_name are constant within a group; carry them along.
    extra = df.groupby(group_cols)[["admitted_codec"]].first().reset_index()
    collapsed = collapsed.merge(extra, on=group_cols, how="left")

    sections = []
    best_rows = []
    for dataset in sorted(collapsed["dataset"].unique()):
        sections.append(f"#### {dataset}\n")
        for ladder in sorted(collapsed["ladder"].unique()):
            sub = collapsed[(collapsed["dataset"] == dataset) & (collapsed["ladder"] == ladder)]
            if sub.empty:
                continue
            sections.append(f"**{ladder}**\n")
            header = ["reorderer", "rung", "admitted_codec", "allowed_count",
                      "payload_bytes", "compression_ratio", "rel_bytes_vs_rung0",
                      "sis_fast_skip"]
            lines = ["| " + " | ".join(header) + " |",
                     "|" + "|".join(["---"] * len(header)) + "|"]
            sub = sub.sort_values(["reorderer", "rung_index"])
            best_idx = sub["compression_ratio"].idxmax() if "compression_ratio" in sub else None
            for idx, row in sub.iterrows():
                bytes_val = fmt(row.get("payload_bytes"), "payload_bytes")
                ratio_val = fmt(row.get("compression_ratio"), "compression_ratio")
                if idx == best_idx:
                    bytes_val, ratio_val = f"**{bytes_val}**", f"**{ratio_val}**"
                lines.append("| " + " | ".join([
                    str(row.get("reorderer", "")), str(row.get("rung_name", "")),
                    str(row.get("admitted_codec", "") or "-"),
                    str(int(row["allowed_count"])) if pd.notna(row.get("allowed_count")) else "n/a",
                    bytes_val, ratio_val,
                    fmt(row.get("rel_bytes_vs_rung0"), "rel_bytes_vs_rung0"),
                    str(row.get("sis_fast_skip", "")),
                ]) + " |")
            sections.append("\n".join(lines) + "\n")
            if best_idx is not None:
                best_rows.append(collapsed.loc[best_idx])
    return "\n".join(sections), pd.DataFrame(best_rows)


# Steps this codebase has no direct equivalent for, keyed by the OpenZL step
# name substring that identifies them -- used to build the "what we lack"
# callout without hand-copying step names per run.
MISSING_TECHNIQUE = {
    "transpose_split": "byte-plane transposition (split each element into its "
        "N byte-planes, treating byte position k across all elements as its "
        "own sub-array) -- no reorderer or section codec in this registry "
        "does this; SubStreamReordererType only offers None/BWT512, both of "
        "which operate on whole elements, never on a byte-plane slice",
    "field_lz": "a field-level LZ dedup pass ahead of the numeric pipeline -- "
        "no registered codec runs LZ-style dedup before its own transform",
    "delta_int": "delta coding applied per byte-plane rather than on the raw "
        "element -- this repo's FrameOfReference/CascadingFOR family delta "
        "the whole element, not a single byte-plane of it",
}


def build_openzl_graph_section(results_dir: Path) -> str:
    path = results_dir / "xmark_bench_openzl_graph.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    df = df[df["dataset"].str.startswith("XMarkPrePost")]
    if df.empty:
        return ""

    lines = ["## bench_openzl_graph: what OpenZL's internal codec-DAG selects\n",
             "Run with `--validate`, `--n 200000`, default level (0). Unlike every "
             "other driver here, OpenZL is not a single registered codec but a "
             "*graph selector* over its own internal transform/entropy-coder "
             "library -- this shows what it actually composed, as a concrete "
             "answer to \"what does OpenZL have that we don't.\"\n"]

    missing = set()
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].sort_values("step_index")
        graph = sub["selected_graph"].iloc[0]
        ratio = sub["compression_ratio"].iloc[0]
        bits = sub["bits_per_element"].iloc[0]
        lines.append(f"### {dataset}\n")
        lines.append(f"Selected graph: `{graph}` -- {bits:.2f} bits/element "
                      f"(ratio {ratio:.4f}x, i.e. {1/ratio:.1f}x compression).\n")
        header = ["step", "codec", "output_bytes", "share_%"]
        rows = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for _, row in sub.iterrows():
            rows.append("| " + " | ".join([
                str(int(row["step_index"])), f"`{row['codec']}`",
                f"{int(row['codec_output_bytes']):,}", f"{row['codec_share_pct']:.1f}",
            ]) + " |")
            for key in MISSING_TECHNIQUE:
                if key in str(row["codec"]):
                    missing.add(key)
        lines.append("\n".join(rows) + "\n")

    if missing:
        lines.append("**Techniques this repo's codec registry has no equivalent for, "
                     "used in the pipeline above:**\n")
        for key in sorted(missing):
            lines.append(f"- **{key}**: {MISSING_TECHNIQUE[key]}\n")
        lines.append(
            "\nThe plain `Zstd` baseline in this sweep applies zstd directly to "
            "the raw interleaved 8-byte-per-element stream and gets only "
            "1.31x-1.32x. OpenZL applies the same zstd *after* byte-plane "
            "transposition and per-plane delta coding and gets 18.8x-22.2x -- "
            "the entropy coder isn't the differentiator, the decorrelating "
            "transform ahead of it is. This is a more general, higher-leverage "
            "gap than the narrow \"Sequence codec\" idea above: a byte-plane "
            "transpose reorderer (offered to SubIntSplit's DP the same way "
            "BWT512 already is) would let *any* downstream section codec see "
            "per-byte-plane structure, not just a dedicated arithmetic-sequence "
            "codec for `pre` specifically.\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent / "results")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "xmark_report.md")
    args = parser.parse_args()

    out_lines = ["# XMark pre/post-id SubIntSplit sweep\n",
                 "Datasets: `XMarkPrePostElements`, `XMarkPrePostFull` "
                 "(level:8|pre:28|post:28 bit-packed tree ids; see "
                 "`Datasets/XMark/README.md`). Best value among non-Raw, "
                 "non-`_Prof` codecs is **bolded** per column.\n",
                 f"> {AUTOSIS_DEFAULT_SAMPLE_NOTE}\n",
                 f"> {AUTOSIS_ACCESS_LATENCY_NOTE}\n",
                 f"> {FOR_VALIDATION_NOTE}\n"]

    all_best = []  # (driver, dataset, metric, encoding, value)

    for driver, metric_cols in DRIVER_METRICS.items():
        df = load(args.results_dir, driver)
        out_lines.append(f"## {driver}\n")
        if df is None or df.empty:
            out_lines.append("_no results_\n")
            continue
        collapsed = collapse(df, ["dataset", "encoding"], metric_cols)
        for dataset in sorted(collapsed["dataset"].unique()):
            sub = collapsed[collapsed["dataset"] == dataset].sort_values("encoding")
            out_lines.append(f"### {dataset}\n")
            out_lines.append(markdown_table(sub, "encoding", metric_cols) + "\n")
            for metric in metric_cols:
                eligible = sub[sub["encoding"] != "Raw"]
                if eligible.empty or eligible[metric].isna().all():
                    continue
                idx = (eligible[metric].idxmax() if metric in HIGHER_IS_BETTER
                       else eligible[metric].idxmin())
                best = eligible.loc[idx]
                all_best.append((driver, dataset, metric, best["encoding"], best[metric]))

    out_lines.append("## bench_ablation: SIS with RA codecs vs SIS with all/most codecs\n")
    abl = load(args.results_dir, "bench_ablation")
    if abl is None or abl.empty:
        out_lines.append("_no results_\n")
    else:
        out_lines.append(f"> {ABLATION_SCOPE_NOTE}\n")
        md, best_ablation_rows = build_ablation_tables(abl)
        out_lines.append(md)
        for _, row in best_ablation_rows.iterrows():
            all_best.append(("bench_ablation", row["dataset"], "compression_ratio",
                             f"{row['ladder']}/{row['rung_name']}", row["compression_ratio"]))

    openzl_section = build_openzl_graph_section(args.results_dir)
    if openzl_section:
        out_lines.append(openzl_section)

    out_lines.append("## Best codec per metric per dataset (summary)\n")
    header = ["driver", "dataset", "metric", "best_codec", "value"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for driver, dataset, metric, codec, value in all_best:
        lines.append("| " + " | ".join([driver, dataset, metric, f"**{codec}**",
                                        fmt(value, metric)]) + " |")
    out_lines.append("\n".join(lines) + "\n")

    out_lines.append("## Suggested follow-up: a \"Sequence\" codec\n")
    out_lines.append(
        "This round only benchmarked the **packed** `prepost_id` (level:8|"
        "pre:28|post:28 bit-packed into one int64), not the unpacked `pre`/"
        "`post`/`level` int32 columns (`Datasets/XMark/level_*.parquet` etc., "
        "generated but not yet registered -- see `DatasetRegistry.hpp`'s "
        "int32Datasets() note) -- so there is no direct 'how well does codec X "
        "compress the raw `pre` column' number yet. What the packed-id sweep "
        "does show: `OpenZL` is the strongest whole-column baseline "
        "(18.8x-22.2x, `bench_compression`), and SubIntSplit's DP -- once "
        "given the full codec universe (`bench_ablation`, `raw_upward_"
        "through_ra`, rung 21) -- reaches **6.54x** via "
        "`CascadingFORPrevFrequencyPartitionEncoding` while keeping "
        "`sis_fast_skip=1` (true) the entire way to rung 28 (all 29 codecs "
        "admitted). Random access was never traded away on this data, at any "
        "point in the ladder.\n\n"
        "That 6.54x is still well short of what's structurally possible for "
        "`pre` alone: by construction `pre` is a permutation of `0..n-1` in "
        "document order, so a maximally-compressed representation is a few "
        "bytes (base + stride), not ~1.6 bits/element. No codec in this "
        "sweep models 'value equals row index' directly -- FOR/delta-style "
        "codecs get close only when the data is *locally* sequential in "
        "physical row order, which `pre` already is by definition, but they "
        "still spend real bits per element on frame references and "
        "residuals rather than recognizing the whole column as one "
        "arithmetic sequence. A dedicated RA-capable **Sequence codec** -- "
        "encode as `base + stride * index` plus a sparse exception list for "
        "any positions that deviate (e.g. `post`, which is *not* globally "
        "sequential but is piecewise-monotonic within a subtree), with O(1) "
        "`decodeAt` via direct arithmetic -- is the natural next comparator, "
        "and this session's numbers are the baseline it would need to beat: "
        "6.54x compression while keeping FastSkip, on the packed id, and "
        "whatever the unpacked `pre` column's own dedicated sweep turns up "
        "as its current best (future work, see above).\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
