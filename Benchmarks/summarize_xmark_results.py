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


def build_oracle_section(results_dir: Path) -> str:
    """bench_costmodel_oracle: is the SIS-vs-OpenZL gap a selection defect
    (same codecs, wrong pick) or a missing-capability one? Distinct question
    from build_openzl_graph_section above, which answers "what capability is
    missing"; this answers "are we even using the capabilities we have well."
    """
    path = results_dir / "xmark_bench_costmodel_oracle.summary.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    df = df[df["dataset"].str.startswith("XMarkPrePost")]
    if df.empty:
        return ""

    lines = ["## bench_costmodel_oracle: is the gap selection or capability?\n",
             "Run with `--sample-sizes 2000 --min-segment-width 8` (the driver's "
             "full default grid is ~300K sample encodes and impractical here; "
             "`min-segment-width 8` shrinks the candidate grid to ~23K, still "
             "the `default` 7-type candidate set only, not `extended`). Compares "
             "AutoSIS's analytical cost-model ranking against a true byte-count "
             "oracle over the *same* candidate segments -- this isolates "
             "selection accuracy from codec-universe coverage.\n"]

    header = ["dataset", "profile", "top1_accuracy", "spearman_rho",
              "mean_abs_rel_err", "regret_bytes_extrapolated"]
    rows = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in df.sort_values(["dataset", "profile"]).iterrows():
        rows.append("| " + " | ".join([
            row["dataset"], row["profile"],
            f"{row['top1_accuracy']:.0%}", f"{row['spearman_rho']:.3f}",
            f"{row['mean_abs_rel_err']:.1%}", f"{row['regret_bytes_extrapolated']:,.0f}",
        ]) + " |")
    lines.append("\n".join(rows) + "\n")

    lines.append(
        "`top1_accuracy` (47-77%, worst on `random`-profile sampling) is the "
        "cost model's #1-ranked candidate matching the oracle's actual #1 "
        "*less than 4 times out of 5* -- SubIntSplit is regularly not using the "
        "best segment plan even among the codecs it already has. On the "
        "`random` profile, `regret_bytes_extrapolated` (~273KB) is comparable "
        "in size to the *entire* best compressed output this sweep found "
        "(244,547 bytes, bench_ablation rung 21) -- a plausible order-of-"
        "magnitude estimate of how much selection error alone could be costing, "
        "separate from any missing capability. The `consec` profile shows far "
        "less regret (651 bytes / 0 bytes), so the practical impact depends on "
        "which sampling profile AutoSIS's production config actually uses for "
        "this kind of high-cardinality tree-id data.\n\n"
        "This is consistent with, and a plausible root cause of, the "
        "monotonicity anomaly in the ablation table above: `raw_upward_"
        "through_ra` rung 21 (22 codecs admitted) reached 6.54x, but rung 22 "
        "(23 codecs -- strictly *more* options) regressed to 4.49x and stayed "
        "there through rung 28. A DP with a superset of codecs should never do "
        "worse than it did with a subset (it can always reproduce the old plan "
        "by not using the new codec) -- the only way this regresses is if the "
        "newly admitted codec's cost estimate mis-ranks against its true byte "
        "cost, exactly what this oracle comparison measures directly.\n")

    return "\n".join(lines)


def _plan_bits_per_elem(plans_df: pd.DataFrame, dataset: str, plan: str) -> tuple[float, list[str]]:
    """Sum a full plan's segments into overall bits/element, plus a
    human-readable line per segment. autosis rows' est_bits are scaled to the
    full N=200,000; oracle_random/oracle_consec rows' est_bits are scaled to
    the 2,000-element sample they were measured over; oracle_merged has no
    est_bits (has_cost_model=0) and uses measured sample_bytes*8 instead.
    """
    sub = plans_df[(plans_df["dataset"] == dataset) & (plans_df["plan"] == plan)]
    if sub.empty:
        return float("nan"), []
    basis = 200000 if plan == "autosis" else 2000
    lines = []
    total_bits = 0.0
    for _, row in sub.sort_values("bit_start").iterrows():
        bits = row["est_bits"] if pd.notna(row["est_bits"]) else row["sample_bytes"] * 8
        bpe = bits / basis
        total_bits += bpe
        lines.append(f"    bits [{int(row['bit_start']):>2}-{int(row['bit_end']):>2}] "
                     f"(width {int(row['width'])}): `{row['encoding']}` -- {bpe:.2f} bits/elem")
    return total_bits, lines


def build_full_choice_comparison(results_dir: Path) -> str:
    """The plain-language headline comparison the PR asks for: OpenZL's full
    pipeline vs SIS's best Auto choice vs SIS's best Oracle choice, each
    shown as its actual full plan, not just a ratio number.
    """
    openzl_path = results_dir / "xmark_bench_openzl_graph.csv"
    comp_path = results_dir / "xmark_bench_compression.csv"
    sections_path = results_dir / "xmark_bench_compression_sections.csv"
    ablation_path = results_dir / "xmark_bench_ablation.csv"
    oracle_plans_path = results_dir / "xmark_bench_costmodel_oracle.plans.csv"
    if not all(p.exists() for p in (openzl_path, comp_path, sections_path,
                                     ablation_path, oracle_plans_path)):
        return ""

    openzl = pd.read_csv(openzl_path)
    comp = pd.read_csv(comp_path)
    sections = pd.read_csv(sections_path)
    ablation = pd.read_csv(ablation_path)
    oracle_plans = pd.read_csv(oracle_plans_path)

    lines = ["## The gap, plainly: OpenZL vs SIS's best Auto vs SIS's best Oracle\n",
             "Four numbers, each a **real full plan** (not just a ratio), for the "
             "same 1,600,000-byte (200,000-element) column:\n"]

    for dataset in sorted(comp["dataset"].unique()):
        lines.append(f"### {dataset}\n")

        oz = openzl[openzl["dataset"] == dataset]
        oz_bits = oz["bits_per_element"].iloc[0] if not oz.empty else float("nan")
        oz_ratio = 64 / oz_bits if oz_bits else float("nan")
        lines.append(f"**1. OpenZL** (`bench_openzl_graph`, real, validated): "
                     f"**{oz_bits:.2f} bits/elem, {oz_ratio:.1f}x**. Full pipeline: "
                     f"struct-of-bytes -> field LZ dedup -> transpose into 8 byte-planes "
                     f"-> delta-code each plane -> zstd each plane. (Full step table "
                     f"above, in bench_openzl_graph.)\n")

        # SIS as shipped: the registered AutoSIS entry, default 100K-sample config.
        default_row = comp[(comp["dataset"] == dataset) & (comp["encoding"] == "AutoSIS_LSB")]
        sec_row = sections[(sections["dataset"] == dataset) &
                            (sections["encoding"] == "AutoSIS_LSB_Prof")]
        if not default_row.empty and not sec_row.empty:
            bpe = 64 / default_row["compression_ratio"].iloc[0]
            enc = sec_row["section_encoding"].iloc[0]
            lines.append(f"**2. SIS as shipped today** (`AutoSIS_LSB`, registry's default "
                         f"100,000-sample config, real, validated): **{bpe:.2f} bits/elem, "
                         f"{default_row['compression_ratio'].iloc[0]:.3f}x**. Full plan: "
                         f"one section, the whole column, `{enc}` -- BWT-reordered then left "
                         f"**uncompressed** (`Raw`). This is the known FINDINGS.md "
                         f"100K-sample collapse (see the note near the top).\n")

        # SIS's best Auto: the best plan bench_ablation's DP actually found and validated.
        abl = ablation[ablation["dataset"] == dataset]
        if abl.empty:
            lines.append("**3. SIS's best Auto found this session**: _not available -- "
                         "`bench_ablation` only completed the `reorderer=none` combination "
                         "on `XMarkPrePostElements` before this session's time budget ran "
                         "out (see the ablation scope note above); this dataset's ablation "
                         "sweep is a follow-up._\n")
        else:
            best_idx = abl["compression_ratio"].idxmax()
            best = abl.loc[best_idx]
            bpe = 64 / best["compression_ratio"]
            lines.append(f"**3. SIS's best Auto found this session** "
                         f"(`bench_ablation`, `{best['ladder']}` rung `{best['rung_name']}`, "
                         f"real, validated, {int(best['allowed_count'])}-codec candidate set): "
                         f"**{bpe:.2f} bits/elem, {best['compression_ratio']:.2f}x**, "
                         f"`FastSkip` kept. Full plan: one section, the whole column, "
                         f"`{best['segment_plan'].split(':', 1)[1]}`.\n")

        # SIS's best Oracle: byte-count oracle over the *smaller* default 7-codec set,
        # sample-estimated (not a full validated encode) -- best of the two profiles.
        best_oracle_bpe, best_oracle_lines, best_oracle_plan = None, None, None
        for plan in ("oracle_consec", "oracle_random"):
            bpe, seg_lines = _plan_bits_per_elem(oracle_plans, dataset, plan)
            if bpe == bpe and (best_oracle_bpe is None or bpe < best_oracle_bpe):  # bpe==bpe: not NaN
                best_oracle_bpe, best_oracle_lines, best_oracle_plan = bpe, seg_lines, plan
        if best_oracle_bpe is not None:
            lines.append(f"**4. SIS's best Oracle** (`bench_costmodel_oracle`, `{best_oracle_plan}` "
                         f"sampling, byte-count oracle over the driver's smaller **default "
                         f"7-codec** candidate set -- *not* the 29-codec set rung 3 used, and "
                         f"estimated on a 2,000-element sample, not a full validated encode): "
                         f"**{best_oracle_bpe:.2f} bits/elem, {64/best_oracle_bpe:.2f}x**. "
                         f"Full plan:\n" + "\n".join(best_oracle_lines) + "\n")

        lines.append("")

    lines.append(
        "**In simple terms:** OpenZL wins mainly because it has a technique "
        "(byte-plane transpose) nothing here has. But look at rows 2-4 for SIS "
        "itself: the version that actually ships today (row 2) gets essentially "
        "*nothing* (a bug -- the default sample size collapses to giving up and "
        "storing the data raw). Letting the same DP search a much bigger set of "
        "codecs (row 3) recovers most of the usable gap on its own, no new "
        "capability needed, while still preserving FastSkip. Row 4 (the oracle) "
        "looks *worse* than row 3 only because it was restricted to a smaller "
        "7-codec set for cost reasons -- it is not proof the oracle is weak, it "
        "is more evidence that codec-set size and selection quality both matter, "
        "separately, and both are currently costing real compression that has "
        "nothing to do with missing a transpose.\n")

    return "\n".join(lines)


RA_PRESERVING_SUGGESTIONS = """## Suggestions that preserve random access / FastSkip

Two separate problems, not one, based on the evidence above:

1. **Selection accuracy** (`bench_costmodel_oracle` above): the cost model's
   top pick matches the true byte-count oracle only 47-77% of the time on
   the *same* candidate codecs, and probably explains the ablation's rung
   21-to-22 regression (6.54x -> 4.49x when admitting one more codec). This
   is a pure bug-fix path with no capability change and no FastSkip
   trade-off at all: investigate the cost model for the codec types that
   enter at rung 22+ (`CascadingFORPrevBlockFrequencyPartitionEncoding`,
   `HuffmanEncoding`, `FSEEncoding`, ...) the same way FINDINGS.md already
   diagnosed the 65,536-unique-value entropy-estimator cap for AutoSIS's
   default 100K-sample collapse -- this is very likely the same family of
   defect. Fixing it recovers real compression using codecs the registry
   *already has*, no new code beyond the cost model.

2. **Missing capability** (`bench_openzl_graph` above): OpenZL's win comes
   from byte-plane transposition + per-plane delta + entropy coding, and
   none of that is available to SubIntSplit's DP today. The transpose step
   itself does not need to cost FastSkip: byte `k` of element `i` sits at a
   fixed, directly computable offset within its byte-plane
   (`plane_k_offset + i`), so `decodeAt(i)` under a pure transpose is still
   O(1) arithmetic across the planes -- it is OpenZL's *subsequent* stages,
   `zl.delta_int` (needs a running reference to decode any single element)
   and `zl.private.zstd` (needs its LZ window materialized), that actually
   break random access, not the transpose.

   A **blocked byte-plane transpose** `SubStreamReordererType` -- transpose
   within small fixed-size blocks (e.g. 128-1024 elements, the same order of
   magnitude as `FOREncoder`'s existing 128-element `FrameSize`) rather than
   globally -- would let `decodeAt(i)` reconstruct just element `i`'s block
   (a few hundred elements) instead of the whole column, preserving FastSkip
   at a bounded, tunable cost. Feeding each transposed byte-plane into the
   *existing* RA-capable section codecs (`BitPacking`, `FrameOfReference`,
   both already random-access per the ablation's own partition logic in
   `CodecSetLadder.hpp`) rather than `delta_int`+`zstd` would trade some of
   OpenZL's ~18-22x for a plan that never drops `FastSkip` -- directly the
   property this whole sweep exists to test. Offered to the DP the same way
   `SubStreamReordererType::BWT512` already is (`--reorderer` in
   `bench_ablation`), it slots into the existing ladder infrastructure with
   no new benchmark-side plumbing.

Both are independent, additive follow-ups, not blocked on each other or on
this sweep's own remaining gaps (BWT512/Full-dataset ablation coverage,
`FOR`'s intermittent validation failure, the unpacked int32 `pre`/`post`/
`level` columns).
"""


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

    comparison_section = build_full_choice_comparison(args.results_dir)
    if comparison_section:
        out_lines.append(comparison_section)

    openzl_section = build_openzl_graph_section(args.results_dir)
    if openzl_section:
        out_lines.append(openzl_section)

    oracle_section = build_oracle_section(args.results_dir)
    if oracle_section:
        out_lines.append(oracle_section)

    if openzl_section or oracle_section:
        out_lines.append(RA_PRESERVING_SUGGESTIONS)

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
