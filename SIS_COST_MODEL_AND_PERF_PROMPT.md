# Session prompt: fix the AutoSIS cost model, then make SIS pay for itself

You are working in `/home/david/Documents/PhD/symbol-store/MetaNimbleProject/EncodingsPlayground`
on branch `benchmarks/microbench-decomposition` (PR #3).

Read `Benchmarks/drivers/FINDINGS.md` §9 and `Benchmarks/drivers/CONVENTIONS.md`
before doing anything. FINDINGS records what has already been measured; do not
re-derive it. CONVENTIONS §3a is the measurement protocol and is not optional —
the run-to-run spread of a per-cell median on this box is 13–46% unpinned, which
is wide enough to "confirm" effects that do not exist. That has already happened
once in this project.

**Distinguish measured fact from hypothesis throughout.** Everything under
"Observed" below was measured. Everything under "Candidate mechanisms" is a guess
with a pointer, and several of them are probably wrong. Confirm or refute them
rather than assuming them.

---

## Part A — The blocker: the DP's plan gets worse with more information

The SubIntSplit DP produces a *worse* plan the more data it samples, and a worse
plan when given *more* candidate codecs. Both are backwards. Until this is fixed,
AutoSIS numbers at the registry default are unusable and the random-access
ablation cannot express its result.

### A1. Observed — sample-size axis

TwitterSnowflake, N=200000, `enablePrune` on and off agreeing exactly at every
row (so this is **not** the pruning bug fixed in commit `72740cc`):

| samples | segments | bytes | plan |
|---|---|---|---|
| 10000 | 3 | **1177554** | `[0..20]` Dictionary, `[21..52]` FreqPartition, `[53..63]` RunLength |
| 50000 | 3 | 1672038 | `[0..20]` FreqPartition, `[21..53]` BitPacking, `[54..63]` RunLength |
| 100000 | **1** | 1603171 | `[0..63]` FreqPartition |

100000 is the default in `makeDefaultAutoSubIntSplitConfig`, so the registry's
AutoSIS entries emit **64.13 bits/element — worse than raw** — against 40.09
bits/element for the hand-written `SIS_Snowflake6` plan on the same data.

### A2. Observed — codec-set axis

`bench_ablation` gives the DP 23–24 candidate codecs and gets a single segment
with byte-identical output at every ladder rung. The same encoder with
`CostModelSet::defaultEncodings()` (seven codecs) splits into three. Adding
candidates makes the plan worse.

### A3. Candidate mechanisms (unconfirmed — this is the investigation)

1. **Fixed costs are extrapolated as if they were per-element.**
   [`IDSubStreamEncodingSelector.hpp:223`](Source/encoders/selectors/IDSubStreamEncodingSelector.hpp#L223):
   ```cpp
   const double totalCost = perSampleCost * double(effectiveCount) / double(numValues);
   ```
   Every cost model returns one number that is then scaled by `N/sample`. A
   dictionary table, an FPE tier dictionary and a section header are **fixed**
   costs — they do not grow with the row count — so scaling them by `N/sample`
   misprices them, and the error is a function of the sample size. That is exactly
   the shape of the A1 symptom.

   Strong supporting hint: the design already knows this distinction exists, but
   only for reorderers. [`:241`](Source/encoders/selectors/IDSubStreamEncodingSelector.hpp#L241)
   calls `rmodel->overheadBits(effectiveCount)` as a *separate* term rather than
   scaling it. `EncodingCostModel` has no equivalent. **Likely fix: split the
   encoding cost-model interface into a per-element term and a fixed term, and
   extrapolate only the former.** Check `Source/encoders/selectors/costs/EncodingCostModel.hpp`
   for which models carry table/header costs (`dictionaryCostBits`,
   `frequencyPartitionCostBits`, `mainlyConstantCostBits` are the obvious ones).

2. **Metric capping silently degrades estimates past a threshold.**
   `kUniqueCountCap = 1 << 16 = 65536`
   ([`MetricCollector.hpp:108`](Source/encoders/selectors/MetricCollector.hpp#L108)).
   Once unique values exceed it, `uniqueCapped = true` and `computeEntropy`
   returns a *binary-entropy lower bound*, `-p·log2(p) − (1−p)·log2(1−p)` with
   `p = cap/n` — a number below 1 bit, for data whose real entropy is tens of
   bits. `uniqueCountCapped` also gates `dictionaryCostBits`. This explains the
   100000-sample collapse (Snowflake IDs are near-unique) but **not** the 50000
   case, which is below the cap. So it is at most part of the story.

3. **One sampler for codecs that need different samples.**
   The selector draws a single block-stratified sample (`blockSize = 32`,
   `maxSamples = 100000` in `makeDefaultAutoSubIntSplitConfig`) and evaluates
   every cost model on it. But RLE, Delta and the Cascading-FOR family can only
   see runs in *consecutive* data, and a block-stratified sample destroys run
   structure between blocks. Note `Source/generators/samplers/EncodingSamplingProfile.hpp`
   already defines `SamplingProfile::{Random, Consecutive, WideContiguous}` and
   `preferredSamplingProfile(EncodingType)` — and the selector does not use them.
   `bench_costmodel_oracle` *does*, via `mergeEncodingGridsByProfile`. Changing
   the sample size changes blocks-versus-block-size, which changes how much run
   structure survives, which is a second mechanism with the right shape for A1.

4. **A specific model is over-optimistic at full width** (the A2 axis). With 24
   candidates something wins `[0..63]` outright; with 7 it does not. Suspects are
   the codecs present only in the wide set: `BlockFrequencyPartitionEncoding`,
   `BlockFSEEncoding`, `BlockFORFPEEncoding`, `RangePack*`, the `CascadingFOR*`
   family, `Huffman`, `FSE`, `LZ4`, `MainlyConstant`, `FrameOfReference`,
   `AdaptiveFramedBitPrefix`.

5. **`splitPenalty` is effectively inert.** It defaults to 100.0 *bits* while
   segment costs are extrapolated to millions of bits, so it cannot influence the
   DP. If splitting is meant to be discouraged at all, the penalty has to be
   expressed relative to the extrapolated scale. Conversely, if it is inert, note
   that the DP currently has no brake on over-splitting either.

### A4. The instrument you already have

**Do not build new tooling first.** `bench_costmodel_oracle` emits exactly the
table this investigation needs: per (bit-range, encoding), `est_bits_per_elem`
against `act_bits_per_elem` with `rel_err`, `model_rank` and `actual_rank`, plus
`top1_accuracy` / `spearman_rho` / `regret_bytes` summaries. It computes actual
encoded bytes for every cell, so it is ground truth.

The decisive experiment is a sweep of that driver:

```
./build/bin/bench_costmodel_oracle --dataset TwitterSnowflake \
    --sample-sizes 2000,10000,50000,100000 --output <out>.csv
```

then, per sample size:
- Does `rel_err` for a *given* (cell, encoding) grow with sample size? → mechanism 1 or 2.
- Does the sign of `rel_err` differ between fixed-overhead codecs (Dictionary,
  FreqPartition) and pure per-element ones (BitPacking, Raw)? → mechanism 1.
- Which encoding has the largest negative `rel_err` (most over-optimistic) on the
  `[0..63]` cell with the wide candidate set? → mechanism 4, and it names the model.
- Does `top1_accuracy` fall as sample size rises? It was 0.64–0.71 at 2000
  samples with the seven-codec set. If it falls, the model is getting worse with
  more data, which localises the bug to the estimator rather than the DP.

Baseline for "what a good plan looks like": `SIS_Snowflake6` reaches 40.09
bits/element on this data, and the oracle plans in `bench_costmodel_oracle`'s
plans table (4–7 segments) are the DP's achievable target.

### A5. Acceptance criteria for Part A

- Plan quality is **monotone non-decreasing** in sample size: more samples never
  produce a worse plan. Assert it in a test.
- Plan quality is **monotone non-decreasing** in candidate-set size: adding a
  codec to the allowed set never makes the chosen plan worse. This is the
  property the whole codec-set ladder depends on, and it is currently false.
- AutoSIS at the registry default beats raw, and is within a stated factor of
  `SIS_Snowflake6` on TwitterSnowflake.
- `Tests/test_subint_plan_agreement.cpp` still passes (pruning stays
  answer-neutral), and gains cases for the two monotonicity properties above.
- Re-run and update `FINDINGS.md` §9b and the PR body.

---

## Part B — Make SIS, AutoSIS and reordering earn their place

The thesis these encodings exist to support is: *accept a modest compression loss
against a purely sequential codec, and get order-of-magnitude better gather and
point throughput in exchange.* The measurements in this refactor say that trade is
currently not being made well. Below are the concrete opportunities found while
building the harness, roughly in decreasing value-per-effort. Each names the
driver that will measure it.

### B1. Per-section gather with span-bounded sequential sections — the big one

**This is the change most directly aimed at the thesis.** SubIntSplit advertises
`FastSkip` only when `allSectionsRandomAccess()`
([`SubIntSplitEncoder.hpp:569`](Source/encoders/SubIntSplitEncoder.hpp#L569)) — a
`min` over sections. One sequential section costs the *whole* encoding its gather
fast path, which is why admitting a single sequential codec looks catastrophic
and why the DP is effectively forced to choose between compression and random
access.

But the whole-encoding flag is coarser than the actual cost. For a gather over
ranges spanning `[lo, hi)`, a sequential section does not need a full decode — it
needs `[lo, hi)`, which is `decodeRangeInto` on that section. The cost becomes
proportional to the *span* of the access rather than to `N`.

Proposal: make `SubIntSplitEncoder::decodeGatherInto` dispatch **per section** —
random-access sections use their skip path, sequential sections decode only the
covering span (or the union of ranges, coalesced) — and report a graded
capability rather than a boolean. Then admitting a sequential codec on one
narrow, high-entropy section costs a bounded amount of gather throughput instead
of all of it, which is exactly the balance the paper wants to demonstrate.

Measure with `bench_decode_gather` across σ, and with `bench_ablation` once
Part A is fixed. Expect the interesting regime at low σ and narrow spans, where
today's behaviour is worst.

### B2. Zero-copy section views — helps bulk, range, gather and cold-path honesty

`SubIntSplitEncoder::slice()`
([`:682`](Source/encoders/SubIntSplitEncoder.hpp#L682)) does
`std::vector<uint8_t> payload(src, src + len)` — it **copies every section's bytes
into decoder-owned storage** on header parse. Consequences:

- A per-decode memcpy of the entire payload, charged to every decode path.
- `cold-payload` measurements are misleading, since the section codecs read the
  copies rather than the flushed payload (this is FINDINGS §6 and is why
  `Decoder::internalBuffers()` exists).

If `EncodedBuffer` gains a non-owning view variant, sections can borrow spans of
the parent buffer. This is the same `EncodedView` refactor that the file-backed
cold path needs (CONVENTIONS §8), so the two motivate each other. Measure with
`bench_decode_bulk` and `bench_decode_gather`, hot and cold.

### B3. FPE full-span range decode is 1.8× its bulk decode

`bench_decode_bulk` and `bench_decode_range` at `(A_frac=0, B_frac=1)` issue the
same access and must agree. RawBitPacked agrees at 1.00, Zstd and OpenZL at 1.05,
but `FPE_PerTierBitmaps` is **1.77×** (1.36–1.84 across runs). `decodeAllPerTierBitmaps`
has a dedicated sequential tier walk while the range path does per-row index work
even when the range covers the whole column. A full-span range request should
dispatch to the bulk path. Cheap, self-contained, and it also lifts SIS, whose
sections are decoded by range. (FINDINGS §2.)

### B4. TierTagArray's point path scans 256 tags per lookup

`decodeAtTierTagArray` jumps to the nearest rank sample and then unpacks up to
`kRankSampleStride = 256` bit-packed tag positions
([`FrequencyPartitionEncoder.hpp:102`](Source/encoders/FrequencyPartitionEncoder.hpp#L102)),
where `decodeAtEliasFano` does one `lower_bound` per tier. Measured 9× slower
than EliasFano under Uniform and 22× under Zipf. It loses on *work*, not
footprint, and no cache state rescues it.

Options: shrink the stride (costs index bytes — `bench_index_oracle` will price
the trade directly), add a second-level rank index, or drop TierTagArray if it
stays Pareto-dominated across datasets. Note FINDINGS §1a: on Zipfian at N=200000
it is on the frontier and *EliasFano* is dominated, so the ranking is
dataset-dependent — settle it with `bench_index_oracle` before removing anything.

### B5. Selection cost is 1300–1900× the encode

Plan selection costs 13.6–14.5 s against a 7.5–12.8 ms encode, scaling with
sample count rather than search strategy (`--exhaustive` is indistinguishable
from the pruned DP). Note the Part A fix will likely make this **worse**, since
pruning was skipping most of the grid.

The dominant term is the grid fill: `BitRangeSegmentBuilder` extracts values for
each of ~2080 `(l, r)` bit ranges and runs `MetricCollector::compute` on each, so
cost is O(64²/2 · sample). Options, cheapest first:

- **Memoize metrics per bit range.** `Source/benchmark/CostModelGrid.hpp` already
  does exactly this for the oracle driver and reduced `MetricCollector::compute`
  calls to one per cell; the selector recomputes.
- **Do not refill the whole grid on retry.** The unpruned retry re-runs every
  cell; only the pruned cells need filling.
- **Coarsen the boundary granularity.** Restricting splits to 2- or 4-bit
  boundaries cuts the grid 4–16×. Measure the compression cost with
  `bench_costmodel_oracle` — if the oracle's own plans land on coarse boundaries
  anyway, this is nearly free.
- **Parallelise over `l`.** The rows are independent.

Measure with `bench_encode`, which reports `selection_ns` separately from
`encode_ns` and sweeps `--sample-sizes`.

### B6. Reordering: pick the permutation format for the access pattern

`ReorderingCodec` now has the `*Into` overrides (commit `e16aab3`), but the
`PermFormat` choice dominates random access: `FlatBitPacked` and `ChunkRelative`
give O(1) lookup, while `DeltaBitPacked`, `DeltaZstd`, `DeltaLZ4`, `ValueGrouped`
and `InverseEliasFano` need a **full permutation unpack** to answer one positional
query.

The promising combination is `WindowedSort` + `ChunkRelative`: intra-chunk ranks
with an implicit base give O(1) random access at a fraction of `FlatBitPacked`'s
size (measured: a flat permutation costs ~17 bits/element at N=100000, and
`DeltaBitPacked` was *larger*, 225011 vs 212510 bytes — delta-coding a sort
permutation does not pay). Sweep window size against permutation bytes and
`perm_lookup_decode_at_ns` with `bench_reordering`.

Also worth testing: BWT inside SIS (`SubStreamReordererType::BWT512`, via
`allowReorderers`) on a *single high-entropy section* rather than the whole
stream. That is the same span-bounded idea as B1 — pay the sequential cost only
where it buys the most.

### B7. Smaller, confirmed items

- **`ISectionCodecIntegral` has no `internalBuffers()`**, so SIS cannot enumerate
  its section codecs' cached state and `cold-all` falls back to an LLC thrash.
  Under-reports in the safe direction, but blocks targeted cold-path measurement
  of nested indexes.
- **`bench_openzl_graph --openzl-per-segment`** is accepted but the per-segment
  attach is not wired.
- **`test_subint_encoder` fails 15 of 43 cases** and is pre-existing (verified by
  stashing the current branch's changes and rebuilding). Unrelated to this work
  but worth triaging.
- **`BlockFORFPE` does not round-trip TwitterSnowflake** (`blockforfpe_decode_bug`),
  and is gated behind `--validate` in the drivers rather than fixed.

---

## Tooling and ground rules

Twelve drivers exist in `Benchmarks/drivers/`, all on a shared core, all with
`--dry-run`, `--validate` (exit 2 on failure, failing codecs excluded), a
`RunManifest` sidecar and typed-null CSV/Parquet output:

`bench_compression`, `bench_encode`, `bench_decode_bulk`, `bench_decode_range`,
`bench_decode_gather`, `bench_decode_point`, `bench_costmodel_oracle`,
`bench_openzl_graph`, `bench_index_oracle`, `bench_reordering`, `bench_ablation`,
`bench_smoke`.

- **Use them rather than writing new ones.** If a question needs a new driver, it
  probably needs a new *axis* on an existing one.
- **Measurement hygiene (CONVENTIONS §3a) is mandatory**: `taskset` pinning,
  `--iterations 21`, three repeats, compare the spread of medians rather than one
  median. This box runs `powersave` with SMT on, so absolute values are not
  paper-grade regardless; rankings that survive three pinned repeats are.
- **Cross-driver agreement is the strongest check available** and has already
  caught one bug that eye-inspection missed. Where two drivers compute the same
  quantity, assert it.
- OpenZL is **retained** as the state-of-the-art baseline; do not remove it.
- Build: `cmake --build build -j$(nproc)`, compiler is `clang++-18`, warnings are
  `-Werror`. Standalone driver builds are documented in the driver headers.
- Commit per logical change with a message that explains *why*, and update
  `FINDINGS.md` when a measurement contradicts something recorded there —
  including retracting it, which has happened twice already and is expected.
