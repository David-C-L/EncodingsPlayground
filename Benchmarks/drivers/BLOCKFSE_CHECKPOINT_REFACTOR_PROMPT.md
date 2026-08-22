# Prompt: decouple BlockFSE's entropy-table granularity from its decode-checkpoint granularity

This is a self-contained brief for a **planning** session (design only, no
implementation) on `EncodingsPlayground`, a C++ compression-codec benchmark
playground. Hand this file to a fresh planning agent/session — it assumes no
memory of any prior conversation. It is **split out from, and designed to run
in parallel with**, a broader, separate prompt,
`Benchmarks/drivers/FASTSKIP_REFACTOR_PROMPT.md`, which covers replacing the
codebase's boolean `FastSkip`/`RandomAccess` properties with a real
access-cost categorization (O(1)/O(log n)/O(block_size)/O(n)) across every
codec and the reordering layer. This prompt is narrower and mostly
independent: it's specifically about *reducing* `BlockFSEEncoder`'s
O(block_size) constant without giving up its compression, which the other
prompt's categorization work can proceed without waiting for (it would just
classify today's `BlockFSEEncoder` as O(block_size) as-is, and benefit
automatically once this work lands and the constant shrinks).

## Problem, with real isolated numbers

`Source/encoders/BlockFSEEncoder.hpp` picks one `blockSize` parameter
(`planBlockSize()`, greedy cost sweep over candidates `{256, 512, 1024, 2048,
4096, 8192}`, `BlockFSEEncoder.hpp:64`) that controls **two different things
at once**:

1. How large a window the FSE/tANS entropy table gets fit to — bigger is
   better for compression (more symbols to build an accurate frequency
   model from, and the table itself has fixed per-block overhead that
   amortizes better over more elements).
2. How much must be re-decoded to answer *one* point query — `decodeAt`
   fully replays the tANS bitstream for the entire containing block on
   **every single call** (`BlockFSEEncoder.hpp:137-150`), i.e. genuinely
   O(block_size). There's an `fseCache_` member, but it's an "LRU-1" cache of
   the *decode table* (the normalized-frequency structure), not of decoded
   values — repeated calls into the same block still re-run the full
   bit-decode loop, they just skip rebuilding the table.

Measured this session, isolating `BlockFSEEncoding` from the `BWT<512>`
wrapper it's normally paired with (a new manual SIS plan,
`SIS_BareBlockFSE`, single full-width section, no reorderer — see
`Benchmarks/results/xmark_report.md`'s "Isolating bare BlockFSE" section for
full detail), on a 200,000-element real dataset:

| metric | bare `BlockFSE` | genuinely O(1) codec in the same sweep (`FPE_NoIndex`) |
|---|---|---|
| point, same-block (ns/probe) | 7,550 | ~5 |
| point, cross-block (ns/probe) | 30,720 | ~5 |
| bulk (Meps) | 4.40 | ~800+ |

The decode-table cache (`fseCache_`) does give a real, measurable same-block
advantage (~4.1x, same-block vs cross-block) — confirming the caching that
exists helps, it's just not enough. Even the *best* case here is
~1,500x slower than a genuinely O(1) codec.

Also measured: applying `BlockFSE` to the *whole* 64-bit column directly
(rather than as one narrow section after bit-splitting) achieved a
compression ratio of **0.72x — worse than storing the data raw**. FSE/tANS
needs a genuinely skewed symbol distribution to win; on this near-unique
64-bit id, that skew only exists in narrow sub-fields `SubIntSplit`'s
bit-splitting isolates first (e.g. an 8-bit tail section), not across the
whole value. Keep this in mind when designing/measuring: a "does this design
compress well" check needs to test `BlockFSE` as a section within a
realistic split plan, not as a standalone whole-column codec — the whole-
column number is not representative of how the codec is actually used when
it wins.

## Proposed direction: decouple table scope from checkpoint stride

The core idea: keep the *entropy table* fit to a large window (for good
compression), but store lightweight decoder-**state** snapshots at a much
finer, independently-tunable interval *within* that window — so a point
query seeks to the nearest checkpoint (an O(1) or O(log blocks) lookup, the
block index already does this) and replays only the *checkpoint stride*
forward, not the whole table-fitting window. This is a known pattern in
fast-compression / checkpointed-entropy-coding literature (interleaved/
checkpointed rANS-family designs) — not a novel algorithm to invent, mostly
an application to this specific codebase's existing format.

Concretely, for tANS/FSE: the decoder state is a single small integer (the
current automaton state index). A checkpoint is "the state value after
decoding element `k`", for `k` at a fixed stride (e.g. every 32 elements)
within the larger table-fitting block. Decoding element `i`: find its block
(existing index), find the nearest checkpoint `<= i` within that block
(new, small, cheap structure — an array of `blocksize/stride` state values
per block), decode forward from there. Worst-case per-query cost becomes
O(checkpoint_stride), independent of the table-fitting block size — you can
keep `blockSize` large (or even go larger than today's 8192 max candidate,
since the cost of doing so no longer scales point-access latency) while
making `checkpoint_stride` small.

## Design questions for the plan to answer

1. **Format**: how do checkpoint state snapshots fit into
   `BlockFSEEncoder`'s existing on-disk layout (header/block-index/payload,
   documented in the class's own comments)? Sizing: one state value per
   checkpoint per block — quantify the storage overhead at a few candidate
   strides (e.g. 16/32/64/128) against this session's real corpus, and
   whether it's worth making the stride itself a `planBlockSize()`-style
   tunable parameter (jointly optimizing table size *and* checkpoint stride
   against a real latency target, using the ns-valued
   `CostModelDimension::DecodeAtSpeed` machinery
   `Source/encoders/selectors/costs/SpeedCostModel.hpp` already has, but
   doesn't currently apply to this kind of *internal* codec parameter
   choice — today `estimateCpuDecodeAtNs` costs `BlockFSEEncoder` as if O(1)
   by omission, a separate but related bug documented in
   `FASTSKIP_REFACTOR_PROMPT.md`).
2. **decodeRange interaction**: `BlockFSEEncoder::decodeRange`
   (`BlockFSEEncoder.hpp:186-...`) already handles multi-block ranges;
   within-block range decode should get the same checkpoint benefit as
   `decodeAt` (seek to nearest checkpoint before the range start, not the
   block start) — design this alongside `decodeAt`, not as an afterthought.
3. **Compatibility / versioning**: new encoded output won't be byte-compatible
   with today's format. Does this become a new `EncodingType` variant
   (e.g. `BlockFSECheckpointedEncoding`) so both remain selectable and
   comparable, or a version-flagged evolution of `BlockFSEEncoding` itself?
   Consider that `CodecSetLadder.hpp`'s and `bench_costmodel_oracle`'s
   candidate-set machinery would need to know about a new type either way.
4. **Does this generalize to `BlockFrequencyPartitionEncoder`?**
   (`Source/encoders/BlockFrequencyPartitionEncoder.hpp`, also confirmed
   O(block_size) this session, no decoded-value caching — its `decodeAt`
   scans a tier-tag bitfield from block-element 0 up to the target index
   every call.) A similar idea applies there too: periodic cumulative-rank
   checkpoints instead of full block scan from 0 (a Fenwick-tree/prefix-sum
   style structure) — evaluate whether this is the same design or a
   sufficiently different data structure to warrant separate treatment.
5. **Does NOT apply to `BWTSectionEncoder` the same way, and that's fine to
   scope out.** `Source/encoders/BWTSectionEncoder.hpp`'s `decodeAt` inverts
   a whole BWT window via `std::stable_sort`-based LF-mapping
   (`BWTSectionEncoder.hpp:191-211`) — you cannot "resume from a checkpoint"
   partway through a BWT inversion the way you can resume a tANS decode from
   a saved state, because the inverse fundamentally needs the whole window's
   sort structure built first. `BWT`'s own lever is simpler and separate:
   its window size `W` is already a template parameter
   (`BWTSectionEncoder.hpp:43`, `template <typename T, size_t W = 512>`),
   currently only instantiated at 512
   (`SubStreamReordererType::BWT512`) — a smaller `W` directly shrinks its
   O(W log W) cost at a compression cost, no new design needed, just
   evaluating whether a smaller instantiation (e.g. 128 or 64) is worth
   exposing. Measuring this trade-off empirically (a few `W` instantiations,
   compression vs decode speed) is worth doing but is a much smaller task
   than the checkpoint design above — note it in the plan as a quick,
   near-free companion action, not the main deliverable.

## Deliverable

A phased implementation plan (this prompt asks for the plan, not the
implementation): the exact format change, the checkpoint-stride selection
strategy (fixed default vs tunable vs cost-model-driven), how `decodeAt` and
`decodeRange` both change, compatibility/versioning decision, and a
verification step per phase (cite the specific test/driver invocation,
e.g. `bench_smoke`, `bench_decode_point` with the same same-block/cross-block
methodology already validated this session, not just "add tests"). Include a
concrete prediction: what should `SIS_BareBlockFSE`-equivalent same-block
`ns_per_probe` look like *after* this change, roughly, given a chosen
checkpoint stride vs today's 7,550 ns/probe at a ~256-8192 block size — this
gives the eventual implementation a number to validate against.
