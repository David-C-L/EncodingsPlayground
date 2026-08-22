# Prompt: design a refactor of `FastSkip`/`RandomAccess` into an access-cost categorization

This is a self-contained brief for a **planning** session (design only, no
implementation) on `EncodingsPlayground`, a C++ compression-codec benchmark
playground. Hand this file to a fresh planning agent/session — it assumes no
memory of any prior conversation.

**Related, split-out work**: `Benchmarks/drivers/BLOCKFSE_CHECKPOINT_REFACTOR_
PROMPT.md` is a separate, narrower prompt specifically about reducing
`BlockFSEEncoder`'s O(block_size) constant (decoupling its entropy-table
window from its decode-checkpoint granularity). It's designed to run in
parallel with this one — this prompt's categorization work only needs to know
`BlockFSEEncoder` currently *is* O(block_size), not what its constant will
become; the other prompt's outcome just makes that constant smaller once it
lands, with no ordering dependency either way.

## Problem

`Source/encodings/EncodingProperty.hpp` defines `EncodingProperty::RandomAccess`
with a doc comment claiming **O(1)** decode-at-a-point access uniformly. This is
false for several codecs that set the flag:

- `Source/encoders/RunLengthEncoder.hpp`: `decodeAt` does interpolation search +
  `std::upper_bound` over run starts — genuinely **~O(log R)** (R = run count),
  never O(N). An honest, cheap codec.
- `Source/encoders/BWTSectionEncoder.hpp`: `decodeAt` inverts an entire
  `W`-element BWT window via `std::stable_sort` (O(W log W)) on **every single
  call**, even repeated calls into the same window. No caching of decoded
  values at all (confirmed: the class has exactly one member, the inner codec
  pointer — no "last window" state). Doc comment at line 153 already says
  `// decodeAt works at O(W) cost`, so this is a known, documented cost that
  the boolean property just doesn't expose.
- `Source/encoders/BlockFSEEncoder.hpp`: `decodeAt` fully decodes the entire
  tANS bitstream for the containing block on every call — **O(block_size)**.
  There is a `fseCache_` member, but it's an "LRU-1" cache of the *decode
  table* (the FSE normalized-frequency structure), not of decoded values —
  repeated calls into the same block still re-run the full bit-decode loop.
- `Source/encoders/BlockFrequencyPartitionEncoder.hpp`: `decodeAt` scans a
  compact tier-tag bitfield from block-element 0 up to the target index every
  call — **O(block_size)** (specifically O(local index) within the block), no
  decoded-value caching either.

**Measured impact** (this session, XMark tree-id benchmarking,
`Benchmarks/results/xmark_report.md` has full detail): a `SubIntSplitEncoder`
plan mixing 2 plain sections with 3 `BWT<512>`-wrapped sections
(`AdaptiveDictionary`, `RunLength`, `BlockFSE`) decoded a full 200,000-element
column ~1,800x slower than a `Raw` baseline in a single **bulk** pass (96%+ of
decode time in the three wrapped sections — `substream_bulk_ns` breakdown:
325ms/305ms/497ms vs <1ms for the two plain sections) and hung for **3.5+
hours** under a gather-access workload before being killed — all while every
section reported `RandomAccess=true`. A block-caching microbenchmark (`bench_
decode_point`, same-block vs cross-block access) found a real but modest
~1.5x locality benefit, not the orders-of-magnitude difference true
memoization would give — the O(block_size) cost dominates regardless.

Separately: a follow-up trial found that when the DP (`bench_ablation
--universe dp-default --reorderer none`) was **denied** the `BWT` reorderer
entirely, it found a **better-compressing** plan (2.48x vs the registered
config's 2.47x on one dataset, 2.54x vs 2.46x on the other) that also decoded
**400-523x faster** in bulk. The registered default paid a large, unforced
decode-speed cost for *no* compression benefit — a selection-quality problem,
not a capability gap, and independent evidence (alongside a documented DP
"admitting more codecs makes the plan worse" regression and an oracle
top-1-accuracy of 0-77% depending on candidate-set size) that the DP's
cost-model estimation is unreliable, on top of the categorization problem this
prompt is about.

## Existing infrastructure to build on, not replace

There is already a richer, **ns-valued** cost framework:
`Source/encoders/selectors/costs/EncodingCostModel.hpp` defines
`CostModelDimension { Compression, EncodeSpeed, DecodeAllSpeed, DecodeAtSpeed,
DecodeRangeSpeed }`, and `SpeedCostModel.hpp`'s `DecodeAtSpeedCostModel`
already computes a per-access ns cost — **but** its `isSequentialDecodeAt()`
(`SpeedCostModel.hpp:151-155`) is a hardcoded 3-type switch
(`Huffman | FSE | LZ4`) with zero awareness of block size. `BWTSectionEncoder`,
`BlockFSEEncoder`, and `BlockFrequencyPartitionEncoder` all fall through to
the *default* branch of `estimateCpuDecodeAtNs` and get costed **as if O(1)**
— directly contradicting their own doc comments' O(block_size)/O(W) claims,
and directly explaining why a speed-aware selector wouldn't currently
penalize BWT correctly even if asked to. This existing framework, not a new
mechanism, is the natural target for the categorization to feed into.

## Proposed categorization

An access-cost **class**, applied independently to two layers that compound:

1. **The section codec itself** — `O(1)` (direct index math: `BitPacking`,
   `RawEncoding`), `O(log n)` (bounded search: `RunLengthEncoder`'s
   interpolation search), `O(block_size)` (must materialize a whole block to
   answer one query: `BlockFSEEncoder`, `BlockFrequencyPartitionEncoder`, and
   their `RangePack`/`CascadingFOR`-block variants — *only the two named
   classes were individually source-verified this session; the rest are
   inferred from the "Block" naming convention and need real verification*),
   `O(n)` (must decode everything: OpenZL-style codecs, confirmed via
   `bench_openzl_graph` this session to have no partial-decode path at all).
2. **The reordering layer** — independently classified the same way. `BWT`
   is `O(block_size log block_size)` (the sort factor makes it strictly worse
   than a same-block-size codec-only cost). A reorderer may or may not carry
   its own index; today none do (`SubStreamReordererType` only offers
   `None`/`BWT512`), but the design should not assume that stays true.
3. **Composition**: a reordered section pays *both* costs — at minimum the
   reorderer's own class cost, plus whatever the wrapped codec's class adds
   on top (e.g. `BWT<512>|BlockFSE` should cost roughly the O(W log W) BWT
   inversion *plus* the wrapped codec's own O(block_size) decode, not just
   one or the other). Design the composition rule explicitly rather than
   picking whichever layer is "worse". This session measured `BlockFSEEncoding`
   truly in isolation (a new manual plan, `SIS_BareBlockFSE`, single
   full-width section, no reorderer: 7,550 ns/probe same-block, 30,720
   ns/probe cross-block), but the only *compound* number available is the
   registered `AutoSIS_LSB` 5-section plan as a whole (78,978/118,288
   ns/probe), which mixes cost from `BWT<512>|BlockFSE` *and* two other
   `BWT`-wrapped sections (`AdaptiveDictionary`, `RunLength`) — **not** a
   clean `BWT<512>|BlockFSE`-alone number. Getting one (e.g. a second manual
   plan, single full-width `BWT<512>|BlockFSE` section, no other sections)
   is a small, cheap first step for whoever picks this up, and would settle
   the composition rule with real measurement instead of the two whole-plan
   numbers above, which only bound the answer loosely.

## Design questions for the plan to answer

1. **Where does the class live?** Extend `EncodingProperties`
   (`Source/encodings/EncodingProperty.hpp`) with a small enum field? A
   parallel per-`EncodingType` table (sibling to
   `defaultAutoSubIntSplitCostModelTypes()` in
   `Source/encoders/SubIntSplitEncoder.hpp`)? Whatever's chosen must be
   queryable per-codec-instance (some codecs are generic over a block-size
   parameter chosen at encode time — `BlockFSEEncoder`'s `planBlockSize()`
   picks from `{256, 512, 1024, 2048, 4096, 8192}` per dataset — so the class
   alone isn't enough, the *value* of block_size matters too for real cost
   estimates).
2. **How does the ablation infrastructure use it?**
   (Explicit requirement, not optional.) `Source/benchmark/registry/
   CodecSetLadder.hpp`'s `partition()`/`sectionRandomAccess()` today only
   splits a universe into a boolean RA/non-RA pair — which is why
   `dpDefaultUniverse()`'s "RA" bucket lumps O(1) `BitPacking` together with
   O(block_size) `BlockFSEEncoding`, indistinguishable to every existing
   ladder. Redesign `partition()`/`buildLadder()` (or add a sibling) to group
   by the 4-way class, so a ladder can express "O(1)+O(log n) only" →
   "+ O(block_size)" → "+ O(n)" as a principled progression, replacing
   today's single RA→non-RA step. `bench_ablation.cpp`'s `--universe`/
   `--ladder` CLI surface should be extended accordingly (a new `--universe`
   value, or a new ladder set) rather than left as `all`/`dp-default` only.
3. **How does `isSequentialDecodeAt`/`estimateCpuDecodeAtNs`
   (`SpeedCostModel.hpp`) generalize** from the hardcoded 3-type switch to
   consult the new classification, including the block-size *value* for
   O(block_size) codecs (not just "is it O(block_size)")?
4. **How is the reorderer layer's class attached and composed** with the
   section's, feeding into the same cost model?
5. **Backward compatibility**: `FastSkip`/`RandomAccess` almost certainly
   become *derived* (`class != O(n)`) rather than removed, since
   `CodecSetLadder.hpp` and driver output columns consume them today as
   booleans. Design the derivation, and audit every existing boolean
   consumer for whether it actually wanted "not O(n)" or something stricter
   ("O(1) specifically") — several probably silently wanted the latter.
6. **Concrete policy change to evaluate, not assume**: should the
   *registered default* `AutoSIS` config
   (`Source/benchmark/registry/EncoderRegistry.hpp`'s `sisAutoEncoders()`,
   via `makeDefaultAutoSubIntSplitConfig` in
   `Source/encoders/SubIntSplitEncoder.hpp`) restrict its codec universe to
   **O(1) and O(log n) only** by default — i.e. exclude O(block_size) and
   O(n) codecs from the default search entirely, so "AutoSIS as shipped"
   guarantees genuinely fast random access — with a separate, explicitly
   opt-in config (e.g. `makeMaxCompressionAutoSubIntSplitConfig`) for callers
   who want to trade access speed for ratio deliberately? This session's own
   data point for what that would cost, measured via the oracle (not yet a
   real DP result, since the categorization to filter by doesn't exist yet):
   on the XMark tree-id data, an oracle restricted to O(1)/O(log n) only
   reached 5.65x-5.93x compression, vs 8.28x-8.52x when O(block_size) codecs
   were allowed (RA-only) — a real, non-trivial trade-off, not a negligible
   one, but still well above the current shipped default's 2.45-2.48x. The
   plan should design the config split and let real data (once the
   categorization exists and the DP can actually search under it) confirm or
   revise this specific number.

## Deliverable

A phased implementation plan (this prompt is asking for the plan, not the
implementation): what changes, in what order, keeping the codebase buildable
and `bench_smoke` passing at each phase, with a verification step per phase
(cite the specific test/driver invocation, not just "add tests").
