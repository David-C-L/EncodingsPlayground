# Session prompt: plan the port of the microbenchmark drivers into nimble

Produce an **implementation plan** (not code) for porting the microbenchmark
drivers from `EncodingsPlayground` into
`/home/david/Documents/PhD/symbol-store/MetaNimbleProject/nimble`, so the paper's
evaluation runs against the real encodings in their real home.

## The overriding constraint

The playground PR added **~12,000 lines**. Porting it as-is would add a second
benchmark framework to a codebase that already has one, and most of it would be
redundant. **The goal is the smallest possible amount of new nimble code that
still measures the same things.**

Judge every proposed file against: *does nimble already have something that does
this, or most of this?* Where it does, the plan must say which symbol, at which
path, and what thin adaptation is needed. A plan that reproduces the playground's
file list is a failed plan. Expect to justify each new file individually.

Read first, in this order:
1. `EncodingsPlayground/Benchmarks/drivers/CONVENTIONS.md` — the measurement
   contract the port must preserve. This is the thing being ported; the code is
   just its current expression.
2. `EncodingsPlayground/Benchmarks/drivers/FINDINGS.md` — what has been measured
   and why several design choices exist. §5 (hot must be passive), §6 (SubIntSplit
   copies section bytes), §7 (clock overhead) are load-bearing for the port.
3. The nimble facts below. They were established by reading the tree; verify
   anything you intend to rely on, but do not re-derive from scratch.

---

## What nimble already has (reuse these)

### Encoding construction and the correct decode path
- **`dwio/nimble/encodings/tests/TestUtils.h`** (363 lines), built as CMake lib
  `nimble_encodings_tests_utils` and already linked by every existing benchmark
  target. Contains `test::Encoder<E>::encode(...)` and
  **`createEncoding(...)` (~:331), which constructs `E` directly** — this is the
  *correct* path for SubIntSplit and FrequencyPartition and it replaces most of
  what the playground's `ArtifactCache` + `PlaygroundTarget` do.
  `TestTrivialEncodingSelectionPolicy<TInner>` (:208-275) has a
  `realNestedSelection` flag (:248-266) that switches nested streams from
  forced-Trivial to real cost-based selection with SubIntSplit stripped out —
  which is how you get diverse per-section encodings inside SIS.
- **Critical gotcha**: `EncodingFactory` does **not** dispatch
  `EncodingType::SubIntSplit` or `FrequencyPartition` (see the comment at
  `TestUtils.h:250-256`). Anything decoding through `EncodingFactory::create()`
  will throw. `encodings/benchmarks/BenchmarkUtils.h`'s `decodeBenchmark`
  (:206-214) does exactly this, so the existing `SubIntSplitBenchmark.cpp` decode
  path is already wrong — do not copy it.

### Data generation
- **`dwio/nimble/encodings/benchmarks/BenchmarkUtils.h`** (242 lines):
  `benchmarkPool()`, `nullFactory()`, and generators `makeRandom`, `makeNarrow`,
  `makeConstant`, `makeMainlyConstant`, `makeRunLength`, `makeIncreasing`,
  `makeLowCardinality`, `makeSparseBool`, `makeDenseBool` (:47-175). These cover
  most of the playground's `DatasetRegistry` generated datasets. Only file-backed
  loading needs anything new, and `tools/encoding_bench/EncodingBench.cpp` already
  loads a CSV column.

### CLI, JSON, and the benchmark harness
- **gflags** is already a dependency and already used for exactly this purpose:
  `EncodingBench.cpp:81-84` (`DEFINE_string(file, ...)`). **This should replace
  the playground's `Cli.hpp` (~450 lines) outright.**
- **folly JSON** is already used: `BenchmarkSuite.h:35` includes
  `<folly/json/json.h>` and `:294` builds a `folly::dynamic`. **This should
  replace `RunManifest.hpp`'s hand-rolled JSON emitter.**
- **`dwio/nimble/benchmarks/BenchmarkSuite.h`** (311 lines) — a folly-based
  harness with per-benchmark state, start/end hooks, `setJsonOutputPath()`,
  `addBenchmark(name, fn)`, `run()`. It is an **orphan**: no CMakeLists in that
  directory, no includers anywhere, and the directory is not referenced from the
  top-level `CMakeLists.txt`. Evaluate adopting it rather than writing a runner.
  Be honest in the plan about whether it can express a *sweep* — the playground
  drivers sweep multi-dimensional axes and folly `Benchmark` does not, which may
  make it a poor fit.

### Selective read (gather) and the production read path
- `RowSet = folly::Range<const vector_size_t*>` — a sorted array of row numbers.
  `ReadWithVisitorParams` (`encodings/common/Encoding.h:85-105`),
  `nimble::callReadWithVisitor(...)` (~:465), and
  `SubIntSplitEncoding::bulkScan<kScatter, Visitor>(...)` are the gather surface.
  FrequencyPartition has `readWithVisitor` but **no `bulkScan`**.
- **`tools/encoding_bench/EncodingBench.cpp`** (554 lines) is the closest existing
  thing to what is being ported and is a goldmine of boilerplate to reuse:
  dense/range/scatter `RowSet` generators (:428-446), `ReaderContext` +
  `makeReaderContext` (:289-362, wiring ReaderBase, StripeStreams, ScanSpec,
  NimbleParams, `buildColumnReader`), `makeReadWithVisitorParams` (:365-388), and
  `runEndToEnd` with `ColumnVisitor`/`ExtractToReader` (:390-416). It has **no
  CMake target** (Buck-only in the OSS export), which is a small, high-value fix.

### Build pattern
```cmake
add_executable(nimble_<name>_benchmark <Name>Benchmark.cpp)
target_link_libraries(nimble_<name>_benchmark
  nimble_encodings_tests_utils nimble_common
  Folly::follybenchmark Folly::folly velox_memory)
```
Only 6–7 of the 22 sources in `encodings/benchmarks/` are wired into CMake; the
directory is Buck-first and there is no `BUCK`/`TARGETS` file in this export.

---

## What nimble does not have (this is the real new code)

Keep this list short and defend every entry.

| Missing | Playground source | Note |
|---|---|---|
| Cache-state control | `CachePolicy.hpp` | Nothing in nimble evicts or warms anything. Operates purely on `std::span<const std::byte>`, so it ports **nearly verbatim** — the largest genuinely-new piece, and the one with the clearest justification. |
| The measurement loop | `MeasureLoop.hpp` | Small. folly `BENCHMARK_SUSPEND` cannot express "evict outside the timed region, per iteration", which is the whole contract. |
| Parameterised access traces | `GatherTraceGen.hpp`, `PointTraceGen.hpp`, `SelectiveTraceGen.hpp` | nimble's `EncodingBench.cpp` has fixed dense/range/scatter shapes only — no (σ, run-length) or (skew, working-set) axes. `GatherTrace::expandToRows()` already emits the flat ascending row list `RowSet` wants. |
| Order statistics and axes | `TimingStats.hpp`, `Axes.hpp` | Tiny, port verbatim. |
| Cost estimate extraction | `OracleGrid.hpp`, `CostModelGrid.hpp` | **`SubIntSplitEncoding.h:760` moves only `selectorResult.segments` and drops the cost**, so estimated-vs-actual is unobtainable through the public path. A driver must call `sampleIntoU64` (`SubIntSplitSampler.h:57`) + `selectSplits` (`SubIntSplitSelector.h:111`) itself. |

### Deliberately do not port
- **`ReorderingRegistry` / `bench_reordering`** — nimble has no reorderer layer
  and no BWT at all. The reordering dimension cannot be ported; say so plainly.
- **`CodecSetLadder` / `bench_ablation` as written** — it is built on
  `CostModelSet`, which nimble does not have. nimble's DP is single-objective via
  `bestCostBits` (`SubIntSplitCostModels.h:255-286`). The ablation would have to
  be re-expressed against nimble's candidate mechanism, and there is no
  `EncodingProperty::FastSkip` in nimble either — so the "min over sections"
  observable needs a different derivation.
- **`OpenZLGraphAnalysis`** — the top-level CMakeLists forces
  `OPENZL_BUILD_BENCHMARKS OFF`, and the graph introspection is playground-specific.
- **`ResultWriter`'s Arrow/Parquet path** — **nimble does not depend on Arrow**
  (no `arrow` in its CMakeLists). Adding Arrow to nimble to emit Parquet is a
  poor trade. Recommend CSV via a ~60-line writer, or `folly::dynamic` → JSON,
  reusing what `BenchmarkSuite.h` already does. State the decision explicitly;
  the typed-null semantics from CONVENTIONS §6 must survive whichever is chosen.

---

## nimble-specific facts that change driver behaviour

- **SubIntSplit is gated** behind `NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS`
  (top `CMakeLists.txt:24-27`, default **OFF**). Any SIS benchmark target and its
  `TestUtils.h` traits must be `#ifdef`-guarded the same way
  (`encodings/tests/TestUtils.h:37-39,159-165`). FrequencyPartition is not gated.
- **There is no `decodeAt`.** A point read is a one-row `RowSet` through the
  visitor path, which is a different cost structure. CONVENTIONS §10 requires it
  be labelled `emulated_point_read=1` so a native lookup is never silently
  compared against an emulated one. This matters: the playground's point numbers
  are `decodeAt` and are **not** comparable to nimble's without that label.
- **`FreqPartIndexType` is chosen via `Encoding::Options::frequencyPartitionIndex`**
  (a `uint8_t`, `encodings/common/Encoding.h:134-137`), not a template parameter
  as in the playground. The four index types are at
  `FrequencyPartitionEncoding.h:88-93`. Test idiom at
  `encodings/tests/FrequencyPartitionEncodingTest.cpp:48-52`.
- **Neither SIS nor FPE is in `EncodingSizeEstimation`**
  (`selection/EncodingSizeEstimation.h`) or in `defaultEncodingReadFactors()`, so
  the production cost-based selector can never pick them; a forced policy is
  required. `RandomEncodingSelectionPolicy` in
  `encodings/selection/tests/` is a ready-made example of a non-default policy.
- **`tools/EncodingUtilities.cpp:69-72,137-139,387`** treats SIS/FPE as having no
  nested streams, so `NimbleDump` will not decompose them — per-section byte
  sizes must come from the wire header instead. The SubIntSplit header layout is
  `u64 N | u8 splitCount | u8 order | splitCount × (u8 bitWidth, u64 sectionBytes)`,
  which gives section count and per-section bytes directly.
- **Defaults differ from the playground**, which matters for any comparison:
  nimble's sampler is `maxSamples = 2048`, `blockSize = 128`
  (`SubIntSplitSampler.h:40-47`) against the playground's 100000/32; `splitPenalty`
  is 10.0 against 100.0; `kUniqueCountCap` is `1 << 14` against `1 << 16`.

### Two cross-repo findings to carry over

1. **The entropy-prune bug is playground-only.** `SubIntSplitSelector.h` has no
   pruning at all — verified by grep. Do not go looking for it in nimble.
2. **The cost-extrapolation concern *does* apply to nimble.**
   `SubIntSplitSelector.h:151-153`:
   ```cpp
   const double fullCost = perSampleCost * double(fullCount) / double(numSamples);
   ```
   This is the same wholesale linear scaling the playground uses, so any fixed
   per-segment cost (dictionary table, header) is extrapolated as though it were
   per-element. `EncodingsPlayground/SIS_COST_MODEL_AND_PERF_PROMPT.md` Part A
   documents the symptom and the evidence. nimble's much smaller default sample
   (2048) may mask it. **A ported `bench_costmodel_oracle` is the instrument that
   would reveal it in nimble, which is a strong argument for porting that driver
   early rather than last.**

---

## What the plan must deliver

1. **A file-by-file justification table**: every proposed new nimble file, the
   playground file it derives from, its estimated size, and the nimble symbol it
   could *not* reuse. Anything reusable must name the reused symbol instead.
2. **A total line-count estimate**, with the playground's ~12,000 as the number
   being beaten, and the reductions attributed (gflags replacing `Cli.hpp`, folly
   JSON replacing the manifest emitter, `BenchmarkUtils` generators replacing the
   dataset registry, `test::Encoder<E>` replacing the artifact/target layer, and
   so on).
3. **A driver list for nimble that is probably shorter than twelve.** The
   playground has twelve because it had no harness at all; nimble has folly
   Benchmark, `BenchmarkSuite.h` and `EncodingBench.cpp`. Consider whether
   compression/encode/bulk collapse into one axis-driven driver. Justify the
   count either way.
4. **A port order**, starting with the driver that proves the seam most cheaply.
   `bench_compression` is the natural first — it has no timing, so it needs
   neither `CachePolicy` nor `MeasureLoop`, and it exercises encoding
   construction, the registry and result output end to end. `bench_decode_gather`
   is the natural second because it is the most demanding consumer of the shared
   core; if it fits, the rest will.
5. **The measurement contract preserved explicitly**: encode-once,
   hoisted buffers and `*Into`-equivalent calls, typed nulls rather than
   sentinels, skipped-not-dropped cells, achieved-versus-nominal reporting, the
   run manifest, and `--validate` excluding a failing codec with exit code 2.
   These are what make the numbers trustworthy and they are the actual payload of
   the port.
6. **A statement of what is not portable** — the reordering dimension, the
   codec-set ladder as written, the OpenZL graph analysis — and what the paper
   loses as a result, so that gap is a decision rather than a discovery.
7. **The CMake work**, which is small but currently blocking: `encodings/benchmarks/`
   wires only 6–7 of 22 sources, `dwio/nimble/benchmarks/` has no CMakeLists and
   is not added from the top level, and `tools/encoding_bench/` has no target.

## Ground rules

- Verify before relying. Every path and line number above was read from the tree,
  but the tree may have moved.
- Prefer extending an existing nimble file over adding one. Prefer an added axis
  over an added driver.
- Do not port a playground abstraction whose only justification was that the
  playground lacked something nimble has.
- Where behaviour must differ between the two repos (emulated point reads,
  different sampler defaults, absent reorderers), the plan must say how the output
  records the difference, so results from the two are never silently pooled.
- Measurement hygiene (CONVENTIONS §3a) applies to nimble too: pinning, ≥21
  iterations, three repeats, spread of medians. Nothing about the port changes it.
