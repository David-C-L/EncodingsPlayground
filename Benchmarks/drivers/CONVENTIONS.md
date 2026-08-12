# Benchmark driver conventions

Every driver in this directory measures **one** thing and follows the contract below.
The contract is written down here so a driver can be written, reviewed or ported
without reading another driver end to end.

The reference implementation is `gather_heatmap_benchmark.cpp` (to become
`bench_decode_gather`). Where this document and that driver disagree, this
document is the intent and the driver is the bug.

---

## 1. One driver, one concern

| Driver | Measures |
|---|---|
| `bench_compression` | encoded size only — no timing |
| `bench_encode` | encode time, selection time, encode peak heap |
| `bench_decode_bulk` | full materialization |
| `bench_decode_range` | one contiguous range |
| `bench_decode_gather` | an ordered list of row ranges with gaps to skip |
| `bench_decode_point` | individual element lookups |
| `bench_costmodel_oracle` | cost-model estimate vs oracle actual |
| `bench_openzl_graph` | OpenZL codec-DAG structure per segment |
| `bench_index_oracle` | measured argmin over FPE index types |
| `bench_ablation` | codec-set and reordering ladders |
| `bench_reordering` | reorderer x permutation format |
| `bench_smoke` | correctness only, small N, wired into ctest |

Separate executables exist so that a `perf`/VTune session on one phase captures
that phase and nothing else, and so a cheap table (compression) does not require
paying for an expensive one (gather sweeps).

## 2. Encode once — except in `bench_encode`

Decode drivers obtain their payload from `ArtifactCache` with
`EncodeMeasurement::None`: the encoder is `reset()` once and `encode()`d once per
(encoder, dataset, N). This matters beyond speed — for AutoSIS encoders every
`encode()` re-runs cost-model selection, so re-encoding per iteration silently
re-does the DP and changes what is being measured.

`bench_encode` is the exception and the reason the split exists: encoding is
measured repeatedly, in its own process, with nothing else running.

**Never** call `encode()` inside a timed decode loop, and never re-encode to
recover a property (per-section bytes, selection time) that the artifact already
carries.

## 3. The measurement contract

All timed work goes through `benchmark::measure()` (`benchmark/MeasureLoop.hpp`):

- Output buffers are **hoisted out of the timed region**. Use the `*Into`
  variants (`decodeAllInto`, `decodeRangeInto`, `decodeGatherInto`), never the
  allocating `decodeAll`/`decodeRange`, which charge a heap allocation and a
  zero-fill to decode throughput.
- Cache preparation (warm or evict) happens **outside** the `t0`/`t1` window and
  is timed separately into `evict_ns`.
- After each iteration the sink is passed through `clobber()` so the optimizer
  cannot elide the work; results must be observably consumed.
- Report `time_ns` (median), `time_p90_ns` and `time_min_ns` from
  `TimingStats::summarize` — never a mean, which one descheduled iteration can
  dominate. Derived quantities that have no such outliers (ratios, throughputs)
  use `MomentSummary` (mean +/- stddev) instead.
- For operations that take tens of nanoseconds (point lookups), time a **batch**
  and report `ns_per_probe`, plus a `clock_overhead_ns` calibration row. Two
  `high_resolution_clock::now()` calls around a single ~50 ns call measure mostly
  the clock.

## 3a. Measurement hygiene — read before believing a number

Measured on this box while porting the gather driver: with the default
`--iterations 5` and no pinning, the run-to-run spread of a per-cell median was
**13–46%**. At that precision almost any comparison between two drivers, two
cache states or two codecs is unfalsifiable, and it is entirely possible to
"confirm" a 1.5x effect that is not there — this happened during the port and
cost a round of investigation.

So, before reporting a comparison:

- **Pin the process** (`taskset -c N`). This alone took one comparison from a
  46% spread to 5%.
- **Raise `--iterations`** to at least ~21 for anything whose per-cell time is
  in the microseconds.
- **Repeat the whole sweep** at least three times and look at the spread of the
  medians, not one median. A single sweep cannot tell you its own error bar.
- **Check the governor.** The manifest records it; on `powersave` with SMT
  enabled, frequency drift is the dominant term. `performance` plus SMT off is
  the configuration a paper number should come from.
- **Distrust the first run of a freshly built binary** — first-touch page faults
  and a cold instruction cache show up as one elevated outlier.

None of this is optional politeness: a driver that reports three significant
figures from five unpinned iterations is reporting noise with a decimal point.

## 4. CLI

Built with `benchmark/Cli.hpp`. Flags bind to variables, and `--help` prints
defaults read from those variables, so help cannot drift from the code.

Common to every driver:

```
--n N                  stream length in elements
--iterations N         timed iterations per cell
--warmup N             untimed iterations per cell
--seed N               all randomness derives from this
--dataset SUBSTR       repeatable; substring filter
--encoder SUBSTR       repeatable; substring filter
--cache-state MODE     hot | cold-payload | cold-all
--evict-method MODE    auto | clflush | llc-thrash | none
--working-set-targets  payload-byte targets; default derived from detected LLC
--output PATH          result file
--format FMT           csv | parquet
--validate             round-trip check before measuring
--dry-run              print the sweep plan and exit
--help
```

An empty filter set means "everything". A filter matching nothing is an **error**
with a non-zero exit, not an empty result file.

## 5. `--dry-run` and `--validate`

`--dry-run` prints the resolved configuration, the **achieved** structure of the
traces (not the requested one) and the total cell and call counts, then exits
without measuring. This is the cheapest way to catch a sweep that would have run
for six hours or produced degenerate cells, so every driver must support it.

`--validate` round-trips each (encoder, dataset) pair against a reference decode
before measuring. A pair that fails is **excluded from the output**, recorded, and
the process exits with code **2** after finishing the rest. A failing codec must
never appear in results — `BlockFORFPE` is known not to round-trip
TwitterSnowflake, and its rows are worthless rather than merely wrong.

For gather drivers, `--validate` additionally asserts that the sigma = 1 slice
equals a direct `decodeRange`, which is what makes the gather and range drivers
comparable.

## 6. Results

Written via `benchmark/ResultWriter.hpp` (Arrow-backed; CSV or Parquet).

- Set fields **by column name**, never positionally. The pre-refactor gather
  driver emitted `",,,,,,"` at a hand-counted offset for a skipped cell, which
  silently corrupts the moment a column is inserted.
- "Not applicable" is a **typed null**, never a sentinel number. A codec with no
  distinct skip phase reports null for `gather_skip_ns`, not `-1` and not `0`.
- A cell that was deliberately not measured is emitted with `skipped=1` and null
  timings, **not dropped**. "Not viable at this range count" is a result; a
  missing row is indistinguishable from a crash.
- Report what was *achieved* alongside what was *requested*:
  `sigma_nominal`/`sigma_achieved`, `k_nominal`/`k_actual`. Grids are indexed by
  the nominal value (it identifies the cell); lines are plotted at the achieved
  one (it is the honest x coordinate).
- Flush per encoder so a long sweep stays inspectable while it runs.

Columns present on every row:

```
driver, dataset, encoding, family, variant, is_sequential, fast_skip,
random_access, N, seed, cache_state, evict_method, evict_ns, payload_bytes,
compression_ratio, iterations, warmup, time_ns, time_p90_ns, time_min_ns,
truncated, skipped
```

## 7. Run manifest

Every run writes `<output>.manifest.json` **before** the sweep starts, and
rewrites it with the finish time and exit code at the end — a killed run must
still have provenance. It records the git SHA and dirty flag the binary was
*built* from (stamped in at configure time, never shelled out at runtime),
compiler and flags, `HAVE_OPENZL`, hostname, CPU model, the **detected** cache
topology, scaling governor / SMT / THP state, seed, argv, and dataset
fingerprints.

Governor, SMT and THP are not optional detail: cold-path and TLB-sensitive
numbers are uninterpretable without them, and a huge-page-backed run must never
be compared against a 4 KiB-page one.

## 8. Cache state

- `hot` is an **active** state, not the absence of one: the payload is
  read-touched before each timed iteration so "hot" means resident rather than
  "whatever the previous iteration left behind".
- `cold-payload` evicts only the encoded bytes (`clflush`, x86 only).
- `cold-all` additionally evicts the sink and codec-internal structures (FPE
  index tables, dictionaries, rank samples) via `internalBuffers()`; where those
  cannot be enumerated it falls back to an LLC thrash, which also disturbs the
  TLB and branch predictors and is therefore a *lower bound on hotness* rather
  than a clean payload-cold measurement.
- A requested cold state that cannot be delivered is an **error**, never a
  silent downgrade to hot under a cold label.
- `madvise(MADV_DONTNEED)` must never be used on the payload: on private
  anonymous heap memory the pages fault back in **zeroed**, corrupting the
  payload rather than cooling it, and silently unless validation runs after
  eviction.
- Cache sizes are **detected**, never hardcoded. Hot and cold only differ while
  the payload fits in LLC, so cache-policy drivers sweep payload size across the
  detected boundary and report achieved `payload_bytes` per row.

## 9. Registries

Encoders and datasets come from `benchmark/registry/`, never from a list inline
in a driver — five drivers previously each carried their own copy with a
different subset commented out. Add an encoder once, to the appropriate family
(`baselineEncoders`, `fpeIndexFamily`, `sisManualPlans`, `sisAutoEncoders`,
`reorderingFamily`).

Dataset paths resolve from `$ENCODINGS_DATASETS` with a repo-relative fallback.
A missing dataset file is a filtered-out dataset with a warning, not an
exception at startup.

## 10. Porting to nimble

Driver bodies are written against the `BenchTargetC` concept
(`benchmark/targets/BenchTarget.hpp`), whose names follow nimble's API
(`materializeAll`, `materializeRange`, `skipThenMaterialize`, `pointRead`), not
the playground's `decode*`. The adapter is a zero-overhead inline pass-through;
`target.native()` is the escape hatch for anything the concept does not express,
so nimble's own harnesses and the production selective-reader path stay
reachable.

Two things do **not** port and are expected to be reimplemented per repo: the
registries (nimble's `EncodingFactory` does not dispatch SubIntSplit or
FrequencyPartition, and gates SIS behind `NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS`)
and the oracle/cost-model grid (nimble's selector discards its estimate, so the
driver must call `sampleIntoU64` + `selectSplits` itself).

Capabilities that must be **emulated** on one side are labelled in the output:
nimble has no `decodeAt`, so `pointRead` there is a one-row visitor read and
carries `emulated_point_read=1`. A native lookup is never silently compared
against an emulated one.
