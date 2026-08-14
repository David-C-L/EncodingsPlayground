# Measured findings from the driver decomposition

Results that came out of building and cross-checking the drivers, recorded here
because several of them contradict assumptions the design started from and would
otherwise survive only in commit messages.

**All absolute numbers below are provisional.** They were taken on a 12th-gen
i7-1260P laptop with the `powersave` governor and SMT enabled, sometimes with
other benchmark processes running. Even pinned with 21 iterations, the
run-to-run spread of a per-cell median ranged from 1% to 95% depending on codec
and size. Rankings that survive three repeats and both statistics (median and
min) are reported as findings; the absolute values are not paper numbers until
they are retaken under `performance` with SMT off on an otherwise idle machine.
See `CONVENTIONS.md` section 3a.

---

## 1. TierTagArray's point path is not one indirection

The design assumed `TierTagArray` traded footprint for a single indirection and
`EliasFano` traded work for a small footprint, so Zipf — which keeps the index
resident and removes the memory-latency term — should compress the gap between
them.

Zipf does remove the memory term: every codec speeds up as the hot set shrinks
(EliasFano 79 → 16 ns/probe from Uniform to θ=2.0). But the ratio between the
two **widens**, from 9× under Uniform to 22× at θ=1.4. Only the absolute gap
narrows, 643 → 228 ns, which is the shared memory component leaving both sides.

The premise was wrong. `decodeAtTierTagArray` jumps to the nearest rank sample
and then scans up to `kRankSampleStride` = 256 bit-packed tag positions,
unpacking each; `decodeAtEliasFano` does one `lower_bound` per tier. TierTagArray
loses on **work**, not on footprint, and no cache state rescues it. Under
`cold-first-probe`, where every codec pays a DRAM miss, all four index types
collapse into 1.3–2.1 µs and become nearly indistinguishable — the same
conclusion from the other direction.

ns/probe, N=10M TwitterSnowflake, 262144 probes, median of 3 medians:

| contract | trace | PerTierBitmaps | TierTagArray | EliasFano | NoIndex |
|---|---|---|---|---|---|
| hot | uniform | 73 | 723 | 79 | 50 |
| hot | zipf θ=1.4 | 30 | 502 | 22 | 11 |
| cold-all, batch | uniform | 84 | 692 | 150 | 78 |
| cold-all, first probe | uniform | 1583 | 2072 | 2108 | 1313 |

`NoIndex` is the honest lower bound rather than a winner: on TwitterSnowflake
every row lands in fallback so `decodeAt` is a direct array read. On Zipfian1.0,
where tiers exist, it reorders rows and **fails per-probe validation** — it can
materialize correctly-ordered output while answering positional queries in its
internal order. A bulk round-trip does not catch that; the per-probe check added
in W8 does.

## 1a. RETRACTED, then corrected: no index type wins everywhere

**The table that stood here was invalid and the conclusion drawn from it was
wrong.** It combined `payload_bytes` and `index_bytes` measured by
`bench_compression` on **Zipfian at N=100000** with ns/probe taken from finding 1,
which was measured on **TwitterSnowflake at N=10M** — two different datasets at
two different sizes, presented as one Pareto table. On that basis it concluded
TierTagArray was dominated by EliasFano. It is not, and the error was mine in
composing the table rather than in either measurement.

`bench_index_oracle` exists precisely to avoid this: it measures both axes on the
same cell. At N=200000, `--validate`, both axes measured together:

| dataset | index type | payload | index bytes | ns/probe (uniform) | on frontier |
|---|---|---|---|---|---|
| Zipfian1.0 | PerTierBitmaps | 970579 | 300000 | 167.6 | **yes** |
| Zipfian1.0 | TierTagArray | 770579 | 100000 | 1596.4 | **yes** |
| Zipfian1.0 | EliasFano | 774198 | 103619 | 2725.7 | no |
| Zipfian1.0 | NoIndex | 670627 | 0 | — | not viable |
| TwitterSnowflake | PerTierBitmaps | 1600014 | 0 | 28.4 | no |
| TwitterSnowflake | TierTagArray | 1625014 | 25000 | 1142.6 | no |
| TwitterSnowflake | EliasFano | 1600014 | 0 | 33.8 | no |
| TwitterSnowflake | NoIndex | 1600014 | 0 | 22.2 | **yes** |

On Zipfian, **EliasFano is the dominated one** — larger payload *and* 1.7x the
latency of TierTagArray — the exact reverse of the retracted claim.

**The real finding is that the ranking does not generalise.** The scalarised
oracle picks TierTagArray at every lambda from 1e-6 to 1e2 on Zipfian, and NoIndex
at every lambda on TwitterSnowflake. No single index type wins across datasets,
which is the argument *for* giving this dimension a cost model rather than a
default — and it is a stronger argument than the one the retracted table made.

Two caveats that matter for how this is used:

- **TwitterSnowflake is degenerate for this dimension.** Three of the four types
  report `index_bytes = 0`: every row lands in fallback, so no tiers exist and no
  index is built. NoIndex therefore wins trivially, and TierTagArray's 25000 bytes
  are pure overhead. Conclusions about index structures should not be drawn from
  this dataset.
- **NoIndex fails validation on Zipfian1.0** (`materializeAll mismatch at row 0`)
  because it reorders rows by tier. It is recorded as non-viable rather than
  winning on bytes, which it otherwise would have.
- The lambda ladder never changes the pick within either dataset, so the
  space-versus-time crossover it was built to locate does not appear here. It
  needs a dataset where two types are genuinely close on both axes.

## 2. FPE's full-range decode is ~1.8x its bulk decode

`bench_decode_bulk` and `bench_decode_range` at `(A_frac=0, B_frac=1)` issue the
same access and must agree. RawBitPacked agrees exactly (1.00). Zstd and OpenZL
agree (1.05). `FPE_PerTierBitmaps` does **not**: 1.77× measured directly, 1.36×
and 1.84× in independent runs, consistent in direction across every run.

So `decodeRangeInto(0, N)` on FPE is materially slower than `decodeAllInto`.
`decodeAllPerTierBitmaps` has a dedicated sequential tier walk, while the range
path does per-row index work even when the range covers the whole column. This is
an available optimisation — a full-span range request could dispatch to the bulk
path — and it is deliberately **not** fixed here, because changing a codec while
building the harness that measures it would confound both.

Consequence for the paper: the range driver's full-column cell is not a
substitute for a bulk number. The driver split is load-bearing, not cosmetic.

## 3. The ~2x gather-vs-range penalty was a harness artifact

This suite documented, in the gather driver's header, that codecs not overriding
`decodeRangeInto` read ~2× slower through the gather API at σ=1, attributed to
the caller-owned-buffer contract forcing an extra materialization.

The gap was real but the comparison was not. The old range driver called the
allocating `decodeRange()` and so skipped the copy. With both drivers honouring
the `*Into` contract, both pay the same base-class fallback for the same codecs
and it cancels: RawBitPacked 1.03, FPE_PerTierBitmaps 0.89, Zstd 0.99, OpenZL
0.95. The asymmetry was in the harness, not the API.

## 4. The cache-state axis only means anything for memory-bound codecs

`Raw` on UniformRandom reads 2.0× slower cold than hot while the payload is
L1-resident, and converges to 1.0 once the payload passes the detected 18 MiB
LLC — which is what `defaultWorkingSetTargets()` is built to straddle.

`RawBitPacked` shows **no** crossover at any size (0.90–1.19, unordered): at
~1.2 GB/s the bit-unpacking arithmetic dominates and the prefetcher hides the
payload read entirely.

That is a property of the codec, not a defect of the measurement, and the axis
now makes it visible instead of leaving it assumed. It also means a cold-path
claim must name the codec it applies to.

## 5. Hot must be passive, and eviction costs more than the work

Defining `Hot` as an active read-touch of the whole payload — the initial
design — inverts its own intent for any access narrower than the payload. Once
the payload exceeds L2, touching all of it evicts the measured window's lines out
to LLC, so "hot" is *colder for the measured region* than doing nothing: FPE's
1.6 MB payload against a 1.25 MB L2 read 1.66× slower under active warm, tracking
payload-versus-L2 rather than anything about the access.

The warmup iterations of the real access are the correct warm, and they are
self-correcting: bulk warms the whole payload, a narrow gather warms its window,
a point trace warms the lines it probes.

Separately, eviction costs ~3.8 ms against ~168 µs of bulk decode — 22×. It has
to stay outside the timed window and be reported as its own `evict_ns`. Under
`cold-first-probe` the ratio reaches 1000–4000×.

## 6. SubIntSplit's sections read copies, not the payload

`slice()` copies each section's bytes into decoder-owned storage, and the section
codecs read those copies. So flushing the payload alone leaves everything
actually decoded resident, and `cold-payload` for SubIntSplit would have been
close to meaningless without `Decoder::internalBuffers()`. Observed
`internalBuffers()` sizes: SubIntSplit 7 spans / 1.23 MB; FPE PerTierBitmaps 21 /
212 KB, EliasFano 15 / 833 KB, TierTagArray 14 / 24 KB.

Not yet covered: recursion into section codecs' own `internalBuffers()`, which
needs a virtual on `ISectionCodecIntegral`. The result under-reports, which is the
safe direction — the driver falls back to an LLC thrash and cools that state
anyway, at the cost of also disturbing the TLB.

## 7. Measurement apparatus is a real term at point-lookup scale

The timing-call pair costs 34–80 ns on this box (median ~50), against a smallest
measured `ns_per_probe` of 7.3 ns. Per-probe timing would have inflated a 7 ns
lookup roughly 8×. Batched over 262144 probes the apparatus is ~1e-5 of the
measurement; under `cold-first-probe` a single 0.8–3.5 µs probe still carries
2–6% apparatus, so those rows must be read against the calibration row.

The calibration also drifted 45 → 80 ns between processes on its own, which is
the governor/SMT effect CONVENTIONS 3a warns about, caught by the apparatus
measuring itself.

## 8. Cost-model accuracy, first numbers from the harness

TwitterSnowflake, N=200000, 2000 samples, the default seven-encoding set. The
model ranks well and estimates within about 7%, but picks the true best encoding
for a bit range only about two thirds of the time:

| profile | top-1 | Spearman ρ | mean abs rel err | regret (sample) | regret (extrapolated) |
|---|---|---|---|---|---|
| random | 0.64 | 0.87 | 6.6% | 487 B | 48.7 KB |
| consecutive | 0.71 | 0.86 | 7.0% | 352 B | 45.8 KB |

Building the oracle grid cost 6.6 s against 0.5 s of selection, which is why the
oracle lives in its own driver and not on any measurement path.

The memoization in `CostModelGrid` is confirmed by the driver's own counter:
`metric_compute_calls` is 14 for 14 cells — exactly one `MetricCollector::compute`
per bit range, where the pre-refactor code ran it per segment per caller.

## 9. AutoSIS was not splitting at all, and still degrades with more information

Two defects, one fixed and one open. Together they are why the codec-set ladder
could not produce a result.

### 9a. The entropy prune scale (FIXED)

`entropyPruneThreshold` was compared directly against `MetricCollector`'s
`entropyEstimate`, which is Shannon entropy in **bits per value** -- unbounded, up
to the range's bit width -- while the threshold defaulted to a bare `1.0`. Every
bit range carrying more than one bit of information per value was pruned, which
on real data is nearly all of them.

It failed silently. The full `[0..63]` range is exempt from both prune tests, so
it always survived, `dp[kBits]` was always finite, and the retry gated on
`!isfinite(total_cost)` could never fire. The DP returned a valid single-section
plan and SubIntSplit stopped splitting. `enablePrune` defaults true in all eight
`makeDefaultAutoSubIntSplit*` overloads, so this was AutoSIS as shipped, not a
benchmark artifact.

Fixed by scaling the threshold to the segment width (`entropy > threshold *
bitWidth`, default 0.95) and by firing the retry when pruning leaves only the
full-width segment. **Measured cost of the bug on TwitterSnowflake at N=100000:
709055 to 587598 bytes, 17.1% compression lost.** Pruning is now answer-neutral --
prune on and prune off give byte-identical output and identical plans -- which is
asserted by `Tests/test_subint_plan_agreement.cpp`.

### 9b. Plan quality degrades as the sample grows (OPEN)

With pruning fixed and *held constant*, the DP's plan gets worse the more of the
data it looks at. TwitterSnowflake, N=200000, prune on and off agreeing exactly at
every row:

| samples | segments | bytes | plan |
|---|---|---|---|
| 10000 | 3 | **1177554** | `[0..20]` Dictionary, `[21..52]` FreqPartition, `[53..63]` RunLength |
| 50000 | 3 | 1672038 | `[0..20]` FreqPartition, `[21..53]` BitPacking, `[54..63]` RunLength |
| 100000 | **1** | 1603171 | `[0..63]` FreqPartition |

100000 samples is the default in `makeDefaultAutoSubIntSplitConfig`, so **the
registry's AutoSIS entries produce a single full-width section at 64.13
bits/element** -- worse than raw, and against 40.09 bits/element for the
hand-written `SIS_Snowflake6` plan on the same data. More samples should tighten
an estimate, never invert a decision.

The same inversion appears along a second axis. The ladder allows 23-24 candidate
codecs and collapses to one segment, while the same encoder with the default
seven-codec set splits into three. So a *wider* candidate set also produces a
worse plan.

One mechanism is identified but does not explain everything: `kUniqueCountCap`
= 1<<16 = 65536 (`MetricCollector.hpp:108`) makes `computeEntropy` switch to a
binary lower-bound formula once unique values exceed the cap, so at 100000
samples of near-unique Snowflake IDs the entropy estimate silently becomes a
small number. That covers the 100000-sample collapse. It does not cover the
50000-sample degradation (50000 < 65536), so at least one cost model is
mis-ranking independently of the cap.

**Consequence: AutoSIS figures at the registry default are not usable**, and the
codec-set ladder still cannot express its intended result -- not because the
ladder is wrong, but because the plan it varies does not respond correctly to the
variation. Fixing this is a cost-model investigation, not a benchmark one.

### What the ladder was built to show, and why it needs 9b fixed first

The result being sought is that admitting more random-access codecs improves
compression while keeping gather and point throughput far better than OpenZL and
than admitting all sequential codecs. That rests on an asymmetry:

- **Compression is a sum over sections** -- a codec that wins one bit range reduces
  total size regardless of the rest.
- **Random access is a `min` over sections** -- SubIntSplit advertises `FastSkip`
  only when `allSectionsRandomAccess()` holds, so one sequential section costs the
  whole encoding its gather fast path.

Admitting an RA codec is therefore monotonically safe, while admitting a
sequential one is a trade: it may win a high-entropy section and in exchange the
entire encoding loses random access. That discontinuity is the finding, and it
lives at the **per-section assignment** level. It is invisible while every plan is
a single 64-bit section, because there is then no per-section specialisation and
no meaningful `min` -- which is exactly the state 9a and 9b leave the DP in.

## 10. Reproducibility defects found in existing code

- Five generators in `CommonGenerators.hpp` seeded from `std::random_device` in
  **both** the constructor and `reset()`. Two processes produced different
  streams, so a repeated sweep could not be compared against itself; worse,
  `reset()` changed the stream mid-process, so a harness that reset between
  encoders handed each encoder different data and then compared their sizes.
  Now a pure function of an explicit seed, with `reset()` rewinding.
- `makeDefaultAutoSubIntSplit*` hardcoded a repo-relative `costGridCsvPath` and
  the selector **throws** if it cannot open it, so AutoSIS only worked from one
  specific working directory.
- `SubIntSplitEncoder::verboseEnabled()` returned `v || true`, making the
  environment check dead and logging unconditional.
- `BenchmarkRunner` seeded from `random_device` while its memory pass used a
  fixed `mt19937(42)`, so the timing and memory passes measured different ranges.
- `sweep_subint_samples.cpp` and `reordering_benchmarks.cpp` carry absolute
  dataset paths that no longer resolve; both throw before doing any work.
- `plot_heatmap.py` imported `matplotlib.cm.get_cmap` (removed in 3.9) and
  `scipy` at module scope, so it failed against its own pre-existing CSV.
