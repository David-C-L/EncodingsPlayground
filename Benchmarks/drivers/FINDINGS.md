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

## 1a. TierTagArray is Pareto-dominated by EliasFano

With `index_bytes` now published as metadata (it was previously computed only
inside the encoder's verbose-logging block, so the one number this comparison
needs was unavailable unless the encoder was also writing to stderr), the space
side can be put next to finding 1's latency side. Zipfian, N=100000:

| index type | payload bytes | vs NoIndex | index bytes | index % | hot uniform ns/probe |
|---|---|---|---|---|---|
| NoIndex | 353231 | — | 0 | 0.0% | 50 (no positional index to pay for) |
| TierTagArray | 403183 | +14.1% | 50000 | 12.4% | 723 |
| EliasFano | 406361 | +15.0% | 53178 | 13.1% | 79 |
| PerTierBitmaps | 503231 | +42.5% | 150048 | 29.8% | 73 |

**TierTagArray is dominated on both axes**: it costs essentially the same index
space as EliasFano (12.4% vs 13.1% of payload) and is 9x slower per probe. There
is no λ in a `bytes + λ·time` objective at which it wins, so it should not appear
on the Pareto frontier the index-oracle driver reports.

The genuine trade-off is the other pair: PerTierBitmaps buys 1.08x the point
latency of EliasFano for 2.8x the index bytes (29.8% vs 13.1% of payload). That is
the choice a cost model for this dimension would have to make, and it is a narrow
one — which is itself worth reporting, since it bounds how much such a model could
possibly gain.

Caveat: these are single-dataset, single-size numbers taken under the conditions
at the top of this file. The ranking is robust (9x is far outside any observed
spread); the percentages are not final.

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

## 8. Reproducibility defects found in existing code

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
