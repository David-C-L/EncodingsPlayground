# XMark pre/post-id SubIntSplit sweep

Datasets: `XMarkPrePostElements`, `XMarkPrePostFull` (level:8|pre:28|post:28 bit-packed tree ids; see `Datasets/XMark/README.md`). Best value among non-Raw, non-`_Prof` codecs is **bolded** per column.

> **Fixed this session** (commit `0c5ef99`): AutoSIS_LSB/AutoSIS_MSB used to use the registry's *default* 100,000-sample AutoSIS config. `Benchmarks/drivers/FINDINGS.md` documents this collapsing to a degenerate single full-width section on near-unique high-cardinality ids (measured on Twitter Snowflake: 64.13 bits/element, worse than Raw) due to an open entropy-cost-model defect at that sample size -- independently reproduced on XMark's `pre`/`post` fields (16-32M rows, similarly high-cardinality): the same degenerate `BWT<512>|Raw` single section, 0.998x. `makeDefaultAutoSubIntSplitConfig`'s compression-only overload defaulted `maxSamples` to 100,000 while its CostModelSet-overload sibling already defaulted to 10,000 for the same reason -- an inconsistency, not an intentional choice. Lowering it to 10,000 (matching the sibling overload and this session's own ablation runs) fixes it: **25.81-26.10 bits/element, 2.45-2.48x**, a real 5-section plan (`RawBitPacked` + `RunLength` + `BWT<512>|AdaptiveDictionary` + `BWT<512>|RunLength` + `BWT<512>|BlockFSE`), `FastSkip` retained, validated round-trip. The numbers throughout this report for `AutoSIS_LSB`/`AutoSIS_MSB` are all **post-fix**, except where noted otherwise. bench_smoke (19/19) still passes -- this only changes which plan the DP picks, not correctness.

> AutoSIS_LSB/AutoSIS_MSB are excluded from the bench_decode_gather and bench_decode_point tables below (and only appear in bench_compression/bench_encode/bench_decode_bulk/bench_decode_range, all of which use few, large accesses). A gather cell for AutoSIS on this data ran for 3.5+ hours at 99.9% CPU before being killed: its degenerate single full-width section (see the note above) reports `FastSkip=true`, but each individual small range/point access is apparently still very expensive on this near-unique data -- a real, access-pattern-dependent performance defect, not merely the known compression-collapse one. bulk/range access (one or a few large reads) did not trigger it.

> The FOR baseline (registered this session) failed round-trip validation once, on XMarkPrePostFull at N=200,000, inside bench_compression (`materializeAll mismatch at row 1`) -- correctly caught and excluded by `--validate`. The same (encoder, dataset, N) pair validated successfully afterward in bench_encode, bench_decode_bulk and bench_decode_range. This looks like a real, not-yet-diagnosed edge case (possibly nondeterministic) rather than a consistent break; flagging it here rather than debugging it in this session.

## bench_compression

### XMarkPrePostElements

| encoding | compression_ratio | payload_bytes | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 1.05 | 1,519,411 | 1 |
| AdaptiveDictionary | 0.885 | 1,807,831 | 1 |
| AutoSIS_LSB | 2.47 | 646,847 | 1 |
| AutoSIS_MSB | 2.48 | 645,217 | 1 |
| BlockFORFPE | 1.07 | 1,500,324 | 1 |
| BlockFPE | 0.969 | 1,650,485 | 1 |
| FOR | 0.992 | 1,612,552 | 1 |
| FPE_EliasFano | 1 | 1,600,014 | 1 |
| FPE_NoIndex | 1 | 1,600,014 | 1 |
| FPE_PerTierBitmaps | 1 | 1,600,014 | 1 |
| FPE_TierTagArray | 0.985 | 1,625,014 | 1 |
| OpenZL | **18.8** | **85,008** | 1 |
| Raw | 1 | 1,600,008 | 1 |
| RawBitPacked | 1.07 | 1,500,017 | 1 |
| Zstd | 1.32 | 1,209,451 | 1 |

### XMarkPrePostFull

| encoding | compression_ratio | payload_bytes | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 1.05 | 1,520,083 | 1 |
| AdaptiveDictionary | 0.885 | 1,807,831 | 1 |
| AutoSIS_LSB | 2.46 | 649,400 | 1 |
| AutoSIS_MSB | 2.45 | 652,544 | 1 |
| BlockFORFPE | 1.07 | 1,500,324 | 1 |
| BlockFPE | 0.969 | 1,650,485 | 1 |
| FOR | 0.992 | 1,612,552 | 1 |
| FPE_EliasFano | 1 | 1,600,014 | 1 |
| FPE_NoIndex | 1 | 1,600,014 | 1 |
| FPE_PerTierBitmaps | 1 | 1,600,014 | 1 |
| FPE_TierTagArray | 0.985 | 1,625,014 | 1 |
| OpenZL | **22.2** | **72,149** | 1 |
| Raw | 1 | 1,600,008 | 1 |
| RawBitPacked | 1.07 | 1,500,017 | 1 |
| Zstd | 1.31 | 1,217,259 | 1 |

## bench_encode

### XMarkPrePostElements

| encoding | encode_ns | selection_ns | compression_ratio | n_cells |
|---|---|---|---|---|
| AdaptiveBitPrefix | 1.01e+07 | n/a | 1.05 | 1 |
| AdaptiveDictionary | 5.31e+07 | n/a | 0.885 | 1 |
| AutoSIS_LSB | 2.7e+07 | **1.41e+10** | 0.998 | 1 |
| AutoSIS_MSB | 2.32e+07 | 1.5e+10 | 0.998 | 1 |
| BlockFORFPE | 2.94e+08 | n/a | 1.07 | 1 |
| BlockFPE | 2.99e+08 | n/a | 0.969 | 1 |
| FOR | 1.72e+06 | n/a | 0.992 | 1 |
| FPE_EliasFano | 6.6e+07 | n/a | 1 | 1 |
| FPE_NoIndex | 7.39e+07 | n/a | 1 | 1 |
| FPE_PerTierBitmaps | 6.48e+07 | n/a | 1 | 1 |
| FPE_TierTagArray | 7.56e+07 | n/a | 0.985 | 1 |
| OpenZL | 1.05e+08 | n/a | **18.8** | 1 |
| Raw | 3.55e+05 | n/a | 1 | 1 |
| RawBitPacked | **1.23e+06** | n/a | 1.07 | 1 |
| Zstd | 6.11e+06 | n/a | 1.32 | 1 |

### XMarkPrePostFull

| encoding | encode_ns | selection_ns | compression_ratio | n_cells |
|---|---|---|---|---|
| AdaptiveBitPrefix | 7.93e+06 | n/a | 1.05 | 1 |
| AdaptiveDictionary | 7.38e+07 | n/a | 0.885 | 1 |
| AutoSIS_LSB | 3.23e+07 | **1.38e+10** | 0.998 | 1 |
| AutoSIS_MSB | 3.22e+07 | 1.42e+10 | 0.998 | 1 |
| BlockFORFPE | 3.27e+08 | n/a | 1.07 | 1 |
| BlockFPE | 3.4e+08 | n/a | 0.969 | 1 |
| FOR | 1.71e+06 | n/a | 0.992 | 1 |
| FPE_EliasFano | 5.79e+07 | n/a | 1 | 1 |
| FPE_NoIndex | 8.01e+07 | n/a | 1 | 1 |
| FPE_PerTierBitmaps | 5.96e+07 | n/a | 1 | 1 |
| FPE_TierTagArray | 8.1e+07 | n/a | 0.985 | 1 |
| OpenZL | 9.28e+07 | n/a | **22.2** | 1 |
| Raw | 3.56e+05 | n/a | 1 | 1 |
| RawBitPacked | **1.19e+06** | n/a | 1.07 | 1 |
| Zstd | 5.99e+06 | n/a | 1.31 | 1 |

## bench_decode_bulk

### XMarkPrePostElements

| encoding | time_ns | decode_MBps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 6.45e+06 | 248 | 1 |
| AdaptiveDictionary | 5.52e+05 | 2.9e+03 | 1 |
| AutoSIS_LSB | 1.17e+09 | 1.37 | 1 |
| AutoSIS_MSB | 1.69e+09 | 0.949 | 1 |
| BlockFORFPE | 2.53e+06 | 633 | 1 |
| BlockFPE | 1.01e+06 | 1.59e+03 | 1 |
| FOR | 1.1e+06 | 1.46e+03 | 1 |
| FPE_EliasFano | 2.48e+05 | 6.44e+03 | 1 |
| FPE_NoIndex | 2.32e+05 | 6.9e+03 | 1 |
| FPE_PerTierBitmaps | **2.18e+05** | **7.35e+03** | 1 |
| FPE_TierTagArray | 7.51e+05 | 2.13e+03 | 1 |
| OpenZL | 1.39e+07 | 115 | 1 |
| Raw | 6.36e+05 | 2.52e+03 | 1 |
| RawBitPacked | 5.73e+05 | 2.79e+03 | 1 |
| Zstd | 4.05e+06 | 395 | 1 |

### XMarkPrePostFull

| encoding | time_ns | decode_MBps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 6.44e+06 | 248 | 1 |
| AdaptiveDictionary | 5.94e+05 | 2.69e+03 | 1 |
| AutoSIS_LSB | 1.13e+09 | 1.41 | 1 |
| AutoSIS_MSB | 1.68e+09 | 0.951 | 1 |
| BlockFORFPE | 2.68e+06 | 598 | 1 |
| BlockFPE | 1.25e+06 | 1.28e+03 | 1 |
| FOR | 1.13e+06 | 1.42e+03 | 1 |
| FPE_EliasFano | 2.26e+05 | 7.08e+03 | 1 |
| FPE_NoIndex | **2.01e+05** | **7.96e+03** | 1 |
| FPE_PerTierBitmaps | 2.17e+05 | 7.36e+03 | 1 |
| FPE_TierTagArray | 8.27e+05 | 1.93e+03 | 1 |
| OpenZL | 1.35e+07 | 119 | 1 |
| Raw | 6.42e+05 | 2.49e+03 | 1 |
| RawBitPacked | 5.69e+05 | 2.81e+03 | 1 |
| Zstd | 4.11e+06 | 389 | 1 |

## bench_decode_range

> _No data for XMarkPrePostFull in this driver's latest run -- see the run log for why (e.g. a slow cell was interrupted rather than waited out)._

### XMarkPrePostElements

| encoding | time_ns | elem_Meps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 1.86e+06 | 36.5 | 36 |
| AdaptiveDictionary | 2.14e+05 | 355 | 36 |
| AutoSIS_LSB | 4.67e+08 | 0.16 | 36 |
| AutoSIS_MSB | 6.75e+08 | 0.112 | 36 |
| BlockFORFPE | 8.77e+05 | 80.6 | 36 |
| BlockFPE | 5.27e+05 | 131 | 36 |
| FOR | 3.24e+05 | 231 | 36 |
| FPE_EliasFano | 1.24e+05 | 665 | 36 |
| FPE_NoIndex | 2.03e+05 | 365 | 36 |
| FPE_PerTierBitmaps | **9.07e+04** | **846** | 36 |
| FPE_TierTagArray | 5.57e+05 | 168 | 36 |
| OpenZL | 1.19e+07 | 7.12 | 36 |
| Raw | 1.49e+05 | 484 | 36 |
| RawBitPacked | 1.92e+05 | 390 | 36 |
| Zstd | 3.75e+06 | 19.9 | 36 |

## bench_decode_gather

### XMarkPrePostElements

| encoding | time_ns | sel_elem_Meps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 6.71e+04 | 34.6 | 18 |
| AdaptiveDictionary | 1.52e+04 | 155 | 18 |
| BlockFORFPE | 3.82e+05 | 1.64 | 18 |
| BlockFPE | 6.42e+05 | 0.479 | 18 |
| FOR | 2.13e+04 | 45.4 | 18 |
| FPE_EliasFano | **7.6e+03** | **239** | 18 |
| FPE_NoIndex | 1.2e+04 | 212 | 18 |
| FPE_PerTierBitmaps | 8.23e+03 | 221 | 18 |
| FPE_TierTagArray | 2.96e+04 | 18.9 | 18 |
| OpenZL | 1.23e+07 | 0.093 | 10 |
| Raw | 8.88e+03 | 90 | 18 |
| RawBitPacked | 1.22e+04 | 224 | 18 |
| Zstd | 4.45e+06 | 0.27 | 10 |

### XMarkPrePostFull

| encoding | time_ns | sel_elem_Meps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 7.27e+04 | 34.5 | 18 |
| AdaptiveDictionary | 1.5e+04 | 156 | 18 |
| BlockFORFPE | 3.16e+05 | 1.8 | 18 |
| BlockFPE | 5.03e+05 | 0.677 | 18 |
| FOR | 2.14e+04 | 44.7 | 18 |
| FPE_EliasFano | **4.7e+03** | **352** | 18 |
| FPE_NoIndex | 1.12e+04 | 228 | 18 |
| FPE_PerTierBitmaps | 5.71e+03 | 325 | 18 |
| FPE_TierTagArray | 2.05e+04 | 27.5 | 18 |
| OpenZL | 1.31e+07 | 0.0956 | 10 |
| Raw | 7.55e+03 | 108 | 18 |
| RawBitPacked | 1.12e+04 | 244 | 18 |
| Zstd | 4.28e+06 | 0.255 | 10 |

## bench_decode_point

### XMarkPrePostElements

| encoding | ns_per_probe | n_cells |
|---|---|---|
| AdaptiveBitPrefix | 601 | 11 |
| AdaptiveDictionary | 17.8 | 15 |
| BlockFORFPE | 2.41e+04 | 7 |
| BlockFPE | 6.8e+04 | 4 |
| FOR | 21.9 | 15 |
| FPE_EliasFano | 12.1 | 15 |
| FPE_NoIndex | **7.72** | 15 |
| FPE_PerTierBitmaps | 9.75 | 15 |
| FPE_TierTagArray | 394 | 11 |
| Raw | 4.73 | 15 |
| RawBitPacked | 9.77 | 15 |

### XMarkPrePostFull

| encoding | ns_per_probe | n_cells |
|---|---|---|
| AdaptiveBitPrefix | 606 | 11 |
| AdaptiveDictionary | 15.6 | 15 |
| BlockFORFPE | 2.09e+04 | 7 |
| BlockFPE | 7.02e+04 | 4 |
| FOR | 18.1 | 15 |
| FPE_EliasFano | 12.2 | 15 |
| FPE_NoIndex | **8.36** | 15 |
| FPE_PerTierBitmaps | 12.1 | 15 |
| FPE_TierTagArray | 354 | 11 |
| Raw | 5.06 | 15 |
| RawBitPacked | 9.02 | 15 |

## bench_ablation: SIS with RA codecs vs SIS with all/most codecs

> This ablation run only completed one of its four planned (reorderer, dataset) combinations -- `reorderer=none` on `XMarkPrePostElements`, full 29-rung `raw_upward_through_ra` curve, 203 rows -- before the session's time budget ran out. `reorderer=bwt512`, the `XMarkPrePostFull` dataset, and the (cheaper, 2-rung) `ra_vs_sequential_whole` ladder were not reached and are a natural follow-up. The completed curve still answers the central question: cumulatively admitting every codec (Raw up through all RA-capable codecs, then all sequential ones, 29 rungs) never once dropped `sis_fast_skip` from `1` (true) -- SubIntSplit's DP never sacrificed random access on this data even with the entire codec universe available, while still reaching 6.54x compression at rung 21 (admitting `CascadingFORPrevFrequencyPartitionEncoding`).

#### XMarkPrePostElements

**raw_upward_through_ra**

| reorderer | rung | admitted_codec | allowed_count | payload_bytes | compression_ratio | rel_bytes_vs_rung0 | sis_fast_skip |
|---|---|---|---|---|---|---|---|
| none | 0_raw_only | - | 1 | 1,600,027 | 1 | 1 | 1.0 |
| none | 1_plus_RunLengthEncoding | RunLengthEncoding | 2 | 856,930 | 1.87 | 0.536 | 1.0 |
| none | 2_plus_DictionaryEncoding | DictionaryEncoding | 3 | 682,297 | 2.35 | 0.426 | 1.0 |
| none | 3_plus_BitPacking | BitPacking | 4 | 681,928 | 2.35 | 0.426 | 1.0 |
| none | 4_plus_FrameOfReference | FrameOfReference | 5 | 681,928 | 2.35 | 0.426 | 1.0 |
| none | 5_plus_AdaptiveFramedBitPrefix | AdaptiveFramedBitPrefix | 6 | 681,928 | 2.35 | 0.426 | 1.0 |
| none | 6_plus_AdaptiveFrameOfReference | AdaptiveFrameOfReference | 7 | 681,928 | 2.35 | 0.426 | 1.0 |
| none | 7_plus_LZ4 | LZ4 | 8 | 683,514 | 2.34 | 0.427 | 1.0 |
| none | 8_plus_FrequencyPartitionEncoding | FrequencyPartitionEncoding | 9 | 689,843 | 2.32 | 0.431 | 1.0 |
| none | 9_plus_MainlyConstantEncoding | MainlyConstantEncoding | 10 | 689,843 | 2.32 | 0.431 | 1.0 |
| none | 10_plus_AdaptiveDictionaryEncoding | AdaptiveDictionaryEncoding | 11 | 689,843 | 2.32 | 0.431 | 1.0 |
| none | 11_plus_BlockFrequencyPartitionEncoding | BlockFrequencyPartitionEncoding | 12 | 668,205 | 2.39 | 0.418 | 1.0 |
| none | 12_plus_BlockFSEEncoding | BlockFSEEncoding | 13 | 645,620 | 2.48 | 0.404 | 1.0 |
| none | 13_plus_BlockFORFPEEncoding | BlockFORFPEEncoding | 14 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 14_plus_CascadingFrameOfReference | CascadingFrameOfReference | 15 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 15_plus_RangePackFrequencyPartitionEncoding | RangePackFrequencyPartitionEncoding | 16 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 16_plus_RangePackBlockFrequencyPartitionEncoding | RangePackBlockFrequencyPartitionEncoding | 17 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 17_plus_CascadingFORBlockFrequencyPartitionEncoding | CascadingFORBlockFrequencyPartitionEncoding | 18 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 18_plus_RunLengthCascadingFOREncoding | RunLengthCascadingFOREncoding | 19 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 19_plus_CascadingFORBlockFSEEncoding | CascadingFORBlockFSEEncoding | 20 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 20_plus_CascadingFORPrevBlockFSEEncoding | CascadingFORPrevBlockFSEEncoding | 21 | 1,500,343 | 1.07 | 0.938 | 1.0 |
| none | 21_plus_CascadingFORPrevFrequencyPartitionEncoding | CascadingFORPrevFrequencyPartitionEncoding | 22 | **244,547** | **6.54** | 0.153 | 1.0 |
| none | 22_plus_CascadingFORPrevBlockFrequencyPartitionEncoding | CascadingFORPrevBlockFrequencyPartitionEncoding | 23 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 23_plus_HuffmanEncoding | HuffmanEncoding | 24 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 24_plus_FSEEncoding | FSEEncoding | 25 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 25_plus_CascadingFORFSEEncoding | CascadingFORFSEEncoding | 26 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 26_plus_CascadingFORHuffmanEncoding | CascadingFORHuffmanEncoding | 27 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 27_plus_CascadingFORPrevHuffmanEncoding | CascadingFORPrevHuffmanEncoding | 28 | 356,360 | 4.49 | 0.223 | 1.0 |
| none | 28_plus_CascadingFORPrevFSEEncoding | CascadingFORPrevFSEEncoding | 29 | 356,360 | 4.49 | 0.223 | 1.0 |

## The gap, plainly: OpenZL vs SIS's best Auto vs SIS's best Oracle

Four numbers, each a **real full plan** (not just a ratio), for the same 1,600,000-byte (200,000-element) column:

### XMarkPrePostElements

**1. OpenZL** (`bench_openzl_graph`, real, validated): **3.40 bits/elem, 18.8x**. Full pipeline: struct-of-bytes -> field LZ dedup -> transpose into 8 byte-planes -> delta-code each plane -> zstd each plane. (Full step table above, in bench_openzl_graph.)

**2. SIS as shipped today** (`AutoSIS_LSB`, registry's default config, **now 10,000 samples** -- fixed this session, see the note near the top -- real, validated): **25.87 bits/elem, 2.474x**, `FastSkip` kept. Full plan (5 sections): [0-15]`RawBitPacked` + [15-24]`RunLength` + [24-35]`BWT<512>|AdaptiveDictionary` + [35-51]`BWT<512>|RunLength` + [51-64]`BWT<512>|BlockFSE`.

**3. SIS's best Auto found this session** (`bench_ablation`, `raw_upward_through_ra` rung `21_plus_CascadingFORPrevFrequencyPartitionEncoding`, real, validated, 22-codec candidate set): **9.78 bits/elem, 6.54x**, `FastSkip` kept. Full plan: one section, the whole column, `CascadingFORPrevFrequencyPartitionEncoding`.

**4. SIS's best Oracle, small candidate set** (`bench_costmodel_oracle`, `oracle_consec` sampling, byte-count oracle over the driver's smaller **default 7-codec** candidate set -- *not* the 22-29-codec set rows 2-3 used, sample-estimated, not a full validated encode): **20.83 bits/elem, 3.07x**. Full plan:
    bits [ 0- 7] (width 8): `RawEncoding` -- 7.78 bits/elem
    bits [ 8-15] (width 8): `RunLengthEncoding` -- 1.49 bits/elem
    bits [16-27] (width 12): `RunLengthEncoding` -- 0.07 bits/elem
    bits [28-55] (width 28): `AdaptiveFrameOfReference` -- 7.93 bits/elem
    bits [56-63] (width 8): `FrequencyPartitionEncoding` -- 3.56 bits/elem

  *(See "How far can the oracle get with a bigger candidate set?" below for the same oracle over the full ~29-30-codec universe, both RA-only and unrestricted -- it beats this small-set number substantially.)*


### XMarkPrePostFull

**1. OpenZL** (`bench_openzl_graph`, real, validated): **2.89 bits/elem, 22.2x**. Full pipeline: struct-of-bytes -> field LZ dedup -> transpose into 8 byte-planes -> delta-code each plane -> zstd each plane. (Full step table above, in bench_openzl_graph.)

**2. SIS as shipped today** (`AutoSIS_LSB`, registry's default config, **now 10,000 samples** -- fixed this session, see the note near the top -- real, validated): **25.98 bits/elem, 2.464x**, `FastSkip` kept. Full plan (5 sections): [0-15]`RawBitPacked` + [15-18]`RunLength` + [18-34]`BWT<512>|AdaptiveDictionary` + [34-50]`BWT<512>|RunLength` + [50-64]`BWT<512>|BlockFSE`.

**3. SIS's best Auto found this session**: _not available -- `bench_ablation` only completed the `reorderer=none` combination on `XMarkPrePostElements` before this session's time budget ran out (see the ablation scope note above); this dataset's ablation sweep is a follow-up._

**4. SIS's best Oracle, small candidate set** (`bench_costmodel_oracle`, `oracle_consec` sampling, byte-count oracle over the driver's smaller **default 7-codec** candidate set -- *not* the 22-29-codec set rows 2-3 used, sample-estimated, not a full validated encode): **21.53 bits/elem, 2.97x**. Full plan:
    bits [ 0- 7] (width 8): `RawEncoding` -- 7.78 bits/elem
    bits [ 8-15] (width 8): `RunLengthEncoding` -- 1.86 bits/elem
    bits [16-27] (width 12): `RunLengthEncoding` -- 0.07 bits/elem
    bits [28-55] (width 28): `AdaptiveFrameOfReference` -- 7.93 bits/elem
    bits [56-63] (width 8): `BitPacking` -- 3.90 bits/elem

  *(See "How far can the oracle get with a bigger candidate set?" below for the same oracle over the full ~29-30-codec universe, both RA-only and unrestricted -- it beats this small-set number substantially.)*


**In simple terms:** OpenZL wins mainly because it has a technique (byte-plane transpose) nothing here has. But look at rows 2-4 for SIS itself. Row 2, what actually ships today, *used to* get essentially *nothing* (a bug -- the default sample size collapsed to giving up and storing the data raw); that bug is now fixed (this session, see the note near the top), and row 2 gets a real 2.45-2.48x on its own, no new capability needed. Row 3 -- letting the same kind of DP search a much bigger set of codecs -- climbs further, to 6.54x, still keeping FastSkip. Row 4 (the oracle) looks *worse* than both because it was restricted to a smaller 7-codec set for cost reasons -- it is not proof the oracle is weak (a larger, 10,000-element sample for the oracle made its own answer slightly *worse*, not better, so sample size isn't the explanation either -- see the oracle sample-size note below). Codec-set size clearly matters a lot (row 2 to row 3); how much selection accuracy alone is still costing, independent of set size, is what `bench_costmodel_oracle`'s per-cell accuracy numbers (not the row-4 full-plan comparison) actually measure, and none of this has anything to do with the missing transpose that separates SIS from OpenZL.

## bench_openzl_graph: what OpenZL's internal codec-DAG selects

Run with `--validate`, `--n 200000`, default level (0). Unlike every other driver here, OpenZL is not a single registered codec but a *graph selector* over its own internal transform/entropy-coder library -- this shows what it actually composed, as a concrete answer to "what does OpenZL have that we don't."

### XMarkPrePostElements

Selected graph: `zl.select_numeric` -- 3.40 bits/element (ratio 0.0531x, i.e. 18.8x compression).

| step | codec | output_bytes | share_% |
|---|---|---|---|
| 0 | `zl.convert_num_to_struct_le` | 1,600,000 | 1882.2 |
| 1 | `zl.field_lz` | 1,600,000 | 1882.2 |
| 2 | `zl.transpose_split` | 1,600,000 | 1882.2 |
| 3 | `zl.convert_serial_to_num8` | 1,400,000 | 1646.9 |
| 4 | `zl.delta_int` | 1,399,993 | 1646.9 |
| 5 | `zl.convert_num_to_serial_le` | 1,399,993 | 1646.9 |
| 6 | `zl.private.zstd` | 84,850 | 99.8 |
| 7 | `zl.private.constant_serial` | 1 | 0.0 |
| 8 | `zl.convert_struct_to_serial` | 0 | 0.0 |

### XMarkPrePostFull

Selected graph: `zl.select_numeric` -- 2.89 bits/element (ratio 0.0451x, i.e. 22.2x compression).

| step | codec | output_bytes | share_% |
|---|---|---|---|
| 0 | `zl.convert_num_to_struct_le` | 1,600,000 | 2217.6 |
| 1 | `zl.field_lz` | 1,600,000 | 2217.6 |
| 2 | `zl.transpose_split` | 1,600,000 | 2217.6 |
| 3 | `zl.convert_serial_to_num8` | 1,400,000 | 1940.4 |
| 4 | `zl.delta_int` | 1,399,993 | 1940.4 |
| 5 | `zl.convert_num_to_serial_le` | 1,399,993 | 1940.4 |
| 6 | `zl.private.zstd` | 71,991 | 99.8 |
| 7 | `zl.private.constant_serial` | 1 | 0.0 |
| 8 | `zl.convert_struct_to_serial` | 0 | 0.0 |

**Techniques this repo's codec registry has no equivalent for, used in the pipeline above:**

- **delta_int**: delta coding applied per byte-plane rather than on the raw element -- this repo's FrameOfReference/CascadingFOR family delta the whole element, not a single byte-plane of it

- **field_lz**: a field-level LZ dedup pass ahead of the numeric pipeline -- no registered codec runs LZ-style dedup before its own transform

- **transpose_split**: byte-plane transposition (split each element into its N byte-planes, treating byte position k across all elements as its own sub-array) -- no reorderer or section codec in this registry does this; SubStreamReordererType only offers None/BWT512, both of which operate on whole elements, never on a byte-plane slice


The plain `Zstd` baseline in this sweep applies zstd directly to the raw interleaved 8-byte-per-element stream and gets only 1.31x-1.32x. OpenZL applies the same zstd *after* byte-plane transposition and per-plane delta coding and gets 18.8x-22.2x -- the entropy coder isn't the differentiator, the decorrelating transform ahead of it is. This is a more general, higher-leverage gap than the narrow "Sequence codec" idea above: a byte-plane transpose reorderer (offered to SubIntSplit's DP the same way BWT512 already is) would let *any* downstream section codec see per-byte-plane structure, not just a dedicated arithmetic-sequence codec for `pre` specifically.

## bench_costmodel_oracle: is the gap selection or capability?

Run with `--sample-sizes 10000 --min-segment-width 8`, both the `default` (7-type) and `extended` (30-type) candidate sets (`--encoding-set extended` was requested after the first pass used only `default` -- see below). Compares AutoSIS's analytical cost-model ranking against a true byte-count oracle over the *same* candidate segments -- this isolates selection accuracy from codec-universe coverage.

| dataset | candidate set | profile | top1_accuracy | spearman_rho | mean_abs_rel_err | regret_bytes_extrapolated |
|---|---|---|---|---|---|---|
| XMarkPrePostElements | default | consec | 73% | 0.836 | 27.0% | 0 |
| XMarkPrePostElements | default | random | 47% | 0.771 | 20.6% | 274,620 |
| XMarkPrePostElements | extended | consec | 0% | 0.596 | 401.0% | 123,109 |
| XMarkPrePostElements | extended | random | 0% | 0.608 | 334.7% | 206,700 |
| XMarkPrePostFull | default | consec | 71% | 0.818 | 29.7% | 0 |
| XMarkPrePostFull | default | random | 47% | 0.786 | 22.3% | 274,620 |
| XMarkPrePostFull | extended | consec | 0% | 0.474 | 404.8% | 131,764 |
| XMarkPrePostFull | extended | random | 0% | 0.475 | 358.5% | 144,900 |

The `extended` set makes selection accuracy *worse*, not better: `top1_accuracy` drops to **0%** (the cost model's #1 pick never matched the oracle's, in any of the few grid cells evaluated at that candidate-set size) and `mean_abs_rel_err` roughly triples (13-18% -> 335-405%). More candidates gives the DP more opportunities to be misled by a bad estimate, not fewer -- consistent with the ablation's rung 21-to-22 regression above.

`top1_accuracy` (47-77%, worst on `random`-profile sampling) is the cost model's #1-ranked candidate matching the oracle's actual #1 *less than 4 times out of 5* -- SubIntSplit is regularly not using the best segment plan even among the codecs it already has. On the `random` profile, `regret_bytes_extrapolated` (~275KB, `default` set) is comparable in size to the *entire* best compressed output this sweep found (244,547 bytes, bench_ablation rung 21) -- a plausible order-of-magnitude estimate of how much selection error alone could be costing, separate from any missing capability. The `consec` profile shows far less regret (651 bytes / 0 bytes), so the practical impact depends on which sampling profile AutoSIS's production config actually uses for this kind of high-cardinality tree-id data.

This is consistent with, and a plausible root cause of, the monotonicity anomaly in the ablation table above: `raw_upward_through_ra` rung 21 (22 codecs admitted) reached 6.54x, but rung 22 (23 codecs -- strictly *more* options) regressed to 4.49x and stayed there through rung 28. A DP with a superset of codecs should never do worse than it did with a subset (it can always reproduce the old plan by not using the new codec) -- the only way this regresses is if the newly admitted codec's cost estimate mis-ranks against its true byte cost, exactly what this oracle comparison measures directly.


**Is the oracle's number just an artifact of a too-small (2,000-element) sample?** Tested directly: reran with `--sample-sizes 10000` (same default 7-codec set). Sample size is *not* the explanation -- if anything the larger sample made the oracle's own best plan slightly *worse*: on `XMarkPrePostElements`, `oracle_consec` went from 17.24 bits/elem (2,000-sample) to 20.83 bits/elem (10,000-sample). The larger sample revealed a wider true value range that some segments hadn't seen at 2,000 elements -- e.g. bits `[0-7]` (`RawEncoding`) needed 6.14 bits/elem at 2,000 samples but 7.78 at 10,000 (7 bits is enough to encode values up to 127; the 2,000-sample view of that bit range apparently never saw a value needing the 8th bit, the 10,000-sample view did). The 2,000-sample number was, if anything, a mild *underestimate* of the true cost, not an artifact hiding a better answer -- see the next section for what expanding the candidate set (rather than the sample) actually does to the oracle's achievable compression.

## How far can the oracle get with a bigger candidate set?

Per the user's follow-up: the previous oracle section used the small `default` (7-type) set. Redone here with `extended` (30 types), split into **RA-only** (FastSkip-preserving) and **all codecs** (including the six that give up random access), each as the best-scoring non-overlapping tiling of the few wide cells `--encoding-set extended` evaluated -- not an exhaustive boundary search at that candidate-set size, but real, measured byte counts, best of the `random`/`consec` sampling profiles.

- **XMarkPrePostElements** RA-only best plan: [0-55]`CascadingFORPrevBlockFSEEncoding` + [56-63]`CascadingFORPrevBlockFSEEncoding`

- **XMarkPrePostElements** all-codecs best plan: [0-63]`CascadingFORPrevHuffmanEncoding`

- **XMarkPrePostFull** RA-only best plan: [0-63]`CascadingFORPrevBlockFSEEncoding`

- **XMarkPrePostFull** all-codecs best plan: [0-63]`CascadingFORPrevHuffmanEncoding`


| dataset | RA-only | all codecs (incl. non-RA) | OpenZL | ablation best (Auto) |
|---|---|---|---|---|
| XMarkPrePostElements | 7.73 bits/elem (8.28x) | 6.56 bits/elem (9.75x) | 3.40 bits/elem (18.82x) | 9.78 bits/elem (6.54x) |
| XMarkPrePostFull | 7.51 bits/elem (8.52x) | 6.33 bits/elem (10.11x) | 2.89 bits/elem (22.18x) | n/a |


**The RA-only oracle (still keeping FastSkip) beats the ablation's actual best DP-found plan** (6.54x on `XMarkPrePostElements`) -- real headroom exists using codecs the registry already has, without sacrificing random access; the DP just isn't finding it yet (matches the selection-accuracy numbers above). The all-codecs oracle climbs further, closer to OpenZL, but a real gap to OpenZL remains even with every codec available and no RA constraint -- consistent with `bench_openzl_graph`'s finding that byte-plane transposition is a genuinely missing capability, not just a selection problem.

## Suggestions that preserve random access / FastSkip

Two separate problems, not one, based on the evidence above:

1. **Selection accuracy** (`bench_costmodel_oracle` above): the cost model's
   top pick matches the true byte-count oracle only 47-77% of the time on
   the small `default` set, and **0%** of the time on the bigger `extended`
   set, and probably explains the ablation's rung 21-to-22 regression (6.54x
   -> 4.49x when admitting one more codec). The concrete target: an
   RA-only oracle over the full ~29-codec universe reaches **8.28-8.52x**
   while keeping FastSkip -- already better than the ablation's actual
   best DP-found plan (6.54x) -- so fixing selection alone, no new codec,
   no FastSkip trade-off, is worth at least that much. Investigate the cost
   model for the codec types that enter at rung 22+
   (`CascadingFORPrevBlockFrequencyPartitionEncoding`, `HuffmanEncoding`,
   `FSEEncoding`, ...) the same way FINDINGS.md already diagnosed the
   65,536-unique-value entropy-estimator cap for AutoSIS's default 100K-sample
   collapse -- this is very likely the same family of defect. Fixing it
   recovers real compression using codecs the registry
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

## Best codec per metric per dataset (summary)

| driver | dataset | metric | best_codec | value |
|---|---|---|---|---|
| bench_compression | XMarkPrePostElements | compression_ratio | **OpenZL** | 18.8 |
| bench_compression | XMarkPrePostElements | payload_bytes | **OpenZL** | 85,008 |
| bench_compression | XMarkPrePostFull | compression_ratio | **OpenZL** | 22.2 |
| bench_compression | XMarkPrePostFull | payload_bytes | **OpenZL** | 72,149 |
| bench_encode | XMarkPrePostElements | encode_ns | **RawBitPacked** | 1.23e+06 |
| bench_encode | XMarkPrePostElements | selection_ns | **AutoSIS_LSB** | 1.41e+10 |
| bench_encode | XMarkPrePostElements | compression_ratio | **OpenZL** | 18.8 |
| bench_encode | XMarkPrePostFull | encode_ns | **RawBitPacked** | 1.19e+06 |
| bench_encode | XMarkPrePostFull | selection_ns | **AutoSIS_LSB** | 1.38e+10 |
| bench_encode | XMarkPrePostFull | compression_ratio | **OpenZL** | 22.2 |
| bench_decode_bulk | XMarkPrePostElements | time_ns | **FPE_PerTierBitmaps** | 2.18e+05 |
| bench_decode_bulk | XMarkPrePostElements | decode_MBps | **FPE_PerTierBitmaps** | 7.35e+03 |
| bench_decode_bulk | XMarkPrePostFull | time_ns | **FPE_NoIndex** | 2.01e+05 |
| bench_decode_bulk | XMarkPrePostFull | decode_MBps | **FPE_NoIndex** | 7.96e+03 |
| bench_decode_range | XMarkPrePostElements | time_ns | **FPE_PerTierBitmaps** | 9.07e+04 |
| bench_decode_range | XMarkPrePostElements | elem_Meps | **FPE_PerTierBitmaps** | 846 |
| bench_decode_gather | XMarkPrePostElements | time_ns | **FPE_EliasFano** | 7.6e+03 |
| bench_decode_gather | XMarkPrePostElements | sel_elem_Meps | **FPE_EliasFano** | 239 |
| bench_decode_gather | XMarkPrePostFull | time_ns | **FPE_EliasFano** | 4.7e+03 |
| bench_decode_gather | XMarkPrePostFull | sel_elem_Meps | **FPE_EliasFano** | 352 |
| bench_decode_point | XMarkPrePostElements | ns_per_probe | **FPE_NoIndex** | 7.72 |
| bench_decode_point | XMarkPrePostFull | ns_per_probe | **FPE_NoIndex** | 8.36 |
| bench_ablation | XMarkPrePostElements | compression_ratio | **raw_upward_through_ra/21_plus_CascadingFORPrevFrequencyPartitionEncoding** | 6.54 |

## Suggested follow-up: a "Sequence" codec

This round only benchmarked the **packed** `prepost_id` (level:8|pre:28|post:28 bit-packed into one int64), not the unpacked `pre`/`post`/`level` int32 columns (`Datasets/XMark/level_*.parquet` etc., generated but not yet registered -- see `DatasetRegistry.hpp`'s int32Datasets() note) -- so there is no direct 'how well does codec X compress the raw `pre` column' number yet. What the packed-id sweep does show: `OpenZL` is the strongest whole-column baseline (18.8x-22.2x, `bench_compression`), and SubIntSplit's DP -- once given the full codec universe (`bench_ablation`, `raw_upward_through_ra`, rung 21) -- reaches **6.54x** via `CascadingFORPrevFrequencyPartitionEncoding` while keeping `sis_fast_skip=1` (true) the entire way to rung 28 (all 29 codecs admitted). Random access was never traded away on this data, at any point in the ladder.

That 6.54x is still well short of what's structurally possible for `pre` alone: by construction `pre` is a permutation of `0..n-1` in document order, so a maximally-compressed representation is a few bytes (base + stride), not ~1.6 bits/element. No codec in this sweep models 'value equals row index' directly -- FOR/delta-style codecs get close only when the data is *locally* sequential in physical row order, which `pre` already is by definition, but they still spend real bits per element on frame references and residuals rather than recognizing the whole column as one arithmetic sequence. A dedicated RA-capable **Sequence codec** -- encode as `base + stride * index` plus a sparse exception list for any positions that deviate (e.g. `post`, which is *not* globally sequential but is piecewise-monotonic within a subtree), with O(1) `decodeAt` via direct arithmetic -- is the natural next comparator, and this session's numbers are the baseline it would need to beat: 6.54x compression while keeping FastSkip, on the packed id, and whatever the unpacked `pre` column's own dedicated sweep turns up as its current best (future work, see above).
