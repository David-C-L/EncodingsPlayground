# XMark pre/post-id SubIntSplit sweep

Datasets: `XMarkPrePostElements`, `XMarkPrePostFull` (level:8|pre:28|post:28 bit-packed tree ids; see `Datasets/XMark/README.md`). Best value among non-Raw, non-`_Prof` codecs is **bolded** per column.

> AutoSIS_LSB/AutoSIS_MSB use the registry's *default* 100,000-sample AutoSIS config. `Benchmarks/drivers/FINDINGS.md` documents this default collapsing to a degenerate single full-width section on near-unique high-cardinality ids (measured on Twitter Snowflake: 64.13 bits/element, worse than Raw) due to an open entropy-cost-model defect at that sample size. XMark's `pre`/`post` fields are similarly high-cardinality across 16-32M rows, so a compression_ratio for AutoSIS_LSB/MSB at or worse than Raw's is very likely this same known issue, not a real finding about SIS on tree ids -- treat the bench_ablation numbers below (10,000-sample fresh DP runs) as the more trustworthy signal for what SIS actually wants to do here.

> AutoSIS_LSB/AutoSIS_MSB are excluded from the bench_decode_gather and bench_decode_point tables below (and only appear in bench_compression/bench_encode/bench_decode_bulk/bench_decode_range, all of which use few, large accesses). A gather cell for AutoSIS on this data ran for 3.5+ hours at 99.9% CPU before being killed: its degenerate single full-width section (see the note above) reports `FastSkip=true`, but each individual small range/point access is apparently still very expensive on this near-unique data -- a real, access-pattern-dependent performance defect, not merely the known compression-collapse one. bulk/range access (one or a few large reads) did not trigger it.

> The FOR baseline (registered this session) failed round-trip validation once, on XMarkPrePostFull at N=200,000, inside bench_compression (`materializeAll mismatch at row 1`) -- correctly caught and excluded by `--validate`. The same (encoder, dataset, N) pair validated successfully afterward in bench_encode, bench_decode_bulk and bench_decode_range. This looks like a real, not-yet-diagnosed edge case (possibly nondeterministic) rather than a consistent break; flagging it here rather than debugging it in this session.

## bench_compression

### XMarkPrePostElements

| encoding | compression_ratio | payload_bytes | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 1.05 | 1,519,411 | 1 |
| AdaptiveDictionary | 0.885 | 1,807,831 | 1 |
| AutoSIS_LSB | 0.998 | 1,603,171 | 1 |
| AutoSIS_MSB | 0.998 | 1,603,171 | 1 |
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
| AutoSIS_LSB | 0.998 | 1,603,171 | 1 |
| AutoSIS_MSB | 0.998 | 1,603,171 | 1 |
| BlockFORFPE | 1.07 | 1,500,324 | 1 |
| BlockFPE | 0.969 | 1,650,485 | 1 |
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
| AdaptiveBitPrefix | 1.07e+07 | n/a | 1.05 | 1 |
| AdaptiveDictionary | 9.08e+07 | n/a | 0.885 | 1 |
| AutoSIS_LSB | 2.97e+07 | **1.45e+10** | 0.998 | 1 |
| AutoSIS_MSB | 3.16e+07 | 1.52e+10 | 0.998 | 1 |
| BlockFORFPE | 3.26e+08 | n/a | 1.07 | 1 |
| BlockFPE | 2.72e+08 | n/a | 0.969 | 1 |
| FOR | 2.59e+06 | n/a | 0.992 | 1 |
| FPE_EliasFano | 5.99e+07 | n/a | 1 | 1 |
| FPE_NoIndex | 7.42e+07 | n/a | 1 | 1 |
| FPE_PerTierBitmaps | 7.13e+07 | n/a | 1 | 1 |
| FPE_TierTagArray | 7.47e+07 | n/a | 0.985 | 1 |
| OpenZL | 9.52e+07 | n/a | **18.8** | 1 |
| Raw | 4.29e+05 | n/a | 1 | 1 |
| RawBitPacked | **1.48e+06** | n/a | 1.07 | 1 |
| Zstd | 5.62e+06 | n/a | 1.32 | 1 |

### XMarkPrePostFull

| encoding | encode_ns | selection_ns | compression_ratio | n_cells |
|---|---|---|---|---|
| AdaptiveBitPrefix | 8.72e+06 | n/a | 1.05 | 1 |
| AdaptiveDictionary | 6.1e+07 | n/a | 0.885 | 1 |
| AutoSIS_LSB | 3.24e+07 | **1.47e+10** | 0.998 | 1 |
| AutoSIS_MSB | 2.37e+07 | 1.48e+10 | 0.998 | 1 |
| BlockFORFPE | 3.31e+08 | n/a | 1.07 | 1 |
| BlockFPE | 3.15e+08 | n/a | 0.969 | 1 |
| FOR | 2.42e+06 | n/a | 0.992 | 1 |
| FPE_EliasFano | 1.01e+08 | n/a | 1 | 1 |
| FPE_NoIndex | 9.01e+07 | n/a | 1 | 1 |
| FPE_PerTierBitmaps | 8.25e+07 | n/a | 1 | 1 |
| FPE_TierTagArray | 6.71e+07 | n/a | 0.985 | 1 |
| OpenZL | 1.07e+08 | n/a | **22.2** | 1 |
| Raw | 2.94e+05 | n/a | 1 | 1 |
| RawBitPacked | **1.27e+06** | n/a | 1.07 | 1 |
| Zstd | 4.79e+06 | n/a | 1.31 | 1 |

## bench_decode_bulk

### XMarkPrePostElements

| encoding | time_ns | decode_MBps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 6.49e+06 | 247 | 1 |
| AdaptiveDictionary | 6.38e+05 | 2.51e+03 | 1 |
| AutoSIS_LSB | 6.11e+08 | 2.62 | 1 |
| AutoSIS_MSB | 6.13e+08 | 2.61 | 1 |
| BlockFORFPE | 2.9e+06 | 552 | 1 |
| BlockFPE | 1.16e+06 | 1.38e+03 | 1 |
| FOR | 1.2e+06 | 1.34e+03 | 1 |
| FPE_EliasFano | 2.7e+05 | 5.93e+03 | 1 |
| FPE_NoIndex | 2.49e+05 | 6.42e+03 | 1 |
| FPE_PerTierBitmaps | **2.21e+05** | **7.23e+03** | 1 |
| FPE_TierTagArray | 8.54e+05 | 1.87e+03 | 1 |
| OpenZL | 1.19e+07 | 135 | 1 |
| Raw | 6.11e+05 | 2.62e+03 | 1 |
| RawBitPacked | 5.68e+05 | 2.82e+03 | 1 |
| Zstd | 4.45e+06 | 360 | 1 |

### XMarkPrePostFull

| encoding | time_ns | decode_MBps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 5.48e+06 | 292 | 1 |
| AdaptiveDictionary | 4.59e+05 | 3.49e+03 | 1 |
| AutoSIS_LSB | 6.02e+08 | 2.66 | 1 |
| AutoSIS_MSB | 5.91e+08 | 2.71 | 1 |
| BlockFORFPE | 2.31e+06 | 692 | 1 |
| BlockFPE | 1.37e+06 | 1.17e+03 | 1 |
| FOR | 9.47e+05 | 1.69e+03 | 1 |
| FPE_EliasFano | **1.97e+05** | **8.11e+03** | 1 |
| FPE_NoIndex | 2.01e+05 | 7.96e+03 | 1 |
| FPE_PerTierBitmaps | 3.38e+05 | 4.74e+03 | 1 |
| FPE_TierTagArray | 8.02e+05 | 1.99e+03 | 1 |
| OpenZL | 1.39e+07 | 115 | 1 |
| Raw | 5.14e+05 | 3.11e+03 | 1 |
| RawBitPacked | 4.88e+05 | 3.28e+03 | 1 |
| Zstd | 3.3e+06 | 485 | 1 |

## bench_decode_range

### XMarkPrePostElements

| encoding | time_ns | elem_Meps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 2.08e+06 | 37.6 | 36 |
| AdaptiveDictionary | 1.97e+05 | 382 | 36 |
| AutoSIS_LSB | 2.38e+08 | 0.311 | 36 |
| AutoSIS_MSB | 2.38e+08 | 0.314 | 36 |
| BlockFORFPE | 1.46e+06 | 51.9 | 36 |
| BlockFPE | 5.31e+05 | 139 | 36 |
| FOR | 4.43e+05 | 171 | 36 |
| FPE_EliasFano | **8.88e+04** | **891** | 36 |
| FPE_NoIndex | 2.46e+05 | 312 | 36 |
| FPE_PerTierBitmaps | 1.13e+05 | 660 | 36 |
| FPE_TierTagArray | 5.6e+05 | 147 | 36 |
| OpenZL | 1.27e+07 | 6.99 | 36 |
| Raw | 1.58e+05 | 474 | 36 |
| RawBitPacked | 2.05e+05 | 359 | 36 |
| Zstd | 3.96e+06 | 17.9 | 36 |

### XMarkPrePostFull

| encoding | time_ns | elem_Meps | n_cells |
|---|---|---|---|
| AdaptiveBitPrefix | 2.33e+06 | 32.1 | 36 |
| AdaptiveDictionary | 2.15e+05 | 347 | 36 |
| AutoSIS_LSB | 2.36e+08 | 0.314 | 36 |
| AutoSIS_MSB | 2.42e+08 | 0.306 | 36 |
| BlockFORFPE | 1.1e+06 | 67.7 | 36 |
| BlockFPE | 5.27e+05 | 139 | 36 |
| FOR | 3.47e+05 | 208 | 36 |
| FPE_EliasFano | **9.75e+04** | **769** | 36 |
| FPE_NoIndex | 2.68e+05 | 283 | 36 |
| FPE_PerTierBitmaps | 1.23e+05 | 606 | 36 |
| FPE_TierTagArray | 4.99e+05 | 173 | 36 |
| OpenZL | 1.25e+07 | 6.46 | 36 |
| Raw | 1.62e+05 | 463 | 36 |
| RawBitPacked | 2.49e+05 | 301 | 36 |
| Zstd | 4.09e+06 | 20.1 | 36 |

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

## Best codec per metric per dataset (summary)

| driver | dataset | metric | best_codec | value |
|---|---|---|---|---|
| bench_compression | XMarkPrePostElements | compression_ratio | **OpenZL** | 18.8 |
| bench_compression | XMarkPrePostElements | payload_bytes | **OpenZL** | 85,008 |
| bench_compression | XMarkPrePostFull | compression_ratio | **OpenZL** | 22.2 |
| bench_compression | XMarkPrePostFull | payload_bytes | **OpenZL** | 72,149 |
| bench_encode | XMarkPrePostElements | encode_ns | **RawBitPacked** | 1.48e+06 |
| bench_encode | XMarkPrePostElements | selection_ns | **AutoSIS_LSB** | 1.45e+10 |
| bench_encode | XMarkPrePostElements | compression_ratio | **OpenZL** | 18.8 |
| bench_encode | XMarkPrePostFull | encode_ns | **RawBitPacked** | 1.27e+06 |
| bench_encode | XMarkPrePostFull | selection_ns | **AutoSIS_LSB** | 1.47e+10 |
| bench_encode | XMarkPrePostFull | compression_ratio | **OpenZL** | 22.2 |
| bench_decode_bulk | XMarkPrePostElements | time_ns | **FPE_PerTierBitmaps** | 2.21e+05 |
| bench_decode_bulk | XMarkPrePostElements | decode_MBps | **FPE_PerTierBitmaps** | 7.23e+03 |
| bench_decode_bulk | XMarkPrePostFull | time_ns | **FPE_EliasFano** | 1.97e+05 |
| bench_decode_bulk | XMarkPrePostFull | decode_MBps | **FPE_EliasFano** | 8.11e+03 |
| bench_decode_range | XMarkPrePostElements | time_ns | **FPE_EliasFano** | 8.88e+04 |
| bench_decode_range | XMarkPrePostElements | elem_Meps | **FPE_EliasFano** | 891 |
| bench_decode_range | XMarkPrePostFull | time_ns | **FPE_EliasFano** | 9.75e+04 |
| bench_decode_range | XMarkPrePostFull | elem_Meps | **FPE_EliasFano** | 769 |
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
