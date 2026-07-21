#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#ifdef __AVX2__
#  include <immintrin.h>
#endif

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "encoders/selectors/costs/SpeedCostModel.hpp"
#include "encoders/selectors/costs/CostModelSet.hpp"
#include "generators/samplers/StreamSampling.hpp"

using encodings::encoders::selectors::costs::RawCostModel;
using encodings::encoders::selectors::costs::RawBitPackedCostModel;
using encodings::encoders::selectors::costs::AdaptiveFORCostModel;
using encodings::encoders::selectors::costs::AdaptiveFramedBitPrefixCostModel;
using encodings::encoders::selectors::costs::FORCostModel;
using encodings::encoders::selectors::costs::DictionaryCostModel;
using encodings::encoders::selectors::costs::AdaptiveDictionaryCostModel;
using encodings::encoders::selectors::costs::RLECostModel;
using encodings::encoders::selectors::costs::HuffmanCostModel;
using encodings::encoders::selectors::costs::LZ4CostModel;
using encodings::encoders::selectors::costs::FSECostModel;
using encodings::encoders::selectors::costs::FrequencyPartitionCostModel;
using encodings::encoders::selectors::costs::BlockFrequencyPartitionCostModel;
using encodings::encoders::selectors::costs::BlockFrequencyPartitionFORCostModel;
using encodings::encoders::selectors::costs::BlockFSECostModel;
using encodings::encoders::selectors::costs::BlockFORFPECostModel;
using encodings::encoders::selectors::costs::MainlyConstantCostModel;
using encodings::encoders::selectors::costs::RangePackCostModel;
using encodings::encoders::selectors::costs::CascadingFORCostModel;
using encodings::encoders::selectors::costs::CascadingFORBlockFPECostModel;
using encodings::encoders::selectors::costs::RunLengthCascadingFORStartsCostModel;
using encodings::encoders::selectors::costs::CascadingFORFSECostModel;
using encodings::encoders::selectors::costs::CascadingFORBlockFSECostModel;
using encodings::encoders::selectors::costs::CascadingFORHuffmanCostModel;
using encodings::encoders::selectors::costs::CascadingFORPrevFSECostModel;
using encodings::encoders::selectors::costs::CascadingFORPrevBlockFSECostModel;
using encodings::encoders::selectors::costs::CascadingFORPrevHuffmanCostModel;
using encodings::encoders::selectors::costs::CascadingFORPrevFrequencyPartitionCostModel;
using encodings::encoders::selectors::costs::CascadingFORPrevBlockFrequencyPartitionCostModel;
using encodings::encoders::selectors::costs::EncodeSpeedCostModel;
using encodings::encoders::selectors::costs::DecodeAllSpeedCostModel;
using encodings::encoders::selectors::costs::DecodeAtSpeedCostModel;
using encodings::encoders::selectors::costs::DecodeRangeSpeedCostModel;
using encodings::encoders::selectors::costs::WeightedCompositeEncodingCostModel;
using encodings::encoders::selectors::costs::CostModelSet;
using encodings::encoders::selectors::costs::CostModelDimension;

namespace encodings::encoders {

// Generic sub-integer split encoder (N-way split where sum(bits)=bit-width of T).
// EnableProfiling=true activates per-section timing and fills
// EncodingMetadata::subStreamEncodeMetrics on each encode() call, and accumulates
// per-section decode times accessible via the subStream*() virtual methods.
// EnableProfiling=false (default) is identical to the pre-profiling code path.
template <typename T, bool EnableProfiling = false>
    requires (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>)
class SubIntSplitEncoder final : public Codec<T, uint8_t> {
public:
    using ValueT = T;
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    static constexpr uint8_t kTotalBits      = static_cast<uint8_t>(sizeof(T) * 8);
    static constexpr bool    kProfileSections = EnableProfiling;
    // Chunk size for the inner decode loop.  Chosen so that one acc slice
    // (kDecodeChunkSize * sizeof(SectionT)) plus per-section scratch (at most
    // kDecodeChunkSize * sizeof(SectionT) total) comfortably fits in L2 cache.
    // 8192 * 8 * 2 = 128 KB — fits in most 256 KB L2 caches.
    static constexpr size_t  kDecodeChunkSize = 8192;
    static_assert(sizeof(T) == sizeof(SectionT),
                  "T and SectionT must be same width for safe reinterpret_cast in decode paths");

    explicit SubIntSplitEncoder(SubIntSplitConfigIntegral<SectionT> cfg) : cfg_(std::move(cfg)) {
        cfg_.validate();
    }

    EncodedBuffer<uint8_t> encode(std::span<const T> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty();
        }

        const size_t splits = cfg_.splitCount();

        // Flat, column-major section buffer: section s occupies [s*N, (s+1)*N).
        // make_unique_for_overwrite skips the zero-initialisation that vector::resize()
        // would perform — the fill loop below writes every element before it is read.
        auto sectionsRaw = std::make_unique_for_overwrite<SectionT[]>(splits * N);

        // Split values into sections according to order
        if (cfg_.order == BitSplitOrder::LSB_TO_MSB) {
            for (size_t i = 0; i < N; ++i) {
                uint64_t shifted = static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(data[i]));
                for (size_t s = 0; s < splits; ++s) {
                    const uint8_t bits = cfg_.bits[s];
                    const uint64_t mask = (bits == 64)
                        ? ~uint64_t{0}
                        : ((uint64_t{1} << bits) - 1);
                    sectionsRaw[s * N + i] = static_cast<SectionT>(shifted & mask);
                    if (bits < kTotalBits) shifted >>= bits;
                }
            }
        } else { // MSB_TO_LSB
            // Precompute starting shifts from MSB side
            std::vector<uint8_t> shiftStart(splits, 0);
            uint8_t remaining = kTotalBits;
            for (size_t s = 0; s < splits; ++s) {
                const uint8_t bits = cfg_.bits[s];
                if (bits > remaining) {
                    throw std::invalid_argument("SubIntSplitEncoder: bits exceed width when splitting");
                }
                remaining = static_cast<uint8_t>(remaining - bits);
                shiftStart[s] = remaining;
            }
            for (size_t i = 0; i < N; ++i) {
                uint64_t v = static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(data[i]));
                for (size_t s = 0; s < splits; ++s) {
                    const uint8_t bits = cfg_.bits[s];
                    const uint64_t mask = (bits == 64)
                        ? ~uint64_t{0}
                        : ((uint64_t{1} << bits) - 1);
                    sectionsRaw[s * N + i] = static_cast<SectionT>((v >> shiftStart[s]) & mask);
                }
            }
        }

        // Encode each section.
        // When parallelism is enabled, sections are encoded concurrently via
        // std::async — each task owns a disjoint slice of sectionsRaw and its
        // own codec instance, so no synchronisation is needed.
        std::vector<EncodedBuffer<uint8_t>> encodedSections(splits);
        std::vector<uint64_t> sectionSizes(splits);
        std::vector<int64_t> sectionEncodeTimes(splits, 0);

        auto encodeSection = [&](size_t s) {
            if constexpr (kProfileSections) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                encodedSections[s] = cfg_.codecs[s]->encode(
                    std::span<const SectionT>(&sectionsRaw[s * N], N));
                sectionEncodeTimes[s] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            } else {
                encodedSections[s] = cfg_.codecs[s]->encode(
                    std::span<const SectionT>(&sectionsRaw[s * N], N));
            }
            sectionSizes[s] = encodedSections[s].data().size();
        };

        if (cfg_.parallelism.enabled && splits > 1) {
            std::vector<std::future<void>> futures(splits);
            for (size_t s = 0; s < splits; ++s)
                futures[s] = std::async(std::launch::async, encodeSection, s);
            for (auto& f : futures) f.get();
        } else {
            for (size_t s = 0; s < splits; ++s)
                encodeSection(s);
        }

        // Build output in a single pre-sized allocation: header + all payloads.
        // Avoids the repeated reallocations from vector::insert that _M_range_insert caused.
        const size_t headerSize = 8 + 1 + 1 + splits * (1 + 8);
        size_t totalPayload = 0;
        for (const auto& enc : encodedSections) totalPayload += enc.data().size();

        std::vector<uint8_t> out(headerSize + totalPayload);
        uint8_t* p = out.data();
        writeU64(p, static_cast<uint64_t>(N)); p += 8;
        *p++ = static_cast<uint8_t>(splits);
        *p++ = static_cast<uint8_t>(cfg_.order);
        for (size_t s = 0; s < splits; ++s) {
            *p++ = cfg_.bits[s];
            writeU64(p, sectionSizes[s]);
            p += 8;
        }
        // p is now at headerSize offset — copy section payloads contiguously
        for (const auto& enc : encodedSections) {
            const size_t sz = enc.data().size();
            std::memcpy(p, enc.data().data(), sz);
            p += sz;
        }

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<T>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(T);
        meta.supportsRandomAccess = allSectionsRandomAccess();

        if constexpr (kProfileSections) {
            meta.subStreamEncodeMetrics.resize(splits);
            for (size_t s = 0; s < splits; ++s) {
                meta.subStreamEncodeMetrics[s] = {
                    cfg_.codecs[s]->name(),
                    cfg_.bits[s],
                    sectionSizes[s],
                    sectionEncodeTimes[s]
                };
            }
        }

        if (verboseEnabled()) {
            const double bitsPerElemTotal = N ? (static_cast<double>(out.size()) * 8.0 / static_cast<double>(N)) : 0.0;
            std::cerr << "[SubIntSplit] N=" << N
                      << " order=" << (cfg_.order == BitSplitOrder::LSB_TO_MSB ? "LSB" : "MSB")
                      << " splits=" << splits
                      << " total=" << out.size() << "B (~" << bitsPerElemTotal << " b/elem)"
                      << " ratio=" << (N ? (static_cast<double>(out.size()) / (N * sizeof(T))) : 0.0)
                      << "\n";
            for (size_t s = 0; s < splits; ++s) {
                const double bitsPerElem = N ? (static_cast<double>(sectionSizes[s]) * 8.0 / static_cast<double>(N)) : 0.0;
                std::cerr << "  sec " << s
                          << " bits=" << static_cast<int>(cfg_.bits[s])
                          << " bytes=" << sectionSizes[s] << "B"
                          << " (~" << bitsPerElem << " b/elem)"
                          << " codec=" << cfg_.codecs[s]->name()
                          << "\n";
            }
        }

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const auto& h = getCachedHeader(encoded);
        if (h.N == 0) return {};

        const size_t splits = h.bits.size();

        // Chunk-based decode: process kDecodeChunkSize elements at a time so that
        // the acc slice and section scratch buffers stay in L2 cache across all
        // splits for a given chunk.
        //
        // Sections that lack RandomAccess (e.g. OpenZL with blockSize=0, LZ4,
        // FSE, Huffman) cannot decode ranges efficiently — their decodeRange
        // calls decodeAll internally, giving O(N²/B) cost if called per chunk.
        // Those sections are decoded once upfront into a full-N SectionT buffer
        // and then accumulated per-chunk from that buffer, preserving L2 reuse.
        std::vector<T> out(h.N);
        SectionT* acc = reinterpret_cast<SectionT*>(out.data());

        if constexpr (kProfileSections)
            bulkDecodeTimeNs_.assign(splits, 0);

        std::vector<std::unique_ptr<SectionT[]>> preDecoded(splits);

        if (cfg_.parallelism.enabled && splits > 1) {
            // Parallel pre-decode: ALL sections are decoded upfront (including RA
            // ones) so the subsequent chunk accumulation phase is pure memory and
            // never calls back into codec instances from worker threads.
            // Each task writes to its own preDecoded[s] buffer — no shared state.
            std::vector<std::future<void>> preFutures(splits);
            for (size_t s = 0; s < splits; ++s) {
                preDecoded[s] = std::make_unique_for_overwrite<SectionT[]>(h.N);
                preFutures[s] = std::async(std::launch::async, [&, s]() {
                    if constexpr (kProfileSections) {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        cfg_.codecs[s]->decodeAllInto(h.views[s], preDecoded[s].get(), h.N);
                        bulkDecodeTimeNs_[s] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    } else {
                        cfg_.codecs[s]->decodeAllInto(h.views[s], preDecoded[s].get(), h.N);
                    }
                });
            }
            for (auto& f : preFutures) f.get();

            // Parallel chunk accumulation: partition [0, h.N) into contiguous
            // sub-ranges, one per thread. Each thread runs its own chunk loop over
            // assign/combineSectionInto — pure memory, disjoint output ranges, no
            // codec calls, no synchronisation needed.
            const size_t numChunks   = (h.N + kDecodeChunkSize - 1) / kDecodeChunkSize;
            const size_t nThreads    = effectiveThreadCount(numChunks);
            const size_t chunksPerThread = (numChunks + nThreads - 1) / nThreads;

            std::vector<std::future<void>> chunkFutures(nThreads);
            for (size_t t = 0; t < nThreads; ++t) {
                const size_t firstChunk = t * chunksPerThread;
                const size_t lastChunk  = std::min(firstChunk + chunksPerThread, numChunks);
                if (firstChunk >= numChunks) break;
                chunkFutures[t] = std::async(std::launch::async, [&, firstChunk, lastChunk]() {
                    for (size_t c = firstChunk; c < lastChunk; ++c) {
                        const size_t chunkStart = c * kDecodeChunkSize;
                        const size_t chunkEnd   = std::min(chunkStart + kDecodeChunkSize, h.N);
                        const size_t chunkCount = chunkEnd - chunkStart;
                        for (size_t s = 0; s < splits; ++s) {
                            if (s == 0) assignSectionInto (preDecoded[s].get() + chunkStart,
                                                           acc + chunkStart, chunkCount, h.shifts[0]);
                            else        combineSectionInto(preDecoded[s].get() + chunkStart,
                                                           acc + chunkStart, chunkCount, h.shifts[s]);
                        }
                    }
                });
            }
            for (auto& f : chunkFutures) if (f.valid()) f.get();
        } else {
            // Sequential path: pre-decode only non-random-access sections, then
            // process chunks in order (preserves cache locality across sections).
            for (size_t s = 0; s < splits; ++s) {
                if (!cfg_.codecs[s]->properties().has(EncodingProperty::RandomAccess)) {
                    preDecoded[s] = std::make_unique_for_overwrite<SectionT[]>(h.N);
                    if constexpr (kProfileSections) {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        cfg_.codecs[s]->decodeAllInto(h.views[s], preDecoded[s].get(), h.N);
                        bulkDecodeTimeNs_[s] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    } else {
                        cfg_.codecs[s]->decodeAllInto(h.views[s], preDecoded[s].get(), h.N);
                    }
                }
            }

            for (size_t chunkStart = 0; chunkStart < h.N; chunkStart += kDecodeChunkSize) {
                const size_t chunkEnd   = std::min(chunkStart + kDecodeChunkSize, h.N);
                const size_t chunkCount = chunkEnd - chunkStart;

                for (size_t s = 0; s < splits; ++s) {
                    if (preDecoded[s]) {
                        if (s == 0) assignSectionInto (preDecoded[s].get() + chunkStart,
                                                       acc + chunkStart, chunkCount, h.shifts[0]);
                        else        combineSectionInto(preDecoded[s].get() + chunkStart,
                                                       acc + chunkStart, chunkCount, h.shifts[s]);
                    } else {
                        if constexpr (kProfileSections) {
                            const auto t0 = std::chrono::high_resolution_clock::now();
                            cfg_.codecs[s]->decodeRangeAndAccumulate(
                                h.views[s], chunkStart, chunkEnd,
                                acc + chunkStart, chunkCount, h.shifts[s], s == 0);
                            bulkDecodeTimeNs_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        } else {
                            cfg_.codecs[s]->decodeRangeAndAccumulate(
                                h.views[s], chunkStart, chunkEnd,
                                acc + chunkStart, chunkCount, h.shifts[s], s == 0);
                        }
                    }
                }
            }
        }

        return out;
    }

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto& h = getCachedHeader(encoded);
        if (index >= h.N) return std::nullopt;

        // Inline combine using precomputed shifts — no allocation.
        SectionT v = 0;
        const size_t splits = h.bits.size();
        if constexpr (kProfileSections) {
            if (decodeAtAccumNs_.size() != splits)
                decodeAtAccumNs_.assign(splits, 0);
        }
        for (size_t s = 0; s < splits; ++s) {
            if constexpr (kProfileSections) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                v |= static_cast<SectionT>(decodeOneSection(*cfg_.codecs[s], h.views[s], index)) << h.shifts[s];
                decodeAtAccumNs_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            } else {
                v |= static_cast<SectionT>(decodeOneSection(*cfg_.codecs[s], h.views[s], index)) << h.shifts[s];
            }
        }
        return static_cast<T>(v);
    }

    // Writes [start, end) into a caller-provided buffer.  The buffer must hold
    // at least (end - start) elements and need not be initialised: section 0
    // always writes every output element before any subsequent OR-accumulation.
    void decodeRangeInto(const EncodedBuffer<uint8_t>& encoded,
                         size_t start, size_t end,
                         T* dst, size_t /*n*/) override {
        const auto& h = getCachedHeader(encoded);
        if (start >= h.N || start >= end) return;
        end = std::min(end, h.N);
        const size_t count = end - start;

        const size_t splits = h.bits.size();
        SectionT* acc = reinterpret_cast<SectionT*>(dst);

        if constexpr (kProfileSections) {
            if (decodeRangeAccumNs_.size() != splits)
                decodeRangeAccumNs_.assign(splits, 0);
        }

        std::vector<std::unique_ptr<SectionT[]>> preDecoded(splits);

        if (cfg_.parallelism.enabled && splits > 1) {
            // Parallel pre-decode: all sections decoded upfront so the chunk
            // accumulation phase is pure memory with no codec calls.
            std::vector<std::future<void>> preFutures(splits);
            for (size_t s = 0; s < splits; ++s) {
                preDecoded[s] = std::make_unique_for_overwrite<SectionT[]>(count);
                preFutures[s] = std::async(std::launch::async, [&, s]() {
                    if constexpr (kProfileSections) {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        cfg_.codecs[s]->decodeRangeInto(h.views[s], start, end, preDecoded[s].get(), count);
                        decodeRangeAccumNs_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    } else {
                        cfg_.codecs[s]->decodeRangeInto(h.views[s], start, end, preDecoded[s].get(), count);
                    }
                });
            }
            for (auto& f : preFutures) f.get();

            // Parallel chunk accumulation over disjoint output sub-ranges.
            const size_t numChunks      = (count + kDecodeChunkSize - 1) / kDecodeChunkSize;
            const size_t nThreads       = effectiveThreadCount(numChunks);
            const size_t chunksPerThread = (numChunks + nThreads - 1) / nThreads;

            std::vector<std::future<void>> chunkFutures(nThreads);
            for (size_t t = 0; t < nThreads; ++t) {
                const size_t firstChunk = t * chunksPerThread;
                const size_t lastChunk  = std::min(firstChunk + chunksPerThread, numChunks);
                if (firstChunk >= numChunks) break;
                chunkFutures[t] = std::async(std::launch::async, [&, firstChunk, lastChunk]() {
                    for (size_t c = firstChunk; c < lastChunk; ++c) {
                        const size_t chunkStart = start + c * kDecodeChunkSize;
                        const size_t chunkEnd   = std::min(chunkStart + kDecodeChunkSize, end);
                        const size_t chunkCount = chunkEnd - chunkStart;
                        const size_t chunkOff   = chunkStart - start;
                        for (size_t s = 0; s < splits; ++s) {
                            if (s == 0) assignSectionInto (preDecoded[s].get() + chunkOff,
                                                           acc + chunkOff, chunkCount, h.shifts[0]);
                            else        combineSectionInto(preDecoded[s].get() + chunkOff,
                                                           acc + chunkOff, chunkCount, h.shifts[s]);
                        }
                    }
                });
            }
            for (auto& f : chunkFutures) if (f.valid()) f.get();
        } else {
            // Sequential path: pre-decode only non-random-access sections.
            for (size_t s = 0; s < splits; ++s) {
                if (!cfg_.codecs[s]->properties().has(EncodingProperty::RandomAccess)) {
                    preDecoded[s] = std::make_unique_for_overwrite<SectionT[]>(count);
                    if constexpr (kProfileSections) {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        cfg_.codecs[s]->decodeRangeInto(h.views[s], start, end, preDecoded[s].get(), count);
                        decodeRangeAccumNs_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    } else {
                        cfg_.codecs[s]->decodeRangeInto(h.views[s], start, end, preDecoded[s].get(), count);
                    }
                }
            }

            for (size_t chunkStart = start; chunkStart < end; chunkStart += kDecodeChunkSize) {
                const size_t chunkEnd   = std::min(chunkStart + kDecodeChunkSize, end);
                const size_t chunkCount = chunkEnd - chunkStart;
                const size_t chunkOff   = chunkStart - start;

                for (size_t s = 0; s < splits; ++s) {
                    if (preDecoded[s]) {
                        if (s == 0) assignSectionInto (preDecoded[s].get() + chunkOff,
                                                       acc + chunkOff, chunkCount, h.shifts[0]);
                        else        combineSectionInto(preDecoded[s].get() + chunkOff,
                                                       acc + chunkOff, chunkCount, h.shifts[s]);
                    } else {
                        if constexpr (kProfileSections) {
                            const auto t0 = std::chrono::high_resolution_clock::now();
                            cfg_.codecs[s]->decodeRangeAndAccumulate(
                                h.views[s], chunkStart, chunkEnd,
                                acc + chunkOff, chunkCount, h.shifts[s], s == 0);
                            decodeRangeAccumNs_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        } else {
                            cfg_.codecs[s]->decodeRangeAndAccumulate(
                                h.views[s], chunkStart, chunkEnd,
                                acc + chunkOff, chunkCount, h.shifts[s], s == 0);
                        }
                    }
                }
            }
        }
    }

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto& h = getCachedHeader(encoded);
        if (start >= h.N) return {};
        end = std::min(end, h.N);
        if (start >= end) return {};
        const size_t count = end - start;
        std::vector<T> out(count);
        decodeRangeInto(encoded, start, end, out.data(), count);
        return out;
    }

    // Fast path for the "selective/gather" access pattern: an ascending,
    // non-overlapping RowRangeList with gaps to skip between surviving runs
    // (mirrors Nimble's TableScan bulkScan pattern). Reuses the same cached
    // header / chunked decodeRangeInto machinery as the range-access path per
    // range, and — when EnableProfiling is on — records the skip vs.
    // materialize time split so the paper's skip-latency argument can be
    // measured directly. There is no physical seek/skip cost to pay here
    // beyond this loop's own iteration, since decodeRangeInto is already
    // offset-addressable (no forward-only cursor to advance) — contrast with
    // a codec backed by a stateful sequential decoder, where skipping a gap
    // has a real, non-zero cost.
    void decodeGatherInto(const EncodedBuffer<uint8_t>& encoded,
                           const RowRangeList& ranges,
                           T* dst, size_t n) override {
        if constexpr (kProfileSections) {
            gatherSkipNs_ = 0;
            gatherMatNs_ = 0;
        }

        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;

            if constexpr (kProfileSections) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                decodeRangeInto(encoded, r.begin, r.end, dst + off, count);
                gatherMatNs_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count();
            } else {
                decodeRangeInto(encoded, r.begin, r.end, dst + off, count);
            }
            off += count;
        }
        if (off != n) [[unlikely]]
            throw std::runtime_error("SubIntSplitEncoder::decodeGatherInto: decoded size mismatch");
    }

    EncodingType encodingType() const override { return EncodingType::Structural; }

    std::string name() const override {
        std::string bitsStr;
        for (size_t i = 0; i < cfg_.bits.size(); ++i) {
            bitsStr += std::to_string(cfg_.bits[i]);
            if (i + 1 < cfg_.bits.size()) bitsStr += "|";
        }
        return "SubIntSplit(" + bitsStr + (cfg_.order == BitSplitOrder::LSB_TO_MSB ? ",LSB" : ",MSB") + ")";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        EncodingProperties props;
        props.add(EP::Lossless)
             .add(EP::PreservesOrder)
             .add(EP::Composable);
        if (allSectionsRandomAccess()) props.add(EP::RandomAccess).add(EP::FastSkip);
        return props;
    }

private:
    static bool verboseEnabled() {
        static bool v = (std::getenv("SUBINTSPLIT_VERBOSE") != nullptr);
        return v || true;
    }

    struct ParsedHeader {
        size_t N{};
        BitSplitOrder order{BitSplitOrder::LSB_TO_MSB};
        std::vector<uint8_t> bits;
        std::vector<uint8_t> shifts;  // precomputed absolute bit-shift per section
        std::vector<uint64_t> bytes;
        std::vector<EncodedBuffer<uint8_t>> views;
    };

    static void writeU64(uint8_t* dst, uint64_t v) { std::memcpy(dst, &v, sizeof(uint64_t)); }
    static uint64_t readU64(const uint8_t* src) { uint64_t v; std::memcpy(&v, src, sizeof(uint64_t)); return v; }

    size_t headerSizeBytes(size_t splits) const { return 8 + 1 + 1 + splits * (1 + 8); }

    const ParsedHeader& getCachedHeader(const EncodedBuffer<uint8_t>& encoded) const {
        const uint8_t* basePtr = encoded.data().data();
        const size_t totalSize = encoded.data().size();
        // Include the first 8 bytes (the N field) in the cache key.  Without this,
        // two encodes that produce the same byte count (e.g. constant data for N=500
        // and N=20000) can map to the same (address, size) pair after the allocator
        // reuses a freed buffer, returning a stale header with the wrong N.
        const uint64_t firstBytes = (totalSize >= 8) ? readU64(basePtr) : 0;
        if (cachedOuterPtr_ != basePtr || cachedOuterSize_ != totalSize
                || cachedFirstBytes_ != firstBytes) {
            cachedHeader_ = parseHeader(encoded);
            cachedOuterPtr_ = basePtr;
            cachedOuterSize_ = totalSize;
            cachedFirstBytes_ = firstBytes;
        }
        return cachedHeader_;
    }

    ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& encoded) const {
        const auto& buf = encoded.data();
        if (buf.size() < 10) {
            throw std::runtime_error("SubIntSplitEncoder: buffer too small for header");
        }
        const uint8_t* p = buf.data();
        ParsedHeader h;
        h.N = static_cast<size_t>(readU64(p)); p += 8;
        const uint8_t splits = *p++;
        h.order = static_cast<BitSplitOrder>(*p++);
        if (splits == 0 || splits > kTotalBits) {
            throw std::runtime_error("SubIntSplitEncoder: invalid split count in header");
        }
        if (cfg_.splitCount() != splits) {
            throw std::runtime_error("SubIntSplitEncoder: header split count does not match config");
        }
        if (cfg_.order != h.order) {
            throw std::runtime_error("SubIntSplitEncoder: header order does not match config");
        }
        const size_t expectedHeader = headerSizeBytes(splits);
        if (buf.size() < expectedHeader) {
            throw std::runtime_error("SubIntSplitEncoder: buffer too small for split metadata");
        }
        h.bits.resize(splits);
        h.bytes.resize(splits);
        for (size_t s = 0; s < splits; ++s) {
            h.bits[s] = *p++;
            h.bytes[s] = readU64(p); p += 8;
        }
        if (h.bits != cfg_.bits) {
            throw std::runtime_error("SubIntSplitEncoder: header bits do not match config");
        }
        // Validate bit sum
        uint16_t totalBits = 0;
        for (uint8_t b : h.bits) {
            if (b == 0 || b > kTotalBits) throw std::runtime_error("SubIntSplitEncoder: invalid bit width in header");
            totalBits = static_cast<uint16_t>(totalBits + b);
        }
        if (totalBits != kTotalBits) {
            throw std::runtime_error("SubIntSplitEncoder: header bits do not sum to expected width");
        }
        // Precompute absolute shifts so combine() can be done with a single OR per section.
        h.shifts.resize(splits);
        if (h.order == BitSplitOrder::LSB_TO_MSB) {
            uint8_t shift = 0;
            for (size_t s = 0; s < splits; ++s) {
                h.shifts[s] = shift;
                shift = static_cast<uint8_t>(shift + h.bits[s]);
            }
        } else {
            // MSB_TO_LSB: section s lands at (kTotalBits - cumsum(bits[0..s]))
            uint8_t remaining = kTotalBits;
            for (size_t s = 0; s < splits; ++s) {
                remaining = static_cast<uint8_t>(remaining - h.bits[s]);
                h.shifts[s] = remaining;
            }
        }
        // Build section views
        h.views.reserve(splits);
        size_t offset = expectedHeader;
        for (size_t s = 0; s < splits; ++s) {
            const uint64_t len = h.bytes[s];
            if (offset + len > buf.size()) {
                throw std::runtime_error("SubIntSplitEncoder: payload sizes exceed buffer");
            }
            h.views.push_back(slice(encoded, offset, len, h.N, h.bits[s]));
            offset += static_cast<size_t>(len);
        }
        return h;
    }

    static EncodedBuffer<uint8_t> slice(const EncodedBuffer<uint8_t>& base,
                                        size_t off, size_t len,
                                        size_t elementCount, uint8_t bits) {
        const uint8_t* src = base.data().data() + off;
        std::vector<uint8_t> payload(src, src + len);

        // Reconstruct minimal metadata so sub-codecs know expected uncompressed size
        const uint8_t chosenBits = (bits <= 8)  ? 8  : (bits <= 16) ? 16
                                   : (bits <= 32) ? 32 : 64;
        const size_t elemBytes   = static_cast<size_t>(chosenBits) / 8;

        encodings::EncodingMetadata meta;
        meta.encodingName     = "SubIntSplitSection";
        meta.elementCount     = elementCount;
        meta.compressedSize   = len;
        meta.uncompressedSize = elementCount * elemBytes;

        switch (chosenBits) {
            case 8:  meta.dataType = encodings::core::typeToDataType<uint8_t>;  break;
            case 16: meta.dataType = encodings::core::typeToDataType<uint16_t>; break;
            case 32: meta.dataType = encodings::core::typeToDataType<uint32_t>; break;
            default: meta.dataType = encodings::core::typeToDataType<uint64_t>; break;
        }
        return encodings::EncodedData(std::move(payload), std::move(meta));
    }

    bool allSectionsRandomAccess() const {
        auto has = [](const EncodingProperties& p) {
            return p.has(EncodingProperty::RandomAccess);
        };
        for (const auto& c : cfg_.codecs) {
            if (!has(c->properties())) return false;
        }
        return true;
    }

    static SectionT decodeOneSection(ISectionCodecIntegral<SectionT>& c, const EncodedBuffer<uint8_t>& view, size_t idx) {
        if (c.properties().has(EncodingProperty::RandomAccess)) {
            auto v = c.decodeAt(view, idx);
            if (!v) throw std::runtime_error("SubIntSplitEncoder::decodeAt: subcodec returned null");
            return *v;
        }
        auto all = c.decodeAll(view);
        if (idx >= all.size()) throw std::runtime_error("SubIntSplitEncoder::decodeAt: index out of range");
        return all[idx];
    }

    // Initialise an uninitialised accumulator from section 0: stores src[i] << shift
    // into dst[i] with no read of dst (pure store, no OR). Required when the
    // accumulator was allocated with make_unique_for_overwrite.
    // shift == 0 fast-paths to memcpy (always true for LSB_TO_MSB section 0).
    static void assignSectionInto(const SectionT* __restrict__ src,
                                   SectionT* __restrict__ dst,
                                   size_t N, uint8_t shift) {
        if (shift == 0) {
            std::memcpy(dst, src, N * sizeof(SectionT));
            return;
        }
#ifdef __AVX2__
        if constexpr (sizeof(SectionT) == 8) {
            const __m128i vshift = _mm_cvtsi64_si128(static_cast<int64_t>(shift));
            size_t i = 0;
            for (; i + 4 <= N; i += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(dst + i + 32), _MM_HINT_T1);
                __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                vs = _mm256_sll_epi64(vs, vshift);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vs);
            }
            for (; i < N; ++i) dst[i] = src[i] << shift;
        } else {
            const __m128i vshift = _mm_cvtsi32_si128(static_cast<int32_t>(shift));
            size_t i = 0;
            for (; i + 8 <= N; i += 8) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(dst + i + 64), _MM_HINT_T1);
                __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                vs = _mm256_sll_epi32(vs, vshift);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vs);
            }
            for (; i < N; ++i) dst[i] = src[i] << shift;
        }
#else
        for (size_t i = 0; i < N; ++i) dst[i] = src[i] << shift;
#endif
    }

    // Column-major combine kernel: OR src[i] << shift into dst[i] for all i.
    // shift is constant across all elements in a section — the loop is trivially
    // SIMD-vectorizable. AVX2 path processes 4 (64-bit) or 8 (32-bit) elements
    // per iteration without any per-element allocation.
    static void combineSectionInto(const SectionT* __restrict__ src,
                                   SectionT* __restrict__ dst,
                                   size_t N, uint8_t shift) {
#ifdef __AVX2__
        if constexpr (sizeof(SectionT) == 8) {
            // 64-bit sections: 4 elements per 256-bit register.
            const __m128i vshift = _mm_cvtsi64_si128(static_cast<int64_t>(shift));
            size_t i = 0;
            for (; i + 4 <= N; i += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(dst + i + 32), _MM_HINT_T1);
                __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
                vs = _mm256_sll_epi64(vs, vshift);
                vd = _mm256_or_si256(vd, vs);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vd);
            }
            for (; i < N; ++i) dst[i] |= src[i] << shift;
        } else {
            // 32-bit sections: 8 elements per 256-bit register.
            const __m128i vshift = _mm_cvtsi32_si128(static_cast<int32_t>(shift));
            size_t i = 0;
            for (; i + 8 <= N; i += 8) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(dst + i + 64), _MM_HINT_T1);
                __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
                vs = _mm256_sll_epi32(vs, vshift);
                vd = _mm256_or_si256(vd, vs);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vd);
            }
            for (; i < N; ++i) dst[i] |= src[i] << shift;
        }
#else
        // Scalar fallback — auto-vectorizer-friendly (constant shift, no aliasing).
        for (size_t i = 0; i < N; ++i) dst[i] |= src[i] << shift;
#endif
    }

    EncodedBuffer<uint8_t> makeEmpty() const {
        const size_t splits = cfg_.splitCount();
        if (splits == 0) {
            throw std::invalid_argument("SubIntSplitEncoder: cannot encode empty config with zero splits");
        }
        const size_t headerSize = headerSizeBytes(splits);
        std::vector<uint8_t> out(headerSize, 0);
        uint8_t* p = out.data();
        writeU64(p, 0); p += 8;
        *p++ = static_cast<uint8_t>(splits);
        *p++ = static_cast<uint8_t>(cfg_.order);
        for (size_t s = 0; s < splits; ++s) {
            *p++ = cfg_.bits[s];
            writeU64(p, 0);
            p += 8;
        }
        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<T>;
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = allSectionsRandomAccess();
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

public:
    // ── Per-section decode-timing virtual overrides ──────────────────────────
    // Only meaningful when EnableProfiling=true; the false variant returns empty
    // via the Codec base-class defaults, so no vtable cost beyond the call site.

    std::vector<int64_t> subStreamBulkDecodeTimeNs() const override {
        if constexpr (kProfileSections) return bulkDecodeTimeNs_; else return {};
    }
    std::vector<int64_t> subStreamDecodeAtAccumNs() const override {
        if constexpr (kProfileSections) return decodeAtAccumNs_;  else return {};
    }
    std::vector<int64_t> subStreamDecodeRangeAccumNs() const override {
        if constexpr (kProfileSections) return decodeRangeAccumNs_; else return {};
    }
    void resetSubStreamDecodeAtAccum() override {
        if constexpr (kProfileSections) decodeAtAccumNs_.assign(cfg_.splitCount(), 0);
    }
    void resetSubStreamDecodeRangeAccum() override {
        if constexpr (kProfileSections) decodeRangeAccumNs_.assign(cfg_.splitCount(), 0);
    }

    // ── Gather (selective row-range) profiling overrides ──────────────────
    int64_t gatherSkipTimeNs() const override {
        if constexpr (kProfileSections) return gatherSkipNs_; else return -1;
    }
    int64_t gatherMaterializeTimeNs() const override {
        if constexpr (kProfileSections) return gatherMatNs_; else return -1;
    }
    void resetGatherProfilingAccum() override {
        if constexpr (kProfileSections) { gatherSkipNs_ = 0; gatherMatNs_ = 0; }
    }

    // Returns the number of threads to use, capped by both workUnits and the
    // configured maxThreads (or hardware_concurrency() when maxThreads==0).
    size_t effectiveThreadCount(size_t workUnits) const {
        const size_t hw  = std::thread::hardware_concurrency();
        const size_t cap = (cfg_.parallelism.maxThreads > 0)
                             ? cfg_.parallelism.maxThreads
                             : (hw > 0 ? hw : 1);
        return std::min(workUnits, cap);
    }

private:
    SubIntSplitConfigIntegral<SectionT> cfg_;
    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t         cachedOuterSize_{0};
    mutable uint64_t       cachedFirstBytes_{0};
    mutable ParsedHeader   cachedHeader_{};

    // Per-section decode timing state — zero-size when EnableProfiling=false.
    struct NoDecodeState {};
    using DecodeTimingVec = std::conditional_t<EnableProfiling, std::vector<int64_t>, NoDecodeState>;
    [[no_unique_address]] mutable DecodeTimingVec bulkDecodeTimeNs_;
    [[no_unique_address]] mutable DecodeTimingVec decodeAtAccumNs_;
    [[no_unique_address]] mutable DecodeTimingVec decodeRangeAccumNs_;

    // Gather (selective row-range) skip/materialize timing accumulators —
    // zero-size when EnableProfiling=false.
    struct NoGatherState {};
    using GatherTimeT = std::conditional_t<EnableProfiling, int64_t, NoGatherState>;
    [[no_unique_address]] mutable GatherTimeT gatherSkipNs_{};
    [[no_unique_address]] mutable GatherTimeT gatherMatNs_{};
};

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

template <typename T>
inline std::shared_ptr<SubIntSplitEncoder<T>> makeSubIntSplitEncoder(SubIntSplitConfigIntegral<std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>> cfg) {
    return std::make_shared<SubIntSplitEncoder<T>>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoder<T>> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    auto cfg = SubIntSplitConfigIntegral<SectionT>::fromSegments(segments);
    return makeSubIntSplitEncoder<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoder<T>> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    auto cfg = SubIntSplitConfigIntegral<SectionT>::fromSegments(segments, orderHint);
    return makeSubIntSplitEncoder<T>(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Manual configuration helpers
// ---------------------------------------------------------------------------

template <typename T>
inline std::shared_ptr<SubIntSplitEncoder<T>> makeSubIntSplitEncoderManual(
    const std::vector<uint8_t>& bits,
    const std::vector<encodings::EncodingType>& encodings,
    BitSplitOrder orderHint = BitSplitOrder::LSB_TO_MSB) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    if (bits.size() != encodings.size()) {
        throw std::invalid_argument("makeSubIntSplitEncoderManual: bits/encodings size mismatch");
    }

    SubIntSplitConfigIntegral<SectionT> cfg;
    cfg.order = orderHint;
    cfg.bits = bits;
    cfg.codecs.reserve(bits.size());

    for (size_t i = 0; i < bits.size(); ++i) {
        const uint8_t width = bits[i];
        switch (encodings[i]) {
            case encodings::EncodingType::RawEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRawSection<SectionT>(width));
                break;
            case encodings::EncodingType::DictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeDictionarySection<SectionT>(width));
                break;
            case encodings::EncodingType::AdaptiveDictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveDictionarySection<SectionT>(width));
                break;
            case encodings::EncodingType::FrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeFORSection<512, SectionT>(width));
                break;
            case encodings::EncodingType::AdaptiveFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFORSection<SectionT>(width));
                break;
            case encodings::EncodingType::OpenZL:
                cfg.codecs.push_back(detail_trisplit::makeOpenZLSection<0, SectionT>(width));
                break;
            case encodings::EncodingType::BitPacking:
                cfg.codecs.push_back(detail_trisplit::makeRawBitPackedSection<SectionT>(width));
                break;
            case encodings::EncodingType::AdaptiveFramedBitPrefix:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFramedBitPrefixSection<SectionT>(width));
                break;
            case encodings::EncodingType::RunLengthEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLESection<SectionT>(width));
                break;
            case encodings::EncodingType::HuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeHuffmanSection<SectionT>(width));
                break;
            case encodings::EncodingType::LZ4:
                cfg.codecs.push_back(detail_trisplit::makeLZ4Section<SectionT>(width));
                break;
            case encodings::EncodingType::FSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeFSESection<SectionT>(width));
                break;
            case encodings::EncodingType::FrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::BlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::BlockFrequencyPartitionFOREncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionFORSection<SectionT>(width));
                break;
            case encodings::EncodingType::BlockFORFPEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFORFPESection<SectionT>(width));
                break;
            case encodings::EncodingType::RangePackFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRangePackFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRangePackBlockFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::RunLengthCascadingFOREncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLECascadingFORSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORFSESection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFSESection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORHuffmanSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFSESection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFSESection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevHuffmanSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFrequencyPartitionSection<SectionT>(width));
                break;
            case encodings::EncodingType::BlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFSESection<SectionT>(width));
                break;
            default:
                throw std::invalid_argument("makeSubIntSplitEncoderManual: unsupported encoding type: " + std::to_string(static_cast<int>(encodings[i])));
        }
    }

    cfg.validate();
    return makeSubIntSplitEncoder<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoder<T>> makeSubIntSplitEncoderManual(
    const std::vector<uint8_t>& bits,
    std::vector<std::shared_ptr<ISectionCodecIntegral<std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>>>> codecs,
    BitSplitOrder orderHint = BitSplitOrder::LSB_TO_MSB) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    SubIntSplitConfigIntegral<SectionT> cfg;
    cfg.order = orderHint;
    cfg.bits = bits;
    cfg.codecs = std::move(codecs);
    cfg.validate();
    return makeSubIntSplitEncoder<T>(std::move(cfg));
}

using SubIntSplitEncoder64 = SubIntSplitEncoder<int64_t>;
using SubIntSplitEncoder32 = SubIntSplitEncoder<int32_t>;

// Profiling variants — identical behaviour but fill subStreamEncodeMetrics and
// expose per-section decode times via the subStream*() virtual methods.
template<typename T> using SubIntSplitEncoderProf = SubIntSplitEncoder<T, true>;
using SubIntSplitEncoder64Prof = SubIntSplitEncoder<int64_t, true>;
using SubIntSplitEncoder32Prof = SubIntSplitEncoder<int32_t, true>;

// ---------------------------------------------------------------------------
// Profiling factory functions for SubIntSplitEncoder<T, true>
// Mirror every non-profiling factory; return profiling variants.
// ---------------------------------------------------------------------------

template <typename T>
inline std::shared_ptr<SubIntSplitEncoderProf<T>> makeSubIntSplitEncoderProf(
    SubIntSplitConfigIntegral<std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>> cfg) {
    return std::make_shared<SubIntSplitEncoderProf<T>>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoderProf<T>> makeSubIntSplitEncoderFromSegmentsProf(
    const std::vector<selectors::SegmentPlan>& segments) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    auto cfg = SubIntSplitConfigIntegral<SectionT>::fromSegments(segments);
    return makeSubIntSplitEncoderProf<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoderProf<T>> makeSubIntSplitEncoderFromSegmentsProf(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    auto cfg = SubIntSplitConfigIntegral<SectionT>::fromSegments(segments, orderHint);
    return makeSubIntSplitEncoderProf<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoderProf<T>> makeSubIntSplitEncoderManualProf(
    const std::vector<uint8_t>& bits,
    const std::vector<encodings::EncodingType>& encs,
    BitSplitOrder orderHint = BitSplitOrder::LSB_TO_MSB) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    if (bits.size() != encs.size())
        throw std::invalid_argument("makeSubIntSplitEncoderManualProf: bits/encodings size mismatch");
    SubIntSplitConfigIntegral<SectionT> cfg;
    cfg.order = orderHint;
    cfg.bits  = bits;
    cfg.codecs.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        const uint8_t width = bits[i];
        switch (encs[i]) {
            case encodings::EncodingType::RawEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRawSection<SectionT>(width)); break;
            case encodings::EncodingType::DictionaryEncoding:
                cfg.codecs.push_back(detail_trisplit::makeDictionarySection<SectionT>(width)); break;
            case encodings::EncodingType::FrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeFORSection<512, SectionT>(width)); break;
            case encodings::EncodingType::AdaptiveFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFORSection<SectionT>(width)); break;
            case encodings::EncodingType::OpenZL:
                cfg.codecs.push_back(detail_trisplit::makeOpenZLSection<0, SectionT>(width)); break;
            case encodings::EncodingType::BitPacking:
                cfg.codecs.push_back(detail_trisplit::makeRawBitPackedSection<SectionT>(width)); break;
            case encodings::EncodingType::AdaptiveFramedBitPrefix:
                cfg.codecs.push_back(detail_trisplit::makeAdaptiveFramedBitPrefixSection<SectionT>(width)); break;
            case encodings::EncodingType::RunLengthEncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLESection<SectionT>(width)); break;
            case encodings::EncodingType::HuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeHuffmanSection<SectionT>(width)); break;
            case encodings::EncodingType::LZ4:
                cfg.codecs.push_back(detail_trisplit::makeLZ4Section<SectionT>(width)); break;
            case encodings::EncodingType::FSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeFSESection<SectionT>(width)); break;
            case encodings::EncodingType::FrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeFrequencyPartitionSection<SectionT>(width)); break;
            case encodings::EncodingType::BlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionSection<SectionT>(width)); break;
            case encodings::EncodingType::BlockFrequencyPartitionFOREncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFrequencyPartitionFORSection<SectionT>(width)); break;
            case encodings::EncodingType::BlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeBlockFSESection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFrameOfReference:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORSection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFrequencyPartitionSection<SectionT>(width)); break;
            case encodings::EncodingType::RunLengthCascadingFOREncoding:
                cfg.codecs.push_back(detail_trisplit::makeRLECascadingFORSection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORFSESection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORBlockFSESection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORHuffmanSection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORPrevFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFSESection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORPrevBlockFSEEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFSESection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORPrevHuffmanEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevHuffmanSection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevFrequencyPartitionSection<SectionT>(width)); break;
            case encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
                cfg.codecs.push_back(detail_trisplit::makeCascadingFORPrevBlockFrequencyPartitionSection<SectionT>(width)); break;
            default:
                throw std::invalid_argument("makeSubIntSplitEncoderManualProf: unsupported encoding type");
        }
    }
    cfg.validate();
    return makeSubIntSplitEncoderProf<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitEncoderProf<T>> makeSubIntSplitEncoderManualProf(
    const std::vector<uint8_t>& bits,
    std::vector<std::shared_ptr<ISectionCodecIntegral<std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>>>> codecs,
    BitSplitOrder orderHint = BitSplitOrder::LSB_TO_MSB) {
    using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
    SubIntSplitConfigIntegral<SectionT> cfg;
    cfg.order  = orderHint;
    cfg.bits   = bits;
    cfg.codecs = std::move(codecs);
    cfg.validate();
    return makeSubIntSplitEncoderProf<T>(std::move(cfg));
}

// int64_t convenience overloads (no explicit template argument needed)
inline std::shared_ptr<SubIntSplitEncoder64Prof> makeSubIntSplitEncoderProf(
    SubIntSplitConfigIntegral<uint64_t> cfg) {
    return makeSubIntSplitEncoderProf<int64_t>(std::move(cfg));
}

inline std::shared_ptr<SubIntSplitEncoder64Prof> makeSubIntSplitEncoderFromSegmentsProf(
    const std::vector<selectors::SegmentPlan>& segments) {
    return makeSubIntSplitEncoderFromSegmentsProf<int64_t>(segments);
}

inline std::shared_ptr<SubIntSplitEncoder64Prof> makeSubIntSplitEncoderFromSegmentsProf(
    const std::vector<selectors::SegmentPlan>& segments, BitSplitOrder orderHint) {
    return makeSubIntSplitEncoderFromSegmentsProf<int64_t>(segments, orderHint);
}

inline std::shared_ptr<SubIntSplitEncoder64Prof> makeSubIntSplitEncoderManualProf(
    const std::vector<uint8_t>& bits,
    const std::vector<encodings::EncodingType>& encs,
    BitSplitOrder orderHint = BitSplitOrder::LSB_TO_MSB) {
    return makeSubIntSplitEncoderManualProf<int64_t>(bits, encs, orderHint);
}

// ---------------------------------------------------------------------------
// Non-profiling int64_t convenience overloads (unchanged)
// ---------------------------------------------------------------------------

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoder(SubIntSplitConfigIntegral<uint64_t> cfg) {
    return makeSubIntSplitEncoder<int64_t>(std::move(cfg));
}

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments) {
    return makeSubIntSplitEncoderFromSegments<int64_t>(segments);
}

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint) {
    return makeSubIntSplitEncoderFromSegments<int64_t>(segments, orderHint);
}

// ---------------------------------------------------------------------------
// Auto-selecting wrapper (lazy instantiation via sampling + selector)
// ---------------------------------------------------------------------------

// Config is independent of EnableProfiling — defined once so that the profiling
// and non-profiling variants share the same Config type and factory helpers can
// return either without any type-conversion overhead.
template <typename T>
struct SubIntSplitAutoEncoderConfig {
    selectors::IDSubStreamEncodingSelector::Config selectorConfig{};
    generators::samplers::StreamSampler<T>::Config samplerConfig{};
    std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>> costModels;
    std::vector<std::unique_ptr<selectors::costs::ISubStreamReordererCostModel>> reordererModels;
    std::optional<BitSplitOrder> orderHint{}; // allow forcing MSB/LSB ordering
    bool debugLogging{true};
    bool enableSelectionTiming{false};
    // Parallelism applied to the resolved inner SubIntSplitEncoder at runtime.
    SubIntSplitParallelismConfig parallelism{};
    // When true, FPE section codecs are constructed with parallelDecode enabled.
    // Defaults to true so AutoSIS benefits from intra-FPE tier parallelism by default.
    bool fpeSectionParallelDecode{true};
};

template <typename T, bool EnableProfiling = false>
    requires (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>)
class SubIntSplitAutoEncoder final : public Codec<T, uint8_t> {
public:
    using Config = SubIntSplitAutoEncoderConfig<T>;

    explicit SubIntSplitAutoEncoder(Config cfg)
        : selector_(cfg.selectorConfig), samplerCfg_(cfg.samplerConfig), orderHint_(cfg.orderHint), debugLogging_(cfg.debugLogging),
          selectionTimingEnabled_(cfg.enableSelectionTiming), parallelism_(cfg.parallelism),
          fpeSectionParallelDecode_(cfg.fpeSectionParallelDecode) {
        if (cfg.costModels.empty()) {
            throw std::invalid_argument("SubIntSplitAutoEncoder: costModels must not be empty");
        }
        costModels_     = std::move(cfg.costModels);
        reordererModels_ = std::move(cfg.reordererModels);
    }

    EncodedBuffer<uint8_t> encode(std::span<const T> data) override {
        ensureEncoder(data);
        auto enc = impl_->encode(data);
        if (selectionTimingEnabled_ && lastSelectionTimeNs_.has_value()) {
            enc.metadata().customMetadata["selectionTime_ns"] = std::to_string(lastSelectionTimeNs_->count());
        }
        if (debugLogging_) {
            logActualEncoding(enc, data.size());
        }
        return enc;
    }

    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        ensureDecoderInitialized();
        return impl_->decodeAll(encoded);
    }

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        ensureDecoderInitialized();
        return impl_->decodeAt(encoded, index);
    }

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        ensureDecoderInitialized();
        return impl_->decodeRange(encoded, start, end);
    }

    void decodeRangeInto(const EncodedBuffer<uint8_t>& encoded,
                          size_t start, size_t end,
                          T* dst, size_t n) override {
        ensureDecoderInitialized();
        impl_->decodeRangeInto(encoded, start, end, dst, n);
    }

    // Forward to impl_'s decodeGatherInto so the auto-selected encoder still
    // gets its genuine skip-then-materialize fast path (and skip/materialize
    // timing split below) instead of silently falling back to the base
    // Decoder::decodeGatherInto's independent-decodeRangeInto-per-range loop.
    void decodeGatherInto(const EncodedBuffer<uint8_t>& encoded,
                           const RowRangeList& ranges,
                           T* dst, size_t n) override {
        ensureDecoderInitialized();
        impl_->decodeGatherInto(encoded, ranges, dst, n);
    }

    EncodingType encodingType() const override { return EncodingType::Structural; }

    std::string name() const override {
        if (impl_) return impl_->name();
        return "SubIntSplitAuto";
    }

    EncodingProperties properties() const override {
        if (impl_) return impl_->properties();
        using EP = EncodingProperty;
        EncodingProperties props;
        props.add(EP::Lossless).add(EP::PreservesOrder).add(EP::Composable);
        // RandomAccess unknown until encoder chosen
        return props;
    }

    // Forward per-section decode-timing methods to impl_ (only populated when
    // EnableProfiling=true; otherwise impl_->subStream*() returns empty by default).
    std::vector<int64_t> subStreamBulkDecodeTimeNs()   const override {
        return impl_ ? impl_->subStreamBulkDecodeTimeNs()   : std::vector<int64_t>{};
    }
    std::vector<int64_t> subStreamDecodeAtAccumNs()    const override {
        return impl_ ? impl_->subStreamDecodeAtAccumNs()    : std::vector<int64_t>{};
    }
    std::vector<int64_t> subStreamDecodeRangeAccumNs() const override {
        return impl_ ? impl_->subStreamDecodeRangeAccumNs() : std::vector<int64_t>{};
    }
    void resetSubStreamDecodeAtAccum() override    { if (impl_) impl_->resetSubStreamDecodeAtAccum(); }
    void resetSubStreamDecodeRangeAccum() override { if (impl_) impl_->resetSubStreamDecodeRangeAccum(); }

    // Forward gather (selective row-range) profiling to impl_ for the same
    // reason as the subStream* forwarding above.
    int64_t gatherSkipTimeNs() const override {
        return impl_ ? impl_->gatherSkipTimeNs() : -1;
    }
    int64_t gatherMaterializeTimeNs() const override {
        return impl_ ? impl_->gatherMaterializeTimeNs() : -1;
    }
    void resetGatherProfilingAccum() override { if (impl_) impl_->resetGatherProfilingAccum(); }

    void reset() override {
        impl_.reset();
        lastSelection_       = selectors::IDSubStreamEncodingSelector::Result{};
        lastSelectionTimeNs_.reset();
    }

private:
    using UnsignedT = std::make_unsigned_t<T>;
    static constexpr uint8_t kTotalBits = static_cast<uint8_t>(sizeof(T) * 8);

    void ensureEncoder(std::span<const T> data) {
        if (impl_) return;
        if (data.empty()) {
            throw std::invalid_argument("SubIntSplitAutoEncoder: cannot auto-select on empty input");
        }

        const auto selectionStart = std::chrono::high_resolution_clock::now();

        // Sample input (auto-adjust stride to cover the full domain if not specified)
        auto effectiveCfg = samplerCfg_;
        if (effectiveCfg.stride == 0) {
            const size_t n = data.size();
            size_t target = 0;
            if (effectiveCfg.maxPercentage > 0.0) {
                target = static_cast<size_t>(std::ceil(effectiveCfg.maxPercentage * static_cast<double>(n)));
            }
            if (effectiveCfg.maxSamples > 0 && (target == 0 || effectiveCfg.maxSamples < target)) {
                target = effectiveCfg.maxSamples;
            }
            if (target == 0) target = 1; // fallback: take at least one
            effectiveCfg.stride = std::max<size_t>(1, n / target);
        }

        auto sample = generators::samplers::StreamSampler<T>::sample(data, effectiveCfg);
        if (sample.empty()) {
            // Fallback to a minimal sample (first element)
            sample.push_back(data.front());
        }

        // If MSB order is requested, mirror bits in the sample so the selector's LSB-oriented DP
        // sees the highest bits first. Later we map segment positions back to the original domain.
        std::vector<T> transformedSample;
        if (orderHint_.has_value() && *orderHint_ == BitSplitOrder::MSB_TO_LSB) {
            transformedSample.reserve(sample.size());
            for (auto v : sample) {
                transformedSample.push_back(static_cast<T>(mirrorBits(static_cast<UnsignedT>(v))));
            }
        }

        // Selector needs non-const cost models vector reference; ensure they exist
        const auto& selectorInput = (orderHint_.has_value() && *orderHint_ == BitSplitOrder::MSB_TO_LSB)
                                        ? transformedSample
                                        : sample;
        lastSelection_ = selector_.select(selectorInput, costModels_, reordererModels_, data.size());
        if (lastSelection_.segments.empty()) {
            throw std::runtime_error("SubIntSplitAutoEncoder: selector returned no segments");
        }

        if (selectionTimingEnabled_) {
            const auto selectionEnd = std::chrono::high_resolution_clock::now();
            lastSelectionTimeNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(selectionEnd - selectionStart);
        }

        if (debugLogging_) {
            std::cout << "[AutoSubIntSplit] sample_size=" << sample.size()
                      << " stride=" << effectiveCfg.stride
                      << " selection_cost_bits=" << lastSelection_.total_cost << "\n";
            for (const auto& seg : lastSelection_.segments) {
                std::cout << "  seg [" << seg.bitStart << ".." << seg.bitEnd << "] "
                          << encodings::encodingTypeToString(seg.encoding)
                          << " cost=" << seg.cost << "\n";
            }
        }

        std::vector<selectors::SegmentPlan> segments = lastSelection_.segments;
        if (orderHint_.has_value() && *orderHint_ == BitSplitOrder::MSB_TO_LSB) {
            segments = remapMirroredSegmentsToOriginal(segments);
        }

        using SectionT = std::conditional_t<(sizeof(T) <= 4), uint32_t, uint64_t>;
        typename SubIntSplitConfigIntegral<SectionT>::SectionBuildOptions sectionOpts;
        sectionOpts.fpeSectionParallelDecode = fpeSectionParallelDecode_;
        auto cfg = orderHint_.has_value()
            ? SubIntSplitConfigIntegral<SectionT>::fromSegments(segments, *orderHint_, sectionOpts)
            : SubIntSplitConfigIntegral<SectionT>::fromSegments(segments, sectionOpts);
        cfg.parallelism = parallelism_;
        // Reflect remapped segments in the cached selection for accurate logging
        lastSelection_.segments = segments;
        impl_ = std::make_shared<SubIntSplitEncoder<T, EnableProfiling>>(std::move(cfg));
    }

    void ensureDecoderInitialized() const {
        if (!impl_) {
            throw std::runtime_error("SubIntSplitAutoEncoder: encoder not initialized; call encode first");
        }
    }

private:
    selectors::IDSubStreamEncodingSelector selector_{};
    generators::samplers::StreamSampler<T>::Config samplerCfg_{};
    std::optional<BitSplitOrder> orderHint_{};
    std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>> costModels_;
    std::vector<std::unique_ptr<selectors::costs::ISubStreamReordererCostModel>> reordererModels_;
    std::shared_ptr<SubIntSplitEncoder<T, EnableProfiling>> impl_{};
    bool debugLogging_{true};
    bool selectionTimingEnabled_{false};
    SubIntSplitParallelismConfig parallelism_{};
    bool fpeSectionParallelDecode_{true};
    std::optional<std::chrono::nanoseconds> lastSelectionTimeNs_{};
    selectors::IDSubStreamEncodingSelector::Result lastSelection_{};

    static UnsignedT mirrorBits(UnsignedT x) {
        // simple bit-reversal; sample sizes are small
        UnsignedT y = 0;
        for (int i = 0; i < kTotalBits; ++i) {
            y <<= 1;
            y |= (x & 1U);
            x >>= 1;
        }
        return y;
    }

    static std::vector<selectors::SegmentPlan> remapMirroredSegmentsToOriginal(const std::vector<selectors::SegmentPlan>& segments) {
        std::vector<selectors::SegmentPlan> out;
        out.reserve(segments.size());
        const int maxBitIndex = static_cast<int>(kTotalBits - 1);
        for (const auto& seg : segments) {
            selectors::SegmentPlan mapped = seg;
            mapped.bitStart = maxBitIndex - seg.bitEnd;
            mapped.bitEnd   = maxBitIndex - seg.bitStart;
            out.push_back(mapped);
        }
        return out;
    }

    void logActualEncoding(const EncodedBuffer<uint8_t>& encoded, size_t /*inputSize*/) const {
        const auto& buf = encoded.data();
        if (buf.size() < 10) {
            std::cout << "[AutoSubIntSplit] encoded buffer too small for debug" << std::endl;
            return;
        }
        auto readU64Local = [](const uint8_t* src) {
            uint64_t v;
            std::memcpy(&v, src, sizeof(uint64_t));
            return v;
        };

        const uint8_t* p = buf.data();
        const uint64_t N = readU64Local(p); p += 8;
        const uint8_t splits = *p++;
        const BitSplitOrder order = static_cast<BitSplitOrder>(*p++);
        std::vector<uint8_t> bits;
        std::vector<uint64_t> bytes;
        bits.reserve(splits);
        bytes.reserve(splits);
        for (size_t s = 0; s < splits; ++s) {
            bits.push_back(*p++);
            bytes.push_back(readU64Local(p));
            p += 8;
        }
        const size_t headerBytes = 8 + 1 + 1 + splits * (1 + 8);
        uint64_t payloadBytes = 0;
        for (auto b : bytes) payloadBytes += b;
        const uint64_t totalBytes = headerBytes + payloadBytes;
        const double bitsPerElem = N ? (static_cast<double>(totalBytes) * 8.0 / static_cast<double>(N)) : 0.0;

        std::cout << "[AutoSubIntSplit] actual encoding: N=" << N
                  << " splits=" << static_cast<int>(splits)
                  << " order=" << (order == BitSplitOrder::LSB_TO_MSB ? "LSB" : "MSB")
                  << " total_bytes=" << totalBytes
                  << " header_bytes=" << headerBytes
                  << " payload_bytes=" << payloadBytes
                  << " bits/elem=" << bitsPerElem
                  << " selection_cost_bits=" << lastSelection_.total_cost
                  << "\n";
        for (size_t s = 0; s < splits; ++s) {
            std::cout << "  sec " << s << " bits=" << static_cast<int>(bits[s])
                      << " bytes=" << bytes[s] << " bits/elem=" << (N ? (static_cast<double>(bytes[s]) * 8.0 / static_cast<double>(N)) : 0.0);
            if (s < lastSelection_.segments.size()) {
                std::cout << " encoding="
                          << encodings::encodingTypeToString(lastSelection_.segments[s].encoding);
            }
            std::cout << "\n";
        }
    }
};

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoder<T>> makeAutoSubIntSplitEncoder(typename SubIntSplitAutoEncoder<T>::Config cfg) {
    return std::make_shared<SubIntSplitAutoEncoder<T>>(std::move(cfg));
}

inline std::vector<encodings::EncodingType> defaultAutoSubIntSplitCostModelTypes() {
    return {
        encodings::EncodingType::RawEncoding,
        encodings::EncodingType::BitPacking,
        encodings::EncodingType::RunLengthEncoding,
        encodings::EncodingType::AdaptiveFrameOfReference,
        encodings::EncodingType::DictionaryEncoding,
        encodings::EncodingType::FrequencyPartitionEncoding,
        encodings::EncodingType::BlockFrequencyPartitionEncoding,
        encodings::EncodingType::BlockFrequencyPartitionFOREncoding,
        encodings::EncodingType::BlockFSEEncoding,
        encodings::EncodingType::MainlyConstantEncoding
    };
}

inline std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>>
makeAutoSubIntSplitCostModelsFromTypes(const std::vector<encodings::EncodingType>& modelTypes) {
    std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>> models;
    models.reserve(modelTypes.size());

    for (const auto type : modelTypes) {
        switch (type) {
            case encodings::EncodingType::RawEncoding:
                models.emplace_back(std::make_unique<RawCostModel>());
                break;
            case encodings::EncodingType::BitPacking:
                models.emplace_back(std::make_unique<RawBitPackedCostModel>());
                break;
            case encodings::EncodingType::RunLengthEncoding:
                models.emplace_back(std::make_unique<RLECostModel>());
                break;
            case encodings::EncodingType::FrameOfReference:
                models.emplace_back(std::make_unique<FORCostModel>());
                break;
            case encodings::EncodingType::AdaptiveFrameOfReference:
                models.emplace_back(std::make_unique<AdaptiveFORCostModel>());
                break;
            case encodings::EncodingType::AdaptiveFramedBitPrefix:
                models.emplace_back(std::make_unique<AdaptiveFramedBitPrefixCostModel>());
                break;
            case encodings::EncodingType::AdaptiveDictionaryEncoding:
                models.emplace_back(std::make_unique<AdaptiveDictionaryCostModel>());
                break;
            case encodings::EncodingType::DictionaryEncoding:
                models.emplace_back(std::make_unique<DictionaryCostModel>());
                break;
            case encodings::EncodingType::HuffmanEncoding:
                models.emplace_back(std::make_unique<HuffmanCostModel>());
                break;
            case encodings::EncodingType::LZ4:
                models.emplace_back(std::make_unique<LZ4CostModel>());
                break;
            case encodings::EncodingType::FSEEncoding:
                models.emplace_back(std::make_unique<FSECostModel>());
                break;
            case encodings::EncodingType::FrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<FrequencyPartitionCostModel>());
                break;
            case encodings::EncodingType::BlockFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<BlockFrequencyPartitionCostModel>());
                break;
            case encodings::EncodingType::BlockFrequencyPartitionFOREncoding:
                models.emplace_back(std::make_unique<BlockFrequencyPartitionFORCostModel>());
                break;
            case encodings::EncodingType::BlockFSEEncoding:
                models.emplace_back(std::make_unique<BlockFSECostModel>());
                break;
            case encodings::EncodingType::BlockFORFPEEncoding:
                models.emplace_back(std::make_unique<BlockFORFPECostModel>());
                break;
            case encodings::EncodingType::MainlyConstantEncoding:
                models.emplace_back(std::make_unique<MainlyConstantCostModel>());
                break;
            case encodings::EncodingType::RangePackFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<RangePackCostModel>(
                    std::make_unique<FrequencyPartitionCostModel>(),
                    encodings::EncodingType::RangePackFrequencyPartitionEncoding));
                break;
            case encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<RangePackCostModel>(
                    std::make_unique<BlockFrequencyPartitionCostModel>(),
                    encodings::EncodingType::RangePackBlockFrequencyPartitionEncoding));
                break;
            case encodings::EncodingType::CascadingFrameOfReference:
                models.emplace_back(std::make_unique<CascadingFORCostModel>());
                break;
            case encodings::EncodingType::CascadingFORBlockFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<CascadingFORBlockFPECostModel>());
                break;
            case encodings::EncodingType::RunLengthCascadingFOREncoding:
                models.emplace_back(std::make_unique<RunLengthCascadingFORStartsCostModel>());
                break;
            case encodings::EncodingType::CascadingFORFSEEncoding:
                models.emplace_back(std::make_unique<CascadingFORFSECostModel>());
                break;
            case encodings::EncodingType::CascadingFORBlockFSEEncoding:
                models.emplace_back(std::make_unique<CascadingFORBlockFSECostModel>());
                break;
            case encodings::EncodingType::CascadingFORHuffmanEncoding:
                models.emplace_back(std::make_unique<CascadingFORHuffmanCostModel>());
                break;
            case encodings::EncodingType::CascadingFORPrevFSEEncoding:
                models.emplace_back(std::make_unique<CascadingFORPrevFSECostModel>());
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFSEEncoding:
                models.emplace_back(std::make_unique<CascadingFORPrevBlockFSECostModel>());
                break;
            case encodings::EncodingType::CascadingFORPrevHuffmanEncoding:
                models.emplace_back(std::make_unique<CascadingFORPrevHuffmanCostModel>());
                break;
            case encodings::EncodingType::CascadingFORPrevFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<CascadingFORPrevFrequencyPartitionCostModel>());
                break;
            case encodings::EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding:
                models.emplace_back(std::make_unique<CascadingFORPrevBlockFrequencyPartitionCostModel>());
                break;
            default:
                throw std::invalid_argument("makeAutoSubIntSplitCostModelsFromTypes: unsupported encoding type for cost model");
        }
    }

    if (models.empty()) {
        throw std::invalid_argument("makeAutoSubIntSplitCostModelsFromTypes: no cost models provided");
    }

    return models;
}

template <typename T>
inline typename SubIntSplitAutoEncoder<T>::Config makeDefaultAutoSubIntSplitConfig(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                         bool enableSelectionTiming = false,
                                                                         std::vector<encodings::EncodingType> costModelTypes = {},
                                                                         int numSplits = -1,
                                                                         bool allowReorderers = true) {
    typename SubIntSplitAutoEncoder<T>::Config cfg;
    cfg.selectorConfig = selectors::IDSubStreamEncodingSelector::Config{}; // defaults
    cfg.selectorConfig.verboseLevel = 1; // leave quiet by default; enable when debugging
    cfg.selectorConfig.minSegmentWidth = 1; // avoid tiny slices that inflate headers/rounding
    cfg.selectorConfig.splitPenalty = 100.0; // discourage excessive splitting (heavier for MSB)
    cfg.selectorConfig.enableMergePhase = false; // merge adjacent in MSB mode to reduce fragmentation
    cfg.samplerConfig.maxSamples = 100'000;   // cap total sampled points
    cfg.samplerConfig.stride = 0;           // unused when blockSize > 0
    cfg.samplerConfig.blockSize = 32;       // 625 blocks × 16 elements — diverse enough to capture slowly-varying fields (e.g. Snowflake timestamps that change every 4096 IDs)
    cfg.samplerConfig.maxPercentage = 0; // or maxSamples, whichever is smaller
    cfg.debugLogging = false;                 // enable instrumentation by default for now
    cfg.enableSelectionTiming = enableSelectionTiming;
    cfg.orderHint = order;
    if (numSplits > 0)
        cfg.selectorConfig.forcedNumSegments = numSplits;

    if (costModelTypes.empty()) {
        costModelTypes = defaultAutoSubIntSplitCostModelTypes();
    }
    cfg.costModels = makeAutoSubIntSplitCostModelsFromTypes(costModelTypes);
    if (allowReorderers)
        cfg.reordererModels.push_back(std::make_unique<selectors::costs::BWTReordererCostModel>());
    return cfg;
}

// CostModelSet overload: accepts a fully configured CostModelSet instead of
// a bare list of encoding types.  Allows callers to compose cost dimensions:
//
//   auto cfg = makeDefaultAutoSubIntSplitConfig<int64_t>(
//       BitSplitOrder::LSB_TO_MSB,
//       CostModelSet{}.add(CostModelDimension::Compression)
//                     .add(CostModelDimension::DecodeAtSpeed, 2.0));
template <typename T>
inline typename SubIntSplitAutoEncoder<T>::Config makeDefaultAutoSubIntSplitConfig(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false)
{
    typename SubIntSplitAutoEncoder<T>::Config cfg;
    cfg.selectorConfig = selectors::IDSubStreamEncodingSelector::Config{};
    cfg.selectorConfig.verboseLevel = 1;
    cfg.selectorConfig.minSegmentWidth = 1;
    cfg.selectorConfig.enableMergePhase = false;
    cfg.samplerConfig.maxSamples = 10'000;
    cfg.samplerConfig.stride = 0;
    cfg.samplerConfig.blockSize = 16;
    cfg.samplerConfig.maxPercentage = 0;
    cfg.debugLogging = false;
    cfg.enableSelectionTiming = enableSelectionTiming;
    cfg.orderHint = order;
    if (numSplits > 0)
        cfg.selectorConfig.forcedNumSegments = numSplits;
    cfg.selectorConfig.splitPenalty =
        costModelSet.recommendedSplitPenalty(cfg.samplerConfig.maxSamples);
    cfg.costModels = costModelSet.build();
    if (allowReorderers)
        cfg.reordererModels.push_back(std::make_unique<selectors::costs::BWTReordererCostModel>());
    return cfg;
}

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoder<T>> makeDefaultAutoSubIntSplitEncoder(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                                   bool exhaustiveSearch = false,
                                                                                   bool enablePrune = true,
                                                                                   bool enableSelectionTiming = false,
                                                                                   std::vector<encodings::EncodingType> costModelTypes = {},
                                                                                   int numSplits = -1,
                                                                                   bool allowReorderers = true) {
    auto cfg = makeDefaultAutoSubIntSplitConfig<T>(order, enableSelectionTiming, std::move(costModelTypes), numSplits, allowReorderers);
    cfg.selectorConfig.orderHint = order;
    cfg.selectorConfig.useExhaustiveSearch = exhaustiveSearch;
    cfg.selectorConfig.enablePrune = enablePrune;
    cfg.selectorConfig.costGridCsvPath = "../Source/encoders/auto_subintsplit_cost_grid_twitter_snowflake_64.csv"; // for debugging/analysis; selector will log evaluated candidates and their costs
    return makeAutoSubIntSplitEncoder<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoder<T>> makeDefaultAutoSubIntSplitEncoder(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false) {
    auto cfg = makeDefaultAutoSubIntSplitConfig<T>(order, std::move(costModelSet), numSplits, allowReorderers, enableSelectionTiming);
    cfg.selectorConfig.orderHint          = order;
    cfg.selectorConfig.useExhaustiveSearch = exhaustiveSearch;
    cfg.selectorConfig.enablePrune         = enablePrune;
    cfg.selectorConfig.costGridCsvPath     = "../Source/encoders/auto_subintsplit_cost_grid_twitter_snowflake_64.csv";
    return makeAutoSubIntSplitEncoder<T>(std::move(cfg));
}

using SubIntSplitAutoEncoder64 = SubIntSplitAutoEncoder<int64_t>;
using SubIntSplitAutoEncoder32 = SubIntSplitAutoEncoder<int32_t>;

// Profiling variants of the auto-selecting encoder.
template<typename T> using SubIntSplitAutoEncoderProf = SubIntSplitAutoEncoder<T, true>;
using SubIntSplitAutoEncoder64Prof = SubIntSplitAutoEncoder<int64_t, true>;
using SubIntSplitAutoEncoder32Prof = SubIntSplitAutoEncoder<int32_t, true>;

// ---------------------------------------------------------------------------
// Profiling factory functions for SubIntSplitAutoEncoder<T, true>
// ---------------------------------------------------------------------------

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoderProf<T>> makeAutoSubIntSplitEncoderProf(
    typename SubIntSplitAutoEncoderProf<T>::Config cfg) {
    return std::make_shared<SubIntSplitAutoEncoderProf<T>>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoderProf<T>> makeDefaultAutoSubIntSplitEncoderProf(
    BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    bool enableSelectionTiming = false,
    std::vector<encodings::EncodingType> costModelTypes = {},
    int numSplits = -1,
    SubIntSplitParallelismConfig parallelism = {}) {
    // Config type is identical to the non-profiling variant (EnableProfiling only affects runtime)
    auto cfg = makeDefaultAutoSubIntSplitConfig<T>(order, enableSelectionTiming, std::move(costModelTypes), numSplits);
    cfg.selectorConfig.orderHint          = order;
    cfg.selectorConfig.useExhaustiveSearch = exhaustiveSearch;
    cfg.selectorConfig.enablePrune         = enablePrune;
    cfg.selectorConfig.costGridCsvPath     = "../Source/encoders/auto_subintsplit_cost_grid_twitter_snowflake_64.csv";
    cfg.parallelism                        = parallelism;
    return makeAutoSubIntSplitEncoderProf<T>(std::move(cfg));
}

template <typename T>
inline std::shared_ptr<SubIntSplitAutoEncoderProf<T>> makeDefaultAutoSubIntSplitEncoderProf(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false,
    SubIntSplitParallelismConfig parallelism = {}) {
    auto cfg = makeDefaultAutoSubIntSplitConfig<T>(order, std::move(costModelSet), numSplits, allowReorderers, enableSelectionTiming);
    cfg.selectorConfig.orderHint          = order;
    cfg.selectorConfig.useExhaustiveSearch = exhaustiveSearch;
    cfg.selectorConfig.enablePrune         = enablePrune;
    cfg.selectorConfig.costGridCsvPath     = "../Source/encoders/auto_subintsplit_cost_grid_twitter_snowflake_64.csv";
    cfg.parallelism                        = parallelism;
    return makeAutoSubIntSplitEncoderProf<T>(std::move(cfg));
}

// int64_t convenience overloads
inline std::shared_ptr<SubIntSplitAutoEncoder64Prof> makeAutoSubIntSplitEncoderProf(
    SubIntSplitAutoEncoder64Prof::Config cfg) {
    return makeAutoSubIntSplitEncoderProf<int64_t>(std::move(cfg));
}

inline std::shared_ptr<SubIntSplitAutoEncoder64Prof> makeDefaultAutoSubIntSplitEncoderProf(
    BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    bool enableSelectionTiming = false,
    std::vector<encodings::EncodingType> costModelTypes = {},
    int numSplits = -1,
    SubIntSplitParallelismConfig parallelism = {}) {
    return makeDefaultAutoSubIntSplitEncoderProf<int64_t>(
        order, exhaustiveSearch, enablePrune, enableSelectionTiming, std::move(costModelTypes), numSplits, parallelism);
}

inline std::shared_ptr<SubIntSplitAutoEncoder64Prof> makeDefaultAutoSubIntSplitEncoderProf(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false,
    SubIntSplitParallelismConfig parallelism = {}) {
    return makeDefaultAutoSubIntSplitEncoderProf<int64_t>(
        order, std::move(costModelSet), exhaustiveSearch, enablePrune, numSplits, allowReorderers, enableSelectionTiming, parallelism);
}

// ---------------------------------------------------------------------------
// Non-profiling int64_t convenience overloads (unchanged)
// ---------------------------------------------------------------------------

inline std::shared_ptr<SubIntSplitAutoEncoder64> makeAutoSubIntSplitEncoder(SubIntSplitAutoEncoder64::Config cfg) {
    return makeAutoSubIntSplitEncoder<int64_t>(std::move(cfg));
}

inline SubIntSplitAutoEncoder64::Config makeDefaultAutoSubIntSplitConfig(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                         bool enableSelectionTiming = false,
                                                                         std::vector<encodings::EncodingType> costModelTypes = {}) {
    return makeDefaultAutoSubIntSplitConfig<int64_t>(order, enableSelectionTiming, std::move(costModelTypes));
}

inline SubIntSplitAutoEncoder64::Config makeDefaultAutoSubIntSplitConfig(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false) {
    return makeDefaultAutoSubIntSplitConfig<int64_t>(order, std::move(costModelSet), numSplits, allowReorderers, enableSelectionTiming);
}

inline std::shared_ptr<SubIntSplitAutoEncoder64> makeDefaultAutoSubIntSplitEncoder(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                                   bool exhaustiveSearch = false,
                                                                                   bool enablePrune = true,
                                                                                   bool enableSelectionTiming = false,
                                                                                   std::vector<encodings::EncodingType> costModelTypes = {}) {
    return makeDefaultAutoSubIntSplitEncoder<int64_t>(order, exhaustiveSearch, enablePrune, enableSelectionTiming, std::move(costModelTypes));
}

inline std::shared_ptr<SubIntSplitAutoEncoder64> makeDefaultAutoSubIntSplitEncoder(
    BitSplitOrder order,
    selectors::costs::CostModelSet costModelSet,
    bool exhaustiveSearch = false,
    bool enablePrune = true,
    int numSplits = -1,
    bool allowReorderers = true,
    bool enableSelectionTiming = false) {
    return makeDefaultAutoSubIntSplitEncoder<int64_t>(
        order, std::move(costModelSet), exhaustiveSearch, enablePrune, numSplits, allowReorderers, enableSelectionTiming);
}

} // namespace encodings::encoders
