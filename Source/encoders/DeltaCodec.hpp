#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <span>
#include <vector>
#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"
#include "encoders/SubIntEncoder.hpp"

namespace encodings::encoders {

// =============================================================================
//  DeltaSubIntEncoder<T, BlockSize>
//
//  A self-contained Codec<T, uint8_t> that fuses:
//
//    1. Delta transform   (T[] → T[] deltas)
//    2. SubIntEncoder<T>  (T[] deltas → uint8_t[] compressed)
//
//  into a single encoder with a sparse skip table that enables O(BlockSize)
//  random access — independent of N.
//
//  Why not ComposedEncoder<DeltaCodec<T,T>, SubIntEncoder<T>>?
//  -----------------------------------------------------------
//  ComposedEncoder passes the entire intermediate EncodedBuffer<T> to the
//  second stage and has no mechanism to carry the skip table through the
//  byte-encoding boundary.  For decodeAt(i) to be fast, we need:
//    (a) the skip table checkpoint at block c = i/B  (absolute value)
//    (b) the SubInt-encoded deltas for positions [c*B .. i]
//  Both (a) and (b) must live in the same EncodedData (uint8_t buffer).
//  Only a fused encoder can guarantee this layout.
//
//  Wire format of EncodedData (uint8_t buffer):
//  -------------------------------------------------------
//  [  0 ..  7 ]   N         (uint64_t, element count)
//  [  8 .. 15 ]   B         (uint64_t, BlockSize used at encode time)
//  [ 16 .. 23 ]   C         (uint64_t, number of skip-table entries = ceil(N/B))
//  [ 24 .. 24+C*sizeof(T)-1 ]  skip table: skipTable[c] = original value at c*B
//  [ 24+C*sizeof(T) .. end ]   SubInt-encoded delta stream (SubIntEncoder output)
//
//  decodeAt(i):
//    1. c    = i / B
//    2. base = skipTable[c]                              ← O(1) from skip table
//    3. lo   = c * B  (inclusive),  hi = i  (inclusive)
//    4. SubInt::decodeRange(lo+1, hi+1) → deltas         ← O(B) SubInt calls
//    5. result = base + sum(deltas)                      ← O(B) additions
//    Total: O(B), independent of N.
//
//  decodeAll: SubInt::decodeAll → prefix-sum. O(N).
//
// =============================================================================

/**
 * @brief Fused Delta + SubInt encoder with sparse skip table for O(BlockSize)
 * random access.
 *
 * @tparam T          int32_t or int64_t
 * @tparam BlockSize  number of elements per skip-table block (default 512)
 */
template <typename T, size_t BlockSize = 512>
    requires (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
class DeltaSubIntEncoder : public Codec<T, uint8_t> {
    static_assert(BlockSize >= 1, "BlockSize must be at least 1");

    using SubInt = SubIntEncoder<T>;

public:
    /**
     * @param subIntCfg   Configuration forwarded to the internal SubIntEncoder.
     *                    Tune the SplitMode / code widths for the delta domain,
     *                    not the original-value domain.
     */
    explicit DeltaSubIntEncoder(SubIntConfig<T> subIntCfg = {})
        : subInt_(subIntCfg)
    {}

    // =========================================================================
    //  Encode
    // =========================================================================

    EncodedData encode(std::span<const T> data) override {
        const size_t N = data.size();
        const size_t C = (N == 0) ? 0 : (N + BlockSize - 1) / BlockSize;

        // ---- 1. Build delta stream + skip table simultaneously ----
        std::vector<T> deltas;
        deltas.reserve(N);
        std::vector<T> skipTable;
        skipTable.reserve(C);

        if (N > 0) {
            deltas.push_back(data[0]);          // first element verbatim
            skipTable.push_back(data[0]);       // checkpoint for block 0

            for (size_t i = 1; i < N; ++i) {
                deltas.push_back(data[i] - data[i - 1]);
                if (i % BlockSize == 0)
                    skipTable.push_back(data[i]);   // checkpoint for block i/BlockSize
            }
        }

        // ---- 2. SubInt-encode the delta stream ----
        EncodedData subIntEncoded = subInt_.encode(deltas);

        // ---- 3. Assemble final buffer ----
        //  Header: [N:8][B:8][C:8]  then [skip table: C*sizeof(T)]  then SubInt payload
        const size_t skipBytes   = C * sizeof(T);
        const size_t headerBytes = 3 * sizeof(uint64_t);
        const size_t subIntSize  = subIntEncoded.size();    // bytes

        std::vector<uint8_t> out;
        out.resize(headerBytes + skipBytes + subIntSize);

        uint8_t* p = out.data();

        auto writeU64 = [&](uint64_t v) {
            std::memcpy(p, &v, sizeof(uint64_t)); p += sizeof(uint64_t);
        };
        writeU64(static_cast<uint64_t>(N));
        writeU64(static_cast<uint64_t>(BlockSize));
        writeU64(static_cast<uint64_t>(C));

        if (!skipTable.empty())
            std::memcpy(p, skipTable.data(), skipBytes);
        p += skipBytes;

        std::memcpy(p, subIntEncoded.data().data(), subIntSize);

        // ---- 4. Metadata ----
        EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = core::typeToDataType<T>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(T);
        meta.supportsRandomAccess = true;
        meta.customMetadata["block_size"]     = std::to_string(BlockSize);
        meta.customMetadata["subint_config"]  = subInt_.name();
        meta.customMetadata["skip_table_bytes"] = std::to_string(skipBytes);

        return EncodedData(std::move(out), std::move(meta));
    }

    // =========================================================================
    //  decodeAll  — O(N)
    // =========================================================================

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        ParsedHeader h = parseHeader(encoded);
        if (h.N == 0) return {};

        // Decode the SubInt payload → deltas
        EncodedData subIntView = makeSubIntView(encoded, h);
        std::vector<T> deltas = subInt_.decodeAll(subIntView);

        // Prefix-sum to reconstruct originals
        std::vector<T> out;
        out.reserve(h.N);
        T cur = deltas[0];
        out.push_back(cur);
        for (size_t i = 1; i < h.N; ++i) {
            cur += deltas[i];
            out.push_back(cur);
        }
        return out;
    }

    // =========================================================================
    //  decodeAt(i)  — O(BlockSize)
    // =========================================================================

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        ParsedHeader h = parseHeader(encoded);
        if (index >= h.N) return std::nullopt;

        // 1. Read checkpoint for the block containing `index`.
        const size_t block = index / BlockSize;
        T base = readCheckpoint(encoded, h, block);

        // 2. Decode deltas from block start+1 to index (inclusive) via SubInt.
        const size_t blockStart = block * BlockSize;
        if (index == blockStart) return base;   // exactly on a checkpoint

        // SubInt decodeRange [blockStart+1, index+1) gives the deltas we need.
        EncodedData subIntView = makeSubIntView(encoded, h);
        std::vector<T> deltas = subInt_.decodeRange(subIntView,
                                                     blockStart + 1,
                                                     index + 1);
        T cur = base;
        for (T d : deltas) cur += d;
        return cur;
    }

    // =========================================================================
    //  decodeRange(start, end)  — O(start%BlockSize + (end-start))
    // =========================================================================

    std::vector<T> decodeRange(const EncodedData& encoded,
                                size_t start, size_t end) override {
        ParsedHeader h = parseHeader(encoded);
        if (start >= h.N) return {};
        end = std::min(end, h.N);
        if (start >= end) return {};

        const size_t block      = start / BlockSize;
        const size_t blockStart = block * BlockSize;
        T base = readCheckpoint(encoded, h, block);

        EncodedData subIntView = makeSubIntView(encoded, h);

        // Accumulate from blockStart+1 to start-1 (to land on the value just
        // before start, or on start itself if start==blockStart).
        std::vector<T> out;
        out.reserve(end - start);

        if (start == blockStart) {
            // base is the value at `start`.
            out.push_back(base);
            if (end > start + 1) {
                std::vector<T> tail = subInt_.decodeRange(subIntView,
                                                           start + 1, end);
                T cur = base;
                for (T d : tail) { cur += d; out.push_back(cur); }
            }
        } else {
            // Need deltas from [blockStart+1, end) and prefix-sum from blockStart.
            std::vector<T> allDeltas = subInt_.decodeRange(subIntView,
                                                            blockStart + 1, end);
            // First (start - blockStart) deltas advance us from base to start.
            T cur = base;
            const size_t skipCount = start - blockStart;
            for (size_t i = 0; i < skipCount; ++i) cur += allDeltas[i];
            out.push_back(cur);
            for (size_t i = skipCount; i < allDeltas.size(); ++i) {
                cur += allDeltas[i];
                out.push_back(cur);
            }
        }
        return out;
    }

    // =========================================================================
    //  Metadata
    // =========================================================================

    EncodingType encodingType() const override { return EncodingType::DeltaEncoding; }

    std::string name() const override {
        return "DeltaSubInt<" + std::to_string(BlockSize) + ">|" + subInt_.name();
    }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::PreservesOrder
             | EncodingProperty::DeltaBased
             | EncodingProperty::OptimizedForSorted
             | EncodingProperty::Composable
             | EncodingProperty::RandomAccess;   // O(BlockSize) via skip table
    }

    size_t estimateEncodedSize(size_t n) const override {
        const size_t C = (n == 0) ? 0 : (n + BlockSize - 1) / BlockSize;
        return 3 * sizeof(uint64_t)
             + C * sizeof(T)
             + subInt_.estimateEncodedSize(n);
    }

private:
    SubInt subInt_;

    // -------------------------------------------------------------------------
    //  Parsed header — just offsets, no allocation
    // -------------------------------------------------------------------------
    struct ParsedHeader {
        uint64_t N;           ///< element count
        uint64_t B;           ///< BlockSize stored at encode time
        uint64_t C;           ///< number of skip-table entries
        size_t   skipOffset;  ///< byte offset in buffer where skip table starts
        size_t   subIntOffset;///< byte offset in buffer where SubInt payload starts
    };

    static ParsedHeader parseHeader(const EncodedData& enc) {
        const uint8_t* p = enc.data().data();
        uint64_t N, B, C;
        std::memcpy(&N, p,                    sizeof(uint64_t));
        std::memcpy(&B, p + sizeof(uint64_t), sizeof(uint64_t));
        std::memcpy(&C, p + 2*sizeof(uint64_t), sizeof(uint64_t));
        const size_t headerBytes = 3 * sizeof(uint64_t);
        const size_t skipBytes   = static_cast<size_t>(C) * sizeof(T);
        return ParsedHeader{N, B, C,
                            /*skipOffset=*/  headerBytes,
                            /*subIntOffset=*/headerBytes + skipBytes};
    }

    /// Read skipTable[block] — the absolute value at position block * B.
    static T readCheckpoint(const EncodedData& enc,
                             const ParsedHeader& h, size_t block) {
        T val;
        const uint8_t* src = enc.data().data()
                           + h.skipOffset
                           + block * sizeof(T);
        std::memcpy(&val, src, sizeof(T));
        return val;
    }

    /// Build a lightweight EncodedData view over just the SubInt payload.
    /// Does a copy of the relevant bytes — SubIntEncoder needs a contiguous
    /// EncodedData; SubIntEncoder's cache uses (ptr, size) identity so the
    /// copy is safe and the cache will still hit across repeated calls with
    /// the same source.
    EncodedData makeSubIntView(const EncodedData& enc,
                                const ParsedHeader& h) const {
        const uint8_t* src = enc.data().data() + h.subIntOffset;
        const size_t   sz  = enc.data().size() - h.subIntOffset;
        std::vector<uint8_t> payload(src, src + sz);

        EncodingMetadata meta;
        meta.encodingName         = subInt_.name();
        meta.dataType             = core::typeToDataType<T>;
        meta.elementCount         = static_cast<size_t>(h.N);
        meta.compressedSize       = sz;
        meta.uncompressedSize     = static_cast<size_t>(h.N) * sizeof(T);
        meta.supportsRandomAccess = true;
        return EncodedData(std::move(payload), std::move(meta));
    }
};

// -----------------------------------------------------------------------
//  Factory helper
// -----------------------------------------------------------------------

/**
 * @brief Create a DeltaSubIntEncoder<T> with the given SubInt config.
 *
 * Example:
 *   auto enc = makeDeltaSubInt<int64_t>(Split44());
 *   // or with a full config:
 *   SubIntConfig<int64_t> cfg; cfg.splitMode = Split26();
 *   auto enc = makeDeltaSubInt<int64_t>(cfg);
 */
template <typename T, size_t BlockSize = 512>
std::shared_ptr<DeltaSubIntEncoder<T, BlockSize>>
makeDeltaSubInt(SubIntConfig<T> cfg = {}) {
    return std::make_shared<DeltaSubIntEncoder<T, BlockSize>>(cfg);
}

} // namespace encodings::encoders
