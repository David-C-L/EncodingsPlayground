#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

namespace encodings::encoders {

// MainlyConstantEncoder<T>
//
// Compresses data where one value dominates the stream, faithful to Nimble's
// MainlyConstantEncoding.  The most frequent value (commonValue) is stored
// inline; a 64-bit-aligned bit-packed bitmap records which positions hold it
// (1=common, 0=uncommon); the remaining values are stored in an otherValues
// sub-stream encoded by an optional inner codec (recursive MC by default).
//
// Wire format:
//   [8]  elementCount (size_t)
//   [4]  bitmapByteCount (uint32_t)  = ((elementCount+63)/64)*8
//   [B]  isCommon bitmap (packed uint64_t words; 1=common, 0=uncommon)
//   [4]  uncommonCount (uint32_t)
//   [4]  otherValuesEncodedSize (uint32_t)
//   [V]  otherValues bytes (raw T[] or inner-encoded)
//   [S]  commonValue (sizeof(T) bytes)
//
// The decoder uses the encoder instance's innerCodec_ to interpret otherValues;
// no codec tag is stored in the stream.
//
// Recursion:
//   maxDepth=0 → otherValues stored raw (flat, used for SubIntSplit segments).
//   maxDepth>0 + innerCodec set → otherValues recursively encoded by innerCodec.
//   makeRecursiveMainlyConstantEncoder<T>(depth) builds the nested chain.

template<typename T>
    requires core::IntegralType<T>
class MainlyConstantEncoder : public Codec<T, uint8_t> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    explicit MainlyConstantEncoder(
            std::shared_ptr<Codec<T, uint8_t>> innerOtherValuesCodec = nullptr,
            uint32_t maxDepth = 0)
        : innerCodec_(std::move(innerOtherValuesCodec))
        , innerMCE_(dynamic_cast<MainlyConstantEncoder<T>*>(innerCodec_.get()))
        , maxDepth_(maxDepth) {}

    // -------------------------------------------------------------------------
    // Encode
    // -------------------------------------------------------------------------
    EncodedData encode(std::span<const T> data) override {
        if (data.empty()) {
            EncodedData result;
            result.metadata().encodingName = name();
            result.metadata().dataType = this->dataType();
            result.metadata().elementCount = 0;
            result.metadata().compressedSize = 0;
            result.metadata().uncompressedSize = 0;
            result.metadata().supportsRandomAccess = true;
            return result;
        }

        const size_t N = data.size();

        // --- Find the most frequent value ---
        ankerl::unordered_dense::map<T, uint32_t> freq;
        freq.reserve(std::min(N, size_t{4096}));
        for (const T& v : data) freq[v]++;

        const T commonValue = std::max_element(freq.begin(), freq.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; })->first;

        // --- Build bitmap and otherValues ---
        const uint32_t numWords = static_cast<uint32_t>((N + 63) / 64);
        const uint32_t bitmapByteCount = numWords * 8u;

        std::vector<uint64_t> bitmap(numWords, ~uint64_t{0});  // all 1 = all common
        std::vector<T> otherValues;
        otherValues.reserve(N - freq[commonValue]);

        for (size_t i = 0; i < N; ++i) {
            if (data[i] != commonValue) {
                bitmap[i / 64] &= ~(uint64_t{1} << (i % 64));
                otherValues.push_back(data[i]);
            }
        }
        // Tail masking: set unused bits in last word to 1 (branchless countNonCommon)
        const uint32_t tailBits = static_cast<uint32_t>(N & 63);
        if (tailBits != 0) {
            bitmap[numWords - 1] |= ~((uint64_t{1} << tailBits) - 1);
        }

        const uint32_t uncommonCount = static_cast<uint32_t>(otherValues.size());

        // --- Encode otherValues ---
        std::vector<uint8_t> otherValuesBytes;
        if (maxDepth_ == 0 || !innerCodec_ || uncommonCount == 0) {
            // Raw: just copy the T values as bytes
            otherValuesBytes.resize(uncommonCount * sizeof(T));
            std::memcpy(otherValuesBytes.data(), otherValues.data(),
                        otherValuesBytes.size());
        } else {
            auto inner = innerCodec_->encode(std::span<const T>(otherValues));
            otherValuesBytes = std::move(inner.data());
        }

        const uint32_t otherValuesEncodedSize =
            static_cast<uint32_t>(otherValuesBytes.size());

        // --- Serialize ---
        const size_t totalSize = sizeof(size_t)      // elementCount
            + sizeof(uint32_t)                        // bitmapByteCount
            + bitmapByteCount                         // bitmap
            + sizeof(uint32_t)                        // uncommonCount
            + sizeof(uint32_t)                        // otherValuesEncodedSize
            + otherValuesEncodedSize                  // otherValues
            + sizeof(T);                              // commonValue

        EncodedData result;
        result.data().resize(totalSize);
        uint8_t* pos = result.data().data();

        write(pos, N);
        write(pos, bitmapByteCount);
        std::memcpy(pos, bitmap.data(), bitmapByteCount); pos += bitmapByteCount;
        write(pos, uncommonCount);
        write(pos, otherValuesEncodedSize);
        if (otherValuesEncodedSize > 0) {
            std::memcpy(pos, otherValuesBytes.data(), otherValuesEncodedSize);
            pos += otherValuesEncodedSize;
        }
        write(pos, commonValue);

        result.metadata().encodingName = name();
        result.metadata().dataType = this->dataType();
        result.metadata().elementCount = N;
        result.metadata().compressedSize = totalSize;
        result.metadata().uncompressedSize = N * sizeof(T);
        result.metadata().supportsRandomAccess = true;
        result.metadata().customMetadata["uncommon_count"] =
            std::to_string(uncommonCount);
        result.metadata().customMetadata["common_value"] =
            std::to_string(static_cast<int64_t>(commonValue));

        return result;
    }

    // -------------------------------------------------------------------------
    // Public decode API — thin wrappers over the *Bytes private implementations.
    // -------------------------------------------------------------------------

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        if (encoded.data().empty()) return std::nullopt;
        return decodeAtBytes(encoded.data().data(), index);
    }

    void decodeAllInto(const EncodedData& encoded, T* dst, size_t n) override {
        if (encoded.data().empty()) return;
        decodeAllBytesInto(encoded.data().data(), dst, n);
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
        if (encoded.data().empty()) return {};
        const ParsedHeader h = parseHeader(encoded.data().data());
        if (h.N == 0) return {};
        std::vector<T> out(h.N, h.commonValue);   // one fill pass, no zero-init
        if (h.uncommonCount > 0)
            scatterAll(h, out.data());
        return out;
    }

    void decodeRangeInto(const EncodedData& encoded,
                         size_t start, size_t end,
                         T* dst, size_t n) override {
        if (encoded.data().empty() || start >= end) return;
        decodeRangeBytesInto(encoded.data().data(), start, end, dst, n);
    }

    std::vector<T> decodeRange(const EncodedData& encoded,
                               size_t start, size_t end) override {
        if (encoded.data().empty() || start >= end) return {};
        const ParsedHeader h = parseHeader(encoded.data().data());
        if (start >= h.N) return {};
        end = std::min(end, h.N);
        const size_t rangeLen = end - start;
        std::vector<T> out(rangeLen, h.commonValue);  // one fill pass, no zero-init
        if (h.uncommonCount > 0)
            scatterRange(h, start, end, out.data());
        return out;
    }

    // Gather (selective row-range) fast path. decodeAt/decodeRange compute the
    // uncommon-value rank via prefixNonCommonCount, a full bitmap popcount
    // scan from bit 0 on every call. Here we carry the rank forward across
    // the whole RowRangeList: the first range seeds it via
    // prefixNonCommonCount (O(begin/64)), every subsequent range advances it
    // by only the GAP since the previous range's end via rangeNonCommonCount
    // (O(gap/64)) instead of restarting from bit 0 -- turns O(K * avg(begin)/64)
    // into O(N/64) total. Works for both the raw/leaf case and the recursive
    // (innerCodec_) case, since scatterRange's existing bulk-decode branch for
    // the inner codec already only needs the rank at the range's start, not a
    // byte-stream cursor.
    void decodeGatherInto(const EncodedData& encoded,
                          const RowRangeList& ranges,
                          T* dst, size_t n) override {
        if (ranges.empty()) {
            if (n != 0) throw std::runtime_error("MainlyConstantEncoder::decodeGatherInto: decoded size mismatch");
            return;
        }
        if (encoded.data().empty()) {
            if (n != 0) throw std::runtime_error("MainlyConstantEncoder::decodeGatherInto: empty buffer, n!=0");
            return;
        }
        const ParsedHeader h = parseHeader(encoded.data().data());

        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t end = std::min(r.end, h.N);
            if (r.begin >= end) continue;
            std::fill(dst + off, dst + off + (end - r.begin), h.commonValue);
            off += (end - r.begin);
        }
        if (off != n) throw std::runtime_error("MainlyConstantEncoder::decodeGatherInto: decoded size mismatch");
        if (h.uncommonCount == 0) return;

        uint32_t rank = 0;
        size_t prevEnd = 0;
        bool haveRank = false;
        size_t dstOff = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t end = std::min(r.end, h.N);
            if (r.begin >= end) continue;

            const uint32_t rankAtBegin = haveRank
                ? rank + rangeNonCommonCount(h.bitmapStart, prevEnd, r.begin)
                : prefixNonCommonCount(h.bitmapStart, r.begin);

            const uint32_t scattered = scatterRange(h, r.begin, end, dst + dstOff, rankAtBegin);
            rank = rankAtBegin + scattered;
            haveRank = true;

            dstOff += (end - r.begin);
            prevEnd = end;
        }
    }

    EncodingType encodingType() const override {
        return EncodingType::MainlyConstantEncoding;
    }

    std::string name() const override {
        return maxDepth_ == 0 ? "MainlyConstantFlat" : "MainlyConstant";
    }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::RandomAccess)
            | EncodingProperty::Lossless
            | EncodingProperty::PreservesOrder
            | EncodingProperty::Composable
            | EncodingProperty::RequiresFullData
            | EncodingProperty::FastSkip;
    }

private:
    std::shared_ptr<Codec<T, uint8_t>> innerCodec_;
    // Non-owning downcast — non-null when innerCodec_ is a MainlyConstantEncoder<T>.
    // Cached at construction to avoid dynamic_cast on every decode call.
    // Used to bypass EncodedData wrapping (and its memmove) for the recursive case.
    MainlyConstantEncoder<T>* innerMCE_ = nullptr;
    uint32_t maxDepth_;

    // -------------------------------------------------------------------------
    // Header parsed into raw pointers into the encoded buffer — no copies.
    // -------------------------------------------------------------------------
    struct ParsedHeader {
        size_t          N;
        uint32_t        numWords;
        const uint8_t*  bitmapStart;
        uint32_t        uncommonCount;
        uint32_t        otherValuesEncodedSize;
        const uint8_t*  otherValuesStart;
        T               commonValue;
    };

    // Unaligned 8-byte load — compiles to a single movq on x86.
    // The bitmap starts at offset 12 and otherValues at offset 20+bitmapByteCount,
    // neither of which is guaranteed to be 8-byte aligned, so we must use memcpy.
    static uint64_t loadWord(const uint8_t* p) noexcept {
        uint64_t w;
        std::memcpy(&w, p, 8);
        return w;
    }

    static ParsedHeader parseHeader(const uint8_t* base) noexcept {
        ParsedHeader h;
        std::memcpy(&h.N,                      base, sizeof(size_t));
        base += sizeof(size_t);
        uint32_t bbc;
        std::memcpy(&bbc,                      base, sizeof(uint32_t));
        base += sizeof(uint32_t);
        h.numWords             = bbc / 8u;
        h.bitmapStart          = base;
        base                  += bbc;
        std::memcpy(&h.uncommonCount,          base, sizeof(uint32_t));
        base += sizeof(uint32_t);
        std::memcpy(&h.otherValuesEncodedSize, base, sizeof(uint32_t));
        base += sizeof(uint32_t);
        h.otherValuesStart     = base;
        base                  += h.otherValuesEncodedSize;
        std::memcpy(&h.commonValue,            base, sizeof(T));
        return h;
    }

    // -------------------------------------------------------------------------
    // Bitmap popcount helpers
    // -------------------------------------------------------------------------

    // Count non-common positions in [0, limit).
    // 4-way independent accumulator breaks the serial RAW dependency on `count`.
    static uint32_t prefixNonCommonCount(const uint8_t* bitmapStart,
                                         size_t limit) noexcept {
        if (limit == 0) return 0;
        const uint32_t fullWords = static_cast<uint32_t>(limit / 64);
        uint32_t w = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (; w + 4 <= fullWords; w += 4) {
            c0 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+0) * 8)));
            c1 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+1) * 8)));
            c2 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+2) * 8)));
            c3 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+3) * 8)));
        }
        uint32_t count = c0 + c1 + c2 + c3;
        for (; w < fullWords; ++w)
            count += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + w * 8)));
        const uint32_t rem = static_cast<uint32_t>(limit % 64);
        if (rem > 0) {
            const uint64_t mask = (uint64_t{1} << rem) - 1;
            count += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + fullWords * 8) & mask));
        }
        return count;
    }

    // Count non-common positions in [limitA, limitB).
    // Starts at word limitA/64, skipping the prefix already counted by prefixNonCommonCount.
    static uint32_t rangeNonCommonCount(const uint8_t* bitmapStart,
                                        size_t limitA, size_t limitB) noexcept {
        if (limitA >= limitB) return 0;
        const uint32_t wA = limitA / 64, remA = limitA % 64;
        const uint32_t wB = limitB / 64, remB = limitB % 64;
        uint32_t count = 0;
        if (wA == wB) {
            uint64_t word = ~loadWord(bitmapStart + wA * 8);
            if (remA) word &= ~((uint64_t{1} << remA) - 1);
            if (remB) word &=  (uint64_t{1} << remB) - 1;
            return static_cast<uint32_t>(__builtin_popcountll(word));
        }
        {
            uint64_t word = ~loadWord(bitmapStart + wA * 8);
            if (remA) word &= ~((uint64_t{1} << remA) - 1);
            count += static_cast<uint32_t>(__builtin_popcountll(word));
        }
        uint32_t w = wA + 1;
        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (; w + 4 <= wB; w += 4) {
            c0 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+0) * 8)));
            c1 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+1) * 8)));
            c2 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+2) * 8)));
            c3 += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + (w+3) * 8)));
        }
        count += c0 + c1 + c2 + c3;
        for (; w < wB; ++w)
            count += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + w * 8)));
        if (remB) {
            const uint64_t mask = (uint64_t{1} << remB) - 1;
            count += static_cast<uint32_t>(
                __builtin_popcountll(~loadWord(bitmapStart + wB * 8) & mask));
        }
        return count;
    }

    // -------------------------------------------------------------------------
    // Private *Bytes implementations — work on raw encoded bytes, recurse via
    // innerMCE_ to avoid EncodedData construction and memmove at each level.
    // -------------------------------------------------------------------------

    std::optional<T> decodeAtBytes(const uint8_t* data, size_t index) const {
        const ParsedHeader h = parseHeader(data);
        if (index >= h.N) return std::nullopt;
        const uint64_t bword = loadWord(h.bitmapStart + (index / 64) * 8);
        if ((bword >> (index % 64)) & uint64_t{1}) return h.commonValue;
        const uint32_t nb = prefixNonCommonCount(h.bitmapStart, index);
        if (maxDepth_ == 0 || !innerCodec_) {
            T val;
            std::memcpy(&val, h.otherValuesStart + nb * sizeof(T), sizeof(T));
            return val;
        }
        if (innerMCE_)
            return innerMCE_->decodeAtBytes(h.otherValuesStart, nb);
        EncodedData inner;
        inner.data().assign(h.otherValuesStart,
                            h.otherValuesStart + h.otherValuesEncodedSize);
        return innerCodec_->decodeAt(inner, nb);
    }

    void decodeAllBytesInto(const uint8_t* data, T* dst, size_t n) const {
        const ParsedHeader h = parseHeader(data);
        if (h.N == 0 || h.N != n) return;
        std::fill(dst, dst + h.N, h.commonValue);
        if (h.uncommonCount == 0) return;
        scatterAll(h, dst);
    }

    void decodeRangeBytesInto(const uint8_t* data,
                               size_t start, size_t end,
                               T* dst, size_t n) const {
        const ParsedHeader h = parseHeader(data);
        if (start >= h.N) return;
        end = std::min(end, h.N);
        if (end - start != n) return;
        std::fill(dst, dst + n, h.commonValue);
        if (h.uncommonCount == 0) return;
        scatterRange(h, start, end, dst);
    }

    // -------------------------------------------------------------------------
    // Scatter helpers — write uncommon values into a pre-filled dst buffer.
    // -------------------------------------------------------------------------

    // Scatter all uncommon values for decodeAll / decodeAllBytesInto.
    void scatterAll(const ParsedHeader& h, T* dst) const {
        if (maxDepth_ == 0 || !innerCodec_) {
            scatterRaw(h, dst);
        } else if (innerMCE_) {
            std::unique_ptr<T[]> other(new T[h.uncommonCount]);
            innerMCE_->decodeAllBytesInto(h.otherValuesStart,
                                          other.get(), h.uncommonCount);
            scatterDecoded(h, other.get(), dst);
        } else {
            EncodedData inner;
            inner.data().assign(h.otherValuesStart,
                                h.otherValuesStart + h.otherValuesEncodedSize);
            auto other = innerCodec_->decodeAll(inner);
            scatterDecoded(h, other.data(), dst);
        }
    }

    // Full-bitmap scatter, raw (maxDepth_==0) case.
    void scatterRaw(const ParsedHeader& h, T* dst) const noexcept {
        uint32_t idx = 0;
        for (uint32_t w = 0; w < h.numWords; ++w) {
            uint64_t bword = loadWord(h.bitmapStart + w * 8);
            if (w == h.numWords - 1) {
                const uint32_t tail = static_cast<uint32_t>(h.N & 63);
                if (tail) bword |= ~((uint64_t{1} << tail) - 1);
            }
            uint64_t unset = ~bword;
            const uint32_t base = w * 64;
            while (unset) {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(unset));
                T val;
                std::memcpy(&val, h.otherValuesStart + idx * sizeof(T), sizeof(T));
                dst[base + bit] = val;
                ++idx;
                unset &= unset - 1;
            }
        }
    }

    // Full-bitmap scatter, inner-codec case.
    void scatterDecoded(const ParsedHeader& h,
                        const T* otherValues, T* dst) const noexcept {
        uint32_t idx = 0;
        for (uint32_t w = 0; w < h.numWords; ++w) {
            uint64_t bword = loadWord(h.bitmapStart + w * 8);
            if (w == h.numWords - 1) {
                const uint32_t tail = static_cast<uint32_t>(h.N & 63);
                if (tail) bword |= ~((uint64_t{1} << tail) - 1);
            }
            uint64_t unset = ~bword;
            const uint32_t base = w * 64;
            while (unset) {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(unset));
                dst[base + bit] = otherValues[idx++];
                unset &= unset - 1;
            }
        }
    }

    // Range scatter [start, end) with peeled head/tail; inner-codec uses innerMCE_.
    // Returns the number of uncommon values scattered (== rank delta across
    // [start, end)) so callers can carry the rank cursor forward across
    // multiple ranges without recomputing prefixNonCommonCount(0, start) from
    // scratch each time -- see decodeGatherInto. `seedOtherStart`, when
    // provided, is used instead of a fresh prefixNonCommonCount(bitmapStart,
    // start) call (which is what every existing single-range caller relies on
    // via the default nullopt).
    uint32_t scatterRange(const ParsedHeader& h,
                          size_t start, size_t end,
                          T* dst,
                          std::optional<uint32_t> seedOtherStart = std::nullopt) const {
        const uint32_t otherStart = seedOtherStart
            ? *seedOtherStart
            : prefixNonCommonCount(h.bitmapStart, start);
        const uint32_t wStart = static_cast<uint32_t>(start / 64);
        const uint32_t wEnd   = static_cast<uint32_t>((end - 1) / 64);
        const uint32_t remS   = static_cast<uint32_t>(start % 64);
        const uint32_t remE   = static_cast<uint32_t>(end   % 64);

        if (maxDepth_ == 0 || !innerCodec_) {
            const uint8_t* rawOther = h.otherValuesStart + otherStart * sizeof(T);
            uint32_t idx = 0;
            for (uint32_t w = wStart; w <= wEnd; ++w) {
                if (__builtin_expect(w + 1 <= wEnd, 1))
                    __builtin_prefetch(&dst[uint64_t{w + 1} * 64 - start], 1, 0);
                uint64_t unset = ~loadWord(h.bitmapStart + w * 8);
                if (__builtin_expect(w == wStart && remS != 0, 0))
                    unset &= ~((uint64_t{1} << remS) - 1);
                if (__builtin_expect(w == wEnd && remE != 0, 0))
                    unset &=  (uint64_t{1} << remE) - 1;
                const uint64_t wordBase = uint64_t{w} * 64;
                while (unset) {
                    const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(unset));
                    T val;
                    std::memcpy(&val, rawOther + idx * sizeof(T), sizeof(T));
                    dst[wordBase + bit - start] = val;
                    ++idx;
                    unset &= unset - 1;
                }
            }
            return idx;
        } else {
            const uint32_t neededCount =
                rangeNonCommonCount(h.bitmapStart, start, end);
            if (neededCount == 0) return 0;

            std::unique_ptr<T[]> slice;
            if (innerMCE_) {
                slice.reset(new T[neededCount]);
                innerMCE_->decodeRangeBytesInto(h.otherValuesStart,
                                                otherStart,
                                                otherStart + neededCount,
                                                slice.get(), neededCount);
            } else {
                EncodedData inner;
                inner.data().assign(h.otherValuesStart,
                                    h.otherValuesStart + h.otherValuesEncodedSize);
                auto tmp = innerCodec_->decodeRange(inner, otherStart,
                                                    otherStart + neededCount);
                slice.reset(new T[neededCount]);
                std::copy(tmp.begin(), tmp.end(), slice.get());
            }

            uint32_t idx = 0;
            for (uint32_t w = wStart; w <= wEnd; ++w) {
                if (__builtin_expect(w + 1 <= wEnd, 1))
                    __builtin_prefetch(&dst[uint64_t{w + 1} * 64 - start], 1, 0);
                uint64_t unset = ~loadWord(h.bitmapStart + w * 8);
                if (__builtin_expect(w == wStart && remS != 0, 0))
                    unset &= ~((uint64_t{1} << remS) - 1);
                if (__builtin_expect(w == wEnd && remE != 0, 0))
                    unset &=  (uint64_t{1} << remE) - 1;
                const uint64_t wordBase = uint64_t{w} * 64;
                while (unset) {
                    const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(unset));
                    dst[wordBase + bit - start] = slice[idx++];
                    unset &= unset - 1;
                }
            }
            return neededCount;
        }
    }

    // --- Wire write helper ---
    template<typename V>
    static void write(uint8_t*& pos, const V& value) {
        std::memcpy(pos, &value, sizeof(V));
        pos += sizeof(V);
    }
};

// ---------------------------------------------------------------------------
// Build a recursive MainlyConstantEncoder chain of the given depth.
// At depth 0: flat (raw otherValues).
// At depth k: otherValues encoded by a depth-(k-1) MC encoder.
// ---------------------------------------------------------------------------
template<typename T>
    requires core::IntegralType<T>
std::shared_ptr<MainlyConstantEncoder<T>>
makeRecursiveMainlyConstantEncoder(uint32_t maxDepth = 3) {
    if (maxDepth == 0)
        return std::make_shared<MainlyConstantEncoder<T>>(nullptr, 0);
    return std::make_shared<MainlyConstantEncoder<T>>(
        makeRecursiveMainlyConstantEncoder<T>(maxDepth - 1), maxDepth);
}

} // namespace encodings::encoders
