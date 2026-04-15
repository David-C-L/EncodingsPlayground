#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/SubIntEncodingUtils.hpp"

namespace encodings::encoders {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct TriSplitConfig64 : SubIntSplitConfigIntegral<uint64_t> {
    uint8_t bitsA{0};
    uint8_t bitsB{0};
    uint8_t bitsC{0};

    std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecA;
    std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecB;
    std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecC;

    TriSplitConfig64() = default;

    TriSplitConfig64(uint8_t a, uint8_t b, uint8_t c,
                     std::shared_ptr<ISectionCodecIntegral<uint64_t>> ca,
                     std::shared_ptr<ISectionCodecIntegral<uint64_t>> cb,
                     std::shared_ptr<ISectionCodecIntegral<uint64_t>> cc,
                     BitSplitOrder ord = BitSplitOrder::LSB_TO_MSB) {
        bitsA  = a;
        bitsB  = b;
        bitsC  = c;
        codecA = std::move(ca);
        codecB = std::move(cb);
        codecC = std::move(cc);
        order  = ord;
        ensureBaseVectors();
    }

    void ensureBaseVectors() {
        if (bits.empty()) {
            bits = {bitsA, bitsB, bitsC};
        }
        if (codecs.empty()) {
            codecs = {codecA, codecB, codecC};
        }
    }
};

// ---------------------------------------------------------------------------
// TriSplitEncoder64
// ---------------------------------------------------------------------------

class TriSplitEncoder64 final : public Codec<int64_t, uint8_t> {
public:
    explicit TriSplitEncoder64(TriSplitConfig64 cfg) : cfg_(std::move(cfg)) {
        cfg_.ensureBaseVectors();
        validateConfig();
    }

    EncodedBuffer<uint8_t> encode(std::span<const int64_t> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty();
        }

        std::vector<uint64_t> secA(N), secB(N), secC(N);
        for (size_t i = 0; i < N; ++i) {
            splitValue(static_cast<uint64_t>(data[i]), secA[i], secB[i], secC[i]);
        }

        auto encA = cfg_.codecA->encode(std::span<const uint64_t>(secA.data(), secA.size()));
        auto encB = cfg_.codecB->encode(std::span<const uint64_t>(secB.data(), secB.size()));
        auto encC = cfg_.codecC->encode(std::span<const uint64_t>(secC.data(), secC.size()));

        const uint64_t bytesA = encA.data().size();
        const uint64_t bytesB = encB.data().size();
        const uint64_t bytesC = encC.data().size();

        // Header: [N:8][bitsA:1][bitsB:1][bitsC:1][order:1][bytesA:8][bytesB:8][bytesC:8]
        std::vector<uint8_t> out;
        out.resize(headerSize());

        uint8_t* p = out.data();
        writeU64(p, static_cast<uint64_t>(N));
        p += 8;
        *p++ = cfg_.bitsA;
        *p++ = cfg_.bitsB;
        *p++ = cfg_.bitsC;
        *p++ = static_cast<uint8_t>(cfg_.order);
        writeU64(p, bytesA); p += 8;
        writeU64(p, bytesB); p += 8;
        writeU64(p, bytesC); p += 8;

        out.insert(out.end(), encA.data().begin(), encA.data().end());
        out.insert(out.end(), encB.data().begin(), encB.data().end());
        out.insert(out.end(), encC.data().begin(), encC.data().end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<int64_t>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(int64_t);
        meta.supportsRandomAccess = allSectionsRandomAccess();

        if (verboseEnabled()) {
            const double bitsPerElemA = N ? (static_cast<double>(bytesA) * 8.0 / static_cast<double>(N)) : 0.0;
            const double bitsPerElemB = N ? (static_cast<double>(bytesB) * 8.0 / static_cast<double>(N)) : 0.0;
            const double bitsPerElemC = N ? (static_cast<double>(bytesC) * 8.0 / static_cast<double>(N)) : 0.0;
            const double bitsPerElemTotal = N ? (static_cast<double>(out.size()) * 8.0 / static_cast<double>(N)) : 0.0;
            std::cerr << "[TriSplit] N=" << N
                      << " order=" << (cfg_.order == BitSplitOrder::LSB_TO_MSB ? "LSB" : "MSB")
                      << " bits=" << static_cast<int>(cfg_.bitsA) << "|" << static_cast<int>(cfg_.bitsB)
                      << "|" << static_cast<int>(cfg_.bitsC)
                      << " A=" << bytesA << "B (~" << bitsPerElemA << " b/elem)"
                      << " B=" << bytesB << "B (~" << bitsPerElemB << " b/elem)"
                      << " C=" << bytesC << "B (~" << bitsPerElemC << " b/elem)"
                      << " total=" << out.size() << "B (~" << bitsPerElemTotal << " b/elem)"
                      << " ratio=" << (N ? (static_cast<double>(out.size()) / (N * sizeof(int64_t))) : 0.0)
                      << "\n";
        }

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    std::vector<int64_t> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const auto& h = getCachedHeader(encoded);
        if (h.N == 0) return {};

        auto secA = cfg_.codecA->decodeAll(h.viewA);
        auto secB = cfg_.codecB->decodeAll(h.viewB);
        auto secC = cfg_.codecC->decodeAll(h.viewC);

        if (secA.size() != h.N || secB.size() != h.N || secC.size() != h.N) {
            throw std::runtime_error("TriSplitEncoder64::decodeAll: section size mismatch");
        }

        std::vector<int64_t> out(h.N);
        for (size_t i = 0; i < h.N; ++i) {
            out[i] = static_cast<int64_t>(combine(secA[i], secB[i], secC[i]));
        }
        return out;
    }

    std::optional<int64_t> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto& h = getCachedHeader(encoded);
        if (index >= h.N) return std::nullopt;

        const uint64_t a = decodeOneSection(*cfg_.codecA, h.viewA, index);
        const uint64_t b = decodeOneSection(*cfg_.codecB, h.viewB, index);
        const uint64_t c = decodeOneSection(*cfg_.codecC, h.viewC, index);

        return static_cast<int64_t>(combine(a, b, c));
    }

    std::vector<int64_t> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto& h = getCachedHeader(encoded);
        if (start >= h.N) return {};
        end = std::min(end, h.N);
        if (start >= end) return {};
        const size_t count = end - start;

        const auto secA = decodeSectionRange(*cfg_.codecA, h.viewA, start, end);
        const auto secB = decodeSectionRange(*cfg_.codecB, h.viewB, start, end);
        const auto secC = decodeSectionRange(*cfg_.codecC, h.viewC, start, end);

        if (secA.size() != count || secB.size() != count || secC.size() != count) {
            throw std::runtime_error("TriSplitEncoder64::decodeRange: section size mismatch");
        }

        std::vector<int64_t> out;
        out.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            out.push_back(static_cast<int64_t>(combine(secA[i], secB[i], secC[i])));
        }
        return out;
    }

    EncodingType encodingType() const override { return EncodingType::Structural; }

    std::string name() const override {
        return "TriSplit(" + std::to_string(cfg_.bitsA) + "|" +
               std::to_string(cfg_.bitsB) + "|" + std::to_string(cfg_.bitsC) +
               (cfg_.order == BitSplitOrder::LSB_TO_MSB ? ",LSB" : ",MSB") + ")";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        EncodingProperties props;
        props.add(EP::Lossless)
             .add(EP::PreservesOrder)
             .add(EP::Composable);
        if (allSectionsRandomAccess()) props.add(EP::RandomAccess);
        return props;
    }

private:
    // -------------------------------------------------------------------
    // Header helpers
    // -------------------------------------------------------------------
    struct ParsedHeader {
        size_t N{};
        uint8_t bitsA{};
        uint8_t bitsB{};
        uint8_t bitsC{};
        BitSplitOrder order{BitSplitOrder::LSB_TO_MSB};
        uint64_t bytesA{};
        uint64_t bytesB{};
        uint64_t bytesC{};
        EncodedBuffer<uint8_t> viewA;
        EncodedBuffer<uint8_t> viewB;
        EncodedBuffer<uint8_t> viewC;
    };

    static constexpr size_t headerSize() {
        return 8 + 1 + 1 + 1 + 1 + 8 + 8 + 8; // 36 bytes
    }

    static void writeU64(uint8_t* dst, uint64_t v) {
        std::memcpy(dst, &v, sizeof(uint64_t));
    }
    static uint64_t readU64(const uint8_t* src) {
        uint64_t v; std::memcpy(&v, src, sizeof(uint64_t)); return v; }

    const ParsedHeader& getCachedHeader(const EncodedBuffer<uint8_t>& encoded) const {
        const uint8_t* basePtr = encoded.data().data();
        const size_t totalSize = encoded.data().size();

        if (cachedOuterPtr_ != basePtr || cachedOuterSize_ != totalSize) {
            cachedHeader_ = parseHeader(encoded);
            cachedOuterPtr_ = basePtr;
            cachedOuterSize_ = totalSize;
        }
        return cachedHeader_;
    }

    ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& encoded) const {
        if (encoded.data().size() < headerSize()) {
            throw std::runtime_error("TriSplitEncoder64: buffer too small for header");
        }

        const uint8_t* p = encoded.data().data();
        const size_t totalSize = encoded.data().size();

        ParsedHeader h;
        h.N     = static_cast<size_t>(readU64(p)); p += 8;
        h.bitsA = *p++;
        h.bitsB = *p++;
        h.bitsC = *p++;
        h.order = static_cast<BitSplitOrder>(*p++);
        h.bytesA = readU64(p); p += 8;
        h.bytesB = readU64(p); p += 8;
        h.bytesC = readU64(p); p += 8;

        if (h.bitsA + h.bitsB + h.bitsC != 64) {
            throw std::runtime_error("TriSplitEncoder64: invalid header bit sum");
        }

        const size_t offA = headerSize();
        const size_t offB = offA + static_cast<size_t>(h.bytesA);
        const size_t offC = offB + static_cast<size_t>(h.bytesB);
        if (offC + static_cast<size_t>(h.bytesC) > totalSize) {
            throw std::runtime_error("TriSplitEncoder64: payload sizes exceed buffer");
        }

        h.viewA = slice(encoded, offA, h.bytesA, h.N, h.bitsA);
        h.viewB = slice(encoded, offB, h.bytesB, h.N, h.bitsB);
        h.viewC = slice(encoded, offC, h.bytesC, h.N, h.bitsC);
        return h;
    }

    static EncodedBuffer<uint8_t> slice(const EncodedBuffer<uint8_t>& base,
                                        size_t off, size_t len,
                                        size_t elementCount, uint8_t bits) {
        const uint8_t* src = base.data().data() + off;
        std::vector<uint8_t> payload(src, src + len);

        // Reconstruct minimal metadata so sub-codecs (notably OpenZL) know the
        // expected uncompressed byte length. Mirror the narrowing logic used
        // when selecting section codecs from bit widths.
        const uint8_t chosenBits = (bits <= 8)  ? 8  : (bits <= 16) ? 16
                                   : (bits <= 32) ? 32 : 64;
        const size_t elemBytes   = static_cast<size_t>(chosenBits) / 8;

        encodings::EncodingMetadata meta;
        meta.encodingName     = "TriSplitSection";
        meta.elementCount     = elementCount;
        meta.compressedSize   = len;
        meta.uncompressedSize = elementCount * elemBytes;

        switch (chosenBits) {
            case 8:  meta.dataType = encodings::core::typeToDataType<uint8_t>;  break;
            case 16: meta.dataType = encodings::core::typeToDataType<uint16_t>; break;
            case 32: meta.dataType = encodings::core::typeToDataType<uint32_t>; break;
            default: meta.dataType = encodings::core::typeToDataType<uint64_t>; break;
        }
        // RA flag is determined by the section codec itself.
        return encodings::EncodedData(std::move(payload), std::move(meta));
    }

    // -------------------------------------------------------------------
    // Split/combine helpers
    // -------------------------------------------------------------------
    void validateConfig() {
        if (cfg_.bitsA == 0 || cfg_.bitsB == 0 || cfg_.bitsC == 0) {
            throw std::invalid_argument("TriSplitEncoder64: bits for each section must be > 0");
        }
        if (static_cast<uint16_t>(cfg_.bitsA) + cfg_.bitsB + cfg_.bitsC != 64) {
            throw std::invalid_argument("TriSplitEncoder64: bitsA+bitsB+bitsC must equal 64");
        }
        if (!cfg_.codecA || !cfg_.codecB || !cfg_.codecC) {
            throw std::invalid_argument("TriSplitEncoder64: codecs must not be null");
        }
        cfg_.ensureBaseVectors();
        if (cfg_.bits.size() != 3 || cfg_.codecs.size() != 3) {
            throw std::invalid_argument("TriSplitEncoder64: base config must have exactly 3 splits");
        }
    }

    void splitValue(uint64_t v, uint64_t& a, uint64_t& b, uint64_t& c) const {
        if (cfg_.order == BitSplitOrder::LSB_TO_MSB) {
            const uint64_t maskA = cfg_.bitsA == 64 ? ~uint64_t{0} : ((uint64_t{1} << cfg_.bitsA) - 1);
            const uint64_t maskB = cfg_.bitsB == 64 ? ~uint64_t{0} : ((uint64_t{1} << cfg_.bitsB) - 1);
            a = v & maskA;
            v >>= cfg_.bitsA;
            b = v & maskB;
            v >>= cfg_.bitsB;
            c = v; // remaining bits
        } else {
            const uint8_t highShift = 64 - cfg_.bitsA;
            a = v >> highShift;
            v &= ((uint64_t{1} << highShift) - 1);
            const uint8_t midShift = highShift - cfg_.bitsB;
            b = v >> midShift;
            c = v & ((uint64_t{1} << midShift) - 1);
        }
    }

    uint64_t combine(uint64_t a, uint64_t b, uint64_t c) const {
        if (cfg_.order == BitSplitOrder::LSB_TO_MSB) {
            uint64_t v = a;
            v |= (b << cfg_.bitsA);
            v |= (c << (cfg_.bitsA + cfg_.bitsB));
            return v;
        } else {
            uint64_t v = a;
            v <<= cfg_.bitsB;
            v |= b;
            v <<= cfg_.bitsC;
            v |= c;
            return v;
        }
    }

    bool allSectionsRandomAccess() const {
        auto has = [](const EncodingProperties& p) {
            return p.has(EncodingProperty::RandomAccess);
        };
        return has(cfg_.codecA->properties()) && has(cfg_.codecB->properties()) && has(cfg_.codecC->properties());
    }

    static uint64_t decodeOneSection(ISectionCodecIntegral<uint64_t>& c, const EncodedBuffer<uint8_t>& view, size_t idx) {
    if (c.properties().has(EncodingProperty::RandomAccess)) {
            auto v = c.decodeAt(view, idx);
            if (!v) throw std::runtime_error("TriSplitEncoder64::decodeAt: subcodec returned null");
            return *v;
        }
        // No RA: decodeAll then index (no persistent caching by design; could be optimized later).
        auto all = c.decodeAll(view);
        if (idx >= all.size()) throw std::runtime_error("TriSplitEncoder64::decodeAt: index out of range after full decode");
        return all[idx];
    }

    static std::vector<uint64_t> decodeSectionRange(ISectionCodecIntegral<uint64_t>& c, const EncodedBuffer<uint8_t>& view, size_t start, size_t end) {
    if (c.properties().has(EncodingProperty::RandomAccess)) {
            return c.decodeRange(view, start, end);
        }
        auto all = c.decodeAll(view);
        if (start > all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<uint64_t>(all.begin() + static_cast<ptrdiff_t>(start), all.begin() + static_cast<ptrdiff_t>(end));
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(headerSize(), 0);
        encodings::EncodingMetadata meta;
        meta.encodingName         = "TriSplit(empty)";
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    static bool verboseEnabled() {
        static bool v = (std::getenv("TRISPLIT_VERBOSE") != nullptr);
        return true || v;
    }

private:
    TriSplitConfig64 cfg_;
    // Cache of parsed header and section views keyed by outer encoded buffer identity.
    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t cachedOuterSize_{0};
    mutable ParsedHeader cachedHeader_{};
};

// ---------------------------------------------------------------------------
// Helpers / presets (Snowflake-friendly)
// ---------------------------------------------------------------------------

namespace detail_trisplit {
enum class SnowflakeVariant {
    Snowflake,
    Twitter
};

struct SnowflakeLayout {
    static constexpr uint8_t timestampBits = 41; // Instagram/Snowflake style
    static constexpr uint8_t machineBits   = 13;
    static constexpr uint8_t sequenceBits  = 10;
};

struct TwitterSnowflakeLayout {
    static constexpr uint8_t timestampBits = 41; // Twitter/Snowflake style
    static constexpr uint8_t machineBits   = 10;
    static constexpr uint8_t sequenceBits  = 13;
};

inline std::tuple<uint8_t, uint8_t, uint8_t> getSnowflakeLayoutBits(SnowflakeVariant variant) {
    switch (variant) {
    case SnowflakeVariant::Twitter:
        return {TwitterSnowflakeLayout::timestampBits,
                TwitterSnowflakeLayout::machineBits,
                TwitterSnowflakeLayout::sequenceBits};
    case SnowflakeVariant::Snowflake:
        return {SnowflakeLayout::timestampBits,
                SnowflakeLayout::machineBits,
                SnowflakeLayout::sequenceBits};
    default:
        return {SnowflakeLayout::timestampBits,
                SnowflakeLayout::machineBits,
                SnowflakeLayout::sequenceBits};
    }
}

TriSplitConfig64 makeSnowflakeConfig(std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecA,
                                     std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecB,
                                     std::shared_ptr<ISectionCodecIntegral<uint64_t>> codecC,
                                     BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
                                     SnowflakeVariant variant = SnowflakeVariant::Snowflake) {
    const auto [baseBitsA, baseBitsB, baseBitsC] = getSnowflakeLayoutBits(variant);

    const uint16_t total = static_cast<uint16_t>(baseBitsA) + baseBitsB + baseBitsC;
    const uint8_t pad = static_cast<uint8_t>(64 - total);

    // Add any remaining pad bits to the sequence partition to satisfy 64-bit total.
    return TriSplitConfig64(
        baseBitsA,
        baseBitsB,
        static_cast<uint8_t>(baseBitsC + pad),
        std::move(codecA),
        std::move(codecB),
        std::move(codecC),
        order);
}

} // namespace detail_trisplit

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

inline std::shared_ptr<TriSplitEncoder64> makeTriSplitEncoder64(TriSplitConfig64 cfg) {
    return std::make_shared<TriSplitEncoder64>(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Ready-made TriSplit variants (Snowflake-friendly)
// ---------------------------------------------------------------------------

inline std::shared_ptr<TriSplitEncoder64> makeTriSplitAllDict(uint8_t bitsA, uint8_t bitsB, uint8_t bitsC,
                                                             BitSplitOrder order = BitSplitOrder::LSB_TO_MSB) {
    auto cfg = TriSplitConfig64(
        bitsA,
        bitsB,
        bitsC,
        detail_trisplit::makeDictionarySection(bitsA),
        detail_trisplit::makeDictionarySection(bitsB),
        detail_trisplit::makeDictionarySection(bitsC),
        order);
    return makeTriSplitEncoder64(std::move(cfg));
}

inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitAllDict(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeDictionarySection(bitsA),
        detail_trisplit::makeDictionarySection(bitsB),
        detail_trisplit::makeDictionarySection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}


inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitAllRawBitPacked(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeRawBitPackedSection(bitsA),
        detail_trisplit::makeRawBitPackedSection(bitsB),
        detail_trisplit::makeRawBitPackedSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitAllHuffman(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeHuffmanSection(bitsA),
        detail_trisplit::makeHuffmanSection(bitsB),
        detail_trisplit::makeHuffmanSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

template <size_t BlockSize = 0>
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitOpenZLOnly(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeOpenZLSection<BlockSize>(bitsA),
        detail_trisplit::makeOpenZLSection<BlockSize>(bitsB),
        detail_trisplit::makeOpenZLSection<BlockSize>(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

// Backwards-compatible alias used by benchmarks
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitDictOnly(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    return makeSnowflakeTriSplitAllDict(order, variant);
}

inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitFORDictFOR(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFORSection(bitsA),
        detail_trisplit::makeDictionarySection(bitsB),
        detail_trisplit::makeAdaptiveFORSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitFORRawFOR(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFORSection(bitsA),
        detail_trisplit::makeRawSection(bitsB),
        detail_trisplit::makeAdaptiveFORSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

// Variant: FOR on timestamp, dictionary on machine, dictionary on sequence (cheap + RA)
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitFORDictDict(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFORSection(bitsA),
        detail_trisplit::makeDictionarySection(bitsB),
        detail_trisplit::makeDictionarySection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

// Variant: FOR on timestamp, dictionary on machine, raw on sequence (fast decode baseline)
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitFORDictRaw(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFORSection(bitsA),
        detail_trisplit::makeDictionarySection(bitsB),
        detail_trisplit::makeRawSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

// All-FOR preset (timestamp/machine/sequence) when per-section deltas are small.
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitFOROnly(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFORSection(bitsA),
        detail_trisplit::makeAdaptiveFORSection(bitsB),
        detail_trisplit::makeAdaptiveFORSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

// All-AdaptiveFramedBitPrefix preset (timestamp/machine/sequence) when per-section prefixes are small.
inline std::shared_ptr<TriSplitEncoder64> makeSnowflakeTriSplitBitPrefixOnly(
    BitSplitOrder order = BitSplitOrder::MSB_TO_LSB,
    detail_trisplit::SnowflakeVariant variant = detail_trisplit::SnowflakeVariant::Snowflake) {
    const auto [bitsA, bitsB, bitsC] = detail_trisplit::getSnowflakeLayoutBits(variant);
    auto cfg = detail_trisplit::makeSnowflakeConfig(
        detail_trisplit::makeAdaptiveFramedBitPrefixSection(bitsA),
        detail_trisplit::makeAdaptiveFramedBitPrefixSection(bitsB),
        detail_trisplit::makeAdaptiveFramedBitPrefixSection(bitsC),
        order,
        variant);
    return makeTriSplitEncoder64(std::move(cfg));
}

} // namespace encodings::encoders
