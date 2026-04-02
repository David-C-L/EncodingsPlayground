#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"
#include "encoders/RawEncoder.hpp"
#include "core/BitPacker.hpp"

namespace encodings::encoders {

// ---------------------------------------------------------------------------
// Adaptive Frame-of-Reference encoder
// ---------------------------------------------------------------------------

/**
 * @brief Adaptive FOR encoder that picks a frame size and residual width at runtime.
 *
 * - Template only on TIn (signed integral).
 * - Chooses frame size from a small candidate set by estimating encoded size on a single pass.
 * - Chooses the narrowest residual type (int8/int16/int32/int64) that fits the data for the selected frame size.
 * - Residuals are stored with RawEncoder for O(1) random access.
 *
 * Wire format (little-endian):
 *   [0 .. 7]   N              (uint64_t)
 *   [8 .. 11]  frameSize      (uint32_t)
 *   [12]       resWidthBytes  (uint8_t) — 1,2,4,8
 *   [13]       refPolicy      (uint8_t) — currently always MIN (=0) for future-proofing
 *   [14..15]   reserved/pad   (uint16_t = 0)
 *   [16 .. 23] refBytes       (uint64_t)
 *   [24 .. 31] resBytes       (uint64_t)
 *   [32 .. 32+refBytes-1]  reference values (TIn[numFrames])
 *   [32+refBytes .. end]    residual values  (ResType[N])
 */
template <typename TIn>
class AdaptiveFOREncoder : public Codec<TIn, uint8_t> {
    static_assert(std::is_integral_v<TIn> && std::is_signed_v<TIn>,
                  "AdaptiveFOREncoder: TIn must be signed integral");

    enum class ResidualWidth : uint8_t { W8 = 1, W16 = 2, W32 = 4, W64 = 8 };
    static constexpr uint8_t kBitpackedFlag = 0x80;

    static bool verboseEnabled() {
        static bool v = (std::getenv("ADAPTIVEFOR_VERBOSE") != nullptr);
        return v;
    }

    static const char* widthName(ResidualWidth w) {
        switch (w) {
            case ResidualWidth::W8:  return "int8";
            case ResidualWidth::W16: return "int16";
            case ResidualWidth::W32: return "int32";
            case ResidualWidth::W64: return "int64";
        }
        return "?";
    }

public:
    AdaptiveFOREncoder() = default;

    // Encode -----------------------------------------------------------------
    EncodedBuffer<uint8_t> encode(std::span<const TIn> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty();
        }

    const auto pick = choosePlan(data);

        // Prepare storage
        const size_t numFrames = (N + pick.frameSize - 1) / pick.frameSize;
        std::vector<TIn> refs(numFrames);

        // Residual buffer typed by width or bit-packed
        EncodedBuffer<uint8_t> residualEncoded;
        const bool bitpacked = pick.resBits != 0;
        if (bitpacked) {
            residualEncoded = encodeResidualsBitpacked(data, refs, pick.frameSize, pick.resBits);
        } else {
            switch (pick.resWidth) {
                case ResidualWidth::W8:  residualEncoded = encodeResiduals<int8_t >(data, refs, pick.frameSize); break;
                case ResidualWidth::W16: residualEncoded = encodeResiduals<int16_t>(data, refs, pick.frameSize); break;
                case ResidualWidth::W32: residualEncoded = encodeResiduals<int32_t>(data, refs, pick.frameSize); break;
                case ResidualWidth::W64: residualEncoded = encodeResiduals<int64_t>(data, refs, pick.frameSize); break;
            }
        }

        const uint64_t refBytes = static_cast<uint64_t>(refs.size() * sizeof(TIn));
        const uint64_t resBytes = static_cast<uint64_t>(residualEncoded.data().size());

        std::vector<uint8_t> out;
        out.reserve(headerSize() + refBytes + resBytes);

        // Header
        appendU64(out, static_cast<uint64_t>(N));
        appendU32(out, static_cast<uint32_t>(pick.frameSize));
        if (bitpacked) {
            out.push_back(static_cast<uint8_t>(kBitpackedFlag | pick.resBits));
        } else {
            out.push_back(static_cast<uint8_t>(pick.resWidth));
        }
        out.push_back(static_cast<uint8_t>(0)); // refPolicy MIN (reserved)
        out.push_back(0); out.push_back(0);     // pad to 16-byte alignment of offsets
        appendU64(out, refBytes);
        appendU64(out, resBytes);

        // Refs
        const size_t refOff = out.size();
        out.resize(refOff + refBytes);
        std::memcpy(out.data() + refOff, refs.data(), refBytes);

        // Residual payload
        out.insert(out.end(), residualEncoded.data().begin(), residualEncoded.data().end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<TIn>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(TIn);
        meta.supportsRandomAccess = true;

        if (verboseEnabled()) {
            const double refBitsPerElem = N ? (static_cast<double>(refBytes) * 8.0 / static_cast<double>(N)) : 0.0;
            const double resBitsPerElem = N ? (static_cast<double>(resBytes) * 8.0 / static_cast<double>(N)) : 0.0;
            const double totalBitsPerElem = N ? (static_cast<double>(out.size()) * 8.0 / static_cast<double>(N)) : 0.0;
            std::cerr << "[AdaptiveFOR] N=" << N
                      << " frame=" << pick.frameSize
                      << " resMode=" << (bitpacked ? ("bitpack(" + std::to_string(static_cast<int>(pick.resBits)) + "b)")
                                                     : widthName(pick.resWidth))
                      << " refs=" << refBytes << "B (~" << refBitsPerElem << " b/elem)"
                      << " residuals=" << resBytes << "B (~" << resBitsPerElem << " b/elem)"
                      << " total=" << out.size() << "B (~" << totalBitsPerElem << " b/elem)"
                      << " ratio=" << (N ? (static_cast<double>(out.size()) / (N * sizeof(TIn))) : 0.0)
                      << "\n";
        }

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // Decode all -------------------------------------------------------------
    std::vector<TIn> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const auto h = parseHeader(encoded);
        if (h.N == 0) return {};

        const auto& residualView = getCachedResidualView(encoded, h);
        const auto residuals = decodeResidualsAll(residualView, h);
        std::vector<TIn> out(h.N);
        const TIn* refs = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        for (size_t i = 0; i < h.N; ++i) {
            const size_t f = i / h.frameSize;
            out[i] = refs[f] + static_cast<TIn>(residuals[i]);
        }
        return out;
    }

    // Decode at --------------------------------------------------------------
    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto h = parseHeader(encoded);
        if (index >= h.N) return std::nullopt;

        const TIn* refs = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        const TIn ref = refs[index / h.frameSize];

        const auto& residualView = getCachedResidualView(encoded, h);
        const auto residual = decodeResidualAt(residualView, h, index);
        if (!residual) return std::nullopt;
        return ref + static_cast<TIn>(*residual);
    }

    // Decode range -----------------------------------------------------------
    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto h = parseHeader(encoded);
        end = std::min(end, h.N);
        if (start >= end) return {};
        const size_t count = end - start;

        const TIn* refs = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        const auto& residualView = getCachedResidualView(encoded, h);
        const auto residuals = decodeResidualRange(residualView, h, start, end);

        std::vector<TIn> out;
        out.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t idx = start + i;
            out.push_back(refs[idx / h.frameSize] + static_cast<TIn>(residuals[i]));
        }
        return out;
    }

    // Identity ---------------------------------------------------------------
    EncodingType encodingType() const override { return EncodingType::AdaptiveFrameOfReference; }

    std::string name() const override {
        return "AdaptiveFOR<" + std::string(typeid(TIn).name()) + ">";
    }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        return EncodingProperties{}
            .add(EP::RandomAccess)
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::FixedSize)
            .add(EP::DeltaBased)
            .add(EP::Composable)
            .add(EP::LowMemoryOverhead);
    }

private:
    struct Plan {
        size_t frameSize{};
        ResidualWidth resWidth{ResidualWidth::W64};
        uint8_t resBits{0}; // when non-zero, residuals are bit-packed with this width
        size_t estimatedBytes{};
    };

    static constexpr size_t headerSize() { return 32; }

    static void appendU64(std::vector<uint8_t>& buf, uint64_t v) {
        const size_t off = buf.size();
        buf.resize(off + sizeof(uint64_t));
        std::memcpy(buf.data() + off, &v, sizeof(uint64_t));
    }
    static void appendU32(std::vector<uint8_t>& buf, uint32_t v) {
        const size_t off = buf.size();
        buf.resize(off + sizeof(uint32_t));
        std::memcpy(buf.data() + off, &v, sizeof(uint32_t));
    }

    // Plan selection ---------------------------------------------------------
    Plan choosePlan(std::span<const TIn> data) const {
        static constexpr size_t kCandidates[] = {8, 32, 64, 128, 256, 512, 1024, 2048, 4096};

        Plan best;
        best.estimatedBytes = std::numeric_limits<size_t>::max();

        for (size_t frame : kCandidates) {
            const auto [resWidth, resBits] = estimateResidualWidth(data, frame);
            const size_t numFrames = (data.size() + frame - 1) / frame;
            const size_t refBytes = numFrames * sizeof(TIn);
            size_t resBytes = 0;
            if (resBits > 0) {
                const size_t bits = data.size() * static_cast<size_t>(resBits);
                resBytes = (bits + 7) / 8;
            } else {
                resBytes = data.size() * static_cast<size_t>(static_cast<uint8_t>(resWidth));
            }
            const size_t total = headerSize() + refBytes + resBytes;
            if (total < best.estimatedBytes) {
                best = Plan{frame, resWidth, static_cast<uint8_t>(resBits), total};
            }
        }
        return best;
    }

    static std::pair<ResidualWidth, uint8_t> estimateResidualWidth(std::span<const TIn> data, size_t frameSize) {
        using std::numeric_limits;
        const size_t N = data.size();
        const size_t numFrames = (N + frameSize - 1) / frameSize;

        TIn globalMin = numeric_limits<TIn>::max();
        TIn globalMax = numeric_limits<TIn>::min();

        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            TIn frameMin = numeric_limits<TIn>::max();
            TIn frameMax = numeric_limits<TIn>::min();
            for (size_t i = lo; i < hi; ++i) {
                frameMin = std::min(frameMin, data[i]);
                frameMax = std::max(frameMax, data[i]);
            }
            const TIn ref = frameMin; // MIN policy
            for (size_t i = lo; i < hi; ++i) {
                const TIn residual = data[i] - ref;
                globalMin = std::min(globalMin, residual);
                globalMax = std::max(globalMax, residual);
            }
        }

        const uint64_t maxResidual = static_cast<uint64_t>(static_cast<int64_t>(globalMax));
        const uint8_t neededBits = maxResidual == 0 ? 1 : static_cast<uint8_t>(64 - std::countl_zero(maxResidual));

        // Compare bit-packed vs typed storage and pick the smaller payload.
        size_t packedBytes = std::numeric_limits<size_t>::max();
        if (neededBits <= 32) {
            packedBytes = ((static_cast<size_t>(neededBits) * N) + 7) / 8;
        }

        const uint8_t widthBytes = (neededBits <= 8) ? 1 : (neededBits <= 16) ? 2 : (neededBits <= 32) ? 4 : 8;
        const size_t typedBytes = N * static_cast<size_t>(widthBytes);

        if (packedBytes < typedBytes && packedBytes != std::numeric_limits<size_t>::max()) {
            return {ResidualWidth::W32, neededBits}; // resWidth ignored when resBits>0
        }

        if (neededBits <= 32) {
            return {ResidualWidth::W32, 0};
        }
        return {ResidualWidth::W64, 0};
    }

    // Residual encoding helpers ---------------------------------------------
    template <typename ResT>
    EncodedBuffer<uint8_t> encodeResiduals(std::span<const TIn> data,
                                           std::vector<TIn>& refs,
                                           size_t frameSize) const {
        const size_t N = data.size();
        const size_t numFrames = refs.size();

        std::vector<ResT> residuals;
        residuals.reserve(N);

        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            const TIn ref = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                              data.begin() + static_cast<ptrdiff_t>(hi));
            refs[f] = ref;
            for (size_t i = lo; i < hi; ++i) {
                const auto residual = data[i] - ref;
                if (residual < static_cast<TIn>(std::numeric_limits<ResT>::min()) ||
                    residual > static_cast<TIn>(std::numeric_limits<ResT>::max())) {
                    throw std::overflow_error("AdaptiveFOR: residual overflow after plan selection");
                }
                residuals.push_back(static_cast<ResT>(residual));
            }
        }

        RawEncoder<ResT> raw;
        return raw.encode(std::span<const ResT>(residuals.data(), residuals.size()));
    }

    EncodedBuffer<uint8_t> encodeResidualsBitpacked(std::span<const TIn> data,
                                                    std::vector<TIn>& refs,
                                                    size_t frameSize,
                                                    uint8_t bits) const {
        const size_t N = data.size();
        const size_t numFrames = refs.size();

        std::vector<uint64_t> residuals;
        residuals.reserve(N);

        uint64_t maxResidual = 0;
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            const TIn ref = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                              data.begin() + static_cast<ptrdiff_t>(hi));
            refs[f] = ref;
            for (size_t i = lo; i < hi; ++i) {
                const uint64_t residual = static_cast<uint64_t>(data[i] - ref);
                residuals.push_back(residual);
                maxResidual = std::max(maxResidual, residual);
            }
        }

        uint8_t bw = bits;
        if (bw == 0) {
            bw = maxResidual == 0 ? 1 : static_cast<uint8_t>(64 - std::countl_zero(maxResidual));
        }
        if (bw > 32) {
            throw std::runtime_error("AdaptiveFOR: bitpacked residual width exceeds 32 bits");
        }

        std::vector<uint8_t> buf;
        buf.reserve((residuals.size() * bw + 7) / 8);
        encodings::core::BitWriter wr(buf, encodings::core::BitOrder::LSB);
        for (uint64_t r : residuals) {
            wr.write(static_cast<uint32_t>(r), bw);
        }
        wr.flush();

        encodings::EncodingMetadata meta;
        meta.encodingName         = "AdaptiveFORResidualBitpack";
        meta.elementCount         = N;
        meta.compressedSize       = buf.size();
        meta.uncompressedSize     = N * sizeof(uint64_t);
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(buf), std::move(meta));
    }

    // Decoding helpers -------------------------------------------------------
    struct ParsedHeader {
        size_t N{};
        size_t frameSize{};
        ResidualWidth resWidth{ResidualWidth::W64};
        uint8_t resBits{0};
        size_t refOffset{};
        size_t resOffset{};
        size_t resBytes{};
    };

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& encoded) {
        if (encoded.data().size() < headerSize()) {
            throw std::runtime_error("AdaptiveFOR: buffer too small for header");
        }
        const uint8_t* p = encoded.data().data();
        ParsedHeader h;
        h.N = readU64(p); p += 8;
        h.frameSize = readU32(p); p += 4;
        const uint8_t resMode = *p++;
        if (resMode & kBitpackedFlag) {
            h.resBits = static_cast<uint8_t>(resMode & 0x7F);
            h.resWidth = ResidualWidth::W32; // logical type unused for bitpacked
        } else {
            h.resWidth = static_cast<ResidualWidth>(resMode);
        }
        p += 3; // skip refPolicy + pad
    const uint64_t refBytes = readU64(p); p += 8;
    const uint64_t resBytes = readU64(p); p += 8;

    h.refOffset = static_cast<size_t>(p - encoded.data().data());
    h.resOffset = h.refOffset + static_cast<size_t>(refBytes);
    h.resBytes  = static_cast<size_t>(resBytes);

    const size_t end = h.resOffset + static_cast<size_t>(resBytes);
        if (end > encoded.data().size()) {
            throw std::runtime_error("AdaptiveFOR: header sizes exceed buffer length");
        }
        return h;
    }

    template <typename ResT>
    static std::vector<ResT> decodeResidualsAllTyped(const EncodedBuffer<uint8_t>& residualView,
                                                     const ParsedHeader& /*h*/) {
        RawEncoder<ResT> raw;
        return raw.decodeAll(residualView);
    }

    static std::vector<int64_t> decodeResidualsAllBitpacked(const EncodedBuffer<uint8_t>& residualView,
                                                            const ParsedHeader& h) {
        if (h.resBits == 0) return std::vector<int64_t>(h.N, 0);
        const uint8_t* data = residualView.data().data();
        const size_t size = residualView.data().size();
        encodings::core::BitReader r(data, size, encodings::core::BitOrder::LSB);
        std::vector<int64_t> out;
        out.reserve(h.N);
        for (size_t i = 0; i < h.N; ++i) {
            out.push_back(static_cast<int64_t>(r.read(h.resBits)));
        }
        return out;
    }

    std::vector<int64_t> decodeResidualsAll(const EncodedBuffer<uint8_t>& residualView,
                                            const ParsedHeader& h) const {
        if (h.resBits) {
            return decodeResidualsAllBitpacked(residualView, h);
        }
        switch (h.resWidth) {
            case ResidualWidth::W8:  return widen(decodeResidualsAllTyped<int8_t >(residualView, h));
            case ResidualWidth::W16: return widen(decodeResidualsAllTyped<int16_t>(residualView, h));
            case ResidualWidth::W32: return widen(decodeResidualsAllTyped<int32_t>(residualView, h));
            case ResidualWidth::W64: return widen(decodeResidualsAllTyped<int64_t>(residualView, h));
        }
        return {};
    }

    template <typename ResT>
    static std::optional<int64_t> decodeResidualAtTyped(const EncodedBuffer<uint8_t>& residualView,
                                                        const ParsedHeader& /*h*/,
                                                        size_t idx) {
        RawEncoder<ResT> raw;
        auto v = raw.decodeAt(residualView, idx);
        if (!v) return std::nullopt;
        return static_cast<int64_t>(*v);
    }

    static std::optional<int64_t> decodeResidualAtBitpacked(const EncodedBuffer<uint8_t>& residualView,
                                                            const ParsedHeader& h,
                                                            size_t idx) {
        if (idx >= h.N) return std::nullopt;
        if (h.resBits == 0) return static_cast<int64_t>(0);
        const uint8_t* data = residualView.data().data();
        const size_t size = residualView.data().size();
        encodings::core::BitReader r(data, size, encodings::core::BitOrder::LSB);
        r.seekToBit(idx * static_cast<size_t>(h.resBits));
        return static_cast<int64_t>(r.read(h.resBits));
    }

    std::optional<int64_t> decodeResidualAt(const EncodedBuffer<uint8_t>& residualView,
                                            const ParsedHeader& h,
                                            size_t idx) const {
        if (h.resBits) {
            return decodeResidualAtBitpacked(residualView, h, idx);
        }
        switch (h.resWidth) {
            case ResidualWidth::W8:  return decodeResidualAtTyped<int8_t >(residualView, h, idx);
            case ResidualWidth::W16: return decodeResidualAtTyped<int16_t>(residualView, h, idx);
            case ResidualWidth::W32: return decodeResidualAtTyped<int32_t>(residualView, h, idx);
            case ResidualWidth::W64: return decodeResidualAtTyped<int64_t>(residualView, h, idx);
        }
        return std::nullopt;
    }

    template <typename ResT>
    static std::vector<int64_t> decodeResidualRangeTyped(const EncodedBuffer<uint8_t>& residualView,
                                                         const ParsedHeader& /*h*/,
                                                         size_t start,
                                                         size_t end) {
        RawEncoder<ResT> raw;
        auto vals = raw.decodeRange(residualView, start, end);
        std::vector<int64_t> out;
        out.reserve(vals.size());
        for (auto v : vals) out.push_back(static_cast<int64_t>(v));
        return out;
    }

    static std::vector<int64_t> decodeResidualRangeBitpacked(const EncodedBuffer<uint8_t>& residualView,
                                                             const ParsedHeader& h,
                                                             size_t start,
                                                             size_t end) {
        if (start >= end) return {};
        if (h.resBits == 0) return std::vector<int64_t>(end - start, 0);
        const uint8_t* data = residualView.data().data();
        const size_t size = residualView.data().size();
        encodings::core::BitReader r(data, size, encodings::core::BitOrder::LSB);
        r.seekToBit(start * static_cast<size_t>(h.resBits));
        std::vector<int64_t> out;
        out.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            out.push_back(static_cast<int64_t>(r.read(h.resBits)));
        }
        return out;
    }

    std::vector<int64_t> decodeResidualRange(const EncodedBuffer<uint8_t>& residualView,
                                             const ParsedHeader& h,
                                             size_t start,
                                             size_t end) const {
        if (h.resBits) {
            return decodeResidualRangeBitpacked(residualView, h, start, end);
        }
        switch (h.resWidth) {
            case ResidualWidth::W8:  return decodeResidualRangeTyped<int8_t >(residualView, h, start, end);
            case ResidualWidth::W16: return decodeResidualRangeTyped<int16_t>(residualView, h, start, end);
            case ResidualWidth::W32: return decodeResidualRangeTyped<int32_t>(residualView, h, start, end);
            case ResidualWidth::W64: return decodeResidualRangeTyped<int64_t>(residualView, h, start, end);
        }
        return {};
    }

    static uint64_t readU64(const uint8_t* p) {
        uint64_t v; std::memcpy(&v, p, sizeof(uint64_t)); return v;
    }
    static uint32_t readU32(const uint8_t* p) {
        uint32_t v; std::memcpy(&v, p, sizeof(uint32_t)); return v;
    }

    static std::vector<int64_t> widen(const std::vector<int8_t>& v)  { return {v.begin(), v.end()}; }
    static std::vector<int64_t> widen(const std::vector<int16_t>& v) { return {v.begin(), v.end()}; }
    static std::vector<int64_t> widen(const std::vector<int32_t>& v) { return {v.begin(), v.end()}; }
    static std::vector<int64_t> widen(const std::vector<int64_t>& v) { return v; }

    const EncodedBuffer<uint8_t>& getCachedResidualView(const EncodedBuffer<uint8_t>& encoded,
                                                        const ParsedHeader& h) const {
        const uint8_t* basePtr = encoded.data().data();
        const size_t totalSize = encoded.data().size();

        if (cachedOuterPtr_ != basePtr || cachedOuterSize_ != totalSize) {
            std::vector<uint8_t> residualPayload(h.resBytes);
            std::memcpy(residualPayload.data(), basePtr + h.resOffset, h.resBytes);

            encodings::EncodingMetadata residualMeta;
            residualMeta.elementCount     = h.N;
            residualMeta.compressedSize   = h.resBytes;
            residualMeta.uncompressedSize = h.resBits
                ? (h.N * static_cast<size_t>(h.resBits) + 7) / 8
                : h.N * static_cast<size_t>(static_cast<uint8_t>(h.resWidth));
            residualMeta.supportsRandomAccess = true;

            cachedResidualView_ = encodings::EncodedData(std::move(residualPayload), std::move(residualMeta));
            cachedOuterPtr_ = basePtr;
            cachedOuterSize_ = totalSize;
        }

        return cachedResidualView_;
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(headerSize(), 0);
        encodings::EncodingMetadata meta;
        meta.encodingName         = "AdaptiveFOR(empty)";
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    // Cache of extracted residual payload keyed by outer encoded buffer identity.
    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t cachedOuterSize_{0};
    mutable EncodedBuffer<uint8_t> cachedResidualView_;
};

// Convenience factory
template <typename TIn>
std::shared_ptr<AdaptiveFOREncoder<TIn>> makeAdaptiveFOR() {
    return std::make_shared<AdaptiveFOREncoder<TIn>>();
}

} // namespace encodings::encoders
