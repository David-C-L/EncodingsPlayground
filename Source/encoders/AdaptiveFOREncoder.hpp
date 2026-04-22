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

        // Build output in one allocation: header + refs + residuals
        const size_t total = headerSize() + static_cast<size_t>(refBytes) + static_cast<size_t>(resBytes);
        std::vector<uint8_t> out(total);
        uint8_t* p = out.data();

        writeU64(p, static_cast<uint64_t>(N));          p += 8;
        writeU32(p, static_cast<uint32_t>(pick.frameSize)); p += 4;
        if (bitpacked) {
            *p++ = static_cast<uint8_t>(kBitpackedFlag | pick.resBits);
        } else {
            *p++ = static_cast<uint8_t>(pick.resWidth);
        }
        *p++ = 0;  // refPolicy MIN (reserved)
        *p++ = 0;  // pad
        *p++ = 0;  // pad
        writeU64(p, refBytes);  p += 8;
        writeU64(p, resBytes);  p += 8;

        std::memcpy(p, refs.data(), static_cast<size_t>(refBytes));
        p += static_cast<size_t>(refBytes);
        std::memcpy(p, residualEncoded.data().data(), static_cast<size_t>(resBytes));

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

        const TIn*     refs    = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        const uint8_t* resData = encoded.data().data() + h.resOffset;

        std::vector<TIn> out(h.N);

        if (h.resBits) {
            encodings::core::BitReader r(resData, h.resBytes, encodings::core::BitOrder::LSB);
            size_t frameIdx = 0;
            size_t frameEnd = h.frameSize;
            for (size_t i = 0; i < h.N; ++i) {
                if (i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                out[i] = refs[frameIdx] + static_cast<TIn>(r.readFast(h.resBits));
            }
        } else {
            size_t frameIdx = 0;
            size_t frameEnd = h.frameSize;
            switch (h.resWidth) {
                case ResidualWidth::W8: {
                    const auto* res = reinterpret_cast<const int8_t*>(resData);
                    for (size_t i = 0; i < h.N; ++i) {
                        if (i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[i]);
                    }
                    break;
                }
                case ResidualWidth::W16: {
                    const auto* res = reinterpret_cast<const int16_t*>(resData);
                    for (size_t i = 0; i < h.N; ++i) {
                        if (i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[i]);
                    }
                    break;
                }
                case ResidualWidth::W32: {
                    const auto* res = reinterpret_cast<const int32_t*>(resData);
                    for (size_t i = 0; i < h.N; ++i) {
                        if (i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[i]);
                    }
                    break;
                }
                case ResidualWidth::W64: {
                    const auto* res = reinterpret_cast<const int64_t*>(resData);
                    for (size_t i = 0; i < h.N; ++i) {
                        if (i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[i]);
                    }
                    break;
                }
            }
        }
        return out;
    }

    // Decode at --------------------------------------------------------------
    std::optional<TIn> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto h = parseHeader(encoded);
        if (index >= h.N) return std::nullopt;

        const TIn*     refs    = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        const TIn      ref     = refs[index / h.frameSize];
        const uint8_t* resData = encoded.data().data() + h.resOffset;

        int64_t residual = 0;
        if (h.resBits) {
            encodings::core::BitReader r(resData, h.resBytes, encodings::core::BitOrder::LSB);
            r.seekToBit(index * static_cast<size_t>(h.resBits));
            residual = static_cast<int64_t>(r.read(h.resBits));
        } else {
            switch (h.resWidth) {
                case ResidualWidth::W8:  residual = reinterpret_cast<const int8_t* >(resData)[index]; break;
                case ResidualWidth::W16: residual = reinterpret_cast<const int16_t*>(resData)[index]; break;
                case ResidualWidth::W32: residual = reinterpret_cast<const int32_t*>(resData)[index]; break;
                case ResidualWidth::W64: residual = reinterpret_cast<const int64_t*>(resData)[index]; break;
            }
        }
        return ref + static_cast<TIn>(residual);
    }

    // Decode range -----------------------------------------------------------
    std::vector<TIn> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto h = parseHeader(encoded);
        end = std::min(end, h.N);
        if (start >= end) return {};
        const size_t count = end - start;

        const TIn*     refs    = reinterpret_cast<const TIn*>(encoded.data().data() + h.refOffset);
        const uint8_t* resData = encoded.data().data() + h.resOffset;

        std::vector<TIn> out(count);

        if (h.resBits) {
            encodings::core::BitReader r(resData, h.resBytes, encodings::core::BitOrder::LSB);
            r.seekToBit(start * static_cast<size_t>(h.resBits));
            size_t frameIdx = start / h.frameSize;
            size_t frameEnd = (frameIdx + 1) * h.frameSize;
            for (size_t i = 0; i < count; ++i) {
                const size_t idx = start + i;
                if (idx == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                out[i] = refs[frameIdx] + static_cast<TIn>(r.readFast(h.resBits));
            }
        } else {
            size_t frameIdx = start / h.frameSize;
            size_t frameEnd = (frameIdx + 1) * h.frameSize;
            switch (h.resWidth) {
                case ResidualWidth::W8: {
                    const auto* res = reinterpret_cast<const int8_t*>(resData);
                    for (size_t i = 0; i < count; ++i) {
                        if (start + i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[start + i]);
                    }
                    break;
                }
                case ResidualWidth::W16: {
                    const auto* res = reinterpret_cast<const int16_t*>(resData);
                    for (size_t i = 0; i < count; ++i) {
                        if (start + i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[start + i]);
                    }
                    break;
                }
                case ResidualWidth::W32: {
                    const auto* res = reinterpret_cast<const int32_t*>(resData);
                    for (size_t i = 0; i < count; ++i) {
                        if (start + i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[start + i]);
                    }
                    break;
                }
                case ResidualWidth::W64: {
                    const auto* res = reinterpret_cast<const int64_t*>(resData);
                    for (size_t i = 0; i < count; ++i) {
                        if (start + i == frameEnd) { ++frameIdx; frameEnd += h.frameSize; }
                        out[i] = refs[frameIdx] + static_cast<TIn>(res[start + i]);
                    }
                    break;
                }
            }
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

    static void writeU64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, sizeof(uint64_t)); }
    static void writeU32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, sizeof(uint32_t)); }
    static uint64_t readU64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, sizeof(uint64_t)); return v; }
    static uint32_t readU32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, sizeof(uint32_t)); return v; }

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

        std::vector<ResT> residuals(N);

        size_t idx = 0;
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            const TIn ref = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                              data.begin() + static_cast<ptrdiff_t>(hi));
            refs[f] = ref;
            for (size_t i = lo; i < hi; ++i, ++idx) {
                const auto residual = data[i] - ref;
                if (residual < static_cast<TIn>(std::numeric_limits<ResT>::min()) ||
                    residual > static_cast<TIn>(std::numeric_limits<ResT>::max())) {
                    throw std::overflow_error("AdaptiveFOR: residual overflow after plan selection");
                }
                residuals[idx] = static_cast<ResT>(residual);
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

        // Pre-allocate output buffer; write residuals directly — no intermediate vector.
        std::vector<uint8_t> buf;
        buf.reserve((N * static_cast<size_t>(bits) + 7) / 8);
        encodings::core::BitWriter wr(buf, encodings::core::BitOrder::LSB);

        for (size_t f = 0; f < numFrames; ++f) {
            const size_t lo = f * frameSize;
            const size_t hi = std::min(lo + frameSize, N);
            const TIn ref = *std::min_element(data.begin() + static_cast<ptrdiff_t>(lo),
                                              data.begin() + static_cast<ptrdiff_t>(hi));
            refs[f] = ref;
            for (size_t i = lo; i < hi; ++i) {
                wr.write(static_cast<uint32_t>(data[i] - ref), bits);
            }
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
};

// Convenience factory
template <typename TIn>
std::shared_ptr<AdaptiveFOREncoder<TIn>> makeAdaptiveFOR() {
    return std::make_shared<AdaptiveFOREncoder<TIn>>();
}

} // namespace encodings::encoders
