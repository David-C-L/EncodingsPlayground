#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "generators/samplers/StreamSampling.hpp"
using encodings::encoders::selectors::costs::RawCostModel;
using encodings::encoders::selectors::costs::RawBitPackedCostModel;
using encodings::encoders::selectors::costs::AdaptiveFORCostModel;
using encodings::encoders::selectors::costs::AdaptiveFramedBitPrefixCostModel;
using encodings::encoders::selectors::costs::FORCostModel;
using encodings::encoders::selectors::costs::DictionaryCostModel;
using encodings::encoders::selectors::costs::RLECostModel;

namespace encodings::encoders {

// Generic sub-integer split encoder (N-way split where sum(bits)=64)
class SubIntSplitEncoder64 final : public Codec<int64_t, uint8_t> {
public:
    explicit SubIntSplitEncoder64(SubIntSplitConfig64 cfg) : cfg_(std::move(cfg)) {
        cfg_.validate();
    }

    EncodedBuffer<uint8_t> encode(std::span<const int64_t> data) override {
        const size_t N = data.size();
        if (N == 0) {
            return makeEmpty();
        }

        const size_t splits = cfg_.splitCount();
        std::vector<std::vector<uint64_t>> sections(splits);
        for (auto& sec : sections) sec.resize(N);

        // Split values into sections according to order
        if (cfg_.order == BitSplitOrder::LSB_TO_MSB) {
            for (size_t i = 0; i < N; ++i) {
                uint64_t v = static_cast<uint64_t>(data[i]);
                uint64_t shifted = v;
                for (size_t s = 0; s < splits; ++s) {
                    const uint8_t bits = cfg_.bits[s];
                    const uint64_t mask = bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
                    sections[s][i] = shifted & mask;
                    if (bits < 64) shifted >>= bits;
                }
            }
        } else { // MSB_TO_LSB
            // Precompute starting shifts from MSB side
            std::vector<uint8_t> shiftStart(splits, 0);
            uint8_t remaining = 64;
            for (size_t s = 0; s < splits; ++s) {
                const uint8_t bits = cfg_.bits[s];
                if (bits > remaining) {
                    throw std::invalid_argument("SubIntSplitEncoder64: bits exceed width when splitting");
                }
                remaining = static_cast<uint8_t>(remaining - bits);
                shiftStart[s] = remaining; // shift to align MSB block
            }
            for (size_t i = 0; i < N; ++i) {
                uint64_t v = static_cast<uint64_t>(data[i]);
                for (size_t s = 0; s < splits; ++s) {
                    const uint8_t bits = cfg_.bits[s];
                    const uint64_t mask = bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
                    sections[s][i] = (v >> shiftStart[s]) & mask;
                }
            }
        }

        // Encode each section
        std::vector<EncodedBuffer<uint8_t>> encodedSections;
        encodedSections.reserve(splits);
        std::vector<uint64_t> sectionSizes;
        sectionSizes.reserve(splits);

        for (size_t s = 0; s < splits; ++s) {
            auto enc = cfg_.codecs[s]->encode(std::span<const uint64_t>(sections[s].data(), sections[s].size()));
            sectionSizes.push_back(enc.data().size());
            encodedSections.push_back(std::move(enc));
        }

        // Build header: [N:8][splitCount:1][order:1][bits_i:1][bytes_i:8]*splitCount
        const size_t headerSize = 8 + 1 + 1 + splits * (1 + 8);
        std::vector<uint8_t> out;
        out.resize(headerSize);

        uint8_t* p = out.data();
        writeU64(p, static_cast<uint64_t>(N)); p += 8;
        *p++ = static_cast<uint8_t>(splits);
        *p++ = static_cast<uint8_t>(cfg_.order);
        for (size_t s = 0; s < splits; ++s) {
            *p++ = cfg_.bits[s];
            writeU64(p, sectionSizes[s]);
            p += 8;
        }

        // Append payloads
        for (const auto& enc : encodedSections) {
            out.insert(out.end(), enc.data().begin(), enc.data().end());
        }

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<int64_t>;
        meta.elementCount         = N;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = N * sizeof(int64_t);
        meta.supportsRandomAccess = allSectionsRandomAccess();

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    std::vector<int64_t> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const auto& h = getCachedHeader(encoded);
        if (h.N == 0) return {};

        const size_t splits = h.bits.size();
        std::vector<std::vector<uint64_t>> sections(splits);
        for (size_t s = 0; s < splits; ++s) {
            sections[s] = cfg_.codecs[s]->decodeAll(h.views[s]);
            if (sections[s].size() != h.N) {
                throw std::runtime_error("SubIntSplitEncoder64::decodeAll: section size mismatch");
            }
        }

        std::vector<int64_t> out(h.N);
        for (size_t i = 0; i < h.N; ++i) {
            out[i] = static_cast<int64_t>(combine(sections, h.bits, h.order, i));
        }
        return out;
    }

    std::optional<int64_t> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto& h = getCachedHeader(encoded);
        if (index >= h.N) return std::nullopt;

        const size_t splits = h.bits.size();
        std::vector<uint64_t> parts(splits);
        for (size_t s = 0; s < splits; ++s) {
            parts[s] = decodeOneSection(*cfg_.codecs[s], h.views[s], index);
        }
        return static_cast<int64_t>(combine(parts, h.bits, h.order));
    }

    std::vector<int64_t> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto& h = getCachedHeader(encoded);
        if (start >= h.N) return {};
        end = std::min(end, h.N);
        if (start >= end) return {};
        const size_t count = end - start;

        const size_t splits = h.bits.size();
        std::vector<std::vector<uint64_t>> sections(splits);
        for (size_t s = 0; s < splits; ++s) {
            sections[s] = decodeSectionRange(*cfg_.codecs[s], h.views[s], start, end);
            if (sections[s].size() != count) {
                throw std::runtime_error("SubIntSplitEncoder64::decodeRange: section size mismatch");
            }
        }

        std::vector<int64_t> out;
        out.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::vector<uint64_t> parts(splits);
            for (size_t s = 0; s < splits; ++s) parts[s] = sections[s][i];
            out.push_back(static_cast<int64_t>(combine(parts, h.bits, h.order)));
        }
        return out;
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
        if (allSectionsRandomAccess()) props.add(EP::RandomAccess);
        return props;
    }

private:
    struct ParsedHeader {
        size_t N{};
        BitSplitOrder order{BitSplitOrder::LSB_TO_MSB};
        std::vector<uint8_t> bits;
        std::vector<uint64_t> bytes;
        std::vector<EncodedBuffer<uint8_t>> views;
    };

    static void writeU64(uint8_t* dst, uint64_t v) { std::memcpy(dst, &v, sizeof(uint64_t)); }
    static uint64_t readU64(const uint8_t* src) { uint64_t v; std::memcpy(&v, src, sizeof(uint64_t)); return v; }

    size_t headerSizeBytes(size_t splits) const { return 8 + 1 + 1 + splits * (1 + 8); }

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
        const auto& buf = encoded.data();
        if (buf.size() < 10) {
            throw std::runtime_error("SubIntSplitEncoder64: buffer too small for header");
        }
        const uint8_t* p = buf.data();
        ParsedHeader h;
        h.N = static_cast<size_t>(readU64(p)); p += 8;
        const uint8_t splits = *p++;
        h.order = static_cast<BitSplitOrder>(*p++);
        if (splits == 0 || splits > 64) {
            throw std::runtime_error("SubIntSplitEncoder64: invalid split count in header");
        }
        if (cfg_.splitCount() != splits) {
            throw std::runtime_error("SubIntSplitEncoder64: header split count does not match config");
        }
        if (cfg_.order != h.order) {
            throw std::runtime_error("SubIntSplitEncoder64: header order does not match config");
        }
        const size_t expectedHeader = headerSizeBytes(splits);
        if (buf.size() < expectedHeader) {
            throw std::runtime_error("SubIntSplitEncoder64: buffer too small for split metadata");
        }
        h.bits.resize(splits);
        h.bytes.resize(splits);
        for (size_t s = 0; s < splits; ++s) {
            h.bits[s] = *p++;
            h.bytes[s] = readU64(p); p += 8;
        }
        if (h.bits != cfg_.bits) {
            throw std::runtime_error("SubIntSplitEncoder64: header bits do not match config");
        }
        // Validate bit sum
        uint16_t totalBits = 0;
        for (uint8_t b : h.bits) {
            if (b == 0 || b > 64) throw std::runtime_error("SubIntSplitEncoder64: invalid bit width in header");
            totalBits = static_cast<uint16_t>(totalBits + b);
        }
        if (totalBits != 64) {
            throw std::runtime_error("SubIntSplitEncoder64: header bits do not sum to 64");
        }
        // Build section views
        h.views.reserve(splits);
        size_t offset = expectedHeader;
        for (size_t s = 0; s < splits; ++s) {
            const uint64_t len = h.bytes[s];
            if (offset + len > buf.size()) {
                throw std::runtime_error("SubIntSplitEncoder64: payload sizes exceed buffer");
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

    static uint64_t decodeOneSection(ISectionCodec64& c, const EncodedBuffer<uint8_t>& view, size_t idx) {
        if (c.properties().has(EncodingProperty::RandomAccess)) {
            auto v = c.decodeAt(view, idx);
            if (!v) throw std::runtime_error("SubIntSplitEncoder64::decodeAt: subcodec returned null");
            return *v;
        }
        auto all = c.decodeAll(view);
        if (idx >= all.size()) throw std::runtime_error("SubIntSplitEncoder64::decodeAt: index out of range");
        return all[idx];
    }

    static std::vector<uint64_t> decodeSectionRange(ISectionCodec64& c, const EncodedBuffer<uint8_t>& view, size_t start, size_t end) {
        if (c.properties().has(EncodingProperty::RandomAccess)) {
            return c.decodeRange(view, start, end);
        }
        auto all = c.decodeAll(view);
        if (start > all.size()) return {};
        end = std::min(end, all.size());
        return std::vector<uint64_t>(all.begin() + static_cast<ptrdiff_t>(start), all.begin() + static_cast<ptrdiff_t>(end));
    }

    static uint64_t combine(const std::vector<std::vector<uint64_t>>& sections,
                            const std::vector<uint8_t>& bits,
                            BitSplitOrder order,
                            size_t idx) {
        std::vector<uint64_t> parts(sections.size());
        for (size_t s = 0; s < sections.size(); ++s) parts[s] = sections[s][idx];
        return combine(parts, bits, order);
    }

    static uint64_t combine(const std::vector<uint64_t>& parts, const std::vector<uint8_t>& bits, BitSplitOrder order) {
        const size_t splits = parts.size();
        if (splits != bits.size()) {
            throw std::invalid_argument("SubIntSplitEncoder64::combine: size mismatch");
        }
        uint64_t v = 0;
        if (splits == 0) return v;
        uint16_t sum = 0;
        for (uint8_t b : bits) sum = static_cast<uint16_t>(sum + b);
        if (sum != 64) throw std::invalid_argument("SubIntSplitEncoder64::combine: bits do not sum to 64");

        if (order == BitSplitOrder::LSB_TO_MSB) {
            uint8_t shift = 0;
            for (size_t s = 0; s < splits; ++s) {
                v |= (parts[s] << shift);
                shift = static_cast<uint8_t>(shift + bits[s]);
            }
        } else { // MSB_TO_LSB
            for (size_t s = 0; s < splits; ++s) {
                v <<= bits[s];
                v |= parts[s];
            }
        }
        return v;
    }

    EncodedBuffer<uint8_t> makeEmpty() const {
        const size_t splits = cfg_.splitCount();
        if (splits == 0) {
            throw std::invalid_argument("SubIntSplitEncoder64: cannot encode empty config with zero splits");
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
        meta.dataType             = encodings::core::typeToDataType<int64_t>;
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = allSectionsRandomAccess();
        return encodings::EncodedData(std::move(out), std::move(meta));
    }

private:
    SubIntSplitConfig64 cfg_;
    mutable const uint8_t* cachedOuterPtr_{nullptr};
    mutable size_t cachedOuterSize_{0};
    mutable ParsedHeader cachedHeader_{};
};

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoder(SubIntSplitConfig64 cfg) {
    return std::make_shared<SubIntSplitEncoder64>(std::move(cfg));
}

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments) {
    auto cfg = SubIntSplitConfig64::fromSegments(segments);
    return makeSubIntSplitEncoder(std::move(cfg));
}

inline std::shared_ptr<SubIntSplitEncoder64> makeSubIntSplitEncoderFromSegments(
    const std::vector<selectors::SegmentPlan>& segments,
    BitSplitOrder orderHint) {
    auto cfg = SubIntSplitConfig64::fromSegments(segments, orderHint);
    return makeSubIntSplitEncoder(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Auto-selecting wrapper (lazy instantiation via sampling + selector)
// ---------------------------------------------------------------------------

class SubIntSplitAutoEncoder64 final : public Codec<int64_t, uint8_t> {
public:
    struct Config {
        selectors::IDSubStreamEncodingSelector::Config selectorConfig{};
        generators::samplers::StreamSampler<int64_t>::Config samplerConfig{};
        std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>> costModels;
        std::optional<BitSplitOrder> orderHint{}; // allow forcing MSB/LSB ordering
        bool debugLogging{true};
        bool enableSelectionTiming{false};
    };

    explicit SubIntSplitAutoEncoder64(Config cfg)
        : selector_(cfg.selectorConfig), samplerCfg_(cfg.samplerConfig), orderHint_(cfg.orderHint), debugLogging_(cfg.debugLogging),
          selectionTimingEnabled_(cfg.enableSelectionTiming) {
        if (cfg.costModels.empty()) {
            throw std::invalid_argument("SubIntSplitAutoEncoder64: costModels must not be empty");
        }
        costModels_ = std::move(cfg.costModels);
    }

    EncodedBuffer<uint8_t> encode(std::span<const int64_t> data) override {
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

    std::vector<int64_t> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        ensureDecoderInitialized();
        return impl_->decodeAll(encoded);
    }

    std::optional<int64_t> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        ensureDecoderInitialized();
        return impl_->decodeAt(encoded, index);
    }

    std::vector<int64_t> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        ensureDecoderInitialized();
        return impl_->decodeRange(encoded, start, end);
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

private:
    void ensureEncoder(std::span<const int64_t> data) {
        if (impl_) return;
        if (data.empty()) {
            throw std::invalid_argument("SubIntSplitAutoEncoder64: cannot auto-select on empty input");
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

        auto sample = generators::samplers::StreamSampler<int64_t>::sample(data, effectiveCfg);
        if (sample.empty()) {
            // Fallback to a minimal sample (first element)
            sample.push_back(data.front());
        }

        // If MSB order is requested, mirror bits in the sample so the selector's LSB-oriented DP
        // sees the highest bits first. Later we map segment positions back to the original domain.
        std::vector<int64_t> transformedSample;
        if (orderHint_.has_value() && *orderHint_ == BitSplitOrder::MSB_TO_LSB) {
            transformedSample.reserve(sample.size());
            for (auto v : sample) {
                transformedSample.push_back(static_cast<int64_t>(mirrorBits64(static_cast<uint64_t>(v))));
            }
        }

        // Selector needs non-const cost models vector reference; ensure they exist
        const auto& selectorInput = (orderHint_.has_value() && *orderHint_ == BitSplitOrder::MSB_TO_LSB)
                                        ? transformedSample
                                        : sample;
        lastSelection_ = selector_.select(selectorInput, costModels_, data.size());
        if (lastSelection_.segments.empty()) {
            throw std::runtime_error("SubIntSplitAutoEncoder64: selector returned no segments");
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

        const auto cfg = orderHint_.has_value()
            ? SubIntSplitConfig64::fromSegments(segments, *orderHint_)
            : SubIntSplitConfig64::fromSegments(segments);
        // Reflect remapped segments in the cached selection for accurate logging
        lastSelection_.segments = segments;
        impl_ = makeSubIntSplitEncoder(std::move(cfg));
    }

    void ensureDecoderInitialized() const {
        if (!impl_) {
            throw std::runtime_error("SubIntSplitAutoEncoder64: encoder not initialized; call encode first");
        }
    }

private:
    selectors::IDSubStreamEncodingSelector selector_{};
    generators::samplers::StreamSampler<int64_t>::Config samplerCfg_{};
    std::optional<BitSplitOrder> orderHint_{};
    std::vector<std::unique_ptr<selectors::costs::EncodingCostModel>> costModels_;
    std::shared_ptr<SubIntSplitEncoder64> impl_{};
    bool debugLogging_{true};
    bool selectionTimingEnabled_{false};
    std::optional<std::chrono::nanoseconds> lastSelectionTimeNs_{};
    selectors::IDSubStreamEncodingSelector::Result lastSelection_{};

    static uint64_t mirrorBits64(uint64_t x) {
        // simple bit-reversal; sample sizes are small
        uint64_t y = 0;
        for (int i = 0; i < 64; ++i) {
            y <<= 1;
            y |= (x & 1);
            x >>= 1;
        }
        return y;
    }

    static std::vector<selectors::SegmentPlan> remapMirroredSegmentsToOriginal(const std::vector<selectors::SegmentPlan>& segments) {
        std::vector<selectors::SegmentPlan> out;
        out.reserve(segments.size());
        for (const auto& seg : segments) {
            selectors::SegmentPlan mapped = seg;
            mapped.bitStart = 63 - seg.bitEnd;
            mapped.bitEnd   = 63 - seg.bitStart;
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

inline std::shared_ptr<SubIntSplitAutoEncoder64> makeAutoSubIntSplitEncoder(SubIntSplitAutoEncoder64::Config cfg) {
    return std::make_shared<SubIntSplitAutoEncoder64>(std::move(cfg));
}

inline SubIntSplitAutoEncoder64::Config makeDefaultAutoSubIntSplitConfig(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                         bool enableSelectionTiming = false) {
    SubIntSplitAutoEncoder64::Config cfg;
    cfg.selectorConfig = selectors::IDSubStreamEncodingSelector::Config{}; // defaults
    cfg.selectorConfig.verboseLevel = 0; // leave quiet by default; enable when debugging
    cfg.selectorConfig.minSegmentWidth = 1; // avoid tiny slices that inflate headers/rounding
    cfg.selectorConfig.splitPenalty = 10.0; // discourage excessive splitting (heavier for MSB)
    cfg.selectorConfig.enableMergePhase = false; // merge adjacent in MSB mode to reduce fragmentation
    cfg.samplerConfig.maxSamples = 10000;   // cap total sampled points
    cfg.samplerConfig.stride = 0;           // auto-compute stride to span the domain
    cfg.samplerConfig.maxPercentage = 0.01; // or maxSamples, whichever is smaller
    cfg.debugLogging = false;                // enable instrumentation by default for now
    cfg.enableSelectionTiming = enableSelectionTiming;
    cfg.orderHint = order;

    cfg.costModels.emplace_back(std::make_unique<RawCostModel>());
    cfg.costModels.emplace_back(std::make_unique<RawBitPackedCostModel>());
    // cfg.costModels.emplace_back(std::make_unique<FORCostModel>());
    cfg.costModels.emplace_back(std::make_unique<AdaptiveFORCostModel>());
    cfg.costModels.emplace_back(std::make_unique<AdaptiveFramedBitPrefixCostModel>());
    cfg.costModels.emplace_back(std::make_unique<DictionaryCostModel>());
    // cfg.costModels.emplace_back(std::make_unique<RLECostModel>());
    return cfg;
}

inline std::shared_ptr<SubIntSplitAutoEncoder64> makeDefaultAutoSubIntSplitEncoder(BitSplitOrder order = BitSplitOrder::LSB_TO_MSB,
                                                                                   bool exhaustiveSearch = false,
                                                                                   bool enablePrune = true,
                                                                                   bool enableSelectionTiming = false) {
    auto cfg = makeDefaultAutoSubIntSplitConfig(order, enableSelectionTiming);
    cfg.selectorConfig.orderHint = order;
    cfg.selectorConfig.useExhaustiveSearch = exhaustiveSearch;
    cfg.selectorConfig.enablePrune = enablePrune;
    // cfg.selectorConfig.costGridCsvPath = "../Source/encoders/auto_subintsplit_cost_grid.csv"; // for debugging/analysis; selector will log evaluated candidates and their costs
    return makeAutoSubIntSplitEncoder(std::move(cfg));
}

} // namespace encodings::encoders
