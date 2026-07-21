#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include "core/BitPacker.hpp"

namespace encodings::encoders {

/**
 * @brief Adaptive framed bit-prefix encoder.
 *
 * For each frame, find the common leading prefix among all values (within the
 * logical bit-width of the data), store that prefix once, and bit-pack the
 * suffixes. Chooses the frame size from {8, 16, 32, 64, 128} by estimated size.
 * Falls back to raw bit-packing when no shared prefix exists.
 */
template <typename T>
    requires std::is_integral_v<T>
class AdaptiveFramedBitPrefixEncoder : public Codec<T, uint8_t> {
    using U = std::make_unsigned_t<T>;

    static constexpr size_t kHeaderBytes = 20; // N(8) + frameSize(4) + bitWidth(1) + pad(3) + numFrames(4)
    static constexpr size_t kFrameRecordBytes = 1 + sizeof(U) + 4; // prefixBits + prefixValue + suffixBytes

    struct FrameInfo {
        uint8_t prefixBits{0};
        U prefixValue{0};
        uint32_t suffixBytes{0};

        uint8_t suffixBits(uint8_t bitWidth) const {
            return (bitWidth > prefixBits) ? static_cast<uint8_t>(bitWidth - prefixBits) : 0;
        }
    };

    struct Plan {
        size_t frameSize{0};
        uint8_t bitWidth{64};
        size_t estimatedBytes{std::numeric_limits<size_t>::max()};
        std::vector<FrameInfo> frames;
    };

public:
    AdaptiveFramedBitPrefixEncoder() = default;

    EncodedBuffer<uint8_t> encode(std::span<const T> data) override {
        if (data.empty()) {
            return makeEmpty();
        }

        const auto plan = choosePlan(data);
        const size_t numFrames = plan.frames.size();

        // Build prefix table
        std::vector<uint8_t> out;
        out.reserve(plan.estimatedBytes);

        appendU64(out, static_cast<uint64_t>(data.size()));
        appendU32(out, static_cast<uint32_t>(plan.frameSize));
        out.push_back(plan.bitWidth);
        out.push_back(0); out.push_back(0); out.push_back(0); // pad
        appendU32(out, static_cast<uint32_t>(numFrames));

        // Reserve space for frame records
        const size_t recordBytes = numFrames * kFrameRecordBytes;
        const size_t recordsOffset = out.size();
        out.resize(recordsOffset + recordBytes, 0);

        // Payloads appended after records
        std::vector<uint8_t> payload;
        payload.reserve(plan.estimatedBytes > out.size() ? plan.estimatedBytes - out.size() : 0);

        size_t frameIdx = 0;
        size_t cursor = recordsOffset;
        for (size_t start = 0; start < data.size(); start += plan.frameSize, ++frameIdx) {
            const size_t end = std::min(start + plan.frameSize, data.size());
            const auto& info = plan.frames[frameIdx];

            // Write frame record
            out[cursor++] = info.prefixBits;
            std::memcpy(out.data() + cursor, &info.prefixValue, sizeof(U));
            cursor += sizeof(U);
            std::memcpy(out.data() + cursor, &info.suffixBytes, sizeof(uint32_t));
            cursor += sizeof(uint32_t);

            const uint8_t suffixBits = info.suffixBits(plan.bitWidth);

            if (suffixBits == 0) {
                continue;
            }

            const auto suffixes = packSuffixes(data.subspan(start, end - start), suffixBits);
            if (suffixes.size() != info.suffixBytes) {
                throw std::runtime_error("AdaptiveFramedBitPrefixEncoder: suffix size mismatch during encode");
            }
            payload.insert(payload.end(), suffixes.begin(), suffixes.end());
        }

        out.insert(out.end(), payload.begin(), payload.end());

        encodings::EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = encodings::core::typeToDataType<T>;
        meta.elementCount         = data.size();
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = data.size() * sizeof(T);
        meta.supportsRandomAccess = true;

        return encodings::EncodedData(std::move(out), std::move(meta));
    }

    std::vector<T> decodeAll(const EncodedBuffer<uint8_t>& encoded) override {
        const auto hdr = parseHeader(encoded);
        if (hdr.N == 0) return {};

        const size_t recordBytes = hdr.numFrames * kFrameRecordBytes;
        const size_t recordsOffset = kHeaderBytes;
        const size_t payloadOffset = recordsOffset + recordBytes;
        if (payloadOffset > encoded.data().size()) {
            throw std::runtime_error("AdaptiveFramedBitPrefix: buffer too small for frame records");
        }

        std::vector<T> out(hdr.N);

        size_t payloadCursor = payloadOffset;
        const uint8_t* records = encoded.data().data() + recordsOffset;
        const uint64_t suffixMask = hdr.bitWidth == 64 ? std::numeric_limits<uint64_t>::max() : ((uint64_t{1} << hdr.bitWidth) - 1);

        size_t frameIdx = 0;
        for (size_t start = 0; start < hdr.N; start += hdr.frameSize, ++frameIdx) {
            const size_t end = std::min(start + hdr.frameSize, hdr.N);
            FrameInfo info = readFrame(records + frameIdx * kFrameRecordBytes);
            const uint8_t suffixBits = info.suffixBits(hdr.bitWidth);

            if (info.suffixBytes == 0 && suffixBits == 0) {
                const T value = static_cast<T>(info.prefixValue & static_cast<U>(suffixMask));
                std::fill(out.data() + start, out.data() + end, value);
                continue;
            }

            unpackSuffixesTo(encoded.data().data() + payloadCursor, info.suffixBytes,
                             end - start, suffixBits, suffixMask, info.prefixValue,
                             out.data() + start);
            payloadCursor += info.suffixBytes;
        }

        return out;
    }

    std::optional<T> decodeAt(const EncodedBuffer<uint8_t>& encoded, size_t index) override {
        const auto hdr = parseHeader(encoded);
        if (index >= hdr.N) return std::nullopt;

        const size_t recordBytes = hdr.numFrames * kFrameRecordBytes;
        const size_t recordsOffset = kHeaderBytes;
        const size_t payloadOffset = recordsOffset + recordBytes;
        if (payloadOffset > encoded.data().size()) return std::nullopt;

        const size_t frameIdx = index / hdr.frameSize;
        const size_t inFrameIdx = index % hdr.frameSize;
        const FrameInfo info = readFrame(encoded.data().data() + recordsOffset + frameIdx * kFrameRecordBytes);
        const uint64_t suffixMask = widthMask(hdr.bitWidth);

        // Compute payload cursor by summing prior suffixBytes
        size_t payloadCursor = payloadOffset;
        for (size_t f = 0; f < frameIdx; ++f) {
            FrameInfo prev = readFrame(encoded.data().data() + recordsOffset + f * kFrameRecordBytes);
            payloadCursor += prev.suffixBytes;
        }

        const uint8_t suffixBits = info.suffixBits(hdr.bitWidth);

        if (suffixBits == 0) {
            return static_cast<T>(info.prefixValue & static_cast<U>(suffixMask));
        }

        const uint8_t* payload = encoded.data().data() + payloadCursor;
        const uint64_t suffix = extractSuffix(payload, info.suffixBytes, inFrameIdx, suffixBits);
        const uint64_t value = (static_cast<uint64_t>(info.prefixValue) << suffixBits) | suffix;
        return static_cast<T>(value & suffixMask);
    }

    std::vector<T> decodeRange(const EncodedBuffer<uint8_t>& encoded, size_t start, size_t end) override {
        const auto hdr = parseHeader(encoded);
        if (start >= hdr.N) return {};
        end = std::min(end, hdr.N);
        if (start >= end) return {};

        const size_t recordBytes = hdr.numFrames * kFrameRecordBytes;
        const size_t recordsOffset = kHeaderBytes;
        const size_t payloadOffset = recordsOffset + recordBytes;
        if (payloadOffset > encoded.data().size()) {
            throw std::runtime_error("AdaptiveFramedBitPrefix: buffer too small for frame records");
        }

        const uint64_t suffixMask = widthMask(hdr.bitWidth);

        std::vector<T> out(end - start);
        size_t writePos = 0;

        // Walk payload offset forward to the first frame we need, then advance frame-by-frame
        const size_t firstFrame = start / hdr.frameSize;
        size_t framePayloadBase = payloadOffset;
        for (size_t f = 0; f < firstFrame; ++f) {
            FrameInfo fi = readFrame(encoded.data().data() + recordsOffset + f * kFrameRecordBytes);
            framePayloadBase += fi.suffixBytes;
        }

        for (size_t i = start; i < end; ) {
            const size_t frameIdx = i / hdr.frameSize;
            const size_t frameStart = frameIdx * hdr.frameSize;
            const size_t frameEnd = std::min(frameStart + hdr.frameSize, hdr.N);
            const FrameInfo info = readFrame(encoded.data().data() + recordsOffset + frameIdx * kFrameRecordBytes);
            const uint8_t suffixBits = info.suffixBits(hdr.bitWidth);
            const size_t localStart = i - frameStart;
            const size_t localEnd = std::min(frameEnd, end) - frameStart;
            const size_t localCount = localEnd - localStart;

            if (suffixBits == 0) {
                const T val = static_cast<T>(info.prefixValue & static_cast<U>(suffixMask));
                std::fill(out.data() + writePos, out.data() + writePos + localCount, val);
            } else {
                const uint8_t* payload = encoded.data().data() + framePayloadBase;
                unpackSuffixesRangeTo(payload, info.suffixBytes, localStart, localEnd,
                                      suffixBits, suffixMask, info.prefixValue,
                                      out.data() + writePos);
            }
            writePos += localCount;
            framePayloadBase += info.suffixBytes;
            i = frameStart + localEnd;
        }

        return out;
    }

    // Gather (selective row-range) fast path. decodeAt/decodeRange both
    // accumulate suffixBytes from frame 0 to the target frame on every call
    // (O(frameIdx)). Here we precompute a local O(numFrames) prefix-sum of
    // per-frame payload byte offsets once, giving O(1) lookup for any frame
    // regardless of range order -- cheaper and simpler than a manually
    // advanced cursor, at the same one-time O(numFrames) cost.
    void decodeGatherInto(const EncodedBuffer<uint8_t>& encoded,
                          const RowRangeList& ranges,
                          T* dst, size_t n) override {
        if (ranges.empty()) {
            if (n != 0) throw std::runtime_error("AdaptiveFramedBitPrefixEncoder::decodeGatherInto: decoded size mismatch");
            return;
        }
        const auto hdr = parseHeader(encoded);
        if (hdr.N == 0) {
            if (n != 0) throw std::runtime_error("AdaptiveFramedBitPrefixEncoder::decodeGatherInto: empty codec, n!=0");
            return;
        }

        const size_t recordBytes   = hdr.numFrames * kFrameRecordBytes;
        const size_t recordsOffset = kHeaderBytes;
        const size_t payloadOffset = recordsOffset + recordBytes;
        if (payloadOffset > encoded.data().size())
            throw std::runtime_error("AdaptiveFramedBitPrefix: buffer too small for frame records");

        std::vector<size_t> frameOffsets(hdr.numFrames + 1);
        frameOffsets[0] = payloadOffset;
        for (size_t f = 0; f < hdr.numFrames; ++f) {
            FrameInfo fi = readFrame(encoded.data().data() + recordsOffset + f * kFrameRecordBytes);
            frameOffsets[f + 1] = frameOffsets[f] + fi.suffixBytes;
        }

        const uint64_t suffixMask = widthMask(hdr.bitWidth);
        size_t off = 0;
        for (const auto& r : ranges) {
            const size_t count = r.size();
            if (count == 0) continue;
            const size_t start = r.begin;
            const size_t end   = std::min(r.end, hdr.N);
            if (start >= end) continue;

            for (size_t i = start; i < end; ) {
                const size_t frameIdx   = i / hdr.frameSize;
                const size_t frameStart = frameIdx * hdr.frameSize;
                const size_t frameEnd   = std::min(frameStart + hdr.frameSize, hdr.N);
                const FrameInfo info = readFrame(encoded.data().data() + recordsOffset + frameIdx * kFrameRecordBytes);
                const uint8_t suffixBits = info.suffixBits(hdr.bitWidth);
                const size_t localStart = i - frameStart;
                const size_t localEnd   = std::min(frameEnd, end) - frameStart;
                const size_t localCount = localEnd - localStart;

                if (suffixBits == 0) {
                    const T val = static_cast<T>(info.prefixValue & static_cast<U>(suffixMask));
                    std::fill(dst + off, dst + off + localCount, val);
                } else {
                    const uint8_t* payload = encoded.data().data() + frameOffsets[frameIdx];
                    unpackSuffixesRangeTo(payload, info.suffixBytes, localStart, localEnd,
                                          suffixBits, suffixMask, info.prefixValue,
                                          dst + off);
                }
                off += localCount;
                i = frameStart + localEnd;
            }
        }
        if (off != n) throw std::runtime_error("AdaptiveFramedBitPrefixEncoder::decodeGatherInto: decoded size mismatch");
    }

    EncodingType encodingType() const override { return EncodingType::AdaptiveFramedBitPrefix; }

    std::string name() const override { return "AdaptiveFramedBitPrefix<" + std::string(typeid(T).name()) + ">"; }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        return EncodingProperties{}
            .add(EP::RandomAccess)
            .add(EP::Lossless)
            .add(EP::PreservesOrder)
            .add(EP::FixedSize)
            .add(EP::Composable)
            .add(EP::LowMemoryOverhead)
            .add(EP::FastSkip);
    }

private:
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

    static uint8_t logicalBitWidth(std::span<const T> data) {
        U maxv = 0;
        for (auto v : data) maxv |= static_cast<U>(v);
        const uint8_t totalBits = static_cast<uint8_t>(sizeof(U) * 8);
        const uint8_t w = maxv == 0 ? 1 : static_cast<uint8_t>(std::bit_width(maxv));
        return w == 0 ? 1 : std::min<uint8_t>(w, totalBits);
    }

    static uint64_t widthMask(uint8_t width) {
        if (width == 0) return 0;
        if (width >= 64) return std::numeric_limits<uint64_t>::max();
        return (uint64_t{1} << width) - 1;
    }

    static uint8_t framePrefixBits(std::span<const T> data, size_t start, size_t end, uint8_t width) {
        const uint64_t mask = widthMask(width);
        uint64_t xor_all = 0;
        const uint64_t first = static_cast<uint64_t>(static_cast<U>(data[start]) & static_cast<U>(mask));
        for (size_t i = start + 1; i < end; ++i) {
            xor_all |= ((static_cast<uint64_t>(static_cast<U>(data[i]) & static_cast<U>(mask))) ^ first);
        }
        if (xor_all == 0) return width;
        if (width == 64) return static_cast<uint8_t>(std::countl_zero(xor_all));
        const uint8_t leading = static_cast<uint8_t>(std::countl_zero(xor_all << (64 - width)));
        return std::min<uint8_t>(width, leading);
    }

    Plan choosePlan(std::span<const T> data) const {
        static constexpr size_t kCandidates[] = {8, 16, 32, 64, 128};
        const uint8_t bitWidth = logicalBitWidth(data);
        const uint64_t mask = widthMask(bitWidth);

        Plan best;

        for (size_t frame : kCandidates) {
            Plan plan;
            plan.frameSize = frame;
            plan.bitWidth = bitWidth;

            const size_t numFrames = (data.size() + frame - 1) / frame;
            plan.frames.reserve(numFrames);
            size_t payloadBytes = 0;

            for (size_t start = 0; start < data.size(); start += frame) {
                const size_t end = std::min(start + frame, data.size());
                const uint8_t prefixBits = framePrefixBits(data, start, end, bitWidth);
                const uint8_t suffixBits = bitWidth - prefixBits;
                FrameInfo info;
                info.prefixBits = prefixBits;
                const uint64_t first = static_cast<uint64_t>(static_cast<U>(data[start]) & static_cast<U>(mask));
                info.prefixValue = suffixBits == 0 ? static_cast<U>(first) : static_cast<U>(first >> suffixBits);
                info.suffixBytes = suffixBits == 0 ? 0 : static_cast<uint32_t>((static_cast<size_t>(suffixBits) * (end - start) + 7) / 8);
                plan.frames.push_back(info);
                payloadBytes += info.suffixBytes;
            }

            const size_t total = kHeaderBytes + numFrames * kFrameRecordBytes + payloadBytes;
            plan.estimatedBytes = total;

            if (total < best.estimatedBytes) {
                best = std::move(plan);
            }
        }

        return best;
    }

    struct ParsedHeader {
        size_t N{0};
        size_t frameSize{0};
        uint8_t bitWidth{64};
        size_t numFrames{0};
    };

    static ParsedHeader parseHeader(const EncodedBuffer<uint8_t>& encoded) {
        if (encoded.data().size() < kHeaderBytes) {
            throw std::runtime_error("AdaptiveFramedBitPrefix: buffer too small for header");
        }
        const uint8_t* p = encoded.data().data();
        ParsedHeader h;
        std::memcpy(&h.N, p, sizeof(uint64_t)); p += 8;
        uint32_t fs; std::memcpy(&fs, p, sizeof(uint32_t)); p += 4;
        h.frameSize = fs;
        h.bitWidth = *p++;
        p += 3; // pad
        uint32_t nf; std::memcpy(&nf, p, sizeof(uint32_t));
        h.numFrames = nf;
        return h;
    }

    static FrameInfo readFrame(const uint8_t* p) {
        FrameInfo info;
        info.prefixBits = *p++;
        std::memcpy(&info.prefixValue, p, sizeof(U));
        p += sizeof(U);
        std::memcpy(&info.suffixBytes, p, sizeof(uint32_t));
        return info;
    }

    static std::vector<uint8_t> packSuffixes(std::span<const T> vals, uint8_t suffixBits) {
        if (suffixBits == 0) return {};
        const size_t bitsTotal = static_cast<size_t>(suffixBits) * vals.size();
        std::vector<uint8_t> out;
        out.reserve((bitsTotal + 7) / 8);
        const uint64_t suffixMask = suffixBits == 64 ? std::numeric_limits<uint64_t>::max() : ((uint64_t{1} << suffixBits) - 1);

        if (suffixBits <= 16) {
            encodings::core::BitWriter wr(out, encodings::core::BitOrder::LSB);
            for (auto v : vals) {
                const uint64_t s = static_cast<uint64_t>(static_cast<U>(v)) & suffixMask;
                wr.write(static_cast<uint32_t>(s), suffixBits);
            }
            wr.flush();
            if (out.size() < (bitsTotal + 7) / 8) {
                out.resize((bitsTotal + 7) / 8, 0);
            }
            return out;
        }

        out.assign((bitsTotal + 7) / 8, 0);
        size_t bitPos = 0;
        for (auto v : vals) {
            uint64_t s = static_cast<uint64_t>(static_cast<U>(v)) & suffixMask;
            size_t remaining = suffixBits;
            while (remaining > 0) {
                const size_t byteIdx = bitPos >> 3;
                const uint8_t offset = static_cast<uint8_t>(bitPos & 7);
                const uint8_t avail = 8 - offset;
                const uint8_t take = static_cast<uint8_t>(std::min<size_t>(remaining, avail));
                const uint8_t bits = static_cast<uint8_t>(s & ((uint64_t{1} << take) - 1));
                out[byteIdx] |= static_cast<uint8_t>(bits << offset);
                s >>= take;
                bitPos += take;
                remaining -= take;
            }
        }
        return out;
    }

    // Writes decoded values for elements [0, count) directly into output[0..count-1].
    static void unpackSuffixesTo(const uint8_t* payload, size_t payloadBytes,
                                 size_t count, uint8_t suffixBits,
                                 uint64_t mask, uint64_t prefix,
                                 T* output) {
        if (suffixBits == 0) {
            std::fill(output, output + count, static_cast<T>(prefix & mask));
            return;
        }
        const uint64_t suffixMask = suffixBits == 64 ? std::numeric_limits<uint64_t>::max()
                                                     : ((uint64_t{1} << suffixBits) - 1);

        if (suffixBits <= 16) {
            const uint64_t shiftedPrefix = (prefix << suffixBits) & mask;
            encodings::core::BitReader r(payload, payloadBytes, encodings::core::BitOrder::LSB);
            for (size_t i = 0; i < count; ++i) {
                const uint64_t s = r.readFast(suffixBits) & suffixMask;
                output[i] = static_cast<T>(shiftedPrefix | s);
            }
            return;
        }

        if (suffixBits == 64) {
            for (size_t i = 0; i < count; ++i) {
                uint64_t s;
                std::memcpy(&s, payload + i * sizeof(uint64_t), sizeof(uint64_t));
                output[i] = static_cast<T>(s & mask);
            }
            return;
        }

        const uint64_t shiftedPrefix = (prefix << suffixBits) & mask;
        size_t bytePos = 0;

        if (suffixBits <= 56) {
            // 64-bit accumulator: acc >>= suffixBits is a single SHR instruction
            uint64_t acc = 0;
            uint32_t accBits = 0;
            while (accBits <= 56 && bytePos < payloadBytes) {
                acc |= static_cast<uint64_t>(payload[bytePos++]) << accBits;
                accBits += 8;
            }
            for (size_t i = 0; i < count; ++i) {
                while (accBits < static_cast<uint32_t>(suffixBits) && accBits <= 56 && bytePos < payloadBytes) {
                    acc |= static_cast<uint64_t>(payload[bytePos++]) << accBits;
                    accBits += 8;
                }
                const uint64_t s = acc & suffixMask;
                acc >>= suffixBits;
                accBits -= static_cast<uint32_t>(suffixBits);
                output[i] = static_cast<T>(shiftedPrefix | s);
            }
        } else {
            // 128-bit fallback for suffixBits in [57, 63]
            __uint128_t acc = 0;
            uint32_t accBits = 0;
            while (accBits <= 56 && bytePos < payloadBytes) {
                acc |= static_cast<__uint128_t>(payload[bytePos++]) << accBits;
                accBits += 8;
            }
            for (size_t i = 0; i < count; ++i) {
                while (accBits < static_cast<uint32_t>(suffixBits) && bytePos < payloadBytes) {
                    acc |= static_cast<__uint128_t>(payload[bytePos++]) << accBits;
                    accBits += 8;
                }
                const uint64_t s = static_cast<uint64_t>(acc) & suffixMask;
                acc >>= suffixBits;
                accBits -= static_cast<uint32_t>(suffixBits);
                output[i] = static_cast<T>(shiftedPrefix | s);
            }
        }
    }

    // Writes decoded values for frame elements [start, end) directly into output[0..end-start-1].
    static void unpackSuffixesRangeTo(const uint8_t* payload, size_t payloadBytes,
                                      size_t start, size_t end,
                                      uint8_t suffixBits, uint64_t mask, uint64_t prefix,
                                      T* output) {
        const size_t count = end - start;
        if (suffixBits == 0) {
            std::fill(output, output + count, static_cast<T>(prefix & mask));
            return;
        }
        const uint64_t suffixMask = suffixBits == 64 ? std::numeric_limits<uint64_t>::max()
                                                     : ((uint64_t{1} << suffixBits) - 1);

        if (suffixBits <= 16) {
            const uint64_t shiftedPrefix = (prefix << suffixBits) & mask;
            encodings::core::BitReader r(payload, payloadBytes, encodings::core::BitOrder::LSB);
            r.seekToBit(static_cast<size_t>(start) * suffixBits);
            for (size_t j = 0; j < count; ++j) {
                const uint64_t s = r.readFast(suffixBits) & suffixMask;
                output[j] = static_cast<T>(shiftedPrefix | s);
            }
            return;
        }

        if (suffixBits == 64) {
            for (size_t j = 0; j < count; ++j) {
                uint64_t s;
                std::memcpy(&s, payload + (start + j) * sizeof(uint64_t), sizeof(uint64_t));
                output[j] = static_cast<T>(s & mask);
            }
            return;
        }

        const uint64_t shiftedPrefix = (prefix << suffixBits) & mask;
        size_t bytePos = (static_cast<size_t>(start) * suffixBits) >> 3;
        const uint32_t bitOff = static_cast<uint32_t>((static_cast<size_t>(start) * suffixBits) & 7);

        if (suffixBits <= 56) {
            // 64-bit accumulator: acc >>= suffixBits is a single SHR instruction
            uint64_t acc = 0;
            uint32_t accBits = 0;
            while (accBits <= 56 && bytePos < payloadBytes) {
                acc |= static_cast<uint64_t>(payload[bytePos++]) << accBits;
                accBits += 8;
            }
            acc >>= bitOff;
            accBits -= bitOff;
            for (size_t j = 0; j < count; ++j) {
                while (accBits < static_cast<uint32_t>(suffixBits) && accBits <= 56 && bytePos < payloadBytes) {
                    acc |= static_cast<uint64_t>(payload[bytePos++]) << accBits;
                    accBits += 8;
                }
                const uint64_t s = acc & suffixMask;
                acc >>= suffixBits;
                accBits -= static_cast<uint32_t>(suffixBits);
                output[j] = static_cast<T>(shiftedPrefix | s);
            }
        } else {
            // 128-bit fallback for suffixBits in [57, 63]
            __uint128_t acc = 0;
            uint32_t accBits = 0;
            while (accBits <= 56 && bytePos < payloadBytes) {
                acc |= static_cast<__uint128_t>(payload[bytePos++]) << accBits;
                accBits += 8;
            }
            acc >>= bitOff;
            accBits -= bitOff;
            for (size_t j = 0; j < count; ++j) {
                while (accBits < static_cast<uint32_t>(suffixBits) && bytePos < payloadBytes) {
                    acc |= static_cast<__uint128_t>(payload[bytePos++]) << accBits;
                    accBits += 8;
                }
                const uint64_t s = static_cast<uint64_t>(acc) & suffixMask;
                acc >>= suffixBits;
                accBits -= static_cast<uint32_t>(suffixBits);
                output[j] = static_cast<T>(shiftedPrefix | s);
            }
        }
    }

    static uint64_t extractSuffix(const uint8_t* payload, size_t payloadBytes, size_t idx, uint8_t suffixBits) {
        if (suffixBits == 0) return 0;
        if (suffixBits <= 16) {
            encodings::core::BitReader r(payload, payloadBytes, encodings::core::BitOrder::LSB);
            r.seekToBit(static_cast<size_t>(suffixBits) * idx);
            return r.read(suffixBits);
        }

        const size_t bitPos = static_cast<size_t>(suffixBits) * idx;
        const uint64_t suffixMask = suffixBits == 64 ? std::numeric_limits<uint64_t>::max() : ((uint64_t{1} << suffixBits) - 1);
        size_t pos = bitPos;
        uint64_t s = 0;
        size_t bitsRead = 0;
        while (bitsRead < suffixBits) {
            const size_t byteIdx = pos >> 3;
            if (byteIdx >= payloadBytes) throw std::runtime_error("AdaptiveFramedBitPrefix: payload underrun");
            const uint8_t offset = static_cast<uint8_t>(pos & 7);
            const uint8_t avail = 8 - offset;
            const uint8_t take = static_cast<uint8_t>(std::min<size_t>(suffixBits - bitsRead, avail));
            const uint8_t bits = static_cast<uint8_t>((payload[byteIdx] >> offset) & ((1u << take) - 1));
            s |= static_cast<uint64_t>(bits) << bitsRead;
            pos += take;
            bitsRead += take;
        }
        return s & suffixMask;
    }

    static EncodedBuffer<uint8_t> makeEmpty() {
        std::vector<uint8_t> out(kHeaderBytes, 0);
        encodings::EncodingMetadata meta;
        meta.encodingName         = "AdaptiveFramedBitPrefix(empty)";
        meta.elementCount         = 0;
        meta.compressedSize       = out.size();
        meta.uncompressedSize     = 0;
        meta.supportsRandomAccess = true;
        return encodings::EncodedData(std::move(out), std::move(meta));
    }
};

template <typename T>
    requires std::is_integral_v<T>
inline std::shared_ptr<AdaptiveFramedBitPrefixEncoder<T>> makeAdaptiveFramedBitPrefixEncoder() {
    return std::make_shared<AdaptiveFramedBitPrefixEncoder<T>>();
}

} // namespace encodings::encoders
