#pragma once

#include <span>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_set>

#include "encodings/Encoder.hpp"
#include "encodings/EncodedData.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"
#include "core/DataType.hpp"

#if defined(__has_include)
#if __has_include(<openzl/openzl.h>)
#include <openzl/openzl.h>
#include <openzl/zl_reflection.h>
#include <openzl/codecs/zl_generic.h>
#define HAVE_OPENZL
#endif
#endif

namespace encodings::encoders {

/**
 * @brief Thin wrapper codec around OpenZL generic compressor.
 *
 * Notes:
 * - Requires OpenZL headers/libs (HAVE_OPENZL). Without them, encode will throw.
 * - Decoding is currently not implemented (OpenZL decompress path not wired); decode* throw.
 */
template <typename T, size_t BlockSize = 0>
    requires (std::is_trivially_copyable_v<T>)
class OpenZLCodec : public Codec<T> {
public:
    EncodedData encode(std::span<const T> data) override {
#ifndef HAVE_OPENZL
        throw std::runtime_error("OpenZLCodec: OpenZL not available (HAVE_OPENZL not defined)");
#else
        if (data.empty()) {
            EncodedData out;
            out.metadata().encodingName         = name();
            out.metadata().dataType             = this->dataType();
            out.metadata().elementCount         = 0;
            out.metadata().compressedSize       = 0;
            out.metadata().uncompressedSize     = 0;
            out.metadata().supportsRandomAccess = false;
            return out;
        }

        // Blocked encoding (BlockSize==0 means single block for compatibility)
        std::vector<uint8_t> buffer;
        buffer.reserve(data.size() * sizeof(T) / 2); // heuristic

        const size_t totalElems = data.size();
        const size_t effectiveBlock = (BlockSize == 0) ? totalElems : BlockSize;
        const size_t blockCount = (totalElems + effectiveBlock - 1) / effectiveBlock;

        if constexpr (BlockSize != 0) {
            appendUint64(buffer, static_cast<uint64_t>(blockCount));
        }

        size_t offset = 0;
        for (size_t b = 0; b < blockCount; ++b) {
            const size_t elems = std::min(effectiveBlock, totalElems - offset);
            const auto blockSpan = data.subspan(offset, elems);
            std::vector<uint8_t> compressed = compressBlock(blockSpan);

            if constexpr (BlockSize != 0) {
                appendUint64(buffer, static_cast<uint64_t>(elems));
                appendUint64(buffer, static_cast<uint64_t>(compressed.size()));
            }

            buffer.insert(buffer.end(), compressed.begin(), compressed.end());
            offset += elems;
        }

        EncodingMetadata meta;
        meta.encodingName         = name();
        meta.dataType             = this->dataType();
        meta.elementCount         = totalElems;
        meta.compressedSize       = buffer.size();
        meta.uncompressedSize     = totalElems * sizeof(T);
        // Mark RA true for now; decodeAt/Range fall back to decodeAll.
        meta.supportsRandomAccess = true;

        return EncodedData(std::move(buffer), std::move(meta));
#endif
    }

    std::vector<T> decodeAll(const EncodedData& encoded) override {
#ifndef HAVE_OPENZL
        throw std::runtime_error("OpenZLCodec: OpenZL not available (HAVE_OPENZL not defined)");
#else
        const size_t bytes = encoded.metadata().uncompressedSize;
        if (bytes == 0) return {};
        const size_t N = bytes / sizeof(T);

        if constexpr (BlockSize == 0) {
            return decompressSingle(encoded.data().data(), encoded.data().size(), N);
        }

        // Blocked decode: parse header
        const uint8_t* p = encoded.data().data();
        const uint8_t* end = p + encoded.data().size();
        if (static_cast<size_t>(end - p) < sizeof(uint64_t)) {
            throw std::runtime_error("OpenZLCodec: corrupted block header (count)");
        }
        const uint64_t blockCount = readUint64(p);

        std::vector<T> out;
        out.reserve(N);

        for (uint64_t b = 0; b < blockCount; ++b) {
            if (static_cast<size_t>(end - p) < 2 * sizeof(uint64_t)) {
                throw std::runtime_error("OpenZLCodec: corrupted block header (sizes)");
            }
            const uint64_t elems = readUint64(p);
            const uint64_t compSize = readUint64(p);
            if (static_cast<size_t>(end - p) < compSize) {
                throw std::runtime_error("OpenZLCodec: corrupted block payload");
            }
            std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
            out.insert(out.end(), block.begin(), block.end());
            p += compSize;
        }

        if (out.size() != N) {
            throw std::runtime_error("OpenZLCodec: decoded element count mismatch");
        }
        return out;
#endif
    }

    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const size_t bytes = encoded.metadata().uncompressedSize;
        if (bytes == 0) return std::nullopt;
        const size_t N = bytes / sizeof(T);
        if (index >= N) return std::nullopt;

        if constexpr (BlockSize == 0) {
            auto all = decodeAll(encoded);
            return all[index];
        }

        const uint8_t* p = encoded.data().data();
        const uint8_t* end = p + encoded.data().size();
        if (static_cast<size_t>(end - p) < sizeof(uint64_t)) {
            throw std::runtime_error("OpenZLCodec: corrupted block header (count)");
        }
        const uint64_t blockCount = readUint64(p);

        uint64_t base = 0;
        for (uint64_t b = 0; b < blockCount; ++b) {
            if (static_cast<size_t>(end - p) < 2 * sizeof(uint64_t)) {
                throw std::runtime_error("OpenZLCodec: corrupted block header (sizes)");
            }
            const uint64_t elems = readUint64(p);
            const uint64_t compSize = readUint64(p);
            if (static_cast<size_t>(end - p) < compSize) {
                throw std::runtime_error("OpenZLCodec: corrupted block payload");
            }
            if (index < base + elems) {
                std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
                return block[static_cast<size_t>(index - base)];
            }
            base += elems;
            p += compSize;
        }
        return std::nullopt;
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const size_t bytes = encoded.metadata().uncompressedSize;
        if (bytes == 0) return {};
        const size_t N = bytes / sizeof(T);
        if (start >= N) return {};
        end = std::min(end, N);

        if constexpr (BlockSize == 0) {
            auto all = decodeAll(encoded);
            return std::vector<T>(all.begin() + static_cast<ptrdiff_t>(start),
                                  all.begin() + static_cast<ptrdiff_t>(end));
        }

        const uint8_t* p = encoded.data().data();
        const uint8_t* readEnd = p + encoded.data().size();
        if (static_cast<size_t>(readEnd - p) < sizeof(uint64_t)) {
            throw std::runtime_error("OpenZLCodec: corrupted block header (count)");
        }
        const uint64_t blockCount = readUint64(p);

        std::vector<T> out;
        out.reserve(end - start);
        uint64_t base = 0;
        for (uint64_t b = 0; b < blockCount && base < end; ++b) {
            if (static_cast<size_t>(readEnd - p) < 2 * sizeof(uint64_t)) {
                throw std::runtime_error("OpenZLCodec: corrupted block header (sizes)");
            }
            const uint64_t elems = readUint64(p);
            const uint64_t compSize = readUint64(p);
            if (static_cast<size_t>(readEnd - p) < compSize) {
                throw std::runtime_error("OpenZLCodec: corrupted block payload");
            }

            const uint64_t blockEnd = base + elems;
            if (blockEnd > start && base < end) {
                // Overlaps requested range
                std::vector<T> block = decompressSingle(p, static_cast<size_t>(compSize), static_cast<size_t>(elems));
                const size_t localStart = static_cast<size_t>(std::max<uint64_t>(start, base) - base);
                const size_t localEnd = static_cast<size_t>(std::min<uint64_t>(end, blockEnd) - base);
                out.insert(out.end(), block.begin() + static_cast<ptrdiff_t>(localStart),
                                      block.begin() + static_cast<ptrdiff_t>(localEnd));
            }

            base = blockEnd;
            p += compSize;
        }

        return out;
    }

    EncodingType encodingType() const override { return EncodingType::OpenZL; }

    std::string name() const override { return "OpenZL"; }

    EncodingProperties properties() const override {
        using EP = EncodingProperty;
        return EncodingProperties{}
            .add(EP::Lossless)
            .add(EP::RandomAccess)
            .add(EP::RequiresFullData);
    }

    size_t estimateEncodedSize(size_t elementCount) const override {
        return elementCount * sizeof(T); // upper bound; OpenZL compressBound requires context
    }

private:
#ifdef HAVE_OPENZL
    static void appendUint64(std::vector<uint8_t>& buf, uint64_t v) {
        uint8_t tmp[sizeof(uint64_t)];
        std::memcpy(tmp, &v, sizeof(uint64_t));
        buf.insert(buf.end(), tmp, tmp + sizeof(uint64_t));
    }

    static uint64_t readUint64(const uint8_t*& p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        return v;
    }

    std::vector<uint8_t> compressBlock(std::span<const T> data) const {
        ZL_CCtx* ctx = ZL_CCtx_create();
        if (!ctx) {
            throw std::runtime_error("OpenZLCodec: failed to create ZL_CCtx");
        }

        const size_t N = data.size();
        const size_t srcBytes = N * sizeof(T);
        const void* srcPtr = static_cast<const void*>(data.data());

        ZL_TypedRef* inRef = ZL_TypedRef_createNumeric(srcPtr, sizeof(T), N);
        if (!inRef) {
            ZL_CCtx_free(ctx);
            throw std::runtime_error("OpenZLCodec: failed to create TypedRef");
        }

        const size_t adjustedSrc = srcBytes * 2;
        const size_t bound = ZL_compressBound(adjustedSrc);
        std::vector<uint8_t> buffer(bound);

        ZL_Compressor* graph = ZL_Compressor_create();
        if (!graph) {
            ZL_TypedRef_free(inRef);
            ZL_CCtx_free(ctx);
            throw std::runtime_error("OpenZLCodec: failed to create compressor");
        }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
        const ZL_GraphID gid = ZL_GRAPH_NUMERIC;
#pragma clang diagnostic pop
        if (ZL_isError(ZL_Compressor_selectStartingGraphID(graph, gid))) {
            ZL_Compressor_free(graph);
            ZL_TypedRef_free(inRef);
            ZL_CCtx_free(ctx);
            throw std::runtime_error("OpenZLCodec: failed to select numeric starting graph");
        }

        if (ZL_isError(ZL_CCtx_refCompressor(ctx, graph))) {
            ZL_Compressor_free(graph);
            ZL_TypedRef_free(inRef);
            ZL_CCtx_free(ctx);
            throw std::runtime_error("OpenZLCodec: failed to ref compressor");
        }

        if (ZL_isError(ZL_CCtx_setParameter(ctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION))) {
            ZL_Compressor_free(graph);
            ZL_TypedRef_free(inRef);
            ZL_CCtx_free(ctx);
            throw std::runtime_error("OpenZLCodec: failed to set format version parameter");
        }

        const bool traceOpenZL = (std::getenv("OPENZL_TRACE") != nullptr);
        struct TraceContext {
            std::unordered_set<std::string> graphs;
            std::unordered_set<std::string> codecs;
        } traceCtx;
        ZL_CompressIntrospectionHooks hooks{};
        if (traceOpenZL) {
            hooks.on_migraphEncode_start = [](void* opaque, ZL_Graph*, const ZL_Compressor* compressor, ZL_GraphID gid, ZL_Edge*[], size_t) noexcept {
                auto* ctx = static_cast<TraceContext*>(opaque);
                if (ctx) {
                    const char* name = ZL_Compressor_Graph_getName(compressor, gid);
                    ctx->graphs.emplace(name ? name : "(anonymous graph)");
                }
            };
            hooks.on_codecEncode_start = [](void* opaque, ZL_Encoder*, const ZL_Compressor* compressor, ZL_NodeID nid, const ZL_Input*[], size_t) noexcept {
                auto* ctx = static_cast<TraceContext*>(opaque);
                if (ctx) {
                    const char* name = ZL_Compressor_Node_getName(compressor, nid);
                    ctx->codecs.emplace(name ? name : "(node)");
                }
            };
            hooks.opaque = &traceCtx;
            ZL_Report hookReport = ZL_CCtx_attachIntrospectionHooks(ctx, &hooks);
            if (ZL_isError(hookReport)) {
                const char* hookErr = ZL_CCtx_getErrorContextString(ctx, hookReport);
                std::cerr << "[OpenZL trace] failed to attach introspection hooks: "
                          << (hookErr ? hookErr : "unknown")
                          << std::endl;
            }
        }

        ZL_Report r = ZL_CCtx_compressTypedRef(ctx, buffer.data(), bound, inRef);

        if (ZL_isError(r)) {
            const char* err = ZL_CCtx_getErrorContextString(ctx, r);
            std::string errMsg = err ? std::string(err) : std::string("unknown");
            ZL_Compressor_free(graph);
            ZL_TypedRef_free(inRef);
            ZL_CCtx_free(ctx);
            throw std::runtime_error(std::string("OpenZLCodec: compress failed: ") + errMsg);
        }

        size_t compressedSize = ZL_validResult(r);
        buffer.resize(compressedSize);

        if (traceOpenZL) {
            ZL_Report detachReport = ZL_CCtx_detachAllIntrospectionHooks(ctx);
            if (ZL_isError(detachReport)) {
                const char* detachErr = ZL_CCtx_getErrorContextString(ctx, detachReport);
                std::cerr << "[OpenZL trace] detach hooks failed: "
                          << (detachErr ? detachErr : "unknown")
                          << std::endl;
            }
            std::cerr << "[OpenZL trace] graphs used: ";
            for (const auto& n : traceCtx.graphs) std::cerr << '"' << n << '"' << ' ';
            std::cerr << "\n[OpenZL trace] codecs used: ";
            for (const auto& n : traceCtx.codecs) std::cerr << '"' << n << '"' << ' ';
            std::cerr << std::endl;
        }

        ZL_Compressor_free(graph);
        ZL_TypedRef_free(inRef);
        ZL_CCtx_free(ctx);

        return buffer;
    }

    std::vector<T> decompressSingle(const uint8_t* data, size_t size, size_t elemCount) const {
        ZL_DCtx* dctx = ZL_DCtx_create();
        if (!dctx) {
            throw std::runtime_error("OpenZLCodec: failed to create ZL_DCtx");
        }

        std::vector<T> out(elemCount);
        ZL_OutputInfo outInfo{};
        ZL_Report r = ZL_DCtx_decompressTyped(
            dctx,
            &outInfo,
            static_cast<void*>(out.data()), elemCount * sizeof(T),
            data, size);

        if (ZL_isError(r)) {
            const char* err = ZL_DCtx_getErrorContextString(dctx, r);
            std::string errMsg = err ? std::string(err) : std::string("unknown");
            ZL_DCtx_free(dctx);
            throw std::runtime_error(std::string("OpenZLCodec: decompress failed: ") + errMsg);
        }

        ZL_DCtx_free(dctx);
        return out;
    }
#endif
};

// Convenience factory
 template <typename T, size_t BlockSize = 0>
 std::shared_ptr<OpenZLCodec<T, BlockSize>> makeOpenZLCodec() {
     return std::make_shared<OpenZLCodec<T, BlockSize>>();
 }

} // namespace encodings::encoders
