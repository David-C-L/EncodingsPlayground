#pragma once

// OpenZL instrumentation: what codec DAG OpenZL actually built, and what each
// node in it cost.
//
// OpenZL is RETAINED as the state-of-the-art baseline — nimble uses it — and this
// instrumentation is a deliberately kept asset for a later SIS-explainability
// project, not leftover scaffolding.  It is a header rather than driver-private
// code so that bench_openzl_graph, explore_best_encoding and any later
// explainability tool report the same graph, and so plot_operator_graph.py has
// exactly one producer of its input schema.
//
// Two independent views of the same compression pass are captured, because
// neither alone is sufficient:
//
//   * introspection hooks (encodeOpenZLWithStats) see execution ORDER and the
//     bytes flowing through each codec invocation, but only as a flat list;
//   * post-hoc reflection (buildOpenZLGraphJson) walks the true stream/codec
//     DAG out of the finished frame, but cannot see per-invocation input bytes.

#include "benchmark/OperatorGraphJson.hpp"
#include "benchmark/OracleGrid.hpp"
#include "encoders/OpenZLEncoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Needed for ZSTD_c_literalCompressionMode/ZSTD_lcm_uncompressed (see
// zstdCompressLiteralsRaw below) -- these are "advanced" params gated behind
// this macro since their numeric ids aren't guaranteed stable across zstd
// versions. They're still dispatched through the fully stable, ABI-safe
// ZSTD_CCtx_setParameter(cctx, int, int) entry point, so this is safe to use
// against a dynamically-linked libzstd -- only the enum VALUES themselves
// come from this header, not any non-stable-ABI internals.
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

namespace encodings::benchmark {

struct OpenZLCodecStep {
    std::string name;
    size_t      headerBytes = 0;   // bytes written to frame for this codec's header
    size_t      inputBytes  = 0;   // total content size of all input streams
    size_t      outputBytes = 0;   // total content size of all output streams

    // zstd is a single opaque leaf as far as OpenZL's own codec/stream model is
    // concerned (one ZSTD_compress2 call -- see encode_zstd_binding.c), so
    // there's no real sub-DAG to report for what it does internally, and no
    // way to ask OpenZL itself to break it into an LZ-matching stage and a
    // literals-entropy-coding stage. This is an ESTIMATE, but obtained from
    // zstd's OWN real behavior rather than a substitute algorithm: we re-run
    // libzstd directly (see zstdCompressLiteralsRaw below) on the exact same
    // bytes this step received (captured via ZL_Input_ptr in
    // on_codecEncode_start), at the same compression level, with
    // ZSTD_c_literalCompressionMode forced to ZSTD_lcm_uncompressed -- i.e.
    // "what would this exact zstd call have produced if literals were stored
    // raw instead of Huffman-coded", isolating literal-entropy-coding as the
    // only difference from the real, unmodified outputBytes above. Only set
    // for steps whose name is "zstd".
    std::optional<size_t> lzOnlyBytes;
};

struct OpenZLEncodeStats {
    std::string                  selectedGraph;  // encoding strategy chosen by selector
    std::vector<OpenZLCodecStep> pipeline;       // codecs in execution order
};

// The introspection hooks (on_codecEncode_start/_end, used below) report a
// codec's own internal name -- e.g. "zl.private.zstd" -- which differs from
// the stripped "zstd" name ZL_ReflectionCtx/the JSON export use (see
// build_openzl_digraph's short_labels in plot_operator_graph.py, which strips
// the same "zl."/"zl.private." prefixes). Matching by suffix keeps both
// naming conventions working without hardcoding either one's exact prefix.
inline bool isZstdCodecName(const std::string& name) {
    return name.ends_with("zstd");
}

// Compresses `data` with plain libzstd at the given level, with literals
// forced to be stored raw (ZSTD_lcm_uncompressed) instead of Huffman-coded --
// see OpenZLCodecStep::lzOnlyBytes for why. Returns std::nullopt on any zstd
// error (best-effort: a failed side-measurement should never affect the real
// OpenZL compression this is measured alongside).
inline std::optional<size_t> zstdCompressLiteralsRaw(std::span<const uint8_t> data, int level) {
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) return std::nullopt;

    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_literalCompressionMode, ZSTD_lcm_uncompressed);

    const size_t bound = ZSTD_compressBound(data.size());
    std::vector<uint8_t> out(bound);
    const size_t r = ZSTD_compress2(cctx, out.data(), bound, data.data(), data.size());
    ZSTD_freeCCtx(cctx);

    if (ZSTD_isError(r)) return std::nullopt;
    return r;
}

// OpenZL's own default level (ZL_COMPRESSIONLEVEL_DEFAULT in zl_compress.h)
// plus the lowest/highest regular zstd levels it forwards to when zstd ends
// up as the selected leaf codec -- swept against the flat, whole-buffer
// OpenZL baseline so that comparison isn't pinned to a single, arbitrary
// (default) point on OpenZL's own tuning range.
constexpr int kOpenZLLevelLowest  = 1;
constexpr int kOpenZLLevelDefault = 6;
constexpr int kOpenZLLevelHighest = 22;

// Encodes `data` with the OpenZL ZL_GRAPH_NUMERIC compressor, attaches
// introspection hooks to capture the codec pipeline details, and returns the
// raw compressed bytes.  Throws on OpenZL unavailability or compression error.
// Templated on the element type so it can run on both a full int64_t dataset
// and uint64_t bit-range section values (see attachOpenZLToSegments).
// `compressionLevel`: 0 (default) leaves ZL_CParam_compressionLevel unset, so
// OpenZL uses its own internal default -- matches the "0 means not set"
// convention OpenZL itself uses for this parameter.
template <typename T>
inline std::vector<uint8_t> encodeOpenZLWithStats(
    std::span<const T> data, OpenZLEncodeStats& stats, int compressionLevel = 0) {
#ifndef HAVE_OPENZL
    (void)data; (void)stats; (void)compressionLevel;
    throw std::runtime_error("OpenZL not available (HAVE_OPENZL not defined)");
#else
    const size_t N        = data.size();
    const size_t srcBytes = N * sizeof(T);
    const void*  srcPtr   = static_cast<const void*>(data.data());

    ZL_CCtx* ctx = ZL_CCtx_create();
    if (!ctx) throw std::runtime_error("encodeOpenZLWithStats: ZL_CCtx_create failed");

    ZL_TypedRef* inRef = ZL_TypedRef_createNumeric(srcPtr, sizeof(T), N);
    if (!inRef) {
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: ZL_TypedRef_createNumeric failed");
    }

    ZL_Compressor* compressor = ZL_Compressor_create();
    if (!compressor) {
        ZL_TypedRef_free(inRef);
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: ZL_Compressor_create failed");
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
    const ZL_GraphID startGid = ZL_GRAPH_NUMERIC;
#pragma clang diagnostic pop

    if (ZL_isError(ZL_Compressor_selectStartingGraphID(compressor, startGid)) ||
        ZL_isError(ZL_CCtx_refCompressor(ctx, compressor)) ||
        ZL_isError(ZL_CCtx_setParameter(ctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)) ||
        (compressionLevel != 0 &&
         ZL_isError(ZL_CCtx_setParameter(ctx, ZL_CParam_compressionLevel, compressionLevel)))) {
        ZL_Compressor_free(compressor);
        ZL_TypedRef_free(inRef);
        ZL_CCtx_free(ctx);
        throw std::runtime_error("encodeOpenZLWithStats: compressor setup failed");
    }

    // Per-call capture state shared between hook callbacks via opaque pointer.
    struct CaptureCtx {
        OpenZLEncodeStats* stats;
        OpenZLCodecStep    pending;  // filled by on_codecEncode_start, pushed by _end
        // Raw bytes zstd receives this invocation (see OpenZLCodecStep::
        // lzOnlyBytes) -- only populated when isZstdCodecName(pending.name).
        std::vector<uint8_t> zstdInputCapture;
        // The compression level THIS pass actually resolved to -- mirrors
        // gcparams.c's own "0 means use ZL_COMPRESSIONLEVEL_DEFAULT" logic,
        // so zstdCompressLiteralsRaw's side-measurement runs at the same
        // level OpenZL's real zstd invocation used, isolating literal
        // compression as the only difference between the two runs.
        int effectiveZstdLevel;
    } capture{&stats, {}, {}, (compressionLevel != 0) ? compressionLevel : kOpenZLLevelDefault};

    ZL_CompressIntrospectionHooks hooks{};
    hooks.opaque = &capture;

    // Capture the encoding strategy name (fires once per block).
    hooks.on_migraphEncode_start = [](void* opaque, ZL_Graph*,
                                      const ZL_Compressor* cpr, ZL_GraphID g,
                                      ZL_Edge*[], size_t) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        const char* name = ZL_Compressor_Graph_getName(cpr, g);
        if (name && name[0] != '\0' && c->stats->selectedGraph.empty())
            c->stats->selectedGraph = name;
    };

    // Start of each codec transform: record name and sum input stream sizes.
    hooks.on_codecEncode_start = [](void* opaque, ZL_Encoder*,
                                    const ZL_Compressor* cpr, ZL_NodeID nid,
                                    const ZL_Input* ins[], size_t nb) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        c->pending = {};
        const char* name = ZL_Compressor_Node_getName(cpr, nid);
        c->pending.name = (name && name[0] != '\0') ? name : "(unknown)";
        for (size_t i = 0; i < nb; ++i)
            c->pending.inputBytes += ZL_Data_contentSize(ZL_codemodInputAsData(ins[i]));

        // zstd is an opaque leaf as far as OpenZL's own model goes (see
        // OpenZLCodecStep::lzOnlyBytes) -- grab the exact bytes it's about to
        // compress so a literals-uncompressed zstd side-measurement can run
        // on them once the real output size is known, in on_codecEncode_end.
        c->zstdInputCapture.clear();
        if (isZstdCodecName(c->pending.name)) {
            for (size_t i = 0; i < nb; ++i) {
                const auto* data = ZL_codemodInputAsData(ins[i]);
                const size_t sz = ZL_Data_contentSize(data);
                const auto* ptr = static_cast<const uint8_t*>(ZL_Input_ptr(ins[i]));
                if (ptr && sz > 0)
                    c->zstdInputCapture.insert(c->zstdInputCapture.end(), ptr, ptr + sz);
            }
        }
    };

    // Header bytes written to the compressed frame for this codec.
    hooks.on_ZL_Encoder_sendCodecHeader = [](void* opaque, ZL_Encoder*,
                                              const void*, size_t sz) noexcept {
        static_cast<CaptureCtx*>(opaque)->pending.headerBytes += sz;
    };

    // End of each codec transform: sum output stream sizes and push to pipeline.
    hooks.on_codecEncode_end = [](void* opaque, ZL_Encoder*,
                                  const ZL_Output* outs[], size_t nb,
                                  ZL_Report) noexcept {
        auto* c = static_cast<CaptureCtx*>(opaque);
        for (size_t i = 0; i < nb; ++i)
            c->pending.outputBytes += ZL_Data_contentSize(ZL_codemodConstOutputAsData(outs[i]));

        // Literals-uncompressed zstd side-measurement (see OpenZLCodecStep::
        // lzOnlyBytes) -- best-effort: leave unset if the side-measurement
        // itself fails so it never affects the real OpenZL compression pass.
        if (isZstdCodecName(c->pending.name) && !c->zstdInputCapture.empty()) {
            c->pending.lzOnlyBytes = zstdCompressLiteralsRaw(
                std::span<const uint8_t>(c->zstdInputCapture.data(), c->zstdInputCapture.size()),
                c->effectiveZstdLevel);
        }
        c->zstdInputCapture.clear();

        c->stats->pipeline.push_back(c->pending);
    };

    (void)ZL_CCtx_attachIntrospectionHooks(ctx, &hooks);

    const size_t bound = ZL_compressBound(srcBytes * 2);
    std::vector<uint8_t> buffer(bound);
    ZL_Report r = ZL_CCtx_compressTypedRef(ctx, buffer.data(), bound, inRef);

    (void)ZL_CCtx_detachAllIntrospectionHooks(ctx);
    ZL_Compressor_free(compressor);
    ZL_TypedRef_free(inRef);
    ZL_CCtx_free(ctx);

    if (ZL_isError(r))
        throw std::runtime_error(std::string("encodeOpenZLWithStats: compression failed"));

    buffer.resize(ZL_validResult(r));
    return buffer;
#endif
}

// Holds both the compressed bytes AND the codec-pipeline stats from a single
// OpenZL compression pass, so a caller can report the resulting size
// immediately AND (later, in the JSON export) build the full operator graph
// from the SAME pass -- avoiding a second, redundant full-dataset OpenZL
// compression just to get the graph.
struct OpenZLCompressResult {
    std::vector<uint8_t> buf;
    OpenZLEncodeStats stats;
};

// Convenience wrapper around encodeOpenZLWithStats, tolerant of OpenZL being
// unavailable or failing on a given buffer (e.g. table-size limits on
// high-cardinality data), so each call site doesn't need its own try/catch.
inline std::optional<OpenZLCompressResult> tryOpenZLCompressFull(std::span<const uint8_t> bytes) {
    try {
        OpenZLCompressResult result;
        result.buf = encodeOpenZLWithStats<uint8_t>(bytes, result.stats);
        return result;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// OpenZL operator-graph JSON export — post-hoc reflection over the already-
// compressed frame via ZL_ReflectionCtx. Unlike the introspection hooks above
// (which only see a flat execution-order list), this walks the true stream/
// codec DAG. Mirrors openzl/tools/streamdump/stream_dump2.c's fill_csize() and
// stream-graph walk, translated to JSON instead of Graphviz DOT. No custom
// decoder registration is needed since ZL_GRAPH_NUMERIC only uses standard
// (built-in) codecs — see stream_dump2_noop_shim.c for the equivalent no-op
// case in OpenZL's own tool.
// ---------------------------------------------------------------------------

#ifdef HAVE_OPENZL
// Recursively computes stream `streamIdx`'s share of total compressed bytes by
// walking forward through consumer codecs. Direct port of stream_dump2.c's
// fill_csize().
inline size_t fillStreamShare(
    const ZL_ReflectionCtx* rctx, std::vector<size_t>& shareBytes, size_t streamIdx) {
    if (shareBytes[streamIdx] != std::numeric_limits<size_t>::max())
        return shareBytes[streamIdx];

    const ZL_DataInfo* info = ZL_ReflectionCtx_getStream_lastChunk(rctx, streamIdx);
    const ZL_CodecInfo* consumer = ZL_DataInfo_getConsumerCodec(info);
    if (!consumer) {
        shareBytes[streamIdx] = ZL_DataInfo_getContentSize(info);
        return shareBytes[streamIdx];
    }

    size_t total = ZL_CodecInfo_getHeaderSize(consumer);
    const size_t nbOutputs = ZL_CodecInfo_getNumOutputs(consumer);
    for (size_t i = 0; i < nbOutputs; ++i) {
        const ZL_DataInfo* out = ZL_CodecInfo_getOutput(consumer, i);
        total += fillStreamShare(rctx, shareBytes, ZL_DataInfo_getIndex(out));
    }
    shareBytes[streamIdx] = total;
    return total;
}

inline const char* zlTypeName(ZL_Type t) {
    switch (t) {
        case ZL_Type_serial:  return "Serialized";
        case ZL_Type_struct:  return "Fixed_Width";
        case ZL_Type_numeric: return "Numeric";
        case ZL_Type_string:  return "Variable_Size";
        default:               return "Unknown";
    }
}

inline operatorgraph::OpenZLGraphJson buildOpenZLGraphJson(
    const std::vector<uint8_t>& compressedBuffer, const OpenZLEncodeStats& stats) {
    operatorgraph::OpenZLGraphJson g;
    g.compressedBytes = compressedBuffer.size();
    g.selectedGraph    = stats.selectedGraph;

    // Aggregate the zstd literals-uncompressed side-measurement across every
    // "zstd" invocation in this pass (see OpenZLCodecStep::lzOnlyBytes) into
    // one number -- Python re-aggregates all same-named codec invocations
    // into a single displayed node anyway (build_openzl_digraph), so there's
    // no need to attribute this per-invocation; only set if at least one
    // zstd step produced a measurement (zstdCompressLiteralsRaw can itself
    // fail/be skipped).
    size_t zstdLzOnlySum = 0;
    bool   haveZstdLzOnly = false;
    for (const auto& step : stats.pipeline) {
        if (isZstdCodecName(step.name) && step.lzOnlyBytes.has_value()) {
            zstdLzOnlySum += *step.lzOnlyBytes;
            haveZstdLzOnly = true;
        }
    }
    if (haveZstdLzOnly) g.zstdLzOnlyBytes = zstdLzOnlySum;

    ZL_ReflectionCtx* rctx = ZL_ReflectionCtx_create();
    if (!rctx) throw std::runtime_error("buildOpenZLGraphJson: ZL_ReflectionCtx_create failed");

    ZL_Report r = ZL_ReflectionCtx_setCompressedFrame(
        rctx, compressedBuffer.data(), compressedBuffer.size());
    if (ZL_isError(r)) {
        ZL_ReflectionCtx_free(rctx);
        throw std::runtime_error("buildOpenZLGraphJson: ZL_ReflectionCtx_setCompressedFrame failed");
    }

    g.frameFormatVersion       = ZL_ReflectionCtx_getFrameFormatVersion(rctx);
    g.frameHeaderSize          = ZL_ReflectionCtx_getFrameHeaderSize(rctx);
    g.frameFooterSize          = ZL_ReflectionCtx_getFrameFooterSize(rctx);
    g.totalTransformHeaderSize = ZL_ReflectionCtx_getTotalTransformHeaderSize_lastChunk(rctx);

    const size_t nbStreams = ZL_ReflectionCtx_getNumStreams_lastChunk(rctx);
    std::vector<size_t> shareBytes(nbStreams, std::numeric_limits<size_t>::max());
    g.streams.reserve(nbStreams);
    for (size_t i = 0; i < nbStreams; ++i) {
        const ZL_DataInfo* info = ZL_ReflectionCtx_getStream_lastChunk(rctx, i);
        const size_t share = fillStreamShare(rctx, shareBytes, i);

        operatorgraph::OpenZLStreamJson sj;
        sj.id                  = i;
        sj.type                = zlTypeName(ZL_DataInfo_getType(info));
        sj.eltWidth             = ZL_DataInfo_getEltWidth(info);
        sj.numElts              = ZL_DataInfo_getNumElts(info);
        sj.contentSize          = ZL_DataInfo_getContentSize(info);
        sj.compressedShareBytes = share;
        sj.compressedSharePct   = compressedBuffer.empty() ? 0.0
            : 100.0 * static_cast<double>(share) / static_cast<double>(compressedBuffer.size());

        if (const ZL_CodecInfo* producer = ZL_DataInfo_getProducerCodec(info))
            sj.producerCodecId = ZL_CodecInfo_getIndex(producer);
        if (const ZL_CodecInfo* consumer = ZL_DataInfo_getConsumerCodec(info))
            sj.consumerCodecId = ZL_CodecInfo_getIndex(consumer);

        g.streams.push_back(std::move(sj));
    }

    const size_t nbCodecs = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);
    g.codecs.reserve(nbCodecs);
    for (size_t i = 0; i < nbCodecs; ++i) {
        const ZL_CodecInfo* info = ZL_ReflectionCtx_getCodec_lastChunk(rctx, i);

        operatorgraph::OpenZLCodecJson cj;
        cj.id  = i;
        const char* nm = ZL_CodecInfo_getName(info);
        cj.name        = nm ? nm : "(unknown)";
        cj.codecId     = ZL_CodecInfo_getCodecID(info);
        cj.isStandard  = ZL_CodecInfo_isStandardCodec(info);
        cj.headerSize  = ZL_CodecInfo_getHeaderSize(info);

        const size_t nbIn = ZL_CodecInfo_getNumInputs(info);
        cj.inputStreamIds.reserve(nbIn);
        for (size_t k = 0; k < nbIn; ++k)
            cj.inputStreamIds.push_back(ZL_DataInfo_getIndex(ZL_CodecInfo_getInput(info, k)));

        const size_t nbOut = ZL_CodecInfo_getNumOutputs(info);
        cj.outputStreamIds.reserve(nbOut);
        for (size_t k = 0; k < nbOut; ++k)
            cj.outputStreamIds.push_back(ZL_DataInfo_getIndex(ZL_CodecInfo_getOutput(info, k)));

        g.codecs.push_back(std::move(cj));
    }

    ZL_ReflectionCtx_free(rctx);
    return g;
}

// ---------------------------------------------------------------------------
// Per-segment OpenZL — applies OpenZL directly to each OracleSIS segment's own
// bit-range data, extracted from the FULL dataset (not the sample used for the
// grid/alternatives), so a segment's chosen SubIntSplit encoding can be
// compared against what OpenZL alone achieves on that exact same data at full
// scale. Mutates each SegmentJson in place, leaving openZlBytes/openZlGraph
// unset (null in the JSON) if OpenZL fails on a given segment.
//
// COST: one ZL_CCtx compress over the whole dataset PER SEGMENT.  This is the
// most expensive thing in this header by a wide margin and dominates a run with
// many segments, which is why bench_openzl_graph puts it behind an explicit
// --openzl-per-segment flag rather than running it unconditionally.
// ---------------------------------------------------------------------------

inline void attachOpenZLToSegments(std::vector<operatorgraph::SegmentJson>& segments,
                                   const std::vector<uint64_t>& uFullData) {
    for (auto& sj : segments) {
        const std::vector<uint64_t> section = extractSection(uFullData, sj.bitStart, sj.width);
        try {
            OpenZLEncodeStats segStats;
            auto buf = encodeOpenZLWithStats<uint64_t>(
                std::span<const uint64_t>(section.data(), section.size()), segStats);
            sj.openZlBytes = buf.size();
            sj.openZlGraph = buildOpenZLGraphJson(buf, segStats);
        } catch (const std::exception&) {
            // OpenZL unavailable or failed on this segment — leave fields unset.
        }
    }
}

// ---------------------------------------------------------------------------
// Double-compression: takes each OracleSIS segment's ALREADY SubIntSplit-
// encoded byte buffer — re-derived from the oracle's chosen section codec over
// the FULL dataset's bit range — and runs OpenZL again on top of it, to see
// whether/how much further OpenZL can squeeze out of already-compressed data
// and what codec graph it picks on that (typically higher-entropy) byte stream.
// Also records the oracle codec's own full-dataset size as fullDatasetBytes, so
// the comparison is apples-to-apples with openZlBytes rather than mixing it
// with the sample-scale sampleBytes field.
// `segmentsJson` and `segmentsPlan` must be the same plan in the same order (as
// produced together by buildSegmentJson over a SegmentPlan vector).
// ---------------------------------------------------------------------------

inline void attachOpenZLOnOracleBytes(std::vector<operatorgraph::SegmentJson>& segmentsJson,
                                      const std::vector<SegmentPlan>& segmentsPlan,
                                      const std::vector<uint64_t>& uFullData) {
    const size_t n = std::min(segmentsJson.size(), segmentsPlan.size());
    for (size_t i = 0; i < n; ++i) {
        auto& sj = segmentsJson[i];
        const auto& seg = segmentsPlan[i];
        const int width = seg.bitEnd - seg.bitStart + 1;
        const std::vector<uint64_t> section = extractSection(uFullData, seg.bitStart, width);

        try {
            auto codec = makeSectionCodec(seg.encoding, static_cast<uint8_t>(width));
            auto encoded = codec->encode(std::span<const uint64_t>(section.data(), section.size()));
            const std::vector<uint8_t>& oracleBytes = encoded.data();
            sj.fullDatasetBytes = oracleBytes.size();

            OpenZLEncodeStats reStats;
            auto buf = encodeOpenZLWithStats<uint8_t>(
                std::span<const uint8_t>(oracleBytes.data(), oracleBytes.size()), reStats);
            sj.openZlOnOracleBytes = buf.size();
            sj.openZlOnOracleGraph = buildOpenZLGraphJson(buf, reStats);
        } catch (const std::exception&) {
            // OpenZL unavailable or failed on this segment — leave fields unset.
        }
    }
}
#endif // HAVE_OPENZL

// ---------------------------------------------------------------------------
// Aggregation and printout
// ---------------------------------------------------------------------------

/// All invocations of one codec name, summed.  ZL_GRAPH_NUMERIC calls fse_v2 once
/// per block, so a per-invocation list is thousands of rows describing one stage.
struct OpenZLAggStep {
    std::string name;
    size_t count       = 0;
    size_t totalIn     = 0;
    size_t totalOut    = 0;
    size_t totalHeader = 0;
};

/// First-occurrence order, which is execution order for the first call of each
/// stage — the only ordering that survives aggregation.
inline std::vector<OpenZLAggStep> aggregatePipeline(const OpenZLEncodeStats& stats) {
    std::vector<std::string> order;
    std::map<std::string, OpenZLAggStep> byName;
    for (const auto& s : stats.pipeline) {
        if (!byName.count(s.name)) order.push_back(s.name);
        auto& a = byName[s.name];
        a.name = s.name;
        ++a.count;
        a.totalIn     += s.inputBytes;
        a.totalOut    += s.outputBytes;
        a.totalHeader += s.headerBytes;
    }
    std::vector<OpenZLAggStep> out;
    out.reserve(order.size());
    for (const auto& name : order) out.push_back(byName[name]);
    return out;
}

inline void printOpenZLAnalysis(const OpenZLEncodeStats& stats, size_t totalCompressedBytes,
                                size_t datasetSize) {
    const double bpe = static_cast<double>(totalCompressedBytes) * 8.0
                       / static_cast<double>(datasetSize);
    const double ratio = static_cast<double>(totalCompressedBytes)
                         / static_cast<double>(datasetSize * sizeof(int64_t));

    std::cout << "\n=== OpenZL Internal Strategy (full dataset) ===\n";
    std::cout << "  Compressed:     " << totalCompressedBytes
              << " bytes  (" << std::fixed << std::setprecision(3) << bpe << " bits/elem"
              << "  ratio=" << ratio << "x)\n";
    std::cout << "  Encoding graph: \""
              << (stats.selectedGraph.empty() ? "(not captured)" : stats.selectedGraph)
              << "\"\n";

    if (stats.pipeline.empty()) {
        std::cout << "  (no codec pipeline data captured)\n";
        return;
    }

    const std::vector<OpenZLAggStep> agg = aggregatePipeline(stats);

    std::cout << "\n  Codec pipeline (" << agg.size()
              << " unique stages, " << stats.pipeline.size() << " total calls):\n";
    size_t totalHeaderBytes = 0;
    for (size_t i = 0; i < agg.size(); ++i) {
        const auto& a = agg[i];
        totalHeaderBytes += a.totalHeader;
        const double r = (a.totalIn > 0)
            ? static_cast<double>(a.totalOut) / static_cast<double>(a.totalIn) : 0.0;

        std::cout << "    #" << std::setw(2) << (i + 1) << "  " << padRight(a.name, 32);
        if (a.count > 1) std::cout << " x" << std::setw(5) << a.count;
        else std::cout << "        ";
        std::cout << "  in:" << std::setw(13) << a.totalIn << " B"
                  << "  out:" << std::setw(13) << a.totalOut << " B"
                  << "  hdr:" << std::setw(6) << a.totalHeader << " B"
                  << "  ratio:" << std::setprecision(3) << r << "x"
                  << "\n";
    }

    std::cout << "\n  Summary:\n";
    std::cout << "    Unique codec stages:  " << agg.size()
              << "  (" << stats.pipeline.size() << " total invocations)\n";
    std::cout << "    Total codec headers:  " << totalHeaderBytes << " bytes\n";
    {
        const size_t firstIn = stats.pipeline.front().inputBytes;
        const double e2e = (firstIn > 0)
            ? static_cast<double>(totalCompressedBytes) / static_cast<double>(firstIn) : 0.0;
        std::cout << "    End-to-end ratio:     " << std::setprecision(3) << e2e << "x"
                  << "  (" << firstIn << " B  ->  " << totalCompressedBytes << " B)\n";
    }

    // Strategy keywords, matched against the graph name and every aggregated
    // codec name, so the tags follow OpenZL's own naming rather than an
    // assumption about which graph it selected.
    auto hasName = [&](const char* kw) {
        if (stats.selectedGraph.find(kw) != std::string::npos) return true;
        for (const auto& a : agg)
            if (a.name.find(kw) != std::string::npos) return true;
        return false;
    };

    const bool usesTranspose = hasName("transpose") || hasName("split");
    const bool usesHuffman   = hasName("huffman");
    const bool usesFieldLz   = hasName("field_lz");
    const bool usesDelta     = hasName("delta") || hasName("Delta");
    const bool usesRangePack = hasName("range_pack");

    std::cout << "\n  Strategy detected: ";
    std::vector<std::string> tags;
    if (usesRangePack) tags.push_back("range-packing");
    if (usesFieldLz)   tags.push_back("field-level LZ");
    if (usesTranspose) tags.push_back("byte-transposition");
    if (usesHuffman)   tags.push_back("Huffman entropy coding");
    if (usesDelta)     tags.push_back("delta coding");
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i) std::cout << " + ";
        std::cout << tags[i];
    }
    if (tags.empty()) std::cout << "(see codec names above)";
    std::cout << "\n";

    std::cout << "\n  Interpretation (why OpenZL beats SubIntSplit):\n";
    if (usesTranspose && usesHuffman) {
        std::cout <<
            "    OpenZL transposes the int64 byte layout (struct-of-arrays): all byte-0\n"
            "    values together, all byte-1 values together, etc. — then Huffman-codes\n"
            "    each byte stream independently. High-order bytes (timestamp bits in\n"
            "    Snowflake IDs) change slowly across consecutive IDs, so those streams\n"
            "    achieve very low entropy and very high compression ratios. This is\n"
            "    conceptually similar to SubIntSplit's bit-decomposition, but at byte\n"
            "    granularity with full Huffman entropy coding.\n";
    } else if (usesDelta) {
        std::cout <<
            "    OpenZL applied delta-coding before entropy compression. Monotonically\n"
            "    increasing ids give small positive deltas — ideal for entropy coding.\n"
            "    SubIntSplit without a reorderer cannot exploit that sequential\n"
            "    structure; enabling BWT should close much of this gap.\n";
    } else {
        std::cout <<
            "    Strategy: \"" << stats.selectedGraph << "\". See codec pipeline above.\n"
            "    The gap likely comes from cross-byte entropy coding that SubIntSplit's\n"
            "    independent per-bit-range encoding cannot replicate.\n";
    }

    // Which aggregated stage removes the most bytes.  Reported rather than
    // inferred from the pipeline order: the last stage is not usually the one
    // doing the compression.
    size_t bestIdx = 0;
    double bestGain = 0.0;
    for (size_t i = 0; i < agg.size(); ++i) {
        const double gain = (agg[i].totalIn > agg[i].totalOut)
            ? static_cast<double>(agg[i].totalIn - agg[i].totalOut) : 0.0;
        if (gain > bestGain) { bestGain = gain; bestIdx = i; }
    }
    if (bestGain > 0) {
        const auto& best = agg[bestIdx];
        const double r = static_cast<double>(best.totalOut) / static_cast<double>(best.totalIn);
        std::cout << "\n  Biggest compression stage: \"" << best.name << "\""
                  << "  saved " << static_cast<size_t>(bestGain) << " bytes"
                  << "  (ratio=" << std::setprecision(3) << r << "x)\n";
    }
}

}  // namespace encodings::benchmark
