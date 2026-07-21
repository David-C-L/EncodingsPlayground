#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace encodings::benchmark::operatorgraph {

// Minimal JSON string escaping, matching JSONWriter::escape() in
// BenchmarkOutput.hpp (not reused directly to avoid pulling in that header's
// heavy BenchmarkRunner.hpp -> MemoryTracker/AllocationTracker/DataGenerator
// dependency chain for a single helper function).
inline std::string jsonEscape(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

inline std::string jsonStr(const std::string& s) {
    return "\"" + jsonEscape(s) + "\"";
}

template <typename T>
inline std::string jsonOpt(const std::optional<T>& v) {
    std::ostringstream ss;
    if (v.has_value()) ss << *v;
    else ss << "null";
    return ss.str();
}

inline std::string jsonOptId(const std::optional<size_t>& v) {
    return v.has_value() ? std::to_string(*v) : "null";
}

// ---------------------------------------------------------------------------
// OpenZL operator graph structs (defined before SegmentJson, which embeds an
// optional per-segment OpenZLGraphJson for OracleSIS splits)
// ---------------------------------------------------------------------------

struct OpenZLStreamJson {
    size_t id{0};
    std::string type;
    size_t eltWidth{0};
    size_t numElts{0};
    size_t contentSize{0};
    size_t compressedShareBytes{0};
    double compressedSharePct{0.0};
    std::optional<size_t> producerCodecId;
    std::optional<size_t> consumerCodecId;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "        \"id\": " << id << ",\n";
        ss << "        \"type\": " << jsonStr(type) << ",\n";
        ss << "        \"eltWidth\": " << eltWidth << ",\n";
        ss << "        \"numElts\": " << numElts << ",\n";
        ss << "        \"contentSize\": " << contentSize << ",\n";
        ss << "        \"compressedShareBytes\": " << compressedShareBytes << ",\n";
        ss << "        \"compressedSharePct\": " << compressedSharePct << ",\n";
        ss << "        \"producerCodecId\": " << jsonOptId(producerCodecId) << ",\n";
        ss << "        \"consumerCodecId\": " << jsonOptId(consumerCodecId) << "\n";
        ss << "      }";
        return ss.str();
    }
};

struct OpenZLCodecJson {
    size_t id{0};
    std::string name;
    unsigned codecId{0};
    bool isStandard{true};
    size_t headerSize{0};
    std::vector<size_t> inputStreamIds;
    std::vector<size_t> outputStreamIds;

    static std::string idArray(const std::vector<size_t>& ids) {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) ss << ", ";
            ss << ids[i];
        }
        ss << "]";
        return ss.str();
    }

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "        \"id\": " << id << ",\n";
        ss << "        \"name\": " << jsonStr(name) << ",\n";
        ss << "        \"codecId\": " << codecId << ",\n";
        ss << "        \"isStandard\": " << (isStandard ? "true" : "false") << ",\n";
        ss << "        \"headerSize\": " << headerSize << ",\n";
        ss << "        \"inputStreamIds\": " << idArray(inputStreamIds) << ",\n";
        ss << "        \"outputStreamIds\": " << idArray(outputStreamIds) << "\n";
        ss << "      }";
        return ss.str();
    }
};

struct OpenZLGraphJson {
    size_t compressedBytes{0};
    std::string selectedGraph;
    uint32_t frameFormatVersion{0};
    size_t frameHeaderSize{0};
    size_t frameFooterSize{0};
    size_t totalTransformHeaderSize{0};
    std::vector<OpenZLStreamJson> streams;
    std::vector<OpenZLCodecJson> codecs;

    // zstd is a single opaque leaf codec as far as OpenZL's own model goes
    // (see explore_best_encoding.cpp's OpenZLCodecStep::lzOnlyBytes) -- this
    // is an ESTIMATE, but obtained from zstd's OWN real behavior rather than
    // a substitute algorithm: the summed output size a direct libzstd call
    // achieves on the exact bytes every "zstd" invocation in this graph
    // received, at the same compression level, with literals forced to be
    // stored raw (ZSTD_lcm_uncompressed) instead of Huffman-coded. The gap
    // between this and the real zstd output (in the "zstd" node's own
    // totalOut, once aggregated) is attributed to literal entropy coding
    // specifically -- see build_openzl_digraph's node-splitting in
    // plot_operator_graph.py. Unset if the graph has no zstd invocation, or
    // the side-measurement couldn't be computed for any of them.
    std::optional<size_t> zstdLzOnlyBytes;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "    \"compressedBytes\": " << compressedBytes << ",\n";
        ss << "    \"selectedGraph\": " << jsonStr(selectedGraph) << ",\n";
        ss << "    \"frameFormatVersion\": " << frameFormatVersion << ",\n";
        ss << "    \"frameHeaderSize\": " << frameHeaderSize << ",\n";
        ss << "    \"frameFooterSize\": " << frameFooterSize << ",\n";
        ss << "    \"totalTransformHeaderSize\": " << totalTransformHeaderSize << ",\n";
        ss << "    \"zstdLzOnlyBytes\": " << jsonOpt(zstdLzOnlyBytes) << ",\n";
        ss << "    \"streams\": [\n";
        for (size_t i = 0; i < streams.size(); ++i) {
            ss << "      " << streams[i].toJson();
            if (i + 1 < streams.size()) ss << ",";
            ss << "\n";
        }
        ss << "    ],\n";
        ss << "    \"codecs\": [\n";
        for (size_t i = 0; i < codecs.size(); ++i) {
            ss << "      " << codecs[i].toJson();
            if (i + 1 < codecs.size()) ss << ",";
            ss << "\n";
        }
        ss << "    ]\n";
        ss << "  }";
        return ss.str();
    }
};

// ---------------------------------------------------------------------------
// Bit-range plan (AutoSIS / OracleSIS) structs
// ---------------------------------------------------------------------------

struct AlternativeJson {
    std::string encoding;
    size_t      sampleBytes{0};
    int         rank{0};

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\"encoding\": " << jsonStr(encoding)
           << ", \"sampleBytes\": " << sampleBytes
           << ", \"rank\": " << rank << "}";
        return ss.str();
    }
};

struct BlockFpeStatsJson {
    uint32_t blockSize{0};
    uint32_t numBlocks{0};
    double   avgNumTiers{0.0};
    double   avgTagBitWidth{0.0};
    double   avgFallbackFraction{0.0};

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "          \"blockSize\": " << blockSize << ",\n";
        ss << "          \"numBlocks\": " << numBlocks << ",\n";
        ss << "          \"avgNumTiers\": " << avgNumTiers << ",\n";
        ss << "          \"avgTagBitWidth\": " << avgTagBitWidth << ",\n";
        ss << "          \"avgFallbackFraction\": " << avgFallbackFraction << "\n";
        ss << "        }";
        return ss.str();
    }
};

struct SegmentJson {
    int bitStart{0};
    int bitEnd{0};
    int width{0};
    std::string encoding;
    std::string reorderer;
    std::optional<double> estimatedCostBits;  // AutoSIS only
    size_t sampleBytes{0};
    std::vector<AlternativeJson> alternatives;
    std::optional<BlockFpeStatsJson> blockFpeStats;
    // The oracle's chosen section codec re-encoded over the FULL dataset's
    // bit range (not the sample) — OracleSIS plans only. Lets the
    // full-dataset-scale openZlBytes/openZlOnOracleBytes below be compared
    // against an apples-to-apples full-dataset oracle size, rather than the
    // sample-scale sampleBytes above (whose fixed per-call framing overhead
    // is proportionally much larger on a ~10k-element sample).
    std::optional<size_t> fullDatasetBytes;
    // OpenZL applied directly to this segment's own bit-range data, run over
    // the FULL dataset — OracleSIS plans only, letting a segment's chosen
    // SubIntSplit encoding be compared against what OpenZL alone would do on
    // that exact same bit range.
    std::optional<size_t> openZlBytes;
    std::optional<OpenZLGraphJson> openZlGraph;
    // OpenZL applied as a *second* pass on top of this segment's already
    // SubIntSplit-encoded bytes (i.e. double-compression), both computed over
    // the FULL dataset — OracleSIS plans only. Shows whether/how much further
    // OpenZL can squeeze out of already-compressed data, and what codec graph
    // it picks when fed that (typically higher-entropy) byte stream.
    std::optional<size_t> openZlOnOracleBytes;
    std::optional<OpenZLGraphJson> openZlOnOracleGraph;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "        \"bitStart\": " << bitStart << ",\n";
        ss << "        \"bitEnd\": " << bitEnd << ",\n";
        ss << "        \"width\": " << width << ",\n";
        ss << "        \"encoding\": " << jsonStr(encoding) << ",\n";
        ss << "        \"reorderer\": " << jsonStr(reorderer) << ",\n";
        ss << "        \"estimatedCostBits\": " << jsonOpt(estimatedCostBits) << ",\n";
        ss << "        \"sampleBytes\": " << sampleBytes << ",\n";
        ss << "        \"alternatives\": [";
        for (size_t i = 0; i < alternatives.size(); ++i) {
            if (i) ss << ", ";
            ss << alternatives[i].toJson();
        }
        ss << "],\n";
        ss << "        \"blockFpeStats\": "
           << (blockFpeStats.has_value() ? blockFpeStats->toJson() : "null") << ",\n";
        ss << "        \"fullDatasetBytes\": " << jsonOpt(fullDatasetBytes) << ",\n";
        ss << "        \"openZlBytes\": " << jsonOpt(openZlBytes) << ",\n";
        ss << "        \"openZlGraph\": "
           << (openZlGraph.has_value() ? openZlGraph->toJson() : "null") << ",\n";
        ss << "        \"openZlOnOracleBytes\": " << jsonOpt(openZlOnOracleBytes) << ",\n";
        ss << "        \"openZlOnOracleGraph\": "
           << (openZlOnOracleGraph.has_value() ? openZlOnOracleGraph->toJson() : "null") << "\n";
        ss << "      }";
        return ss.str();
    }
};

struct BitRangePlanJson {
    std::optional<double> totalCost;
    std::vector<SegmentJson> segments;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"totalCost\": " << jsonOpt(totalCost) << ",\n";
        ss << "      \"segments\": [\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            ss << "      " << segments[i].toJson();
            if (i + 1 < segments.size()) ss << ",";
            ss << "\n";
        }
        ss << "      ]\n";
        ss << "    }";
        return ss.str();
    }
};

// ---------------------------------------------------------------------------
// Dataset / summary / top-level export
// ---------------------------------------------------------------------------

struct DatasetInfoJson {
    std::string label, path, column;
    size_t datasetSize{0}, sampleSize{0}, blockSize{0}, consecBlockSize{0};
    bool allowReorderers{false};
    std::vector<std::string> encodingTypes;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "    \"label\": " << jsonStr(label) << ",\n";
        ss << "    \"path\": " << jsonStr(path) << ",\n";
        ss << "    \"column\": " << jsonStr(column) << ",\n";
        ss << "    \"datasetSize\": " << datasetSize << ",\n";
        ss << "    \"sampleSize\": " << sampleSize << ",\n";
        ss << "    \"blockSize\": " << blockSize << ",\n";
        ss << "    \"consecBlockSize\": " << consecBlockSize << ",\n";
        ss << "    \"allowReorderers\": " << (allowReorderers ? "true" : "false") << ",\n";
        ss << "    \"encodingTypes\": [";
        for (size_t i = 0; i < encodingTypes.size(); ++i) {
            if (i) ss << ", ";
            ss << jsonStr(encodingTypes[i]);
        }
        ss << "]\n";
        ss << "  }";
        return ss.str();
    }
};

struct SummaryJson {
    size_t autoSisBytes{0}, oracleRandomBytes{0};
    std::optional<size_t> oracleConsecBytes, oracleMergedBytes, openZlBytes;
    double autoSisBpe{0.0}, oracleRandomBpe{0.0};
    std::optional<double> oracleConsecBpe, oracleMergedBpe, openZlBpe;

    // Flat, whole-buffer OpenZL baseline re-run at OpenZL's lowest/highest
    // ZL_CParam_compressionLevel, alongside the default-level openZlBytes/Bpe
    // above -- lets a comparison against OpenZL span its own tuning range
    // instead of being pinned to a single (default) point.
    std::optional<size_t> openZlLowestBytes, openZlHighestBytes;
    std::optional<double> openZlLowestBpe, openZlHighestBpe;
    double overheadVsOraclePct{0.0}, efficiencyPct{0.0};
    size_t segmentsMatching{0}, segmentsTotal{0};

    // Double-compression: each plan's own already-encoded full-dataset output,
    // fed through OpenZL as a second pass (see tryOpenZLCompressFull in
    // explore_best_encoding.cpp). Distinct from the per-segment
    // openZlOnOracleBytes in SegmentJson below, which sums independently
    // per-segment OpenZL passes rather than compressing the whole plan's
    // combined output as one buffer.
    std::optional<size_t> autoSisThenOpenZlBytes, oracleRandomThenOpenZlBytes,
        oracleConsecThenOpenZlBytes, oracleMergedThenOpenZlBytes;
    std::optional<double> autoSisThenOpenZlBpe, oracleRandomThenOpenZlBpe,
        oracleConsecThenOpenZlBpe, oracleMergedThenOpenZlBpe;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "    \"autoSisBytes\": " << autoSisBytes << ",\n";
        ss << "    \"oracleRandomBytes\": " << oracleRandomBytes << ",\n";
        ss << "    \"oracleConsecBytes\": " << jsonOpt(oracleConsecBytes) << ",\n";
        ss << "    \"oracleMergedBytes\": " << jsonOpt(oracleMergedBytes) << ",\n";
        ss << "    \"openZlBytes\": " << jsonOpt(openZlBytes) << ",\n";
        ss << "    \"autoSisBpe\": " << autoSisBpe << ",\n";
        ss << "    \"oracleRandomBpe\": " << oracleRandomBpe << ",\n";
        ss << "    \"oracleConsecBpe\": " << jsonOpt(oracleConsecBpe) << ",\n";
        ss << "    \"oracleMergedBpe\": " << jsonOpt(oracleMergedBpe) << ",\n";
        ss << "    \"openZlBpe\": " << jsonOpt(openZlBpe) << ",\n";
        ss << "    \"openZlLowestBytes\": " << jsonOpt(openZlLowestBytes) << ",\n";
        ss << "    \"openZlHighestBytes\": " << jsonOpt(openZlHighestBytes) << ",\n";
        ss << "    \"openZlLowestBpe\": " << jsonOpt(openZlLowestBpe) << ",\n";
        ss << "    \"openZlHighestBpe\": " << jsonOpt(openZlHighestBpe) << ",\n";
        ss << "    \"autoSisThenOpenZlBytes\": " << jsonOpt(autoSisThenOpenZlBytes) << ",\n";
        ss << "    \"oracleRandomThenOpenZlBytes\": " << jsonOpt(oracleRandomThenOpenZlBytes) << ",\n";
        ss << "    \"oracleConsecThenOpenZlBytes\": " << jsonOpt(oracleConsecThenOpenZlBytes) << ",\n";
        ss << "    \"oracleMergedThenOpenZlBytes\": " << jsonOpt(oracleMergedThenOpenZlBytes) << ",\n";
        ss << "    \"autoSisThenOpenZlBpe\": " << jsonOpt(autoSisThenOpenZlBpe) << ",\n";
        ss << "    \"oracleRandomThenOpenZlBpe\": " << jsonOpt(oracleRandomThenOpenZlBpe) << ",\n";
        ss << "    \"oracleConsecThenOpenZlBpe\": " << jsonOpt(oracleConsecThenOpenZlBpe) << ",\n";
        ss << "    \"oracleMergedThenOpenZlBpe\": " << jsonOpt(oracleMergedThenOpenZlBpe) << ",\n";
        ss << "    \"overheadVsOraclePct\": " << overheadVsOraclePct << ",\n";
        ss << "    \"efficiencyPct\": " << efficiencyPct << ",\n";
        ss << "    \"segmentsMatching\": " << segmentsMatching << ",\n";
        ss << "    \"segmentsTotal\": " << segmentsTotal << "\n";
        ss << "  }";
        return ss.str();
    }
};

struct OperatorGraphExport {
    int schemaVersion{1};
    DatasetInfoJson dataset;
    SummaryJson summary;
    BitRangePlanJson autoSis, oracleRandom, oracleConsecutive, oracleMerged;
    std::optional<OpenZLGraphJson> openZl;

    // Operator graphs for the double-compression passes summarised in
    // SummaryJson's autoSisThenOpenZlBytes/etc. -- the codec DAG OpenZL chose
    // when compressing each plan's OWN already-encoded output as a single
    // buffer (distinct from the per-segment openZlOnOracleGraph fields in
    // SegmentJson, which are per-segment, not whole-plan, graphs).
    std::optional<OpenZLGraphJson> autoSisThenOpenZl, oracleRandomThenOpenZl,
        oracleConsecThenOpenZl, oracleMergedThenOpenZl;

    std::string toJson() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"schemaVersion\": " << schemaVersion << ",\n";
        ss << "  \"dataset\": " << dataset.toJson() << ",\n";
        ss << "  \"summary\": " << summary.toJson() << ",\n";
        ss << "  \"bitRangePlans\": {\n";
        ss << "    \"autoSis\": " << autoSis.toJson() << ",\n";
        ss << "    \"oracleRandom\": " << oracleRandom.toJson() << ",\n";
        ss << "    \"oracleConsecutive\": " << oracleConsecutive.toJson() << ",\n";
        ss << "    \"oracleMerged\": " << oracleMerged.toJson() << "\n";
        ss << "  },\n";
        ss << "  \"openZl\": " << (openZl.has_value() ? openZl->toJson() : "null") << ",\n";
        ss << "  \"autoSisThenOpenZl\": " << (autoSisThenOpenZl.has_value() ? autoSisThenOpenZl->toJson() : "null") << ",\n";
        ss << "  \"oracleRandomThenOpenZl\": " << (oracleRandomThenOpenZl.has_value() ? oracleRandomThenOpenZl->toJson() : "null") << ",\n";
        ss << "  \"oracleConsecThenOpenZl\": " << (oracleConsecThenOpenZl.has_value() ? oracleConsecThenOpenZl->toJson() : "null") << ",\n";
        ss << "  \"oracleMergedThenOpenZl\": " << (oracleMergedThenOpenZl.has_value() ? oracleMergedThenOpenZl->toJson() : "null") << "\n";
        ss << "}\n";
        return ss.str();
    }

    bool save(const std::string& filepath) const {
        try {
            std::filesystem::path path(filepath);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream file(filepath);
            if (!file) {
                std::cerr << "Failed to open file: " << filepath << std::endl;
                return false;
            }
            file << toJson();
            std::cout << "Operator graph JSON saved to: " << filepath << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving operator graph JSON: " << e.what() << std::endl;
            return false;
        }
    }
};

} // namespace encodings::benchmark::operatorgraph
