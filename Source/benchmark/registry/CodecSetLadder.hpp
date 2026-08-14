#pragma once

// Ladders of allowed codec sets for the SubIntSplit random-access ablations.
//
// SubIntSplit's FastSkip is the MIN over its sections
// (SubIntSplitEncoder::allSectionsRandomAccess(), consulted by properties() at
// SubIntSplitEncoder.hpp ~line 569): one section whose codec cannot answer a
// positional query costs the WHOLE encoding its random-access property, however
// many bits it covers.  The ablation therefore does not compare finished
// encoders; it varies the set of codecs the DP is allowed to choose from and
// asks what the DP does with the extra freedom and what the encoding loses.
//
// Which codecs are "sequential" is QUERIED, never listed.  A rung is built by
// constructing the section codec the SubIntSplit factory would construct for an
// EncodingType and reading `properties().has(EncodingProperty::RandomAccess)`
// off it — the same predicate allSectionsRandomAccess() uses.  A hardcoded list
// would go stale the first time a codec gains a positional index, and would do
// so silently: the ladders would still run and the numbers would still look
// plausible.
//
// This header only builds the SETS.  It runs no DP, encodes nothing and knows
// nothing about measurement; bench_ablation.cpp turns a rung into a plan.

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "encoders/BitSplitOrder.hpp"
#include "encoders/SubIntEncodingUtils.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/SubStreamReordererType.hpp"
#include "encoders/selectors/costs/CostModelSet.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::benchmark {

using encodings::EncodingType;
using encodings::encoders::selectors::SubStreamReordererType;
using encodings::encoders::selectors::costs::CostModelDimension;

/// One step of a ladder: the codec set the DP may choose from, plus which codec
/// this rung added relative to the previous one.
///
/// `admitted` is empty on rung 0 and on rungs that are an endpoint rather than an
/// increment.  It carries the attribution for the headline result — the codec
/// that cost the encoding its FastSkip — so it is a name and not an index into
/// `allowed`, which shifts as the universe grows.
struct LadderRung {
    std::string name;
    std::vector<EncodingType> allowed;
    std::string admitted;
};

/// Each ladder answers a question the others cannot; see buildLadder().
enum class Ladder {
    RaOnlyThenSequentialOneByOne,
    RawUpwardThroughRa,
    SequentialOnly,
    RaVsSequentialWhole,
};

inline const char* ladderName(Ladder l) {
    switch (l) {
        case Ladder::RaOnlyThenSequentialOneByOne: return "ra_then_sequential_one_by_one";
        case Ladder::RawUpwardThroughRa:           return "raw_upward_through_ra";
        case Ladder::SequentialOnly:               return "sequential_only";
        case Ladder::RaVsSequentialWhole:          return "ra_vs_sequential_whole";
    }
    return "unknown";
}

namespace detail {

/// Builds the section codec the SubIntSplit factory would build for `et` and
/// returns its properties.
///
/// Full width (bits 0..63) so the single segment satisfies fromSegments()'s
/// full-coverage check; the properties of a section codec are a property of its
/// type, not of the width it was instantiated at, so the probe width does not
/// enter the answer.
///
/// A throwing type is not in the universe: fromSegments() throws
/// `unsupported encoding type for section` for an EncodingType the factory
/// cannot build, and that is the honest answer to "can the DP use this?" —
/// reporting it as non-random-access instead would put a codec on a ladder that
/// cannot be encoded at all.
inline bool probeSectionProperties(EncodingType et, encodings::EncodingProperties& out) {
    using encodings::encoders::SubIntSplitConfigIntegral;
    encodings::encoders::selectors::SegmentPlan seg;
    seg.bitStart  = 0;
    seg.bitEnd    = 63;
    seg.encoding  = et;
    seg.reorderer = SubStreamReordererType::None;
    try {
        const auto cfg = SubIntSplitConfigIntegral<uint64_t>::fromSegments(
            {seg}, encodings::encoders::BitSplitOrder::LSB_TO_MSB);
        out = cfg.codecs.front()->properties();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/// Memoised: the probe constructs a codec (and for a few types allocates
/// tables), and the ladders ask the same question of the same types repeatedly.
struct SectionCapability {
    bool constructible{};
    bool randomAccess{};
};

inline const SectionCapability& sectionCapability(EncodingType et) {
    static std::map<EncodingType, SectionCapability> cache;
    auto it = cache.find(et);
    if (it != cache.end()) return it->second;

    SectionCapability cap;
    encodings::EncodingProperties props;
    cap.constructible = probeSectionProperties(et, props);
    cap.randomAccess =
        cap.constructible && props.has(encodings::EncodingProperty::RandomAccess);
    return cache.emplace(et, cap).first->second;
}

}  // namespace detail

/// True when a SubIntSplit section using `et` can answer a positional query, and
/// therefore does not cost the enclosing encoding its FastSkip.
inline bool sectionRandomAccess(EncodingType et) {
    return detail::sectionCapability(et).randomAccess;
}

/// True when the SubIntSplit section factory can build a codec for `et` at all.
inline bool sectionConstructible(EncodingType et) {
    return detail::sectionCapability(et).constructible;
}

/// True when the DP has a compression cost model for `et`.
///
/// A codec the DP cannot cost is invisible to it: leaving such a type in a rung
/// would produce a rung indistinguishable from the previous one and read as "the
/// DP declined the codec" when in fact the codec was never offered.
///
/// The factory THROWS on an unsupported type rather than skipping it, so the
/// probe is a try — and so a rung must be filtered through this before it
/// reaches the factory, or one unmodelled type aborts the whole sweep.
inline bool hasDpCostModel(EncodingType et) {
    try {
        return !encodings::encoders::makeAutoSubIntSplitCostModelsFromTypes({et}).empty();
    } catch (const std::exception&) {
        return false;
    }
}

/// Every codec a SubIntSplit DP could actually be given: costable by the selector
/// AND constructible as a section.
///
/// Measured on this universe (31 pass both filters), exactly six declare no
/// RandomAccess: Huffman, FSE, and the four CascadingFOR variants with a Huffman
/// or FSE leaf.  That is why the ladders draw on this set rather than on
/// `dpDefaultUniverse()`, whose nine members are ALL random-access — a real
/// finding about the shipped selector configuration, and one that would leave
/// three of the four ladders with a single rung and nothing to ablate.
///
/// The enumeration bound is the last declared EncodingType.  A type appended
/// after it is silently outside the universe; there is no reflection over the
/// enum to prevent that, so the name is spelled out here to at least fail the
/// build if that enumerator is renamed or removed.
inline std::vector<EncodingType> codecUniverse() {
    constexpr int kLast =
        static_cast<int>(EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding);
    std::vector<EncodingType> out;
    for (int i = 0; i <= kLast; ++i) {
        const EncodingType et = static_cast<EncodingType>(i);
        if (hasDpCostModel(et) && sectionConstructible(et)) out.push_back(et);
    }
    return out;
}

/// The nine types AutoSIS is configured with by default, same filtering.  Kept
/// selectable so a ladder can be run over the set the shipped selector actually
/// uses, where the answer to "what does admitting a sequential codec cost?" is
/// "the question does not arise".
inline std::vector<EncodingType> dpDefaultUniverse() {
    std::vector<EncodingType> out;
    for (const EncodingType et : encodings::encoders::defaultAutoSubIntSplitCostModelTypes()) {
        if (hasDpCostModel(et) && sectionConstructible(et)) out.push_back(et);
    }
    return out;
}

namespace detail {

inline std::vector<EncodingType> partition(const std::vector<EncodingType>& universe,
                                          bool wantRandomAccess) {
    std::vector<EncodingType> out;
    for (const EncodingType et : universe) {
        if (sectionRandomAccess(et) == wantRandomAccess) out.push_back(et);
    }
    return out;
}

inline std::string typeName(EncodingType et) { return encodings::encodingTypeToString(et); }

/// Rungs are named `<index>_<what changed>` so the string sorts into ladder order
/// in a plot's category axis without the plot needing the rung index.
inline std::string rungName(size_t index, const std::string& suffix) {
    return std::to_string(index) + "_" + suffix;
}

}  // namespace detail

/// Build the rungs of one ladder over `universe`.
///
/// 1. RaOnlyThenSequentialOneByOne — rung 0 is every random-access codec; each
///    later rung adds exactly ONE sequential codec to that same base.  The rungs
///    are independent of each other, which is what isolates a single sequential
///    codec's contribution: cumulative admission cannot say which of two codecs
///    the DP actually wanted.  This is the ladder the FastSkip flip is attributed
///    from.
/// 2. RawUpwardThroughRa — rung 0 is {Raw} alone, then the random-access codecs
///    one at a time, then the sequential ones, each rung CUMULATIVE.  Marginal
///    value of each codec measured from the bottom instead of from a full set,
///    where a codec's contribution is masked by its substitutes.
/// 3. SequentialOnly — the sequential codecs, all together and then one at a
///    time.  Answers what neither of the above can: whether the random-access
///    codecs are paying for their property in bytes.  If ladder 1's rung 0
///    matches or beats this rung 0, random access is free on this data.
/// 4. RaVsSequentialWhole — the two endpoints only, for the headline table
///    across every access pattern and cache state.
///
/// A ladder whose relevant partition is empty comes back with the rungs it can
/// build (possibly just rung 0) rather than throwing: an empty partition is a
/// fact about the universe, and a driver reporting "no sequential codecs exist"
/// is more useful than one that aborts.
inline std::vector<LadderRung> buildLadder(Ladder ladder,
                                           const std::vector<EncodingType>& universe) {
    const std::vector<EncodingType> ra  = detail::partition(universe, /*wantRandomAccess=*/true);
    const std::vector<EncodingType> seq = detail::partition(universe, /*wantRandomAccess=*/false);
    std::vector<LadderRung> rungs;

    switch (ladder) {
        case Ladder::RaOnlyThenSequentialOneByOne: {
            if (ra.empty()) return rungs;
            rungs.push_back({detail::rungName(0, "ra_only"), ra, ""});
            for (const EncodingType s : seq) {
                std::vector<EncodingType> allowed = ra;
                allowed.push_back(s);
                rungs.push_back({detail::rungName(rungs.size(), "plus_" + detail::typeName(s)),
                                 std::move(allowed), detail::typeName(s)});
            }
            return rungs;
        }

        case Ladder::RawUpwardThroughRa: {
            std::vector<EncodingType> allowed{EncodingType::RawEncoding};
            rungs.push_back({detail::rungName(0, "raw_only"), allowed, ""});
            // Raw is already in, so it must not be admitted a second time — that
            // rung would duplicate rung 0 and read as "the DP declined Raw".
            const auto appendEach = [&](const std::vector<EncodingType>& part) {
                for (const EncodingType et : part) {
                    if (et == EncodingType::RawEncoding) continue;
                    allowed.push_back(et);
                    rungs.push_back({detail::rungName(rungs.size(), "plus_" + detail::typeName(et)),
                                     allowed, detail::typeName(et)});
                }
            };
            appendEach(ra);
            appendEach(seq);
            return rungs;
        }

        case Ladder::SequentialOnly: {
            if (seq.empty()) return rungs;
            rungs.push_back({detail::rungName(0, "sequential_all"), seq, ""});
            if (seq.size() > 1) {
                for (const EncodingType s : seq) {
                    rungs.push_back({detail::rungName(rungs.size(), "only_" + detail::typeName(s)),
                                     std::vector<EncodingType>{s}, detail::typeName(s)});
                }
            }
            return rungs;
        }

        case Ladder::RaVsSequentialWhole: {
            if (!ra.empty()) rungs.push_back({detail::rungName(0, "ra_only"), ra, ""});
            if (!seq.empty()) {
                rungs.push_back({detail::rungName(rungs.size(), "sequential_all"), seq, ""});
            }
            return rungs;
        }
    }
    return rungs;
}

/// Compact, stable rendering of a codec set for the `allowed_codecs` column.
inline std::string describeCodecSet(const std::vector<EncodingType>& allowed) {
    std::string out;
    for (const EncodingType et : allowed) {
        if (!out.empty()) out += "|";
        out += detail::typeName(et);
    }
    return out;
}

/// One cost dimension and its weight, as bench_ablation exposes it on the CLI.
struct CostDim {
    CostModelDimension dim{CostModelDimension::Compression};
    double weight{1.0};
};

/// The DP cost models for one rung.
///
/// Compression-only goes through makeAutoSubIntSplitCostModelsFromTypes rather
/// than CostModelSet, because CostModelSet's compression factory
/// (makeCompressionCostModel) has no model for BlockFrequencyPartitionEncoding,
/// BlockFSEEncoding or MainlyConstantEncoding and drops them silently — and
/// BlockFPE is exactly the codec AutoSIS picks for the wide sections of
/// TwitterSnowflake, so routing the default through CostModelSet would delete the
/// most interesting rung of the ladder while still producing a full table.  For
/// the two paths' shared types the models are identical objects, so the
/// compression rung is not a different measurement.
///
/// Any other dimension set does go through CostModelSet (it is the only thing
/// that can weight dimensions), and `dropped` names the types it could not build
/// so a driver can report the loss instead of absorbing it.
inline std::vector<std::unique_ptr<encodings::encoders::selectors::costs::EncodingCostModel>>
buildRungCostModels(const std::vector<EncodingType>& allowed,
                    const std::vector<CostDim>& dims,
                    std::vector<EncodingType>& dropped) {
    using encodings::encoders::selectors::costs::CostModelSet;
    dropped.clear();

    const bool compressionOnly =
        dims.size() == 1 && dims.front().dim == CostModelDimension::Compression;
    if (compressionOnly) {
        // Attribute drops here too.  This path used to return before touching
        // `dropped`, so models_dropped was "-" by construction and the column was
        // evidence of nothing -- on the DEFAULT run, which is the one people read.
        // It happens to be empty today because the factory throws on an unmodelled
        // type rather than dropping it, but a column that cannot report a drop must
        // not look like a column that found none.
        auto models = encodings::encoders::makeAutoSubIntSplitCostModelsFromTypes(allowed);
        std::vector<EncodingType> built;
        built.reserve(models.size());
        for (const auto& m : models) built.push_back(m->encodingType());
        for (const EncodingType et : allowed) {
            if (std::find(built.begin(), built.end(), et) == built.end()) dropped.push_back(et);
        }
        return models;
    }

    CostModelSet set;
    set.forEncodings(allowed);
    for (const CostDim& d : dims) set.add(d.dim, d.weight);
    auto models = set.build();

    // Attribute the loss by comparing what came back against what was asked for.
    // CostModelSet reports neither which types it dropped nor how many, so the
    // only honest source is the built vector's own encodingType() list.
    std::vector<EncodingType> built;
    built.reserve(models.size());
    for (const auto& m : models) built.push_back(m->encodingType());
    for (const EncodingType et : allowed) {
        if (std::find(built.begin(), built.end(), et) == built.end()) dropped.push_back(et);
    }
    return models;
}

/// Recommended DP split penalty for a dimension set, mirroring the AutoSIS
/// factories: compression-only keeps their hardcoded 100.0, anything with a
/// speed dimension takes CostModelSet's sample-count-scaled recommendation.
inline double recommendedSplitPenalty(const std::vector<CostDim>& dims, size_t sampleCount) {
    const bool compressionOnly =
        dims.size() == 1 && dims.front().dim == CostModelDimension::Compression;
    if (compressionOnly) return 100.0;
    encodings::encoders::selectors::costs::CostModelSet set;
    for (const CostDim& d : dims) set.add(d.dim, d.weight);
    return set.recommendedSplitPenalty(sampleCount);
}

}  // namespace encodings::benchmark
