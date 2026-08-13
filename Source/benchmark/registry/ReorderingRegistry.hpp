#pragma once

// The reordering family: reorderer x permutation format x inner codec.
//
// Separate from registry/EncoderRegistry.hpp on purpose.  These codecs need
// `encodings_reorderers`, which most drivers do not link, and a header that
// dragged that dependency into every driver would make the reordering work a
// build-time cost of measuring anything at all.  The shape is the same one
// EncoderRegistry establishes — an EncoderEntry per codec, `family` and `variant`
// carried on the entry rather than parsed back out of the name — so
// applyFilters(), ArtifactCache and DecodeHarness take these entries unchanged.
//
// WHAT THE AXES ARE FOR.  A reordering codec pays permutation bytes and
// permutation-inversion time to buy compression on the reordered values.  Both
// halves of that trade depend on the permutation FORMAT, and the formats split
// into two classes that behave nothing alike: FlatBitPacked and ChunkRelative
// answer "where did row i go?" with a bit extraction, while DeltaBitPacked,
// DeltaZstd, DeltaLZ4, ValueGrouped and InverseEliasFano can only answer it by
// unpacking the whole permutation.  Both classes are registered so the driver can
// report the cost of that difference rather than asserting it.

#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/registry/EncoderRegistry.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "encodings/Encoder.hpp"
#include "reorderers/BWTReorderer.hpp"
#include "reorderers/MTFReorderer.hpp"
#include "reorderers/PermutationStore.hpp"
#include "reorderers/ReorderingCodec.hpp"
#include "reorderers/SortReorderer.hpp"
#include "reorderers/WindowedSortReorderer.hpp"

namespace encodings::benchmark {

using encodings::reorderers::PermFormat;
using encodings::reorderers::PermResidency;
using encodings::reorderers::permFormatSupportsRandomAccess;

/// The reordering-specific facts about one entry.
///
/// These cannot live on EncoderEntry (which is shared with every other family)
/// and must not be re-derived from the entry name by string matching, because the
/// result columns are grouped by them: a name-parsing convention breaks silently
/// the first time an entry is renamed.
struct ReorderingSpec {
    std::string reorderer;   ///< "Sort", "WindowedSort<256>", "BWT<256>", "MTF<256>"
    std::string innerCodec;  ///< name of the codec the reordered values go through
    PermFormat  permFormat{PermFormat::None};

    /// True for the reorderers that store a position permutation at all (Sort,
    /// WindowedSort).  A value transform (BWT, MTF) has no "where did row i go"
    /// question to answer, so its permutation columns are meaningless rather than
    /// zero and the driver writes them as nulls.
    bool positionPermutation{false};

    /// From permFormatSupportsRandomAccess<F>, i.e. the format's own claim that a
    /// single position can be looked up without unpacking the permutation, AND
    /// only where such a lookup exists at all.  It is the axis the point-lookup
    /// rows are read against.
    bool permRandomAccess{false};

    /// Whether an unpacked permutation is allowed to stay resident between calls.
    /// Registered as a separate entry rather than a driver flag so that the cached
    /// and uncached costs of the same format appear as two rows of one table.
    PermResidency residency{PermResidency::PerCall};

    /// The reordering profiling hooks (reorder_encode_ns, unreorder_decode_all_ns,
    /// perm_lookup_*) only report on ReorderingCodec<T, true>.  Instrumentation
    /// perturbs what it reports — two steady_clock reads per decodeAt against a
    /// lookup of a few nanoseconds — so a control entry with profiling off is
    /// registered for the same codec and the driver reports both.
    bool profiling{true};

    /// 0 = no limit.  Encode cost, not decode cost: BWT's forward transform is
    /// O(W^2 log W) per window, so a full sweep at N = 10M would spend hours in
    /// encode() before measuring anything.
    size_t maxViableN{0};

    /// 0 = no limit.  MTF holds the whole value alphabet and copies it per window,
    /// so its cost scales with the column's distinct-value count, not with W.  On a
    /// high-cardinality column (TwitterSnowflake: one distinct value per row) it is
    /// not slow, it is quadratic — the driver emits those cells as not viable
    /// rather than either running them or dropping the row.
    size_t maxViableCardinality{0};

    /// Routes the *Into entry points back through Decoder's allocate-and-copy
    /// fallback, on this exact codec object.  Bound here because only the registry
    /// knows the concrete ReorderingCodec type; a driver holds Codec<T>.
    std::function<void(bool)> setBypassIntoOverrides;
};

namespace detail {

/// Specs live beside the entries, keyed by entry name, because EncoderEntry has no
/// room for them and the driver needs both.  Populated by reorderingFamily(),
/// which a driver therefore has to call before looking a spec up.
inline std::map<std::string, ReorderingSpec>& reorderingSpecStore() {
    static std::map<std::string, ReorderingSpec> specs;
    return specs;
}

/// Builds one (reorderer, inner codec) entry and records its spec.
///
/// `Reorder` is a factory rather than a made object so that each entry owns its
/// own reorderer and inner codec: ReorderingCodec keeps decode scratch and a copy
/// of the inner section bytes, so two entries sharing one inner codec would share
/// that state and evicting one artifact would cool the other.
template <PermFormat F, bool Profiling, typename ReordererT, typename InnerT>
inline EncoderEntry<int64_t> reorderingEntry(std::string name,
                                             std::string reordererLabel,
                                             std::shared_ptr<ReordererT> reorderer,
                                             std::shared_ptr<InnerT> inner,
                                             encodings::reorderers::ReorderingType rtype,
                                             PermResidency residency = PermResidency::PerCall,
                                             size_t maxViableN = 0,
                                             size_t maxViableCardinality = 0) {
    auto codec = encodings::reorderers::makeReorderingCodec<int64_t, Profiling>(
        reorderer, inner, rtype, name, residency);

    const bool positional = rtype == encodings::reorderers::ReorderingType::Sort ||
                            rtype == encodings::reorderers::ReorderingType::WindowedSort;

    ReorderingSpec spec;
    spec.reorderer            = std::move(reordererLabel);
    spec.innerCodec           = inner->name();
    spec.permFormat           = F;
    spec.positionPermutation  = positional;
    spec.permRandomAccess     = positional && permFormatSupportsRandomAccess<F>;
    spec.residency            = residency;
    spec.profiling            = Profiling;
    spec.maxViableN           = maxViableN;
    spec.maxViableCardinality = maxViableCardinality;
    spec.setBypassIntoOverrides = [codec](bool bypass) { codec->bypassIntoOverrides(bypass); };
    reorderingSpecStore()[name] = std::move(spec);

    // variant is "<reorderer>/<format>/<inner>" so a plot can group by any one of
    // the three axes without re-splitting the encoder name — and so that two
    // entries differing only in inner codec are distinguishable, which they would
    // not be if the inner codec were left in the name alone.
    const ReorderingSpec& stored = reorderingSpecStore()[name];
    std::string variant = stored.reorderer;
    variant += '/';
    variant += encodings::reorderers::PermutationStore::formatName(F);
    variant += '/';
    variant += stored.innerCodec;
    if (residency == PermResidency::Resident) variant += "/resident";
    if (!Profiling) variant += "/noprof";

    return detail::entry<int64_t>(std::move(name), std::move(codec), "reordering",
                                 std::move(variant));
}

}  // namespace detail

/// Spec for a registered reordering entry, or nullptr for a name that is not one.
inline const ReorderingSpec* reorderingSpec(const std::string& encoderName) {
    const auto& specs = detail::reorderingSpecStore();
    const auto it = specs.find(encoderName);
    return it == specs.end() ? nullptr : &it->second;
}

/// Reorderer x PermFormat x inner codec.
///
/// Two inner codecs, deliberately of different capability: RawBitPacked is
/// RandomAccess, so a mapped position can be read on its own and the permutation
/// is the only cost a selective read pays; Zstd is not, so every access — however
/// narrow — inverts the whole stream.  Without the second one the tables would
/// only describe the case where reordering is cheap.
inline std::vector<EncoderEntry<int64_t>> reorderingFamily() {
    using encodings::reorderers::BWTReorderer;
    using encodings::reorderers::MTFReorderer;
    using encodings::reorderers::ReorderingType;
    using encodings::reorderers::SortReorderer;
    using encodings::reorderers::WindowedSortReorderer;
    using encodings::encoders::RawBitPackedEncoder;
    using encodings::encoders::ZstdEncoder;

    std::vector<EncoderEntry<int64_t>> e;

    const auto bitPacked = [] { return std::make_shared<RawBitPackedEncoder<int64_t>>(); };
    const auto zstd      = [] { return std::make_shared<ZstdEncoder<int64_t>>(); };

    // ── Sort: the full permutation-format axis ──────────────────────────────
    // A global sort is the reorderer whose permutation is least structured, so it
    // is where the formats differ most and where a format that needs a full
    // unpack costs the most to use.
    const auto sortEntry = [&]<PermFormat F, bool Prof = true>(
                               PermResidency residency = PermResidency::PerCall) {
        std::string name = std::string("Sort[") +
                           encodings::reorderers::PermutationStore::formatName(F) + "]|BitPacked";
        if (residency == PermResidency::Resident) name += "+resident";
        if constexpr (!Prof) name += "+noprof";
        return detail::reorderingEntry<F, Prof>(
            std::move(name), "Sort", std::make_shared<SortReorderer<int64_t>>(F), bitPacked(),
            ReorderingType::Sort, residency);
    };

    e.push_back(sortEntry.template operator()<PermFormat::FlatBitPacked>());
    e.push_back(sortEntry.template operator()<PermFormat::DeltaBitPacked>());
    e.push_back(sortEntry.template operator()<PermFormat::DeltaZstd>());
    e.push_back(sortEntry.template operator()<PermFormat::DeltaLZ4>());
    e.push_back(sortEntry.template operator()<PermFormat::ValueGrouped>());
    e.push_back(sortEntry.template operator()<PermFormat::InverseEliasFano>());

    // The same two sequential formats with the unpacked permutation kept.  Their
    // per-call rows above and their resident rows here bound the same format from
    // the two ends a system could actually implement.
    e.push_back(sortEntry.template operator()<PermFormat::DeltaZstd>(PermResidency::Resident));
    e.push_back(
        sortEntry.template operator()<PermFormat::InverseEliasFano>(PermResidency::Resident));

    // Profiling-off control for the cheapest format, so the instrumentation's own
    // cost is a measured quantity rather than an assumption.
    e.push_back(sortEntry.template operator()<PermFormat::FlatBitPacked, false>());

    // Sequential inner codec: the permutation is random-access but the values are
    // not, which is the combination where a positional index buys nothing.
    e.push_back(detail::reorderingEntry<PermFormat::FlatBitPacked, true>(
        "Sort[FlatBitPacked]|Zstd", "Sort",
        std::make_shared<SortReorderer<int64_t>>(PermFormat::FlatBitPacked), zstd(),
        ReorderingType::Sort));

    // ── WindowedSort: the chunk-relative formats ────────────────────────────
    // Sorting inside windows of W keeps every target position inside its own
    // window, which is what makes the permutation storable as an intra-window rank
    // (log2(W) bits, random-access) instead of a global position.
    const auto wsortEntry = [&]<size_t W, PermFormat F>() {
        return detail::reorderingEntry<F, true>(
            std::string("WSort") + std::to_string(W) + "[" +
                encodings::reorderers::PermutationStore::formatName(F) + "]|BitPacked",
            "WindowedSort<" + std::to_string(W) + ">",
            std::make_shared<WindowedSortReorderer<int64_t, W>>(F), bitPacked(),
            ReorderingType::WindowedSort);
    };

    e.push_back(wsortEntry.template operator()<256, PermFormat::ChunkRelative>());
    e.push_back(wsortEntry.template operator()<256, PermFormat::ChunkRelativeZstd>());
    e.push_back(wsortEntry.template operator()<256, PermFormat::ChunkRelativeLZ4>());
    e.push_back(wsortEntry.template operator()<4096, PermFormat::ChunkRelative>());

    // ── Value transforms: no position permutation at all ────────────────────
    // Registered in the same family because they are alternatives a cost model
    // would choose between, and because their range and gather behaviour — one
    // full inversion regardless of how few rows were asked for — is the contrast
    // that gives the permutation formats' cost a scale.
    //
    // maxViableN for BWT: the forward transform sorts W rotations with an O(W)
    // comparator per window.
    e.push_back(detail::reorderingEntry<PermFormat::None, true>(
        "BWT256|BitPacked", "BWT<256>", std::make_shared<BWTReorderer<int64_t, 256>>(),
        bitPacked(), ReorderingType::BWT, PermResidency::PerCall, /*maxViableN=*/2'000'000));
    // maxViableCardinality for MTF: it copies the whole alphabet per window and
    // shifts it per element.
    e.push_back(detail::reorderingEntry<PermFormat::None, true>(
        "MTF256|BitPacked", "MTF<256>", std::make_shared<MTFReorderer<int64_t, 256>>(),
        bitPacked(), ReorderingType::MTF, PermResidency::PerCall, /*maxViableN=*/2'000'000,
        /*maxViableCardinality=*/65'536));

    return e;
}

}  // namespace encodings::benchmark
