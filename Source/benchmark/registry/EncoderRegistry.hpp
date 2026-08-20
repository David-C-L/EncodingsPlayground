#pragma once

// The single list of codecs under test, grouped into families.
//
// Five drivers previously each carried their own copy of this list with a
// different subset commented out, so "the same encoder" meant a different plan
// in two tables that were then plotted together.  An encoder is added here
// once, to exactly one family, and every driver sees it.
//
// No `reorderingFamily()` here on purpose: reordering codecs live behind the
// `encodings_reorderers` target, which not every driver links, and pulling them
// into this header would force that dependency on all of them.  A later
// workstream adds `registry/ReorderingRegistry.hpp` with the same shape.

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

#include "encoders/AdaptiveFramedBitPrefixEncoder.hpp"
#include "encoders/BlockFORFPEEncoder.hpp"
#include "encoders/BlockFrequencyPartitionEncoder.hpp"
#include "encoders/FrequencyPartitionEncoder.hpp"
#include "encoders/OpenZLEncoder.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"

namespace encodings::benchmark {

/// One codec under test.
///
/// `family` and `variant` are carried on the entry rather than parsed back out
/// of `name` at write time, because the result columns `family`/`variant` are
/// what the plots group by and a name-prefix convention drifts the moment an
/// encoder is renamed.
template <typename T>
struct EncoderEntry {
    std::string name;
    std::shared_ptr<encodings::Codec<T>> codec;
    bool isSequential{};      ///< derived from properties(); see deriveIsSequential
    std::string family;       ///< "baseline" | "fpe-index" | "sis-manual" | "sis-auto" | "reordering"
    std::string variant;      ///< index type, plan signature, ... ; "" when the family has one member
    bool knownBroken{false};  ///< decode is known not to round-trip; drivers must gate on --validate
};

/// "A selective read must touch the whole payload per range."
///
/// Derived rather than hand-written: the three drivers that carried this flag
/// inline disagreed with each other about BlockFPE, and a flag that can be
/// computed cannot go stale when a codec gains a skip path.
///
/// `FastSkip` alone is not sufficient. It means the codec overrides
/// `decodeGatherInto()` with a genuine skip phase; a codec without it can still
/// seek — `RawEncoder` is `RandomAccess` and the base-class per-range fallback
/// reads only the selected rows. What actually makes a codec sequential is
/// having *neither*: then every range costs a decode from the start of the
/// stream (Zstd at BlockSize == 0 is the canonical case).
template <typename T>
inline bool deriveIsSequential(const encodings::Codec<T>& codec) {
    const auto props = codec.properties();
    return !props.has(encodings::EncodingProperty::FastSkip) &&
           !props.has(encodings::EncodingProperty::RandomAccess);
}

namespace detail {

/// Builds an entry with `isSequential` derived.  `sequentialOverride` exists for
/// codecs whose declared properties overstate their capability — OpenZL
/// declares `RandomAccess` unconditionally even at `BlockSize == 0`, where
/// decoding one row decompresses the entire frame.
template <typename T>
inline EncoderEntry<T> entry(std::string name,
                             std::shared_ptr<encodings::Codec<T>> codec,
                             std::string family,
                             std::string variant = {},
                             bool knownBroken = false,
                             std::optional<bool> sequentialOverride = std::nullopt) {
    EncoderEntry<T> e;
    e.isSequential = sequentialOverride.value_or(deriveIsSequential<T>(*codec));
    e.name = std::move(name);
    e.codec = std::move(codec);
    e.family = std::move(family);
    e.variant = std::move(variant);
    e.knownBroken = knownBroken;
    return e;
}

/// Compact, stable rendering of a manual SubIntSplit plan, e.g.
/// "13:BlockFPE|1:BitPacking|...".  Stable is the point: it is written into the
/// `variant` column, so two runs of the same plan must produce the same string.
inline std::string planSignature(const std::vector<uint8_t>& bits,
                                 const std::vector<encodings::EncodingType>& encs) {
    std::string s;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (i != 0) s += '|';
        s += std::to_string(static_cast<unsigned>(bits[i]));
        s += ':';
        s += encodings::encodingTypeToString(encs[i]);
    }
    return s;
}

}  // namespace detail

/// Whole-column codecs: the reference points every other family is measured
/// against.
inline std::vector<EncoderEntry<int64_t>> baselineEncoders() {
    using Elem = int64_t;
    using namespace encodings::encoders;

    std::vector<EncoderEntry<Elem>> e;
    e.push_back(detail::entry<Elem>("Raw", std::make_shared<RawEncoder<Elem>>(), "baseline"));
    e.push_back(detail::entry<Elem>("RawBitPacked",
                                    std::make_shared<RawBitPackedEncoder<Elem>>(), "baseline"));
    e.push_back(detail::entry<Elem>("AdaptiveBitPrefix",
                                    std::make_shared<AdaptiveFramedBitPrefixEncoder<Elem>>(),
                                    "baseline"));
    e.push_back(detail::entry<Elem>("BlockFPE",
                                    std::make_shared<BlockFrequencyPartitionEncoder<Elem>>(),
                                    "baseline"));
    // Registered, not omitted: silently dropping it would hide the bug and make
    // the missing rows look like a crashed sweep.  Drivers must exclude it via
    // --validate, which is where the exclusion gets recorded and sets exit 2.
    e.push_back(detail::entry<Elem>("BlockFORFPE",
                                    std::make_shared<BlockFORFPEEncoder<Elem>>(),
                                    "baseline", /*variant=*/{}, /*knownBroken=*/true));
    e.push_back(detail::entry<Elem>("Zstd", std::make_shared<ZstdEncoder<Elem>>(), "baseline"));
#ifdef HAVE_OPENZL
    e.push_back(detail::entry<Elem>("OpenZL", makeOpenZLCodec<Elem>(), "baseline", /*variant=*/{},
                                    /*knownBroken=*/false, /*sequentialOverride=*/true));
#endif
    return e;
}

/// The positional-index axis of FrequencyPartitionEncoding.  The index type is a
/// template parameter, so each one is a distinct codec type and the four must be
/// listed rather than looped over.
inline std::vector<EncoderEntry<int64_t>> fpeIndexFamily() {
    using Elem = int64_t;
    using encodings::encoders::FreqPartIndexType;
    using encodings::encoders::FrequencyPartitionEncoder;

    std::vector<EncoderEntry<Elem>> e;
    e.push_back(detail::entry<Elem>(
        "FPE_PerTierBitmaps",
        std::make_shared<FrequencyPartitionEncoder<Elem, FreqPartIndexType::PerTierBitmaps>>(),
        "fpe-index", "PerTierBitmaps"));
    e.push_back(detail::entry<Elem>(
        "FPE_TierTagArray",
        std::make_shared<FrequencyPartitionEncoder<Elem, FreqPartIndexType::TierTagArray>>(),
        "fpe-index", "TierTagArray"));
    e.push_back(detail::entry<Elem>(
        "FPE_EliasFano",
        std::make_shared<FrequencyPartitionEncoder<Elem, FreqPartIndexType::EliasFano>>(),
        "fpe-index", "EliasFano"));
    // NoIndex reorders rows by tier, so it cannot answer a positional query at
    // all; it is the lower bound on payload size that the other three pay an
    // index for.
    e.push_back(detail::entry<Elem>(
        "FPE_NoIndex",
        std::make_shared<FrequencyPartitionEncoder<Elem, FreqPartIndexType::NoIndex>>(),
        "fpe-index", "NoIndex"));
    return e;
}

/// Fixed SubIntSplit plans.  These are the plans the auto-selector picked on
/// TwitterSnowflake, frozen so that a driver measuring a plan is not also
/// measuring a DP run — and so a selector change shows up as a difference
/// against a fixed reference rather than moving both sides at once.
inline std::vector<EncoderEntry<int64_t>> sisManualPlans() {
    using Elem = int64_t;
    using encodings::EncodingType;
    using encodings::encoders::makeSubIntSplitEncoderManual;
    using encodings::encoders::makeSubIntSplitEncoderManualProf;

    std::vector<EncoderEntry<Elem>> e;

    const std::vector<uint8_t> snowflakeBits{13, 1, 8, 28, 8, 6};
    const std::vector<EncodingType> snowflakeEncs{
        EncodingType::BlockFrequencyPartitionEncoding,
        EncodingType::BitPacking,
        EncodingType::BlockFrequencyPartitionEncoding,
        EncodingType::BitPacking,
        EncodingType::RunLengthCascadingFOREncoding,
        EncodingType::RunLengthEncoding};

    const std::vector<uint8_t> deltaBits{13, 8, 1, 25, 17};
    const std::vector<EncodingType> deltaEncs{
        EncodingType::BlockFrequencyPartitionEncoding,
        EncodingType::AdaptiveDictionaryEncoding,
        EncodingType::BitPacking,
        EncodingType::BitPacking,
        EncodingType::CascadingFORPrevBlockFrequencyPartitionEncoding};

    e.push_back(detail::entry<Elem>(
        "SIS_Snowflake6", makeSubIntSplitEncoderManual<Elem>(snowflakeBits, snowflakeEncs),
        "sis-manual", detail::planSignature(snowflakeBits, snowflakeEncs)));
    // The Prof instantiation is a separate entry rather than a flag because only
    // SubIntSplitEncoder<T, true> reports the per-section and skip/materialize
    // splits, and its instrumentation perturbs the timings it reports — so it
    // must never stand in for the plain plan in a timing table.
    e.push_back(detail::entry<Elem>(
        "SIS_Snowflake6_Prof",
        makeSubIntSplitEncoderManualProf<Elem>(snowflakeBits, snowflakeEncs),
        "sis-manual", detail::planSignature(snowflakeBits, snowflakeEncs)));
    e.push_back(detail::entry<Elem>(
        "SIS_Delta5", makeSubIntSplitEncoderManual<Elem>(deltaBits, deltaEncs),
        "sis-manual", detail::planSignature(deltaBits, deltaEncs)));
    return e;
}

/// Cost-model-driven SubIntSplit.
///
/// Selection timing is enabled on every entry so `selectionTime_ns` reaches the
/// artifact: it is the one number that distinguishes these from `sisManualPlans`
/// and it is unrecoverable after the fact without re-running the DP.
inline std::vector<EncoderEntry<int64_t>> sisAutoEncoders() {
    using Elem = int64_t;
    using encodings::encoders::BitSplitOrder;

    // The config is assembled here instead of calling
    // makeDefaultAutoSubIntSplitEncoder(), which is a debugging helper: it sets
    // selectorConfig.costGridCsvPath to a path relative to "..", and the
    // selector THROWS if it cannot open that file for writing — so an AutoSIS
    // encoder built that way only works when the process happens to run one
    // directory below the repo root.  It also sets verboseLevel = 1, which
    // prints DP choices onto a driver's stdout.  Neither belongs in a
    // measurement run.
    const auto config = [](BitSplitOrder order) {
        auto cfg = encodings::encoders::makeDefaultAutoSubIntSplitConfig<Elem>(
            order, /*enableSelectionTiming=*/true);
        cfg.selectorConfig.orderHint = order;
        cfg.selectorConfig.useExhaustiveSearch = false;
        cfg.selectorConfig.enablePrune = true;
        cfg.selectorConfig.verboseLevel = 0;
        cfg.selectorConfig.costGridCsvPath.reset();
        return cfg;
    };

    // sequentialOverride: SubIntSplitAutoEncoder::properties() cannot report
    // RandomAccess before encode() has run, because the answer depends on the
    // plan the DP has not chosen yet — so the derivation would call every
    // AutoSIS entry sequential and drivers would drop its high-range-count
    // cells.  The post-encode truth is on the artifact (fastSkip/randomAccess),
    // which ArtifactCache reads after the encode.
    std::vector<EncoderEntry<Elem>> e;
    e.push_back(detail::entry<Elem>(
        "AutoSIS_LSB",
        encodings::encoders::makeAutoSubIntSplitEncoder<Elem>(config(BitSplitOrder::LSB_TO_MSB)),
        "sis-auto", "LSB_TO_MSB", /*knownBroken=*/false, /*sequentialOverride=*/false));
    e.push_back(detail::entry<Elem>(
        "AutoSIS_MSB",
        encodings::encoders::makeAutoSubIntSplitEncoder<Elem>(config(BitSplitOrder::MSB_TO_LSB)),
        "sis-auto", "MSB_TO_LSB", /*knownBroken=*/false, /*sequentialOverride=*/false));
    e.push_back(detail::entry<Elem>(
        "AutoSIS_LSB_Prof",
        encodings::encoders::makeAutoSubIntSplitEncoderProf<Elem>(
            config(BitSplitOrder::LSB_TO_MSB)),
        "sis-auto", "LSB_TO_MSB", /*knownBroken=*/false, /*sequentialOverride=*/false));
    return e;
}

/// Every registered encoder, in family order.
inline std::vector<EncoderEntry<int64_t>> allEncoders() {
    std::vector<EncoderEntry<int64_t>> all = baselineEncoders();
    const auto append = [&all](std::vector<EncoderEntry<int64_t>> part) {
        all.insert(all.end(), std::make_move_iterator(part.begin()),
                   std::make_move_iterator(part.end()));
    };
    append(fpeIndexFamily());
    append(sisManualPlans());
    append(sisAutoEncoders());
    return all;
}

/// Substring filter over `.name`, preserving registry order.
///
/// Empty `substrings` means "everything".  Matching is case-sensitive: filters
/// name registry entries, and a case-insensitive match would make `--encoder
/// raw` also select `AutoSIS_...` variants once someone adds one.
///
/// Returns an EMPTY vector when nothing matches, and does not throw.  Treating
/// that as an error is the DRIVER's job (per CONVENTIONS section 4: a filter
/// matching nothing must exit non-zero, never write an empty result file) —
/// throwing here would deny the driver the chance to report which filter failed
/// and to still print its --dry-run plan.
template <typename E>
inline std::vector<E> applyFilters(std::vector<E> in, const std::vector<std::string>& substrings) {
    if (substrings.empty()) return in;
    std::vector<E> out;
    for (auto& entry : in) {
        for (const auto& needle : substrings) {
            if (entry.name.find(needle) != std::string::npos) {
                out.push_back(std::move(entry));
                break;
            }
        }
    }
    return out;
}

}  // namespace encodings::benchmark
