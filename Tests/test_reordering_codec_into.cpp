// Correctness of ReorderingCodec's caller-owned-buffer decode paths.
//
// decodeAllInto, decodeRangeInto and decodeGatherInto used to be inherited from
// Decoder, which decodes into its own vector and copies; they are now overridden
// to write into dst directly, and for the position-permuted reorderers to touch
// only the rows asked for.  That makes them new code on the path every benchmark
// driver measures, so each one is checked against decodeAll() — the entry point
// that was already trusted — for every (reorderer, permutation format, inner
// codec) combination the registry can produce.
//
// The override is also checked against the fallback it replaced, on the same
// codec object and the same artifact via bypassIntoOverrides(): a fast path that
// disagrees with the slow one is the failure mode that a round-trip against the
// original data alone can hide, because both paths could be wrong the same way
// only if they shared code, and these deliberately do not.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "GatherTestHelpers.hpp"

#include "benchmark/SelectiveTraceGen.hpp"
#include "encoders/RawBitPackedEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include "reorderers/BWTReorderer.hpp"
#include "reorderers/MTFReorderer.hpp"
#include "reorderers/PermutationStore.hpp"
#include "reorderers/ReorderingCodec.hpp"
#include "reorderers/SortReorderer.hpp"
#include "reorderers/WindowedSortReorderer.hpp"

using namespace encodings;
using namespace encodings::reorderers;
using encodings::encoders::RawBitPackedEncoder;
using encodings::encoders::ZstdEncoder;

namespace {

using Elem = int64_t;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << "\n";
        ++failures;
    }
}

/// Ranges that exercise the boundaries of the mapping loops: the ends of the
/// stream, a window boundary for the windowed reorderers, and the degenerate
/// empty and full spans.
std::vector<std::pair<size_t, size_t>> rangeShapes(size_t n) {
    return {
        {0, 0},            // empty
        {0, n},            // full
        {0, 1},            // first element alone
        {n - 1, n},        // last element alone
        {n / 3, n / 3 + 1},
        {255, 258},        // straddles a W=256 window boundary
        {n / 4, (3 * n) / 4},
    };
}

/// decodeAllInto / decodeRangeInto / decodeGatherInto against the data, then the
/// same three against the base-class fallback on the same object.
void checkCodec(Codec<Elem>& codec, const std::vector<Elem>& original, const std::string& label) {
    const size_t n = original.size();
    const auto encoded = codec.encode(std::span<const Elem>(original));

    std::vector<Elem> dst(n, Elem{-1});
    codec.decodeAllInto(encoded, dst.data(), n);
    expect(dst == original, label + " / decodeAllInto");

    for (const auto& [begin, end] : rangeShapes(n)) {
        if (end > n) continue;
        const size_t count = end - begin;
        std::vector<Elem> got(count == 0 ? 1 : count, Elem{-1});
        codec.decodeRangeInto(encoded, begin, end, got.data(), count);
        bool ok = true;
        for (size_t i = 0; i < count; ++i) ok = ok && got[i] == original[begin + i];
        expect(ok, label + " / decodeRangeInto [" + std::to_string(begin) + ", " +
                       std::to_string(end) + ")");
    }

    expect(encodings::testutil::checkGatherAllTraceShapes(codec, encoded, original),
           label + " / decodeGatherInto trace shapes");

    // A gather whose ranges are dense enough to cover most of the stream, and one
    // sparse enough that almost every range is a single row: the two ends of the
    // range-count axis the driver sweeps.
    for (double selectivity : {0.9, 0.05}) {
        const auto ranges = encodings::benchmark::makeSelectiveTrace(
            n, {.selectivity = selectivity, .meanRunLength = selectivity > 0.5 ? 32.0 : 1.0});
        expect(encodings::testutil::checkGatherMatchesDecodeAll(
                   codec, encoded, original, ranges,
                   label + " sigma=" + std::to_string(selectivity)),
               label + " / gather sigma=" + std::to_string(selectivity));
    }
}

/// The override must agree with Decoder's fallback element for element.
void checkAgainstFallback(ReorderingCodec<Elem, true>& codec, const std::vector<Elem>& original,
                          const std::string& label) {
    const size_t n = original.size();
    const auto encoded = codec.encode(std::span<const Elem>(original));

    const auto ranges = encodings::benchmark::makeSelectiveTrace(
        n, {.selectivity = 0.25, .meanRunLength = 3.0});
    size_t selected = 0;
    for (const auto& r : ranges) selected += r.size();

    std::vector<Elem> fast(n, Elem{-1}), slow(n, Elem{-2});
    std::vector<Elem> fastGather(selected, Elem{-1}), slowGather(selected, Elem{-2});
    const size_t rBegin = n / 5, rEnd = (4 * n) / 5;
    std::vector<Elem> fastRange(rEnd - rBegin, Elem{-1}), slowRange(rEnd - rBegin, Elem{-2});

    codec.bypassIntoOverrides(false);
    codec.decodeAllInto(encoded, fast.data(), n);
    codec.decodeRangeInto(encoded, rBegin, rEnd, fastRange.data(), rEnd - rBegin);
    codec.decodeGatherInto(encoded, ranges, fastGather.data(), selected);

    codec.bypassIntoOverrides(true);
    codec.decodeAllInto(encoded, slow.data(), n);
    codec.decodeRangeInto(encoded, rBegin, rEnd, slowRange.data(), rEnd - rBegin);
    codec.decodeGatherInto(encoded, ranges, slowGather.data(), selected);
    codec.bypassIntoOverrides(false);

    expect(fast == slow, label + " / override == fallback (all)");
    expect(fastRange == slowRange, label + " / override == fallback (range)");
    expect(fastGather == slowGather, label + " / override == fallback (gather)");
}

/// cold-all needs every decoder-owned structure enumerated, and the resident
/// permutation is the one this workstream added.  A Resident codec must report
/// more bytes than a PerCall one reports for its recomputed copy is irrelevant —
/// what matters is that the permutation is reported at all, and that the reported
/// spans are non-empty so the cache controller has something to flush.
void checkInternalBuffers(const std::string& label, PermResidency residency) {
    std::vector<Elem> data(4096);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<Elem>((i * 7919) % 1000);

    auto codec = makeReorderingCodec<Elem, true>(
        std::make_shared<SortReorderer<Elem>>(PermFormat::FlatBitPacked),
        std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::Sort, "", residency);

    const auto encoded = codec->encode(std::span<const Elem>(data));
    expect(codec->internalBuffers().empty(), label + " / no internal buffers before a decode");

    std::vector<Elem> dst(data.size());
    codec->decodeAllInto(encoded, dst.data(), data.size());

    const auto buffers = codec->internalBuffers();
    expect(!buffers.empty(), label + " / internal buffers reported after a decode");
    size_t total = 0;
    for (const auto& b : buffers) {
        expect(!b.empty(), label + " / no empty span is reported");
        total += b.size();
    }
    expect(total >= data.size() * sizeof(size_t),
           label + " / the unpacked permutation is included");
    expect(codec->residentPermutationBytes() == data.size() * sizeof(size_t),
           label + " / residentPermutationBytes agrees with the unpack");
}

/// Low cardinality with long runs, plus a stretch of unique values: value groups
/// for ValueGrouped and InverseEliasFano, runs for BWT and MTF, and enough
/// distinct values that a permutation is not the identity.
std::vector<Elem> mixedData(size_t n) {
    std::vector<Elem> v(n);
    std::mt19937_64 rng(1234);
    for (size_t i = 0; i < n; ++i) {
        if (i % 5 == 0) v[i] = static_cast<Elem>(i);                       // unique
        else if (i % 3 == 0) v[i] = 42;                                    // long groups
        else v[i] = static_cast<Elem>(rng() % 64);                         // small alphabet
    }
    return v;
}

template <PermFormat F>
void checkSortFormat(const std::vector<Elem>& data) {
    const std::string label = std::string("Sort[") + PermutationStore::formatName(F) + "]";
    for (PermResidency residency : {PermResidency::PerCall, PermResidency::Resident}) {
        auto codec = makeReorderingCodec<Elem, true>(
            std::make_shared<SortReorderer<Elem>>(F),
            std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::Sort, "", residency);
        const std::string full = label + "/" + permResidencyToString(residency);
        checkCodec(*codec, data, full);
        checkAgainstFallback(*codec, data, full);
    }
}

template <size_t W, PermFormat F>
void checkWindowedSort(const std::vector<Elem>& data) {
    const std::string label = "WSort<" + std::to_string(W) + ">[" +
                              PermutationStore::formatName(F) + "]";
    auto codec = makeReorderingCodec<Elem, true>(
        std::make_shared<WindowedSortReorderer<Elem, W>>(F),
        std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::WindowedSort);
    checkCodec(*codec, data, label);
    checkAgainstFallback(*codec, data, label);
}

}  // namespace

int main() {
    // 3000 is small enough for BWT's O(W^2 log W) forward transform at W = 256 and
    // large enough to contain several windows and a straddled window boundary.
    const std::vector<Elem> data = mixedData(3000);

    checkSortFormat<PermFormat::FlatBitPacked>(data);
    checkSortFormat<PermFormat::DeltaBitPacked>(data);
    checkSortFormat<PermFormat::DeltaZstd>(data);
    checkSortFormat<PermFormat::DeltaLZ4>(data);
    checkSortFormat<PermFormat::ValueGrouped>(data);
    checkSortFormat<PermFormat::InverseEliasFano>(data);

    checkWindowedSort<256, PermFormat::ChunkRelative>(data);
    checkWindowedSort<256, PermFormat::ChunkRelativeZstd>(data);
    checkWindowedSort<256, PermFormat::ChunkRelativeLZ4>(data);
    checkWindowedSort<4096, PermFormat::ChunkRelative>(data);

    // A sequential inner codec: no inner random access, so every override falls
    // back to inverting the whole stream and must still produce the right rows.
    {
        auto codec = makeReorderingCodec<Elem, true>(
            std::make_shared<SortReorderer<Elem>>(PermFormat::FlatBitPacked),
            std::make_shared<ZstdEncoder<Elem>>(), ReorderingType::Sort);
        checkCodec(*codec, data, "Sort[FlatBitPacked]|Zstd");
        checkAgainstFallback(*codec, data, "Sort[FlatBitPacked]|Zstd");
    }

    // Value transforms: no position permutation at all, so these exercise the
    // whole-stream inversion path of all three overrides.
    {
        auto codec = makeReorderingCodec<Elem, true>(
            std::make_shared<BWTReorderer<Elem, 256>>(),
            std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::BWT);
        checkCodec(*codec, data, "BWT<256>");
        checkAgainstFallback(*codec, data, "BWT<256>");
    }
    {
        auto codec = makeReorderingCodec<Elem, true>(
            std::make_shared<MTFReorderer<Elem, 256>>(),
            std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::MTF);
        checkCodec(*codec, data, "MTF<256>");
        checkAgainstFallback(*codec, data, "MTF<256>");
    }

    // Non-profiling instantiation: the overrides must not depend on the profiling
    // branches that only exist in ReorderingCodec<T, true>.
    {
        auto codec = makeReorderingCodec<Elem, false>(
            std::make_shared<SortReorderer<Elem>>(PermFormat::FlatBitPacked),
            std::make_shared<RawBitPackedEncoder<Elem>>(), ReorderingType::Sort);
        checkCodec(*codec, data, "Sort[FlatBitPacked] (no profiling)");
    }

    checkInternalBuffers("internalBuffers/per-call", PermResidency::PerCall);
    checkInternalBuffers("internalBuffers/resident", PermResidency::Resident);

    if (failures != 0) {
        std::cerr << failures << " check(s) FAILED\n";
        return 1;
    }
    std::cout << "ALL PASS\n";
    return 0;
}
