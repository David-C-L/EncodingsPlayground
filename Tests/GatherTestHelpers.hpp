// Shared test helper for verifying Codec::decodeGatherInto() correctness
// against decodeAll(), across a set of representative RowRangeList traces.
#pragma once

#include <cassert>
#include <iostream>
#include <vector>

#include "encodings/Encoder.hpp"
#include "encodings/RowRange.hpp"
#include "benchmark/SelectiveTraceGen.hpp"

namespace encodings::testutil {

// Roundtrip-checks decodeGatherInto against decodeAll() sliced by the same ranges.
// Returns true on match; prints a diagnostic and returns false on mismatch.
template <typename T>
bool checkGatherMatchesDecodeAll(encodings::Codec<T>& codec,
                                  const encodings::EncodedBuffer<uint8_t>& encoded,
                                  const std::vector<T>& original,
                                  const encodings::RowRangeList& ranges,
                                  const std::string& label) {
    size_t total = 0;
    for (const auto& r : ranges) total += r.size();

    std::vector<T> dst(total);
    codec.decodeGatherInto(encoded, ranges, dst.data(), total);

    size_t off = 0;
    for (const auto& r : ranges) {
        for (size_t i = r.begin; i < r.end; ++i, ++off) {
            if (dst[off] != original[i]) {
                std::cerr << "FAIL decodeGatherInto [" << label << "]: mismatch at row " << i
                          << " (dst[" << off << "]=" << dst[off]
                          << ", expected=" << original[i] << ")\n";
                return false;
            }
        }
    }
    return true;
}

// Runs checkGatherMatchesDecodeAll across a standard set of representative
// trace shapes: empty, single full range, many small scattered ranges, and a
// boundary-touching pair. Returns true iff all traces pass.
template <typename T>
bool checkGatherAllTraceShapes(encodings::Codec<T>& codec,
                                const encodings::EncodedBuffer<uint8_t>& encoded,
                                const std::vector<T>& original) {
    const size_t n = original.size();
    bool ok = true;

    ok &= checkGatherMatchesDecodeAll(codec, encoded, original, encodings::RowRangeList{}, "empty");

    ok &= checkGatherMatchesDecodeAll(codec, encoded, original,
        encodings::RowRangeList{{0, n}}, "full");

    ok &= checkGatherMatchesDecodeAll(codec, encoded, original,
        encodings::benchmark::makeSelectiveTrace(n, {.selectivity = 0.3, .meanRunLength = 2.0}),
        "scattered");

    if (n >= 2) {
        ok &= checkGatherMatchesDecodeAll(codec, encoded, original,
            encodings::RowRangeList{{0, 1}, {n - 1, n}}, "boundary");
    }

    return ok;
}

}  // namespace encodings::testutil
