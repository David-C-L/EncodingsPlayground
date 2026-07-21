#pragma once

#include <cstddef>
#include <vector>

namespace encodings {

/**
 * @brief A half-open row range [begin, end), mirroring Nimble's RowRange.
 *
 * Used to describe the surviving rows of a TableScan-style filtered read:
 * an ascending, non-overlapping list of contiguous runs with gaps between
 * them that must be skipped rather than materialized.
 */
struct RowRange {
    size_t begin{0};
    size_t end{0};  // exclusive

    size_t size() const { return end > begin ? end - begin : 0; }
};

/// Ascending, non-overlapping list of surviving row ranges (ranges[i].end <=
/// ranges[i+1].begin), as produced by a TableScan filter's active-ranges
/// narrowing.
using RowRangeList = std::vector<RowRange>;

}  // namespace encodings
