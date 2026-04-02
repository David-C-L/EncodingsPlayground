#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <type_traits>

#include "encoders/selectors/BitRangeSegmentBuilder.hpp"
#include "encoders/selectors/MetricCollector.hpp"
#include "encoders/selectors/costs/EncodingCostModel.hpp"
#include "encoders/BitSplitOrder.hpp" // for BitSplitOrder
#include "encodings/EncodingType.hpp"

namespace encodings::encoders::selectors {

struct SegmentPlan {
	int bitStart{0};
	int bitEnd{0};
	encodings::EncodingType encoding{encodings::EncodingType::RawEncoding};
	double cost{0.0};


    std::string toString() const {
        std::ostringstream oss;
        oss << "[" << bitStart << ".." << bitEnd << "] "
            << encodings::encodingTypeToString(encoding)
            << " cost=" << std::fixed << std::setprecision(2) << cost;
        return oss.str();
    }
};


class IDSubStreamEncodingSelector {
public:
	struct Result {
		std::vector<SegmentPlan> segments;
		double total_cost{0.0};


        std::string toString() const {
            std::ostringstream oss;
            oss << "IDSubStreamEncodingSelector Result\n";
            oss << "  total_cost: " << std::fixed << std::setprecision(2) << total_cost << "\n";
            oss << "  segments (" << segments.size() << "):\n";
            for (const auto& seg : segments) {
                oss << "    - " << seg.toString() << "\n";
            }
            return oss.str();
        }
	};

	struct Config {
		int minSegmentWidth{1};
		double splitPenalty{0.0};
		bool enablePrune{false};
		double entropyPruneThreshold{1.0};
		int verboseLevel{0}; // 0=off, 1=final DP choices, 2=full trace
		bool enableMergePhase{false}; // optional post-pass to merge adjacent segments
		std::optional<encodings::encoders::BitSplitOrder> orderHint{}; // optional downstream bit-order preference
		bool useExhaustiveSearch{false}; // when true, enumerate all split combinations instead of DP
		std::optional<std::string> costGridCsvPath{}; // optional: write per-encoding bits/elem grid
	};

	static constexpr int kBits = 64;

	IDSubStreamEncodingSelector() = default;
	explicit IDSubStreamEncodingSelector(const Config& config) : config_(config) {}

	template <typename SampleT>
		requires std::is_integral_v<SampleT>
	Result select(
		const std::vector<SampleT>& sample,
		const std::vector<std::unique_ptr<encoders::selectors::costs::EncodingCostModel>>& encodings,
		std::optional<size_t> fullCount = std::nullopt
	) const {
		if (sample.empty()) {
			return {};
		}
		if (encodings.empty()) {
			throw std::invalid_argument("IDSubStreamEncodingSelector::select: no encodings provided");
		}

		const size_t effectiveCount = fullCount.value_or(sample.size());

		BitRangeSegmentBuilder<SampleT, uint64_t> builder(sample);
		MetricCollector<uint64_t> collector;

		// Keep encoding names for CSV output
		std::vector<std::string> encodingNames;
		encodingNames.reserve(encodings.size());
		for (const auto& enc : encodings) {
			encodingNames.push_back(encodings::encodingTypeToString(enc->encodingType()));
		}

		std::array<std::array<SegmentChoice, kBits>, kBits> bestCost{};
		for (int l = 0; l < kBits; ++l) {
			for (int r = 0; r < kBits; ++r) {
				bestCost[l][r].cost = std::numeric_limits<double>::infinity();
			}
		}

		if (config_.verboseLevel >= 2) {
			std::cout << "[Selector] Evaluating segments (l..r)" << std::endl;
		}
		// Precompute cost per segment (min across encodings).
		// Optional per-encoding cost grid (bits/elem) for heatmap
		std::vector<std::vector<double>> perEncodingCostGrid;
		if (config_.costGridCsvPath.has_value()) {
			perEncodingCostGrid.assign(kBits * kBits, std::vector<double>(encodings.size(), std::numeric_limits<double>::infinity()));
		}

		for (int l = 0; l < kBits; ++l) {
			builder.reset(l);
			for (int r = l; r < kBits; ++r) {
				builder.extend(r);
				const auto& values = builder.values();
				SegmentMetrics metrics = collector.compute(values);
				const size_t numValues = values.size();
				const size_t bitWidth = static_cast<size_t>(r - l + 1);

				if (config_.enablePrune) {
					if (metrics.entropyEstimate > config_.entropyPruneThreshold) {
						continue;
					}
					if (bitWidth < 64) {
						if (metrics.range >= ((uint64_t{1} << bitWidth) - 1)) {
							continue;
						}
					}
				}

				double bestEncodingCost = std::numeric_limits<double>::infinity();
				encodings::EncodingType bestEncoding = encodings::EncodingType::RawEncoding;
				if (config_.verboseLevel >= 2) {
					std::cout << "  Segment [" << l << ".." << r << "] width=" << bitWidth << " bits" << std::endl;
				}
				for (size_t encIdx = 0; encIdx < encodings.size(); ++encIdx) {
					const auto& encoding = encodings[encIdx];
					const double perSampleCost = encoding->computeCost(metrics, numValues, bitWidth);
					const double totalCost = perSampleCost * static_cast<double>(effectiveCount) / static_cast<double>(numValues);
					const double bitsPerElem = totalCost / static_cast<double>(effectiveCount);
					if (config_.verboseLevel >= 2) {
						std::cout << "    - " << encodingNames[encIdx]
								  << " cost=" << std::fixed << std::setprecision(2) << totalCost
								  << " bits/elem=" << std::fixed << std::setprecision(4) << bitsPerElem
								  << std::endl;
					}
					if (config_.costGridCsvPath.has_value()) {
						perEncodingCostGrid[l * kBits + r][encIdx] = bitsPerElem;
					}
					if (totalCost < bestEncodingCost) {
						bestEncodingCost = totalCost;
						bestEncoding = encoding->encodingType();
					}
				}
				if (bestEncodingCost < bestCost[l][r].cost) {
					bestCost[l][r].cost = bestEncodingCost;
					bestCost[l][r].encoding = bestEncoding;
				}
				if (config_.verboseLevel >= 2 && std::isfinite(bestCost[l][r].cost)) {
					std::cout << "    -> best=" << encodings::encodingTypeToString(bestCost[l][r].encoding)
							  << " cost=" << std::fixed << std::setprecision(2) << bestCost[l][r].cost
							  << std::endl;
				}
			}
		}

		Result result;

		if (config_.useExhaustiveSearch) {
			if (config_.verboseLevel >= 1) {
				std::cout << "[Selector] Running exhaustive search over split combinations" << std::endl;
			}
			result = runExhaustiveSearch(bestCost);
		} else {
			result = runDynamicProgramming(bestCost);
		}

		// Emit optional cost grid CSV: columns = start,end,width,<encodings...>
		if (config_.costGridCsvPath.has_value()) {
			std::ofstream csv(*config_.costGridCsvPath);
			if (!csv) {
				throw std::runtime_error("IDSubStreamEncodingSelector: failed to open costGridCsvPath for writing");
			}
			csv << "start,end,width";
			for (const auto& name : encodingNames) {
				csv << ',' << name;
			}
			csv << '\n';
			for (int l = 0; l < kBits; ++l) {
				for (int r = 0; r < kBits; ++r) {
					const size_t idx = l * kBits + r;
					const int width = (r >= l) ? (r - l + 1) : 0;
					csv << l << ',' << r << ',' << width;
					for (double v : perEncodingCostGrid[idx]) {
						if (std::isfinite(v)) {
							csv << ',' << v;
						} else {
							csv << ',';
						}
					}
					csv << '\n';
				}
			}
		}

		// Optional merge phase to escape local optima
		if (config_.enableMergePhase && result.segments.size() > 1) {
			tryMergeAdjacentSegments(sample, encodings, effectiveCount, result);
		}

		if (config_.verboseLevel >= 1) {
			std::cout << result.toString();
		}

		return result;
	}

private:
	template <typename SampleT>
	void tryMergeAdjacentSegments(
		const std::vector<SampleT>& sample,
		const std::vector<std::unique_ptr<encoders::selectors::costs::EncodingCostModel>>& encodings,
		size_t effectiveCount,
		Result& result) const {
		BitRangeSegmentBuilder<SampleT, uint64_t> mergeBuilder(sample);
		MetricCollector<uint64_t> mergeCollector;
		bool merged_any = true;
		while (merged_any) {
			merged_any = false;
			std::vector<SegmentPlan> newSegments;
			size_t iSeg = 0;
			while (iSeg < result.segments.size()) {
				if (iSeg + 1 < result.segments.size()) {
					const auto& A = result.segments[iSeg];
					const auto& B = result.segments[iSeg + 1];
					const int l = A.bitStart;
					const int r = B.bitEnd;
					const size_t mergedWidth = static_cast<size_t>(r - l + 1);

					mergeBuilder.reset(l);
					for (int bit = l; bit <= r; ++bit) {
						mergeBuilder.extend(bit);
					}
					const auto& values = mergeBuilder.values();
					SegmentMetrics metrics = mergeCollector.compute(values);
					const size_t numValues = values.size();

					double bestMergedCost = std::numeric_limits<double>::infinity();
					encodings::EncodingType bestEncoding = encodings::EncodingType::RawEncoding;
					for (const auto& enc : encodings) {
						double cost = enc->computeCost(metrics, numValues, mergedWidth);
						cost *= static_cast<double>(effectiveCount) / static_cast<double>(numValues);
						if (cost < bestMergedCost) {
							bestMergedCost = cost;
							bestEncoding = enc->encodingType();
						}
					}

					const double splitCost = A.cost + B.cost + config_.splitPenalty;
                    if (config_.verboseLevel >= 1) {
                        std::cout << "[Selector][Merge Phase] Evaluating merge of segments [" << A.bitStart << ".." << A.bitEnd << "] and ["
                                  << B.bitStart << ".." << B.bitEnd << "] into [" << l << ".." << r << "] with encoding "
                                  << encodings::encodingTypeToString(bestEncoding) << " cost=" << std::fixed
                                  << std::setprecision(2) << bestMergedCost << " bits/elem=" << (bestMergedCost / effectiveCount)
                                  << " vs split cost=" << std::fixed << std::setprecision(2) << splitCost
                                  << " bits/elem=" << (splitCost / effectiveCount) << std::endl;
                    }

					if (bestMergedCost < splitCost) {
						SegmentPlan merged;
						merged.bitStart = l;
						merged.bitEnd = r;
						merged.encoding = bestEncoding;
						merged.cost = bestMergedCost;
						newSegments.push_back(std::move(merged));
						iSeg += 2;
						merged_any = true;
						continue;
					}
				}
				newSegments.push_back(result.segments[iSeg]);
				++iSeg;
			}
			result.segments = std::move(newSegments);
		}

		// Recompute total cost with split penalties after merges
		double total = 0.0;
		for (const auto& seg : result.segments) {
			total += seg.cost;
		}
		total += config_.splitPenalty * static_cast<double>(result.segments.size() > 0 ? (result.segments.size() - 1) : 0);
		result.total_cost = total;
	}

	struct SegmentChoice {
		double cost{std::numeric_limits<double>::infinity()};
		encodings::EncodingType encoding{encodings::EncodingType::RawEncoding};
	};

	// Standard DP reconstruction helper
	Result runDynamicProgramming(const std::array<std::array<SegmentChoice, kBits>, kBits>& bestCost) const {
		std::array<double, kBits + 1> dp{};
		std::array<int, kBits + 1> prev{};
		std::array<encodings::EncodingType, kBits + 1> chosen{};
		for (int i = 0; i <= kBits; ++i) {
			dp[i] = std::numeric_limits<double>::infinity();
			prev[i] = -1;
			chosen[i] = encodings::EncodingType::RawEncoding;
		}
		dp[0] = 0.0;

		if (config_.verboseLevel >= 2) {
			std::cout << "[Selector] Running DP" << std::endl;
		}
		for (int i = 1; i <= kBits; ++i) {
			for (int j = 0; j < i; ++j) {
				const int width = i - j;
				if (width < config_.minSegmentWidth) {
					continue;
				}
				const auto& choice = bestCost[j][i - 1];
				if (!std::isfinite(choice.cost)) {
					continue;
				}
				const double cost = dp[j] + choice.cost + (j == 0 ? 0.0 : config_.splitPenalty);
				if (config_.verboseLevel >= 2) {
					std::cout << "  dp[" << i << "] candidate: split at " << j
							  << " (segment [" << j << ".." << (i - 1) << "] "
							  << encodings::encodingTypeToString(choice.encoding)
							  << ") cost=" << std::fixed << std::setprecision(2) << cost
							  << std::endl;
				}
				if (cost < dp[i]) {
					dp[i] = cost;
					prev[i] = j;
					chosen[i] = choice.encoding;
					if (config_.verboseLevel >= 2) {
						std::cout << "    -> dp[" << i << "] updated: cost="
								  << std::fixed << std::setprecision(2) << dp[i]
								  << " via split " << j << std::endl;
					}
				}
			}
			if (config_.verboseLevel >= 1 && std::isfinite(dp[i])) {
				std::cout << "[Selector] dp[" << i << "] choose split " << prev[i]
						  << " (segment [" << prev[i] << ".." << (i - 1) << "] "
						  << encodings::encodingTypeToString(chosen[i])
						  << ") cost=" << std::fixed << std::setprecision(2) << dp[i]
						  << std::endl;
			}
		}

		Result result;
		result.total_cost = dp[kBits];
		if (!std::isfinite(result.total_cost)) {
			return result;
		}

		// Backtrack
		int idx = kBits;
		while (idx > 0) {
			const int start = prev[idx];
			if (start < 0) {
				break;
			}
			SegmentPlan plan;
			plan.bitStart = start;
			plan.bitEnd = idx - 1;
			plan.encoding = chosen[idx];
			plan.cost = bestCost[start][idx - 1].cost;
			result.segments.push_back(plan);
			idx = start;
		}

		std::reverse(result.segments.begin(), result.segments.end());
		return result;
	}

	Result runExhaustiveSearch(const std::array<std::array<SegmentChoice, kBits>, kBits>& bestCost) const {
		struct CacheEntry {
			bool valid{false};
			double cost{0.0};
			std::vector<SegmentPlan> segments;
		};

		std::array<CacheEntry, kBits + 1> memo{};

		std::function<CacheEntry(int)> dfs = [&](int start) -> CacheEntry {
			if (memo[start].valid) return memo[start];
			CacheEntry best;
			best.valid = true;
			best.cost = std::numeric_limits<double>::infinity();

			if (start == kBits) {
				best.cost = 0.0;
				return memo[start] = best;
			}

			for (int end = start; end < kBits; ++end) {
				const int width = end - start + 1;
				if (width < config_.minSegmentWidth) continue;
				const auto& choice = bestCost[start][end];
				if (!std::isfinite(choice.cost)) continue;

				const auto suffix = dfs(end + 1);
				if (!std::isfinite(suffix.cost)) continue;

				double total = choice.cost + suffix.cost;
				if (end + 1 <= kBits - 1) {
					total += config_.splitPenalty; // apply penalty between segments
				}
				if (total < best.cost) {
					best.cost = total;
					best.segments.clear();
					SegmentPlan seg;
					seg.bitStart = start;
					seg.bitEnd = end;
					seg.encoding = choice.encoding;
					seg.cost = choice.cost;
					best.segments.push_back(seg);
					best.segments.insert(best.segments.end(), suffix.segments.begin(), suffix.segments.end());
				}
			}

			return memo[start] = best;
		};

		CacheEntry solution = dfs(0);
		Result result;
		result.total_cost = solution.cost;
		result.segments = std::move(solution.segments);
		return result;
	}

	Config config_{};
};

} // namespace encodings::encoders::selectors
