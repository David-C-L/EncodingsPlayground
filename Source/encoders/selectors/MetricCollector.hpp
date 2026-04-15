#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>

namespace encodings::encoders::selectors {

// Bitset flags declaring which metric groups a cost model requires.
// IDSubStreamEncodingSelector unions the flags from all registered cost models
// and passes the result to MetricCollector::compute() to skip unused work.
enum class MetricFlag : uint32_t {
    None           = 0,
    MinMax         = 1u << 0,  // min, max, range
    RunStats       = 1u << 1,  // runCount, avgRunLength
    FreqStats      = 1u << 2,  // uniqueCount, hllEstimatedCardinality, f1, f2, entropyEstimate
    SuffixFrames   = 1u << 3,  // frameMaxSuffixBits, frameAvgSuffixBits
    ResidualFrames = 1u << 4,  // frameMaxResidualBits, frameAvgResidualBits
    All            = (1u << 5) - 1,
};
using MetricFlags = uint32_t;

inline constexpr MetricFlags operator|(MetricFlag a, MetricFlag b) noexcept {
    return static_cast<MetricFlags>(a) | static_cast<MetricFlags>(b);
}
inline constexpr MetricFlags operator|(MetricFlags a, MetricFlag b) noexcept {
    return a | static_cast<MetricFlags>(b);
}
inline constexpr bool hasFlag(MetricFlags flags, MetricFlag f) noexcept {
    return (flags & static_cast<MetricFlags>(f)) != 0;
}

struct SegmentMetrics {
	uint64_t min{0};
	uint64_t max{0};
	uint64_t range{0};

	size_t uniqueCount{0};
	bool uniqueCountCapped{false};
	size_t runCount{0};

	double avgRunLength{0.0};
	// Tail-frequency support
	size_t f1{0}; // values seen exactly once
	size_t f2{0}; // values seen exactly twice

	// HLL cardinality estimate precomputed once in MetricCollector::compute().
	// Replaces the raw register vector — callers never need the registers themselves.
	double hllEstimatedCardinality{0.0};

	// BitPrefix frames (small): sizes {8,16,32,64,128} — suffix-width stats only.
	// Residual  frames (large): sizes {256,512,1024,2048,4096} — residual-width stats only.
	// The two sets are kept separate because BitPrefix benefits from small frames
	// (shared prefix bits are stable inside short windows) while FOR needs large frames
	// (amortises the per-frame reference overhead).
	static constexpr size_t kBitPrefixFrameCandidateCount = 5;
	static constexpr size_t kResidualFrameCandidateCount  = 5;

	std::array<uint8_t, kResidualFrameCandidateCount>  frameMaxResidualBits{};
	std::array<double,  kResidualFrameCandidateCount>  frameAvgResidualBits{};
	std::array<uint8_t, kBitPrefixFrameCandidateCount> frameMaxSuffixBits{};
	std::array<double,  kBitPrefixFrameCandidateCount> frameAvgSuffixBits{};

	// Optional
	double entropyEstimate{0.0};
};

// MAY WANT INPUT CONFIG TO SELECT WINDOW SIZE AND UNIQUE CAP
template <typename T = uint64_t>
    requires std::is_integral_v<T>
class MetricCollector {
public:
	static constexpr size_t kUniqueCountCap = 1 << 16;
	// Precision-10 HLL: 1024 registers, standard error ~3.25%.
	static constexpr uint8_t kHllPrecision = 10;

	// Frame candidate sets — all powers of 2 so the countr_zero trick applies.
	// BitPrefix: 2^3..2^7  (countr_zero offset = 3)
	// Residual:  2^8..2^12 (countr_zero offset = 8)
	inline static constexpr std::array<size_t, SegmentMetrics::kBitPrefixFrameCandidateCount>
		kBitPrefixFrameCandidates{ 8, 16, 32, 64, 128 };
	inline static constexpr std::array<size_t, SegmentMetrics::kResidualFrameCandidateCount>
		kResidualFrameCandidates{ 256, 512, 1024, 2048, 4096 };

private:
	static constexpr size_t kBpN   = SegmentMetrics::kBitPrefixFrameCandidateCount;
	static constexpr size_t kResN  = SegmentMetrics::kResidualFrameCandidateCount;
	static constexpr size_t kNumRegisters = 1u << kHllPrecision;

	// -------------------------------------------------------------------------
	// Frame tracking state structs — one per set.
	// Fields without member-initializers are set by initFromFirst() before use.
	// -------------------------------------------------------------------------

	struct SuffixFrameState {
		std::array<uint64_t, kBpN>  fmin;
		std::array<uint64_t, kBpN>  fmax;
		std::array<uint64_t, kBpN>  fxor;
		std::array<size_t,   kBpN>  fcount;
		// Accumulated suffix stats — zero-initialized.
		std::array<uint8_t,  kBpN>  incrMaxSuffix{};
		std::array<double,   kBpN>  incrSumWeightedSuffix{};
		std::array<double,   kBpN>  incrTotalVals{};

		void initFromFirst(uint64_t uv) noexcept {
			fmin.fill(uv); fmax.fill(uv); fxor.fill(0); fcount.fill(1);
		}
	};

	struct ResidualFrameState {
		std::array<uint64_t, kResN> fmin;
		std::array<uint64_t, kResN> fmax;
		std::array<size_t,   kResN> fcount;
		// Accumulated residual stats — zero-initialized.
		std::array<double,   kResN> fbitsum{};
		std::array<size_t,   kResN> fbitseen{};

		void initFromFirst(uint64_t uv) noexcept {
			fmin.fill(uv); fmax.fill(uv); fcount.fill(1);
		}
	};

	// -------------------------------------------------------------------------
	// Hot-path helpers — named so they appear as distinct scopes in profiler
	// call trees even when inlined (source-level attribution via debug info).
	// -------------------------------------------------------------------------

	// HLL register update: branchless max instead of conditional store.
	static void hllAdd(uint8_t* __restrict__ regs, uint64_t hash) {
		const uint32_t idx = static_cast<uint32_t>(hash & (kNumRegisters - 1));
		const uint64_t w   = hash >> kHllPrecision;
		const uint8_t rank = (w == 0)
			? static_cast<uint8_t>(64 - kHllPrecision + 1)
			: static_cast<uint8_t>(std::countl_zero(w) + 1);
		regs[idx] = std::max(regs[idx], rank);
	}

	// Frequency-map update with f1/f2 maintenance.
	static void updateFreq(
		T v,
		ankerl::unordered_dense::map<T, uint32_t>& freqMap,
		bool& uniqueCapped,
		size_t& f1,
		size_t& f2)
	{
		auto [it, inserted] = freqMap.emplace(v, 1);
		if (inserted) {
			++f1;
		} else {
			if (it->second == 1) { --f1; ++f2; }
			else if (it->second == 2) { --f2; }
			++it->second;
		}
		if (freqMap.size() > kUniqueCountCap) uniqueCapped = true;
	}

	// Finalize one suffix (BitPrefix) frame: compute shared-prefix length over the
	// accumulated XOR, derive suffix bits, accumulate stats, reset state.
	static void finalizeSuffixFrame(
		size_t c,
		SuffixFrameState& s,
		uint64_t curMax)
	{
		const size_t fcount = s.fcount[c];
		if (fcount == 0) return;

		const uint8_t curBitWidth = static_cast<uint8_t>(std::bit_width(curMax));
		const uint64_t xorv = s.fxor[c];
		uint8_t prefixBits;
		if (xorv == 0 || curBitWidth == 0) {
			prefixBits = curBitWidth;
		} else if (curBitWidth >= 64) {
			prefixBits = static_cast<uint8_t>(std::countl_zero(xorv));
		} else {
			prefixBits = std::min<uint8_t>(curBitWidth,
				static_cast<uint8_t>(std::countl_zero(xorv << (64u - curBitWidth))));
		}
		const uint8_t suffixBits = (curBitWidth > prefixBits)
			? static_cast<uint8_t>(curBitWidth - prefixBits) : 0;

		s.incrMaxSuffix[c] = std::max(s.incrMaxSuffix[c], suffixBits);
		s.incrSumWeightedSuffix[c] += static_cast<double>(suffixBits) * static_cast<double>(fcount);
		s.incrTotalVals[c]         += static_cast<double>(fcount);

		s.fcount[c] = 0;
		s.fmin[c]   = std::numeric_limits<uint64_t>::max();
		s.fmax[c]   = std::numeric_limits<uint64_t>::min();
		// fxor resets implicitly: next element sets fmin=fmax=uv → xor=0.
	}

	// Finalize one residual (FOR-style) frame: compute span width, accumulate stats,
	// reset state.
	static void finalizeResidualFrame(
		size_t c,
		ResidualFrameState& s,
		SegmentMetrics& out)
	{
		const size_t fcount = s.fcount[c];
		if (fcount == 0) return;

		const uint64_t span = s.fmax[c] - s.fmin[c];
		const uint8_t bits  = span == 0 ? 0 : static_cast<uint8_t>(std::bit_width(span));
		out.frameMaxResidualBits[c] = std::max(out.frameMaxResidualBits[c], bits);
		s.fbitsum[c]  += static_cast<double>(bits);
		++s.fbitseen[c];

		s.fcount[c] = 0;
		s.fmin[c]   = std::numeric_limits<uint64_t>::max();
		s.fmax[c]   = std::numeric_limits<uint64_t>::min();
	}

	// Per-element update for all BitPrefix frames, plus conditional finalization.
	// Frame c (size 2^(3+c)) finalizes when countr_zero(i+1) >= 3+c.
	static void updateSuffixFrames(
		uint64_t uv,
		SuffixFrameState& s,
		size_t i,
		uint64_t curMax)
	{
		for (size_t c = 0; c < kBpN; ++c) {
			if (uv < s.fmin[c]) s.fmin[c] = uv;
			if (uv > s.fmax[c]) s.fmax[c] = uv;
			s.fxor[c] = s.fmax[c] ^ s.fmin[c];
			++s.fcount[c];
		}
		const int tz = static_cast<int>(std::countr_zero(i + 1));
		if (tz >= 3) {
			const int maxC = std::min(tz - 3, static_cast<int>(kBpN) - 1);
			for (int c = 0; c <= maxC; ++c)
				finalizeSuffixFrame(static_cast<size_t>(c), s, curMax);
		}
	}

	// Per-element update for all Residual frames, plus conditional finalization.
	// Frame c (size 2^(8+c)) finalizes when countr_zero(i+1) >= 8+c.
	static void updateResidualFrames(
		uint64_t uv,
		ResidualFrameState& s,
		SegmentMetrics& out,
		size_t i)
	{
		for (size_t c = 0; c < kResN; ++c) {
			if (uv < s.fmin[c]) s.fmin[c] = uv;
			if (uv > s.fmax[c]) s.fmax[c] = uv;
			++s.fcount[c];
		}
		const int tz = static_cast<int>(std::countr_zero(i + 1));
		if (tz >= 8) {
			const int maxC = std::min(tz - 8, static_cast<int>(kResN) - 1);
			for (int c = 0; c <= maxC; ++c)
				finalizeResidualFrame(static_cast<size_t>(c), s, out);
		}
	}

	// -------------------------------------------------------------------------
	// Cold-path post-processing helpers.
	// -------------------------------------------------------------------------

	// Compute HLL cardinality estimate from the register array.
	// Called once per compute() after the full pass.
	static double computeHllEstimate(const uint8_t* regs) {
		static constexpr auto kPow2Neg = []() constexpr {
			std::array<double, 65> t{};
			t[0] = 1.0;
			for (int i = 1; i < 65; ++i) t[i] = t[i - 1] * 0.5;
			return t;
		}();
		double sum = 0.0;
		size_t zeros = 0;
		for (size_t k = 0; k < kNumRegisters; ++k) {
			const uint8_t r = regs[k];
			if (r == 0) ++zeros;
			sum += kPow2Neg[r];
		}
		constexpr double m = static_cast<double>(kNumRegisters);
		const double alpha = 0.7213 / (1.0 + 1.079 / m);
		double estimate = alpha * m * m / sum;
		if (estimate <= 2.5 * m && zeros > 0)
			estimate = m * std::log(m / static_cast<double>(zeros));
		return estimate;
	}

	// Compute entropy estimate from the frequency map.
	// Called once per compute() after the full pass.
	static double computeEntropy(
		const ankerl::unordered_dense::map<T, uint32_t>& freqMap,
		size_t n,
		bool uniqueCapped)
	{
		if (uniqueCapped) {
			// Lower bound on entropy from observed unique count.
			const double p = static_cast<double>(kUniqueCountCap) / static_cast<double>(n);
			return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
		}
		if (freqMap.empty()) return 0.0;
		double entropy = 0.0;
		const double total = static_cast<double>(n);
		for (const auto& [key, count] : freqMap) {
			(void)key;
			const double p = static_cast<double>(count) / total;
			if (p > 0.0) entropy -= p * std::log2(p);
		}
		return entropy;
	}

public:
	// Compute metrics for the given values. Only the metric groups indicated by
	// `flags` are computed; all others are left at their zero-initialised defaults.
	// This allows callers that have a fixed set of cost models to skip expensive
	// work (frequency-map construction, HLL hashing, frame tracking) when the
	// corresponding metrics are not needed.
	SegmentMetrics compute(
		const std::vector<T>& values,
		MetricFlags flags = static_cast<MetricFlags>(MetricFlag::All)) const
	{
		const bool doRun      = hasFlag(flags, MetricFlag::RunStats);
		const bool doFreq     = hasFlag(flags, MetricFlag::FreqStats);
		const bool doSuffix   = hasFlag(flags, MetricFlag::SuffixFrames);
		const bool doResidual = hasFlag(flags, MetricFlag::ResidualFrames);

		SegmentMetrics out;
		if (values.empty()) return out;

		const size_t n = values.size();

		// Stack HLL registers (1024 bytes — fits in L1 cache).
		uint8_t hllRegs[kNumRegisters]{};

		// --- Prolog: initialise from values[0], eliminating if(i>0) from the hot loop ---
		const T        v0  = values[0];
		const uint64_t uv0 = static_cast<uint64_t>(v0);
		out.min = uv0;
		out.max = uv0;

		if (doRun) {
			out.runCount = 1;
		}

		ankerl::unordered_dense::map<T, uint32_t> freqMap;
		bool uniqueCapped = false;
		if (doFreq) {
			hllAdd(hllRegs, std::hash<T>{}(v0));
			freqMap.reserve(std::min(n, kUniqueCountCap));
			freqMap[v0] = 1;
			out.f1 = 1;
		}

		SuffixFrameState  sfx{};
		if (doSuffix)   sfx.initFromFirst(uv0);

		ResidualFrameState res{};
		if (doResidual) res.initFromFirst(uv0);

		T      prev             = v0;
		size_t currentRunLength = 1;

		// --- Main loop (i=0 already handled in prolog) ---
		for (size_t i = 1; i < n; ++i) {
			const T        v  = values[i];
			const uint64_t uv = static_cast<uint64_t>(v);

			if (uv < out.min) out.min = uv;
			if (uv > out.max) out.max = uv;

			if (doRun) {
				if (v == prev) {
					++currentRunLength;
				} else {
					++out.runCount;
					currentRunLength = 1;
				}
			}

			if (doFreq) {
				hllAdd(hllRegs, std::hash<T>{}(v));
				if (!uniqueCapped) updateFreq(v, freqMap, uniqueCapped, out.f1, out.f2);
			}

			if (doSuffix)   updateSuffixFrames(uv, sfx, i, out.max);
			if (doResidual) updateResidualFrames(uv, res, out, i);

			prev = v;
		}

		// --- Flush any partial frames ---
		if (doSuffix) {
			for (size_t c = 0; c < kBpN;  ++c) finalizeSuffixFrame(c, sfx, out.max);
		}
		if (doResidual) {
			for (size_t c = 0; c < kResN; ++c) finalizeResidualFrame(c, res, out);
		}

		// --- Commit per-frame averages ---
		if (doSuffix) {
			for (size_t c = 0; c < kBpN; ++c) {
				out.frameMaxSuffixBits[c] = sfx.incrMaxSuffix[c];
				out.frameAvgSuffixBits[c] = (sfx.incrTotalVals[c] > 0.0)
					? sfx.incrSumWeightedSuffix[c] / sfx.incrTotalVals[c] : 0.0;
			}
		}
		if (doResidual) {
			for (size_t c = 0; c < kResN; ++c) {
				if (res.fbitseen[c] > 0)
					out.frameAvgResidualBits[c] = res.fbitsum[c] / static_cast<double>(res.fbitseen[c]);
			}
		}

		out.range = out.max - out.min;

		if (doRun) {
			out.avgRunLength = static_cast<double>(n) / static_cast<double>(out.runCount);
		}

		if (doFreq) {
			out.uniqueCount         = uniqueCapped ? (kUniqueCountCap + 1) : freqMap.size();
			out.uniqueCountCapped   = uniqueCapped;
			out.entropyEstimate     = computeEntropy(freqMap, n, uniqueCapped);
			out.hllEstimatedCardinality = computeHllEstimate(hllRegs);
		}

		return out;
	}
};

} // namespace encodings::encoders::selectors
