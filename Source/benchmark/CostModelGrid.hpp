#pragma once

// Estimated-vs-actual for every (bit range, encoding): the cost model's
// prediction next to the oracle grid's measured byte count.
//
// This is the memoized replacement for explore_best_encoding.cpp's
// computeCostModelAccuracy(), which was called once per segment PER CALLER and
// each time re-extracted the section values, re-ran MetricCollector::compute()
// over the whole sample with MetricFlag::All, and rebuilt the entire cost-model
// vector from scratch.  With three callers over a ~6-segment plan that is ~18
// full metric passes and ~18 model-vector constructions where 6 of each are
// needed, and the uSample copy was itself rebuilt in three places.
//
// So: the sample is passed in ONCE by reference, the models are built ONCE in
// the constructor, and at(l, r) memoizes per bit range.  Callers become pure
// readers of that cache.

#include "benchmark/OracleGrid.hpp"
#include "encoders/SubIntSplitEncoder.hpp"
#include "encoders/selectors/IDSubStreamEncodingSelector.hpp"
#include "encoders/selectors/MetricCollector.hpp"
#include "encoders/selectors/costs/CostModelSet.hpp"
#include "encodings/EncodingType.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace encodings::benchmark {

/// One (bit range, encoding) cell of the accuracy table.
///
/// `hasCostModel` is an explicit field rather than a filter applied before the
/// caller sees the row: OpenZL and the Cascading* compositions are in the
/// oracle's candidate set but not in CostModelSet::defaultEncodings(), and a
/// silently dropped row makes the model look better than it is by hiding the
/// candidates it cannot even rank.  `estBits` is NaN exactly when hasCostModel
/// is false.
struct CellEstimate {
    EncodingType enc{EncodingType::RawEncoding};
    double       estBits{std::numeric_limits<double>::quiet_NaN()};
    size_t       actualBytes{std::numeric_limits<size_t>::max()};
    int          modelRank{0};   ///< 1-based by estBits ascending; 0 without a model
    int          actualRank{0};  ///< 1-based by actual bytes ascending; 0 if it failed to encode
    bool         hasCostModel{false};
};

class CostModelGrid {
  public:
    /// `uSample` and `grid` are borrowed and must outlive this object: the point
    /// of the class is that neither is copied per segment.  `uSample` must be the
    /// same sample `grid` was computed from, or est and act describe different
    /// data — nothing here can detect that, so it is the caller's invariant.
    CostModelGrid(const std::vector<uint64_t>& uSample,
                  const std::vector<EncodingType>& encodingTypes,
                  const EncodingGrid& grid)
        : uSample_(uSample), grid_(grid), sectionValues_(uSample.size()) {
        // Built one type at a time: makeAutoSubIntSplitCostModelsFromTypes throws
        // on the first type it does not know, which would take the whole driver
        // down for an oracle-only candidate rather than marking that candidate
        // has_cost_model=false.
        for (EncodingType t : encodingTypes) {
            try {
                auto one = encodings::encoders::makeAutoSubIntSplitCostModelsFromTypes({t});
                if (!one.empty()) models_.push_back(std::move(one.front()));
                else oracleOnly_.push_back(t);
            } catch (const std::exception&) {
                oracleOnly_.push_back(t);
            }
            ++modelBuildAttempts_;
        }
    }

    CostModelGrid(const CostModelGrid&)            = delete;
    CostModelGrid& operator=(const CostModelGrid&) = delete;

    /// Rows for bit range [l..r], one per candidate encoding, sorted by the cost
    /// model's ranking (modelled candidates first, ascending estBits; then the
    /// oracle-only ones, ascending actual bytes).  Computed on first request and
    /// cached for every later one.
    const std::vector<CellEstimate>& at(int l, int r) {
        const auto key = std::make_pair(l, r);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;

        const int    width = r - l + 1;
        const size_t n     = uSample_.size();
        const uint64_t mask = sectionMask(width);
        for (size_t i = 0; i < n; ++i) sectionValues_[i] = (uSample_[i] >> l) & mask;

        const auto metrics = collector_.compute(
            sectionValues_,
            static_cast<encodings::encoders::selectors::MetricFlags>(
                encodings::encoders::selectors::MetricFlag::All));
        ++metricComputeCalls_;

        const EncodingCell& cell = grid_[l][r];

        std::vector<CellEstimate> entries;
        entries.reserve(models_.size() + oracleOnly_.size());
        for (const auto& model : models_) {
            CellEstimate e;
            e.enc          = model->encodingType();
            e.estBits      = model->computeCost(metrics, n, static_cast<size_t>(width));
            e.actualBytes  = cell.bytesFor(e.enc);
            e.actualRank   = cell.rankOf(e.enc);
            e.hasCostModel = true;
            entries.push_back(e);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const CellEstimate& a, const CellEstimate& b) { return a.estBits < b.estBits; });
        for (size_t i = 0; i < entries.size(); ++i) entries[i].modelRank = static_cast<int>(i + 1);

        std::vector<CellEstimate> unmodelled;
        unmodelled.reserve(oracleOnly_.size());
        for (EncodingType t : oracleOnly_) {
            CellEstimate e;
            e.enc         = t;
            e.actualBytes = cell.bytesFor(t);
            e.actualRank  = cell.rankOf(t);
            unmodelled.push_back(e);
        }
        std::sort(unmodelled.begin(), unmodelled.end(),
                  [](const CellEstimate& a, const CellEstimate& b) {
                      return a.actualBytes < b.actualBytes;
                  });
        entries.insert(entries.end(), unmodelled.begin(), unmodelled.end());

        return cache_.emplace(key, std::move(entries)).first->second;
    }

    /// The cost model's pick for this cell — the lowest-estBits candidate, i.e.
    /// what AutoSIS's DP would choose if it used these boundaries.
    EncodingType modelPick(int l, int r) {
        const auto& e = at(l, r);
        for (const auto& c : e)
            if (c.hasCostModel) return c.enc;
        return grid_[l][r].bestEncoding;
    }

    /// Best ACTUAL candidate restricted to those the model can rank.  The right
    /// comparison target for a top-1 statistic: scoring the model against an
    /// oracle pick it has no model for measures the candidate set, not the model.
    EncodingType oraclePickAmongModelled(int l, int r) {
        const auto& entries = at(l, r);
        EncodingType best{};
        size_t bestBytes = std::numeric_limits<size_t>::max();
        bool found = false;
        for (const auto& c : entries) {
            if (!c.hasCostModel || c.actualBytes == std::numeric_limits<size_t>::max()) continue;
            if (!found || c.actualBytes < bestBytes) {
                bestBytes = c.actualBytes;
                best      = c.enc;
                found     = true;
            }
        }
        return found ? best : grid_[l][r].bestEncoding;
    }

    bool hasCostModelFor(EncodingType t) const {
        for (const auto& m : models_)
            if (m->encodingType() == t) return true;
        return false;
    }

    size_t modelledCount() const { return models_.size(); }
    size_t cachedCells() const { return cache_.size(); }

    /// Instrumentation for the memoization claim: one MetricCollector::compute
    /// per DISTINCT bit range regardless of how many callers ask for it, and one
    /// model-vector construction per grid rather than per segment per caller.
    size_t metricComputeCalls() const { return metricComputeCalls_; }
    size_t modelBuildAttempts() const { return modelBuildAttempts_; }

  private:
    const std::vector<uint64_t>& uSample_;
    const EncodingGrid&          grid_;
    std::vector<uint64_t>        sectionValues_;
    encodings::encoders::selectors::MetricCollector<uint64_t> collector_;
    std::vector<std::unique_ptr<encodings::encoders::selectors::costs::EncodingCostModel>> models_;
    std::vector<EncodingType> oracleOnly_;
    std::map<std::pair<int, int>, std::vector<CellEstimate>> cache_;
    size_t metricComputeCalls_{0};
    size_t modelBuildAttempts_{0};
};

// ─── Derived accuracy statistics ─────────────────────────────────────────────

inline double bitsPerElem(size_t bytes, size_t n) {
    return (bytes == std::numeric_limits<size_t>::max() || n == 0)
               ? std::numeric_limits<double>::quiet_NaN()
               : static_cast<double>(bytes) * 8.0 / static_cast<double>(n);
}

/// Spearman rank correlation between the model's ranking and the actual one over
/// one cell's modelled candidates.  NaN below three usable candidates, where the
/// coefficient is either undefined or can only take the values -1, 0 and 1 and
/// would drag a mean around.
inline double spearmanRho(const std::vector<CellEstimate>& entries) {
    std::vector<std::pair<int, int>> pairs;
    for (const auto& c : entries)
        if (c.hasCostModel && c.actualRank > 0) pairs.emplace_back(c.modelRank, c.actualRank);
    const size_t n = pairs.size();
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();

    // Ranks are re-derived within the usable subset: modelRank/actualRank are
    // positions in the full candidate list, and a gap left by a candidate that
    // failed to encode would otherwise be read as a rank disagreement.
    std::vector<size_t> byModel(n), byActual(n);
    for (size_t i = 0; i < n; ++i) { byModel[i] = i; byActual[i] = i; }
    std::sort(byModel.begin(), byModel.end(),
              [&](size_t a, size_t b) { return pairs[a].first < pairs[b].first; });
    std::sort(byActual.begin(), byActual.end(),
              [&](size_t a, size_t b) { return pairs[a].second < pairs[b].second; });
    std::vector<double> rm(n), ra(n);
    for (size_t i = 0; i < n; ++i) {
        rm[byModel[i]]  = static_cast<double>(i + 1);
        ra[byActual[i]] = static_cast<double>(i + 1);
    }

    double sumD2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = rm[i] - ra[i];
        sumD2 += d * d;
    }
    const double nd = static_cast<double>(n);
    return 1.0 - (6.0 * sumD2) / (nd * (nd * nd - 1.0));
}

// ─── The nimble port seam ────────────────────────────────────────────────────

struct AutoPlanOptions {
    std::vector<EncodingType> encodingTypes;
    encodings::encoders::BitSplitOrder order{encodings::encoders::BitSplitOrder::LSB_TO_MSB};
    bool   allowReorderers{false};
    int    numSplits{-1};
    /// Element count the estimate is extrapolated to; sample.size() when unset.
    size_t fullCount{0};
    int    verboseLevel{0};
};

/// An AutoSIS plan together with the per-segment cost-model estimate that chose
/// it — the quantity every estimated-vs-actual table needs and the one thing the
/// production selector path does not hand back.
struct PlanWithEstimates {
    std::vector<SegmentPlan> segments;
    std::vector<double>      estBitsPerSegment;
    double                   totalEstBits{0.0};
};

/// Runs the same selection AutoSIS_C runs (CostModelSet with the Compression
/// dimension over `encodingTypes`) and keeps the estimate.
///
/// PORTING NOTE — this function does not port through nimble's public selector.
/// nimble's SubIntSplitEncoding.h moves only the chosen segments out of the
/// selector and DROPS the cost, so estimated-vs-actual is unobtainable through
/// its selector path: a nimble implementation must call `sampleIntoU64` +
/// `selectSplits` itself and read the per-segment cost off the DP result, which
/// is exactly what CONVENTIONS section 10 lists as one of the two things that
/// must be reimplemented per repo.
inline PlanWithEstimates autoPlan(std::span<const int64_t> sample, const AutoPlanOptions& opts) {
    using namespace encodings::encoders;
    using namespace encodings::encoders::selectors;

    costs::CostModelSet cms;
    cms.add(costs::CostModelDimension::Compression);
    if (!opts.encodingTypes.empty()) cms.forEncodings(opts.encodingTypes);

    auto cfg = makeDefaultAutoSubIntSplitConfig<int64_t>(
        opts.order, std::move(cms), opts.numSplits, opts.allowReorderers);
    cfg.selectorConfig.verboseLevel = opts.verboseLevel;
    // A cost-grid dump path makes the selector THROW when it cannot open the
    // file, which would make plan selection depend on the working directory.
    cfg.selectorConfig.costGridCsvPath.reset();

    const std::vector<int64_t> sampleVec(sample.begin(), sample.end());
    IDSubStreamEncodingSelector selector(cfg.selectorConfig);
    auto result = selector.select(sampleVec, cfg.costModels, cfg.reordererModels,
                                 opts.fullCount ? opts.fullCount : sampleVec.size());

    PlanWithEstimates out;
    out.segments     = std::move(result.segments);
    out.totalEstBits = result.total_cost;
    out.estBitsPerSegment.reserve(out.segments.size());
    for (const auto& seg : out.segments) out.estBitsPerSegment.push_back(seg.cost);
    return out;
}

}  // namespace encodings::benchmark
