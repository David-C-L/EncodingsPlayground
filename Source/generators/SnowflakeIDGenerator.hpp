#pragma once

#include <random>
#include <stdexcept>
#include <unordered_map>
#include "DataGenerator.hpp"

namespace encodings::datagen {

    struct SnowflakeIDConfig {
        const char* name_{"Default"}; // Optional name for this config (e.g., "Instagram Snowflake")
        const size_t numTopZeroBits_{1}; // Default 1 bit for sign (unused, always 0)
        const size_t numTimestampBits_{41}; // Default 41 bits for timestamp (enough for ~69 years at ms precision)
        const size_t numMachineIDBits_{10}; // Default 10 bits for machine ID (enough for 1024 machines)
        const size_t numSequenceBits_{12};  // Default 12 bits for sequence number (enough for 4096 IDs per ms per machine)

        template<typename T>
        requires std::is_integral_v<T>
        bool isValidConfig() const {
            return numTopZeroBits_ + numTimestampBits_ + numMachineIDBits_ + numSequenceBits_ <= sizeof(T) * 8;
        }
    };

    static constexpr SnowflakeIDConfig INSTAGRAM_SNOWFLAKE_CONFIG =
        SnowflakeIDConfig{
            .name_ = "InstagramSnowflakeInt64",
            .numTopZeroBits_ = 0,
            .numTimestampBits_ = 41,
            .numMachineIDBits_ = 13,
            .numSequenceBits_ = 10
        };

    static constexpr SnowflakeIDConfig INSTAGRAM_SNOWFLAKE_INT_CONFIG =
        SnowflakeIDConfig{
            .name_ = "InstagramSnowflakeInt32",
            .numTopZeroBits_ = 0,
            .numTimestampBits_ = 21,
            .numMachineIDBits_ = 7,
            .numSequenceBits_ = 4
        };

    /**
     * @brief Generates Snowflake-style IDs that mimic real distributed-system ID streams.
     *
     * Layout (MSB → LSB):
     *   [ numTopZeroBits_ | numTimestampBits_ | numMachineIDBits_ | numSequenceBits_ ]
     *
     * Generation model
     * ----------------
     * - Timestamp advances monotonically.  The average per-ID advance is chosen
     *   so that the full generate(count) call consumes approximately
     *   `tsFillFraction` of the available timestamp range.  This prevents
     *   premature saturation (the old fixed-ms-per-ID model would exhaust a
     *   21-bit timestamp in under 1M IDs when advancing 2 ms per ID on average).
     *   Per-ID the advance is drawn from Bernoulli(p) when p < 1 (bursty, many
     *   IDs share a timestamp) or Uniform(0, 2*avgAdv) when avgAdv >= 1 (sparse,
     *   each ID typically gets its own timestamp slot).
     * - Machine ID is drawn uniformly at random from [0, numMachines) for each
     *   ID independently — any machine can serve any request at any time.
     * - Sequence number counts IDs issued by a given machine within a single
     *   timestamp tick.  If the sequence counter for (timestamp, machine) would
     *   overflow 2^numSequenceBits the timestamp is forced to advance by 1 first.
     * - reset() restores the full RNG + counter state so repeated calls to
     *   generate() produce identical output (important for reproducible benchmarks).
     *
     * @tparam T  Integral type wide enough to hold the full ID.
     *            Use int64_t for standard 64-bit Snowflakes, int32_t for compact
     *            variants (INSTAGRAM_SNOWFLAKE_INT_CONFIG fits in 32 bits).
     */
    template<typename T>
    requires std::is_integral_v<T>
    class SnowflakeIDGenerator : public DataGenerator<T> {
    public:
        /**
         * @param config          Bit-field layout configuration.
         * @param numMachines     Number of distinct machines; must be ≤ 2^numMachineIDBits.
         * @param seed            RNG seed for reproducibility (default 42).
         * @param tsFillFraction  What fraction of the available timestamp range a
         *                        single generate(count) call should consume
         *                        (default 0.5 = use half the range).
         *                        Values near 0 produce highly bursty output
         *                        (many IDs share a timestamp); values near 1
         *                        spread IDs across the full timestamp range.
         */
        explicit SnowflakeIDGenerator(SnowflakeIDConfig config,
                                      size_t numMachines,
                                      int64_t seed          = 42,
                                      double  tsFillFraction = 0.5)
            : config_(std::move(config))
            , numMachines_(numMachines)
            , seed_(seed)
            , tsFillFraction_(tsFillFraction)
            , rng_(static_cast<uint64_t>(seed))
        {
            if (!config_.isValidConfig<T>()) {
                throw std::invalid_argument(
                    "SnowflakeIDGenerator: total bits in SnowflakeIDConfig exceed bit width of T");
            }
            const uint64_t maxMachineID = (uint64_t{1} << config_.numMachineIDBits_) - 1;
            if (numMachines_ == 0 || numMachines_ > maxMachineID + 1) {
                throw std::invalid_argument(
                    "SnowflakeIDGenerator: numMachines must be in [1, 2^numMachineIDBits]");
            }
            if (tsFillFraction_ <= 0.0 || tsFillFraction_ > 1.0) {
                throw std::invalid_argument(
                    "SnowflakeIDGenerator: tsFillFraction must be in (0, 1]");
            }

            // Precompute layout constants
            seqMask_        = (uint64_t{1} << config_.numSequenceBits_) - 1;
            machineShift_   = config_.numSequenceBits_;
            timestampShift_ = config_.numSequenceBits_ + config_.numMachineIDBits_;
            maxTimestamp_   = (uint64_t{1} << config_.numTimestampBits_) - 1;

            machineDist_ = std::uniform_int_distribution<uint64_t>(0, numMachines_ - 1);
        }

        std::vector<T> generate(size_t count) override {
            std::vector<T> result;
            result.reserve(count);

            // Compute the average per-ID timestamp advance so that this call
            // consumes approximately tsFillFraction_ of the timestamp range.
            // avgAdv = (maxTimestamp_ * tsFillFraction_) / count
            const double targetSpan = static_cast<double>(maxTimestamp_) * tsFillFraction_;
            const double avgAdv     = (count > 0) ? targetSpan / static_cast<double>(count) : 0.0;

            // Build a tick distribution for this call:
            //   avgAdv < 1  → Bernoulli(avgAdv): advance by 1 with prob avgAdv, else 0
            //   avgAdv >= 1 → Uniform(0, 2*avgAdv): symmetric around avgAdv
            // We pre-draw ticks as doubles for the Bernoulli case.
            std::uniform_real_distribution<double> bernoulliDist(0.0, 1.0);
            std::uniform_int_distribution<uint64_t> uniformTickDist;
            const bool useBernoulli = avgAdv < 1.0;
            if (!useBernoulli) {
                const uint64_t tickMax = static_cast<uint64_t>(2.0 * avgAdv);
                uniformTickDist = std::uniform_int_distribution<uint64_t>(0, tickMax);
            }

            for (size_t i = 0; i < count; ++i) {
                // 1. Pick a machine uniformly at random.
                const uint64_t machineID = machineDist_(rng_);

                // 2. Advance the simulated clock.
                uint64_t tick = 0;
                if (useBernoulli) {
                    tick = (bernoulliDist(rng_) < avgAdv) ? 1u : 0u;
                } else {
                    tick = uniformTickDist(rng_);
                }
                uint64_t candidateTs = currentTimestamp_ + tick;

                // 3. Claim the next sequence slot for (candidateTs, machine).
                //    If exhausted, force a 1-tick advance (guaranteed free slot).
                uint64_t& seq = seqCounters_[{candidateTs, machineID}];
                if (seq > seqMask_) {
                    candidateTs += 1;
                    seq = seqCounters_[{candidateTs, machineID}]; // may be 0
                }

                // 4. Clamp to the maximum representable timestamp.
                if (candidateTs > maxTimestamp_) {
                    candidateTs = maxTimestamp_;
                }

                // 5. The clock only moves forward.
                if (candidateTs > currentTimestamp_) {
                    currentTimestamp_ = candidateTs;
                }

                const uint64_t seqNum = seqCounters_[{currentTimestamp_, machineID}]++;

                // 6. Pack the three fields into one integer.
                const uint64_t id =
                    (currentTimestamp_ << timestampShift_) |
                    (machineID         << machineShift_)   |
                    seqNum;

                result.push_back(static_cast<T>(id));
            }

            return result;
        }

        std::string name() const override {
            return "SnowflakeID_"
                 + std::to_string(numMachines_) + "Machines_"
                 + std::to_string(sizeof(T) * 8) + "Bits_"
                 + config_.name_ + "Config";
        }

        void reset() override {
            rng_.seed(static_cast<uint64_t>(seed_));
            currentTimestamp_ = 1;
            seqCounters_.clear();
        }

        std::map<std::string, std::string> getConfig() const override {
            return {
                {"config",          config_.name_},
                {"numMachines",     std::to_string(numMachines_)},
                {"seed",            std::to_string(seed_)},
                {"tsFillFraction",  std::to_string(tsFillFraction_)},
            };
        }

    private:
        // Hash for std::pair<uint64_t, uint64_t> used as sequence-counter key
        struct PairHash {
            size_t operator()(const std::pair<uint64_t, uint64_t>& p) const noexcept {
                size_t h = std::hash<uint64_t>{}(p.first);
                h ^= std::hash<uint64_t>{}(p.second)
                     + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        SnowflakeIDConfig config_;
        size_t   numMachines_;
        int64_t  seed_;
        double   tsFillFraction_;

        // Layout constants (derived from config, immutable after construction)
        uint64_t seqMask_        = 0;
        uint64_t machineShift_   = 0;
        uint64_t timestampShift_ = 0;
        uint64_t maxTimestamp_   = 0;

        // Mutable generation state — fully restored by reset()
        std::mt19937_64 rng_;
        uint64_t currentTimestamp_ = 1;
        // Maps (timestamp_tick, machineID) → next sequence number for that slot
        std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, PairHash> seqCounters_;

        std::uniform_int_distribution<uint64_t> machineDist_;
    };

} // namespace encodings::datagen