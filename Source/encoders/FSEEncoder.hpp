#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "encodings/EncodedData.hpp"
#include "encodings/Encoder.hpp"
#include "encodings/EncodingProperty.hpp"
#include "encodings/EncodingType.hpp"

namespace encodings::encoders {

/// Finite State Entropy codec (tANS variant).
///
/// Achieves close to entropy limit, like Huffman, but without the power-of-2
/// code-length restriction — useful for skewed non-dyadic distributions.
///
/// Sequential decode only (no random access).
///
/// Wire format:
///   [8 bytes]  numElements   (uint64_t)
///   [1 byte]   tableLog      (uint8_t, L = 2^tableLog, range 5..20)
///   [4 bytes]  numSymbols    (uint32_t)
///   [numSymbols × (sizeof(T) + 2)]  symbol+normFreq table:
///              Each entry: sizeof(T) bytes for the symbol value,
///              2 bytes for the normalized frequency (uint16_t).
///              Entries are sorted by symbol value (ascending).
///   [4 bytes]  initialDecoderState (uint32_t, in [L, 2L))
///              Written by encoder as its final state; the decoder starts here.
///   [ceil(payloadBits/8) bytes]  LSB-first bit-packed ANS stream
///              Bits for element 0 are at the front (lowest addresses).
///
template <typename T>
    requires std::is_integral_v<T>
class FSEEncoder : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    // Default table log (L = 1024 states).
    // For compatibility/perf stability, we preserve legacy tuning up to 4096
    // symbols (tableLog<=12), and only use larger logs for higher-cardinality
    // benchmark-only scenarios.
    static constexpr int kDefaultTableLog = 10;
    static constexpr int kLegacyMaxTableLog = 12;
    static constexpr int kMaxTableLog       = 20;
    static constexpr int kMinTableLog     = 5;

    // Fixed header before the symbol table.
    // numElements(8) + tableLog(1) + numSymbols(4) + initialState(4) = 17 bytes.
    static constexpr size_t kHeaderFixed = 8 + 1 + 4 + 4;

    // -------------------------------------------------------------------------
    // Decode table entry: one entry per state slot in [0, L).
    // Public so BlockFSEEncoder can cache and reuse the built decode table.
    // -------------------------------------------------------------------------
    struct DecodeEntry {
        T        symbol;
        uint8_t  nbBits;   // number of bits to read from the stream
        uint32_t newState; // base for the next state: final = newState + readBits(nbBits)
    };

    // -------------------------------------------------------------------------
    // Decode-table cache for BlockFSEEncoder (and any other block-oriented
    // caller that repeatedly decodes blocks with the same symbol distribution).
    //
    // Keyed by (tableLog, numSymbols, 64-bit FNV hash of raw symbol-table bytes).
    // A LRU-1 cache: if the next block's symbol table matches, buildSpread and
    // buildDecodeTable are skipped entirely.
    // -------------------------------------------------------------------------
    struct BlockDecodeContext {
        uint64_t symTableHash{0};
        uint32_t numSymbols{0};
        int      tableLog{0};
        std::vector<DecodeEntry> dt; // L = 2^tableLog entries; empty means uncached
    };

    // -------------------------------------------------------------------------
    // Decode a single self-contained FSE block from raw bytes into dst[0..maxDst).
    //
    // Parses the FSE wire format in-place (no copy of the block bytes).
    // If ctx is non-null and the block's symbol table matches ctx (same hash,
    // tableLog, numSymbols), the cached decode table is reused and
    // buildSpread/buildDecodeTable are skipped entirely.
    // On a cache miss, ctx->dt is rebuilt and ctx is updated.
    //
    // Writes exactly min(numElements_from_header, maxDst) elements to dst.
    // Returns the number of elements written.
    // -------------------------------------------------------------------------
    size_t decodeBlockInto(const uint8_t* data, size_t size,
                           T* dst, size_t maxDst,
                           BlockDecodeContext* ctx) const
    {
        if (size < kHeaderFixed) return 0;
        const uint8_t* p = data;

        uint64_t numElements;
        uint8_t  tableLog8;
        uint32_t numSymbols32;
        uint32_t initState;
        std::memcpy(&numElements,  p, 8); p += 8;
        tableLog8 = *p++;
        std::memcpy(&numSymbols32, p, 4); p += 4;
        std::memcpy(&initState,    p, 4); p += 4;

        if (numElements == 0) return 0;
        const int    tableLog   = static_cast<int>(tableLog8);
        if (tableLog < kMinTableLog || tableLog > kMaxTableLog)
            throw std::runtime_error("FSEEncoder::decodeBlockInto: invalid tableLog");
        const size_t numSymbols = static_cast<size_t>(numSymbols32);
        if (numSymbols == 0)
            throw std::runtime_error("FSEEncoder::decodeBlockInto: empty symbol table");
        const uint32_t L = 1u << tableLog;
        if (initState < L || initState >= (2u * L))
            throw std::runtime_error("FSEEncoder::decodeBlockInto: invalid initial state");

        const size_t symTableBytes = numSymbols * (sizeof(T) + 2);
        if (kHeaderFixed + symTableBytes > size)
            throw std::runtime_error("FSEEncoder::decodeBlockInto: symbol table exceeds buffer");

        // Determine whether the cached decode table is still valid.
        const bool useCachedDt = [&]() -> bool {
            if (!ctx || ctx->dt.empty()) return false;
            if (ctx->tableLog != tableLog || ctx->numSymbols != numSymbols) return false;
            const uint64_t h = fnv64(p, symTableBytes);
            if (ctx->symTableHash != h) return false;
            return true;
        }();

        // Parse symbol table and (re)build decode table on cache miss.
        const std::vector<DecodeEntry>* dtPtr = nullptr;
        if (useCachedDt) {
            dtPtr = &ctx->dt;
            p += symTableBytes; // advance past symbol table
        } else {
            std::vector<std::pair<T, uint32_t>> norm;
            norm.reserve(numSymbols);
            uint32_t normSum = 0;
            const uint8_t* symStart = p;
            for (size_t i = 0; i < numSymbols; ++i) {
                T        sym;
                uint16_t nf16;
                std::memcpy(&sym,  p, sizeof(T)); p += sizeof(T);
                std::memcpy(&nf16, p, 2);         p += 2;
                if (nf16 == 0)
                    throw std::runtime_error("FSEEncoder::decodeBlockInto: normFreq == 0");
                norm.push_back({sym, static_cast<uint32_t>(nf16)});
                normSum += static_cast<uint32_t>(nf16);
            }
            if (normSum != L)
                throw std::runtime_error("FSEEncoder::decodeBlockInto: normFreq sum != L");

            const auto spread = buildSpread(norm, tableLog);
            auto       newDt  = buildDecodeTable(spread, norm, tableLog);

            if (ctx) {
                ctx->symTableHash = fnv64(symStart, symTableBytes);
                ctx->numSymbols   = static_cast<uint32_t>(numSymbols);
                ctx->tableLog     = tableLog;
                ctx->dt           = std::move(newDt);
            } else {
                // No cache — store locally and use a pointer to it.
                // Avoid a second allocation by temporarily storing in a local var.
                // We'll handle this via the non-cached code path below.
                ctx = nullptr; // handled after the branch
                // Re-assign: we need to keep newDt alive through runDecodeLoop.
                // Use a static thread_local would be UB; use a scope-local instead.
                const uint8_t* payloadStart2 = p;
                const uint8_t* payloadEnd2   = data + size;
                const size_t   toWrite       = static_cast<size_t>(std::min<uint64_t>(numElements, maxDst));
                runDecodeLoop(newDt, L, initState, payloadStart2, payloadEnd2, toWrite, dst);
                return toWrite;
            }
            dtPtr = &ctx->dt;
        }

        const uint8_t* payloadStart = p;
        const uint8_t* payloadEnd   = data + size;
        const size_t   toWrite      = static_cast<size_t>(std::min<uint64_t>(numElements, maxDst));
        runDecodeLoop(*dtPtr, L, initState, payloadStart, payloadEnd, toWrite, dst);
        return toWrite;
    }

private:
    // -------------------------------------------------------------------------
    // FNV-64 hash of raw bytes — used to key the BlockDecodeContext cache.
    // -------------------------------------------------------------------------
    static uint64_t fnv64(const uint8_t* data, size_t size) noexcept {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < size; ++i)
            h = (h ^ static_cast<uint64_t>(data[i])) * 1099511628211ULL;
        return h;
    }

    // -------------------------------------------------------------------------
    // ANS decode inner loop — writes numElements symbols into dst[0..n).
    // Extracted so both decodeAll and decodeBlockInto share the same kernel.
    // -------------------------------------------------------------------------
    static void runDecodeLoop(
        const std::vector<DecodeEntry>& dt,
        uint32_t L, uint32_t initState,
        const uint8_t* payloadStart, const uint8_t* payloadEnd,
        size_t numElements,
        T* dst)
    {
        LSBBitReader br{payloadStart, payloadEnd};
        uint32_t state = initState;
        for (size_t i = 0; i < numElements; ++i) {
            if (state < L || state >= (2u * L)) [[unlikely]]
                throw std::runtime_error("FSEEncoder: state out of bounds during decode");
            const DecodeEntry& e = dt[state - L];
            dst[i] = e.symbol;
            state  = e.newState + br.read(static_cast<int>(e.nbBits));
        }
    }

    // -------------------------------------------------------------------------
    // Per-symbol encoding info (precomputed before encoding).
    // -------------------------------------------------------------------------
    struct EncInfo {
        uint32_t f;          // normalized frequency
        uint32_t threshold;  // states ≥ threshold use nbBitsMin+1; others use nbBitsMin
        int      nbBitsMin;  // tableLog - bit_width(f)
        uint32_t cumStart;   // offset into encodeStates[] for this symbol
    };

    // -------------------------------------------------------------------------
    // LSB-first bit writer/reader (matches the ANS bit ordering).
    // -------------------------------------------------------------------------
    struct LSBBitWriter {
        std::vector<uint8_t>& out;
        uint64_t buf{0};
        int      filled{0};

        void write(uint32_t bits, int nb) {
            if (nb == 0) return;
            buf    |= static_cast<uint64_t>(bits) << filled;
            filled += nb;
            while (filled >= 8) {
                out.push_back(static_cast<uint8_t>(buf & 0xFF));
                buf    >>= 8;
                filled  -= 8;
            }
        }

        void flush() {
            if (filled > 0) {
                out.push_back(static_cast<uint8_t>(buf & 0xFF));
                buf    = 0;
                filled = 0;
            }
        }
    };

    struct LSBBitReader {
        const uint8_t* cur;
        const uint8_t* end;
        uint64_t buf{0};
        int      avail{0};

        void refill() {
            while (avail <= 56 && cur < end) {
                buf   |= static_cast<uint64_t>(*cur++) << avail;
                avail += 8;
            }
        }

        uint32_t read(int nb) {
            if (nb == 0) return 0;
            if (avail < nb) {
                refill();
                if (avail < nb) {
                    throw std::runtime_error("FSEEncoder::decodeAll: invalid or truncated bitstream");
                }
            }
            const uint32_t bits = static_cast<uint32_t>(buf) & ((1u << nb) - 1u);
            buf   >>= nb;
            avail  -= nb;
            return bits;
        }
    };

    // -------------------------------------------------------------------------
    // Choose tableLog based on number of unique symbols.
    // -------------------------------------------------------------------------
    static int chooseTableLog(size_t numSymbols) {
        if (numSymbols <= 2) return kMinTableLog;
        // L must be >= numSymbols (each symbol gets >= 1 slot).
        const int minLog = static_cast<int>(std::bit_width(numSymbols - 1)); // ceil(log2(numSymbols))

        // Keep historical behavior for <=4096 unique symbols to avoid
        // perturbing performance on streams that already perform well.
        if (numSymbols <= (size_t{1} << kLegacyMaxTableLog)) {
            return std::clamp(std::max(minLog + 2, kDefaultTableLog), kMinTableLog, kLegacyMaxTableLog);
        }

        // For very high-cardinality benchmark workloads, grow just enough to
        // satisfy feasibility while capping memory growth.
        return std::clamp(minLog, kLegacyMaxTableLog + 1, kMaxTableLog);
    }

    // -------------------------------------------------------------------------
    // Normalize raw frequencies to sum exactly to L = 2^tableLog.
    // Every present symbol gets normFreq ≥ 1.
    // Returns sorted vector of (sym, normFreq) pairs (ascending by sym).
    // -------------------------------------------------------------------------
    static std::vector<std::pair<T, uint32_t>> normalizeFreqs(
        const ankerl::unordered_dense::map<T, uint64_t>& rawFreq,
        int tableLog)
    {
        const uint32_t L = 1u << tableLog;
        uint64_t total = 0;
        for (auto& [sym, f] : rawFreq) total += f;

        // Collect (sym, raw) sorted ascending by symbol for reproducibility.
        std::vector<std::pair<T, uint64_t>> sorted(rawFreq.begin(), rawFreq.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        std::vector<std::pair<T, uint32_t>> norm;
        norm.reserve(sorted.size());
        uint32_t sumNorm = 0;
        size_t largestIdx = 0;
        uint64_t largestRaw = 0;

        for (auto& [sym, f] : sorted) {
            uint32_t nf = static_cast<uint32_t>(std::max<uint64_t>(1, (f * L) / total));
            norm.push_back({sym, nf});
            sumNorm += nf;
            if (f > largestRaw) { largestRaw = f; largestIdx = norm.size() - 1; }
        }

        // Rebalance to exact L safely (no unsigned underflow).
        if (sumNorm > L) {
            uint32_t excess = sumNorm - L;
            // Deterministic order: reduce larger buckets first while keeping nf >= 1.
            std::vector<size_t> idx(norm.size());
            for (size_t i = 0; i < norm.size(); ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                if (norm[a].second != norm[b].second) return norm[a].second > norm[b].second;
                return sorted[a].second > sorted[b].second;
            });

            for (size_t id : idx) {
                if (excess == 0) break;
                const uint32_t canDrop = norm[id].second > 1 ? (norm[id].second - 1) : 0;
                if (canDrop == 0) continue;
                const uint32_t drop = std::min(canDrop, excess);
                norm[id].second -= drop;
                excess -= drop;
            }
            if (excess != 0) {
                throw std::runtime_error("FSEEncoder::encode: failed to normalize frequencies (sum > L)");
            }
        } else if (sumNorm < L) {
            norm[largestIdx].second += (L - sumNorm);
        }

        return norm;
    }

    // -------------------------------------------------------------------------
    // Build the spread table: L slots, each assigned a symbol proportional to
    // normFreq.  Uses the zstd-style odd step to interleave symbols.
    // -------------------------------------------------------------------------
    static std::vector<T> buildSpread(
        const std::vector<std::pair<T, uint32_t>>& norm,
        int tableLog)
    {
        const uint32_t L    = 1u << tableLog;
        const uint32_t step = (L >> 1) + (L >> 3) + 3; // odd → coprime with L = 2^k

        std::vector<T> spread(L);
        uint32_t pos = 0;
        for (auto& [sym, f] : norm) {
            for (uint32_t k = 0; k < f; ++k) {
                spread[pos] = sym;
                pos = (pos + step) & (L - 1);
            }
        }
        return spread;
    }

    // -------------------------------------------------------------------------
    // Build decode table (L entries) from the spread table.
    // Each entry: symbol, nbBits to read, newState base for the next decode state.
    // State machine runs with states in [L, 2L); table is indexed by state - L.
    // -------------------------------------------------------------------------
    static std::vector<DecodeEntry> buildDecodeTable(
        const std::vector<T>& spread,
        const std::vector<std::pair<T, uint32_t>>& norm,
        int tableLog)
    {
        const uint32_t L = 1u << tableLog;

        // symbolNext[sym] starts at normFreq[sym] and increments as we visit each slot.
        ankerl::unordered_dense::map<T, uint32_t> symbolNext;
        symbolNext.reserve(norm.size());
        for (auto& [sym, f] : norm) symbolNext[sym] = f;

        std::vector<DecodeEntry> dt(L);
        for (uint32_t pos = 0; pos < L; ++pos) {
            const T      sym = spread[pos];
            const uint32_t x = symbolNext[sym]++;
            // nb: number of bits the decoder reads for this slot.
            // Chosen so newState = x << nb ∈ [L, 2L).
            const int    nb  = tableLog - static_cast<int>(std::bit_width(x)) + 1;
            const uint32_t newState = x << nb; // guaranteed ∈ [L, 2L)
            dt[pos] = {sym, static_cast<uint8_t>(nb), newState};
        }
        return dt;
    }

    // -------------------------------------------------------------------------
    // Build the encode state table (size L, one entry per rank across all syms)
    // and the per-symbol EncInfo lookup table.
    //
    // encodeStates[cumStart[s] + k] = L + decode_table_position
    // for the k-th occurrence of symbol s in the spread (k = 0-indexed).
    // -------------------------------------------------------------------------
    static void buildEncodeTables(
        const std::vector<DecodeEntry>&          dt,
        const std::vector<std::pair<T, uint32_t>>& norm,
        int                                      tableLog,
        std::vector<uint32_t>&                   encodeStates,
        ankerl::unordered_dense::map<T, EncInfo>& encInfo)
    {
        const uint32_t L = 1u << tableLog;
        encodeStates.resize(L);

        // Assign cumulative start positions.
        uint32_t cum = 0;
        for (auto& [sym, f] : norm) {
            const int bw       = static_cast<int>(std::bit_width(f));
            const int nbMin    = tableLog - bw;
            const uint32_t thr = f << (tableLog - bw + 1); // f * 2^(tableLog-bw+1)
            encInfo[sym] = {f, thr, nbMin, cum};
            cum += f;
        }

        // Iterate decode table to fill encodeStates.
        // For each pos, determine which occurrence k of its symbol this is.
        ankerl::unordered_dense::map<T, uint32_t> occCount;
        occCount.reserve(norm.size());
        for (auto& [sym, f] : norm) occCount[sym] = 0;

        for (uint32_t pos = 0; pos < L; ++pos) {
            const T      sym = dt[pos].symbol;
            const uint32_t k = occCount[sym]++;
            encodeStates[encInfo.at(sym).cumStart + k] = L + pos;
        }
    }

    // -------------------------------------------------------------------------
    // Core encode loop (processes symbols right-to-left).
    // Collects (bits, nbBits) pairs in right-to-left order.
    // -------------------------------------------------------------------------
    static uint32_t encodeSymbols(
        std::span<const T>                             data,
        const ankerl::unordered_dense::map<T, EncInfo>& encInfo,
        const std::vector<uint32_t>&                   encodeStates,
        int                                            tableLog,
        std::vector<std::pair<uint32_t, int>>&         bitsQueue)
    {
        const uint32_t L = 1u << tableLog;
        bitsQueue.reserve(data.size());
        uint32_t state = L; // start at lower bound of [L, 2L)

        for (int i = static_cast<int>(data.size()) - 1; i >= 0; --i) {
            const EncInfo& ei = encInfo.at(data[i]);
            const int nb   = ei.nbBitsMin + (state >= ei.threshold ? 1 : 0);
            const uint32_t bits = state & ((1u << nb) - 1u);
            state >>= nb;
            // state now in [ei.f, 2*ei.f); k = state - ei.f
            const uint32_t k = state - ei.f;
            state = encodeStates[ei.cumStart + k];
            bitsQueue.push_back({bits, nb});
        }
        return state; // final encoder state → initial decoder state
    }

public:
    EncodingType encodingType() const override { return EncodingType::FSEEncoding; }
    std::string name() const override { return "FSE"; }

    EncodingProperties properties() const override {
        // RandomAccess deliberately NOT claimed: decodeAt/decodeRange (below)
        // call decodeAll() fresh on every invocation, with no caching -- any
        // code that branches on this flag to decide whether it's safe to skip
        // its own caching (e.g. FOREncoder::decodeAt, CascadingFOREncoder's
        // decodeLeafAt, RangePackSectionCodec, RunLengthEncoder's ComposedView)
        // would otherwise be silently misled into O(N)-per-query behaviour.
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::PreservesOrder
             | EncodingProperty::RequiresFullData
             | EncodingProperty::EntropyCoding
             | EncodingProperty::VariableSize;
    }

    // -------------------------------------------------------------------------
    // encode
    // -------------------------------------------------------------------------
    EncodedData encode(std::span<const T> data) override {
        EncodedData result;
        result.metadata().elementCount         = data.size();
        result.metadata().encodingName         = name();
        result.metadata().supportsRandomAccess = false;

        if (data.empty()) {
            result.data().resize(kHeaderFixed, 0);
            return result;
        }

        // --- Frequency count ---
        ankerl::unordered_dense::map<T, uint64_t> rawFreq;
        rawFreq.reserve(data.size());
        for (T v : data) ++rawFreq[v];

        // --- Choose tableLog and normalize ---
        const int tableLog = chooseTableLog(rawFreq.size());
        const uint32_t L = 1u << tableLog;

        if (rawFreq.size() > static_cast<size_t>(L)) {
            // Too many unique symbols for the table — fall back to raw copy.
            // (The cost model should have avoided selecting FSE in this case.)
            throw std::runtime_error("FSEEncoder::encode: uniqueCount > L (" + std::to_string(rawFreq.size()) + " > " + std::to_string(L) + "), cannot build table");
        }

        const auto norm = normalizeFreqs(rawFreq, tableLog);
        const size_t numSymbols = norm.size();
        uint32_t normSum = 0;
        for (const auto& [_, nf] : norm) normSum += nf;
        if (normSum != L) {
            throw std::runtime_error("FSEEncoder::encode: normalized frequencies do not sum to table size");
        }

        // --- Build tables ---
        const auto spread  = buildSpread(norm, tableLog);
        const auto dt      = buildDecodeTable(spread, norm, tableLog);
        std::vector<uint32_t> encodeStates;
        ankerl::unordered_dense::map<T, EncInfo> encInfo;
        encInfo.reserve(numSymbols);
        buildEncodeTables(dt, norm, tableLog, encodeStates, encInfo);

        // --- Encode symbols right-to-left ---
        std::vector<std::pair<uint32_t, int>> bitsQueue;
        const uint32_t initState = encodeSymbols(data, encInfo, encodeStates, tableLog, bitsQueue);

        // --- Allocate output ---
        const size_t symTableBytes = numSymbols * (sizeof(T) + 2);
        const size_t headerBytes   = kHeaderFixed + symTableBytes;

        std::vector<uint8_t> out;
        out.reserve(headerBytes + data.size() * 2); // rough upper bound
        out.resize(headerBytes);

        // --- Write fixed header ---
        uint8_t* h = out.data();
        const uint64_t ne = static_cast<uint64_t>(data.size());
        const uint32_t ns = static_cast<uint32_t>(numSymbols);
        std::memcpy(h, &ne, 8);         h += 8;
        *h++ = static_cast<uint8_t>(tableLog);
        std::memcpy(h, &ns, 4);         h += 4;
        std::memcpy(h, &initState, 4);  h += 4;

        // --- Write symbol+normFreq table (sorted by symbol, ascending) ---
        for (auto& [sym, nf] : norm) {
            std::memcpy(h, &sym, sizeof(T));              h += sizeof(T);
            const uint16_t nf16 = static_cast<uint16_t>(nf);
            std::memcpy(h, &nf16, 2);                    h += 2;
        }

        // --- Write payload: bits in forward (element 0 first) order ---
        // bitsQueue is [data[N-1], data[N-2], ..., data[0]]; reverse iterate.
        LSBBitWriter bw{out};
        for (auto it = bitsQueue.rbegin(); it != bitsQueue.rend(); ++it) {
            bw.write(it->first, it->second);
        }
        bw.flush();

        result.data() = std::move(out);
        result.metadata().compressedSize   = result.data().size();
        result.metadata().uncompressedSize = data.size() * sizeof(T);
        return result;
    }

    // -------------------------------------------------------------------------
    // decodeAll
    // -------------------------------------------------------------------------
    std::vector<T> decodeAll(const EncodedData& encoded) override {
        const uint8_t* p     = encoded.data().data();
        const size_t   total = encoded.data().size();
        if (total < kHeaderFixed) return {};

        uint64_t numElements;
        uint8_t  tableLog8;
        uint32_t numSymbols32;
        uint32_t initState;
        std::memcpy(&numElements,  p, 8); p += 8;
        tableLog8 = *p++;
        std::memcpy(&numSymbols32, p, 4); p += 4;
        std::memcpy(&initState,    p, 4); p += 4;

        if (numElements == 0) return {};
        const int      tableLog  = static_cast<int>(tableLog8);
        if (tableLog < kMinTableLog || tableLog > kMaxTableLog) {
            throw std::runtime_error("FSEEncoder::decodeAll: invalid tableLog");
        }
        const size_t   numSymbols = static_cast<size_t>(numSymbols32);
        if (numSymbols == 0) {
            throw std::runtime_error("FSEEncoder::decodeAll: empty symbol table for non-empty payload");
        }

        const uint32_t L = 1u << tableLog;
        if (initState < L || initState >= (2u * L)) {
            throw std::runtime_error("FSEEncoder::decodeAll: invalid initial decoder state");
        }

        // --- Parse symbol+normFreq table ---
        const size_t symTableBytes = numSymbols * (sizeof(T) + 2);
        if (kHeaderFixed + symTableBytes > total) {
            throw std::runtime_error("FSEEncoder::decodeAll: symbol table exceeds buffer");
        }

        std::vector<std::pair<T, uint32_t>> norm;
        norm.reserve(numSymbols);
        uint32_t normSum = 0;
        for (size_t i = 0; i < numSymbols; ++i) {
            T        sym;
            uint16_t nf16;
            std::memcpy(&sym,  p, sizeof(T)); p += sizeof(T);
            std::memcpy(&nf16, p, 2);         p += 2;
            if (nf16 == 0) {
                throw std::runtime_error("FSEEncoder::decodeAll: invalid normalized frequency 0");
            }
            norm.push_back({sym, static_cast<uint32_t>(nf16)});
            normSum += static_cast<uint32_t>(nf16);
        }

        if (normSum != L) {
            throw std::runtime_error("FSEEncoder::decodeAll: invalid normalized frequency sum");
        }

        // --- Rebuild tables ---
        const auto spread = buildSpread(norm, tableLog);
        const auto dt     = buildDecodeTable(spread, norm, tableLog);

        // --- Decode ---
        const uint8_t* payloadStart = p;
        const uint8_t* payloadEnd   = encoded.data().data() + total;

        std::vector<T> result(static_cast<size_t>(numElements));
        runDecodeLoop(dt, L, initState, payloadStart, payloadEnd,
                      static_cast<size_t>(numElements), result.data());
        return result;
    }

    // -------------------------------------------------------------------------
    // Random access — not efficiently supported; fall back to decodeAll.
    // -------------------------------------------------------------------------
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const auto all = decodeAll(encoded);
        if (index >= all.size()) return std::nullopt;
        return all[index];
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        const auto all = decodeAll(encoded);
        if (start >= all.size()) return {};
        const size_t realEnd = std::min(end, all.size());
        return std::vector<T>(all.begin() + static_cast<std::ptrdiff_t>(start),
                              all.begin() + static_cast<std::ptrdiff_t>(realEnd));
    }
};

} // namespace encodings::encoders
