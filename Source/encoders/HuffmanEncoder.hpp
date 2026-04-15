#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <queue>
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

/// Canonical Huffman entropy codec.
///
/// Supports encode() and decodeAll() only — random access is not possible with
/// variable-length codes, so decodeAt() and decodeRange() throw.
///
/// Wire format:
///   [8 bytes]  numElements   (uint64_t)
///   [2 bytes]  numSymbols    (uint16_t, ≤ 65535)
///   [8 bytes]  payloadBits   (uint64_t — total bits in the payload section)
///   [numSymbols × (sizeof(T)+1)]  canonical symbol table
///              Each entry: sizeof(T) bytes for the symbol value, 1 byte for the
///              code length.  Entries are sorted by (code_length ASC, symbol ASC),
///              which is exactly the canonical Huffman ordering.  The canonical
///              codes are implicit and can be reconstructed from this sorted list.
///   [ceil(payloadBits/8) bytes]  MSB-first bit-packed Huffman codes
///
template <typename T>
    requires std::is_integral_v<T>
class HuffmanEncoder : public Codec<T> {
public:
    using EncodedData = EncodedBuffer<uint8_t>;

    // Fast decode table covers codes ≤ kMaxFastBits (12) bits.
    // For 4096 entries × up to 9 bytes/entry this is ≤ 36 KB — fits in L1/L2.
    static constexpr int kMaxFastBits = 12;

    // Sentinel: fastTable entry with len=0 means "code longer than kMaxFastBits".
    static constexpr uint8_t kSlowPathLen = 0;

    static constexpr size_t kMaxCodeBits = 24; // safety cap; never triggered in practice

    // Fixed header size before the variable-length symbol table.
    static constexpr size_t kHeaderFixed = 8 + 2 + 8; // numElements + numSymbols + payloadBits

    // -------------------------------------------------------------------------
    // Private types
    // -------------------------------------------------------------------------
private:
    struct EncEntry {
        T        sym;
        uint8_t  len;
        uint32_t code;
    };

    struct FastEntry {
        T       sym;
        uint8_t len; // 0 = kSlowPathLen = this slot maps to a code > kMaxFastBits
    };

    struct SlowEntry {
        uint32_t code;
        uint8_t  len;
        T        sym;
    };

    // -------------------------------------------------------------------------
    // Bit I/O
    // -------------------------------------------------------------------------

    // Writes bits MSB-first into an output byte vector.
    struct BitWriter {
        std::vector<uint8_t>& out;
        uint64_t buf{0};
        int      filled{0}; // bits currently in buf

        void write(uint32_t code, int len) {
            buf     = (buf << len) | static_cast<uint64_t>(code);
            filled += len;
            while (filled >= 8) {
                filled -= 8;
                out.push_back(static_cast<uint8_t>(buf >> filled));
            }
        }

        void flush() {
            if (filled > 0) {
                out.push_back(static_cast<uint8_t>(buf << (8 - filled)));
                filled = 0;
            }
        }
    };

    // Reads bits MSB-first from a byte range.
    struct BitReader {
        const uint8_t* cur;
        const uint8_t* end;
        uint64_t buf{0};
        int      avail{0}; // valid bits in buf (at the top/MSB side)

        // Fill the buffer until we have at least 57 bits or exhaust the source.
        void refill() {
            while (avail <= 56 && cur < end) {
                buf    = (buf << 8) | static_cast<uint64_t>(*cur++);
                avail += 8;
            }
        }

        // Peek the top n bits without consuming them.
        uint32_t peek(int n) const {
            return static_cast<uint32_t>(buf >> (avail - n)) & ((1u << n) - 1);
        }

        void consume(int n) { avail -= n; }

        bool exhausted() const { return avail == 0 && cur >= end; }
    };

    // -------------------------------------------------------------------------
    // Huffman tree
    // -------------------------------------------------------------------------

    struct HNode {
        uint64_t freq;
        int32_t  left{-1}, right{-1}; // -1 = leaf
        T        sym{};
    };

    // Build tree + compute canonical (symbol, length) pairs, sorted by
    // (code_length ASC, symbol ASC).  Returns an empty vector if data is empty.
    static std::vector<EncEntry> buildCanonical(const std::vector<T>& data) {
        if (data.empty()) return {};

        // Count frequencies.
        ankerl::unordered_dense::map<T, uint64_t> freq;
        freq.reserve(data.size());
        for (T v : data) ++freq[v];

        // Build Huffman tree.
        std::vector<HNode> pool;
        pool.reserve(2 * freq.size() + 1);

        struct PQEntry { uint64_t freq; int32_t idx; };
        auto cmp = [](const PQEntry& a, const PQEntry& b) { return a.freq > b.freq; };
        std::priority_queue<PQEntry, std::vector<PQEntry>, decltype(cmp)> pq(cmp);

        for (auto& [sym, f] : freq) {
            int32_t idx = static_cast<int32_t>(pool.size());
            pool.push_back({f, -1, -1, sym});
            pq.push({f, idx});
        }

        // Merge until one root remains.
        while (pq.size() > 1) {
            auto [f1, i1] = pq.top(); pq.pop();
            auto [f2, i2] = pq.top(); pq.pop();
            int32_t n = static_cast<int32_t>(pool.size());
            pool.push_back({f1 + f2, i1, i2, {}});
            pq.push({f1 + f2, n});
        }
        int32_t root = pq.top().idx;

        // Assign code lengths via iterative DFS.
        std::vector<std::pair<T, uint8_t>> symLens;
        symLens.reserve(freq.size());
        {
            struct Frame { int32_t node; uint8_t depth; };
            std::vector<Frame> stk;
            stk.push_back({root, 0});
            while (!stk.empty()) {
                auto [n, d] = stk.back();
                stk.pop_back();
                if (pool[n].left == -1) {
                    // Minimum code length is 1 (handles single-symbol edge case).
                    symLens.push_back({pool[n].sym, std::max<uint8_t>(d, 1)});
                } else {
                    stk.push_back({pool[n].right, static_cast<uint8_t>(d + 1)});
                    stk.push_back({pool[n].left,  static_cast<uint8_t>(d + 1)});
                }
            }
        }

        // Sort for canonical ordering: (len ASC, sym ASC).
        std::sort(symLens.begin(), symLens.end(), [](const auto& a, const auto& b) {
            return a.second < b.second || (a.second == b.second && a.first < b.first);
        });

        // Assign canonical codes.
        std::vector<EncEntry> entries;
        entries.reserve(symLens.size());
        uint32_t code    = 0;
        uint8_t  prevLen = symLens[0].second;
        for (auto& [sym, len] : symLens) {
            if (len > prevLen) {
                code <<= (len - prevLen);
                prevLen = len;
            }
            if (len > kMaxCodeBits) {
                throw std::runtime_error("HuffmanEncoder: code length exceeds kMaxCodeBits");
            }
            entries.push_back({sym, len, code});
            ++code;
        }

        return entries;
    }

    // -------------------------------------------------------------------------
    // Build decode tables from a canonical entry list.
    // -------------------------------------------------------------------------
    static void buildDecodeTables(
        const std::vector<EncEntry>& entries,
        std::vector<FastEntry>& fastTable,
        std::vector<SlowEntry>& slowTable,
        uint8_t& maxCodeLen)
    {
        static constexpr uint32_t kFastSize = 1u << kMaxFastBits;
        fastTable.assign(kFastSize, {T{}, kSlowPathLen});
        slowTable.clear();

        maxCodeLen = 0;
        for (auto& e : entries) {
            maxCodeLen = std::max(maxCodeLen, e.len);
            if (e.len <= kMaxFastBits) {
                // Fill all slots that share this code as a prefix.
                const uint32_t base  = e.code << (kMaxFastBits - e.len);
                const uint32_t count = 1u << (kMaxFastBits - e.len);
                const FastEntry fe{e.sym, e.len};
                for (uint32_t k = 0; k < count; ++k) {
                    fastTable[base + k] = fe;
                }
            } else {
                slowTable.push_back({e.code, e.len, e.sym});
            }
        }
    }

    // -------------------------------------------------------------------------
    // Decode a single symbol from the bit reader.
    // -------------------------------------------------------------------------
    [[nodiscard]] static T decodeOneSymbol(
        BitReader& br,
        const std::vector<FastEntry>& fastTable,
        const std::vector<SlowEntry>& slowTable,
        uint8_t maxCodeLen)
    {
        br.refill();

        if (br.avail >= kMaxFastBits) [[likely]] {
            const FastEntry& fe = fastTable[br.peek(kMaxFastBits)];
            if (fe.len != kSlowPathLen) [[likely]] {
                // Fast path: code fits in kMaxFastBits bits.
                br.consume(static_cast<int>(fe.len));
                return fe.sym;
            }
            // Slow path: code is longer than kMaxFastBits.
            // We already have kMaxFastBits bits; extend bit by bit.
            uint32_t code = br.peek(kMaxFastBits);
            br.consume(kMaxFastBits);
            uint8_t len = kMaxFastBits;
            while (len <= maxCodeLen) {
                if (br.avail == 0) br.refill();
                code = (code << 1) | br.peek(1);
                br.consume(1);
                ++len;
                for (const auto& se : slowTable) {
                    if (se.len == len && se.code == code) return se.sym;
                }
            }
        } else {
            // End-of-stream: fewer than kMaxFastBits bits remain.
            // Decode bit by bit, checking both tables.
            uint32_t code = 0;
            uint8_t  len  = 0;
            while (len <= maxCodeLen) {
                if (br.avail == 0) {
                    br.refill();
                    if (br.avail == 0) break;
                }
                code = (code << 1) | br.peek(1);
                br.consume(1);
                ++len;
                if (len <= kMaxFastBits) {
                    const FastEntry& fe = fastTable[code << (kMaxFastBits - len)];
                    if (fe.len == len) return fe.sym;
                } else {
                    for (const auto& se : slowTable) {
                        if (se.len == len && se.code == code) return se.sym;
                    }
                }
            }
        }
        throw std::runtime_error("HuffmanEncoder::decodeAll: invalid or truncated bitstream");
    }

public:
    // -------------------------------------------------------------------------
    // encode
    // -------------------------------------------------------------------------
    EncodedData encode(std::span<const T> data) override {
        EncodedData result;
        result.metadata().elementCount      = data.size();
        result.metadata().encodingName      = name();
        result.metadata().supportsRandomAccess = false;

        if (data.empty()) {
            result.data().resize(kHeaderFixed, 0);
            return result;
        }

        // Build canonical code table.
        const std::vector<EncEntry> entries = buildCanonical({data.begin(), data.end()});
        const size_t numSymbols = entries.size();

        // Build encode lookup: symbol → (code, len).
        ankerl::unordered_dense::map<T, std::pair<uint32_t, uint8_t>> encTable;
        encTable.reserve(numSymbols);
        for (auto& e : entries) encTable[e.sym] = {e.code, e.len};

        // Compute total payload bits.
        uint64_t totalBits = 0;
        for (T v : data) totalBits += encTable[v].second;

        // Allocate output: fixed header + symbol table + payload.
        const size_t symTableBytes = numSymbols * (sizeof(T) + 1);
        const size_t headerBytes   = kHeaderFixed + symTableBytes;
        const size_t payloadBytes  = (totalBits + 7) / 8;

        std::vector<uint8_t> out;
        out.reserve(headerBytes + payloadBytes);
        out.resize(headerBytes);

        // Write fixed header.
        uint8_t* h = out.data();
        const uint64_t ne = static_cast<uint64_t>(data.size());
        const uint16_t ns = static_cast<uint16_t>(numSymbols);
        std::memcpy(h, &ne, 8);        h += 8;
        std::memcpy(h, &ns, 2);        h += 2;
        std::memcpy(h, &totalBits, 8); h += 8;

        // Write canonical symbol table (sorted by len ASC, sym ASC).
        for (auto& e : entries) {
            std::memcpy(h, &e.sym, sizeof(T)); h += sizeof(T);
            *h++ = e.len;
        }

        // Encode payload as MSB-first bit stream.
        BitWriter bw{out};
        for (T v : data) {
            auto [code, len] = encTable[v];
            bw.write(code, static_cast<int>(len));
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
        const uint8_t* p   = encoded.data().data();
        const size_t   total = encoded.data().size();
        if (total < kHeaderFixed) return {};

        uint64_t numElements;
        uint16_t numSymbols16;
        uint64_t totalPayloadBits;
        std::memcpy(&numElements,     p, 8); p += 8;
        std::memcpy(&numSymbols16,    p, 2); p += 2;
        std::memcpy(&totalPayloadBits, p, 8); p += 8;

        const size_t numSymbols = numSymbols16;
        if (numElements == 0 || numSymbols == 0) return {};

        // Parse canonical symbol table and reconstruct codes.
        std::vector<EncEntry> entries;
        entries.reserve(numSymbols);
        for (size_t i = 0; i < numSymbols; ++i) {
            T       sym;
            uint8_t len;
            std::memcpy(&sym, p, sizeof(T)); p += sizeof(T);
            len = *p++;
            entries.push_back({sym, len, 0});
        }
        {
            uint32_t code    = 0;
            uint8_t  prevLen = entries[0].len;
            for (auto& e : entries) {
                if (e.len > prevLen) {
                    code <<= (e.len - prevLen);
                    prevLen = e.len;
                }
                e.code = code;
                ++code;
            }
        }

        // Build decode tables.
        std::vector<FastEntry> fastTable;
        std::vector<SlowEntry> slowTable;
        uint8_t maxCodeLen = 0;
        buildDecodeTables(entries, fastTable, slowTable, maxCodeLen);

        // Decode bit stream.
        const uint8_t* payloadStart = p;
        const uint8_t* payloadEnd   = encoded.data().data() + encoded.data().size();
        BitReader br{payloadStart, payloadEnd};

        std::vector<T> result;
        result.reserve(static_cast<size_t>(numElements));
        for (uint64_t i = 0; i < numElements; ++i) {
            result.push_back(decodeOneSymbol(br, fastTable, slowTable, maxCodeLen));
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Random access — not supported
    // -------------------------------------------------------------------------
    std::optional<T> decodeAt(const EncodedData& encoded, size_t index) override {
        const auto all = decodeAll(encoded);
        if (index >= all.size()) {
            return std::nullopt;
        }
        return all[index];
    }

    std::vector<T> decodeRange(const EncodedData& encoded, size_t start, size_t end) override {
        if (start >= end) {
            return {};
        }

        auto all = decodeAll(encoded);
        if (start >= all.size()) {
            return {};
        }

        end = std::min(end, all.size());
        return std::vector<T>(all.begin() + static_cast<std::ptrdiff_t>(start),
                              all.begin() + static_cast<std::ptrdiff_t>(end));
    }

    // -------------------------------------------------------------------------
    // Metadata
    // -------------------------------------------------------------------------
    EncodingType encodingType() const override {
        return EncodingType::HuffmanEncoding;
    }

    std::string name() const override { return "Huffman"; }

    EncodingProperties properties() const override {
        return EncodingProperties(EncodingProperty::Lossless)
             | EncodingProperty::PreservesOrder
            //  | EncodingProperty::SequentialOnly
             | EncodingProperty::RequiresFullData
             | EncodingProperty::EntropyCoding
             | EncodingProperty::VariableSize
             | EncodingProperty::RandomAccess; // Note: RandomAccess is technically false since we don't support decodeAt/decodeRange efficiently, but it signals that the encoding can represent any sequence without restrictions, and that the entire data is needed to decode any part of it.
    }
};

} // namespace encodings::encoders
