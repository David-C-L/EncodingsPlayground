/**
 * @file test_subint_encoder.cpp
 * @brief Tests for SubIntEncoder<int32_t> and SubIntEncoder<int64_t>
 *
 * Build (normal):  cmake --build build --target test_subint_encoder
 * Build (debug):   cmake --build build --target test_subint_encoder_dbg
 *   The _dbg target compiles with -DSUBINT_DEBUG=1 which enables per-element
 *   encode/decode trace output via the SUBINT_DBG() macro in SubIntEncoder.hpp.
 */

// The _dbg CMake target defines SUBINT_DEBUG=1 on the command line.
// Default to 0 here so the normal target is quiet.
#ifndef SUBINT_DEBUG
#  define SUBINT_DEBUG 0
#endif
#include "encoders/SubIntEncoder.hpp"
#include "generators/SnowflakeIDGenerator.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace encodings::encoders;
using namespace encodings::datagen;

// ── helpers ────────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (cond) { std::cout << "  ✓ PASS  " << msg << "\n"; ++g_pass; } \
        else      { std::cout << "  ✗ FAIL  " << msg << "\n"; ++g_fail; } \
    } while (0)

template<typename T>
static bool roundtrip(const std::string& label,
                      SplitMode sm,
                      const std::vector<T>& data,
                      bool verbose = false)
{
    SubIntConfig<T> cfg;
    cfg.splitMode = sm;
    SubIntEncoder<T> enc(cfg);

    std::cout << "\n--- " << label << " (n=" << data.size() << ") ---\n";

    auto encoded = enc.encode(std::span<const T>(data));

    if (verbose) {
        std::cout << "  encoded " << encoded.size() << " bytes\n";
        // print first 64 bytes of encoded buffer
        std::cout << "  header hex: ";
        for (size_t i = 0; i < std::min(encoded.size(), size_t{64}); ++i)
            printf("%02X ", encoded.data()[i]);
        std::cout << "\n";
    }

    auto decoded = enc.decodeAll(encoded);

    bool ok = (decoded.size() == data.size());
    for (size_t i = 0; ok && i < data.size(); ++i)
        if (decoded[i] != data[i]) ok = false;

    if (!ok) {
        std::cout << "  MISMATCH details:\n";
        for (size_t i = 0; i < data.size(); ++i) {
            bool match = (i < decoded.size() && decoded[i] == data[i]);
            if (!match) {
                std::cout << "    [" << i << "]  expected 0x"
                          << std::hex << std::setw(sizeof(T)*2)
                          << std::setfill('0') << (uint64_t)(typename std::make_unsigned<T>::type)data[i]
                          << "  got 0x"
                          << std::setw(sizeof(T)*2)
                          << (i < decoded.size()
                              ? (uint64_t)(typename std::make_unsigned<T>::type)decoded[i]
                              : uint64_t(-1))
                          << std::dec << std::setfill(' ') << "\n";
            }
        }
    }

    CHECK(ok, label);
    return ok;
}

// ── int32 tests ─────────────────────────────────────────────────────────────

void test_int32_split13() {
    std::vector<int32_t> data = {
        0x01020304, 0x01020305, 0x02020304, 0x01030304,
        0x00000000, 0x7FFFFFFF, (int32_t)0x80000000, (int32_t)0xFFFFFFFF,
    };
    roundtrip<int32_t>("int32 Split{1,3}", Split13(), data, true);
}

void test_int32_split22() {
    std::vector<int32_t> data = {
        0x01020304, 0x01020305, 0x02020304, 0x01030304,
        0x00000000, 0x7FFFFFFF, (int32_t)0x80000000, (int32_t)0xFFFFFFFF,
    };
    roundtrip<int32_t>("int32 Split{2,2}", Split22(), data, true);
}

void test_int32_split31() {
    std::vector<int32_t> data = {
        0x01020304, 0x01020305, 0x02020304, 0x01030304,
        0x00000000, 0x7FFFFFFF, (int32_t)0x80000000, (int32_t)0xFFFFFFFF,
    };
    roundtrip<int32_t>("int32 Split{3,1}", Split31(), data, true);
}

// ── int64 tests ─────────────────────────────────────────────────────────────

// A simple set of 10 values that exercise distinct left+right groups for every split
static const std::vector<int64_t> kSimple10 = {
    0x0102030405060708LL,
    0x0102030405060709LL,
    0x0102030405060710LL,
    0x0102030405060718LL,
    0x0102030405060720LL,
    0x0102030405070708LL,
    0x0102030405080708LL,
    0x0102030406060708LL,
    0x0102030505060708LL,
    0x0102040405060708LL,
};

void test_int64_split17() { roundtrip<int64_t>("int64 Split{1,7}", Split17(), kSimple10, true); }
void test_int64_split26() { roundtrip<int64_t>("int64 Split{2,6}", Split26(), kSimple10, true); }
void test_int64_split35() { roundtrip<int64_t>("int64 Split{3,5}", Split35(), kSimple10, true); }
void test_int64_split44() { roundtrip<int64_t>("int64 Split{4,4}", Split44(), kSimple10, true); }
void test_int64_split53() { roundtrip<int64_t>("int64 Split{5,3}", Split53(), kSimple10, true); }
void test_int64_split62() { roundtrip<int64_t>("int64 Split{6,2}", Split62(), kSimple10, true); }
void test_int64_split71() { roundtrip<int64_t>("int64 Split{7,1}", Split71(), kSimple10, true); }

// Extreme values
void test_int64_extremes() {
    std::vector<int64_t> data = {
        0LL, 1LL, -1LL,
        (int64_t)0x7FFFFFFFFFFFFFFFLL,
        (int64_t)0x8000000000000000LL,
        (int64_t)0xFFFFFFFFFFFFFFFFLL,
        (int64_t)0x0102030405060708LL,
        (int64_t)0xFEFDFCFBFAF9F8F7LL,
    };
    roundtrip<int64_t>("int64 extremes Split{4,4}", Split44(), data, true);
    roundtrip<int64_t>("int64 extremes Split{1,7}", Split17(), data, true);
    roundtrip<int64_t>("int64 extremes Split{7,1}", Split71(), data, true);
}

// Construction-time validation
void test_invalid_split() {
    std::cout << "\n--- Construction-time validation ---\n";
    bool threw = false;
    try {
        SubIntEncoder<int32_t> bad(SubIntConfig<int32_t>{ Split17() });
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "int32 with Split17 (1+7≠4) throws");

    threw = false;
    try {
        SubIntEncoder<int64_t> bad(SubIntConfig<int64_t>{ Split22() });
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "int64 with Split22 (2+2≠8) throws");
}

// Stress test: large n forces wider code widths (tests 32-bit code path)
void test_int64_large_n() {
    // 70 000 values with fully distinct right-7-byte groups forces rightCW=32
    // (bit_width(69999)=17, rounds up to 32)
    const size_t N = 70000;
    std::vector<int64_t> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i)
        data.push_back(static_cast<int64_t>((uint64_t(i + 1) << 8) | 0xAB));
    // left byte = 0xAB (same for all → leftCW=1, leftRaw=false for Split17)
    // right 7 bytes = distinct for each → rightCW=32 for Split17

    std::cout << "\n--- int64 large-n Split{1,7} (n=" << N << ", forces rightCW=32) ---\n";
    roundtrip<int64_t>("int64 large-n Split{1,7}", Split17(), data, /*verbose=*/false);
    roundtrip<int64_t>("int64 large-n Split{7,1}", Split71(), data, /*verbose=*/false);
    roundtrip<int64_t>("int64 large-n Split{4,4}", Split44(), data, /*verbose=*/false);
}

// Force raw fallback on the LEFT side with lb > 4 (the actual bug path).
// Split{5,3}: lb=5, lMaxCW=32. With N>2^32 distinct left groups raw kicks in,
// but we can force it more cheaply with N=70000 distinct left-5-byte values.
// bit_width(69999)=17, rounds up to 32 — still fits in uint32_t, no raw yet.
// To truly force leftRaw we need > 2^32 distinct left groups — not practical.
// Instead, verify that lb=5 raw writes work correctly by encoding values whose
// left-5-byte group cannot be represented in a code dictionary at all: use the
// overflow path by providing more than lCap = 2^32 distinct groups.  That isn't
// feasible in a unit test, so instead we verify correctness with many distinct
// left groups that DO fit in dict (32-bit codes) and confirm round-trip.
void test_int64_raw_large_lb() {
    // For Split{5,3}: lb=5, rb=3. With N=70000 distinct left-5-byte values
    // AND N=70000 distinct right-3-byte values: both require 32-bit codes
    // (bit_width(69999)=17 → rounds to 32).  This is the widest code path
    // WITHOUT raw.  We test correctness at the 32-bit code boundary.
    const size_t N = 70000;
    std::vector<int64_t> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        // Each value has a unique left-5-byte AND unique right-3-byte group.
        const uint64_t lv = (uint64_t(i + 1) & 0xFFFFFFFFFFULL);  // 5 bytes
        const uint64_t rv = uint64_t(i + 1) & 0xFFFFFFULL;         // 3 bytes
        data.push_back(static_cast<int64_t>((lv << 24) | rv));
    }
    roundtrip<int64_t>("int64 32-bit-code Split{5,3}", Split53(), data);

    // Force rightRaw on Split{5,3}: make the right-3-byte group overflow (>16M distinct).
    // 3 bytes = max 2^24=16M values → we can't exceed that with reasonable N.
    // Instead, force rightRaw directly by providing all 256 left-1-byte groups and
    // many distinct right-7-byte groups using Split{1,7} (rb=7, rMaxCW=32):
    // with N=70001 distinct right-7-byte groups: rightCW=32, no raw needed.
    // Nothing further to add here — the large-n test above already covers this.

    // Snowflake Split{6,2} with n=100000 triggers rightRaw (right overflow warning).
    // We verify that encode+decode still produces correct values after the fix.
    SnowflakeIDGenerator<int64_t> gen(INSTAGRAM_SNOWFLAKE_CONFIG, 4096, 42, 0.05);
    auto sf = gen.generate(100000);
    roundtrip<int64_t>("Snowflake Split{6,2} rightRaw n=100000", Split62(), sf);
    roundtrip<int64_t>("Snowflake Split{5,3} rightRaw n=100000", Split53(), sf);
    roundtrip<int64_t>("Snowflake Split{4,4} raw n=100000",     Split44(), sf);
}


void test_int64_snowflake() {
    for (size_t n : {1000UL, 100000UL, 1000000UL}) {
        SnowflakeIDGenerator<int64_t> gen(INSTAGRAM_SNOWFLAKE_CONFIG, 4096, 42, 0.05);
        auto data = gen.generate(n);

        roundtrip<int64_t>("Snowflake Split{1,7} n=" + std::to_string(n), Split17(), data);
        roundtrip<int64_t>("Snowflake Split{2,6} n=" + std::to_string(n), Split26(), data);
        roundtrip<int64_t>("Snowflake Split{3,5} n=" + std::to_string(n), Split35(), data);
        roundtrip<int64_t>("Snowflake Split{4,4} n=" + std::to_string(n), Split44(), data);
        roundtrip<int64_t>("Snowflake Split{5,3} n=" + std::to_string(n), Split53(), data);
        roundtrip<int64_t>("Snowflake Split{6,2} n=" + std::to_string(n), Split62(), data);
        roundtrip<int64_t>("Snowflake Split{7,1} n=" + std::to_string(n), Split71(), data);
    }
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== SubIntEncoder Tests ===\n";

    test_int32_split13();
    test_int32_split22();
    test_int32_split31();

    test_int64_split17();
    test_int64_split26();
    test_int64_split35();
    test_int64_split44();
    test_int64_split53();
    test_int64_split62();
    test_int64_split71();

    test_int64_extremes();
    test_int64_large_n();
    test_int64_raw_large_lb();
    test_int64_snowflake();
    test_invalid_split();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return (g_fail == 0) ? 0 : 1;
}
