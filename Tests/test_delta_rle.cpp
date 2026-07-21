/**
 * @file test_delta_rle.cpp
 * @brief Test the DeltaRunLengthEncoder
 */

#include "encoders/DeltaRunLengthEncoder.hpp"
#include "GatherTestHelpers.hpp"
#include <iostream>
#include <vector>

using namespace encodings::encoders;

void testSequentialData() {
    std::cout << "Test 1: Sequential data (constant delta = 1)\n";
    std::vector<int32_t> data = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    
    DeltaRunLengthEncoder<int32_t> encoder;
    auto encoded = encoder.encode(data);
    
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size:  " << encoded.size() << " bytes\n";
    std::cout << "  Num runs:      " << encoded.metadata().customMetadata["num_runs"] << "\n";
    std::cout << "  Compression:   " << encoded.metadata().customMetadata["compression_ratio"] << "\n";
    
    auto decoded = encoder.decodeAll(encoded);
    bool match = (data == decoded);
    std::cout << "  Decode match:  " << (match ? "✓ PASS" : "✗ FAIL") << "\n\n";
}

void testPlateauData() {
    std::cout << "Test 2: Plateau data (alternating delta = 0 and 5)\n";
    std::vector<int32_t> data = {0, 0, 0, 0, 5, 5, 5, 10, 10, 10, 10, 10};
    
    DeltaRunLengthEncoder<int32_t> encoder;
    auto encoded = encoder.encode(data);
    
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size:  " << encoded.size() << " bytes\n";
    std::cout << "  Num runs:      " << encoded.metadata().customMetadata["num_runs"] << "\n";
    std::cout << "  Compression:   " << encoded.metadata().customMetadata["compression_ratio"] << "\n";
    
    auto decoded = encoder.decodeAll(encoded);
    bool match = (data == decoded);
    std::cout << "  Decode match:  " << (match ? "✓ PASS" : "✗ FAIL") << "\n\n";
}

void testRandomAccess() {
    std::cout << "Test 3: Random access\n";
    std::vector<int32_t> data = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    
    DeltaRunLengthEncoder<int32_t> encoder;
    auto encoded = encoder.encode(data);
    
    std::cout << "  Testing decodeAt(5): ";
    auto val = encoder.decodeAt(encoded, 5);
    bool match = (val.has_value() && val.value() == 30);
    std::cout << (match ? "✓ PASS" : "✗ FAIL") << " (got " << (val.has_value() ? std::to_string(val.value()) : "nullopt") << ", expected 30)\n";
    
    std::cout << "  Testing decodeRange(3, 7): ";
    auto range = encoder.decodeRange(encoded, 3, 7);
    std::vector<int32_t> expected = {20, 25, 30, 35};
    match = (range == expected);
    std::cout << (match ? "✓ PASS" : "✗ FAIL") << "\n\n";
}

void testWorstCase() {
    std::cout << "Test 4: Worst case (every delta different)\n";
    std::vector<int32_t> data = {1, 2, 4, 7, 11, 16, 22, 29, 37, 46};
    
    DeltaRunLengthEncoder<int32_t> encoder;
    auto encoded = encoder.encode(data);
    
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size:  " << encoded.size() << " bytes\n";
    std::cout << "  Num runs:      " << encoded.metadata().customMetadata["num_runs"] << "\n";
    std::cout << "  Compression:   " << encoded.metadata().customMetadata["compression_ratio"] << "\n";
    
    auto decoded = encoder.decodeAll(encoded);
    bool match = (data == decoded);
    std::cout << "  Decode match:  " << (match ? "✓ PASS" : "✗ FAIL") << "\n\n";
}

bool testGatherAllTraceShapes() {
    std::cout << "Test 5: decodeGatherInto against decodeAll() across trace shapes\n";
    bool allOk = true;
    for (auto& data : std::vector<std::vector<int32_t>>{
             {5, 10, 15, 20, 25, 30, 35, 40, 45, 50},              // sequential deltas
             {0, 0, 0, 0, 5, 5, 5, 10, 10, 10, 10, 10},            // plateau
             {1, 2, 4, 7, 11, 16, 22, 29, 37, 46},                 // worst case, unique deltas
         }) {
        DeltaRunLengthEncoder<int32_t> encoder;
        auto encoded = encoder.encode(data);
        bool ok = encodings::testutil::checkGatherAllTraceShapes<int32_t>(encoder, encoded, data);
        allOk &= ok;
    }
    std::cout << "  " << (allOk ? "✓ PASS" : "✗ FAIL") << "\n\n";
    return allOk;
}

int main() {
    std::cout << "=== DeltaRunLengthEncoder Tests ===\n\n";

    testSequentialData();
    testPlateauData();
    testRandomAccess();
    testWorstCase();
    bool gatherOk = testGatherAllTraceShapes();

    std::cout << "All tests complete!\n";
    return gatherOk ? 0 : 1;
}
