#include <iostream>
#include <vector>
#include <cassert>
#include "encoders/BitParenthesesEncoder.hpp"

using namespace encodings::encoders;

void test_basic_encoding() {
    std::cout << "Test 1: Basic encoding [3, 5, 4]\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {3, 5, 4};
    
    auto encoded = encoder.encode(data);
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size: " << encoded.size() << " bytes\n";
    std::cout << "  Compression ratio: " << encoded.metadata().compressionRatio() << "x\n";
    
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded.size() == data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        assert(decoded[i] == data[i]);
    }
    std::cout << "  ✓ Decode successful: [" << decoded[0] << ", " << decoded[1] << ", " << decoded[2] << "]\n";
}

void test_random_access() {
    std::cout << "\nTest 2: Random access\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {3, 5, 4, 2, 7, 1};
    
    auto encoded = encoder.encode(data);
    
    for (size_t i = 0; i < data.size(); ++i) {
        auto value = encoder.decodeAt(encoded, i);
        assert(value.has_value());
        assert(value.value() == data[i]);
        std::cout << "  Index " << i << ": " << value.value() << " (expected " << data[i] << ") ✓\n";
    }
}

void test_range_access() {
    std::cout << "\nTest 3: Range access\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {3, 5, 4, 2, 7, 1, 6};
    
    auto encoded = encoder.encode(data);
    
    // Test range [2, 5)
    auto range = encoder.decodeRange(encoded, 2, 5);
    assert(range.size() == 3);
    assert(range[0] == 4 && range[1] == 2 && range[2] == 7);
    std::cout << "  Range [2, 5): [" << range[0] << ", " << range[1] << ", " << range[2] << "] ✓\n";
}

void test_repeated_values() {
    std::cout << "\nTest 4: Repeated values (good for RLE)\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {5, 5, 5, 5, 5, 3, 3, 3, 7, 7};
    
    auto encoded = encoder.encode(data);
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size: " << encoded.size() << " bytes\n";
    std::cout << "  Compression ratio: " << encoded.metadata().compressionRatio() << "x\n";
    
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded == data);
    std::cout << "  ✓ Decode successful\n";
}

void test_small_values() {
    std::cout << "\nTest 5: Small values (optimal case)\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {1, 2, 1, 3, 2, 1, 2, 3, 1, 1};
    
    auto encoded = encoder.encode(data);
    std::cout << "  Original size: " << data.size() * sizeof(int32_t) << " bytes\n";
    std::cout << "  Encoded size: " << encoded.size() << " bytes\n";
    std::cout << "  Compression ratio: " << encoded.metadata().compressionRatio() << "x\n";
    
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded == data);
    std::cout << "  ✓ Decode successful\n";
}

void test_empty() {
    std::cout << "\nTest 6: Empty data\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data;
    
    auto encoded = encoder.encode(data);
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded.empty());
    std::cout << "  ✓ Empty data handled correctly\n";
}

void test_single_element() {
    std::cout << "\nTest 7: Single element\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {42};
    
    auto encoded = encoder.encode(data);
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded.size() == 1 && decoded[0] == 42);
    std::cout << "  ✓ Single element: " << decoded[0] << "\n";
}

void test_zeros() {
    std::cout << "\nTest 8: Zeros (edge case)\n";
    
    BitParenthesesEncoder<int32_t> encoder;
    std::vector<int32_t> data = {0, 0, 5, 0, 3};
    
    auto encoded = encoder.encode(data);
    auto decoded = encoder.decodeAll(encoded);
    assert(decoded == data);
    std::cout << "  ✓ Zeros handled correctly: [";
    for (size_t i = 0; i < decoded.size(); ++i) {
        std::cout << decoded[i];
        if (i < decoded.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== BitParenthesesEncoder Test Suite ===\n\n";
    
    try {
        test_basic_encoding();
        test_random_access();
        test_range_access();
        test_repeated_values();
        test_small_values();
        test_empty();
        test_single_element();
        test_zeros();
        
        std::cout << "\n=== All tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
