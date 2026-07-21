#include "core/Bitset.hpp"
#include "encoders/RawEncoder.hpp"
#include "encoders/ZstdEncoder.hpp"
#include <iostream>
#include <cassert>

using namespace encodings;
using namespace encodings::core;
using namespace encodings::encoders;

void testBasicOperations() {
    std::cout << "Testing basic operations..." << std::endl;
    
    FastBitset bitset(100); // Default: valuesPerBit = 1
    
    assert(bitset.size() == 100);
    assert(bitset.sizeInBits() == 100);
    assert(!bitset.test(10));
    
    bitset.set(10);
    assert(bitset.test(10));
    assert(bitset.numSetBits() == 1);
    assert(bitset.numSetValues() == 1);
    
    bitset.set(20);
    bitset.set(30);
    assert(bitset.numSetBits() == 3);
    
    bitset.reset(20);
    assert(!bitset.test(20));
    assert(bitset.numSetBits() == 2);
    
    bitset.clear();
    assert(bitset.numSetBits() == 0);
    
    std::cout << "  ✓ Basic operations passed" << std::endl;
}

void testTestAndSet() {
    std::cout << "Testing test_and_set..." << std::endl;
    
    FastBitset bitset(100);
    
    // First set should return false (bit was not set)
    bool wasSet = bitset.test_and_set(42);
    if (wasSet) {
        std::cerr << "Error: test_and_set returned true on first set" << std::endl;
    }
    assert(!wasSet);
    assert(bitset.test(42));
    
    // Second set should return true (bit was already set)
    wasSet = bitset.test_and_set(42);
    if (!wasSet) {
        std::cerr << "Error: test_and_set returned false on second set" << std::endl;
    }
    assert(wasSet);
    assert(bitset.test(42));
    
    std::cout << "  ✓ test_and_set passed" << std::endl;
}

void testRangeOperations() {
    std::cout << "Testing range operations..." << std::endl;
    
    FastBitset bitset(1000);
    
    // Set range
    bitset.setRange(100, 50);
    assert(bitset.test(100));
    assert(bitset.test(125));
    assert(bitset.test(149));
    assert(!bitset.test(99));
    assert(!bitset.test(150));
    
    // Test range
    assert(bitset.testRange(100, 50));
    assert(!bitset.testRange(100, 51)); // One bit outside
    assert(!bitset.testRange(99, 50));  // Starts one bit early
    
    std::cout << "  ✓ Range operations passed" << std::endl;
}

void testValueGrouping() {
    std::cout << "Testing value grouping..." << std::endl;
    
    // 2 values per bit
    FastBitset bitset2(100, 2);
    assert(bitset2.size() == 100);
    assert(bitset2.sizeInBits() == 50); // 100 values / 2 per bit
    
    bitset2.set(10);
    assert(bitset2.test(10));
    assert(bitset2.test(11)); // Both map to same bit
    
    bitset2.set(11);
    assert(bitset2.numSetBits() == 1); // Still just 1 bit
    assert(bitset2.numSetValues() == 2); // But 2 values
    
    // 4 values per bit
    FastBitset bitset4(100, 4);
    assert(bitset4.sizeInBits() == 25); // 100 values / 4 per bit
    
    bitset4.set(8);  // Values 8-11 map to bit 2
    bitset4.set(10); // Same bit
    assert(bitset4.numSetBits() == 1);
    assert(bitset4.numSetValues() == 4);
    
    std::cout << "  ✓ Value grouping passed" << std::endl;
}

void testEncodeDecodeRaw() {
    std::cout << "Testing encode/decode with RawEncoder..." << std::endl;
    
    FastBitset original(1000);
    original.set(10);
    original.set(100);
    original.set(500);
    original.set(999);
    
    size_t originalSetBits = original.numSetBits();
    if (originalSetBits != 4) {
        std::cerr << "Error: Expected 4 set bits, got " << originalSetBits << std::endl;
    }
    assert(originalSetBits == 4);

    
    // Encode using default RawEncoder
    EncodedData encoded = original.encode();
    
    std::cout << "  Original size: " << original.bits_.size() * sizeof(uint64_t) << " bytes" << std::endl;
    std::cout << "  Encoded size: " << encoded.size() << " bytes" << std::endl;
    std::cout << "  Metadata: " << encoded.metadata().encodingName << std::endl;
    
    // Decode
    FastBitset decoded = FastBitset::decode(encoded);
    
    assert(decoded.size() == original.size());
    assert(decoded.sizeInBits() == original.sizeInBits());
    assert(decoded.numSetBits() == originalSetBits);
    assert(decoded.test(10));
    assert(decoded.test(100));
    assert(decoded.test(500));
    assert(decoded.test(999));
    assert(!decoded.test(50));
    
    std::cout << "  ✓ RawEncoder encode/decode passed" << std::endl;
}

void testEncodeDecodeZstd() {
    std::cout << "Testing encode/decode with ZstdEncoder..." << std::endl;
    
    auto zstdCodec = std::make_shared<ZstdEncoder<uint64_t>>();
    FastBitset original(10000, 1, zstdCodec);
    
    // Set many bits to get better compression
    for (size_t i = 0; i < 10000; i += 10) {
        original.set(i);
    }
    
    size_t originalSetBits = original.numSetBits();
    if (originalSetBits != 1000) {
        std::cerr << "Error: Expected 1000 set bits, got " << originalSetBits << std::endl;
    }
    assert(originalSetBits == 1000);
    
    // Encode
    EncodedData encoded = original.encode();
    
    std::cout << "  Original size: " << original.bits_.size() * sizeof(uint64_t) << " bytes" << std::endl;
    std::cout << "  Encoded size: " << encoded.size() << " bytes" << std::endl;
    std::cout << "  Compression ratio: " 
              << (double)(original.bits_.size() * sizeof(uint64_t)) / encoded.size() 
              << "x" << std::endl;
    std::cout << "  Metadata: " << encoded.metadata().encodingName << std::endl;
    
    // Decode with codec
    FastBitset decoded = FastBitset::decode(encoded, zstdCodec);
    
    assert(decoded.size() == original.size());
    assert(decoded.numSetBits() == originalSetBits);
    
    // Verify some bits
    for (size_t i = 0; i < 10000; i += 10) {
        assert(decoded.test(i));
    }
    assert(!decoded.test(5));
    assert(!decoded.test(15));
    
    std::cout << "  ✓ ZstdEncoder encode/decode passed" << std::endl;
}

void testSparsePattern() {
    std::cout << "Testing sparse bitset pattern..." << std::endl;
    
    // Simulate null tracking in sparse columnar data
    FastBitset nullBitmap(10000);
    
    // Only 1% of values are present (99% sparse)
    for (size_t i = 0; i < 10000; i += 100) {
        nullBitmap.set(i);
    }
    
    assert(nullBitmap.numSetBits() == 100);
    
    // Encode with compression
    auto zstdCodec = std::make_shared<ZstdEncoder<uint64_t>>();
    FastBitset compressed(10000, 1, zstdCodec);
    for (size_t i = 0; i < 10000; i += 100) {
        compressed.set(i);
    }
    
    EncodedData encoded = compressed.encode();
    size_t uncompressedSize = compressed.bits_.size() * sizeof(uint64_t);
    
    std::cout << "  Sparsity: 99% empty" << std::endl;
    std::cout << "  Uncompressed: " << uncompressedSize << " bytes" << std::endl;
    std::cout << "  Compressed: " << encoded.size() << " bytes" << std::endl;
    std::cout << "  Ratio: " << (double)uncompressedSize / encoded.size() << "x" << std::endl;
    
    // Decode and verify
    FastBitset decoded = FastBitset::decode(encoded, zstdCodec);
    for (size_t i = 0; i < 10000; i++) {
        if (i % 100 == 0) {
            assert(decoded.test(i));
        } else {
            assert(!decoded.test(i));
        }
    }
    
    std::cout << "  ✓ Sparse pattern passed" << std::endl;
}

void testNullTracking() {
    std::cout << "Testing null tracking use case (map group keys)..." << std::endl;
    
    // Simulate 5 maps with 10 possible keys each
    // Not every map has every key
    const size_t numMaps = 5;
    
    // Key 0: present in maps 0, 2, 4
    FastBitset key0Presence(numMaps);
    key0Presence.set(0);
    key0Presence.set(2);
    key0Presence.set(4);
    
    // Key 1: present in all maps
    FastBitset key1Presence(numMaps);
    for (size_t i = 0; i < numMaps; i++) {
        key1Presence.set(i);
    }
    
    // Key 2: present in no maps
    FastBitset key2Presence(numMaps);
    
    assert(key0Presence.numSetBits() == 3);
    assert(key1Presence.numSetBits() == 5);
    assert(key2Presence.numSetBits() == 0);
    
    // Check specific map presence
    assert(key0Presence.test(0));
    assert(!key0Presence.test(1));
    assert(key0Presence.test(2));
    
    std::cout << "  ✓ Null tracking use case passed" << std::endl;
}

int main() {
    std::cout << "Running FastBitset tests...\n" << std::endl;
    
    try {
        testBasicOperations();
        testTestAndSet();
        testRangeOperations();
        testValueGrouping();
        testEncodeDecodeRaw();
        testEncodeDecodeZstd();
        testSparsePattern();
        testNullTracking();
        
        std::cout << "\n✓ All FastBitset tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
