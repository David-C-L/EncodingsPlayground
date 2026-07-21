#include <iostream>
#include "encoders/BitParenthesesEncoder.hpp"

using namespace encodings::encoders;

int main() {
    BitParenthesesEncoder<int32_t> encoder;
    
    std::vector<int32_t> largeValues = {1000, 2000, 1500, 3000, 2500};
    
    auto encoded = encoder.encode(largeValues);
    
    std::cout << "Large values: [1000, 2000, 1500, 3000, 2500]\n";
    std::cout << "Total 1s + 0s: " << (1000+1 + 2000+1 + 1500+1 + 3000+1 + 2500+1) << " bits\n";
    std::cout << "Bytes needed: " << (10005 + 7) / 8 << " bytes (for bit vector)\n";
    std::cout << "Boundaries: " << 5 * 4 << " bytes\n";
    std::cout << "Header: 8 bytes\n";
    std::cout << "Expected total: ~" << (1251 + 20 + 8) << " bytes\n\n";
    
    std::cout << "Actual encoded size: " << encoded.size() << " bytes\n";
    std::cout << "Raw encoding would be: " << 5 * sizeof(int32_t) + sizeof(size_t) << " bytes\n\n";
    
    std::cout << "Analysis:\n";
    std::cout << "- BitParentheses stores " << 1251 << " bytes of mostly-ones bit vector\n";
    std::cout << "- With RLE on the bit vector, those long runs of 1s would compress well!\n";
    std::cout << "- Example: 1000 ones → 1 RLE entry (start=0, value=1, length=1000)\n";
    std::cout << "- 5 values → ~10 RLE entries → ~80 bytes instead of 1251 bytes!\n";
    
    return 0;
}
