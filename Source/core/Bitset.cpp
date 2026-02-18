#include "Bitset.hpp"
#include "encoders/RawEncoder.hpp"

namespace encodings::core {
    std::shared_ptr<encodings::Codec<uint64_t>> FastBitset::createDefaultCodec()
    {
        return std::make_shared<encodings::encoders::RawEncoder<uint64_t>>();
    }
} // namespace encodings::core
