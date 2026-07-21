#include <iostream>
#include <algorithm>
#include <set>
#include "generators/TPCHLineitemPartKeyGenerator.hpp"

using namespace encodings::datagen;

void analyzeDistribution(const std::vector<int32_t>& data) {
    if (data.empty()) return;
    
    auto minVal = *std::min_element(data.begin(), data.end());
    auto maxVal = *std::max_element(data.begin(), data.end());
    
    std::set<int32_t> unique(data.begin(), data.end());
    
    // Count frequency of top 10 values
    std::map<int32_t, int> freq;
    for (auto val : data) {
        freq[val]++;
    }
    
    std::vector<std::pair<int32_t, int>> sortedFreq(freq.begin(), freq.end());
    std::sort(sortedFreq.begin(), sortedFreq.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "  Min: " << minVal << "\n";
    std::cout << "  Max: " << maxVal << "\n";
    std::cout << "  Unique values: " << unique.size() << "\n";
    std::cout << "  Top 5 most frequent:\n";
    for (size_t i = 0; i < std::min(size_t(5), sortedFreq.size()); ++i) {
        std::cout << "    PartKey " << sortedFreq[i].first 
                  << ": " << sortedFreq[i].second << " times ("
                  << (100.0 * sortedFreq[i].second / data.size()) << "%)\n";
    }
}

int main() {
    std::cout << "=== TPC-H Lineitem L_PARTKEY Generator Tests ===\n\n";
    
    // Test 1: Small scale factor with uniform distribution
    {
        std::cout << "Test 1: SF=0.01 (2,000 parts), Uniform distribution\n";
        TPCHLineitemPartKeyGenerator gen(0.01, 42, 0.0);
        auto data = gen.generate(10000);
        
        std::cout << "Generated 10,000 part keys\n";
        std::cout << "Cardinality: " << gen.getCardinality() << "\n";
        analyzeDistribution(data);
        std::cout << "\n";
    }
    
    // Test 2: Scale factor 1.0 with Zipfian skew
    {
        std::cout << "Test 2: SF=1.0 (200,000 parts), Zipfian skew=0.3\n";
        TPCHLineitemPartKeyGenerator gen(1.0, 42, 0.3);
        auto data = gen.generate(10000);
        
        std::cout << "Generated 10,000 part keys\n";
        std::cout << "Cardinality: " << gen.getCardinality() << "\n";
        analyzeDistribution(data);
        std::cout << "\n";
    }
    
    // Test 3: Strong Zipfian skew
    {
        std::cout << "Test 3: SF=1.0, Strong Zipfian skew=1.0\n";
        TPCHLineitemPartKeyGenerator gen(1.0, 42, 1.0);
        auto data = gen.generate(10000);
        
        std::cout << "Generated 10,000 part keys\n";
        std::cout << "Note: This will be slow due to Zipfian computation\n";
        analyzeDistribution(data);
        std::cout << "\n";
    }
    
    // Test 4: Clustered distribution
    {
        std::cout << "Test 4: SF=1.0, Clustered (cluster_size=100, hot_parts=1000)\n";
        TPCHLineitemPartKeyClusteredGenerator gen(1.0, 42, 100, 1000);
        auto data = gen.generate(10000);
        
        std::cout << "Generated 10,000 part keys\n";
        analyzeDistribution(data);
        std::cout << "\n";
    }
    
    // Test 5: Very small dataset
    {
        std::cout << "Test 5: SF=1.0, Uniform, Small dataset (100 values)\n";
        TPCHLineitemPartKeyGenerator gen(1.0, 123, 0.0); // zipfExponent = 0.0 for uniform
        auto data = gen.generate(100);
        
        std::cout << "Generated 100 part keys\n";
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), data.size()); ++i) {
            std::cout << data[i];
            if (i < std::min(size_t(20), data.size()) - 1) std::cout << ", ";
        }
        std::cout << "\n\n";
    }
    
    std::cout << "=== All tests complete ===\n";
    std::cout << "\nRecommendations:\n";
    std::cout << "  - Use uniform (zipf=0.0) for general testing\n";
    std::cout << "  - Use zipf=0.3 for realistic TPC-H workloads\n";
    std::cout << "  - Use clustered generator for temporal locality patterns\n";
    std::cout << "  - SF=0.01 for quick tests, SF=1.0 for standard benchmarks\n";
    
    return 0;
}
