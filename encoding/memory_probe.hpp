#pragma once
#include "core/execution/multimodal_engine.hpp"
#include <map>
#include <string>
#include <vector>

namespace nca::encoding {

/**
 * MemoryProbe
 * Performs 'Expert Attribution' to localize knowledge within the silicon.
 * Helps identify where specific bit-patterns are stored in the 1024 expert pool.
 */
class MemoryProbe {
public:
    explicit MemoryProbe(std::shared_ptr<nca::execution::MultimodalEngine> engine);

    /**
     * Scans the expert pool for a specific alphabet combination.
     * Returns a 'Saliency Score' for each expert.
     */
    std::vector<float> attribute_knowledge(const std::string& bit_pattern);

    /**
     * Generates a summary of the 'Silicon Memory Map'.
     */
    std::map<std::string, std::vector<int>> map_foundation_sectors();

private:
    std::shared_ptr<nca::execution::MultimodalEngine> engine_;
};

} // namespace nca::encoding
