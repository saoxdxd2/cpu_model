#pragma once
#include <string>
#include <vector>
#include <filesystem>

namespace nca::automation {

/**
 * SiliconAutomation
 * C++ based automation engine to replace slow Python scripts.
 * Performs high-speed batch operations (Refactoring, Build Mgmt, Git Ops).
 */
class SiliconAutomation {
public:
    SiliconAutomation(const std::string& workspace_root);

    // BATCH REFACTORING: Replaces code patterns across the entire workspace in O(1) sweeps.
    void batch_refactor(const std::string& pattern, const std::string& replacement);

    // BUILD ORCHESTRATION: High-speed parallel build management.
    void trigger_fast_build(const std::string& config);

    // CLEANUP OPS: Aggressive removal of dead code and redundant telemetry.
    void silicon_wipe(const std::vector<std::string>& targets);

    // METRICS
    size_t get_last_op_latency_ms() const { return last_ms_; }

private:
    std::string root_;
    size_t last_ms_;

    void walk_and_replace(const std::filesystem::path& path, const std::string& pat, const std::string& rep);
};

} // namespace nca::automation
