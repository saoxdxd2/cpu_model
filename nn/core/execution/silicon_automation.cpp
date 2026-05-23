#include "nn/core/execution/silicon_automation.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

namespace nca::automation {

SiliconAutomation::SiliconAutomation(const std::string& workspace_root) 
    : root_(workspace_root), last_ms_(0) {}

void SiliconAutomation::batch_refactor(const std::string& pattern, const std::string& replacement) {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[Automation] Batch Refactoring: '" << pattern << "' -> '" << replacement << "'\n";
    
    walk_and_replace(std::filesystem::path(root_), pattern, replacement);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "  [OK] Refactor complete in " << last_ms_ << " ms.\n";
}

void SiliconAutomation::walk_and_replace(const std::filesystem::path& path, const std::string& pat, const std::string& rep) {
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path, std::filesystem::directory_options::skip_permission_denied)) {
            try {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".ts" || ext == ".js" || ext == ".cpp" || ext == ".hpp" || ext == ".h" || ext == ".json") {
                        // In-place replacement
                        std::ifstream in(entry.path());
                        if (!in.is_open()) continue;

                        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                        in.close();

                        size_t pos = 0;
                        bool changed = false;
                        while ((pos = content.find(pat, pos)) != std::string::npos) {
                            content.replace(pos, pat.length(), rep);
                            pos += rep.length();
                            changed = true;
                        }

                        if (changed) {
                            std::ofstream out(entry.path());
                            if (out.is_open()) {
                                out << content;
                                out.close();
                            }
                        }
                    }
                }
            } catch (...) { continue; }
        }
    } catch (...) {
        std::cerr << "  [WARN] Automation walk encountered restricted directory. Skipping...\n";
    }
}

void SiliconAutomation::trigger_fast_build(const std::string& config) {
    std::cout << "[Automation] Triggering Saturated Build: " << config << "\n";
    // System call for cmake --build
}

void SiliconAutomation::silicon_wipe(const std::vector<std::string>& targets) {
    std::cout << "[Automation] Initiating Silicon Wipe on " << targets.size() << " target patterns...\n";
    
    // Hardcoded Bloat Targets for the Freedom Refactor
    std::vector<std::string> bloat = {
        "workbench.welcomePage.enabled",
        "extensions.autoCheckUpdates",
        "workbench.settings.enableSync",
        "telemetry.enableTelemetry",
        "api.enableTelemetry"
    };

    for(const auto& t : targets) {
        batch_refactor(t, "// NCA_WIPED_FOR_FREEDOM");
    }
    
    for(const auto& b : bloat) {
        batch_refactor(b, "false /* NCA_DISABLED */");
    }
}

} // namespace nca::automation
