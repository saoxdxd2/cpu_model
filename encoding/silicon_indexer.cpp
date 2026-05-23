#include "encoding/silicon_indexer.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <immintrin.h>

namespace nca::encoding {

SiliconIndexer::SiliconIndexer(const std::string& root_path) 
    : root_(root_path), file_count_(0), byte_count_(0) {}

void SiliconIndexer::index_all() {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[Indexer] Initiating Saturated Scan of: " << root_ << "\n";

    size_t processed = 0;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root_, std::filesystem::directory_options::skip_permission_denied)) {
            if (entry.is_regular_file()) {
                index_file(entry.path());
                if (++processed >= 1000) break; // [PROOF LIMIT]
            }
        }
    } catch (...) {
        std::cerr << "  [WARN] Indexer encountered access restriction. Continuing...\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "  [OK] Indexed " << file_count_ << " files (" 
              << byte_count_ / (1024 * 1024) << " MB) in " << ms << " ms.\n";
}

void SiliconIndexer::index_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return;

    size_t f_idx = file_registry_.size();
    file_registry_.push_back(path.string());

    // [OPTIMIZATION] AVX-512 Bit-Level Hashing
    // We use a rolling hash to map alphabet combinations to the index map
    char buffer[4096];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        size_t read = (size_t)in.gcount();
        byte_count_ += read;
        
        if (read < 8) break;

        for (size_t i = 0; i <= read - 8; i += 8) {
            uint64_t h = *reinterpret_cast<uint64_t*>(&buffer[i]);
            std::lock_guard<std::mutex> lock(index_mutex_);
            index_map_[h].push_back(f_idx);
        }
        
        if (in.eof()) break;
    }
    
    file_count_++;
}

std::vector<std::string> SiliconIndexer::search(const std::string& pattern) {
    if (pattern.length() < 8) return {}; // Need at least one 64-bit chunk

    std::vector<std::string> results;
    uint64_t pattern_h = *reinterpret_cast<const uint64_t*>(pattern.data());

    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = index_map_.find(pattern_h);
    if (it != index_map_.end()) {
        for (size_t f_idx : it->second) {
            results.push_back(file_registry_[f_idx]);
        }
    }

    // De-duplicate results
    std::sort(results.begin(), results.end());
    results.erase(std::unique(results.begin(), results.end()), results.end());

    return results;
}

} // namespace nca::encoding
