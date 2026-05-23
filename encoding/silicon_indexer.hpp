#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <mutex>
#include <map>

namespace nca::encoding {

/**
 * SiliconIndexer
 * High-speed C++ indexer for massive workspaces.
 * Uses bit-level hashing and parallel AVX-512 sweeps to solve 'Unbound Indexing'.
 */
class SiliconIndexer {
public:
    explicit SiliconIndexer(const std::string& root_path);

    // Performs a full workspace scan and generates a bit-level map
    void index_all();

    // High-speed search for specific alphabet combinations
    std::vector<std::string> search(const std::string& pattern);

    // Metrics
    size_t total_files() const { return file_count_; }
    size_t total_bytes() const { return byte_count_; }

private:
    std::string root_;
    size_t file_count_;
    size_t byte_count_;
    
    // Bit-level index map: Hash -> List of file indices
    std::map<uint64_t, std::vector<size_t>> index_map_;
    std::vector<std::string> file_registry_;
    std::mutex index_mutex_;

    void index_file(const std::filesystem::path& path);
};

} // namespace nca::encoding
