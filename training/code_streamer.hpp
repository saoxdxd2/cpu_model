#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

namespace nca::training {

/**
 * CodeStreamer
 * Recursively crawls source directories and provides a stream of 
 * alphabet-level primitives for training.
 */
class CodeStreamer {
public:
    explicit CodeStreamer(const std::string& root_path);

    // Returns the next block of code as a vector of alphabet primitives
    // Returns false if end of codebase reached.
    bool get_next_chunk(std::vector<uint8_t>& chunk_out, size_t chunk_size = 2048);

    // Metadata
    size_t total_files_found() const { return source_files_.size(); }
    void   reset();

private:
    std::string root_;
    std::vector<std::string> source_files_;
    size_t current_file_idx_;
    std::ifstream current_stream_;

    void discover_files(const std::filesystem::path& path);
};

} // namespace nca::training
