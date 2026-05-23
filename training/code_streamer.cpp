#include "training/code_streamer.hpp"
#include <iostream>

namespace nca::training {

CodeStreamer::CodeStreamer(const std::string& root_path) 
    : root_(root_path), current_file_idx_(0) {
    discover_files(std::filesystem::path(root_));
    std::cout << "[CodeStreamer] Discovered " << source_files_.size() << " source files in " << root_ << "\n";
}

void CodeStreamer::discover_files(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return;

    std::vector<std::string> agent_files;
    std::vector<std::string> other_files;

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path, std::filesystem::directory_options::skip_permission_denied)) {
            if (entry.is_regular_file()) {
                std::string path_str = entry.path().string();
                std::string ext = entry.path().extension().string();

                if (ext == ".ts" || ext == ".js" || ext == ".cpp" || 
                    ext == ".hpp" || ext == ".h" || ext == ".json" || ext == ".md" || ext == ".txt") {
                    
                    if (path_str.find(".agents") != std::string::npos || 
                        path_str.find(".github") != std::string::npos) {
                        agent_files.push_back(path_str);
                    } else {
                        other_files.push_back(path_str);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[CodeStreamer] Crawl error: " << e.what() << "\n";
    }

    source_files_.clear();
    source_files_.insert(source_files_.end(), agent_files.begin(), agent_files.end());
    source_files_.insert(source_files_.end(), other_files.begin(), other_files.end());
}

bool CodeStreamer::get_next_chunk(std::vector<uint8_t>& chunk_out, size_t chunk_size) {
    chunk_out.clear();
    chunk_out.reserve(chunk_size);

    while (chunk_out.size() < chunk_size) {
        // Open next file if current is exhausted
        if (!current_stream_.is_open() || current_stream_.eof()) {
            if (current_file_idx_ >= source_files_.size()) {
                return !chunk_out.empty(); // No more files
            }
            
            if (current_stream_.is_open()) current_stream_.close();
            current_stream_.open(source_files_[current_file_idx_++], std::ios::binary);
            
            if (!current_stream_.is_open()) continue;
        }

        // Read bytes into the chunk
        char buffer[1024];
        size_t to_read = std::min(sizeof(buffer), chunk_size - chunk_out.size());
        current_stream_.read(buffer, to_read);
        
        std::streamsize read = current_stream_.gcount();
        if (read > 0) {
            for (std::streamsize i = 0; i < read; ++i) {
                chunk_out.push_back(static_cast<uint8_t>(buffer[i]));
            }
        }
    }

    return !chunk_out.empty();
}

void CodeStreamer::reset() {
    current_file_idx_ = 0;
    if (current_stream_.is_open()) current_stream_.close();
}

} // namespace nca::training
