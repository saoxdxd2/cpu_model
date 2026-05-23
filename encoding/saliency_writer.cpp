#include "encoding/saliency_writer.hpp"
#include <iostream>
#include <fstream>

namespace nca::encoding {

SaliencyWriter::SaliencyWriter(SaliencyTokenizer* tokenizer) : tokenizer_(tokenizer) {}

void SaliencyWriter::write_patch(const float* latent_seq, size_t seq_len, std::string& patch_out) {
    patch_out = "";
    const size_t D_MODEL = 2048;

    for (size_t i = 0; i < seq_len; ++i) {
        const float* latent = latent_seq + (i * D_MODEL);
        
        uint32_t char_id;
        tokenizer_->tokenize_primitive(latent, char_id);
        
        // Silicon-to-ASCII Translation
        if (char_id < 256) {
            patch_out += static_cast<char>(char_id);
        } else {
            patch_out += '?'; // Signal lost
        }
    }
}

bool SaliencyWriter::write_saturated(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << content;
    out.close();
    return true;
}

} // namespace nca::encoding
