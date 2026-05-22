#pragma once
#include <vector>
#include <cstdint>

namespace nca::encoding {

/**
 * SaliencyTokenizer
 * Discretizes continuous latent vectors into high-saliency token embeddings.
 * Maps tactical situations to discrete 'Thought Anchors'.
 */
class SaliencyTokenizer {
public:
    // Unified Tokenizer: Adapts behavior based on vocab_size (e.g., 256 for ASCII, 4096 for Concepts)
    SaliencyTokenizer(size_t vocab_size = 4096, size_t d_model = 2048);

    // --- Semantic Level (Chatbot/Concepts) ---
    void tokenize(const float* latent, std::vector<uint32_t>& tokens_out);
    const float* get_embedding(uint32_t token_id) const;

    // --- Primitive Level (Raw Alphabet/Bitboards) ---
    void tokenize_primitive(const float* latent, uint32_t& char_out);
    const float* get_char_embedding(uint8_t c) const;

private:
    size_t vocab_size_;
    size_t d_model_;
    std::vector<float> codebook_; // [vocab_size * d_model]
};

} // namespace nca::encoding
