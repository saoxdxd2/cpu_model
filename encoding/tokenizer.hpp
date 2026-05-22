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
    SaliencyTokenizer(size_t vocab_size = 4096, size_t d_model = 2048);

    /**
     * Discretizes a latent vector into a set of tokens.
     * Returns a sparse bitset of active token indices.
     */
    void tokenize(const float* latent, std::vector<uint32_t>& tokens_out);

    /**
     * Decodes a token index back into its high-precision embedding.
     */
    const float* get_embedding(uint32_t token_id) const;

private:
    size_t vocab_size_;
    size_t d_model_;
    std::vector<float> codebook_; // [vocab_size * d_model]
};

} // namespace nca::encoding
