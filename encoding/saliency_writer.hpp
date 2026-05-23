#pragma once
#include "encoding/tokenizer.hpp"
#include <string>
#include <vector>

namespace nca::encoding {

/**
 * SaliencyWriter
 * The 'Silicon Mouthpiece' for code generation.
 * Translates model thought wavefronts into a sequence of alphabet-level patches.
 */
class SaliencyWriter {
public:
    explicit SaliencyWriter(SaliencyTokenizer* tokenizer);

    /**
     * Translates a sequence of latent vectors into an English/Code patch.
     */
    void write_patch(const float* latent_seq, size_t seq_len, std::string& patch_out);

    /**
     * Applies a saturated ASCII patch directly to the specified file.
     * Part of the 'Freedom Refactor' surgical patching logic.
     */
    bool write_saturated(const std::string& path, const std::string& content);

private:
    SaliencyTokenizer* tokenizer_;
};

} // namespace nca::encoding
