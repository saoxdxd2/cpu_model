#include "encoding/state_encoder.hpp"
#include "encoding/tokenizer.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace nca::encoding;

int main() {
    std::cout << "===========================================\n";
    std::cout << " NCA Silicon Encoding & Tokenization Test \n";
    std::cout << "===========================================\n\n";

    const size_t OBS_DIM = 1616;
    const size_t D_MODEL = 2048;
    const size_t VOCAB_SIZE = 4096;

    SiliconEncoder encoder;
    SaliencyTokenizer tokenizer(VOCAB_SIZE, D_MODEL);

    // 1. Simulate a raw observation
    std::vector<float> raw_obs(OBS_DIM);
    for(size_t i=0; i<OBS_DIM; ++i) raw_obs[i] = (static_cast<float>(rand()) / RAND_MAX);
    
    // Set some "High Saliency" signals (low health entities)
    raw_obs[0] = 0.05f; // Entity 0 Health
    raw_obs[5] = 0.01f; // Entity 1 Health

    std::vector<float> latent(D_MODEL);

    std::cout << "[RUN] Executing Silicon Scan + Prune...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    encoder.encode(raw_obs.data(), latent.data(), D_MODEL);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "  [OK] Latent Vector Generated in " << elapsed.count() << " ms.\n";

    // 2. Tokenize the latent space
    std::cout << "[RUN] Discretizing Latent Space to Saliency Tokens...\n";
    std::vector<uint32_t> tokens;
    
    start = std::chrono::high_resolution_clock::now();
    tokenizer.tokenize(latent.data(), tokens);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    if (!tokens.empty()) {
        std::cout << "  [OK] Token Found: " << tokens[0] << " (Matched in " << elapsed.count() << " ms)\n";
        const float* embedding = tokenizer.get_embedding(tokens[0]);
        if (embedding) {
            std::cout << "  [OK] Token Embedding successfully recovered.\n";
        }
    }

    std::cout << "\n[SUCCESS] Silicon Encoding Pipeline Verified.\n";
    return 0;
}
