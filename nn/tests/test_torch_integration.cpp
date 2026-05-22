// ============================================================================
// NCA — PyTorch/LibTorch Integration Test (External Training Bridge)
// tests/test_torch_integration.cpp
// ============================================================================

#include <iostream>
#include <vector>
#include <torch/torch.h>

// This test demonstrates how the external training team can wrap the 
// custom NCA C++ arrays into PyTorch Tensors for traditional backprop
// without modifying the core inference C++ engine.

int main() {
    std::cout << "+---------------------------------------------------+\n";
    std::cout << "|  NCA -- External Training Integration Test        |\n";
    std::cout << "+---------------------------------------------------+\n\n";

    // 1. Simulate the C++ model parameters (raw float arrays)
    // Normally these would be pointers inside MultimodalEngine
    const size_t D_MODEL = 2048;
    alignas(64) float raw_weights[D_MODEL];
    
    // Initialize with some dummy values
    for (size_t i = 0; i < D_MODEL; ++i) {
        raw_weights[i] = (float)i / D_MODEL;
    }

    // 2. The Bridge: Wrap the raw C++ memory in a torch::Tensor
    // This allows PyTorch to perform autograd and optimizers directly on the C++ memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    torch::Tensor weight_tensor = torch::from_blob(raw_weights, {D_MODEL}, options);

    std::cout << "  [CHECK] Memory Bridge Established.\n";
    std::cout << "          Tensor Size: [" << weight_tensor.size(0) << "]\n";
    std::cout << "          Requires Grad: " << (weight_tensor.requires_grad() ? "Yes" : "No") << "\n\n";

    // 3. Simulate a traditional PyTorch training step
    auto optimizer = torch::optim::Adam({weight_tensor}, torch::optim::AdamOptions(0.01));

    // Dummy target and input
    torch::Tensor input = torch::rand({D_MODEL});
    torch::Tensor target = torch::ones({D_MODEL});

    std::cout << "  [RUN ] Executing Traditional Backprop Step...\n";

    optimizer.zero_grad();
    
    // Forward pass
    torch::Tensor output = weight_tensor * input;
    
    // Loss
    torch::Tensor loss = torch::mse_loss(output, target);
    
    // Backward pass
    loss.backward();
    
    // Optimizer step
    optimizer.step();

    std::cout << "  [CHECK] Backward Pass Complete. Loss: " << loss.item<float>() << "\n";
    std::cout << "  [CHECK] Verifying C++ memory update... ";

    // Verify that the original C++ array was actually modified by PyTorch
    bool updated = false;
    for (size_t i = 0; i < D_MODEL; ++i) {
        if (raw_weights[i] != (float)i / D_MODEL) {
            updated = true;
            break;
        }
    }

    if (updated) {
        std::cout << "OK (raw_weights modified by LibTorch Adam Optimizer)\n\n";
        std::cout << "  [PASS] External training pipeline can successfully mutate NCA memory.\n";
        return 0;
    } else {
        std::cout << "FAIL (C++ memory unchanged!)\n";
        return 1;
    }
}
