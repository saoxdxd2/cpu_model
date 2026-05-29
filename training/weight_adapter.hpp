#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>

namespace nca::execution { class MultimodalEngine; }

namespace nca::training {

class WeightAdapter {
public:
    static void inject_params(const torch::Tensor& source, 
                              float* target_ptr, 
                              size_t target_size);

    static void adopt_state_dict(const std::map<std::string, torch::Tensor>& dict, 
                                 nca::execution::MultimodalEngine& engine);
};

} // namespace nca::training
