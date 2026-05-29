// ============================================================================
// NCA -- Halting Gate (Phase 7)
// core/layers/halting.cpp
// ============================================================================

#include "core/layers/halting.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>
#include <algorithm>

namespace nca::layers {

//    pt = ex * pc + (1.f - ex) * pt;
//    state.p_sum += pt; state.remainder = ex * pt; should_halt = (ex > .5f); state.steps++;
//    return pt;
//}

} // namespace nca::layers
