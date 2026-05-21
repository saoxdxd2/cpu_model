# NCA — Nano-Core Architecture Engineering Directives

## 1. Core Philosophy: Absolute Saturation
- **Geometric Saturated**: All Matrix-Vector operations MUST use Rank-16 unrolling.
- **Complexity Destruction**: Replace $O(N \log N)$ and Dense $O(N^2)$ with $O(N)$ and Hashed $O(1)$ proxies.
- **L1 Eternal**: Keep all active data structures and working sets under the 32KB L1d cache limit.
- **CPU Advantage**: JUMP instead of WALK. Use branching and pointer-chasing as a weapon.

## 2. Model Integrity: High-Signal Implementation
- **Condense to Silicon**: Strip redundant code to minimize instruction cache pressure.
- **Wavefront Execution**: Parallelize recurrent states along mathematically independent diagonals.
- **Recursive ACT**: Intelligence is a function of time (cycles), not just parameters (depth).
- **Online Grounding**: Weights are NOT static. The engine MUST adapt to High-Saliency Facts via Gaussian Moment Updates during inference.

## 3. Current Status: GROUNDED BREAKTHROUGH (v27.0)
- **Spectral Logic**: FWHT-domain state refinement using Kronecker-RLS ($O(N \log N)$).
- **SDMS v11.0**: Saliency-Driven Multi-expert System with 1024 Micro-Experts.
- **Adaptive ACT**: Dynamic recursion depth based on token importance/novelty.
- **Verification**: Saturated VNNI kernels achieving bit-perfect "Golden Contract" stability.
