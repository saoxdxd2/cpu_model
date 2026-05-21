# NCA — Nano-Core Architecture Engineering Directives

## 1. Core Philosophy: Absolute Saturation
- **Geometric Saturated**: All Matrix-Vector operations MUST use Rank-16 unrolling.
- **Complexity Destruction**: Replace $O(N \log N)$ and Dense $O(N^2)$ with $O(N)$ and Hashed $O(1)$ proxies.
- **L1 Eternal**: Keep all active data structures and working sets under the 32KB L1d cache limit.
- **CPU Advantage**: JUMP instead of WALK. Use branching and pointer-chasing as a weapon.

## 2. Model Integrity: Neural Circuit Synthesis
- **Hashed Logic Routing (HLR)**: Use randomized projections to select active weights. Only 2% of the model is activated per token.
- **Weights as Anchors**: Parameters are stable associative keys. Never re-initialize in the hot loop.
- **Recursive ACT**: Intelligence is a function of time (cycles), not just parameters (depth).

## 3. Current Status: NEURAL CIRCUIT SYNTHESIS COMPLETE (v6.0)
- **Vision Core**: Saturated (Scanner + Gated Pruner).
- **Logic Core**: Sparse Synthesis (Hashed Router + Triple-Hybrid Expert Pool).
- **Bridge Core**: Saturated (Rank-16 Adaptive Anchors).
- **Verification**: Running at ~190 tokens/s with sparse logic and high structural variance.
