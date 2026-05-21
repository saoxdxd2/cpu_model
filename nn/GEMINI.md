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

## 3. Current Status: FULLY CONDENSED & HARDENED (v6.3)
- **Vision Core**: Wavefront Saturated (Diagonal Scanning + Gated Pruner).
- **Logic Core**: Sparse Synthesis (Hashed Router + Triple-Hybrid Expert Pool).
- **Verification**: Running at ~270 tokens/s with bit-perfect implementation contract.
