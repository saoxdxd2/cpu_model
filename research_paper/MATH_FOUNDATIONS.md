# Volume I: Mathematical Foundations of the Aether Pipeline

The Nano-Core Architecture (NCA) replaces standard discrete attention with a continuous **Recursive Wavefront** $(\Psi)$. This volume details the calculus of state transition, spectral memory, and deductive grounding.

## 1. The Recursive Wavefront Calculus

Traditional Transformers compute attention as a static product $QK^T$. In contrast, the NCA defines the mental state at time $t$ as a recursive differential:

$$\Psi_t = \sigma( \alpha \Psi_{t-1} + \beta \mathcal{E}(\mathbf{x}_t) + \mathcal{L}(\Psi_{t-1}, \mathbf{x}_t) )$$

Where:
*   $\alpha$: Persistence factor (decay coefficient).
*   $\beta$: Injection gain for the incoming alphabet primitive $\mathbf{x}_t$.
*   $\mathcal{E}(\cdot)$: The Silicon Encoder (Spatial to Latent mapping).
*   $\mathcal{L}(\cdot)$: The L-Cycle Thought Operator (Recursive Logic).

The objective of the engine is to find the steady-state $\Psi^*$ where the predictive surprise (entropy) is minimized.

## 2. Kronecker-Factored RLS (Spectral Memory)

Long-term memory in the NCA is not a fixed lookup table but a **Recursive Least Squares (RLS)** filter operating in the spectral domain. To achieve $O(1)$ complexity for $D=2048$, we employ a Kronecker-factored covariance matrix:

$$P \approx A \otimes B$$

Where $A \in \mathbb{R}^{64 \times 64}$ and $B \in \mathbb{R}^{32 \times 32}$. The update rule for the spectral factors follows the Woodbury Identity:

$$A_{t+1} = \lambda^{-1} A_t - \frac{\lambda^{-2} A_t \mathbf{u} \mathbf{u}^T A_t}{1 + \lambda^{-1} \mathbf{u}^T A_t \mathbf{u}}$$

By factorizing $P$, we reduce the memory overhead from $D^2$ ($4,194,304$ parameters) to $D_{factor}^2$ ($5,120$ parameters), a **1000x reduction in memory cost** for long-term intelligence.

## 3. Gaussian Moment Updates (Grounding)

To ground the Gemma-4 intelligence into the agentic environment, we use a **Gaussian Moment Update** for the micro-expert pool. This connects the Reinforcement Learning (RL) advantage $A_t$ directly to the neural weights $W$:

$$\Delta W_{expert} = \eta \cdot A_t \cdot \nabla_W \log \pi(a_t | \Psi_t)$$

In the NCA, this is implemented as an $O(1)$ online update:
1.  **Saliency Check**: The Hashed Router identifies top-16 experts.
2.  **Moment Accumulation**: The expert's mean and variance are shifted by the advantage signal.
3.  **Saturation**: Weights are re-clamped to the MXINT8 boundary to prevent numerical drift.

## 4. Alphabet-Level Deductive Reasoning

The NCA avoids the "Tokenization Trap." The probability of a word is deduced from the joint probability of its alphabet primitives:

$$P(\text{word}) = \prod_{i=1}^n P(\text{char}_i | \Psi_{i-1})$$

The engine is trained using **Next-Primitive Prediction (NPP)**, which forces the model to learn the structural combinations of alphabets. This leads to superior "Silicon Grammar" understanding, as the model sees the raw bitboards of code instead of abstract vocab indices.

---
*Next Volume: [Hardware Saturation](HARDWARE_SATURATION.md)*
