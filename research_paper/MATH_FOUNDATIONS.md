# Volume I: Mathematical Foundations of the CENTAUR Engine

The CENTAUR Architecture replaces standard discrete attention with a continuous **Recursive Wavefront** $(\Psi)$. This volume details the calculus of state transition, spectral memory, and deductive grounding.

## 1. The Recursive Wavefront Calculus

Traditional Transformers compute attention as a static product $QK^T$. In contrast, CENTAUR defines the mental state at time $t$ as a recursive differential:

$$\Psi_t = \sigma( \alpha \Psi_{t-1} + \beta \mathcal{E}(\mathbf{x}_t) + \mathcal{L}(\Psi_{t-1}, \mathbf{x}_t) )$$

Where:
*   $\alpha$: Persistence factor (decay coefficient).
*   $\beta$: Injection gain for the incoming alphabet primitive $\mathbf{x}_t$.
*   $\mathcal{L}(\cdot)$: The L-Cycle Thought Operator (Recursive Logic).

The objective of the engine is to find the steady-state $\Psi^*$ where the predictive surprise (entropy) is minimized.

## 2. Kronecker-Factored RLS (Spectral Memory)

Long-term memory in CENTAUR is not a fixed lookup table but a **Recursive Least Squares (RLS)** filter operating in the spectral domain. To achieve $O(1)$ complexity for $D=2048$, we employ a Kronecker-factored covariance matrix:

$$P \approx A \otimes B$$

Where $A \in \mathbb{R}^{64 \times 64}$ and $B \in \mathbb{R}^{32 \times 32}$. The update rule for the spectral factors follows the Woodbury Identity:

$$A_{t+1} = \lambda^{-1} A_t - \frac{\lambda^{-2} A_t \mathbf{u} \mathbf{u}^T A_t}{1 + \lambda^{-1} \mathbf{u}^T A_t \mathbf{u}}$$

By factorizing $P$, we reduce the memory overhead from $D^2$ to $D_{factor}^2$, a **1000x reduction in memory cost** for long-term intelligence.

## 3. Gaussian Moment Updates (Grounding)

To ground the intelligence into the agentic environment, we use a **Gaussian Moment Update** for the micro-expert pool. This connects the Reinforcement Learning (RL) advantage $A_t$ directly to the neural weights $W$:

$$\Delta W_{expert} = \eta \cdot A_t \cdot \nabla_W \log \pi(a_t | \Psi_t)$$

In CENTAUR, this is implemented as an $O(1)$ online update:
1.  **Saliency Check**: The Bipolar Phase-Collapse Router (BPCR) identifies experts in $O(B)$ time.
2.  **Moment Accumulation**: The expert's mean and variance are shifted by the advantage signal.
3.  **Saturation**: Weights are strictly managed within the cache-aligned memory buffers.

## 4. Alphabet-Level Deductive Reasoning

CENTAUR avoids the "Tokenization Trap." The probability of a word is deduced from the joint probability of its alphabet primitives:

$$P(\text{word}) = \prod_{i=1}^n P(\text{char}_i | \Psi_{i-1})$$

The engine is trained using **Next-Primitive Prediction (NPP)**, which forces the model to learn the structural combinations of alphabets, leading to superior "Silicon Grammar" understanding.

## 5. Knowledge Localization & Expert Attribution

To identify "where the memory resides," we employ an **Attribution Heuristic** ($S$) that measures the energy resonance between a target alphabet pattern ($\mathbf{x}_{target}$) and the Expert Pool ($W_{pool}$):

$$S_i = \sum_{b=1}^{B} \left| \text{Gate}_i(\mathbf{x}_{target}) \right| \cdot \text{Energy}(W_{up, i})$$

Where:
*   $S_i$: Saliency score for Expert $i$.
*   $\text{Gate}_i(\cdot)$: The sparse routing probability for the target pattern.

This allows CENTAUR to generate a **Silicon Memory Map**, identifying specialized logic sectors within the shared weight foundation.

---
*Next Volume: [Hardware Saturation](HARDWARE_SATURATION.md)*
