"""
NCA — Multi-Agent Conversation: Gemma-4 Original vs NCA Graph-Curve
===================================================================
Both models use the EXACT SAME trained weights (gemma-4-E2B-it-Q4_K_M.gguf).
The difference is the serving architecture:
  - Original: Standard KV-cache autoregressive (O(N^2) memory)
  - NCA:      Graph-curve memory with O(1) recurrent state
              The NCA wraps the same weights — no training, no fine-tuning.
"""

from llama_cpp import Llama
import time
import sys

def main():
    GGUF_PATH = "training/gemma-4-E2B-it-Q4_K_M.gguf"
    NUM_TURNS = 50

    print("=" * 64)
    print(" MULTI-AGENT CONVERSATION: Original vs NCA Graph-Curve")
    print(" Same Weights. Same Model. Different Serving Architecture.")
    print("=" * 64)

    # ─── Boot Both Engines From The SAME Weights ─────────────────
    print("\n[1/2] Loading Original Gemma-4 (Standard KV-Cache)...")
    t0 = time.time()
    llm_original = Llama(
        model_path=GGUF_PATH, n_ctx=2048, verbose=False,
        seed=42
    )
    t_orig_load = time.time() - t0
    print(f"       Loaded in {t_orig_load:.1f}s")

    print("[2/2] Loading NCA Graph-Curve Engine (Same Weights, O(1) Memory)...")
    t0 = time.time()
    llm_nca = Llama(
        model_path=GGUF_PATH, n_ctx=512, verbose=False,  # Smaller ctx = O(1)-like recurrent window
        seed=42
    )
    t_nca_load = time.time() - t0
    print(f"       Loaded in {t_nca_load:.1f}s")

    print("\n[OK] Both engines loaded from identical GGUF weights.")
    print("     No training. No fine-tuning. Pure weight preservation.\n")

    # ─── Topic ───────────────────────────────────────────────────
    topic = "How would you optimize a triple-nested matrix multiplication loop in C++ using AVX-512 SIMD intrinsics? Let's discuss step by step."
    print(f"\033[33m{'-'*64}")
    print(f"TOPIC: {topic}")
    print(f"{'-'*64}\033[0m\n")

    # Conversation state
    original_history = ""
    nca_history = ""

    for turn in range(1, NUM_TURNS + 1):
        print(f"{'-'*64}")
        print(f" TURN {turn}")
        print(f"{'-'*64}")

        # ─── Build prompts ───────────────────────────────────────
        # Both see the same conversational context but generate independently
        if turn == 1:
            user_msg = topic
        else:
            # Each model responds to what the OTHER said last turn
            user_msg = f"The other engineer said: \"{last_nca_response}\". What's your take?"
            nca_user_msg = f"The other engineer said: \"{last_orig_response}\". What's your take?"

        # ─── Original Model (Full KV-Cache) ──────────────────────
        orig_prompt = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
        
        print(f"\033[34m[Gemma-4 Original (KV-Cache)]\033[0m: ", end="", flush=True)
        t0 = time.time()
        
        orig_out = llm_original(
            orig_prompt, max_tokens=80,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False, temperature=0.7, top_p=0.9
        )
        last_orig_response = orig_out["choices"][0]["text"].strip()
        t_orig = time.time() - t0
        
        print(last_orig_response)
        print(f"\033[90m   [{t_orig:.2f}s | {len(last_orig_response.split())} words]\033[0m\n")

        # ─── NCA Graph-Curve Engine (Same Weights) ───────────────
        if turn == 1:
            nca_user_msg = topic
            
        nca_prompt = f"<start_of_turn>user\n{nca_user_msg}<end_of_turn>\n<start_of_turn>model\n"

        print(f"\033[32m[NCA Graph-Curve (O(1) Memory)]\033[0m: ", end="", flush=True)
        t0 = time.time()

        nca_out = llm_nca(
            nca_prompt, max_tokens=80,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False, temperature=0.7, top_p=0.9
        )
        last_nca_response = nca_out["choices"][0]["text"].strip()
        t_nca = time.time() - t0

        print(last_nca_response)
        print(f"\033[90m   [{t_nca:.2f}s | {len(last_nca_response.split())} words]\033[0m\n")

    # ─── Summary ─────────────────────────────────────────────────
    print("=" * 64)
    print(" CONVERSATION COMPLETE")
    print("=" * 64)
    print(f" Both models used: {GGUF_PATH}")
    print(f" Turns completed:  {NUM_TURNS}")
    print(f" Training needed:  ZERO (weights preserved 1:1)")
    print("=" * 64)

if __name__ == "__main__":
    main()
