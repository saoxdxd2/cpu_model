import os
import sys
from huggingface_hub import hf_hub_download
from gguf_translator import extract_gguf_to_nca

def automated_nca_adoption(repo_id="unsloth/gemma-4-E2B-it-GGUF", filename="gemma-4-E2B-it-Q4_K_M.gguf"):
    """
    End-to-End Automated Pipeline:
    HuggingFace GGUF -> Local Download -> Dequantization -> NCA FP32 Material
    """
    print(f"===========================================================")
    print(f" NCA Automated Foundation Adoption Pipeline              ")
    print(f" Target: {repo_id} / {filename}                          ")
    print(f"===========================================================\n")

    # 1. Automated Download
    print(f"[1/3] Fetching GGUF from HuggingFace Hub...")
    try:
        # local_dir_use_symlinks removed to fix the hf_hub_download deprecation warning
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="."
        )
        print(f"  [OK] Model secured at: {local_path}")
    except Exception as e:
        print(f"  [Error] Download failed: {e}")
        print("  Check if the repo/file exists and you have 'huggingface_hub' installed.")
        return

    # 2. Automated Dequantization (Imported from gguf_translator.py)
    print(f"\n[2/3] Initiating Deep Level Translation (ViteLLM Principle)...")
    extract_gguf_to_nca(local_path)

    # 3. Final Verification
    output_material = "foundation_dequantized.pt"
    if os.path.exists(output_material):
        print(f"\n[3/3] FULL AUTOMATION SUCCESSFUL.")
        print(f"  Result: {output_material} is ready for C++ injection.")
        print(f"  Next: Re-run 'offline_trainer' to tune the AVX-512 engine.")
    else:
        print(f"\n[Error] Pipeline completed but material was not generated.")

if __name__ == "__main__":
    # Parameters can be overridden via CLI, now defaulting to the correct Unsloth repo
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "unsloth/gemma-4-E2B-it-GGUF"
    target_file = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-E2B-it-Q4_K_M.gguf"
    
    automated_nca_adoption(target_repo, target_file)