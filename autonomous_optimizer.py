import os
import time
import subprocess
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

root_dir = r"c:\Users\sao\Documents\cpu_model"
grapify_file = os.path.join(root_dir, "research_paper", "PROJECT_DOCUMENTATION_AND_FUNCTIONS.md")
build_dir = os.path.join(root_dir, "build")

# 1. Pydantic Schemas
class OptimizationTarget(BaseModel):
    target_file: str = Field(description="Relative path to the file to optimize.")
    reasoning: str = Field(description="Why this file was selected based on the Grapify docs.")

class CodeReplacement(BaseModel):
    rationale: str = Field(description="Explanation of the specific AVX-512/Zero-Cost improvements or fixes made.")
    new_code: str = Field(description="The COMPLETE, fully optimized new C++ code.")

def load_grapify():
    if not os.path.exists(grapify_file):
        return "Error: Documentation manifest not found."
    with open(grapify_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def propose_target(llm, grapify_context: str) -> OptimizationTarget:
    """Phase 1: Analyzes the architecture map to propose a file to optimize."""
    parser = PydanticOutputParser(pydantic_object=OptimizationTarget)
    prompt = PromptTemplate(
        template=(
            "You are the CENTAUR Neural Engine Optimizer. Analyze the 'Grapify' architecture map below.\n"
            "Identify exactly ONE C++ file (.cpp or .hpp) that would benefit most from AVX-512 optimization, "
            "heapless refactoring, or duplicate code pruning.\n\n"
            "GRAPIFY MAP:\n```markdown\n{grapify_context}\n```\n\n"
            "{format_instructions}"
        ),
        input_variables=["grapify_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    return chain.invoke({"grapify_context": grapify_context})

def safe_optimize(llm, original_code: str) -> CodeReplacement:
    """Phase 2: Passes the ACTUAL source code to the LLM for modification."""
    parser = PydanticOutputParser(pydantic_object=CodeReplacement)
    prompt = PromptTemplate(
        template=(
            "You are the CENTAUR Neural Engine Optimizer. Optimize this specific file for AVX-512 "
            "and zero-cost abstractions. DO NOT hallucinate. Output the COMPLETE revised file.\n\n"
            "ORIGINAL CODE:\n```cpp\n{original_code}\n```\n\n"
            "{format_instructions}"
        ),
        input_variables=["original_code"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    return chain.invoke({"original_code": original_code})

def attempt_fix(llm, original_code: str, broken_code: str, error_msg: str) -> CodeReplacement:
    """Phase 3 (Self-Correction): Analyzes compiler/test errors and attempts to fix the broken optimization."""
    parser = PydanticOutputParser(pydantic_object=CodeReplacement)
    prompt = PromptTemplate(
        template=(
            "You are the CENTAUR Neural Engine Optimizer. Your previous AVX-512 optimization broke the build or failed the tests.\n\n"
            "ERROR LOG:\n```\n{error_msg}\n```\n\n"
            "ORIGINAL, WORKING CODE:\n```cpp\n{original_code}\n```\n\n"
            "BROKEN OPTIMIZED CODE:\n```cpp\n{broken_code}\n```\n\n"
            "Analyze the error log. The failure is likely due to strict SIMD intrinsic typing (__m512 vs __m512i), "
            "memory alignment faults, or a broken unit test mathematical assertion.\n"
            "Output the COMPLETE, fixed C++ code that retains your optimizations but resolves the error.\n\n"
            "{format_instructions}"
        ),
        input_variables=["error_msg", "original_code", "broken_code"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    return chain.invoke({"error_msg": error_msg, "original_code": original_code, "broken_code": broken_code})

def apply_and_test(filepath: str, new_code: str) -> tuple[bool, str]:
    """Writes code, compiles, and runs CTest. Returns (Success, ErrorMsg)."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_code)
    
    print(f"\n[VERIFICATION] Compiling {os.path.basename(filepath)}...")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        try:
            subprocess.run(["cmake", ".."], cwd=build_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return False, f"CMake Config Error:\n{e.stderr}"

    # 1. Compile Phase
    try:
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, f"Compiler Error:\n{e.stderr}"

    # 2. Testing Phase
    print(f"[TESTING] Execution phase: Running CTest suite...")
    try:
        # Run tests with detailed output on failure
        subprocess.run(["ctest", "--output-on-failure", "-C", "Release"], cwd=build_dir, check=True, capture_output=True, text=True)
        print("[VERIFICATION] SUCCESS: Optimization is stable and mathematically verified by tests.")
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Test Failure:\n{e.stdout}\n{e.stderr}"

def main():
    print("Initializing CENTAUR Self-Healing Compiler Agent with Test Integration...")
    
    # Use gemini-3.1-pro-preview for advanced reasoning in fixing errors
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.5,
        max_output_tokens=8192
    )

    iteration = 1
    MAX_FIX_RETRIES = 3

    while True:
        print(f"\n================ Iteration {iteration} ================")
        print("Loading architecture graph...")
        grapify_context = load_grapify()
        
        if "Error" in grapify_context:
            print("Grapify context missing. Retrying in 10s...")
            time.sleep(10)
            continue
            
        print("Phase 1: Identifying target...")
        try:
            target_proposal = propose_target(llm, grapify_context)
            target_filepath = os.path.join(root_dir, target_proposal.target_file)
            
            if not os.path.exists(target_filepath):
                print(f"[SKIP] Target file {target_proposal.target_file} does not exist. Retrying next loop.")
                time.sleep(5)
                continue
                
            print(f"[TARGET IDENTIFIED] {target_proposal.target_file}")
            print(f"[REASONING] {target_proposal.reasoning}")
            
            # Cache the pristine working code
            with open(target_filepath, "r", encoding="utf-8") as f:
                original_code = f.read()
                
            print("Phase 2: Generating optimization...")
            replacement = safe_optimize(llm, original_code)
            
            if original_code.strip() == replacement.new_code.strip():
                print("[SKIP] Proposed code is identical to current codebase.")
                time.sleep(15)
                iteration += 1
                continue
                
            current_code = replacement.new_code
            is_fixed = False
            
            # The Self-Healing Test/Fix Loop
            for attempt in range(MAX_FIX_RETRIES):
                success, error_msg = apply_and_test(target_filepath, current_code)
                
                if success:
                    print(f"[RATIONALE] {replacement.rationale}")
                    is_fixed = True
                    break
                
                print(f"[FAILURE CAUGHT] Attempt {attempt+1}/{MAX_FIX_RETRIES} failed.")
                print(f"Error preview: {error_msg[:300]}...")
                
                if attempt < MAX_FIX_RETRIES - 1:
                    print("\n[SELF-HEALING] Phase 3: Analyzing error log and generating fix...")
                    fix = attempt_fix(llm, original_code, current_code, error_msg)
                    current_code = fix.new_code
                    replacement.rationale = fix.rationale # Update rationale to the fix logic
                    print("[SELF-HEALING] Re-applying patched code...")
            
            # Rollback if all self-healing attempts failed
            if not is_fixed:
                print(f"\n[ROLLBACK] Exhausted all {MAX_FIX_RETRIES} self-healing attempts.")
                print("[ROLLBACK] Reverting codebase to original mathematical state.")
                with open(target_filepath, "w", encoding="utf-8") as f:
                    f.write(original_code)
                
        except Exception as e:
            print(f"[ERROR] Agent encountered a critical failure in the loop: {e}")
            
        print("\nSleeping for 15 seconds before the next iteration...")
        time.sleep(15)
        iteration += 1

if __name__ == "__main__":
    main()
