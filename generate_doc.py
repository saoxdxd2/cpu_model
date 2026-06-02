import os
import time
import concurrent.futures
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

root_dir = r"c:\Users\sao\Documents\cpu_model"
output_file = os.path.join(root_dir, "research_paper", "PROJECT_DOCUMENTATION_AND_FUNCTIONS.md")
exclude_dirs = {'.git', 'build', 'training', 'deployment', 'research_paper', '.gemini', '.sixth'}

def process_file(filepath, chain):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return f"Could not read file: {e}\n"
        
    rel_path = os.path.relpath(filepath, root_dir)
    print(f"Processing {rel_path}...")
    
    # Retry mechanism wrapping LangChain execution to handle severe 429s
    for attempt in range(5):
        try:
            doc = chain.invoke({"rel_path": rel_path, "content": content})
            return f"## File: `{rel_path}`\n\n{doc}\n\n---\n\n"
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limited on {rel_path}. Retrying in {2 ** attempt * 2}s...")
                time.sleep(2 ** attempt * 2)
            else:
                print(f"Error on {rel_path}: {e}. Retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)
                
    return f"## File: `{rel_path}`\n\nError: Maximum retries exceeded due to rate limiting.\n\n---\n\n"

def main():
    print("Starting LangChain-powered documentation generation...")
    files_to_process = []
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(('.cpp', '.hpp', '.h', '.c', '.cmake', 'CMakeLists.txt', '.py')):
                files_to_process.append(os.path.join(root, file))

    # Initialize LangChain LLM
    # We use gemini-3.1 for advanced architectural insight, matching the autonomous agent.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.1,
        max_retries=4,  # LangChain native retries
        max_output_tokens=2048
    )
    
    prompt = PromptTemplate(
        template=(
            "You are an expert C++ AVX-512 systems architect working on the CENTAUR Neural Engine. "
            "Analyze this file:\nPath: {rel_path}\n"
            "Content:\n```\n{content}\n```\n\n"
            "Provide a concise, highly technical architectural breakdown. Explain the core purpose, critical functions, "
            "and how it contributes to the zero-cost, heapless, pure C++ AVX-512 physics architecture. "
            "IMPORTANT: DO NOT use repetitive boilerplate text. DO NOT copy-paste the same description for multiple functions. "
            "Be direct, insightful, and avoid filler words. Use Markdown."
        ),
        input_variables=["rel_path", "content"]
    )
    
    # Construct the LangChain Pipeline
    chain = prompt | llm | StrOutputParser()

    results = {}
    # Use ThreadPoolExecutor to achieve concurrency while respecting API limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_file = {executor.submit(process_file, filepath, chain): filepath for filepath in files_to_process}
        for future in concurrent.futures.as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                results[filepath] = future.result()
            except Exception as exc:
                results[filepath] = f"## File: `{os.path.relpath(filepath, root_dir)}`\n\nError: {exc}\n\n---\n\n"

    print("Writing results to documentation file...")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("# CENTAUR Neural Engine: Exhaustive Codebase Documentation (LangChain AI Generated)\n\n")
        out_f.write("This document was generated concurrently using LangChain and Gemini 1.5 Pro for precise, deduplicated architectural insights.\n\n---\n\n")
        
        for filepath in sorted(files_to_process):
            out_f.write(results[filepath])

    print(f"Successfully wrote optimized documentation to {output_file}")

if __name__ == "__main__":
    main()
