import os
import json
import urllib.request

API_KEY = "AIzaSyCkZHQUuoCkMkhApmpWUbACAA78YF1PuwE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"

root_dir = r"c:\Users\sao\Documents\cpu_model"
output_file = os.path.join(root_dir, "research_paper", "PROJECT_DOCUMENTATION_AND_FUNCTIONS.md")
exclude_dirs = {'.git', 'build', 'training', 'deployment', 'research_paper', '.gemini'}

def ask_gemini(prompt):
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3
        }
    }
    req = urllib.request.Request(
        URL, 
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Error: Unexpected response format: {result}"
    except urllib.error.HTTPError as e:
        # Fallback to older model names if 3.1 is not registered in this specific v1beta endpoint version
        if e.code == 404:
            print("Model gemini-3.1-flash-lite not found, trying gemini-2.0-flash-lite...")
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={API_KEY}"
            req.full_url = fallback_url
            try:
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e2:
                return f"Error calling Gemini API (Fallback): {e2}"
        return f"HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return f"Could not read file: {e}\n"
        
    rel_path = os.path.relpath(filepath, root_dir)
    print(f"Processing {rel_path}...")
    
    prompt = (
        f"You are an expert C++ AVX-512 systems architect working on the CENTAUR Neural Engine. "
        f"Your task is to provide an EXTREMELY detailed, massive, function-by-function breakdown "
        f"of the following file. Explain every single function, its purpose, its parameters, return type, "
        f"and how it relates to the zero-cost, heapless, pure C++ AVX-512 physics architecture of CENTAUR.\n\n"
        f"File Path: {rel_path}\n"
        f"File Content:\n```\n{content}\n```\n\n"
        f"Generate at least 400 words for this file, explaining every minute detail. Use Markdown."
    )
    
    doc = ask_gemini(prompt)
    
    out = f"## File: `{rel_path}`\n\n"
    out += doc
    out += "\n\n---\n\n"
    return out

def main():
    print("Starting generation using AI...")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("# CENTAUR Neural Engine: Exhaustive Codebase Documentation (AI Generated)\n\n")
        out_f.write("This document was generated using the Gemini AI API for extreme accuracy and depth.\n\n")
        out_f.write("---\n\n")
        
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(('.cpp', '.hpp', '.h', '.c', '.cmake', 'CMakeLists.txt', '.py')):
                    filepath = os.path.join(root, file)
                    out_f.write(process_file(filepath))
                    out_f.flush()

    print(f"Successfully wrote extensive documentation to {output_file}")

if __name__ == "__main__":
    main()
