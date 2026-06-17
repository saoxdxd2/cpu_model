import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
import ctypes
import llama_cpp
from pathlib import Path
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "automation_runs"
STATE_FILE = ROOT / "automation_state.json"
GOAL_FILE = ROOT / "automation_goal.md"

DEFAULT_SERVER = "http://127.0.0.1:8080"
ERROR_LOG = ROOT / "automation_errors.jsonl"


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def now_iso():
    return dt.datetime.now(dt.UTC).isoformat()


def bytes_to_mb(value):
    return round(value / (1024 * 1024), 2)


def append_error(stage, exc, extra=None):
    event = {
        "time": now_iso(),
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "extra": extra or {},
    }
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def memory_status():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    if not ok:
        return {"error": "GlobalMemoryStatusEx failed"}
    return {
        "memory_load_percent": int(stat.dwMemoryLoad),
        "total_physical_mb": bytes_to_mb(stat.ullTotalPhys),
        "available_physical_mb": bytes_to_mb(stat.ullAvailPhys),
        "total_pagefile_mb": bytes_to_mb(stat.ullTotalPageFile),
        "available_pagefile_mb": bytes_to_mb(stat.ullAvailPageFile),
    }


def tasklist_processes(image_name):
    result = run_cmd(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        timeout=10,
    )
    rows = []
    for raw in result["stdout"].splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("INFO:"):
            continue
        try:
            parts = [p.strip().strip('"') for p in raw.split('","')]
            if len(parts) >= 5:
                mem_kb = int(parts[4].replace(",", "").replace(" K", ""))
                rows.append(
                    {
                        "image": parts[0],
                        "pid": parts[1],
                        "session": parts[2],
                        "session_number": parts[3],
                        "memory_mb": round(mem_kb / 1024, 2),
                    }
                )
        except ValueError:
            rows.append({"raw": raw})
    return rows


def resource_snapshot():
    disk = shutil.disk_usage(ROOT)
    return {
        "time": now_iso(),
        "memory": memory_status(),
        "disk": {
            "total_mb": bytes_to_mb(disk.total),
            "used_mb": bytes_to_mb(disk.used),
            "free_mb": bytes_to_mb(disk.free),
        },
        "llama_server_processes": tasklist_processes("llama-server.exe"),
    }


def run_cmd(args, timeout=30):
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(args),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }
    except Exception as exc:
        return {
            "command": " ".join(args),
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def read_text(path, max_chars=12000):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    return text[:max_chars]


def load_state():
    if not STATE_FILE.exists():
        return {"iteration": 0, "reports": []}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"iteration": 0, "reports": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


_LLM_INSTANCE = None

def get_llm(ctx_size=16384):
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        draft = None
        try:
            print(f"[worker] Loading Draft Model (E2B) natively via llama_cpp (n_ctx={ctx_size}, Weights map into RAM)...")
            draft_llm = llama_cpp.Llama.from_pretrained(
                repo_id="unsloth/gemma-4-E2B-it-GGUF",
                filename="*Q4_K_M.gguf",
                n_ctx=ctx_size,
                n_threads=4,
                n_batch=4,
                n_gpu_layers=0,
                flash_attn=True,
                type_k=2,
                type_v=2
            )
            draft = llama_cpp.LlamaDraftModel(draft_llm)
            print("[worker] Speculative Decoding enabled via Draft Model.")
        except Exception as e:
            print(f"[worker] Failed to load draft model: {e}. Falling back to standard decoding.")
            
        print(f"[worker] Loading Target Model (E4B) natively via llama_cpp (n_ctx={ctx_size}, Weights map into RAM once)...")
        _LLM_INSTANCE = llama_cpp.Llama.from_pretrained(
            repo_id="unsloth/gemma-4-E4B-it-GGUF",
            filename="*Q4_K_M.gguf",
            n_ctx=ctx_size,
            n_threads=4,
            n_batch=4,
            n_gpu_layers=0,
            draft_model=draft,
            flash_attn=True,
            type_k=2,
            type_v=2
        )
    return _LLM_INSTANCE


async def chat_with_agent(name, agent_data, session, system_prompt, user_prompt, args, temperature=0.2, max_tokens=-1):
    llm = get_llm(args.ctx_size)
    
    print(f"[swarm] {name} Agent is taking the CPU...")
    
    # Fast Context Swapping: Load the ~90MB brain specific to this agent
    if agent_data["state"] is not None:
        try:
            llm.load_state(agent_data["state"])
            print(f"[swarm] Loaded {name}'s KV Cache state into the engine.")
        except Exception as e:
            print(f"[swarm] Failed to load state for {name}: {e}. Continuing without state.")
            
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Create or completely overwrite a file with new code. Use this to actively implement your logic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to repository root"},
                        "content": {"type": "string", "description": "The exact new content of the file"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]
    
    if session:
        try:
            tools_response = await session.list_tools()
            for mcp_tool in tools_response.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description,
                        "parameters": mcp_tool.inputSchema
                    }
                })
        except Exception as e:
            print(f"[worker] Failed to load MCP tools: {e}")

    messages = [{"role": "system", "content": system_prompt}] + agent_data["history"] + [{"role": "user", "content": user_prompt}]
    
    # Context window dynamic shift (~3.5 chars per token)
    MAX_CHARS = int(args.ctx_size * 3.5)
    while len(messages) > 2 and sum(len(m.get("content", "")) for m in messages) > MAX_CHARS:
        print(f"[{name}] Context limit approached ({MAX_CHARS} chars). Truncating oldest history message.")
        messages.pop(1)

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto"
        )
    except ValueError as exc:
        if "context" in str(exc).lower() or "token" in str(exc).lower():
            print(f"[{name}] 400 Context Limit Exceeded. Falling back to zero-history.")
            truncated_prompt = user_prompt[:16000] + "\n\n[TRUNCATED DUE TO CONTEXT LIMIT]"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": truncated_prompt}]
            agent_data["history"] = [] # Clear history on overflow
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto"
            )
        else:
            raise

    message = response["choices"][0]["message"]
    report_lines = [message.get("content", "")]
    
    # Process tools locally
    if "tool_calls" in message and message["tool_calls"]:
        for tc in message["tool_calls"]:
            if tc["type"] == "function":
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {}
                    
                if tool_name == "edit_file":
                    try:
                        file_path = ROOT / args["path"]
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(args.get("content", ""), encoding="utf-8")
                        action_msg = f"-> Tool execution successful: Overwrote {args.get('path', '')}"
                        print(f"[{name}] {action_msg}")
                        report_lines.append(f"\n**Tool Execution:** {action_msg}")
                    except Exception as e:
                        err_msg = f"-> Tool execution failed: {e}"
                        print(f"[{name}] {err_msg}")
                        report_lines.append(f"\n**Tool Error:** {err_msg}")
                else:
                    # MCP Tool Execution
                    if session:
                        try:
                            result = await session.call_tool(tool_name, args)
                            mcp_result = "\n".join([c.text for c in result.content if c.type == "text"])
                            action_msg = f"-> MCP Tool '{tool_name}' successful: {mcp_result[:300]}..."
                            print(f"[{name}] {action_msg}")
                            report_lines.append(f"\n**MCP Tool Execution:** {action_msg}")
                        except Exception as e:
                            err_msg = f"-> MCP Tool '{tool_name}' failed: {e}"
                            print(f"[{name}] {err_msg}")
                            report_lines.append(f"\n**MCP Tool Error:** {err_msg}")
                            
    final_output = "\n".join([line for line in report_lines if line is not None])
    
    # Save the continuous conversational state so the Agent remembers what it did
    agent_data["history"].append({"role": "user", "content": user_prompt})
    agent_data["history"].append({"role": "assistant", "content": final_output})
    
    # Save the physical KV Cache (~90MB) back to Python Memory
    agent_data["state"] = llm.save_state()
    print(f"[swarm] Preserved {name}'s state to memory.")
    
    return final_output


import re

def extract_cpp_ast(root_dir):
    ast_graph = {}
    try:
        for path in root_dir.rglob("*.hpp"):
            if "build" in path.parts or ".git" in path.parts: continue
            try:
                content = path.read_text(encoding="utf-8")
                structs = re.findall(r'(?:struct|class)\s+\w+(?:\s*:\s*(?:public|private|protected)\s+[\w<>:]+)?', content)
                funcs = re.findall(r'^\s*(?:virtual|inline|static|constexpr|template.*?)?\s*[\w<>:]+\s+[\w<>:]+\s*\([^)]*\)\s*(?:const|noexcept|override)?\s*(?=\{)', content, re.MULTILINE)
                if structs or funcs:
                    ast_graph[path.name] = {"structures": structs, "functions": [f.strip().replace('\n', '') for f in funcs]}
            except Exception: pass
    except Exception: pass
    return json.dumps(ast_graph, indent=2)

def collect_context(run_build):
    context = {}
    context["resources"] = resource_snapshot()
    context["git_status"] = run_cmd(["git", "status", "--short"])
    context["files"] = run_cmd(["rg", "--files"], timeout=20)
    context["ast_graph"] = extract_cpp_ast(ROOT)
    context["cmake_top"] = read_text(ROOT / "CMakeLists.txt", 4000)
    context["nn_tests_cmake"] = read_text(ROOT / "nn" / "tests" / "CMakeLists.txt", 8000)
    context["goal"] = read_text(GOAL_FILE, 8000)
    context["tiny_eval"] = read_text(ROOT / "tiny_honest_eval.py", 8000)

    if run_build:
        context["build"] = run_cmd(
            ["cmake", "--build", str(ROOT / "build"), "--config", "Release", "--parallel", "4"],
            timeout=180,
        )
        context["ctest"] = run_cmd(
            ["ctest", "--test-dir", str(ROOT / "build"), "-C", "Release", "--output-on-failure"],
            timeout=120,
        )
    return context


def write_report(iteration, report):
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}_iteration_{iteration:04d}.md"
    path.write_text(report, encoding="utf-8")
    return path


async def run_iteration(args, state, session, agent_manager):
    iteration = int(state.get("iteration", 0)) + 1
    
    context = collect_context(run_build=args.run_build)
    
    # ---------------------------------------------------------
    # MULTI-AGENT SWARM WORKFLOW
    # ---------------------------------------------------------
    
    # 1. ARCHITECT AGENT (Tree of Thoughts Branching)
    print(f"\n--- SWARM PHASE 1: ARCHITECT (Tree of Thoughts) ---")
    architect_sys = "You are the Architect Agent. You MUST begin your response with a <|think|> block. Inside this block, mathematically evaluate L1 cache locality, memory alignment, and dual-port SIMD saturation (specifically mixing AVX-512 and AVX2 to maximize throughput on Ice Lake CPUs with a single AVX-512 FMA unit). Then output your high-level blueprint. Do NOT edit files."
    architect_user = f"Context Goal:\n{context['goal']}\n\nMathematical AST Graph of Codebase:\n{context['ast_graph'][:6000]}\n\nDraft a concise high-level blueprint for the next phase."
    
    branches = []
    num_branches = 3
    for i in range(num_branches):
        branch_agent = {
            "state": agent_manager["Architect"]["state"], 
            "history": list(agent_manager["Architect"]["history"])
        }
        prompt_variation = architect_user + f"\n\nVariation {i+1}: Approach this problem from a completely unique, different angle than your normal path. Explore an alternative mathematical or structural approach."
        
        report = await chat_with_agent(f"Architect-Branch-{i+1}", branch_agent, session, architect_sys, prompt_variation, args, temperature=0.6)
        branches.append({"report": report, "agent_data": branch_agent})

    # Reviewer scores the branches for A* Queue
    reviewer_tot_user = "The Architect generated three possible blueprints:\n"
    for i, b in enumerate(branches):
        reviewer_tot_user += f"\n--- BLUEPRINT {i+1} ---\n{b['report']}\n"
    reviewer_tot_user += "\nCritique these 3 blueprints. You MUST assign a score out of 10 to each based on cache-locality and hardware-specific SIMD applicability (mixing AVX-512 with AVX2). You MUST format your scoring as 'BLUEPRINT 1 SCORE: X', 'BLUEPRINT 2 SCORE: Y', etc."
    
    tot_evaluation = await chat_with_agent("Reviewer", agent_manager["Reviewer"], session, "You are the Reviewer Agent. Evaluate and score blueprints.", reviewer_tot_user, args, temperature=0.1)
    
    # Extract scores for A* Priority Queue
    import re
    scores = []
    for i in range(num_branches):
        match = re.search(f"BLUEPRINT {i+1} SCORE:\s*(\d+)", tot_evaluation.upper())
        score = int(match.group(1)) if match else 5
        scores.append((score, i))
    
    scores.sort(reverse=True, key=lambda x: x[0]) # Highest score first
    print(f"[swarm] A* Priority Queue established: {scores}")
    
    coder_sys = "You are the Coder Agent. You MUST begin your response with a <|think|> block to map out exact line numbers and memory structs you will modify. Then strictly implement the blueprint using your `edit_file` tool."
    reviewer_sys = "You are the Reviewer Agent. You MUST begin with a <|think|> block auditing the code for memory flaws, L1 cache misses, and missed opportunities to parallelize AVX-512 with AVX2 intrinsics. If flawless, output 'STATUS: APPROVED'. If there are errors, output 'STATUS: REJECTED' and list the exact flaws."
    
    path_successful = False
    
    # A* Search Backtracking Loop
    for score, selected_idx in scores:
        print(f"\n[A* Search] Expanding Node: Blueprint {selected_idx+1} (Heuristic Score: {score}/10)...")
        
        agent_manager["Architect"] = branches[selected_idx]["agent_data"]
        architect_report = branches[selected_idx]["report"]
        
        print(f"\n--- SWARM PHASE 2: CODER ---")
        coder_user = f"The Architect has provided the winning blueprint:\n{architect_report}\n\nExecute this plan by actively writing the C++ files."
        coder_report = await chat_with_agent("Coder", agent_manager["Coder"], session, coder_sys, coder_user, args, temperature=0.1)
        
        print(f"\n--- SWARM PHASE 3: EXECUTION FEEDBACK & CRITIC ---")
        max_critique_loops = 3
        critique_history = f"### ToT Selection & A* Path\n{tot_evaluation}\n\nSelected Path: Blueprint {selected_idx+1}\n\n"
        
        node_success = False
        
        for loop_idx in range(max_critique_loops):
            # 3A. Execution Feedback (Self-Debugging)
            print(f"\n[swarm] Compiling current codebase for verification...")
            build_result = run_cmd(["cmake", "--build", str(ROOT / "build"), "--config", "Release", "--parallel", "4"], timeout=180)
            
            if build_result["returncode"] != 0:
                print("[swarm] MSVC Compilation FAILED. Instantly feeding stderr back to Coder.")
                compiler_err = build_result['stderr'][-4000:] if build_result['stderr'] else build_result['stdout'][-4000:]
                coder_revision_user = f"Your code FAILED to compile! MSVC compiler output:\n{compiler_err}\n\nPlease use your tools to fix these compilation errors immediately."
                coder_report = await chat_with_agent("Coder", agent_manager["Coder"], session, coder_sys, coder_revision_user, args, temperature=0.1)
                critique_history += f"\n### Coder Compiler Fixes (Loop {loop_idx+1})\n{coder_report}\n"
                continue # Recompile
                
            # 3B. Architectural Critique
            reviewer_user = f"The Coder's updates compiled successfully.\n\nCurrent Git Status:\n{context['git_status']['stdout']}\n\nCritique the logical implementation. Start your response with STATUS: APPROVED or STATUS: REJECTED."
            reviewer_report = await chat_with_agent("Reviewer", agent_manager["Reviewer"], session, reviewer_sys, reviewer_user, args, temperature=0.2)
            critique_history += f"\n### Reviewer (Loop {loop_idx+1})\n{reviewer_report}\n"
            
            if "STATUS: APPROVED" in reviewer_report.upper():
                print(f"[swarm] Reviewer APPROVED the implementation on loop {loop_idx+1}.")
                node_success = True
                break
                
            print(f"\n--- SWARM PHASE 2B: CODER REVISION (Loop {loop_idx+1}) ---")
            coder_revision_user = f"The Reviewer REJECTED your logic with the following feedback:\n{reviewer_report}\n\nPlease use your tools to apply the requested fixes immediately."
            coder_report = await chat_with_agent("Coder", agent_manager["Coder"], session, coder_sys, coder_revision_user, args, temperature=0.1)
            critique_history += f"\n### Coder Logic Revision (Loop {loop_idx+1})\n{coder_report}\n"
            
        if node_success:
            path_successful = True
            break
        else:
            print(f"[A* Search] Path {selected_idx+1} dead-ended (failed to compile or pass review 3 times). Backtracking to next best node in Priority Queue...")
            # Note: For perfect backtracking, we would git checkout / reset the files to pristine state here.
            run_cmd(["git", "restore", "."])
    
    if not path_successful:
        print("[A* Search] FATAL: All ToT branches failed. Forcing iteration commit to allow agent to observe failure.")
        
    # ---------------------------------------------------------
    
    full_report = f"# Multi-Agent Swarm Iteration {iteration}\n\n## 1. Architect's Selected Blueprint\n{architect_report}\n\n## 2. Coder's Execution\n{coder_report}\n\n## 3. CRITIC Review Log\n{critique_history}"

    report_path = write_report(iteration, full_report)
    state["iteration"] = iteration
    state.setdefault("reports", []).append(str(report_path.relative_to(ROOT)))
    state["last_report"] = str(report_path.relative_to(ROOT))
    state["last_resource_snapshot"] = context["resources"]
    save_state(state)
    print(f"[worker] Swarm Iteration Complete. Wrote {report_path}")


async def main():
    parser = argparse.ArgumentParser(description="Local LLaMA 24/7 Multi-Agent Swarm Worker with MCP.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="llama-server base URL")
    parser.add_argument("--ctx-size", type=int, default=16384, help="Context size for the model (scales RAM footprint)")
    parser.add_argument("--interval", type=int, default=1800, help="seconds between iterations")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--once", action="store_true", help="run one iteration and exit")
    parser.add_argument("--run-build", action="store_true", help="run build and ctest each iteration")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=-1, help="Max tokens to generate (-1 means no limit, up to context size)")
    parser.add_argument("--min-free-mb", type=float, default=1024.0, help="skip LLM calls below this free RAM")
    parser.add_argument("--retry-base", type=int, default=30, help="initial retry delay after failures")
    parser.add_argument("--retry-max", type=int, default=900, help="maximum retry delay after repeated failures")
    args = parser.parse_args()

    state = load_state()
    completed = 0
    retry_delay = args.retry_base
    
    # Define our lightweight Swarm Agents
    agent_manager = {
        "Architect": {"state": None, "history": []},
        "Coder":     {"state": None, "history": []},
        "Reviewer":  {"state": None, "history": []}
    }
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"]
    )
    
    print("[worker] Starting MCP stdio_client to @modelcontextprotocol/server-memory ...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("[worker] MCP Session initialized. Graph memory available.")
                
                while True:
                    try:
                        await run_iteration(args, state, session, agent_manager)
                        if args.once:
                            return
                        completed += 1
                        if args.max_iterations and completed >= args.max_iterations:
                            return
                        # No artificial delays. Let the Swarm think infinitely.
                    except Exception as exc:
                        append_error("iteration", exc)
                        print(f"[worker] iteration failed: {type(exc).__name__}: {exc}")
                        if args.once:
                            return
                        await asyncio.sleep(10) # brief pause only on crashes
                    except KeyboardInterrupt:
                        print("[worker] stopped")
                        return
                    
    except Exception as e:
        print(f"[worker] MCP server startup failed: {e}. Falling back to standard loop.")
        while True:
            try:
                await run_iteration(args, state, None, agent_manager)
                if args.once:
                    return
                completed += 1
                if args.max_iterations and completed >= args.max_iterations:
                    return
            except Exception as exc:
                print(f"[worker] error: {exc}")
                await asyncio.sleep(10)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
