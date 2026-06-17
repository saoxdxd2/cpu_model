import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
import ctypes
import llama_cpp
import ctypes
from pathlib import Path


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

def get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        print("[worker] Loading model natively via llama_cpp...")
        _LLM_INSTANCE = llama_cpp.Llama.from_pretrained(
            repo_id="unsloth/gemma-4-E4B-it-GGUF",
            filename="*Q4_K_M.gguf",
            n_ctx=8192,
            n_threads=4,
            n_batch=4,
            n_gpu_layers=0,
            flash_attn=True,
            type_k=2, # q4_0
            type_v=2
        )
    return _LLM_INSTANCE

def llama_chat(messages, temperature=0.2, max_tokens=2048):
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]


def collect_context(run_build):
    context = {}
    context["resources"] = resource_snapshot()
    context["git_status"] = run_cmd(["git", "status", "--short"])
    context["files"] = run_cmd(["rg", "--files"], timeout=20)
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


def build_prompt(context):
    return f"""
You are the local Agentless research worker for this repository.

Project direction:
{context["goal"]}

Workflow:
1. Localization: identify the smallest file/function/experiment that matters.
2. Proposal: describe the next patch or experiment. Do not edit files.
3. Validation: define exact commands and pass/fail criteria.

Hard rules:
- Do not claim the project works unless verified by commands or source evidence.
- Prefer small, testable steps toward the Weight-Space Trading Engine.
- Focus on model optimization, tensor inspection, quantization atlas, reward oracle, and safe evaluation.
- Avoid agent frameworks, CrewAI/LangGraph-style orchestration, and open-ended tool loops.
- Do not recommend direct autonomous source rewrites. Propose patch plans only.
- Identify blockers that could waste months.
- Output a concise Markdown report with:
  1. Current status
  2. Most important finding
  3. Next 3 concrete actions
  4. Files likely involved
  5. Risks / falsification checks

Repository context:

RESOURCE SNAPSHOT:
```json
{json.dumps(context["resources"], indent=2)}
```

GIT STATUS:
```text
{json.dumps(context["git_status"], indent=2)}
```

FILES:
```text
{context["files"]["stdout"][:12000]}
```

TOP CMAKE:
```cmake
{context["cmake_top"]}
```

TESTS CMAKE:
```cmake
{context["nn_tests_cmake"]}
```

TINY HONEST EVAL, IF PRESENT:
```python
{context["tiny_eval"]}
```

BUILD RESULT, IF RUN:
```text
{json.dumps(context.get("build", {}), indent=2)}
```

CTEST RESULT, IF RUN:
```text
{json.dumps(context.get("ctest", {}), indent=2)}
```
"""


def write_report(iteration, report):
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}_iteration_{iteration:04d}.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_iteration(args, state):
    iteration = int(state.get("iteration", 0)) + 1
    resources = resource_snapshot()
    available_mb = resources.get("memory", {}).get("available_physical_mb", 0)
    if available_mb and available_mb < args.min_free_mb:
        report = (
            f"# Iteration {iteration}: skipped\n\n"
            f"Free RAM is {available_mb} MB, below threshold {args.min_free_mb} MB.\n\n"
            "The worker skipped the llama.cpp call to avoid destabilizing the laptop.\n"
        )
        report_path = write_report(iteration, report)
        state["iteration"] = iteration
        state.setdefault("reports", []).append(str(report_path.relative_to(ROOT)))
        state["last_report"] = str(report_path.relative_to(ROOT))
        state["last_resource_snapshot"] = resources
        save_state(state)
        print(f"[worker] skipped low-memory iteration, wrote {report_path}")
        return

    context = collect_context(run_build=args.run_build)

    prompt = build_prompt(context)
    system_msg = {
        "role": "system",
        "content": "You are a rigorous Agentless local engineering worker. Be blunt, concrete, and evidence-driven.",
    }
    
    # Load history to resume and not reset to 0
    history_msgs = []
    reports = state.get("reports", [])
    for report_rel in reports[-5:]:  # Keep up to 5 recent iterations
        report_path = ROOT / report_rel
        if report_path.exists():
            history_msgs.append({"role": "assistant", "content": read_text(report_path, max_chars=1500)})
            history_msgs.append({"role": "user", "content": "Continue with the next iteration based on the updated state."})

    # Context window shifting mechanism (like cline/antigravity)
    # Approximate limit: ~8192 tokens = ~32000 chars. We reserve 4000 for generation and keep prompt safe.
    MAX_CHARS = 24000
    
    messages = [system_msg] + history_msgs + [{"role": "user", "content": prompt}]
    
    # Truncate oldest history if we exceed the limit
    while len(messages) > 2 and sum(len(m["content"]) for m in messages) > MAX_CHARS:
        print("[worker] Context limit approached. Truncating oldest history message to prevent reset.")
        messages.pop(1) # Remove oldest message after system prompt

    try:
        report = llama_chat(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        if "context" in str(exc).lower() or "token" in str(exc).lower():
            print("[worker] 400 Context Limit Exceeded. Falling back to zero-history truncated prompt.")
            # Aggressively truncate if the prompt itself is too large
            truncated_prompt = prompt[:16000] + "\n\n[TRUNCATED DUE TO CONTEXT LIMIT]"
            messages = [system_msg, {"role": "user", "content": truncated_prompt}]
            report = llama_chat(
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            raise

    report_path = write_report(iteration, report)
    state["iteration"] = iteration
    state.setdefault("reports", []).append(str(report_path.relative_to(ROOT)))
    state["last_report"] = str(report_path.relative_to(ROOT))
    state["last_resource_snapshot"] = context["resources"]
    save_state(state)
    print(f"[worker] wrote {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Local llama.cpp 24/7 project worker.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="llama-server base URL")
    parser.add_argument("--interval", type=int, default=1800, help="seconds between iterations")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--once", action="store_true", help="run one iteration and exit")
    parser.add_argument("--run-build", action="store_true", help="run build and ctest each iteration")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-free-mb", type=float, default=1024.0, help="skip LLM calls below this free RAM")
    parser.add_argument("--retry-base", type=int, default=30, help="initial retry delay after failures")
    parser.add_argument("--retry-max", type=int, default=900, help="maximum retry delay after repeated failures")
    args = parser.parse_args()

    state = load_state()
    completed = 0
    retry_delay = args.retry_base
    while True:
        try:
            run_iteration(args, state)
            retry_delay = args.retry_base
            if args.once:
                return
            print(f"[worker] retry delay is {retry_delay}s")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, args.retry_max)
            continue
        except Exception as exc:
            append_error("iteration", exc)
            print(f"[worker] iteration failed: {type(exc).__name__}: {exc}")
            if args.once:
                return
            print(f"[worker] retry delay is {retry_delay}s")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, args.retry_max)
            continue
        except KeyboardInterrupt:
            print("[worker] stopped")
            return

        completed += 1
        if args.once or (args.max_iterations and completed >= args.max_iterations):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
