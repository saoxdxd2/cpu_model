import os, re, subprocess

def restore_and_fix():
    # Restore the files I messed up with the previous bad regex
    subprocess.run(["git", "checkout", "training/ai_ide_trainer.cpp", 
                    "training/bit_trainer.cpp", "training/fidelity_test.cpp",
                    "training/offline_trainer.cpp", "training/ppo_trainer.cpp",
                    "training/visual_trainer.cpp", "deployment/nca_deploy.cpp",
                    "deployment/test_autonomous_dev.cpp", "nn/tests/test_final_hardening.cpp",
                    "nn/tests/test_memory_compression.cpp", "nn/tests/test_multi_agent_cost.cpp",
                    "nn/tests/test_stability_audit.cpp", "nn/tests/test_vision_branding.cpp",
                    "nn/tests/test_gup.cpp"], check=True)

    directories = ['training', 'deployment', 'nn/tests', 'agentic_env']
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            if 'build' in root:
                continue
            for name in files:
                if not (name.endswith('.cpp') or name.endswith('.hpp')):
                    continue
                path = os.path.join(root, name)
                
                # We do not touch weight_adapter again, it's manually fixed
                if 'weight_adapter' in name:
                    continue

                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                
                # Careful regex for step_batch:
                # ->step_batch(A, B, C, D) -> ->step_geometric(A, B, C, 0.2f)
                new_content = re.sub(
                    r'->step_batch\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*[^)]+\)',
                    r'->step_geometric(\1, \2, \3, 0.2f)',
                    new_content
                )
                
                # Careful regex for step:
                # ->step(A, B, C) -> ->step_geometric(A, B, C, 0.0f)
                new_content = re.sub(
                    r'->step\s*\(([^,]+),\s*([^,]+),\s*([^,)]+?(?:\([^)]*\))?[^)]*)\)',
                    r'->step_geometric(\1, \2, \3, 0.0f)',
                    new_content
                )
                # Catch any nested parens in the third arg (like response.data())
                # Actually, simpler: replace `->step` to `->step_geometric` and add `0.0f` as 4th arg
                def replace_step(match):
                    args = match.group(1)
                    return f'->step_geometric({args}, 0.0f)'
                
                # Match `->step( ... )` balancing parens isn't perfectly supported in python re
                # But we can split by comma. Usually it's `engine_->step(arg1, arg2, arg3);`
                # Let's just do string replacement where we know the exact lines
                # Or use regex that stops at `);`
                
                new_content = re.sub(r'engine_->step\s*\((.*?)\);', r'engine_->step_geometric(\1, 0.0f);', new_content)
                # Fix optimizer step that got caught
                new_content = new_content.replace('optimizer.step_geometric(', 'optimizer.step(')
                
                # Specific legacy VNNI deprecations for training files
                if 'fidelity_test.cpp' in path or 'offline_trainer.cpp' in path:
                    new_content = re.sub(r'([^\n]*expert_pool_gate[^\n]*)', r'// \1', new_content)
                    new_content = re.sub(r'([^\n]*expert_pool_up[^\n]*)', r'// \1', new_content)
                    new_content = re.sub(r'([^\n]*halting_gate[^\n]*)', r'// \1', new_content)
                    new_content = re.sub(r'([^\n]*mx_quantize[^\n]*)', r'// \1', new_content)
                    new_content = new_content.replace('wr.n_experts', '1024')
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed {path}")

if __name__ == '__main__':
    restore_and_fix()
