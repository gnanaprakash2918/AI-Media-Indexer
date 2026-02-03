
import os
import py_compile
import subprocess

IGNORE_DIRS = {
    ".uv_cache", "__pycache__", ".cache", ".github", ".idea", ".venv", "node_modules", ".git"
}

def check_syntax(start_path):
    print(f"[-] Starting Compilation Check from: {start_path}")
    errors = []
    
    for root, dirs, files in os.walk(start_path):
        # Filter ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    py_compile.compile(full_path, doraise=True)
                except py_compile.PyCompileError as e:
                    print(f"[!] Syntax Error in {full_path}")
                    print(e)
                    errors.append(full_path)
                except Exception as e:
                    print(f"[!] Compilation Failed for {full_path}: {e}")
                    errors.append(full_path)

    if not errors:
        print("[P] All files compiled successfully!")
    else:
        print(f"[F] Found syntax errors in {len(errors)} files.")

def run_ruff_targeted():
    print("\n[-] Running Ruff (Targeted Critical Checks)...")
    try:
        # Check for: Undefined names (F821), Syntax Errors (E9), Invalid usage (F63), etc.
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "F821,E9,F63,F7", "--output-format", "full"], 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("[P] Ruff check passed (No undefined variables or syntax errors)!")
        else:
            print(f"[!] Ruff found critical issues (Exit Code: {result.returncode})")
    except FileNotFoundError:
        print("[!] Ruff not found. Skipping linting.")

if __name__ == "__main__":
    base_dir = os.getcwd()
    check_syntax(base_dir)
    run_ruff_targeted()
