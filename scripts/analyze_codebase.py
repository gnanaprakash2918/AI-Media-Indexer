import os
import re
from pathlib import Path

PROJECT_ROOT = Path("d:/AI-Media-Indexer")
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".idea", ".vscode", "node_modules", ".ruff_cache", "site-packages", "logs", "qdrant_data"}
ENTRY_POINTS = {"main.py", "debug_main.py", "search_cli.py", "agent_main.py", "agent_cli.py", "server.py", "asr_server.py"}

def get_python_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames in-place to exclude directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return files

def get_all_file_contents(exclude_files):
    content = ""
    for f in get_python_files(PROJECT_ROOT):
        if f in exclude_files:
            continue
        try:
            content += f.read_text(encoding="utf-8", errors="ignore")
        except:
            pass
    return content

def is_low_value(file_path):
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        return len(lines) < 5, len(lines)
    except:
        return False, 0

def check_usage():
    all_files = get_python_files(PROJECT_ROOT)
    
    print(f"Reading {len(all_files)} files...")
    file_contents = {}
    for f in all_files:
        try:
            file_contents[f] = f.read_text(encoding="utf-8", errors="ignore")
        except:
            pass

    unused_files = []
    low_value_files = []
    
    # Pre-compute all content combined (excluding the file itself is tricky with a single blob, 
    # so we'll just check against the global blob and subtract 1 count if we want to be pedantic,
    # but practically, checking if it appears *at least twice* (once in itself, once elsewhere) is better,
    # OR just iterate over the map which is fast in memory)
    
    print("Analyzing usage...")
    for file_path in all_files:
        if file_path.name in ENTRY_POINTS:
            continue
            
        is_low, loc = is_low_value(file_path)
        if is_low:
            low_value_files.append((str(file_path.relative_to(PROJECT_ROOT)), loc))

        module_name = file_path.stem
        # Special case: init files
        if module_name == "__init__":
            continue
            
        found = False
        for other_path, content in file_contents.items():
            if other_path == file_path:
                continue
            if module_name in content:
                found = True
                break
        
        if not found:
             unused_files.append(str(file_path.relative_to(PROJECT_ROOT)))

    print("\n=== POTENTIALLY UNUSED FILES (Not referenced by name in other files) ===")
    for f in sorted(unused_files):
        print(f)

    print("\n=== LOW VALUE FILES (< 5 LOC) ===")
    for f, loc in sorted(low_value_files):
        print(f"{f} ({loc} lines)")

if __name__ == "__main__":
    check_usage()
