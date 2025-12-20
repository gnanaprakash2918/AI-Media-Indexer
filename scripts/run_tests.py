import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

console = Console()

def check_environment(args):
    if os.environ.get("ENV") == "production":
        console.print("[bold red]CRITICAL: Production environment detected![/]")
        sys.exit(1)

    if args.dry_run:
        console.print("[yellow]DRY-RUN: No changes will be made.[/]")
        return

    if not args.force and os.environ.get("AI_INDEXER_TEST_MODE") != "1":
        console.print("[bold red]WARNING: Safe directories (qdrant, postgres, logs) will be wiped![/]")
        if not Confirm.ask("Proceed?"):
            sys.exit(0)

def cleanup_artifacts(args):
    targets = ["qdrant_data", "qdrant_data_embedded", "postgres_data", "logs", ".pytest_cache", "htmlcov", "reports"]
    pycache_dirs = list(Path(".").rglob("__pycache__"))

    if args.dry_run:
        console.print(f"[cyan]Would delete: {targets} + {len(pycache_dirs)} __pycache__ dirs[/]")
        return

    for target in targets:
        path = Path(target)
        if path.exists():
            try:
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
                console.print(f"[green]Deleted: {target}[/]")
            except Exception as e:
                console.print(f"[red]Failed {target}: {e}[/]")

    for p in pycache_dirs:
        shutil.rmtree(p, ignore_errors=True)

def run_suite(args, extra_args=None):
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / f"report_{timestamp}.html"
    cov_path = report_dir / f"coverage_{timestamp}"
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"--cov-report=html:{cov_path}",
        "--self-contained-html",
        "tests"
    ]

    
    if args.markers:
        cmd.extend(["-m", args.markers])
    if extra_args:
        cmd.extend(extra_args)
    
    console.print(f"[bold green]Running pytest...[/]")
    console.print(f"[cyan]Report: {report_path}[/]")
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--markers", type=str)
    parser.add_argument("--llm-provider", type=str, default="mock", help="mock, ollama, gemini")
    
    args = parser.parse_args()
    check_environment(args)
    cleanup_artifacts(args)
    
    # Set test mode environment variable
    os.environ["AI_INDEXER_TEST_MODE"] = "1"

    # Pass LLM Provider to pytest
    cmd_args = argparse.Namespace(**vars(args))
    extra_args = []
    if args.llm_provider:
        extra_args.append(f"--llm-provider={args.llm_provider}")
    
    sys.exit(run_suite(cmd_args, extra_args))

if __name__ == "__main__":
    main()
