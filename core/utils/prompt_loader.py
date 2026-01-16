"""Centralized Prompt Loader Utility.

All prompts are loaded from external .txt files in the prompts/ directory.
This ensures:
- No hardcoding in Python files
- Easy customization without code changes
- Version control of prompts separately
- Works for ANY video content worldwide
"""

from __future__ import annotations

import functools
from pathlib import Path

from core.utils.logger import get_logger

log = get_logger(__name__)

# Base prompts directory
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@functools.lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt from external file with caching.

    Args:
        name: Prompt file name (without .txt extension).

    Returns:
        Prompt content as string, empty if not found.
    """
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        content = prompt_file.read_text(encoding="utf-8")
        log.debug(
            f"[PromptLoader] Loaded prompt: {name} ({len(content)} chars)"
        )
        return content
    else:
        log.warning(f"[PromptLoader] Prompt file not found: {prompt_file}")
        return ""


def get_prompt(name: str, **kwargs: str) -> str:
    """Load and format a prompt with variables.

    Args:
        name: Prompt file name (without .txt extension).
        **kwargs: Variables to substitute in the prompt.

    Returns:
        Formatted prompt string.
    """
    template = load_prompt(name)
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        log.error(f"[PromptLoader] Missing variable in prompt {name}: {e}")
        return template


def list_prompts() -> list[str]:
    """List all available prompt files.

    Returns:
        List of prompt names (without .txt extension).
    """
    if not PROMPTS_DIR.exists():
        return []
    return [f.stem for f in PROMPTS_DIR.glob("*.txt")]


def reload_prompts() -> None:
    """Clear the prompt cache to reload from disk."""
    load_prompt.cache_clear()
    log.info("[PromptLoader] Prompt cache cleared")
