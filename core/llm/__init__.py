"""LLM and VLM provider package.

Provides unified interfaces for Text LLMs and Vision-Language Models (VLMs).

Default implementation: Qwen3-VL via vLLM REST endpoint per AGENTS.md.
"""

from __future__ import annotations

from core.llm.factory import LLMFactory
from core.llm.interface import LLMInterface
from core.llm.text_factory import OllamaText, VLLMText, get_text_client
from core.llm.video_vlm import VideoVLM
from core.llm.vlm_factory import OllamaVLM, VLLMVLMClient, get_vlm_client

__all__ = [
    "LLMFactory",
    "LLMInterface",
    "VideoVLM",
    "VLLMVLMClient",
    "OllamaVLM",
    "get_vlm_client",
    "VLLMText",
    "OllamaText",
    "get_text_client",
]
