"""Unified abstract LLM/VLM interface.

This module provides an abstract base class that concrete LLM adapters must
implement, plus helper methods for loading prompts and parsing structured
JSON responses.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from config import settings
from core.utils.logger import log

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Base class that combines abstract text and vision generation methods.
    
    Provides utilities for JSON repair, structured generation, and prompt caching.
    """

    def __init__(self, prompt_dir: str | Path | None = None):
        if prompt_dir is None:
            self.prompt_dir = settings.prompt_dir
        elif isinstance(prompt_dir, str):
            if prompt_dir.strip() == "":
                raise ValueError("prompt_dir cannot be an empty string")
            self.prompt_dir = Path(prompt_dir)
        else:
            self.prompt_dir = prompt_dir

        self._prompt_cache: dict[str, str] = {}

        if not self.prompt_dir.exists():
            log(f"Prompt directory '{self.prompt_dir}' does not exist. Creating it.")
            try:
                os.makedirs(self.prompt_dir, exist_ok=True)
            except OSError as exc:
                raise OSError(f"Failed to create prompt directory '{self.prompt_dir}'.") from exc

    def load_prompt(self, filename: str) -> str:
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]
        file_path = self.prompt_dir / filename
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    self._prompt_cache[filename] = content
                    return content
            except Exception as exc:
                raise FileNotFoundError(f"Failed to load prompt from '{file_path}'.") from exc
        raise FileNotFoundError(f"Prompt '{filename}' not found on disk or in defaults.")

    def parse_json_response(self, response_text: str, schema: type[T]) -> T:
        if not response_text or not response_text.strip():
            raise RuntimeError("Empty response received from LLM")

        clean_text = re.sub(r"```[a-zA-Z]*", "", response_text).replace("```", "").strip()
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

        match = re.search(r"(\{.*\}|\[.*\])", clean_text, re.DOTALL)
        candidate = match.group(1) if match else clean_text

        try:
            data = json.loads(candidate)
            return schema.model_validate(data)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON format received from LLM: {e}")

    def construct_system_prompt(self, schema: type[BaseModel], filename: str = "system_prompt.txt") -> str:
        raw_prompt = self.load_prompt(filename)
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        if "{{JSON_SCHEMA}}" in raw_prompt:
            return raw_prompt.replace("{{JSON_SCHEMA}}", schema_json)
        return f"{raw_prompt}\n\n## JSON Output Schema\n{schema_json}"

    def construct_user_prompt(self, filename: str = "user_prompt.txt") -> str:
        template = self.load_prompt(filename)
        try:
            few_shot = self.load_prompt("few_shot_examples.txt")
        except FileNotFoundError:
            few_shot = ""
        return template.replace("{{FEW_SHOT}}", few_shot)

    # --- ABSTRACT GENERATION METHODS ---

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a string response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(self, schema: type[T], prompt: str, system_prompt: str = "", **kwargs: Any) -> T:
        """Generate a structured response and validate it against the schema."""
        raise NotImplementedError

    # --- VISION METHODS ---
    
    @abstractmethod
    async def describe_image(self, prompt: str, image_path: str | Path, system_prompt: str = "", **kwargs: Any) -> str:
        """Describe an image and return the textual description."""
        raise NotImplementedError
        
    @abstractmethod
    async def describe_image_from_bytes(self, prompt: str, image_bytes: bytes, system_prompt: str = "", **kwargs: Any) -> str:
        """Describe an image from raw bytes."""
        raise NotImplementedError

    async def describe_image_structured(self, schema: type[T], prompt: str, image_path: str | Path, system_prompt: str = "", **kwargs: Any) -> T:
        if "JSON Output Schema" not in prompt and "JSON Output Schema" not in system_prompt:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"{prompt}\n\n## JSON Output Schema\n{schema_json}\n\nOUTPUT VALID JSON ONLY."

        kwargs.setdefault("response_format", {"type": "json_object"})
        response = await self.describe_image(prompt, image_path, system_prompt, **kwargs)
        return self.parse_json_response(response, schema)

    async def unload_model(self) -> None:
        """Explicitly unload the model from memory/VRAM."""
        pass
