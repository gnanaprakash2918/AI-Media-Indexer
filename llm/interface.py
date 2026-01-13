"""Abstract LLM interface and common prompt utilities.

This module provides an abstract base class that concrete LLM adapters must
implement, plus helper methods for loading prompts and parsing structured
JSON responses returned by language models.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC):
    """Base class that combines abstract generation methods and prompt helpers.

    This class implements prompt loading with an in-memory cache and provides
    utilities for parsing JSON responses into Pydantic models.
    """

    def __init__(self, prompt_dir: str | Path = "./prompts"):
        """Initialize prompt directory and in-memory prompt cache.

        Args:
            prompt_dir: Path to the directory containing prompt templates.
        """
        if isinstance(prompt_dir, str):
            if prompt_dir.strip() == "":
                raise ValueError("prompt_dir cannot be an empty string")
            self.prompt_dir = Path(prompt_dir)
        else:
            self.prompt_dir = prompt_dir

        self._prompt_cache: dict[str, str] = {}

        if not self.prompt_dir.exists():
            # Create the prompt directory if it does not exist.
            print(
                f"Prompt directory '{self.prompt_dir}' does not exist. Creating it."
            )
            try:
                os.makedirs(self.prompt_dir, exist_ok=True)
                print(f"Prompt directory '{self.prompt_dir}' created.")
            except OSError as exc:
                print(f"Failed to create prompt directory '{self.prompt_dir}'.")
                raise OSError(
                    f"Failed to create prompt directory '{self.prompt_dir}'."
                ) from exc

    def load_prompt(self, filename: str) -> str:
        """Load a prompt template from cache or disk.

        Args:
            filename: Name of the template file relative to prompt_dir.

        Returns:
            The loaded prompt content as a string.

        Raises:
            FileNotFoundError: If the file is not present on disk.
        """
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]

        file_path = self.prompt_dir / filename

        if file_path.exists():
            print(f"Loading prompt from '{file_path}'")
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    self._prompt_cache[filename] = content
                    return content
            except Exception as exc:
                print(f"Failed to load prompt from '{file_path}'.")
                raise FileNotFoundError(
                    f"Failed to load prompt from '{file_path}'."
                ) from exc

        raise FileNotFoundError(
            f"Prompt '{filename}' not found on disk or in defaults."
        )

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON errors from LLMs."""
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Count braces and add missing closing braces
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0:
            text = text.rstrip() + "}" * open_braces
        if open_brackets > 0:
            text = text.rstrip() + "]" * open_brackets

        return text

    def parse_json_response(self, response_text: str, schema: type[T]) -> T:
        """Extract JSON from a model response and validate via Pydantic."""
        clean_text = (
            re.sub(r"```[a-zA-Z]*", "", response_text)
            .replace("```", "")
            .strip()
        )

        match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)

        # Try parsing as-is first
        try:
            data = json.loads(clean_text)
            return schema.model_validate(data)
        except json.JSONDecodeError:
            pass

        # Try with JSON repair
        repaired = self._repair_json(clean_text)
        try:
            data = json.loads(repaired)
            return schema.model_validate(data)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Failed: {e}")
            print(f"Failed Payload: {clean_text[:500]}...")
            raise RuntimeError("Invalid JSON format received from LLM") from e
        except Exception as e:
            print(f"Schema Validation Failed: {e}")
            raise RuntimeError(f"Parsed JSON did not match schema: {e}") from e

    def construct_system_prompt(
        self, schema: type[BaseModel], filename: str = "system_prompt.txt"
    ) -> str:
        """Construct a system prompt that includes the JSON schema.

        If the prompt template contains the token {{JSON_SCHEMA}} it will be
        replaced; otherwise a small schema footer is appended.
        """
        raw_prompt = self.load_prompt(filename)
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        if "{{JSON_SCHEMA}}" in raw_prompt:
            return raw_prompt.replace("{{JSON_SCHEMA}}", schema_json)

        return f"{raw_prompt}\n\n## JSON Output Schema\n{schema_json}"

    def construct_user_prompt(self, filename: str = "user_prompt.txt") -> str:
        """Construct a user prompt inserting few-shot examples if present."""
        template = self.load_prompt(filename)
        try:
            few_shot = self.load_prompt("few_shot_examples.txt")
        except FileNotFoundError:
            few_shot = ""

        content = template.replace("{{FEW_SHOT}}", few_shot)
        return content

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a string response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generate a structured response and validate it against the schema."""
        raise NotImplementedError

    @abstractmethod
    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Describe an image and return the textual description."""
        raise NotImplementedError

    async def describe_image_structured(
        self,
        schema: type[T],
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Describe an image and return a structured Pydantic object.

        Default implementation: describe image, then parse JSON.
        Subclasses can override to use native format=json with image.
        """
        # Default: get text description, then try to parse as JSON
        response = await self.describe_image(
            prompt, image_path, system_prompt, **kwargs
        )
        return self.parse_json_response(response, schema)
