import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC):
    """Base class that combines:
    1. Abstract methods for Generation (interface).
    2. Concrete methods for Prompt Loading with caching.
    """

    def __init__(self, prompt_dir: str | Path = "./prompts"):
        if isinstance(prompt_dir, str):
            if prompt_dir.strip() == "":
                raise ValueError("prompt_dir cannot be an empty string")
            self.prompt_dir = Path(prompt_dir)
        else:
            self.prompt_dir = prompt_dir

        self._prompt_cache: dict[str, str] = {}

        if not self.prompt_dir.exists():
            print(
                f"Prompt directory '{self.prompt_dir}' does not exist. Creating it."
            )
            try:
                os.makedirs(self.prompt_dir, exist_ok=True)
                print(f"Prompt directory '{self.prompt_dir}' created.")
            except OSError:
                print(f"Failed to create prompt directory '{self.prompt_dir}'.")
                raise

    def load_prompt(self, filename: str) -> str:
        """Loads a prompt template from memory cache or disk."""
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
            except Exception:
                print(f"Failed to load prompt from '{file_path}'.")
                raise

        raise FileNotFoundError(
            f"Prompt '{filename}' not found on disk or in defaults."
        )

    def parse_json_response(self, response_text: str, schema: type[T]) -> T:
        """Extracts JSON from text and validates against Pydantic."""
        clean_text = (
            re.sub(r"```[a-zA-Z]*", "", response_text)
            .replace("```", "")
            .strip()
        )

        match = re.search(r"(\{.*})", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)

        try:
            data = json.loads(clean_text)
            return schema.model_validate(data)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Failed: {e}")
            print(f"Failed Payload: {clean_text}")
            raise RuntimeError("Invalid JSON format received from LLM")
        except Exception as e:
            print(f"Schema Validation Failed: {e}")
            raise RuntimeError(f"Parsed JSON did not match schema: {e}")

    def construct_system_prompt(
        self, schema: type[BaseModel], filename: str = "system_prompt.txt"
    ) -> str:
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

        content = template.replace("{{FEW_SHOT}}", few_shot)
        return content

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        raise NotImplementedError

    @abstractmethod
    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
