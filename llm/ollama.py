"""Ollama LLM adapter.

Wraps the `ollama` async client and adapts it to the LLMInterface.
Uses Ollama's native format='json' for structured output.
Includes rate limiting and cooldown to prevent system overload.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import ollama

from core.utils.observe import observe

from .interface import LLMInterface, T


class OllamaLLM(LLMInterface):
    """Adapter for the Ollama LLM service with rate limiting."""

    # Class-level rate limiting
    _semaphore: asyncio.Semaphore | None = None
    _last_call_time: float = 0
    _consecutive_errors: int = 0
    _cooldown_until: float = 0

    # Configuration
    MAX_CONCURRENT_CALLS = 1  # Reduced from 2 to prevent VRAM spikes
    MIN_CALL_INTERVAL = 0.5  # Seconds between calls
    MAX_CONSECUTIVE_ERRORS = 3  # Errors before cooldown
    COOLDOWN_DURATION = 15.0  # Seconds to wait after errors
    MAX_RETRIES = 3  # Internal retries for transient errors
    RUNNER_CRASH_COOLDOWN = 30.0  # Wait longer for model runner to restart

    def __init__(
        self,
        model_name: str | None = None,
        base_url_env: str = "OLLAMA_BASE_URL",
        prompt_dir: str = "./prompts",
    ):
        """Initialize the Ollama AsyncClient.

        Args:
            model_name: Optional model override.
            base_url_env: Environment variable name for the base URL.
            prompt_dir: Prompt template directory.
        """
        super().__init__(prompt_dir=prompt_dir)

        self.model = model_name or os.getenv("OLLAMA_MODEL", "llava:7b")
        base_url = os.getenv(base_url_env, "http://localhost:11434")

        print(
            f"Initializing Ollama client with model={self.model}, base_url={base_url}"
        )

        # Initialize class-level semaphore if not already done
        if OllamaLLM._semaphore is None:
            OllamaLLM._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CALLS)

        try:
            self.client = ollama.AsyncClient(host=base_url)
            print("Ollama AsyncClient initialized successfully.")
        except Exception as e:
            print(f"Failed to construct Ollama AsyncClient: {e}")
            raise RuntimeError(
                f"Failed to construct Ollama AsyncClient: {e}"
            ) from e

    async def _rate_limit(self) -> None:
        """Applies rate limiting and cooldown before making an Ollama API call.

        Enforces minimum call intervals and waits if the system is currently
        in a cooldown period due to recent errors.
        """
        # Check cooldown
        now = time.time()
        if now < OllamaLLM._cooldown_until:
            wait_time = OllamaLLM._cooldown_until - now
            print(f"[Ollama] In cooldown, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)

        # Enforce minimum interval between calls
        elapsed = now - OllamaLLM._last_call_time
        if elapsed < self.MIN_CALL_INTERVAL:
            await asyncio.sleep(self.MIN_CALL_INTERVAL - elapsed)

        OllamaLLM._last_call_time = time.time()

    def _record_success(self) -> None:
        """Resets the consecutive error counter upon a successful API call."""
        OllamaLLM._consecutive_errors = 0

    def _record_error(self, error: Any = None) -> None:
        """Tracks consecutive errors and triggers cooldowns if necessary.

        Detects specific error types like CUDA OOM and model runner crashes
        to trigger immediate long cooldowns or system recovery actions.

        Args:
            error: The exception or error object encountered.
        """
        OllamaLLM._consecutive_errors += 1

        error_str = str(error).lower() if error else ""

        # Check for CUDA OOM (VRAM exhaustion)
        is_cuda_oom = (
            "cuda" in error_str
            and (
                "out of memory" in error_str
                or "unable to allocate" in error_str
            )
        ) or "cudamalloc failed" in error_str

        # Check for model runner crash or 500 errors (VRAM exhaustion)
        is_runner_crash = (
            "model runner has unexpectedly stopped" in error_str
            or "500" in error_str
            or "internal server error" in error_str
            or ("unexpected" in error_str and "stop" in error_str)
        )

        if is_cuda_oom:
            print(
                "[Ollama] CUDA OOM detected! Moving encoder to CPU as fallback..."
            )
            try:
                # Try to free VRAM by moving encoder to CPU
                from core.ingestion.pipeline import IngestionPipeline

                if (
                    hasattr(IngestionPipeline, "_instance")
                    and IngestionPipeline._instance  # type: ignore
                ):
                    try:
                        IngestionPipeline._instance.db.encoder_to_cpu()  # type: ignore
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Ollama] Could not move encoder to CPU: {e}")

            OllamaLLM._cooldown_until = time.time() + self.RUNNER_CRASH_COOLDOWN
            OllamaLLM._consecutive_errors = 0
        elif is_runner_crash:
            OllamaLLM._cooldown_until = time.time() + self.RUNNER_CRASH_COOLDOWN
            print(
                f"[Ollama] Model crash/500 detected! Waiting {self.RUNNER_CRASH_COOLDOWN}s for recovery..."
            )
            OllamaLLM._consecutive_errors = (
                0  # Reset count since we take immediate long cooldown
            )
        elif OllamaLLM._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            OllamaLLM._cooldown_until = time.time() + self.COOLDOWN_DURATION
            print(
                f"[Ollama] Too many errors, entering {self.COOLDOWN_DURATION}s cooldown"
            )
            OllamaLLM._consecutive_errors = 0

    @observe("llm_generate")
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generates a text response from Ollama for a given prompt.

        Args:
            prompt: The user prompt to send to the model.
            **kwargs: Additional model parameters (e.g., temperature).

        Returns:
            The generated text response.
        """
        debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"Ollama generating text for prompt: {debug_prompt}")

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()

            for attempt in range(self.MAX_RETRIES):
                try:
                    resp = await self.client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": kwargs.get("temperature", 0.0)},
                    )
                    content = resp.get("message", {}).get("content")
                    self._record_success()
                    return str(content) if content else ""
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = 2 ** (attempt + 1)
                        print(
                            f"[Ollama] Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue

                    self._record_error(e)
                    print(
                        f"Ollama generation failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise RuntimeError(f"Ollama generation failed: {e}") from e
            return ""  # Should not reach here

    @observe("llm_generate_structured")
    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generates a structured JSON response using Ollama's native schema mode.

        Uses Ollama 0.5+ format=schema feature for guaranteed JSON structure.
        Falls back to format='json' with prompt engineering for older versions.
        """
        print("Ollama structured generation requested.")

        # Get the JSON schema from Pydantic model
        json_schema = schema.model_json_schema()

        # Build the prompt with schema hint (helps model even with native schema)
        schema_hint = f"Return JSON matching this schema: {self._build_schema_example(schema)}"
        full_prompt = f"{schema_hint}\n\n{prompt}"

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{full_prompt}"

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()

            for attempt in range(self.MAX_RETRIES):
                try:
                    # Try Ollama 0.5+ schema-based format first
                    try:
                        resp = await self.client.chat(
                            model=self.model,
                            messages=[{"role": "user", "content": full_prompt}],
                            options={"temperature": kwargs.get("temperature", 0.0)},
                            format=json_schema,  # Ollama 0.5+ native schema
                        )
                    except (TypeError, Exception) as schema_err:
                        # Fall back to simple JSON mode for older Ollama
                        if "format" in str(schema_err) or "schema" in str(schema_err).lower():
                            print("[Ollama] Schema format not supported, using json mode")
                            resp = await self.client.chat(
                                model=self.model,
                                messages=[{"role": "user", "content": full_prompt}],
                                options={"temperature": kwargs.get("temperature", 0.0)},
                                format="json",  # Fallback to simple JSON mode
                            )
                        else:
                            raise

                    response_text = resp.get("message", {}).get("content", "")

                    if not response_text:
                        raise RuntimeError("Empty response from Ollama")

                    result = self.parse_json_response(response_text, schema)
                    self._record_success()
                    return result
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = 2 ** (attempt + 1)
                        print(
                            f"[Ollama] Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue

                    self._record_error(e)
                    print(
                        f"Ollama structured generation failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Ollama structured generation failed: {e}"
                    ) from e
            raise RuntimeError("Exhausted retries")

    def _build_schema_example(self, schema: type) -> str:
        """Builds a JSON example string from a Pydantic schema for prompting.

        Recursively traverses the schema to create a representative JSON object
        that can be used as a template in the LLM prompt.

        Args:
            schema: The Pydantic model class or primitive type.

        Returns:
            A formatted JSON string representing the structure of the schema.
        """
        import json
        from typing import get_args, get_origin

        def build_example(model_class: type) -> dict | str | int | list:
            """Recursively build example for a Pydantic model or primitive."""
            if not hasattr(model_class, "model_fields"):
                # It's a primitive type
                return f"<{model_class.__name__ if hasattr(model_class, '__name__') else 'value'}>"

            example = {}
            for name, field in model_class.model_fields.items():
                annotation = field.annotation
                desc = field.description or name

                # Get the origin (e.g., list, dict, Optional)
                origin = get_origin(annotation)

                if origin is list:
                    # list[EntityDetail] -> show nested example
                    args = get_args(annotation)
                    if args and hasattr(args[0], "model_fields"):
                        # It's a list of Pydantic models
                        example[name] = [build_example(args[0])]
                    else:
                        # Simple list like list[str] or list[int]
                        example[name] = ["<item>"]
                elif hasattr(annotation, "model_fields"):
                    # It's a nested Pydantic model like SceneContext
                    example[name] = build_example(annotation)
                elif annotation is str or (
                    origin is None and "str" in str(annotation).lower()
                ):
                    # String field - use description as hint
                    example[name] = (
                        desc[:60] + "..." if len(desc) > 60 else desc
                    )
                elif annotation is int or (
                    origin is None and "int" in str(annotation).lower()
                ):
                    example[name] = 0
                elif annotation is bool:
                    example[name] = False
                else:
                    # Handle Optional[str], Optional[X], or unknowns
                    args = get_args(annotation)
                    if args:
                        if args[0] is str:
                            example[name] = (
                                desc[:60] + "..." if len(desc) > 60 else desc
                            )
                        elif hasattr(args[0], "model_fields"):
                            example[name] = build_example(args[0])
                        else:
                            example[name] = f"<{name}>"
                    else:
                        example[name] = f"<{name}>"

            return example

        try:
            result = build_example(schema)
            return json.dumps(result, indent=2)
        except Exception as e:
            print(f"Schema example build failed: {e}")
            return "{}"

    @observe("llm_describe_image")
    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generates a text description for a given image.

        Args:
            prompt: The text prompt or question about the image.
            image_path: Path to the image file to analyze.
            system_prompt: Optional system prompt.
            **kwargs: Additional model parameters.

        Returns:
            The model's description of the image.
        """
        img_path = str(Path(image_path))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": [img_path],
            }
        )

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()

            for attempt in range(self.MAX_RETRIES):
                try:
                    resp = await self.client.chat(
                        model=self.model,
                        messages=messages,
                    )

                    content = resp.get("message", {}).get("content")
                    self._record_success()
                    return str(content) if content else ""
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = 2 ** (attempt + 1)
                        print(
                            f"[Ollama] Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue

                    self._record_error(e)
                    print(
                        f"Ollama image description failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Ollama image description failed: {e}"
                    ) from e
            return ""

    @observe("llm_describe_image_structured")
    async def describe_image_structured(
        self,
        schema: type[T],
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Describes an image with Ollama's native JSON schema mode.

        Uses Ollama 0.5+ format=schema for guaranteed structure.
        Falls back to format='json' for older versions.
        """
        img_path = str(Path(image_path))

        # Get JSON schema from Pydantic
        json_schema = schema.model_json_schema()
        schema_example = self._build_schema_example(schema)

        # Prompt with schema hint
        json_prompt = f"""Analyze this image. Return JSON matching this schema:

{schema_example}

Be specific with names (e.g., "Tesla Model 3" not "car", "Idly" not "food").

{prompt}"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": json_prompt,
                "images": [img_path],
            }
        )

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()

            for attempt in range(self.MAX_RETRIES):
                try:
                    # Try Ollama 0.5+ schema-based format first
                    try:
                        resp = await self.client.chat(
                            model=self.model,
                            messages=messages,
                            format=json_schema,  # Native schema format
                        )
                    except (TypeError, Exception) as schema_err:
                        if "format" in str(schema_err) or "schema" in str(schema_err).lower():
                            resp = await self.client.chat(
                                model=self.model,
                                messages=messages,
                                format="json",  # Fallback
                            )
                        else:
                            raise

                    content = resp.get("message", {}).get("content", "")
                    if not content:
                        raise RuntimeError("Empty response from Ollama vision")

                    result = self.parse_json_response(content, schema)
                    self._record_success()
                    return result

                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = 2 ** (attempt + 1)
                        print(
                            f"[Ollama] Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue

                    self._record_error(e)
                    print(
                        f"Ollama structured image description failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Ollama structured image description failed: {e}"
                    ) from e
            raise RuntimeError("Exhausted retries")
