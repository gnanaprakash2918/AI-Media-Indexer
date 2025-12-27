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

from .interface import LLMInterface, T
from core.utils.observe import observe


class OllamaLLM(LLMInterface):
    """Adapter for the Ollama LLM service with rate limiting."""

    # Class-level rate limiting
    _semaphore: asyncio.Semaphore | None = None
    _last_call_time: float = 0
    _consecutive_errors: int = 0
    _cooldown_until: float = 0
    
    # Configuration
    MAX_CONCURRENT_CALLS = 2  # Max parallel Ollama requests
    MIN_CALL_INTERVAL = 0.5  # Seconds between calls
    MAX_CONSECUTIVE_ERRORS = 3  # Errors before cooldown
    COOLDOWN_DURATION = 10.0  # Seconds to wait after errors

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
            raise RuntimeError(f"Failed to construct Ollama AsyncClient: {e}") from e

    async def _rate_limit(self) -> None:
        """Apply rate limiting and cooldown before making a call."""
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
        """Reset error counter on successful call."""
        OllamaLLM._consecutive_errors = 0

    def _record_error(self) -> None:
        """Track errors and trigger cooldown if needed."""
        OllamaLLM._consecutive_errors += 1
        if OllamaLLM._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            OllamaLLM._cooldown_until = time.time() + self.COOLDOWN_DURATION
            print(f"[Ollama] Too many errors, entering {self.COOLDOWN_DURATION}s cooldown")
            OllamaLLM._consecutive_errors = 0

    @observe("llm_generate")
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from the Ollama service for a user prompt."""
        debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"Ollama generating text for prompt: {debug_prompt}")

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()
            
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
                self._record_error()
                print(f"Ollama generation failed: {e}")
                raise RuntimeError(f"Ollama generation failed: {e}") from e

    @observe("llm_generate_structured")
    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generate structured JSON output matching the Pydantic schema.
        
        Uses Ollama's native format='json' for structured output.
        """
        print("Ollama structured generation requested.")

        # Build JSON schema example with nested structure
        schema_example = self._build_schema_example(schema)
        
        # Construct prompt with explicit JSON instructions
        json_instructions = f"""You MUST respond with ONLY valid JSON matching this EXACT structure:

{schema_example}

CRITICAL RULES:
1. Output ONLY the JSON object, no other text
2. Do not include markdown code blocks or backticks
3. Follow the nested structure EXACTLY - entities and scene must be OBJECTS, not strings
4. Use specific names (Idly not food, Tesla Model 3 not car)

Now respond with JSON:"""

        full_prompt = f"{json_instructions}\n\n{prompt}"
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{full_prompt}"

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()
            
            try:
                # Use format='json' to force JSON mode in Ollama
                resp = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    options={"temperature": kwargs.get("temperature", 0.0)},
                    format="json",
                )
                response_text = resp.get("message", {}).get("content", "")
                
                if not response_text:
                    raise RuntimeError("Empty response from Ollama")
                
                result = self.parse_json_response(response_text, schema)
                self._record_success()
                return result
            except Exception as e:
                self._record_error()
                print(f"Ollama structured generation failed: {e}")
                raise RuntimeError(f"Ollama structured generation failed: {e}") from e

    def _build_schema_example(self, schema: type) -> str:
        """Build a JSON example from a Pydantic schema, recursively handling nested models."""
        import json
        from typing import get_origin, get_args
        
        def build_example(model_class: type) -> dict | str | int | list:
            """Recursively build example for a Pydantic model or primitive."""
            if not hasattr(model_class, 'model_fields'):
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
                    if args and hasattr(args[0], 'model_fields'):
                        # It's a list of Pydantic models
                        example[name] = [build_example(args[0])]
                    else:
                        # Simple list like list[str] or list[int]
                        example[name] = ["<item>"]
                elif hasattr(annotation, 'model_fields'):
                    # It's a nested Pydantic model like SceneContext
                    example[name] = build_example(annotation)
                elif annotation == str or (origin is None and 'str' in str(annotation).lower()):
                    # String field - use description as hint
                    example[name] = desc[:60] + "..." if len(desc) > 60 else desc
                elif annotation == int or (origin is None and 'int' in str(annotation).lower()):
                    example[name] = 0
                elif annotation == bool:
                    example[name] = False
                else:
                    # Handle Optional[str], Optional[X], or unknowns
                    args = get_args(annotation)
                    if args:
                        if args[0] == str:
                            example[name] = desc[:60] + "..." if len(desc) > 60 else desc
                        elif hasattr(args[0], 'model_fields'):
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
        """Generate a description for an image using the Ollama client."""
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
            
            try:
                resp = await self.client.chat(
                    model=self.model,
                    messages=messages,
                )

                content = resp.get("message", {}).get("content")
                self._record_success()
                return str(content) if content else ""
            except Exception as e:
                self._record_error()
                print(f"Ollama image description failed: {e}")
                raise RuntimeError(f"Ollama image description failed: {e}") from e

    @observe("llm_describe_image_structured")
    async def describe_image_structured(
        self,
        schema: type[T],
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Describe an image and return structured JSON output.
        
        Uses format='json' for proper JSON output from vision model.
        """
        img_path = str(Path(image_path))
        
        # Build schema example
        schema_example = self._build_schema_example(schema)
        
        # Construct prompt with JSON instructions  
        # Note: face_cluster_ids is excluded - it's filled by the pipeline, not LLM
        json_prompt = f"""Analyze this image and return ONLY valid JSON matching this EXACT structure:

{schema_example}

CRITICAL RULES:
1. "entities" must be a LIST OF OBJECTS with "name", "category", "visual_details" keys
2. "scene" must be an OBJECT with "location", "action_narrative", "cultural_context", "visible_text" keys
3. DO NOT return strings for "entities" or "scene" - they must be OBJECTS
4. "face_cluster_ids" should be an empty list: []
5. Be SPECIFIC: "Idly" not "food", "Tesla Model 3" not "car", "Nike Air Jordan" not "shoes"

{prompt}

Return JSON:"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": json_prompt,
            "images": [img_path],
        })

        # Apply rate limiting
        assert OllamaLLM._semaphore is not None, "Semaphore not initialized"
        async with OllamaLLM._semaphore:
            await self._rate_limit()
            
            try:
                resp = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    format="json",  # Force JSON output
                )

                content = resp.get("message", {}).get("content", "")
                if not content:
                    raise RuntimeError("Empty response from Ollama vision")
                
                result = self.parse_json_response(content, schema)
                self._record_success()
                return result
                
            except Exception as e:
                self._record_error()
                print(f"Ollama structured image description failed: {e}")
                raise RuntimeError(f"Ollama structured image description failed: {e}") from e
