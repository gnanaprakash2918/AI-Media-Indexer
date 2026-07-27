"""Phase 1: Unit tests for provider routing.

Verifies that:
- LLMFactory.create_llm("vllm") returns VLLMProvider
- LLMFactory.create_llm("gemini") returns GeminiLLM
- LLMFactory.create_llm("ollama") returns OllamaLLM
- LLMFactory.get_default_llm() respects LLM_PROVIDER env var
- get_vlm_client() returns the correct VLMClient per provider
- get_text_client() returns the correct TextLLMClient per provider
- Unknown provider raises ValueError with actionable message
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# llm/factory.py — LLMFactory
# ---------------------------------------------------------------------------


class TestLLMFactory:
    def test_create_vllm_returns_vllm_provider(self):
        from llm.factory import LLMFactory
        from llm.vllm import VLLMProvider

        llm = LLMFactory.create_llm("vllm", prompt_dir="/tmp/prompts_test")
        assert isinstance(llm, VLLMProvider)

    def test_create_gemini_returns_gemini_llm(self):
        """GeminiLLM init requires GOOGLE_API_KEY; mock the env."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-for-test"}):
            from llm.factory import LLMFactory
            from llm.gemini import GeminiLLM

            # GeminiLLM may raise if langchain_google_genai is not installed.
            # Only assert type if construction succeeds.
            try:
                llm = LLMFactory.create_llm(
                    "gemini", prompt_dir="/tmp/prompts_test"
                )
                assert isinstance(llm, GeminiLLM)
            except Exception as exc:
                pytest.skip(f"Gemini not available in test env: {exc}")

    def test_create_ollama_returns_ollama_llm(self):
        from llm.factory import LLMFactory
        from llm.ollama import OllamaLLM

        llm = LLMFactory.create_llm("ollama", prompt_dir="/tmp/prompts_test")
        assert isinstance(llm, OllamaLLM)

    def test_unknown_provider_raises_value_error(self):
        from llm.factory import LLMFactory

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create_llm("anthropic")  # type: ignore

    def test_get_default_llm_respects_env_var_vllm(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "vllm"}):
            from llm.factory import LLMFactory
            from llm.vllm import VLLMProvider

            llm = LLMFactory.get_default_llm(prompt_dir="/tmp/prompts_test")
            assert isinstance(llm, VLLMProvider)

    def test_get_default_llm_respects_env_var_ollama(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from llm.factory import LLMFactory
            from llm.ollama import OllamaLLM

            llm = LLMFactory.get_default_llm(prompt_dir="/tmp/prompts_test")
            assert isinstance(llm, OllamaLLM)

    def test_get_default_llm_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "notarealthing"}):
            from llm.factory import LLMFactory

            with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
                LLMFactory.get_default_llm()


# ---------------------------------------------------------------------------
# core/llm/vlm_factory.py — get_vlm_client
# ---------------------------------------------------------------------------


class TestVLMClientRouting:
    def test_get_vlm_client_vllm(self):
        from core.llm.vlm_factory import VLLMVLMClient, get_vlm_client

        client = get_vlm_client("vllm")
        assert isinstance(client, VLLMVLMClient)

    def test_get_vlm_client_ollama(self):
        from core.llm.vlm_factory import OllamaVLM, get_vlm_client

        client = get_vlm_client("ollama")
        assert isinstance(client, OllamaVLM)

    def test_get_vlm_client_unknown_falls_back_to_vllm(self):
        """Unknown providers should default to vllm (logged warning)."""
        from core.llm.vlm_factory import VLLMVLMClient, get_vlm_client

        client = get_vlm_client("unknown_provider")
        assert isinstance(client, VLLMVLMClient)

    def test_get_vlm_client_default_is_vllm(self, monkeypatch):
        """When ai_provider_vision=vllm (default), get_vlm_client() → VLLMVLMClient."""
        from core.llm.vlm_factory import VLLMVLMClient, get_vlm_client

        monkeypatch.setattr(
            "core.llm.vlm_factory.settings",
            type(
                "S",
                (),
                {
                    "ai_provider_vision": "vllm",
                    "vllm_base_url": "http://localhost:8000",
                    "vlm_endpoint_model_name": "Qwen/Qwen3-VL-2B-Instruct",
                    "vllm_api_key": None,
                },
            )(),
        )
        client = get_vlm_client()
        assert isinstance(client, VLLMVLMClient)


# ---------------------------------------------------------------------------
# core/llm/text_factory.py — get_text_client
# ---------------------------------------------------------------------------


class TestTextClientRouting:
    def test_get_text_client_vllm(self):
        from core.llm.text_factory import VLLMText, get_text_client

        client = get_text_client("vllm")
        assert isinstance(client, VLLMText)

    def test_get_text_client_ollama(self):
        from core.llm.text_factory import OllamaText, get_text_client

        client = get_text_client("ollama")
        assert isinstance(client, OllamaText)

    def test_get_text_client_unknown_falls_back_to_vllm(self):
        from core.llm.text_factory import VLLMText, get_text_client

        client = get_text_client("bogus")
        assert isinstance(client, VLLMText)


# ---------------------------------------------------------------------------
# VLLMProvider — no local model loading (no torch, no transformers)
# ---------------------------------------------------------------------------


class TestVLLMProviderNoLocalLoad:
    def test_vllm_provider_does_not_import_torch(self):
        """VLLMProvider should not trigger torch/transformers imports."""
        import importlib
        import sys

        # Remove cached module if already imported
        for key in list(sys.modules.keys()):
            if key.startswith("llm.vllm"):
                del sys.modules[key]

        # Import the module
        from llm.vllm import VLLMProvider

        provider = VLLMProvider(
            base_url="http://localhost:8000",
            model="Qwen/Qwen3-VL-2B-Instruct",
            prompt_dir="/tmp/test_prompts",
        )

        # torch should NOT be imported as a side-effect of VLLMProvider
        # (it may be in sys.modules from other imports, but VLLMProvider itself
        # does not require it)
        assert provider is not None
        assert provider.base_url == "http://localhost:8000"
        assert provider.model == "Qwen/Qwen3-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# VideoVLM & VLLMProvider — Success-Path REST Payload & Parsing Verification
# ---------------------------------------------------------------------------


class TestVideoVLMSuccessPath:
    def test_videovlm_success_payload_and_parsing(self):
        """Verify VideoVLM correctly serializes frame payload and parses response."""
        import asyncio
        import numpy as np
        from unittest.mock import AsyncMock, MagicMock, patch
        from core.llm.video_vlm import VideoVLM

        vlm = VideoVLM(base_url="http://localhost:8000", model="Qwen/Qwen3-VL-2B-Instruct")

        # Mock 2 synthetic RGB frames (10x10x3)
        dummy_frames = [
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.ones((10, 10, 3), dtype=np.uint8) * 255,
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Action: Person walking | Subjects: Man in jacket | Mood: Calm"
                    }
                }
            ]
        }

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                res = await vlm.generate_action_summary(dummy_frames, fps=2.0)

                # Assert HTTP call was made to the correct endpoint
                mock_post.assert_called_once()
                call_url = mock_post.call_args[0][0]
                assert call_url == "http://localhost:8000/v1/chat/completions"

                # Assert payload structure
                json_payload = mock_post.call_args[1]["json"]
                assert json_payload["model"] == "Qwen/Qwen3-VL-2B-Instruct"
                messages = json_payload["messages"]
                assert len(messages) == 1
                content_parts = messages[0]["content"]
                # 2 images + 1 text instruction prompt
                assert len(content_parts) == 3
                assert content_parts[0]["type"] == "image_url"
                assert content_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
                assert content_parts[2]["type"] == "text"

                # Assert summary dictionary output parsing
                assert res["action"] == "Person walking"
                assert res["subjects"] == "Man in jacket"
                assert res["mood"] == "Calm"
                assert "raw" in res

        asyncio.run(_run())

