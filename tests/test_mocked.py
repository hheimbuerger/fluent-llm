"""Mocked tests for the public `fluent_llm` interface.

These tests patch only the single integration point with OpenAI –
`fluent_llm.openai.invoker.call_llm_api` – so the rest of the library runs
unchanged without any network access.

Covered surface areas:
1. Fluent builder DSL (`LLMPromptBuilder`).
2. Correct propagation of the requested `ResponseType`.
3. Usage-statistics tracking via `fluent_llm.usage_tracker`.
4. Behaviour of the module-level `llm` instance and package version exposure.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fluent_llm import ResponseType
from fluent_llm.builder import LLMPromptBuilder, llm
from fluent_llm.usage_tracker import tracker

# ---------------------------------------------------------------------------
# Fixture: patch the OpenAI invoker with an AsyncMock
# ---------------------------------------------------------------------------

@pytest.fixture()
def patch_call_llm_api(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace `call_llm_api` with an AsyncMock that returns minimal stubs."""

    async def _stub(*_args: Any, **kwargs: Any):
        expect_type: ResponseType = kwargs["expect_type"]
        if expect_type is ResponseType.TEXT:
            return "MOCK_TEXT_RESPONSE"
        if expect_type is ResponseType.IMAGE:
            return b"\x89PNG\r\n\x1a\n"  # PNG signature – sufficient for tests
        raise NotImplementedError(expect_type)

    mock = AsyncMock(side_effect=_stub)
    # Patch at the invoker source *and* the alias imported into builder.py
    monkeypatch.setattr("fluent_llm.openai.invoker.call_llm_api", mock, raising=True)
    monkeypatch.setattr("fluent_llm.builder.call_llm_api", mock, raising=True)
    return mock

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _builder() -> LLMPromptBuilder:
    """Return a minimal builder reused in tests."""
    return (
        LLMPromptBuilder()
        .agent("You are a cyber-security assistant.")
        .request("Scan the provided code for vulnerabilities.")
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_generation(patch_call_llm_api: AsyncMock) -> None:
    """`ResponseType.TEXT` propagates and the mock string is returned."""
    response = await _builder().expect(ResponseType.TEXT).call()
    assert response == "MOCK_TEXT_RESPONSE"

    patch_call_llm_api.assert_awaited_once()
    assert patch_call_llm_api.await_args.kwargs["expect_type"] is ResponseType.TEXT


@pytest.mark.asyncio
async def test_image_generation(patch_call_llm_api: AsyncMock) -> None:
    """`ResponseType.IMAGE` returns bytes that start with a PNG header."""
    img = await _builder().expect(ResponseType.IMAGE).call()
    assert isinstance(img, bytes)
    assert img.startswith(b"\x89PNG")


def test_usage_stats_reset_and_track() -> None:
    """`tracker` should reset and accumulate usage statistics correctly."""

    class _Resp:  # minimal object with attributes expected by tracker
        def __init__(self, model: str, in_t: int, out_t: int):
            self.model = model
            self.usage = type("usage", (), {
                "input_tokens": in_t,
                "output_tokens": out_t,
                "total_tokens": in_t + out_t,
            })()

    tracker.reset_usage()
    tracker.track_usage(_Resp("gpt-4o", 10, 15))
    tracker.track_usage(_Resp("gpt-4o", 5, 5))

    stats = tracker.get_usage("gpt-4o")
    assert stats["input_tokens"] == 15
    assert stats["output_tokens"] == 20
    assert stats["total_tokens"] == 35
    assert stats["call_count"] == 2


@pytest.mark.asyncio
async def test_public_llm_instance(patch_call_llm_api: AsyncMock) -> None:
    """Smoke-test the convenience `llm` builder instance."""
    txt = await (
        llm.agent("You are terse.")
           .request("Say hi in one word.")
           .expect(ResponseType.TEXT)
           .call()
    )
    assert txt == "MOCK_TEXT_RESPONSE"


def test_package_has_version() -> None:
    """Package must expose a version string via importlib.metadata."""
    import importlib.metadata as _md

    assert _md.version("fluent-llm")
