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

from typing import Any
from unittest.mock import AsyncMock

import pytest

from fluent_llm.messages import ResponseType
from fluent_llm.builder import LLMPromptBuilder, llm
from fluent_llm.usage_tracker import tracker
from fluent_llm.model_selector import UnresolvableModelError

# ---------------------------------------------------------------------------
# Fixture: patch the OpenAI invoker with an AsyncMock
# ---------------------------------------------------------------------------

@pytest.fixture()
def patch_call_llm_api(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the OpenAI provider's prompt_via_api with an AsyncMock that returns minimal stubs."""
    from fluent_llm.providers.openai.gpt import OpenAIProvider

    async def _stub(*args, **kwargs):
        # The first argument is 'self', which we can ignore
        if len(args) > 1:
            model, messages, expect_type = args[1:4]
        else:
            expect_type = kwargs.get('expect_type')
            
        if expect_type is ResponseType.TEXT:
            return "MOCK_TEXT_RESPONSE"
        if expect_type is ResponseType.IMAGE:
            return b"\x89PNG\r\n\x1a\n"  # PNG signature
        raise NotImplementedError(f"Unsupported expect_type: {expect_type}")

    mock = AsyncMock(spec=OpenAIProvider.prompt_via_api)
    mock.side_effect = _stub
    monkeypatch.setattr("fluent_llm.providers.openai.gpt.OpenAIProvider.prompt_via_api", mock)
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
    """`ResponseType.IMAGE` returns image generation call with correct parameters."""
    # Make the API call using the builder pattern with prompt_for_image()
    builder = LLMPromptBuilder()
    result = await builder.image("A test image").prompt_for_image()

    # Verify the mock was called with the correct parameters
    patch_call_llm_api.assert_awaited_once()
    
    # Get the arguments from the mock call
    args, kwargs = patch_call_llm_api.await_args
    
    # Verify the expect_type is set to IMAGE
    assert kwargs['expect_type'] == ResponseType.IMAGE
    
    # Verify the response is the raw image data
    assert result == b"\x89PNG\r\n\x1a\n"


# def test_usage_stats_reset_and_track() -> None:
#     """`tracker` should reset and accumulate usage statistics correctly."""
#     from fluent_llm.usage_tracker import UsageStats, UsageTracker
    
#     # Create a new tracker for testing
#     test_tracker = UsageTracker()
    
#     # Reset the tracker before starting
#     test_tracker.reset_usage()
    
#     # Track usage for the same model
#     from fluent_llm.usage_tracker import UsageStats
    
#     test_tracker.track_usage("gpt-4o", UsageStats(
#         input_tokens=10,
#         output_tokens=15,
#         total_tokens=25,
#         call_count=1
#     ))

#     test_tracker.track_usage("gpt-4o", UsageStats(
#         input_tokens=5,
#         output_tokens=5,
#         total_tokens=10,
#         call_count=1
#     ))
    
#     # Get and verify the stats
#     stats = test_tracker.get_usage("gpt-4o")
#     assert stats["input_tokens"] == 15
#     assert stats["output_tokens"] == 20
#     assert stats["total_tokens"] == 35
#     assert stats["call_count"] == 2


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


# ---------------------------------------------------------------------------
# Provider and Model Preference Tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def patch_anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the Anthropic provider's prompt_via_api with an AsyncMock."""
    from fluent_llm.providers.anthropic.claude import AnthropicProvider

    async def _anthropic_stub(*args, **kwargs):
        # The first argument is 'self', which we can ignore
        if len(args) > 1:
            model, messages, expect_type = args[1:4]
        else:
            expect_type = kwargs.get('expect_type')
            
        if expect_type is ResponseType.TEXT:
            return "MOCK_ANTHROPIC_TEXT_RESPONSE"
        raise NotImplementedError(f"Unsupported expect_type: {expect_type}")

    mock = AsyncMock(spec=AnthropicProvider.prompt_via_api)
    mock.side_effect = _anthropic_stub
    monkeypatch.setattr("fluent_llm.providers.anthropic.claude.AnthropicProvider.prompt_via_api", mock)
    return mock


@pytest.mark.asyncio
async def test_provider_preference_anthropic_text(patch_anthropic_provider: AsyncMock) -> None:
    """Test that specifying 'anthropic' provider works for text requests."""
    response = await llm \
        .provider("anthropic") \
        .request("Hello, how are you?") \
        .prompt()
    
    assert response == "MOCK_ANTHROPIC_TEXT_RESPONSE"
    # Verify the Anthropic provider was called
    patch_anthropic_provider.assert_called_once()


@pytest.mark.asyncio
async def test_provider_preference_impossible_capability() -> None:
    """Test that requesting impossible capabilities raises UnresolvableModelError."""
    with pytest.raises(UnresolvableModelError, match="has no models supporting required capabilities.*image output"):
        await llm \
            .provider("anthropic") \
            .request("Create an image of a cat") \
            .prompt_for_image()


@pytest.mark.asyncio
async def test_model_preference_specific_model(patch_call_llm_api: AsyncMock) -> None:
    """Test that specifying a specific model works when capabilities match."""
    response = await llm \
        .model("gpt-4o-mini") \
        .request("Hello, how are you?") \
        .prompt()
    
    assert response == "MOCK_TEXT_RESPONSE"
    patch_call_llm_api.assert_called_once()


@pytest.mark.asyncio
async def test_model_preference_impossible_capability() -> None:
    """Test that requesting impossible model capabilities raises UnresolvableModelError."""
    with pytest.raises(UnresolvableModelError, match="Model.*does not support image output"):
        await llm \
            .model("gpt-4o-mini") \
            .request("Create an image") \
            .prompt_for_image()


@pytest.mark.asyncio
async def test_unknown_provider() -> None:
    """Test that specifying an unknown provider raises UnresolvableModelError."""
    with pytest.raises(UnresolvableModelError, match="Provider 'unknown' not supported"):
        await llm \
            .provider("unknown") \
            .request("Hello") \
            .prompt()


@pytest.mark.asyncio
async def test_unknown_model() -> None:
    """Test that specifying an unknown model raises UnresolvableModelError."""
    with pytest.raises(UnresolvableModelError, match="Model 'unknown-model' not found in any provider"):
        await llm \
            .model("unknown-model") \
            .request("Hello") \
            .prompt()
