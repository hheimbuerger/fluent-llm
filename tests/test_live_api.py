import pytest
from PIL import Image

from fluent_llm import llm, ResponseType

@pytest.mark.asyncio
async def test_text_generation_live():
    """Live test: text generation with the fluent interface (real API)."""
    response = await llm\
        .agent("You are an art evaluator.")\
        .context("You received this painting from your client.")\
        .image("tests/painting.png")\
        .request("Please evaluate this painting and state your opinion whether it's museum-worthy.")\
        .expect(ResponseType.TEXT)\
        .call()
    assert isinstance(response, str)
    print("Text response:", response)

@pytest.mark.asyncio
async def test_image_generation_live():
    """Live test: image generation with the fluent interface (real API)."""
    response = await llm\
        .agent("You are a 17th century classic painter.")\
        .context("You were paid 10 francs for creating a portrait.")\
        .request('Create a portrait of Louis XIV.')\
        .expect(ResponseType.IMAGE)\
        .call()
    assert isinstance(response, Image)
    print("Image response is an Image instance.")
    # Optionally display the image (uncomment if running interactively)
    # response.show()

    # Basic validation that stats were returned
    stats = llm.get_last_call_stats()
    assert "No usage information" not in stats  # Should have usage info

    print("\nUsage stats:", stats)
    print("Response:", response)


@pytest.mark.asyncio
async def test_usage_stats_live():
    """Live test: verify that get_last_call_stats works after an API call."""
    # Make a simple API call
    response = await llm\
        .request("How tall is the Fernsehturm in Berlin?")\
        .expect(ResponseType.TEXT)\
        .call()
    assert 'Berlin' in response

    # Get the usage stats
    stats = llm.get_last_call_stats()

    # Basic validation that stats were returned
    assert stats
    assert "No usage information" not in stats  # Should have usage info

    # Check that we have both input and output tokens in the stats
    has_input = any(token_type in stats for token_type in ("input_tokens", "prompt_tokens"))
    has_output = any(token_type in stats for token_type in ("output_tokens", "completion_tokens"))
    assert has_input, f"Expected input tokens in stats, got: {stats}"
    assert has_output, f"Expected output tokens in stats, got: {stats}"

    # Check that costs are included (should have a $ symbol)
    assert "$" in stats, f"Expected cost information with $ in stats, got: {stats}"

    print("\nUsage stats:", stats)
    print("Response:", response)
