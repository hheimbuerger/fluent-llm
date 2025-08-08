import pytest
from PIL.Image import Image
from pydantic import BaseModel
from fluent_llm import llm


@pytest.mark.asyncio
async def test_text_generation_live_async():
    """Live test: asynchronous text generation with the fluent interface (real API)."""
    response = await llm\
        .agent("You are terse.")\
        .request("Say hi in one word.")\
        .prompt()
    assert isinstance(response, str)
    print("Text response:", response)


def test_text_generation_live_sync():
    """Live test: synchronous text generation with the fluent interface (real API)."""
    response = llm\
        .agent("You are terse.")\
        .request("Say hi in one word.")\
        .prompt()
    assert isinstance(response, str)
    print("Text response:", response)


@pytest.mark.asyncio
async def test_structured_output_live():
    """Live test: async text generation with structured output (real API)."""
    class EvaluationResult(BaseModel):
        """Example type for structured output."""
        score: int
        reason: str

    response = await llm\
        .agent("You are an art evaluator.")\
        .request("Rate the Mona Lisa on a scale of 1-10 and explain your rating.")\
        .prompt_for_type(EvaluationResult)

    assert isinstance(response, EvaluationResult)
    assert 1 <= response.score <= 10
    assert isinstance(response.reason, str) and len(response.reason) > 0
    print(f"Structured response - Score: {response.score}, Reason: {response.reason}")


@pytest.mark.asyncio
async def test_image_in():
    """Live test: text generation with image in."""
    response = await llm\
        .agent("You are an art evaluator.")\
        .context("You received this painting from your client.")\
        .image("tests/painting.png")\
        .request("Please evaluate this painting and state your opinion whether it's museum-worthy.")\
        .prompt()
    assert isinstance(response, str)
    print("Text response:", response)


@pytest.mark.asyncio
async def test_audio_in():
    """Live test: text generation with the fluent interface (real API)."""
    response = await llm\
        .agent("You are a biologist.")\
        .audio("tests/maybe_cat.mp3")\
        .request("What animal is this?")\
        .prompt()
    assert isinstance(response, str)
    assert 'cat' in response
    print("Text response:", response)


@pytest.mark.asyncio
async def test_image_generation_live():
    """
    Live test: image generation with the fluent interface (real API).

    This test verifies that:
    1. The API returns a valid image generation response
    2. The response contains the expected fields
    3. The image data can be loaded as a PIL Image
    4. Usage statistics are properly tracked
    """
    # Make the API call with specific image generation parameters
    image = await llm\
        .agent("You are a 17th century classic painter.")\
        .context("You were paid 10 francs for creating a portrait.")\
        .request('Create a portrait of Louis XIV.')\
        .prompt_for_image()

    # Verify the response is an image generation call object
    assert isinstance(image, Image)
    assert image.format in ['PNG', 'JPEG', 'WEBP'], f"Unexpected image format: {image.format}"
    print(f"Generated image: {image.size[0]}x{image.size[1]} {image.format}")

    # Verify usage statistics
    stats = llm.usage.generate_report()
    # TODO: ensure there are image generation and image output tokens in the last usage stats
    assert "No usage information" not in stats, "Should have usage information"

    # Optionally display the image (uncomment if running interactively)
    # img.show()


@pytest.mark.asyncio
async def test_usage_stats_live():
    """Live test: verify that get_last_call_stats works after an API call."""
    # Make a simple API call
    response = await llm\
        .request("How tall is the Fernsehturm in Berlin?")\
        .prompt()
    assert 'Berlin' in response

    # Get the usage stats
    assert llm.usage.cost.total_call_cost_usd > 0
    assert len(str(llm.usage)) > 0

    # Check that we have both input and output tokens in the stats
    assert 'input_tokens' in llm.usage.cost.breakdown, f"Expected input tokens in cost breakdown, got: {llm.usage.cost}"
    assert 'output_tokens' in llm.usage.cost.breakdown, f"Expected output tokens in cost breakdown, got: {llm.usage.cost}"

    # --- Case 2: image in -> text out
    img_to_text = await llm\
        .context("You received this painting from your client.")\
        .image("tests/painting.png")\
        .request("Please describe this painting briefly.")\
        .prompt()
    assert isinstance(img_to_text, str)

    # Get the usage stats
    assert llm.usage.cost.total_call_cost_usd > 0
    assert len(str(llm.usage)) > 0

    # --- Case 3: text in -> image out
    generated_img = await llm\
        .agent("You are an abstract artist.")\
        .request("Create an abstract painting representing freedom.")\
        .prompt_for_image()
    assert isinstance(generated_img, Image)

    # Get the usage stats
    assert llm.usage.cost.total_call_cost_usd > 0
    assert len(str(llm.usage)) > 0
