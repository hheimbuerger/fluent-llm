import pytest
from PIL.Image import Image
from pydantic import BaseModel
from fluent_llm.builder import LLMPromptBuilder
from fluent_llm.model_selector import ModelSelectionStrategy
from fluent_llm.providers.anthropic.claude import AnthropicProvider


@pytest.fixture(scope="module", autouse=True)
def anthropic_llm():
    """Configure a separate builder instance with an Anthropic model selector."""
    selector_class = type('', \
                    (ModelSelectionStrategy,), \
                    dict(
                        select_model = lambda self, messages, expect_type: (AnthropicProvider(), "claude-sonnet-4",)
                    )
               )
    return LLMPromptBuilder(model_selector=selector_class())


@pytest.mark.asyncio
async def test_text_generation_live_async(anthropic_llm):
    """Live test: asynchronous text generation with the fluent interface (real API)."""
    response = await anthropic_llm\
        .agent("You are terse.")\
        .request("Say hi in one word.")\
        .prompt()
    assert isinstance(response, str)
    print("Text response:", response)


@pytest.mark.asyncio
async def test_image_in(anthropic_llm):
    """Live test: text generation with image in."""
    response = await anthropic_llm\
        .agent("You are an art evaluator.")\
        .context("You received this painting from your client.")\
        .image("tests/painting.png")\
        .request("Please evaluate this painting and state your opinion whether it's museum-worthy.")\
        .prompt()
    assert isinstance(response, str)
    print("Text response:", response)


@pytest.mark.asyncio
async def test_usage_stats_live(anthropic_llm):
    """Live test: verify that get_last_call_stats works after various Anthropic API calls."""
    # --- Case 1: text in -> text out
    response = await anthropic_llm.request("What is the capital of France?").prompt()
    assert "Paris" in response

    # Get the usage stats
    assert anthropic_llm.usage.cost.total_call_cost_usd > 0
    assert len(str(anthropic_llm.usage)) > 0

    # Check that we have both input and output tokens in the stats
    assert 'input_tokens' in anthropic_llm.usage.cost.breakdown, f"Expected input tokens in cost breakdown, got: {anthropic_llm.usage.cost}"
    assert 'output_tokens' in anthropic_llm.usage.cost.breakdown, f"Expected output tokens in cost breakdown, got: {anthropic_llm.usage.cost}"

    # # --- Case 2: image in -> text out
    # img_to_text = await anthropic_llm\
    #     .context("You received this painting from your client.")\
    #     .image("tests/painting.png")\
    #     .request("Please describe this painting briefly.")\
    #     .prompt()
    # assert isinstance(img_to_text, str)

    # # Get the usage stats
    # assert anthropic_llm.usage.cost.total_call_cost_usd > 0
    # assert len(str(anthropic_llm.usage)) > 0

    print(anthropic_llm.usage)

    # --- Case 3: text in -> image out
    # !NOT SUPPORTED BY CLAUDE!
    # generated_img = await anthropic_llm\
    #     .agent("You are an abstract artist.")\
    #     .request("Create an abstract painting representing freedom.")\
    #     .prompt_for_image()
    # assert isinstance(generated_img, Image)

    # stats3 = anthropic_llm.get_last_call_stats()
    # assert stats3 and "No usage information" not in stats3
