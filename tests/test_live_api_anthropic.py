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
