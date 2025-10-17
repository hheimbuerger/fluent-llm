import pytest
from fluent_llm import llm
from fluent_llm.messages import ToolCallMessage, TextMessage


@pytest.fixture(scope="module", autouse=True)
def anthropic_llm():
    """"""
    return llm.provider('anthropic')


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


@pytest.mark.asyncio
async def test_tool_calling_conversation_live():
    """Test complete tool calling conversation flow with real Anthropic API."""

    def get_weather(location: str) -> str:
        """Get current weather for a location.
        
        Args:
            location: The city or location to get weather for
            
        Returns:
            Weather information as a string
        """
        return f"Weather in {location}: Sunny, {20+len(location)}°C"

    print("Starting tool calling conversation test...")
    
    # New API: prompt_conversation() returns LLMConversation
    conversation = (
        llm
        .model('claude-sonnet-4-5-20250929')
        .agent("You are a helpful assistant. When asked about weather, use the get_weather tool. Always explain what you're doing before calling tools.")
        .tool(get_weather)
        .request("What's the weather in Paris?")
        .prompt_conversation()
    )
    
    # Collect messages by iterating
    messages = []
    async for message in conversation:
        messages.append(message)
        print(f"Message {len(messages)}: {type(message).__name__} - {str(message)[:100]}...")
    
    print(f"Received {len(messages)} messages total")
    
    # Should contain AI responses
    assert len(messages) >= 1
    
    # Verify roles are correct
    from fluent_llm.messages import Role
    assistant_messages = [msg for msg in messages if hasattr(msg, 'role') and msg.role == Role.ASSISTANT]
    
    assert len(assistant_messages) >= 1, "Should have at least one assistant message"
    
    # Check if tool was actually called by looking for weather data in the response
    final_response = str(messages[-1])
    assert "Paris" in final_response
    
    # Verify that the weather function was called with the expected temperature calculation
    # Our function returns temperature = 20 + len(location), so Paris (5 chars) should be 25°C
    assert "25" in final_response or "25°C" in final_response
    print(f"Final response: {final_response}")
    
    # Test conversation continuation
    print("Testing conversation continuation...")
    continuation = conversation.continuation
    next_conversation = continuation.request("What about London?").prompt_conversation()
    
    final_messages = []
    async for message in next_conversation:
        final_messages.append(message)
    
    print(f"Continuation has {len(final_messages)} new messages")
    final_response_msg = final_messages[-1]
    final_response = str(final_response_msg)
    print(f"Final response: {final_response}")
    
    assert "London" in final_response or "london" in final_response.lower()
    
    # Verify that the weather function was called for London (6 chars) should be 26°C
    assert "26" in final_response or "26°C" in final_response
