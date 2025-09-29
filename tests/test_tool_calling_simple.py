"""
Simple integration test for tool calling functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fluent_llm import llm


def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 22°C"


@pytest.mark.asyncio
async def test_simple_tool_calling():
    """Test basic tool calling with mocked Anthropic API."""
    # Mock the Anthropic API response for tool calling
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [
        MagicMock(
            type="tool_use",
            id="tool_call_1",
            name="get_weather",
            input={"location": "Paris"}
        )
    ]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    
    # Mock the continuation response
    mock_final_response = MagicMock()
    mock_final_response.stop_reason = "end_turn"
    mock_final_response.content = [
        MagicMock(type="text", text="The weather in Paris is sunny and 22°C.")
    ]
    mock_final_response.usage = MagicMock(input_tokens=150, output_tokens=25)
    
    with patch('anthropic.AsyncAnthropic') as mock_anthropic:
        mock_client = AsyncMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [mock_response, mock_final_response]
        
        messages, continuation = await (
            llm
            .agent("You are a helpful assistant")
            .tools([get_weather])
            .prompt_conversation("What's the weather in Paris?")
        )
        
        # Basic verification
        assert len(messages) >= 2
        assert mock_client.messages.create.call_count == 2
        
        # Verify tools were passed to the API
        first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_kwargs
        assert len(first_call_kwargs["tools"]) == 1
        assert first_call_kwargs["tools"][0]["name"] == "get_weather"


@pytest.mark.asyncio 
async def test_tool_validation_error():
    """Test that tools are validated when added."""
    def bad_function(param):  # No type annotation
        return "result"
    
    with pytest.raises(ValueError, match="lacks type annotation"):
        llm.tools([bad_function])


@pytest.mark.asyncio
async def test_unsupported_provider():
    """Test error when using tools with unsupported provider."""
    with pytest.raises(ValueError, match="does not support tool calling"):
        await (
            llm
            .provider("openai")
            .tools([get_weather])
            .prompt_conversation("What's the weather?")
        )