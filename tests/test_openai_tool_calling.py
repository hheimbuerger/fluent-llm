"""Unit tests for OpenAI provider tool calling support."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.fluent_llm.providers.openai.gpt import OpenAIProvider
from src.fluent_llm.tools import Tool
from src.fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, Role
from src.fluent_llm.conversation import MessageList
from src.fluent_llm.prompt import Prompt, ResponseType


# Sample tool functions for testing
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"Weather in {location}: Sunny, 22°C"


def calculate(operation: str, a: float, b: float) -> float:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0.0


class TestOpenAIToolSupport:
    """Test OpenAI provider tool calling capabilities."""
    
    def test_supports_tools(self):
        """Test that OpenAI provider reports tool support."""
        provider = OpenAIProvider()
        assert provider.supports_tools() is True
    
    def test_convert_tools_to_api_format(self):
        """Test tool definition conversion to OpenAI format."""
        provider = OpenAIProvider()
        
        # Create tools
        weather_tool = Tool.from_function(get_weather)
        calc_tool = Tool.from_function(calculate)
        
        # Convert to API format
        api_tools = provider._convert_tools_to_api_format([weather_tool, calc_tool])
        
        # Verify structure
        assert len(api_tools) == 2
        
        # Check weather tool
        assert api_tools[0]["type"] == "function"
        assert api_tools[0]["function"]["name"] == "get_weather"
        assert api_tools[0]["function"]["description"] == "Get the weather for a location."
        assert "parameters" in api_tools[0]["function"]
        assert api_tools[0]["function"]["parameters"]["type"] == "object"
        assert "location" in api_tools[0]["function"]["parameters"]["properties"]
        
        # Check calculate tool
        assert api_tools[1]["type"] == "function"
        assert api_tools[1]["function"]["name"] == "calculate"
        assert "operation" in api_tools[1]["function"]["parameters"]["properties"]
        assert "a" in api_tools[1]["function"]["parameters"]["properties"]
        assert "b" in api_tools[1]["function"]["parameters"]["properties"]
    
    def test_convert_tool_call_message_to_openai_format(self):
        """Test ToolCallMessage conversion to OpenAI format."""
        provider = OpenAIProvider()
        
        # Create a ToolCallMessage
        tool_msg = ToolCallMessage(
            message="Let me check the weather",
            tool_name="get_weather",
            tool_call_id="call_123",
            arguments={"location": "Paris"},
            result="Weather in Paris: Sunny, 22°C",
            error=None
        )
        
        messages = MessageList([tool_msg])
        openai_messages = provider._convert_messages_to_openai_format(messages)
        
        # Should produce two messages: assistant with tool_calls, and tool with result
        assert len(openai_messages) == 2
        
        # Check assistant message
        assert openai_messages[0]["role"] == "assistant"
        assert openai_messages[0]["content"] == "Let me check the weather"
        assert "tool_calls" in openai_messages[0]
        assert len(openai_messages[0]["tool_calls"]) == 1
        
        tool_call = openai_messages[0]["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": "Paris"}
        
        # Check tool result message
        assert openai_messages[1]["role"] == "tool"
        assert openai_messages[1]["tool_call_id"] == "call_123"
        assert openai_messages[1]["content"] == "Weather in Paris: Sunny, 22°C"
    
    def test_convert_tool_call_message_with_error(self):
        """Test ToolCallMessage with error conversion."""
        provider = OpenAIProvider()
        
        # Create a ToolCallMessage with error
        tool_msg = ToolCallMessage(
            message="",
            tool_name="get_weather",
            tool_call_id="call_456",
            arguments={"location": "InvalidPlace"},
            result=None,
            error=ValueError("Location not found")
        )
        
        messages = MessageList([tool_msg])
        openai_messages = provider._convert_messages_to_openai_format(messages)
        
        # Check tool result contains error
        assert openai_messages[1]["role"] == "tool"
        assert "Location not found" in openai_messages[1]["content"]
    
    def test_convert_mixed_messages_to_openai_format(self):
        """Test conversion of mixed message types including tool calls."""
        provider = OpenAIProvider()
        
        messages = MessageList([
            AgentMessage("You are a helpful assistant"),
            TextMessage("What's the weather in London?", Role.USER),
            ToolCallMessage(
                message="Let me check",
                tool_name="get_weather",
                tool_call_id="call_789",
                arguments={"location": "London"},
                result="Weather in London: Cloudy, 15°C",
                error=None
            ),
            TextMessage("The weather in London is cloudy.", Role.ASSISTANT)
        ])
        
        openai_messages = provider._convert_messages_to_openai_format(messages)
        
        # Should have: system, user, assistant (tool call), tool (result), assistant (text)
        assert len(openai_messages) == 5
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[2]["role"] == "assistant"
        assert "tool_calls" in openai_messages[2]
        assert openai_messages[3]["role"] == "tool"
        assert openai_messages[4]["role"] == "assistant"
    
    def test_handle_tool_calls_response_chat_completion(self):
        """Test handling tool calls from Chat Completion API response."""
        provider = OpenAIProvider()
        
        # Mock Chat Completion response with tool calls
        import openai.types.chat.chat_completion
        mock_response = MagicMock(spec=openai.types.chat.chat_completion.ChatCompletion)
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Let me check that for you"
        
        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "Tokyo"})
        
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Create a minimal prompt
        messages = MessageList([TextMessage("What's the weather?", Role.USER)])
        prompt = Prompt(
            messages=messages,
            expect_type=ResponseType.TEXT,
            tools=[Tool.from_function(get_weather)]
        )
        
        # Handle the response
        result = provider._handle_tool_calls_response(mock_response, prompt)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["text"] == "Let me check that for you"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        
        tool_call = result["tool_calls"][0]
        assert tool_call["id"] == "call_abc123"
        assert tool_call["name"] == "get_weather"
        assert tool_call["arguments"] == {"location": "Tokyo"}
    
    def test_handle_multiple_tool_calls(self):
        """Test handling multiple tool calls in one response."""
        provider = OpenAIProvider()
        
        # Mock response with multiple tool calls
        import openai.types.chat.chat_completion
        mock_response = MagicMock(spec=openai.types.chat.chat_completion.ChatCompletion)
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = ""
        
        # Mock multiple tool calls
        mock_tool_call_1 = MagicMock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function = MagicMock()
        mock_tool_call_1.function.name = "get_weather"
        mock_tool_call_1.function.arguments = json.dumps({"location": "Paris"})
        
        mock_tool_call_2 = MagicMock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function = MagicMock()
        mock_tool_call_2.function.name = "calculate"
        mock_tool_call_2.function.arguments = json.dumps({"operation": "add", "a": 5, "b": 3})
        
        mock_response.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        
        messages = MessageList([TextMessage("Test", Role.USER)])
        prompt = Prompt(
            messages=messages,
            expect_type=ResponseType.TEXT,
            tools=[Tool.from_function(get_weather), Tool.from_function(calculate)]
        )
        
        result = provider._handle_tool_calls_response(mock_response, prompt)
        
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][1]["name"] == "calculate"
    
    def test_handle_tool_calls_response_no_tool_calls(self):
        """Test error handling when no tool calls are present."""
        provider = OpenAIProvider()
        
        # Mock response without tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Some text"
        mock_response.choices[0].message.tool_calls = None
        
        messages = MessageList([TextMessage("Test", Role.USER)])
        prompt = Prompt(messages=messages, expect_type=ResponseType.TEXT)
        
        with pytest.raises(RuntimeError, match="Expected tool calls in response"):
            provider._handle_tool_calls_response(mock_response, prompt)
    



class TestOpenAIToolCallingIntegration:
    """Integration tests for OpenAI tool calling with prompt_via_api."""
    
    @pytest.mark.asyncio
    async def test_prompt_via_api_with_tools_chat_completion(self):
        """Test that tools are properly added to Responses API calls."""
        provider = OpenAIProvider()
        
        # Create prompt with tools
        messages = MessageList([
            AgentMessage("You are helpful"),
            TextMessage("What's the weather?", Role.USER)
        ])
        
        tools = [Tool.from_function(get_weather)]
        
        prompt = Prompt(
            messages=messages,
            expect_type=ResponseType.TEXT,
            tools=tools
        )
        
        # Mock the OpenAI client
        with patch('openai.AsyncOpenAI') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock response - use Response type for Responses API
            import openai.types.responses.response
            mock_response = MagicMock(spec=openai.types.responses.response.Response)
            mock_response.status = "completed"
            mock_response.output_text = "It's sunny!"
            
            # Mock usage with proper values
            mock_usage = MagicMock()
            mock_usage.input_tokens = 10
            mock_usage.output_tokens = 5
            mock_response.usage = mock_usage
            mock_response.model = "gpt-4o-mini"
            
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            
            # Call the API
            result = await provider.prompt_via_api(
                model="gpt-4o-mini",
                p=prompt
            )
            
            # Verify tools were passed
            call_args = mock_client.responses.create.call_args
            assert call_args is not None
            assert "tools" in call_args[1]
            assert len(call_args[1]["tools"]) == 1
            assert call_args[1]["tools"][0]["type"] == "function"
            assert call_args[1]["tools"][0]["function"]["name"] == "get_weather"
    
    @pytest.mark.asyncio
    async def test_prompt_via_api_handles_tool_calls_finish_reason(self):
        """Test that finish_reason='tool_calls' is handled correctly."""
        provider = OpenAIProvider()
        
        messages = MessageList([TextMessage("What's the weather?", Role.USER)])
        tools = [Tool.from_function(get_weather)]
        
        prompt = Prompt(
            messages=messages,
            expect_type=ResponseType.TEXT,
            tools=tools
        )
        
        with patch('openai.AsyncOpenAI') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock response with tool_calls finish_reason
            import openai.types.chat.chat_completion
            mock_response = MagicMock(spec=openai.types.chat.chat_completion.ChatCompletion)
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = ""
            
            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function = MagicMock()
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = json.dumps({"location": "Paris"})
            
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            
            # Mock usage with proper values
            mock_usage = MagicMock()
            mock_usage.input_tokens = 10
            mock_usage.output_tokens = 5
            mock_response.usage = mock_usage
            mock_response.model = "gpt-4o-mini"
            
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            
            # Call the API
            result = await provider.prompt_via_api(
                model="gpt-4o-mini",
                p=prompt
            )
            
            # Verify result is a dict with tool_calls
            assert isinstance(result, dict)
            assert "tool_calls" in result
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "get_weather"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
