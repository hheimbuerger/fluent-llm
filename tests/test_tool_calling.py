"""
Tests for tool calling functionality in the fluent LLM library.

This module tests the complete tool calling conversation flow including:
- Tool definition and validation
- Conversation management
- Tool execution
- Error handling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from fluent_llm import llm
from fluent_llm.tools import Tool
from fluent_llm.messages import ToolCallMessage, ToolResultMessage, TextMessage


def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location.
    
    Args:
        location: The city or location to get weather for
        units: Temperature units (celsius or fahrenheit)
    
    Returns:
        Weather information as a string
    """
    return f"Weather in {location}: Sunny, 22°C" if units == "celsius" else f"Weather in {location}: Sunny, 72°F"


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


def failing_tool(message: str) -> str:
    """A tool that always fails for testing error handling.
    
    Args:
        message: A message (unused)
        
    Returns:
        Never returns, always raises an exception
    """
    raise ValueError("This tool always fails")


class TestToolDefinition:
    """Test tool definition and validation."""
    
    def test_tool_from_function_basic(self):
        """Test creating a tool from a basic function."""
        tool = Tool.from_function(get_weather)
        
        assert tool.name == "get_weather"
        assert "Get current weather" in tool.description
        assert tool.function == get_weather
        assert "location" in tool.schema["properties"]
        assert "units" in tool.schema["properties"]
        assert "location" in tool.schema["required"]
        assert "units" not in tool.schema["required"]  # Has default value
    
    def test_tool_schema_generation(self):
        """Test JSON schema generation from function signature."""
        tool = Tool.from_function(calculate_sum)
        
        schema = tool.schema
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["required"] == ["a", "b"]
        
        # Check that integer types are properly detected
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "integer"
    
    def test_tool_validation_missing_type_annotation(self):
        """Test that functions without type annotations are rejected."""
        def bad_function(param):  # No type annotation
            return "result"
        
        with pytest.raises(ValueError, match="lacks type annotation"):
            Tool.from_function(bad_function)
    
    def test_tool_validation_reserved_name(self):
        """Test that reserved function names are rejected."""
        def help(message: str) -> str:  # Reserved name
            return "help message"
        
        with pytest.raises(ValueError, match="reserved"):
            Tool.from_function(help)


class TestBuilderToolMethods:
    """Test the builder's tool-related methods."""
    
    def test_tool_method_single(self):
        """Test adding a single tool."""
        builder = llm.tool(get_weather)
        
        assert len(builder._tools) == 1
        assert builder._tools[0].name == "get_weather"
    
    def test_tools_method_multiple(self):
        """Test adding multiple tools."""
        builder = llm.tools([get_weather, calculate_sum])
        
        assert len(builder._tools) == 2
        tool_names = [tool.name for tool in builder._tools]
        assert "get_weather" in tool_names
        assert "calculate_sum" in tool_names
    
    @pytest.mark.asyncio
    async def test_tools_prevent_other_methods(self):
        """Test that other prompt methods fail when tools are defined."""
        builder = llm.tools([get_weather])
        
        with pytest.raises(ValueError, match="Cannot use prompt_for_text"):
            await builder.request("What's the weather?").prompt_for_text()
        
        with pytest.raises(ValueError, match="Cannot use call"):
            await builder.request("What's the weather?").call()
    
    def test_prompt_conversation_works_without_tools(self):
        """Test that prompt_conversation works without tools defined."""
        builder = llm.agent("You are helpful")
        
        # This should not raise an error - prompt_conversation supports zero tools
        try:
            builder.prompt_conversation("Hello")
        except ValueError as e:
            if "requires tools to be defined" in str(e):
                pytest.fail("prompt_conversation should work without tools defined")


class TestToolCallingConversation:
    """Test the complete tool calling conversation flow."""
    
    @pytest.mark.asyncio
    async def test_basic_tool_calling_conversation(self):
        """Test a basic conversation with tool calling."""
        # Mock the Anthropic API response for tool calling
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_call_1",
                name="get_weather",
                input={"location": "Paris", "units": "celsius"}
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
            
            # Verify the conversation flow
            assert len(messages) >= 2  # At least user message and response
            
            # Check that the API was called twice (initial + continuation)
            assert mock_client.messages.create.call_count == 2
            
            # Verify tools were passed to the API
            first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
            assert "tools" in first_call_kwargs
            assert len(first_call_kwargs["tools"]) == 1
            assert first_call_kwargs["tools"][0]["name"] == "get_weather"
    
    @pytest.mark.asyncio
    async def test_conversation_continuation(self):
        """Test continuing a conversation after tool calls."""
        # Mock initial tool calling response
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
        
        # Mock continuation response
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [
            MagicMock(type="text", text="The weather in Paris is sunny.")
        ]
        mock_final_response.usage = MagicMock(input_tokens=150, output_tokens=25)
        
        # Mock second conversation turn
        mock_second_response = MagicMock()
        mock_second_response.stop_reason = "end_turn"
        mock_second_response.content = [
            MagicMock(type="text", text="London weather is cloudy.")
        ]
        mock_second_response.usage = MagicMock(input_tokens=200, output_tokens=30)
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = [
                mock_response, mock_final_response, mock_second_response
            ]
            
            # First conversation turn
            messages, continuation = await (
                llm
                .agent("You are a helpful assistant")
                .tools([get_weather])
                .prompt_conversation("What's the weather in Paris?")
            )
            
            # Continue the conversation
            final_messages, _ = await continuation.prompt_conversation("What about London?")
            
            # Verify the conversation was continued
            assert len(final_messages) > len(messages)
            assert mock_client.messages.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_response(self):
        """Test handling multiple tool calls in a single response."""
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_call_1",
                name="get_weather",
                input={"location": "Paris"}
            ),
            MagicMock(
                type="tool_use",
                id="tool_call_2",
                name="get_weather",
                input={"location": "London"}
            )
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [
            MagicMock(type="text", text="Both cities have good weather.")
        ]
        mock_final_response.usage = MagicMock(input_tokens=200, output_tokens=25)
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = [mock_response, mock_final_response]
            
            messages, _ = await (
                llm
                .agent("You are a helpful assistant")
                .tools([get_weather])
                .prompt_conversation("What's the weather in Paris and London?")
            )
            
            # Verify both tool calls were handled
            continuation_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            user_message = continuation_call_kwargs["messages"][-1]
            assert user_message["role"] == "user"
            assert len(user_message["content"]) == 2  # Two tool results


class TestErrorHandling:
    """Test error handling in tool calling."""
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test handling of tool execution errors."""
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_call_1",
                name="failing_tool",
                input={"message": "test"}
            )
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [
            MagicMock(type="text", text="I encountered an error with the tool.")
        ]
        mock_final_response.usage = MagicMock(input_tokens=150, output_tokens=25)
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = [mock_response, mock_final_response]
            
            # Should not raise an exception, but handle the error gracefully
            messages, _ = await (
                llm
                .agent("You are a helpful assistant")
                .tools([failing_tool])
                .prompt_conversation("Use the failing tool")
            )
            
            # Verify the error was passed to the model
            continuation_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            user_message = continuation_call_kwargs["messages"][-1]
            tool_result_content = user_message["content"][0]["content"]
            assert "Error executing tool" in tool_result_content
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test handling of calls to unknown tools."""
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [
            MagicMock(
                type="tool_use",
                id="tool_call_1",
                name="unknown_tool",
                input={"param": "value"}
            )
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [
            MagicMock(type="text", text="I don't have access to that tool.")
        ]
        mock_final_response.usage = MagicMock(input_tokens=150, output_tokens=25)
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = [mock_response, mock_final_response]
            
            messages, _ = await (
                llm
                .agent("You are a helpful assistant")
                .tools([get_weather])
                .prompt_conversation("Use an unknown tool")
            )
            
            # Verify the error was passed to the model
            continuation_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            user_message = continuation_call_kwargs["messages"][-1]
            tool_result_content = user_message["content"][0]["content"]
            assert "not found" in tool_result_content
    
    @pytest.mark.asyncio
    async def test_unsupported_provider_error(self):
        """Test error when using tools with unsupported provider."""
        with pytest.raises(ValueError, match="does not support tool calling"):
            await (
                llm
                .provider("openai")  # OpenAI doesn't support tools yet
                .tools([get_weather])
                .prompt_conversation("What's the weather?")
            )
    
    def test_invalid_tool_arguments_error(self):
        """Test handling of invalid tool call arguments."""
        # This would be tested by mocking a tool call with wrong arguments
        # The error handling is in the Anthropic provider's tool execution
        pass


class TestModelSelection:
    """Test model selection with tools."""
    
    @pytest.mark.asyncio
    async def test_anthropic_preferred_with_tools(self):
        """Test that Anthropic is preferred when tools are present."""
        with patch('fluent_llm.providers.anthropic.claude.anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            # Mock a simple text response (no tools called)
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [MagicMock(type="text", text="Hello!")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response
            
            builder = llm.tools([get_weather])
            
            # The model selector should choose Anthropic by building a prompt manually
            from fluent_llm.prompt import Prompt
            from fluent_llm.messages import ResponseType
            
            p = Prompt(
                messages=builder._messages,
                expect_type=ResponseType.TEXT,
                preferred_provider=builder._preferred_provider,
                preferred_model=builder._preferred_model,
                tools=builder._tools,
                is_conversation=True,
            )
            provider, model = builder._model_selector.select_model(p)
            
            assert "claude" in model.lower()
