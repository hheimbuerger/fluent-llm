"""Tests for async generator conversation functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fluent_llm import llm
from fluent_llm.messages import TextMessage, ToolCallMessage, MessageList, Role
from fluent_llm.builder import LLMPromptBuilder


class TestAsyncGeneratorConversations:
    """Test suite for async generator conversation functionality."""

    @pytest.mark.asyncio
    async def test_manual_async_generator_iteration(self):
        """Test manual async generator iteration using __anext__()."""
        
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        # Mock the provider response
        mock_response = MagicMock()
        mock_response.text = "I'll check the weather for you."
        mock_response.tool_calls = [
            MagicMock(
                id="tool_1",
                name="get_weather",
                arguments={"location": "Paris"}
            )
        ]
        
        with patch('fluent_llm.builder.DefaultModelSelectionStrategy') as mock_strategy:
            mock_provider = AsyncMock()
            mock_provider.supports_tools.return_value = True
            mock_provider.check_capabilities = MagicMock()
            mock_provider.prompt_via_api.return_value = mock_response
            
            mock_strategy.return_value.select_model.return_value = (mock_provider, "test-model")
            
            conversation = llm \
                .agent("You are a helpful assistant") \
                .tool(get_weather) \
                .request("What's the weather in Paris?") \
                .prompt_conversation()
            
            # Manual iteration
            messages = []
            try:
                while True:
                    message = await conversation.__anext__()
                    messages.append(message)
            except StopAsyncIteration:
                continuation_builder = conversation.llm_continuation
            
            # Verify message types and content
            assert len(messages) >= 1
            assert any(isinstance(msg, ToolCallMessage) for msg in messages)
            assert isinstance(continuation_builder, LLMPromptBuilder)
            
            # Check ToolCallMessage structure
            tool_message = next(msg for msg in messages if isinstance(msg, ToolCallMessage))
            assert tool_message.tool_name == "get_weather"
            assert tool_message.arguments == {"location": "Paris"}
            assert tool_message.result == "Weather in Paris: Sunny, 72°F"
            assert tool_message.error is None

    @pytest.mark.asyncio
    async def test_async_for_loop_iteration(self):
        """Test async for loop iteration pattern."""
        
        def calculate(x: int, y: int) -> int:
            return x + y
        
        # Mock the provider response
        mock_response = MagicMock()
        mock_response.text = "I'll calculate that for you."
        mock_response.tool_calls = [
            MagicMock(
                id="tool_1",
                name="calculate",
                arguments={"x": 5, "y": 3}
            )
        ]
        
        with patch('fluent_llm.builder.DefaultModelSelectionStrategy') as mock_strategy:
            mock_provider = AsyncMock()
            mock_provider.supports_tools.return_value = True
            mock_provider.check_capabilities = MagicMock()
            mock_provider.prompt_via_api.return_value = mock_response
            
            mock_strategy.return_value.select_model.return_value = (mock_provider, "test-model")
            
            conversation = llm \
                .agent("You are a calculator") \
                .tool(calculate) \
                .request("What is 5 + 3?") \
                .prompt_conversation()
            
            messages = []
            async for message in conversation:
                messages.append(message)
                
            # Verify expected message flow
            assert len(messages) >= 1
            assert any(isinstance(msg, ToolCallMessage) for msg in messages)

    @pytest.mark.asyncio
    async def test_tool_call_message_structure_with_error(self):
        """Test that ToolCallMessage contains complete information including errors."""
        
        def failing_function() -> str:
            raise ValueError("This function always fails")
        
        # Mock the provider response
        mock_response = MagicMock()
        mock_response.text = "I'll try to execute that function."
        mock_response.tool_calls = [
            MagicMock(
                id="tool_1",
                name="failing_function",
                arguments={}
            )
        ]
        
        with patch('fluent_llm.builder.DefaultModelSelectionStrategy') as mock_strategy:
            mock_provider = AsyncMock()
            mock_provider.supports_tools.return_value = True
            mock_provider.check_capabilities = MagicMock()
            mock_provider.prompt_via_api.return_value = mock_response
            
            mock_strategy.return_value.select_model.return_value = (mock_provider, "test-model")
            
            conversation = llm \
                .agent("You are a test assistant") \
                .tool(failing_function) \
                .request("Execute the failing function") \
                .prompt_conversation()
            
            tool_message = None
            async for message in conversation:
                if isinstance(message, ToolCallMessage):
                    tool_message = message
                    break
            
            assert tool_message is not None
            assert tool_message.tool_name == "failing_function"
            assert tool_message.arguments == {}
            assert tool_message.result is None
            assert isinstance(tool_message.error, ValueError)
            assert "This function always fails" in str(tool_message.error)
            assert hasattr(tool_message, 'message')  # Assistant text

    @pytest.mark.asyncio
    async def test_prompt_agentically_compatibility(self):
        """Test that prompt_agentically still works as expected."""
        
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        # Mock the provider response
        mock_response = MagicMock()
        mock_response.text = "The weather is nice today."
        mock_response.tool_calls = [
            MagicMock(
                id="tool_1",
                name="get_weather",
                arguments={"location": "Paris"}
            )
        ]
        
        with patch('fluent_llm.builder.DefaultModelSelectionStrategy') as mock_strategy:
            mock_provider = AsyncMock()
            mock_provider.supports_tools.return_value = True
            mock_provider.check_capabilities = MagicMock()
            mock_provider.prompt_via_api.return_value = mock_response
            
            mock_strategy.return_value.select_model.return_value = (mock_provider, "test-model")
            
            messages, continuation = await llm \
                .agent("You are a helpful assistant") \
                .tool(get_weather) \
                .request("What's the weather in Paris?") \
                .prompt_agentically(max_calls=5)
            
            # Should return complete conversation
            assert isinstance(messages, MessageList)
            assert len(messages) >= 1
            assert any(isinstance(msg, ToolCallMessage) for msg in messages)
            assert isinstance(continuation, LLMPromptBuilder)

    @pytest.mark.asyncio
    async def test_stopasynciteration_handling_and_continuation_builder(self):
        """Test StopAsyncIteration handling and continuation builder access."""
        
        def simple_tool() -> str:
            return "Tool executed successfully"
        
        # Mock the provider response
        mock_response = MagicMock()
        mock_response.text = "Task completed."
        mock_response.tool_calls = [
            MagicMock(
                id="tool_1",
                name="simple_tool",
                arguments={}
            )
        ]
        
        with patch('fluent_llm.builder.DefaultModelSelectionStrategy') as mock_strategy:
            mock_provider = AsyncMock()
            mock_provider.supports_tools.return_value = True
            mock_provider.check_capabilities = MagicMock()
            mock_provider.prompt_via_api.return_value = mock_response
            
            mock_strategy.return_value.select_model.return_value = (mock_provider, "test-model")
            
            conversation = llm \
                .agent("You are a test assistant") \
                .tool(simple_tool) \
                .request("Execute the simple tool") \
                .prompt_conversation()
            
            messages = []
            continuation_builder = None
            
            async for message in conversation:
                messages.append(message)
            
            # After async for completes, continuation builder should be available
            continuation_builder = conversation.llm_continuation
            
            # Verify continuation builder is available
            assert continuation_builder is not None
            assert isinstance(continuation_builder, LLMPromptBuilder)
            assert len(messages) >= 1

    def test_prompt_conversation_no_message_parameter_error(self):
        """Test that prompt_conversation doesn't accept message parameter."""
        
        # This should work - no message parameter
        conversation_builder = llm.agent("test").prompt_conversation
        
        # This should fail - trying to pass message parameter would be a TypeError
        # since the method signature doesn't accept a message parameter
        import inspect
        sig = inspect.signature(conversation_builder)
        assert 'message' not in sig.parameters

    @pytest.mark.asyncio
    async def test_tool_call_message_str_representation(self):
        """Test the __str__ method of ToolCallMessage for both success and error cases."""
        
        # Test successful tool call
        success_msg = ToolCallMessage(
            message="I'll calculate that",
            tool_name="add",
            tool_call_id="tool_1",
            arguments={"x": 5, "y": 3},
            result=8,
            error=None
        )
        
        expected_success = "Tool call: add({'x': 5, 'y': 3}) -> 8"
        assert str(success_msg) == expected_success
        
        # Test failed tool call
        error_msg = ToolCallMessage(
            message="I'll try to execute this",
            tool_name="divide",
            tool_call_id="tool_2",
            arguments={"x": 5, "y": 0},
            result=None,
            error=ZeroDivisionError("division by zero")
        )
        
        expected_error = "Tool call: divide({'x': 5, 'y': 0}) -> ERROR: division by zero"
        assert str(error_msg) == expected_error