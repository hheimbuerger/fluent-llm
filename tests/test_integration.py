"""Integration tests for complete conversation flows."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fluent_llm import llm, MessageList, LLMConversation
from fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, Role
from fluent_llm.tools import Tool


class TestBuilderToConversationToContinuation:
    """Test the complete flow from builder to conversation to continuation."""

    @pytest.mark.asyncio
    async def test_basic_flow(self):
        """Test basic builder → conversation → continuation flow."""
        # Mock the provider
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            # Mock response
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [
                MagicMock(type="text", text="Hello! How can I help?")
            ]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response
            
            # Build
            builder = llm.agent("You are helpful").request("Hi")
            
            # Execute conversation
            conversation = builder.prompt_conversation()
            
            messages = []
            async for message in conversation:
                messages.append(message)
            
            # Verify we got a response
            assert len(messages) == 1
            assert isinstance(messages[0], TextMessage)
            assert messages[0].role == Role.ASSISTANT
            
            # Get continuation
            continuation = conversation.continuation
            
            # Verify continuation references same conversation
            assert continuation._conversation is conversation
            
            # Verify conversation has all messages
            assert len(conversation.messages) >= 2  # At least user + assistant

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with continuation."""
        from fluent_llm.builder import LLMPromptBuilder
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            # Mock responses for multiple turns
            mock_response1 = MagicMock()
            mock_response1.stop_reason = "end_turn"
            mock_response1.content = [MagicMock(type="text", text="First response")]
            mock_response1.usage = MagicMock(input_tokens=10, output_tokens=5)
            
            mock_response2 = MagicMock()
            mock_response2.stop_reason = "end_turn"
            mock_response2.content = [MagicMock(type="text", text="Second response")]
            mock_response2.usage = MagicMock(input_tokens=15, output_tokens=5)
            
            mock_client.messages.create.side_effect = [mock_response1, mock_response2]
            
            # First turn
            builder1 = LLMPromptBuilder().agent("You are helpful").request("First question")
            conversation1 = builder1.prompt_conversation()
            
            messages1 = []
            async for message in conversation1:
                messages1.append(message)
            
            assert len(messages1) == 1
            assert len(conversation1.messages) >= 2  # agent + question + response
            
            # Second turn using continuation
            # Note: continuation references the same conversation, so we need to reset the generator
            continuation = conversation1.continuation
            builder2 = continuation.request("Second question")
            
            # Apply deltas to add the new question
            conversation1.messages.extend(builder2._delta_messages)
            
            # Reset generator for next iteration
            conversation1._generator = None
            
            messages2 = []
            async for message in conversation1:
                messages2.append(message)
            
            assert len(messages2) == 1
            
            # Verify conversation has all messages from both turns
            assert len(conversation1.messages) >= 4  # agent + q1 + a1 + q2 + a2


class TestSerializationRestorationContinuation:
    """Test serialization → restoration → continuation flow."""

    @pytest.mark.asyncio
    async def test_serialize_and_restore(self):
        """Test serializing a conversation and restoring it."""
        # Create a conversation with messages
        conversation = LLMConversation()
        conversation.messages.append(AgentMessage("You are helpful"))
        conversation.messages.append(TextMessage("Hello", Role.USER))
        conversation.messages.append(TextMessage("Hi there!", Role.ASSISTANT))
        
        # Serialize
        data = conversation.messages.to_dict()
        
        # Restore to new conversation
        restored_messages = MessageList.from_dict(data)
        restored_conversation = LLMConversation(messages=restored_messages)
        
        # Verify restoration
        assert len(restored_conversation.messages) == 3
        assert isinstance(restored_conversation.messages[0], AgentMessage)
        assert restored_conversation.messages[0].text == "You are helpful"
        assert isinstance(restored_conversation.messages[1], TextMessage)
        assert restored_conversation.messages[1].role == Role.USER
        assert isinstance(restored_conversation.messages[2], TextMessage)
        assert restored_conversation.messages[2].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_continue_after_restoration(self):
        """Test continuing a conversation after restoration."""
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [MagicMock(type="text", text="Continued response")]
            mock_response.usage = MagicMock(input_tokens=20, output_tokens=5)
            mock_client.messages.create.return_value = mock_response
            
            # Create and serialize
            conversation = LLMConversation()
            conversation.messages.append(AgentMessage("You are helpful"))
            conversation.messages.append(TextMessage("Hello", Role.USER))
            conversation.messages.append(TextMessage("Hi!", Role.ASSISTANT))
            
            data = conversation.messages.to_dict()
            
            # Restore
            restored_messages = MessageList.from_dict(data)
            restored_conversation = LLMConversation(messages=restored_messages)
            
            # Continue
            continuation = restored_conversation.continuation
            builder = continuation.request("Follow up question")
            next_conversation = builder.prompt_conversation()
            
            messages = []
            async for message in next_conversation:
                messages.append(message)
            
            # Verify we got a response
            assert len(messages) == 1
            assert messages[0].text == "Continued response"
            
            # Verify all messages are preserved
            assert len(next_conversation.messages) >= 4

    @pytest.mark.asyncio
    async def test_serialize_with_tool_calls(self):
        """Test serializing conversations with tool calls."""
        conversation = LLMConversation()
        conversation.messages.append(AgentMessage("You are helpful"))
        conversation.messages.append(TextMessage("Get weather", Role.USER))
        conversation.messages.append(ToolCallMessage(
            message="Getting weather",
            tool_name="get_weather",
            tool_call_id="call_123",
            arguments={"location": "Paris"},
            result="Sunny, 22°C",
            error=None
        ))
        conversation.messages.append(TextMessage("The weather is sunny!", Role.ASSISTANT))
        
        # Serialize
        data = conversation.messages.to_dict()
        
        # Restore
        restored_messages = MessageList.from_dict(data)
        
        # Verify tool call is preserved
        assert len(restored_messages) == 4
        assert isinstance(restored_messages[2], ToolCallMessage)
        assert restored_messages[2].tool_name == "get_weather"
        assert restored_messages[2].result == "Sunny, 22°C"


class TestConfigurationManagement:
    """Test configuration management via deltas."""

    def test_config_deltas_accumulate(self):
        """Test that config deltas accumulate through builder chain."""
        builder = (llm
                   .model("gpt-4")
                   .provider("openai"))
        
        assert "preferred_model" in builder._delta_config
        assert "preferred_provider" in builder._delta_config
        assert builder._delta_config["preferred_model"] == "gpt-4"
        assert builder._delta_config["preferred_provider"] == "openai"

    def test_config_deltas_applied_to_conversation(self):
        """Test that config deltas are applied to conversation."""
        builder = llm.model("gpt-4").provider("openai")
        conversation = builder.prompt_conversation()
        
        assert conversation._config.preferred_model == "gpt-4"
        assert conversation._config.preferred_provider == "openai"

    def test_tool_config_accumulation(self):
        """Test that tool configurations accumulate."""
        def tool1(x: int) -> int:
            """Tool 1."""
            return x * 2
        
        def tool2(y: str) -> str:
            """Tool 2."""
            return y.upper()
        
        builder = llm.tool(tool1).tool(tool2)
        
        assert "tools" in builder._delta_config
        assert len(builder._delta_config["tools"]) == 2

    def test_tools_applied_to_conversation(self):
        """Test that tools are applied to conversation."""
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2
        
        builder = llm.tool(test_tool)
        conversation = builder.prompt_conversation()
        
        assert len(conversation._config.tools) == 1
        assert conversation._config.tools[0].name == "test_tool"


class TestCrossComponentInteractions:
    """Test interactions between different components."""

    def test_builder_creates_conversation_with_messages(self):
        """Test that builder creates conversation with correct messages."""
        from fluent_llm.builder import LLMPromptBuilder
        
        # Create fresh builder
        builder = (LLMPromptBuilder()
                   .agent("System message")
                   .request("User message")
                   .assistant("Assistant message"))
        
        conversation = builder.prompt_conversation()
        
        # Messages should be applied
        assert len(conversation.messages) == 3
        assert isinstance(conversation.messages[0], AgentMessage)
        assert isinstance(conversation.messages[1], TextMessage)
        assert conversation.messages[1].role == Role.USER
        assert isinstance(conversation.messages[2], TextMessage)
        assert conversation.messages[2].role == Role.ASSISTANT

    def test_continuation_preserves_conversation_state(self):
        """Test that continuation preserves conversation state."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("Test", Role.USER))
        conversation._config.preferred_model = "gpt-4"
        
        continuation = conversation.continuation
        
        # Should reference same conversation
        assert continuation._conversation is conversation
        assert len(continuation._conversation.messages) == 1
        assert continuation._conversation._config.preferred_model == "gpt-4"

    def test_multiple_continuations_share_conversation(self):
        """Test that multiple continuations share the same conversation."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("Test", Role.USER))
        
        continuation1 = conversation.continuation
        continuation2 = conversation.continuation
        
        # Both should reference the same conversation
        assert continuation1._conversation is conversation
        assert continuation2._conversation is conversation
        assert continuation1._conversation is continuation2._conversation

    @pytest.mark.asyncio
    async def test_assistant_injection_in_flow(self):
        """Test assistant message injection in conversation flow."""
        from fluent_llm.builder import LLMPromptBuilder
        
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [MagicMock(type="text", text="Final response")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response
            
            # Build with assistant injection
            builder = (LLMPromptBuilder()
                       .agent("You are helpful")
                       .request("First question")
                       .assistant("I understand")
                       .request("Second question"))
            
            conversation = builder.prompt_conversation()
            
            # Verify messages before execution
            assert len(conversation.messages) == 4
            assert conversation.messages[2].role == Role.ASSISTANT
            assert conversation.messages[2].text == "I understand"
            
            # Execute
            messages = []
            async for message in conversation:
                messages.append(message)
            
            # Verify execution completed
            assert len(messages) == 1

    def test_empty_builder_creates_empty_conversation(self):
        """Test that empty builder creates conversation with no messages."""
        from fluent_llm.builder import LLMPromptBuilder
        
        builder = LLMPromptBuilder()
        conversation = builder.prompt_conversation()
        
        assert len(conversation.messages) == 0

    def test_config_only_builder(self):
        """Test builder with only config changes."""
        from fluent_llm.builder import LLMPromptBuilder
        
        builder = LLMPromptBuilder().model("gpt-4").provider("openai")
        conversation = builder.prompt_conversation()
        
        assert len(conversation.messages) == 0
        assert conversation._config.preferred_model == "gpt-4"
        assert conversation._config.preferred_provider == "openai"


class TestEarlyAccess:
    """Test early access to continuation during iteration."""

    @pytest.mark.asyncio
    async def test_continuation_available_before_completion(self):
        """Test that continuation is available before iteration completes."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("Test", Role.USER))
        
        # Continuation should be available immediately
        continuation = conversation.continuation
        
        assert continuation is not None
        assert continuation._conversation is conversation

    def test_continuation_reflects_current_state(self):
        """Test that continuation reflects current conversation state."""
        conversation = LLMConversation()
        
        # Get continuation when empty
        continuation1 = conversation.continuation
        assert len(continuation1._conversation.messages) == 0
        
        # Add message
        conversation.messages.append(TextMessage("Test", Role.USER))
        
        # Get new continuation
        continuation2 = conversation.continuation
        assert len(continuation2._conversation.messages) == 1
        
        # Both should reference same conversation
        assert continuation1._conversation is continuation2._conversation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
