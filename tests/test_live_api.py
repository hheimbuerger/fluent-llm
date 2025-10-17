"""Live API integration tests for the new architecture.

These tests make real API calls and require valid API keys.
They are skipped if the required environment variables are not set.
"""
import pytest
import os
from fluent_llm import llm, MessageList, LLMConversation
from fluent_llm.builder import LLMPromptBuilder
from fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, Role


# Skip all tests in this module if API keys are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping live API tests"
)


class TestRealConversationFlows:
    """Test real conversation flows with API calls."""

    @pytest.mark.asyncio
    async def test_simple_conversation(self):
        """Test a simple conversation with real API."""
        builder = LLMPromptBuilder().agent("You are a helpful assistant. Be concise.").request("Say hello in one word.")
        conversation = builder.prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
        
        # Verify we got a response
        assert len(messages) >= 1
        assert isinstance(messages[-1], TextMessage)
        assert messages[-1].role == Role.ASSISTANT
        assert len(messages[-1].text) > 0
        
        # Verify conversation has all messages
        assert len(conversation.messages) >= 2  # At least user + assistant

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with real API."""
        # First turn
        builder1 = LLMPromptBuilder().agent("You are a helpful assistant. Be very concise.").request("What is 2+2?")
        conversation1 = builder1.prompt_conversation()
        
        messages1 = []
        async for message in conversation1:
            messages1.append(message)
        
        assert len(messages1) >= 1
        first_response = messages1[-1].text
        assert "4" in first_response or "four" in first_response.lower()
        
        # Second turn using continuation
        continuation = conversation1.continuation
        builder2 = continuation.request("What is that number plus 3?")
        
        # Apply deltas and reset generator
        conversation1.messages.extend(builder2._delta_messages)
        conversation1._generator = None
        
        messages2 = []
        async for message in conversation1:
            messages2.append(message)
        
        assert len(messages2) >= 1
        second_response = messages2[-1].text
        assert "7" in second_response or "seven" in second_response.lower()

    @pytest.mark.asyncio
    async def test_assistant_message_injection(self):
        """Test injecting assistant messages in conversation."""
        builder = (LLMPromptBuilder()
                   .agent("You are a helpful assistant.")
                   .request("What is the capital of France?")
                   .assistant("The capital of France is Paris.")
                   .request("What about Germany?"))
        
        conversation = builder.prompt_conversation()
        
        # Verify assistant message was injected
        assert len(conversation.messages) == 4
        assert conversation.messages[2].role == Role.ASSISTANT
        assert "Paris" in conversation.messages[2].text
        
        # Execute and get response
        messages = []
        async for message in conversation:
            messages.append(message)
        
        assert len(messages) >= 1
        response = messages[-1].text
        assert "Berlin" in response or "berlin" in response.lower()


class TestSerializationWithRealData:
    """Test serialization with actual conversation data."""

    @pytest.mark.asyncio
    async def test_serialize_real_conversation(self):
        """Test serializing a real conversation."""
        # Have a real conversation
        builder = LLMPromptBuilder().agent("You are helpful. Be concise.").request("What is Python?")
        conversation = builder.prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
        
        # Serialize
        data = conversation.messages.to_dict()
        
        # Verify serialization
        assert "version" in data
        assert "messages" in data
        assert len(data["messages"]) >= 2
        
        # Deserialize
        restored_messages = MessageList.from_dict(data)
        
        # Verify restoration
        assert len(restored_messages) == len(conversation.messages)
        for i, msg in enumerate(restored_messages):
            assert type(msg) == type(conversation.messages[i])
            if hasattr(msg, 'text'):
                assert msg.text == conversation.messages[i].text

    @pytest.mark.asyncio
    async def test_continue_after_serialization(self):
        """Test continuing a conversation after serialization/restoration."""
        # First conversation
        builder1 = LLMPromptBuilder().agent("You are helpful. Be concise.").request("What is 5+5?")
        conversation1 = builder1.prompt_conversation()
        
        messages1 = []
        async for message in conversation1:
            messages1.append(message)
        
        # Serialize
        data = conversation1.messages.to_dict()
        
        # Restore to new conversation
        restored_messages = MessageList.from_dict(data)
        conversation2 = LLMConversation(messages=restored_messages)
        
        # Continue the conversation
        continuation = conversation2.continuation
        builder2 = continuation.request("What is that times 2?")
        
        # Apply deltas and reset generator
        conversation2.messages.extend(builder2._delta_messages)
        conversation2._generator = None
        
        messages2 = []
        async for message in conversation2:
            messages2.append(message)
        
        assert len(messages2) >= 1
        response = messages2[-1].text
        assert "20" in response or "twenty" in response.lower()


class TestToolCallingWithNewArchitecture:
    """Test tool calling with the new architecture."""

    @pytest.mark.asyncio
    async def test_simple_tool_call(self):
        """Test a simple tool call."""
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny and 22°C."
        
        builder = (LLMPromptBuilder()
                   .agent("You are a helpful assistant.")
                   .tool(get_weather)
                   .request("What's the weather in Paris?"))
        
        conversation = builder.prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
        
        # Should have at least tool call and final response
        assert len(messages) >= 2
        
        # Find tool call message
        tool_call_msg = None
        for msg in messages:
            if isinstance(msg, ToolCallMessage):
                tool_call_msg = msg
                break
        
        assert tool_call_msg is not None
        assert tool_call_msg.tool_name == "get_weather"
        assert "Paris" in str(tool_call_msg.arguments)
        assert tool_call_msg.result is not None
        assert "sunny" in tool_call_msg.result

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Test multiple tool calls in sequence."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        builder = (LLMPromptBuilder()
                   .agent("You are a helpful math assistant.")
                   .tools(add, multiply)
                   .request("What is (5 + 3) * 2?"))
        
        conversation = builder.prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
        
        # Should have tool calls
        tool_calls = [msg for msg in messages if isinstance(msg, ToolCallMessage)]
        assert len(tool_calls) >= 1
        
        # Final response should have the answer
        final_response = messages[-1]
        assert isinstance(final_response, TextMessage)
        assert "16" in final_response.text or "sixteen" in final_response.text.lower()

    @pytest.mark.asyncio
    async def test_serialize_conversation_with_tool_calls(self):
        """Test serializing a conversation that includes tool calls."""
        def get_time() -> str:
            """Get the current time."""
            return "12:00 PM"
        
        builder = (LLMPromptBuilder()
                   .agent("You are helpful.")
                   .tool(get_time)
                   .request("What time is it?"))
        
        conversation = builder.prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
        
        # Serialize
        data = conversation.messages.to_dict()
        
        # Verify tool call is in serialized data
        tool_call_found = False
        for msg_data in data["messages"]:
            if msg_data.get("type") == "ToolCallMessage":
                tool_call_found = True
                assert msg_data["tool_name"] == "get_time"
                break
        
        assert tool_call_found
        
        # Deserialize
        restored_messages = MessageList.from_dict(data)
        
        # Verify tool call is restored
        tool_call_msg = None
        for msg in restored_messages:
            if isinstance(msg, ToolCallMessage):
                tool_call_msg = msg
                break
        
        assert tool_call_msg is not None
        assert tool_call_msg.tool_name == "get_time"


class TestCrossModelContinuation:
    """Test continuing conversations across different models."""

    @pytest.mark.asyncio
    async def test_continue_with_different_model(self):
        """Test continuing a conversation with a different model."""
        # First conversation with one model
        builder1 = (LLMPromptBuilder()
                    .agent("You are helpful. Be concise.")
                    .model("claude-3-haiku")
                    .request("What is 10+10?"))
        
        conversation1 = builder1.prompt_conversation()
        
        messages1 = []
        async for message in conversation1:
            messages1.append(message)
        
        assert len(messages1) >= 1
        
        # Serialize
        data = conversation1.messages.to_dict()
        
        # Restore and continue with different model
        restored_messages = MessageList.from_dict(data)
        conversation2 = LLMConversation(messages=restored_messages)
        
        continuation = conversation2.continuation
        builder2 = continuation.model("claude-3-sonnet").request("What is that plus 5?")
        
        # Apply deltas and reset generator
        conversation2.messages.extend(builder2._delta_messages)
        conversation2.apply_config_deltas(builder2._delta_config)
        conversation2._generator = None
        
        messages2 = []
        async for message in conversation2:
            messages2.append(message)
        
        assert len(messages2) >= 1
        response = messages2[-1].text
        assert "25" in response or "twenty" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
