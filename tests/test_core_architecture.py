"""Comprehensive unit tests for the new three-class architecture."""
import pytest
from fluent_llm import llm, MessageList, LLMConversation
from fluent_llm.conversation import (
    MessageListDeserializationError,
    ConversationConfigurationError,
    DeltaApplicationError,
    ConversationConfig,
)
from fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, ImageMessage, AudioMessage, Role
from fluent_llm.tools import Tool
from pathlib import Path


class TestMessageListSerialization:
    """Test MessageList serialization and deserialization."""

    def test_serialize_text_messages(self):
        """Test serialization of text messages."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        ml.append(TextMessage("Hi there!", Role.ASSISTANT))
        
        data = ml.to_dict()
        assert data["version"] == "1.0"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["type"] == "TextMessage"
        assert data["messages"][0]["text"] == "Hello"
        assert data["messages"][0]["role"] == "user"

    def test_serialize_agent_messages(self):
        """Test serialization of agent messages."""
        ml = MessageList()
        ml.append(AgentMessage("You are helpful"))
        
        data = ml.to_dict()
        assert data["messages"][0]["type"] == "AgentMessage"
        assert data["messages"][0]["text"] == "You are helpful"
        assert data["messages"][0]["role"] == "system"

    def test_serialize_tool_call_messages(self):
        """Test serialization of tool call messages."""
        ml = MessageList()
        ml.append(ToolCallMessage(
            message="Calling tool",
            tool_name="get_weather",
            tool_call_id="call_123",
            arguments={"location": "Paris"},
            result="Sunny",
            error=None
        ))
        
        data = ml.to_dict()
        assert data["messages"][0]["type"] == "ToolCallMessage"
        assert data["messages"][0]["tool_name"] == "get_weather"
        assert data["messages"][0]["arguments"] == {"location": "Paris"}
        assert data["messages"][0]["result"] == "Sunny"

    def test_serialize_tool_call_with_error(self):
        """Test serialization of tool call with error."""
        ml = MessageList()
        ml.append(ToolCallMessage(
            message="Calling tool",
            tool_name="failing_tool",
            tool_call_id="call_456",
            arguments={"param": "value"},
            result=None,
            error=ValueError("Tool failed")
        ))
        
        data = ml.to_dict()
        assert data["messages"][0]["error"] == "Tool failed"
        assert data["messages"][0]["result"] is None

    def test_round_trip_all_message_types(self):
        """Test round-trip serialization for all message types."""
        ml = MessageList()
        ml.append(AgentMessage("System message"))
        ml.append(TextMessage("User message", Role.USER))
        ml.append(TextMessage("Assistant message", Role.ASSISTANT))
        ml.append(ToolCallMessage(
            message="Tool call",
            tool_name="test_tool",
            tool_call_id="call_789",
            arguments={"arg": "value"},
            result="result",
            error=None
        ))
        
        # Serialize
        data = ml.to_dict()
        
        # Deserialize
        ml2 = MessageList.from_dict(data)
        
        # Verify
        assert len(ml2) == 4
        assert isinstance(ml2[0], AgentMessage)
        assert ml2[0].text == "System message"
        assert isinstance(ml2[1], TextMessage)
        assert ml2[1].role == Role.USER
        assert isinstance(ml2[2], TextMessage)
        assert ml2[2].role == Role.ASSISTANT
        assert isinstance(ml2[3], ToolCallMessage)
        assert ml2[3].tool_name == "test_tool"

    def test_version_handling(self):
        """Test version handling in serialization."""
        ml = MessageList()
        ml.append(TextMessage("Test", Role.USER))
        
        data = ml.to_dict()
        assert "version" in data
        assert data["version"] == "1.0"

    def test_deserialize_invalid_data_type(self):
        """Test error handling for invalid data type."""
        with pytest.raises(MessageListDeserializationError, match="Expected dict"):
            MessageList.from_dict("not a dict")

    def test_deserialize_invalid_version(self):
        """Test error handling for unsupported version."""
        data = {"version": "2.0", "messages": []}
        with pytest.raises(MessageListDeserializationError, match="Unsupported version"):
            MessageList.from_dict(data)

    def test_deserialize_missing_messages(self):
        """Test error handling for missing messages field."""
        data = {"version": "1.0"}
        with pytest.raises(MessageListDeserializationError, match="Expected 'messages' to be a list"):
            MessageList.from_dict(data)

    def test_deserialize_invalid_message_type(self):
        """Test error handling for unknown message type."""
        data = {
            "version": "1.0",
            "messages": [{"type": "UnknownMessage", "text": "test"}]
        }
        with pytest.raises(MessageListDeserializationError, match="Unknown message type"):
            MessageList.from_dict(data)

    def test_deserialize_invalid_message_data(self):
        """Test error handling for invalid message data."""
        data = {
            "version": "1.0",
            "messages": ["not a dict"]
        }
        with pytest.raises(MessageListDeserializationError, match="Expected dict"):
            MessageList.from_dict(data)


class TestMessageListMutableOperations:
    """Test MessageList mutable operations."""

    def test_append(self):
        """Test appending messages."""
        ml = MessageList()
        ml.append(TextMessage("First", Role.USER))
        ml.append(TextMessage("Second", Role.USER))
        
        assert len(ml) == 2
        assert ml[0].text == "First"
        assert ml[1].text == "Second"

    def test_extend_with_message_list(self):
        """Test extending with another MessageList."""
        ml1 = MessageList()
        ml1.append(TextMessage("First", Role.USER))
        
        ml2 = MessageList()
        ml2.append(TextMessage("Second", Role.USER))
        ml2.append(TextMessage("Third", Role.USER))
        
        ml1.extend(ml2)
        
        assert len(ml1) == 3
        assert ml1[2].text == "Third"

    def test_extend_with_list(self):
        """Test extending with a regular list."""
        ml = MessageList()
        ml.append(TextMessage("First", Role.USER))
        
        messages = [
            TextMessage("Second", Role.USER),
            TextMessage("Third", Role.USER)
        ]
        ml.extend(messages)
        
        assert len(ml) == 3

    def test_copy(self):
        """Test copying MessageList."""
        ml1 = MessageList()
        ml1.append(TextMessage("First", Role.USER))
        ml1.append(TextMessage("Second", Role.USER))
        
        ml2 = ml1.copy()
        
        # Verify it's a copy
        assert len(ml2) == 2
        assert ml2[0].text == "First"
        
        # Verify it's independent
        ml2.append(TextMessage("Third", Role.USER))
        assert len(ml1) == 2
        assert len(ml2) == 3

    def test_iteration(self):
        """Test iterating over MessageList."""
        ml = MessageList()
        ml.append(TextMessage("First", Role.USER))
        ml.append(TextMessage("Second", Role.USER))
        
        messages = list(ml)
        assert len(messages) == 2
        assert messages[0].text == "First"

    def test_indexing(self):
        """Test indexing MessageList."""
        ml = MessageList()
        ml.append(TextMessage("First", Role.USER))
        ml.append(TextMessage("Second", Role.USER))
        
        assert ml[0].text == "First"
        assert ml[1].text == "Second"
        assert ml[-1].text == "Second"


class TestBuilderImmutability:
    """Test builder immutability and delta pattern."""

    def test_all_methods_return_new_instances(self):
        """Test that all builder methods return new instances."""
        b1 = llm
        b2 = b1.agent("System")
        b3 = b2.request("User")
        b4 = b3.assistant("Assistant")
        b5 = b4.model("gpt-4")
        b6 = b5.provider("openai")
        
        # All should be different instances
        assert b1 is not b2
        assert b2 is not b3
        assert b3 is not b4
        assert b4 is not b5
        assert b5 is not b6

    def test_delta_accumulation(self):
        """Test that deltas accumulate correctly."""
        b1 = llm.agent("System")
        assert len(b1._delta_messages) == 1
        
        b2 = b1.request("User")
        assert len(b2._delta_messages) == 2
        
        b3 = b2.assistant("Assistant")
        assert len(b3._delta_messages) == 3

    def test_original_builders_unchanged(self):
        """Test that original builders remain unchanged."""
        b1 = llm.agent("System")
        original_len = len(b1._delta_messages)
        
        b2 = b1.request("User")
        
        # b1 should be unchanged
        assert len(b1._delta_messages) == original_len
        assert len(b2._delta_messages) == original_len + 1

    def test_config_delta_accumulation(self):
        """Test that config deltas accumulate."""
        b1 = llm.model("gpt-4")
        assert "preferred_model" in b1._delta_config
        assert b1._delta_config["preferred_model"] == "gpt-4"
        
        b2 = b1.provider("openai")
        assert "preferred_model" in b2._delta_config
        assert "preferred_provider" in b2._delta_config

    def test_tool_delta_accumulation(self):
        """Test that tool deltas accumulate."""
        def tool1(x: int) -> int:
            """Tool 1."""
            return x * 2
        
        def tool2(y: str) -> str:
            """Tool 2."""
            return y.upper()
        
        b1 = llm.tool(tool1)
        assert "tools" in b1._delta_config
        assert len(b1._delta_config["tools"]) == 1
        
        b2 = b1.tool(tool2)
        assert len(b2._delta_config["tools"]) == 2


class TestAssistantMessageInjection:
    """Test assistant message injection."""

    def test_assistant_method_creates_assistant_message(self):
        """Test that .assistant() creates an assistant message."""
        builder = llm.assistant("Hello from assistant")
        
        assert len(builder._delta_messages) == 1
        msg = builder._delta_messages[0]
        assert isinstance(msg, TextMessage)
        assert msg.role == Role.ASSISTANT
        assert msg.text == "Hello from assistant"

    def test_assistant_strips_whitespace(self):
        """Test that assistant message strips whitespace."""
        builder = llm.assistant("  Hello  ")
        
        msg = builder._delta_messages[0]
        assert msg.text == "Hello"

    def test_assistant_in_conversation_flow(self):
        """Test assistant message in conversation flow."""
        builder = (llm
                   .agent("You are helpful")
                   .request("User message")
                   .assistant("Assistant response")
                   .request("Follow up"))
        
        assert len(builder._delta_messages) == 4
        assert builder._delta_messages[2].role == Role.ASSISTANT


class TestDeltaApplication:
    """Test delta application to conversations."""

    def test_message_delta_application(self):
        """Test applying message deltas."""
        conversation = LLMConversation()
        builder = llm.agent("System").request("User")
        
        # Apply deltas
        conversation.messages.extend(builder._delta_messages)
        
        assert len(conversation.messages) == 2
        assert isinstance(conversation.messages[0], AgentMessage)
        assert isinstance(conversation.messages[1], TextMessage)

    def test_config_delta_application(self):
        """Test applying config deltas."""
        conversation = LLMConversation()
        delta_config = {
            "preferred_model": "gpt-4",
            "preferred_provider": "openai"
        }
        
        conversation.apply_config_deltas(delta_config)
        
        assert conversation._config.preferred_model == "gpt-4"
        assert conversation._config.preferred_provider == "openai"

    def test_tool_delta_application(self):
        """Test applying tool deltas."""
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2
        
        conversation = LLMConversation()
        tool = Tool.from_function(test_tool)
        
        conversation.apply_config_deltas({"tools": [tool]})
        
        assert len(conversation._config.tools) == 1
        assert conversation._config.tools[0].name == "test_tool"

    def test_invalid_config_key_raises_error(self):
        """Test that invalid config keys raise errors."""
        conversation = LLMConversation()
        
        with pytest.raises(ConversationConfigurationError, match="Unknown config key"):
            conversation.apply_config_deltas({"invalid_key": "value"})

    def test_invalid_tools_type_raises_error(self):
        """Test that invalid tools type raises error."""
        conversation = LLMConversation()
        
        with pytest.raises(ConversationConfigurationError, match="must be a list"):
            conversation.apply_config_deltas({"tools": "not a list"})


class TestContinuationSystem:
    """Test conversation continuation system."""

    def test_continuation_builder_creation(self):
        """Test that continuation property creates a builder."""
        conversation = LLMConversation()
        continuation = conversation.continuation
        
        from fluent_llm.builder import LLMPromptBuilder
        assert isinstance(continuation, LLMPromptBuilder)

    def test_conversation_reference_preservation(self):
        """Test that continuation references the same conversation."""
        conversation = LLMConversation()
        continuation = conversation.continuation
        
        assert continuation._conversation is conversation

    def test_continuation_state_reflection(self):
        """Test that continuation reflects conversation state."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("Test", Role.USER))
        
        continuation = conversation.continuation
        
        # The continuation should reference the same conversation
        assert len(continuation._conversation.messages) == 1

    def test_continuation_can_add_messages(self):
        """Test that continuation can add messages."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("First", Role.USER))
        
        continuation = conversation.continuation
        continuation2 = continuation.request("Second")
        
        # continuation2 should have delta messages
        assert len(continuation2._delta_messages) == 1


class TestConversationMutability:
    """Test conversation mutability and execution."""

    def test_message_list_growth(self):
        """Test that MessageList grows during conversation."""
        conversation = LLMConversation()
        
        initial_len = len(conversation.messages)
        conversation.messages.append(TextMessage("New message", Role.USER))
        
        assert len(conversation.messages) == initial_len + 1

    def test_config_delta_application_mutates_conversation(self):
        """Test that config deltas mutate the conversation."""
        conversation = LLMConversation()
        
        assert conversation._config.preferred_model is None
        
        conversation.apply_config_deltas({"preferred_model": "gpt-4"})
        
        assert conversation._config.preferred_model == "gpt-4"

    def test_continuation_builder_creation_from_conversation(self):
        """Test creating continuation builder from conversation."""
        conversation = LLMConversation()
        conversation.messages.append(TextMessage("Test", Role.USER))
        
        continuation = conversation.continuation
        
        assert continuation._conversation is conversation


class TestErrorHandling:
    """Test error handling throughout the architecture."""

    def test_invalid_deserialization_error_message_clarity(self):
        """Test that deserialization errors have clear messages."""
        with pytest.raises(MessageListDeserializationError) as exc_info:
            MessageList.from_dict({"version": "1.0", "messages": "not a list"})
        
        assert "Expected 'messages' to be a list" in str(exc_info.value)

    def test_configuration_error_message_clarity(self):
        """Test that configuration errors have clear messages."""
        conversation = LLMConversation()
        
        with pytest.raises(ConversationConfigurationError) as exc_info:
            conversation.apply_config_deltas({"unknown_key": "value"})
        
        assert "Unknown config key" in str(exc_info.value)

    def test_delta_application_error_wrapping(self):
        """Test that delta application errors are wrapped properly."""
        builder = llm.agent("Test")
        
        # Create a conversation and manually break it
        conversation = LLMConversation()
        builder._conversation = conversation
        
        # Force an error by providing invalid config
        builder._delta_config = {"tools": "not a list"}
        
        with pytest.raises(DeltaApplicationError) as exc_info:
            builder.prompt_conversation()
        
        assert "Failed to apply deltas" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
