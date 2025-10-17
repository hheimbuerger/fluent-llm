"""Unit tests for MessageList serialization and operations."""
import pytest
from fluent_llm.conversation import MessageList, MessageListDeserializationError
from fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, Role


class TestMessageListSerialization:
    """Test MessageList serialization and deserialization."""
    
    def test_empty_message_list_serialization(self):
        """Test serializing an empty MessageList."""
        ml = MessageList()
        data = ml.to_dict()
        
        assert data["version"] == "1.0"
        assert data["messages"] == []
    
    def test_empty_message_list_deserialization(self):
        """Test deserializing an empty MessageList."""
        data = {"version": "1.0", "messages": []}
        ml = MessageList.from_dict(data)
        
        assert len(ml) == 0
    
    def test_text_message_serialization(self):
        """Test serializing TextMessage."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        
        data = ml.to_dict()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["type"] == "TextMessage"
        assert data["messages"][0]["text"] == "Hello"
        assert data["messages"][0]["role"] == "user"
    
    def test_text_message_deserialization(self):
        """Test deserializing TextMessage."""
        data = {
            "version": "1.0",
            "messages": [
                {"type": "TextMessage", "text": "Hello", "role": "user"}
            ]
        }
        ml = MessageList.from_dict(data)
        
        assert len(ml) == 1
        assert isinstance(ml[0], TextMessage)
        assert ml[0].text == "Hello"
        assert ml[0].role == Role.USER
    
    def test_agent_message_serialization(self):
        """Test serializing AgentMessage."""
        ml = MessageList()
        ml.append(AgentMessage("You are helpful"))
        
        data = ml.to_dict()
        assert data["messages"][0]["type"] == "AgentMessage"
        assert data["messages"][0]["text"] == "You are helpful"
        assert data["messages"][0]["role"] == "system"
    
    def test_agent_message_deserialization(self):
        """Test deserializing AgentMessage."""
        data = {
            "version": "1.0",
            "messages": [
                {"type": "AgentMessage", "text": "You are helpful", "role": "system"}
            ]
        }
        ml = MessageList.from_dict(data)
        
        assert isinstance(ml[0], AgentMessage)
        assert ml[0].text == "You are helpful"
    
    def test_tool_call_message_serialization(self):
        """Test serializing ToolCallMessage."""
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
        msg_data = data["messages"][0]
        assert msg_data["type"] == "ToolCallMessage"
        assert msg_data["tool_name"] == "get_weather"
        assert msg_data["arguments"] == {"location": "Paris"}
        assert msg_data["result"] == "Sunny"
    
    def test_tool_call_message_with_error_serialization(self):
        """Test serializing ToolCallMessage with error."""
        ml = MessageList()
        ml.append(ToolCallMessage(
            message="",
            tool_name="failing_tool",
            tool_call_id="call_456",
            arguments={},
            result=None,
            error=ValueError("Tool failed")
        ))
        
        data = ml.to_dict()
        msg_data = data["messages"][0]
        assert msg_data["error"] == "Tool failed"
        assert msg_data["result"] is None
    
    def test_mixed_messages_round_trip(self):
        """Test round-trip serialization with mixed message types."""
        ml = MessageList()
        ml.append(AgentMessage("System prompt"))
        ml.append(TextMessage("User message", Role.USER))
        ml.append(TextMessage("Assistant response", Role.ASSISTANT))
        ml.append(ToolCallMessage(
            message="",
            tool_name="test_tool",
            tool_call_id="call_789",
            arguments={"arg": "value"},
            result="result"
        ))
        
        # Serialize
        data = ml.to_dict()
        
        # Deserialize
        ml2 = MessageList.from_dict(data)
        
        assert len(ml2) == 4
        assert isinstance(ml2[0], AgentMessage)
        assert isinstance(ml2[1], TextMessage)
        assert ml2[1].role == Role.USER
        assert isinstance(ml2[2], TextMessage)
        assert ml2[2].role == Role.ASSISTANT
        assert isinstance(ml2[3], ToolCallMessage)
    
    def test_invalid_version_raises_error(self):
        """Test that invalid version raises error."""
        data = {"version": "2.0", "messages": []}
        
        with pytest.raises(MessageListDeserializationError, match="Unsupported version"):
            MessageList.from_dict(data)
    
    def test_missing_version_raises_error(self):
        """Test that missing version raises error."""
        data = {"messages": []}
        
        with pytest.raises(MessageListDeserializationError):
            MessageList.from_dict(data)
    
    def test_invalid_data_type_raises_error(self):
        """Test that invalid data type raises error."""
        with pytest.raises(MessageListDeserializationError, match="Expected dict"):
            MessageList.from_dict("not a dict")
    
    def test_invalid_messages_type_raises_error(self):
        """Test that invalid messages type raises error."""
        data = {"version": "1.0", "messages": "not a list"}
        
        with pytest.raises(MessageListDeserializationError, match="Expected 'messages' to be a list"):
            MessageList.from_dict(data)
    
    def test_unknown_message_type_raises_error(self):
        """Test that unknown message type raises error."""
        data = {
            "version": "1.0",
            "messages": [
                {"type": "UnknownMessage", "text": "test"}
            ]
        }
        
        with pytest.raises(MessageListDeserializationError, match="Unknown message type"):
            MessageList.from_dict(data)


class TestMessageListOperations:
    """Test MessageList mutable operations."""
    
    def test_append(self):
        """Test appending messages."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        
        assert len(ml) == 1
        assert ml[0].text == "Hello"
    
    def test_extend_with_list(self):
        """Test extending with a list of messages."""
        ml = MessageList()
        ml.extend([
            TextMessage("Hello", Role.USER),
            TextMessage("Hi", Role.ASSISTANT)
        ])
        
        assert len(ml) == 2
    
    def test_extend_with_message_list(self):
        """Test extending with another MessageList."""
        ml1 = MessageList()
        ml1.append(TextMessage("Hello", Role.USER))
        
        ml2 = MessageList()
        ml2.append(TextMessage("Hi", Role.ASSISTANT))
        
        ml1.extend(ml2)
        
        assert len(ml1) == 2
    
    def test_copy(self):
        """Test copying a MessageList."""
        ml1 = MessageList()
        ml1.append(TextMessage("Hello", Role.USER))
        
        ml2 = ml1.copy()
        
        assert len(ml2) == 1
        assert ml1 is not ml2
        assert ml1[0] is ml2[0]  # Shallow copy
    
    def test_iteration(self):
        """Test iterating over MessageList."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        ml.append(TextMessage("Hi", Role.ASSISTANT))
        
        messages = list(ml)
        assert len(messages) == 2
    
    def test_indexing(self):
        """Test indexing MessageList."""
        ml = MessageList()
        ml.append(TextMessage("First", Role.USER))
        ml.append(TextMessage("Second", Role.USER))
        
        assert ml[0].text == "First"
        assert ml[1].text == "Second"
    
    def test_has_type(self):
        """Test has_type method."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        ml.append(AgentMessage("System"))
        
        assert ml.has_type(TextMessage)
        assert ml.has_type(AgentMessage)
        assert not ml.has_type(ToolCallMessage)
    
    def test_has_text_property(self):
        """Test has_text property."""
        ml = MessageList()
        assert not ml.has_text
        
        ml.append(TextMessage("Hello", Role.USER))
        assert ml.has_text
    
    def test_has_agent_property(self):
        """Test has_agent property."""
        ml = MessageList()
        assert not ml.has_agent
        
        ml.append(AgentMessage("System"))
        assert ml.has_agent
    
    def test_has_tool_call_property(self):
        """Test has_tool_call property."""
        ml = MessageList()
        assert not ml.has_tool_call
        
        ml.append(ToolCallMessage(
            message="",
            tool_name="test",
            tool_call_id="123",
            arguments={}
        ))
        assert ml.has_tool_call
    
    def test_to_dict_list(self):
        """Test to_dict_list method."""
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        ml.append(AgentMessage("System"))
        
        dict_list = ml.to_dict_list()
        
        assert len(dict_list) == 2
        assert dict_list[0]["type"] == "TextMessage"
        assert dict_list[1]["type"] == "AgentMessage"
