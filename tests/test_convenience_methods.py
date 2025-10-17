"""Unit tests for convenience load/save methods."""
import json
import tempfile
import pytest
from pathlib import Path
from io import StringIO, BytesIO

from fluent_llm.builder import llm
from fluent_llm.conversation import MessageList, MessageListDeserializationError
from fluent_llm.messages import TextMessage, AgentMessage, ToolCallMessage, Role


class TestLoadConversation:
    """Tests for load_conversation() method."""
    
    def test_load_from_string_path(self):
        """Test loading conversation from a string file path."""
        # Create a temporary file with conversation data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "messages": [
                    {"type": "AgentMessage", "text": "You are helpful", "role": "system"},
                    {"type": "TextMessage", "text": "Hello", "role": "user"}
                ],
                "version": "1.0"
            }
            json.dump(data, f)
            temp_path = f.name
        
        try:
            # Load conversation using string path
            conversation = llm.load_conversation(temp_path)
            
            # Verify messages were loaded
            assert len(conversation.messages) == 2
            assert isinstance(conversation.messages[0], AgentMessage)
            assert conversation.messages[0].text == "You are helpful"
            assert isinstance(conversation.messages[1], TextMessage)
            assert conversation.messages[1].text == "Hello"
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_load_from_path_object(self):
        """Test loading conversation from a Path object."""
        # Create a temporary file with conversation data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "messages": [
                    {"type": "TextMessage", "text": "Test message", "role": "user"}
                ],
                "version": "1.0"
            }
            json.dump(data, f)
            temp_path = Path(f.name)
        
        try:
            # Load conversation using Path object
            conversation = llm.load_conversation(temp_path)
            
            # Verify messages were loaded
            assert len(conversation.messages) == 1
            assert conversation.messages[0].text == "Test message"
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_load_from_io_stream(self):
        """Test loading conversation from an IO stream."""
        # Create conversation data in a StringIO stream
        data = {
            "messages": [
                {"type": "AgentMessage", "text": "System prompt", "role": "system"},
                {"type": "TextMessage", "text": "User query", "role": "user"},
                {"type": "TextMessage", "text": "Assistant response", "role": "assistant"}
            ],
            "version": "1.0"
        }
        stream = StringIO(json.dumps(data))
        
        # Load conversation from stream
        conversation = llm.load_conversation(stream)
        
        # Verify messages were loaded
        assert len(conversation.messages) == 3
        assert conversation.messages[0].text == "System prompt"
        assert conversation.messages[1].text == "User query"
        assert conversation.messages[2].text == "Assistant response"
        assert conversation.messages[2].role == Role.ASSISTANT
    
    def test_load_from_dict(self):
        """Test loading conversation from a dictionary."""
        # Create conversation data as dict
        data = {
            "messages": [
                {"type": "TextMessage", "text": "Direct dict load", "role": "user"}
            ],
            "version": "1.0"
        }
        
        # Load conversation from dict
        conversation = llm.load_conversation(data)
        
        # Verify messages were loaded
        assert len(conversation.messages) == 1
        assert conversation.messages[0].text == "Direct dict load"
    
    def test_load_with_tool_call_messages(self):
        """Test loading conversation with ToolCallMessage."""
        data = {
            "messages": [
                {"type": "TextMessage", "text": "What's the weather?", "role": "user"},
                {
                    "type": "ToolCallMessage",
                    "message": "Checking weather",
                    "tool_name": "get_weather",
                    "tool_call_id": "call_123",
                    "arguments": {"location": "Paris"},
                    "result": "Sunny, 22°C",
                    "error": None,
                    "role": "assistant"
                }
            ],
            "version": "1.0"
        }
        
        # Load conversation
        conversation = llm.load_conversation(data)
        
        # Verify tool call message was loaded correctly
        assert len(conversation.messages) == 2
        tool_msg = conversation.messages[1]
        assert isinstance(tool_msg, ToolCallMessage)
        assert tool_msg.tool_name == "get_weather"
        assert tool_msg.result == "Sunny, 22°C"
        assert tool_msg.error is None
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            llm.load_conversation("nonexistent_file.json")
    
    def test_load_invalid_json_raises_error(self):
        """Test that loading invalid JSON raises JSONDecodeError."""
        # Create a file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                llm.load_conversation(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_version_raises_error(self):
        """Test that loading unsupported version raises error."""
        data = {
            "messages": [],
            "version": "2.0"  # Unsupported version
        }
        
        with pytest.raises(MessageListDeserializationError, match="Unsupported version"):
            llm.load_conversation(data)
    
    def test_load_missing_messages_field_raises_error(self):
        """Test that loading data without messages field raises error."""
        data = {
            "version": "1.0"
            # Missing "messages" field
        }
        
        with pytest.raises(MessageListDeserializationError):
            llm.load_conversation(data)
    
    def test_load_with_builder_config(self):
        """Test that loaded conversation can be configured with builder methods."""
        data = {
            "messages": [
                {"type": "TextMessage", "text": "Previous message", "role": "user"}
            ],
            "version": "1.0"
        }
        
        # Load conversation with builder that has config
        conversation = llm.provider("anthropic").model("claude-3-sonnet").load_conversation(data)
        
        # Verify conversation was created and config was applied
        assert len(conversation.messages) == 1
        assert conversation._config.preferred_provider == "anthropic"
        assert conversation._config.preferred_model == "claude-3-sonnet"
    
    def test_load_unsupported_source_type_raises_error(self):
        """Test that loading from unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            llm.load_conversation(12345)  # Invalid type


class TestSaveConversation:
    """Tests for save() method on LLMConversation."""
    
    def test_save_to_string_path(self):
        """Test saving conversation to a string file path."""
        # Create a conversation with messages
        data = {
            "messages": [
                {"type": "AgentMessage", "text": "You are helpful", "role": "system"},
                {"type": "TextMessage", "text": "Hello", "role": "user"}
            ],
            "version": "1.0"
        }
        conversation = llm.load_conversation(data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            conversation.save(temp_path)
            
            # Verify file was created and contains correct data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["version"] == "1.0"
            assert len(saved_data["messages"]) == 2
            assert saved_data["messages"][0]["text"] == "You are helpful"
            assert saved_data["messages"][1]["text"] == "Hello"
        finally:
            Path(temp_path).unlink()
    
    def test_save_to_path_object(self):
        """Test saving conversation to a Path object."""
        # Create a conversation
        data = {
            "messages": [
                {"type": "TextMessage", "text": "Test save", "role": "user"}
            ],
            "version": "1.0"
        }
        conversation = llm.load_conversation(data)
        
        # Save to Path object
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            conversation.save(temp_path)
            
            # Verify file was created
            assert temp_path.exists()
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert len(saved_data["messages"]) == 1
        finally:
            temp_path.unlink()
    
    def test_save_to_io_stream(self):
        """Test saving conversation to an IO stream."""
        # Create a conversation
        data = {
            "messages": [
                {"type": "AgentMessage", "text": "System", "role": "system"},
                {"type": "TextMessage", "text": "User", "role": "user"},
                {"type": "TextMessage", "text": "Assistant", "role": "assistant"}
            ],
            "version": "1.0"
        }
        conversation = llm.load_conversation(data)
        
        # Save to StringIO stream
        stream = StringIO()
        conversation.save(stream)
        
        # Verify stream contains correct data
        stream.seek(0)
        saved_data = json.load(stream)
        assert len(saved_data["messages"]) == 3
        assert saved_data["messages"][0]["text"] == "System"
        assert saved_data["messages"][2]["role"] == "assistant"
    
    def test_save_with_tool_call_messages(self):
        """Test saving conversation with ToolCallMessage."""
        data = {
            "messages": [
                {
                    "type": "ToolCallMessage",
                    "message": "Calling tool",
                    "tool_name": "get_weather",
                    "tool_call_id": "call_456",
                    "arguments": {"city": "London"},
                    "result": "Rainy, 15°C",
                    "error": None,
                    "role": "assistant"
                }
            ],
            "version": "1.0"
        }
        conversation = llm.load_conversation(data)
        
        # Save and reload
        stream = StringIO()
        conversation.save(stream)
        stream.seek(0)
        saved_data = json.load(stream)
        
        # Verify tool call was saved correctly
        tool_msg = saved_data["messages"][0]
        assert tool_msg["type"] == "ToolCallMessage"
        assert tool_msg["tool_name"] == "get_weather"
        assert tool_msg["result"] == "Rainy, 15°C"
    
    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if they don't exist."""
        # Create a path with non-existent parent directories
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "subdir1" / "subdir2" / "conversation.json"
            
            # Create a conversation
            data = {
                "messages": [
                    {"type": "TextMessage", "text": "Test", "role": "user"}
                ],
                "version": "1.0"
            }
            conversation = llm.load_conversation(data)
            
            # Save should create parent directories
            conversation.save(nested_path)
            
            # Verify file was created
            assert nested_path.exists()
            with open(nested_path, 'r') as f:
                saved_data = json.load(f)
            assert len(saved_data["messages"]) == 1
    
    def test_save_unsupported_destination_type_raises_error(self):
        """Test that saving to unsupported type raises IOError."""
        data = {
            "messages": [
                {"type": "TextMessage", "text": "Test", "role": "user"}
            ],
            "version": "1.0"
        }
        conversation = llm.load_conversation(data)
        
        with pytest.raises(IOError, match="Failed to save conversation"):
            conversation.save(12345)  # Invalid type


class TestRoundTripSerialization:
    """Tests for complete save/load round-trip."""
    
    def test_round_trip_with_file(self):
        """Test that save and load preserve all message data."""
        # Create original conversation
        original_data = {
            "messages": [
                {"type": "AgentMessage", "text": "You are helpful", "role": "system"},
                {"type": "TextMessage", "text": "Hello", "role": "user"},
                {"type": "TextMessage", "text": "Hi there!", "role": "assistant"},
                {
                    "type": "ToolCallMessage",
                    "message": "Using tool",
                    "tool_name": "calculator",
                    "tool_call_id": "call_789",
                    "arguments": {"operation": "add", "a": 5, "b": 3},
                    "result": 8,
                    "error": None,
                    "role": "assistant"
                }
            ],
            "version": "1.0"
        }
        original_conversation = llm.load_conversation(original_data)
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original_conversation.save(temp_path)
            
            # Load from file
            restored_conversation = llm.load_conversation(temp_path)
            
            # Verify all messages match
            assert len(restored_conversation.messages) == len(original_conversation.messages)
            
            for i, (orig, restored) in enumerate(zip(original_conversation.messages, restored_conversation.messages)):
                assert type(orig) == type(restored), f"Message {i} type mismatch"
                if isinstance(orig, (TextMessage, AgentMessage)):
                    assert orig.text == restored.text, f"Message {i} text mismatch"
                    assert orig.role == restored.role, f"Message {i} role mismatch"
                elif isinstance(orig, ToolCallMessage):
                    assert orig.tool_name == restored.tool_name
                    assert orig.tool_call_id == restored.tool_call_id
                    assert orig.arguments == restored.arguments
                    assert orig.result == restored.result
        finally:
            Path(temp_path).unlink()
    
    def test_round_trip_with_stream(self):
        """Test round-trip serialization using streams."""
        # Create conversation
        original_data = {
            "messages": [
                {"type": "TextMessage", "text": "Stream test", "role": "user"}
            ],
            "version": "1.0"
        }
        original_conversation = llm.load_conversation(original_data)
        
        # Save to stream
        save_stream = StringIO()
        original_conversation.save(save_stream)
        
        # Load from stream
        save_stream.seek(0)
        restored_conversation = llm.load_conversation(save_stream)
        
        # Verify message matches
        assert len(restored_conversation.messages) == 1
        assert restored_conversation.messages[0].text == "Stream test"
    
    def test_continuation_after_load(self):
        """Test that continuation works correctly after loading."""
        # Create and save conversation
        original_data = {
            "messages": [
                {"type": "TextMessage", "text": "First message", "role": "user"}
            ],
            "version": "1.0"
        }
        original_conversation = llm.load_conversation(original_data)
        
        # Save and reload
        stream = StringIO()
        original_conversation.save(stream)
        stream.seek(0)
        restored_conversation = llm.load_conversation(stream)
        
        # Get continuation and add more messages
        continuation = restored_conversation.continuation
        updated_builder = continuation.request("Second message")
        
        # Apply deltas
        updated_conversation = updated_builder.prompt_conversation()
        
        # Verify both messages are present
        assert len(updated_conversation.messages) == 2
        assert updated_conversation.messages[0].text == "First message"
        assert updated_conversation.messages[1].text == "Second message"


class TestIntegrationWithMessageList:
    """Tests for integration between convenience methods and MessageList."""
    
    def test_load_uses_message_list_from_dict(self):
        """Test that load_conversation uses MessageList.from_dict."""
        data = {
            "messages": [
                {"type": "TextMessage", "text": "Test", "role": "user"}
            ],
            "version": "1.0"
        }
        
        conversation = llm.load_conversation(data)
        
        # Verify conversation has MessageList
        assert isinstance(conversation.messages, MessageList)
        assert len(conversation.messages) == 1
    
    def test_save_uses_message_list_to_dict(self):
        """Test that save uses MessageList.to_dict."""
        # Create conversation with MessageList
        messages = MessageList([
            AgentMessage(text="System prompt"),
            TextMessage(text="User message", role=Role.USER)
        ])
        from fluent_llm.conversation import LLMConversation
        conversation = LLMConversation(messages=messages)
        
        # Save to stream
        stream = StringIO()
        conversation.save(stream)
        
        # Verify saved data structure
        stream.seek(0)
        saved_data = json.load(stream)
        assert "messages" in saved_data
        assert "version" in saved_data
        assert len(saved_data["messages"]) == 2