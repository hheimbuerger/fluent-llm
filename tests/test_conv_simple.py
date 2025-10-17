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
