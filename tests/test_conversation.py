"""Unit tests for LLMConversation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fluent_llm.conversation import LLMConversation, ConversationConfig, ConversationConfigurationError
from fluent_llm.messages import TextMessage, AgentMessage, Role


class TestConversationInitialization:
    """Test conversation initialization."""
    
    def test_default_initialization(self):
        """Test default conversation initialization."""
        conv = LLMConversation()
        
        assert len(conv.messages) == 0
        assert conv._config is not None
        assert isinstance(conv._config, ConversationConfig)
    
    def test_initialization_with_messages(self):
        """Test initialization with existing messages."""
        from fluent_llm.conversation import MessageList
        
        ml = MessageList()
        ml.append(TextMessage("Hello", Role.USER))
        
        conv = LLMConversation(messages=ml)
        
        assert len(conv.messages) == 1
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = ConversationConfig(preferred_provider="anthropic")
        conv = LLMConversation(config=config)
        
        assert conv._config.preferred_provider == "anthropic"


class TestConversationContinuation:
    """Test conversation continuation pattern."""
    
    def test_continuation_property_returns_builder(self):
        """Test that continuation property returns a builder."""
        from fluent_llm.builder import LLMPromptBuilder
        
        conv = LLMConversation()
        continuation = conv.continuation
        
        assert isinstance(continuation, LLMPromptBuilder)
    
    def test_continuation_references_same_conversation(self):
        """Test that continuation builder references the same conversation."""
        conv = LLMConversation()
        continuation = conv.continuation
        
        assert continuation._conversation is conv
    
    def test_continuation_reflects_conversation_state(self):
        """Test that continuation reflects current conversation state."""
        from fluent_llm.conversation import MessageList
        
        conv = LLMConversation()
        conv.messages.append(TextMessage("Hello", Role.USER))
        
        continuation = conv.continuation
        
        # The continuation should reference the same conversation
        assert len(continuation._conversation.messages) == 1


class TestConfigDeltaApplication:
    """Test configuration delta application."""
    
    def test_apply_config_deltas_updates_provider(self):
        """Test applying provider config delta."""
        conv = LLMConversation()
        conv.apply_config_deltas({"preferred_provider": "anthropic"})
        
        assert conv._config.preferred_provider == "anthropic"
    
    def test_apply_config_deltas_updates_model(self):
        """Test applying model config delta."""
        conv = LLMConversation()
        conv.apply_config_deltas({"preferred_model": "gpt-4"})
        
        assert conv._config.preferred_model == "gpt-4"
    
    def test_apply_config_deltas_updates_tools(self):
        """Test applying tools config delta."""
        from fluent_llm.tools import Tool
        
        def my_tool(x: int) -> int:
            """Test tool."""
            return x
        
        tool = Tool.from_function(my_tool)
        conv = LLMConversation()
        conv.apply_config_deltas({"tools": [tool]})
        
        assert len(conv._config.tools) == 1
    
    def test_apply_invalid_config_key_raises_error(self):
        """Test that invalid config key raises error."""
        conv = LLMConversation()
        
        with pytest.raises(ConversationConfigurationError, match="Unknown config key"):
            conv.apply_config_deltas({"invalid_key": "value"})
    
    def test_apply_invalid_tools_type_raises_error(self):
        """Test that invalid tools type raises error."""
        conv = LLMConversation()
        
        with pytest.raises(ConversationConfigurationError, match="must be a list"):
            conv.apply_config_deltas({"tools": "not a list"})


class TestConversationMutability:
    """Test that conversation is mutable."""
    
    def test_messages_can_be_appended(self):
        """Test that messages can be appended to conversation."""
        conv = LLMConversation()
        conv.messages.append(TextMessage("Hello", Role.USER))
        
        assert len(conv.messages) == 1
    
    def test_messages_grow_during_execution(self):
        """Test that messages grow during execution (mocked)."""
        # This will be tested more thoroughly in integration tests
        conv = LLMConversation()
        initial_count = len(conv.messages)
        
        conv.messages.append(TextMessage("New message", Role.ASSISTANT))
        
        assert len(conv.messages) == initial_count + 1


class TestAsyncIteratorProtocol:
    """Test async iterator protocol."""
    
    def test_conversation_has_aiter(self):
        """Test that conversation has __aiter__ method."""
        conv = LLMConversation()
        
        assert hasattr(conv, '__aiter__')
        assert callable(conv.__aiter__)
    
    def test_conversation_has_anext(self):
        """Test that conversation has __anext__ method."""
        conv = LLMConversation()
        
        assert hasattr(conv, '__anext__')
        assert callable(conv.__anext__)
    
    @pytest.mark.asyncio
    async def test_conversation_is_async_iterable(self):
        """Test that conversation can be used in async for loop."""
        conv = LLMConversation()
        
        # Mock the generator creation to avoid actual API calls
        async def mock_generator():
            yield TextMessage("Response", Role.ASSISTANT)
        
        # Replace the _create_generator method instead of setting _generator directly
        # since __aiter__ resets _generator
        conv._create_generator = mock_generator
        
        messages = []
        try:
            async for message in conv:
                messages.append(message)
        except StopAsyncIteration:
            pass
        
        assert len(messages) == 1


class TestToolExecution:
    """Test tool execution in conversation."""
    
    def test_execute_tool_call_success(self):
        """Test successful tool execution."""
        from fluent_llm.tools import Tool
        
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2
        
        tool = Tool.from_function(my_tool)
        config = ConversationConfig(tools=[tool])
        conv = LLMConversation(config=config)
        
        result, error = conv._execute_tool_call("my_tool", {"x": 5})
        
        assert result == 10
        assert error is None
    
    def test_execute_tool_call_error(self):
        """Test tool execution with error."""
        from fluent_llm.tools import Tool
        
        def failing_tool(x: int) -> int:
            """Failing tool."""
            raise ValueError("Tool failed")
        
        tool = Tool.from_function(failing_tool)
        config = ConversationConfig(tools=[tool])
        conv = LLMConversation(config=config)
        
        result, error = conv._execute_tool_call("failing_tool", {"x": 5})
        
        assert result is None
        assert isinstance(error, ValueError)
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        conv = LLMConversation()
        
        result, error = conv._execute_tool_call("unknown_tool", {})
        
        assert result is None
        assert isinstance(error, ValueError)
        assert "not found" in str(error)
