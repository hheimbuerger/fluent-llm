"""Unit tests for LLMPromptBuilder delta pattern and immutability."""
import pytest
from fluent_llm import llm
from fluent_llm.builder import LLMPromptBuilder
from fluent_llm.messages import TextMessage, AgentMessage, Role
from fluent_llm.conversation import MessageList


class TestBuilderImmutability:
    """Test that builder methods return new instances."""
    
    def test_agent_returns_new_instance(self):
        """Test that agent() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.agent("You are helpful")
        
        assert builder1 is not builder2
        assert len(builder1._delta_messages) == 0
        assert len(builder2._delta_messages) == 1
    
    def test_request_returns_new_instance(self):
        """Test that request() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.request("Hello")
        
        assert builder1 is not builder2
        assert len(builder1._delta_messages) == 0
        assert len(builder2._delta_messages) == 1
    
    def test_assistant_returns_new_instance(self):
        """Test that assistant() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.assistant("Hi there")
        
        assert builder1 is not builder2
        assert len(builder1._delta_messages) == 0
        assert len(builder2._delta_messages) == 1
    
    def test_context_returns_new_instance(self):
        """Test that context() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.context("Background info")
        
        assert builder1 is not builder2
    
    def test_model_returns_new_instance(self):
        """Test that model() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.model("gpt-4")
        
        assert builder1 is not builder2
        assert len(builder1._delta_config) == 0
        assert "preferred_model" in builder2._delta_config
    
    def test_provider_returns_new_instance(self):
        """Test that provider() returns a new builder instance."""
        builder1 = llm
        builder2 = builder1.provider("anthropic")
        
        assert builder1 is not builder2
        assert "preferred_provider" in builder2._delta_config
    
    def test_chaining_creates_multiple_instances(self):
        """Test that chaining creates multiple distinct instances."""
        builder1 = llm
        builder2 = builder1.agent("System")
        builder3 = builder2.request("User")
        builder4 = builder3.assistant("Assistant")
        
        assert builder1 is not builder2
        assert builder2 is not builder3
        assert builder3 is not builder4
        
        assert len(builder1._delta_messages) == 0
        assert len(builder2._delta_messages) == 1
        assert len(builder3._delta_messages) == 2
        assert len(builder4._delta_messages) == 3


class TestDeltaAccumulation:
    """Test delta message and config accumulation."""
    
    def test_message_deltas_accumulate(self):
        """Test that message deltas accumulate correctly."""
        builder = llm.agent("System").request("User").assistant("Assistant")
        
        assert len(builder._delta_messages) == 3
        assert isinstance(builder._delta_messages[0], AgentMessage)
        assert isinstance(builder._delta_messages[1], TextMessage)
        assert builder._delta_messages[1].role == Role.USER
        assert isinstance(builder._delta_messages[2], TextMessage)
        assert builder._delta_messages[2].role == Role.ASSISTANT
    
    def test_config_deltas_accumulate(self):
        """Test that config deltas accumulate correctly."""
        builder = llm.model("gpt-4").provider("openai")
        
        assert builder._delta_config["preferred_model"] == "gpt-4"
        assert builder._delta_config["preferred_provider"] == "openai"
    
    def test_config_deltas_override(self):
        """Test that later config deltas override earlier ones."""
        builder = llm.model("gpt-3.5").model("gpt-4")
        
        assert builder._delta_config["preferred_model"] == "gpt-4"
    
    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from messages."""
        builder = llm.agent("  System  ").request("  User  ")
        
        assert builder._delta_messages[0].text == "System"
        assert builder._delta_messages[1].text == "User"


class TestAssistantMessageInjection:
    """Test assistant message injection."""
    
    def test_assistant_creates_assistant_role_message(self):
        """Test that assistant() creates a message with ASSISTANT role."""
        builder = llm.assistant("Hello")
        
        assert len(builder._delta_messages) == 1
        assert isinstance(builder._delta_messages[0], TextMessage)
        assert builder._delta_messages[0].role == Role.ASSISTANT
        assert builder._delta_messages[0].text == "Hello"
    
    def test_assistant_in_conversation_flow(self):
        """Test assistant message in a conversation flow."""
        builder = (llm
                   .agent("You are helpful")
                   .request("Hi")
                   .assistant("Hello! How can I help?")
                   .request("Tell me about Python"))
        
        assert len(builder._delta_messages) == 4
        assert builder._delta_messages[2].role == Role.ASSISTANT
    
    def test_multiple_assistant_messages(self):
        """Test multiple assistant messages."""
        builder = llm.assistant("First").assistant("Second")
        
        assert len(builder._delta_messages) == 2
        assert all(msg.role == Role.ASSISTANT for msg in builder._delta_messages)


class TestToolMethods:
    """Test tool-related builder methods."""
    
    def test_tool_adds_to_config(self):
        """Test that tool() adds tool to config."""
        def my_tool(x: int) -> int:
            """A test tool."""
            return x * 2
        
        builder = llm.tool(my_tool)
        
        assert "tools" in builder._delta_config
        assert len(builder._delta_config["tools"]) == 1
        assert builder._delta_config["tools"][0].name == "my_tool"
    
    def test_tools_with_multiple_functions(self):
        """Test tools() with multiple functions."""
        def tool1(x: int) -> int:
            """Tool 1."""
            return x
        
        def tool2(y: str) -> str:
            """Tool 2."""
            return y
        
        builder = llm.tools(tool1, tool2)
        
        assert len(builder._delta_config["tools"]) == 2
    
    def test_tools_with_list(self):
        """Test tools() with a list of functions."""
        def tool1(x: int) -> int:
            """Tool 1."""
            return x
        
        def tool2(y: str) -> str:
            """Tool 2."""
            return y
        
        builder = llm.tools([tool1, tool2])
        
        assert len(builder._delta_config["tools"]) == 2
    
    def test_chaining_tool_calls(self):
        """Test chaining multiple tool() calls."""
        def tool1(x: int) -> int:
            """Tool 1."""
            return x
        
        def tool2(y: str) -> str:
            """Tool 2."""
            return y
        
        builder = llm.tool(tool1).tool(tool2)
        
        assert len(builder._delta_config["tools"]) == 2


class TestBuilderInitialization:
    """Test builder initialization."""
    
    def test_default_initialization(self):
        """Test default builder initialization."""
        builder = LLMPromptBuilder()
        
        assert len(builder._delta_messages) == 0
        assert len(builder._delta_config) == 0
        assert builder._conversation is not None
    
    def test_global_llm_instance(self):
        """Test that global llm instance exists."""
        assert llm is not None
        assert isinstance(llm, LLMPromptBuilder)
