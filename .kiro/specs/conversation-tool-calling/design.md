# Design Document

## Overview

This feature adds conversation support and tool calling capabilities to the fluent LLM library. The design extends the existing builder pattern with new methods for tool definition and conversation management, while maintaining type safety and the library's fluent API philosophy.

The implementation focuses on Anthropic's tool calling format initially, with a clear extension path for other providers. The design leverages Pydantic's existing JSON schema generation capabilities to automatically convert Python function signatures into tool schemas.

## Architecture

### Core Components

1. **Tool Definition System**
   - `Tool` dataclass for tool metadata and function references
   - Automatic JSON schema generation from Python function type annotations
   - Tool registry management within the builder

2. **Conversation Management**
   - New `prompt_conversation()` method that returns responses and continuation builder
   - Conversation state preservation across multiple turns
   - Tool call detection and execution flow

3. **Provider Extensions**
   - Enhanced Anthropic provider with tool calling support
   - Tool call parsing and response formatting
   - Error handling for tool execution failures

### Data Flow

```mermaid
graph TD
    A[User defines tools with .tools()] --> B[Builder validates and stores tools]
    B --> C[User calls prompt_conversation()]
    C --> D[Provider formats tools for API]
    D --> E[API call with tool definitions]
    E --> F[Response contains tool calls?]
    F -->|Yes| G[Execute tool functions]
    F -->|No| H[Return text response]
    G --> I[Format tool results]
    I --> J[Continue conversation with results]
    J --> E
    H --> K[Return responses and continuation builder]
    K --> L[User can continue conversation]
```

## Components and Interfaces

### Tool Definition

```python
@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., Any]
    schema: dict  # Auto-generated from function signature
    
    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "Tool":
        """Create Tool from function, deriving metadata automatically."""
        return cls(
            name=func.__name__,
            description=func.__doc__ or f"Tool: {func.__name__}",
            function=func,
            schema=generate_tool_schema(func)
        )
```

### Builder Extensions

```python
class LLMPromptBuilder:
    def tool(self, tool_function: Callable[..., Any]) -> LLMPromptBuilder:
        """Add a single tool definition from function, auto-deriving metadata."""

    def tools(self, tool_functions: List[Callable[..., Any]]) -> LLMPromptBuilder:
        """Add multiple tool definitions from functions, auto-deriving metadata."""

    def prompt_conversation(self, message: str = None, **kwargs) -> Tuple[List[Message], LLMPromptBuilder]:
        """Execute conversation with tool calling support.
        
        Similar to prompt_for_text but supports tool calling and returns conversation state.
        Only supports text messages - no image, audio, or structured output support.
        
        Args:
            message: Optional text message to add to conversation
            **kwargs: Additional arguments passed to the provider API
        
        Returns:
            Tuple of (message_list, continuation_builder) where message_list
            contains all messages including tool calls and responses.
        """
```

### Provider Interface Extensions

```python
class LLMProvider:
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        
    async def prompt_with_tools(
        self, 
        model: str, 
        p: Prompt, 
        tools: List[Tool],
        **kwargs
    ) -> Any:
        """Execute prompt with tool calling support."""
```

## Data Models

### Message Extensions

New message types to support tool calling:

```python
@dataclass
class ToolCallMessage(Message):
    """Message representing a tool call made by the AI."""
    tool_name: str
    tool_call_id: str
    arguments: dict
    role: MessageRole = MessageRole.ASSISTANT

@dataclass  
class ToolResultMessage(Message):
    """Message representing the result of a tool call."""
    tool_call_id: str
    result: Any
    role: MessageRole = MessageRole.USER
```

### Internal Conversation State

The builder maintains internal conversation state with Anthropic API-level messages:

```python
@dataclass
class ConversationState:
    api_messages: List[dict]  # Anthropic API format messages (includes user messages)
    tools: List[Tool]
    message_history: MessageList  # Our abstract message format
```



### Tool Schema Generation

The system will use Pydantic's `TypeAdapter` to generate JSON schemas from function signatures:

```python
from pydantic import TypeAdapter
from typing import get_type_hints
import inspect

def generate_tool_schema(func: Callable) -> dict:
    """Generate JSON schema from function signature."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Build parameter schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name in type_hints:
            param_type = type_hints[param_name]
            adapter = TypeAdapter(param_type)
            properties[param_name] = adapter.json_schema()
            
            # Check if parameter is required by looking at type annotation
            # Optional[T] or Union[T, None] indicates optional parameter
            origin = getattr(param_type, '__origin__', None)
            if origin is Union:
                args = getattr(param_type, '__args__', ())
                is_optional = type(None) in args
            else:
                is_optional = False
                
            # Also check for default values as fallback
            has_default = param.default != inspect.Parameter.empty
            
            if not is_optional and not has_default:
                required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }
```



## Error Handling

### Tool Validation Errors
- Invalid function signatures
- Missing type annotations
- Unsupported parameter types

### Runtime Errors
- Tool execution failures
- Provider API errors
- Tool call parsing errors

### Provider Compatibility
- Clear error messages when tools are used with unsupported providers
- Graceful degradation when tool calling is not available

### Conversation Limitations
- Text-only conversations (no image, audio, or structured output support initially)
- Simple message passing interface similar to prompt_for_text
- Focus on tool calling functionality rather than multimodal capabilities

## Testing Strategy

### Example Test Scenario

```python
def test_tool_calling_conversation():
    """Test complete tool calling conversation flow."""
    
    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    messages, continuation = await (
        llm()
        .agent("You are a helpful assistant")
        .tools([get_weather])  # Just pass the function
        .prompt_conversation("What's the weather in Paris?")
    )
    
    # Should contain user message, tool call, tool result, and AI response
    assert len(messages) >= 3
    assert any(isinstance(msg, ToolCallMessage) for msg in messages)
    assert any(isinstance(msg, ToolResultMessage) for msg in messages)
    
    # Test conversation continuation
    final_messages, _ = await continuation.prompt_conversation(
        "What about London?"
    )
    
    assert any("London" in str(msg) for msg in final_messages)
```

## Implementation Notes

### Pydantic Integration
- Leverage existing Pydantic dependency for schema generation
- Use `TypeAdapter` for converting Python types to JSON schemas
- Support for complex types (List, Dict, Optional, Union)

### Model Selection Strategy
- Default model selection strategy must prefer Anthropic when tools are present
- When `prompt_conversation()` is used, automatically select Anthropic provider
- Clear error messages when tools are used with unsupported providers

### Anthropic Specifics
- Tool calls have `stop_reason` of "tool_use"
- Tool results must be sent back in specific format
- Multiple tool calls can occur in single response

### Future OpenAI Support
- OpenAI uses different tool calling format
- Function calling vs tool calling terminology
- Different response structure and continuation patterns

### Type Safety
- Full typing for tool definitions and responses
- Generic types for tool function parameters
- Proper return type annotations for conversation methods
- Message type hierarchy with tool-specific message types