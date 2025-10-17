# Design Document

## Overview

This design implements a three-class architecture for assistant message injection and conversation serialization in fluent_llm. The architecture eliminates state duplication through a delta pattern where immutable builders accumulate changes and apply them to mutable conversations that own the complete message state.

### Mental Model

```
MessageList (Mutable Data Container)
    ↑ owned by
LLMConversation (Mutable Execution Context)
    ↑ referenced by
LLMPromptBuilder (Immutable Composition Tool)
    ↓ accumulates deltas
    ↓ applies to conversation on execution
```

**Key Principles:**
- **Single Source of Truth**: Messages exist only in MessageList
- **Delta Pattern**: Builders accumulate changes, apply on execution
- **Clear Mutability**: MessageList/Conversation mutable, Builder immutable
- **Reference-Based Continuation**: Builders reference conversations directly

## Architecture

### Core Classes

#### 1. MessageList (Mutable Data Container)

```python
class MessageList:
    """Serializable message container - the single source of truth for conversation data."""
    
    def __init__(self, messages: list[Message] = None):
        self._messages: list[Message] = messages or []
    
    def append(self, message: Message) -> None:
        """Add message to the list (mutable operation)."""
        self._messages.append(message)
    
    def extend(self, messages: list[Message]) -> None:
        """Add multiple messages (mutable operation)."""
        self._messages.extend(messages)
    
    def copy(self) -> 'MessageList':
        """Create a copy of this MessageList."""
        return MessageList([msg for msg in self._messages])
    
    def __iter__(self):
        return iter(self._messages)
    
    def __len__(self):
        return len(self._messages)
    
    def __getitem__(self, index):
        return self._messages[index]
    
    # Serialization (only on MessageList)
    def to_dict(self) -> dict:
        """Serialize messages to dictionary."""
        return {
            "messages": [msg.to_serializable_dict() for msg in self._messages],
            "version": "1.0"
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageList':
        """Deserialize messages from dictionary."""
        if data.get("version") != "1.0":
            raise MessageListDeserializationError(f"Unsupported version: {data.get('version')}")
        
        messages = []
        for msg_data in data["messages"]:
            msg_type = msg_data["type"]
            if msg_type == "TextMessage":
                messages.append(TextMessage.from_serializable_dict(msg_data))
            elif msg_type == "AgentMessage":
                messages.append(AgentMessage.from_serializable_dict(msg_data))
            elif msg_type == "ToolCallMessage":
                messages.append(ToolCallMessage.from_serializable_dict(msg_data))
            else:
                raise MessageListDeserializationError(f"Unknown message type: {msg_type}")
        
        return cls(messages)
```

#### 2. LLMConversation (Mutable Execution Context)

```python
@dataclass
class ConversationConfig:
    """Configuration for conversation execution."""
    preferred_provider: str | None = None
    preferred_model: str | None = None
    model_selector: ModelSelectionStrategy = field(default_factory=DefaultModelSelectionStrategy)
    tools: list[Tool] = field(default_factory=list)

class LLMConversation:
    """Mutable execution context - owns MessageList and handles API calls."""
    
    def __init__(self, messages: MessageList = None, config: ConversationConfig = None):
        # State ownership
        self.messages: MessageList = messages or MessageList()  # Single source of truth
        self._config: ConversationConfig = config or ConversationConfig()
        
        # Execution state
        self._generator = None
        self._kwargs = {}
    
    @property
    def continuation(self) -> 'LLMPromptBuilder':
        """Create builder that references this conversation."""
        return LLMPromptBuilder(conversation=self)
    
    def apply_config_deltas(self, delta_config: dict) -> None:
        """Apply configuration changes (mutable operation)."""
        for key, value in delta_config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
    
    # Async iterator protocol
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._generator is None:
            self._generator = self._create_generator()
        
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            raise
    
    async def _create_generator(self):
        """Execute conversation and yield messages."""
        while True:
            # Build prompt from current messages
            p = Prompt(
                messages=self.messages,  # Use our MessageList directly
                expect_type=ResponseType.TEXT,
                preferred_provider=self._config.preferred_provider,
                preferred_model=self._config.preferred_model,
                tools=self._config.tools,
                is_conversation=True,
            )
            
            # Select model and make API call
            provider, model = self._config.model_selector.select_model(p)
            provider.check_capabilities(model, p)
            
            response = await provider.prompt_via_api(
                model=model,
                p=p,
                conversation_state=None,  # We manage our own state
                **self._kwargs
            )
            
            # Process response and update our MessageList
            if isinstance(response, dict) and response.get('tool_calls'):
                # Handle tool calls
                for tool_call in response['tool_calls']:
                    result, error = self._execute_tool_call(
                        tool_call["name"], 
                        tool_call.get("input", tool_call.get("arguments", {}))
                    )
                    
                    tool_message = ToolCallMessage(
                        message=response.get('text', ""),
                        tool_name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        arguments=tool_call.get("input", tool_call.get("arguments", {})),
                        result=result,
                        error=error
                    )
                    
                    self.messages.append(tool_message)  # Mutate our MessageList
                    yield tool_message
                
                continue  # Continue conversation loop
            else:
                # Final text response
                if isinstance(response, dict):
                    text_content = response.get('text', str(response))
                else:
                    text_content = str(response)
                
                text_message = TextMessage(text=text_content, role=Role.ASSISTANT)
                self.messages.append(text_message)  # Mutate our MessageList
                yield text_message
                return  # End conversation
    
    def _execute_tool_call(self, tool_name: str, arguments: dict) -> tuple[Any | None, Exception | None]:
        """Execute tool call using configured tools."""
        for tool in self._config.tools:
            if tool.name == tool_name:
                try:
                    result = tool.function(**arguments)
                    return result, None
                except Exception as e:
                    return None, e
        
        return None, ValueError(f"Tool '{tool_name}' not found")
```

#### 3. LLMPromptBuilder (Immutable Composition Tool)

```python
class LLMPromptBuilder:
    """Immutable composition tool with conversation reference and delta pattern."""
    
    def __init__(self, conversation: LLMConversation = None, delta_messages: MessageList = None, delta_config: dict = None):
        self._conversation: LLMConversation = conversation or LLMConversation()
        self._delta_messages: MessageList = delta_messages or MessageList()
        self._delta_config: dict = delta_config or {}
    
    def _copy_with_delta_message(self, message: Message) -> 'LLMPromptBuilder':
        """Create new builder with additional delta message."""
        new_delta_messages = self._delta_messages.copy()
        new_delta_messages.append(message)
        return LLMPromptBuilder(self._conversation, new_delta_messages, self._delta_config)
    
    def _copy_with_delta_config(self, **config_updates) -> 'LLMPromptBuilder':
        """Create new builder with additional config deltas."""
        new_delta_config = {**self._delta_config, **config_updates}
        return LLMPromptBuilder(self._conversation, self._delta_messages, new_delta_config)
    
    # Message composition methods (all return new instances)
    def agent(self, text: str) -> 'LLMPromptBuilder':
        return self._copy_with_delta_message(AgentMessage(text=text.strip()))
    
    def assistant(self, text: str) -> 'LLMPromptBuilder':
        """NEW: Add assistant message to delta composition."""
        return self._copy_with_delta_message(TextMessage(text=text.strip(), role=Role.ASSISTANT))
    
    def context(self, text: str) -> 'LLMPromptBuilder':
        return self._copy_with_delta_message(TextMessage(text=text.strip(), role=Role.USER))
    
    def request(self, text: str) -> 'LLMPromptBuilder':
        return self._copy_with_delta_message(TextMessage(text=text.strip(), role=Role.USER))
    
    # Configuration methods (all return new instances)
    def provider(self, provider_name: str) -> 'LLMPromptBuilder':
        return self._copy_with_delta_config(preferred_provider=provider_name)
    
    def model(self, model_name: str) -> 'LLMPromptBuilder':
        return self._copy_with_delta_config(preferred_model=model_name)
    
    def tools(self, *tool_functions) -> 'LLMPromptBuilder':
        tools = [Tool.from_function(f) for f in tool_functions]
        current_tools = self._conversation._config.tools + [Tool.from_function(f) for f in tool_functions]
        return self._copy_with_delta_config(tools=current_tools)
    
    # Execution methods
    def prompt_conversation(self, **kwargs) -> LLMConversation:
        """Apply deltas to conversation and return it for execution."""
        # Apply delta messages
        self._conversation.messages.extend(self._delta_messages)
        
        # Apply delta config
        self._conversation.apply_config_deltas(self._delta_config)
        
        # Store execution kwargs
        self._conversation._kwargs.update(kwargs)
        
        return self._conversation
    
    @asyncify
    async def prompt(self, **kwargs) -> str:
        """One-shot execution via conversation."""
        conversation = self.prompt_conversation(**kwargs)
        
        # Execute conversation and extract final text response
        final_message = None
        async for message in conversation:
            final_message = message
        
        if isinstance(final_message, TextMessage) and final_message.role == Role.ASSISTANT:
            return final_message.text
        else:
            return str(final_message)
    
    @asyncify
    async def prompt_for_image(self, **kwargs) -> Any:
        """One-shot image execution via conversation."""
        # Similar pattern - create conversation, execute, extract image
        pass
    
    # Other one-shot methods follow same pattern...
```

## Data Flow Examples

### 1. Fresh Conversation Creation

```python
# Builder accumulates deltas (immutable)
builder1 = llm.agent("You are helpful")  # delta_messages: [AgentMessage]
builder2 = builder1.assistant("Hello!")  # delta_messages: [AgentMessage, TextMessage(assistant)]
builder3 = builder2.request("What is Python?")  # delta_messages: [AgentMessage, TextMessage(assistant), TextMessage(user)]

# Apply deltas to conversation
conversation = builder3.prompt_conversation()
# conversation.messages now has: [AgentMessage, TextMessage(assistant), TextMessage(user)]

# Execute and grow MessageList
async for message in conversation:
    print(f"Total messages: {len(conversation.messages)}")
    # 3 -> 4 -> 5 as API responses are added to conversation.messages
```

### 2. Conversation Continuation

```python
# Get continuation builder (references same conversation)
continuation = conversation.continuation
# continuation._conversation is the same object as conversation
# continuation._delta_messages is empty (fresh deltas)

# Compose new deltas
follow_up = continuation.request("Tell me more about functions")
# follow_up._delta_messages: [TextMessage(user)]

# Apply deltas to same conversation
updated_conversation = follow_up.prompt_conversation()
# updated_conversation is the SAME object as conversation
# conversation.messages now includes the new user message
```

### 3. Configuration Changes via Deltas

```python
# Change provider via delta
anthropic_continuation = conversation.continuation \
    .provider("anthropic") \
    .tools(get_weather) \
    .request("What's the weather?")

# Apply config deltas to conversation
conversation = anthropic_continuation.prompt_conversation()
# conversation._config.preferred_provider is now "anthropic"
# conversation._config.tools now includes get_weather
```

### 4. Serialization and Restoration

```python
# Serialize MessageList only
data = conversation.messages.to_dict()
json.dump(data, open('messages.json', 'w'))

# Restore MessageList
restored_messages = MessageList.from_dict(json.load(open('messages.json')))

# Create new conversation with restored messages
restored_conversation = LLMConversation(messages=restored_messages)

# Continue with any configuration
continuation = restored_conversation.continuation \
    .provider("anthropic") \
    .tools(my_tool) \
    .request("Continue the conversation")
```

## Components and Interfaces

### Message Serialization

Each message type implements serialization:

```python
class TextMessage(Message):
    def to_serializable_dict(self) -> dict:
        return {
            "type": "TextMessage",
            "text": self.text,
            "role": self.role.value
        }
    
    @classmethod
    def from_serializable_dict(cls, data: dict) -> 'TextMessage':
        return cls(
            text=data["text"],
            role=Role(data["role"])
        )
```

### Delta Application

```python
def apply_deltas_to_conversation(conversation: LLMConversation, delta_messages: MessageList, delta_config: dict):
    """Apply builder deltas to conversation."""
    # Apply message deltas
    conversation.messages.extend(delta_messages)
    
    # Apply config deltas
    conversation.apply_config_deltas(delta_config)
```

## Data Models

### Serialization Format (MessageList Only)

```python
{
    "messages": [
        {
            "type": "AgentMessage",
            "text": "You are a helpful assistant",
            "role": "system"
        },
        {
            "type": "TextMessage",
            "text": "Hello! How can I help you today?",
            "role": "assistant"
        },
        {
            "type": "TextMessage",
            "text": "What's the weather in Paris?",
            "role": "user"
        },
        {
            "type": "ToolCallMessage",
            "message": "I'll check the weather for you",
            "tool_name": "get_weather",
            "tool_call_id": "call_123",
            "arguments": {"location": "Paris"},
            "result": "Sunny, 22°C",
            "error": null,
            "role": "assistant"
        }
    ],
    "version": "1.0"
}
```

### Delta Structures

```python
# Delta messages (accumulated in builder)
delta_messages = MessageList([
    TextMessage("Follow up question", Role.USER),
    TextMessage("Another message", Role.USER)
])

# Delta config (accumulated in builder)
delta_config = {
    "preferred_provider": "anthropic",
    "tools": [get_weather_tool, calculate_tool]
}
```

## Error Handling

### Custom Exceptions

```python
class MessageListDeserializationError(Exception):
    """Raised when MessageList data is invalid."""
    pass

class DeltaApplicationError(Exception):
    """Raised when deltas cannot be applied to conversation."""
    pass

class ConversationConfigurationError(Exception):
    """Raised when conversation configuration is invalid."""
    pass
```

## Testing Strategy

### Unit Tests

1. **MessageList Serialization**
   - Test round-trip for all message types
   - Test version handling and error cases
   - Test mutability operations (append, extend)

2. **Builder Immutability and Deltas**
   - Test all methods return new instances
   - Test delta accumulation for messages and config
   - Test original builders remain unchanged

3. **Conversation Mutability**
   - Test MessageList growth during execution
   - Test config delta application
   - Test continuation builder creation

4. **Delta Application**
   - Test message deltas applied correctly
   - Test config deltas applied correctly
   - Test error handling for invalid deltas

### Integration Tests

1. **End-to-End Flow**
   - Test builder → conversation → continuation flow
   - Test serialization → restoration → continuation
   - Test early continuation access during iteration

2. **Configuration Management**
   - Test config deltas override conversation config
   - Test tool registration via deltas
   - Test provider/model changes via deltas

### Live API Tests

1. **Real Conversation Flows**
   - Test with actual API calls and message accumulation
   - Test serialization with real conversation data
   - Test cross-model continuation with restored conversations

## Migration Strategy

### Breaking Changes (Acceptable)

1. **ConversationGenerator/ConversationState Removal**
   - Replace with new three-class architecture
   - Update all execution methods to use new pattern

2. **Builder Method Changes**
   - `prompt_conversation()` now returns just LLMConversation
   - All execution methods use conversation internally

### Implementation Phases

1. **Phase 1**: Implement MessageList with serialization
2. **Phase 2**: Implement new LLMConversation class
3. **Phase 3**: Implement delta-based LLMPromptBuilder
4. **Phase 4**: Add `.assistant()` method and continuation support
5. **Phase 5**: Update all execution methods to use conversations
6. **Phase 6**: Remove old classes and update integration

## Performance Considerations

### Memory Efficiency

- Single MessageList per conversation (no duplication)
- Delta accumulation minimizes copying
- Immutable builders share conversation references

### Execution Efficiency

- Direct MessageList mutation during execution
- Lazy delta application only on execution
- Efficient message type dispatch for serialization

## Security Considerations

### Serialization Safety

- Only serialize message data, never functions
- Validate all deserialized message data
- Version serialization format for compatibility
- Clear error messages for invalid data

### Delta Application Safety

- Validate config deltas before application
- Sanitize message content during delta creation
- Prevent unauthorized configuration changes