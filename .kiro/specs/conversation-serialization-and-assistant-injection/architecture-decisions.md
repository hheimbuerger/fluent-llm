# Architecture Decisions Document

## Overview

This document captures the key architectural decisions made during the design of the conversation serialization and assistant injection feature for fluent_llm. It explains the reasoning behind our three-class architecture and the mutability choices that drive the system design.

## Core Goals

### Primary Goals
1. **Assistant Message Injection**: Enable developers to inject assistant messages into conversation chains for priming and continuation
2. **Conversation Serialization**: Allow saving and restoring conversation state across sessions
3. **Clean Architecture**: Eliminate duplication and create clear separation of concerns
4. **Flexible Continuation**: Support aborting async iteration and continuing with new requests

### Secondary Goals
1. **Immutable Builder Pattern**: Maintain functional programming patterns for composition
2. **Model Agnostic Serialization**: Serialized conversations should work with any provider/model
3. **Early Continuation Access**: Allow accessing continuation builders before async iteration completes

## Key Requirements That Shaped the Architecture

### Requirement 1: No State Duplication
**Problem**: Original design had both builders and conversations storing messages and configuration, leading to synchronization issues and unclear ownership.

**Solution**: Single source of truth - messages live only in MessageList, configuration lives only in conversations, builders work with deltas.

### Requirement 2: Async Iterator Mutability Problem
**Problem**: If conversations are immutable, async iteration becomes impossible because:
- User gets conversation copy from builder
- Async iteration needs to mutate conversation to add new messages
- User's continuation access would be from stale conversation state

**Solution**: Conversations must be mutable to support async iteration and real-time continuation access.

### Requirement 3: Builder Reusability
**Problem**: If builders are mutable, they can't be reused safely:
```python
builder = llm.agent("...")
conv1 = builder.request("Question 1").prompt_conversation()
conv2 = builder.request("Question 2").prompt_conversation()  # Would be contaminated
```

**Solution**: Builders must be immutable to support functional composition patterns.

### Requirement 4: Serialization Clarity
**Problem**: What should be serializable vs non-serializable was unclear, leading to "partial serialization" confusion.

**Solution**: Only MessageList is serializable. Tools, model selectors, and other configuration must be recreated on restoration.

### Requirement 5: Continuation Without Manual State Passing
**Problem**: Requiring users to manually pass conversation references breaks the fluent API:
```python
# Bad UX
continuation = conversation.continuation.request("...").prompt_conversation(conversation)
```

**Solution**: Builders hold conversation references and automatically apply deltas to the referenced conversation.

## Architectural Decisions

### Decision 1: Three-Class Architecture

**Classes**:
1. **MessageList**: Serializable data container
2. **LLMConversation**: Mutable execution context
3. **LLMPromptBuilder**: Immutable composition tool with conversation reference

**Rationale**: 
- Separates data (MessageList), execution (Conversation), and composition (Builder)
- Each class has single responsibility
- Clear serialization boundary (only MessageList)

### Decision 2: Mutability Strategy

**MessageList**: Mutable
- **Why**: Needs to grow during conversation execution
- **Impact**: Can be serialized at any point, represents current state

**LLMConversation**: Mutable  
- **Why**: Must accumulate messages during async iteration
- **Impact**: Long-lived sessions that grow over time, continuation always reflects current state

**LLMPromptBuilder**: Immutable
- **Why**: Enables functional composition and safe reuse
- **Impact**: Each method returns new instance, no side effects

### Decision 3: Delta Pattern for Builders

**Pattern**: Builders accumulate deltas (new messages + config changes) and apply them on execution

**Rationale**:
- Avoids duplication between builder and conversation
- Enables immutable builders with mutable conversation targets
- Clear separation between composition (deltas) and execution (application)

**Implementation**:
```python
class LLMPromptBuilder:
    def __init__(self, conversation, delta_messages, delta_config):
        self._conversation = conversation  # Reference to target
        self._delta_messages = delta_messages  # New messages to add
        self._delta_config = delta_config  # Config changes to apply
```

### Decision 4: Conversation Reference in Builders

**Decision**: Builders hold references to conversations rather than requiring manual passing

**Rationale**:
- Improves user experience - no need to pass conversation repeatedly
- Enables fluent continuation patterns
- Makes builder behavior predictable (always applies to same conversation)

**Trade-off**: Creates hidden state, but the UX benefit outweighs the complexity

### Decision 5: Configuration Storage Location

**Decision**: Configuration lives on conversations, not builders

**Rationale**:
- Conversations need config for execution (API calls)
- Builders work with config deltas, not full config
- Avoids duplication between builder and conversation
- Conversations are execution contexts, so they own execution config

### Decision 6: Serialization Scope

**Decision**: Only serialize MessageList, not configuration or tools

**Rationale**:
- Tools contain function objects (not serializable)
- Model selectors are complex objects (not serializable)  
- Configuration should be recreated for flexibility (model-agnostic restoration)
- MessageList contains all the conversational context needed

## Usage Patterns Enabled

### Pattern 1: Fresh Conversation Creation
```python
conversation = llm.agent("...").assistant("...").request("...").prompt_conversation()
```
- Builder composes deltas immutably
- Final call creates mutable conversation and applies deltas

### Pattern 2: Conversation Continuation
```python
continuation = conversation.continuation.request("...").prompt_conversation()
```
- Continuation builder references original conversation
- New deltas applied to same conversation object
- No manual state passing required

### Pattern 3: Early Continuation Access
```python
conversation = builder.prompt_conversation()
message1 = await conversation.__anext__()
# Stop iteration early, continue with new request
continuation = conversation.continuation.request("Follow up").prompt_conversation()
```
- Continuation always reflects current conversation state
- Can abort async iteration at any point

### Pattern 4: Serialization and Restoration
```python
# Serialize
data = conversation.messages.to_dict()

# Restore
messages = MessageList.from_dict(data)
restored_conversation = LLMConversation(messages)
continuation = restored_conversation.continuation.provider("anthropic").tools(my_tool)
```
- Only message history is preserved
- Configuration recreated on restoration
- Full flexibility in restoration setup

## Alternative Approaches Considered

### Alternative 1: Fully Immutable System
**Rejected because**: Async iteration requires mutation to accumulate messages. Immutable conversations would require complex state threading.

### Alternative 2: Fully Mutable System  
**Rejected because**: Builder reusability would be impossible. Functional composition patterns would break.

### Alternative 3: Configuration on Builders
**Rejected because**: Would require duplication between builder and conversation, or complex synchronization mechanisms.

### Alternative 4: Separate Execution Classes
**Rejected because**: Added complexity without clear benefits. Three classes already provide good separation.

## Key Insights

1. **Mutability is contextual**: Different classes need different mutability based on their role
2. **Delta patterns solve duplication**: Accumulating changes rather than duplicating state
3. **References enable fluent APIs**: Hidden state can be acceptable for UX benefits
4. **Serialization boundaries matter**: Clear rules about what can/cannot be serialized
5. **Single responsibility**: Each class should have one clear purpose

## Future Considerations

1. **Performance**: Delta application might need optimization for large conversations
2. **Memory**: Long-lived conversations will accumulate large message lists
3. **Concurrency**: Multiple builders referencing same conversation might need synchronization
4. **Versioning**: Serialization format should support future evolution

## Validation

This architecture successfully addresses all core requirements:
- ✅ Assistant message injection via `.assistant()` method
- ✅ Conversation serialization via MessageList
- ✅ Clean separation of concerns (no duplication)
- ✅ Flexible continuation patterns
- ✅ Immutable builder composition
- ✅ Model-agnostic serialization
- ✅ Early continuation access