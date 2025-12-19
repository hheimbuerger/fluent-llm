# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-17

### Added

#### Three-Class Architecture
- **MessageList**: New mutable data container for conversation messages with serialization support
- **LLMConversation**: New mutable execution context that owns MessageList and handles API calls
- **LLMPromptBuilder**: Refactored to use delta pattern - immutable composition with conversation references

#### Assistant Message Injection
- `.assistant(text)` method on LLMPromptBuilder for injecting assistant messages
- Enables few-shot learning by providing example responses
- Supports conversation priming and restoration with specific assistant states
- Maintains proper message ordering with user, agent, and assistant messages

#### Conversation Serialization
- Model-agnostic serialization that only includes message data (no configuration)
- `conversation.save(destination)` method for saving conversations to files or streams
- `llm.load_conversation(source)` method for loading conversations from files, streams, or dicts
- Support for string paths, Path objects, IO streams, and dictionaries
- JSON-based serialization format with version handling
- Enables cross-model conversation continuation (start with OpenAI, continue with Anthropic)

#### Continuation System
- `conversation.continuation` property returns a builder that references the source conversation
- Automatic conversation reference preservation in continuation builders
- Early continuation access during async iteration
- Continuation builders reflect current mutable conversation state
- Seamless multi-turn conversation flow

#### Delta Pattern
- Builders accumulate changes as deltas (messages and config)
- Deltas applied to conversation on execution via `.prompt_conversation()`
- Immutable builder instances for composition safety
- Mutable conversation for execution efficiency
- Single source of truth for messages (no duplication)

#### Configuration Management
- `ConversationConfig` dataclass for execution configuration
- `apply_config_deltas()` method for updating conversation configuration
- Configuration changes via builder methods (`.provider()`, `.model()`, `.tools()`)
- Separate configuration from serialization for flexibility

#### Error Handling
- `MessageListDeserializationError` for invalid serialization data
- `DeltaApplicationError` for delta application failures
- `ConversationConfigurationError` for configuration issues
- Clear, actionable error messages

#### OpenAI Tool Calling Support
- Full tool calling support in OpenAI provider
- Tool definition conversion to OpenAI format
- Tool call response parsing and execution
- Tool result submission back to API
- Multi-turn conversation support with tools
- `supports_tools()` capability detection

### Changed

#### Breaking Changes
- **Removed `ConversationGenerator` class** - Use `LLMConversation` with async iteration
- **Removed `ConversationState` class** - State now managed by `LLMConversation`
- **Renamed `.start_conversation()` to `.prompt_conversation()`** - Returns `LLMConversation` directly
- **Renamed `.llm_continuation` to `.continuation`** - Property on `LLMConversation`
- **Serialization format changed** - Configuration no longer included (model-agnostic)

#### Architecture Improvements
- Single source of truth for messages (in `MessageList`)
- No state duplication between builder and conversation
- Clear mutability model: MessageList/Conversation mutable, Builder immutable
- Reference-based continuation system (no manual conversation passing)
- Unified execution model (all methods use conversation internally)

#### API Improvements
- All builder methods return new instances (immutability)
- Conversation owns complete message history
- Builders accumulate deltas and apply on execution
- Configuration changes via deltas
- Automatic conversation reference in continuations

### Migration Guide

#### Starting Conversations
**Before:**
```python
conversation = llm.request("Hello").start_conversation()
```

**After:**
```python
conversation = llm.request("Hello").prompt_conversation()
```

#### Accessing Continuations
**Before:**
```python
continuation_builder = conversation.llm_continuation
```

**After:**
```python
continuation_builder = conversation.continuation
```

#### Serialization
**Before:**
```python
import json
data = {"messages": [msg.to_dict() for msg in conversation.messages]}
json.dump(data, open("conv.json", "w"))
```

**After:**
```python
conversation.save("conv.json")
conversation = llm.load_conversation("conv.json")
```

### Fixed
- State duplication issues between builder and conversation
- Manual conversation state management complexity
- Configuration coupling with serialization
- Continuation builder creation complexity

### Documentation
- Comprehensive README updates with new architecture explanation
- API reference for all three core classes
- Examples for assistant message injection
- Examples for serialization and restoration
- Examples for continuation patterns
- Migration guide from old architecture
- Code examples for all public methods

## [1.0.0] - Previous Release

Initial release with basic functionality.

