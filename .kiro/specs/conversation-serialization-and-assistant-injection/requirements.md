# Requirements Document

## Introduction

This feature implements assistant message injection and conversation serialization through a three-class architecture: MessageList (serializable data), LLMConversation (mutable execution context), and LLMPromptBuilder (immutable composition tool with conversation reference and delta pattern). The design eliminates state duplication while enabling flexible conversation management and model-agnostic serialization.

## Requirements

### Requirement 1: Assistant Message Injection API

**User Story:** As a developer using fluent_llm, I want to inject assistant messages into my conversation builder chain, so that I can prime conversations with example exchanges or continue from a specific conversation state.

#### Acceptance Criteria

1. WHEN I call `.assistant(text)` on an LLMPromptBuilder THEN the system SHALL add a TextMessage with Role.ASSISTANT to the builder's delta messages
2. WHEN I chain multiple `.assistant()` calls THEN the system SHALL preserve the order of all messages (user, agent, assistant) in the delta
3. WHEN I use `.assistant()` in combination with existing methods like `.agent()`, `.context()`, `.request()` THEN all messages SHALL be properly ordered in the delta
4. WHEN I call `.assistant()` with empty or whitespace-only text THEN the system SHALL strip whitespace and add the message
5. WHEN the builder applies deltas to a conversation THEN all composed messages including assistant messages SHALL be added to the conversation's MessageList

### Requirement 2: Three-Class Architecture with Clear Separation

**User Story:** As a developer, I want a clean architecture where MessageList handles data, LLMConversation handles execution, and LLMPromptBuilder handles composition, so that there's no duplication and clear responsibility separation.

#### Acceptance Criteria

1. WHEN I work with MessageList THEN it SHALL only contain message data and serialization methods
2. WHEN I work with LLMConversation THEN it SHALL own the complete MessageList and handle execution
3. WHEN I work with LLMPromptBuilder THEN it SHALL accumulate deltas (new messages + config changes) without duplicating conversation state
4. WHEN I create any of these classes THEN there SHALL be no state duplication between them
5. WHEN I access message history THEN it SHALL exist only in the conversation's MessageList
6. WHEN I need execution configuration THEN it SHALL exist only on the conversation

### Requirement 3: Delta Pattern for Immutable Builders

**User Story:** As a developer, I want builders to accumulate changes as deltas and apply them on execution, so that I can have immutable composition with mutable execution targets.

#### Acceptance Criteria

1. WHEN I call builder methods THEN they SHALL return new builder instances with updated deltas
2. WHEN I chain builder methods THEN each SHALL accumulate additional delta messages or config changes
3. WHEN I call `.prompt_conversation()` THEN the builder SHALL apply all deltas to the referenced conversation
4. WHEN I reuse a builder THEN the original SHALL be unchanged (immutable)
5. WHEN deltas are applied THEN new messages SHALL be appended to conversation.messages and config changes SHALL update conversation config

### Requirement 4: Conversation Reference in Builders

**User Story:** As a developer, I want continuation builders to automatically reference their source conversation, so that I don't need to manually pass conversation objects around.

#### Acceptance Criteria

1. WHEN I access `conversation.continuation` THEN it SHALL return a builder that references the source conversation
2. WHEN I use a continuation builder THEN calling `.prompt_conversation()` SHALL automatically apply deltas to the referenced conversation
3. WHEN I chain methods on a continuation builder THEN the conversation reference SHALL be preserved
4. WHEN I create a fresh builder (not from continuation) THEN it SHALL create a new conversation on first execution
5. WHEN I use continuation builders THEN I SHALL NOT need to manually pass conversation objects

### Requirement 5: Mutability Strategy

**User Story:** As a developer, I want a clear mutability model where MessageList and Conversation are mutable for execution needs, while Builder remains immutable for composition safety.

#### Acceptance Criteria

1. WHEN I use MessageList THEN it SHALL be mutable and grow as messages are added during execution
2. WHEN I use LLMConversation THEN it SHALL be mutable and accumulate state during async iteration
3. WHEN I use LLMPromptBuilder THEN it SHALL be immutable and each method SHALL return a new instance
4. WHEN I iterate over a conversation THEN the conversation's MessageList SHALL grow with new messages
5. WHEN I access `conversation.continuation` at any time THEN it SHALL reflect the current mutable state
6. WHEN I reuse builders THEN immutability SHALL prevent contamination between uses

### Requirement 6: MessageList-Only Serialization

**User Story:** As a developer, I want to serialize only the MessageList without configuration or tools, so that I have model-agnostic conversation persistence that works across different setups.

#### Acceptance Criteria

1. WHEN I call `message_list.to_dict()` THEN the system SHALL return a dictionary containing only message data
2. WHEN I call `MessageList.from_dict(data)` THEN the system SHALL reconstruct a MessageList from the serialized data
3. WHEN I serialize conversations THEN configuration (provider, model, tools) SHALL NOT be included
4. WHEN I deserialize a MessageList THEN I SHALL be able to create conversations with any configuration
5. WHEN I serialize/deserialize THEN all message types SHALL be preserved with complete state
6. WHEN I want to save to file THEN I SHALL be able to use `json.dump(conversation.messages.to_dict(), file)`

### Requirement 7: Flexible Conversation Restoration

**User Story:** As a developer, I want to restore conversations from serialized MessageList and continue them with any configuration, so that I have maximum flexibility in session restoration.

#### Acceptance Criteria

1. WHEN I restore a MessageList THEN I SHALL be able to create a new LLMConversation with it
2. WHEN I create a conversation from restored MessageList THEN I SHALL be able to configure it with any provider, model, or tools
3. WHEN I access continuation on a restored conversation THEN it SHALL work identically to original conversations
4. WHEN I continue a restored conversation THEN I SHALL NOT need to know anything about the original configuration
5. WHEN I serialize and restore conversations THEN the behavior SHALL be identical to the original

### Requirement 8: Early Continuation Access

**User Story:** As a developer, I want to access continuation builders at any point during conversation execution, so that I can abort async iteration and continue with new requests.

#### Acceptance Criteria

1. WHEN I access `conversation.continuation` before starting iteration THEN it SHALL return a builder with the current MessageList state
2. WHEN I access `conversation.continuation` during iteration THEN it SHALL return a builder reflecting all messages received so far
3. WHEN I stop iterating early and use continuation THEN I SHALL be able to make new requests with full conversation history
4. WHEN I access continuation multiple times THEN each SHALL reflect the current conversation state
5. WHEN the conversation is mutated by async iteration THEN continuation builders SHALL automatically reflect the updated state

### Requirement 9: Unified Execution Model

**User Story:** As a developer, I want all LLM interactions to go through the conversation execution model internally, so that there's consistency between one-shot and multi-turn interactions.

#### Acceptance Criteria

1. WHEN I call `.prompt()` for one-shot responses THEN the system SHALL internally create a conversation and extract the final result
2. WHEN I call `.prompt_conversation()` THEN the system SHALL return the conversation for multi-turn interaction
3. WHEN I use specialized methods like `.prompt_for_image()` THEN they SHALL use the same conversation execution model
4. WHEN I compare one-shot and multi-turn approaches THEN they SHALL use identical underlying execution mechanisms
5. WHEN any execution occurs THEN it SHALL go through the conversation's async iteration system

### Requirement 10: Configuration Delta Management

**User Story:** As a developer, I want builders to handle configuration changes as deltas, so that I can modify provider, model, and tools without duplicating configuration state.

#### Acceptance Criteria

1. WHEN I call `.provider()`, `.model()`, or `.tools()` on a builder THEN it SHALL accumulate config deltas
2. WHEN I apply builder deltas THEN config changes SHALL update the conversation's configuration
3. WHEN I chain configuration methods THEN deltas SHALL accumulate properly
4. WHEN I use continuation builders THEN they SHALL be able to modify the conversation's configuration via deltas
5. WHEN configuration deltas are applied THEN the conversation SHALL use the updated configuration for subsequent API calls

### Requirement 11: Convenience Load/Save Methods

**User Story:** As a developer, I want convenient methods to load conversations from files/streams and save them, so that I can easily persist and restore conversation state without manual JSON handling.

#### Acceptance Criteria

1. WHEN I call `builder.load_conversation(filename)` with a string THEN it SHALL load MessageList from the file and create a conversation
2. WHEN I call `builder.load_conversation(stream)` with an IO stream THEN it SHALL load MessageList from the stream and create a conversation  
3. WHEN I call `builder.load_conversation(path_obj)` with a Path object THEN it SHALL load MessageList from the path and create a conversation
4. WHEN I call `builder.load_conversation(dict_data)` with a dictionary THEN it SHALL load MessageList from the dict and create a conversation
5. WHEN I call `conversation.save(filename)` with a string THEN it SHALL save the MessageList to the file using JSON
6. WHEN I call `conversation.save(stream)` with an IO stream THEN it SHALL save the MessageList to the stream using JSON
7. WHEN I call `conversation.save(path_obj)` with a Path object THEN it SHALL save the MessageList to the path using JSON
8. WHEN save/load operations fail THEN they SHALL provide clear error messages about file access or data format issues

### Requirement 12: Error Handling and Validation

**User Story:** As a developer, I want clear error messages and validation for the new architecture, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. WHEN I try to deserialize invalid MessageList data THEN the system SHALL raise a descriptive error indicating what is invalid
2. WHEN serialization fails THEN the error message SHALL indicate which specific message cannot be serialized
3. WHEN I use assistant message injection incorrectly THEN the system SHALL provide helpful error messages
4. WHEN delta application fails THEN the system SHALL provide clear guidance on the issue
5. WHEN I access continuation on a malformed conversation THEN the system SHALL provide actionable error messages