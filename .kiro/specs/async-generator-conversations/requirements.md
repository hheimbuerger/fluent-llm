# Requirements Document

## Introduction

This feature refactors the existing conversation handling in the fluent LLM library to use an async generator pattern instead of the current tuple-based approach. The new design provides a more intuitive and Pythonic way to handle multi-turn conversations with tool calling, where each yield returns individual messages and tool calls, allowing for fine-grained control over conversation flow.

## Requirements

### Requirement 1

**User Story:** As a developer using the fluent LLM library, I want to iterate through conversation messages one by one using an async generator, so that I can have fine-grained control over the conversation flow and handle each message individually.

#### Acceptance Criteria

1. WHEN I call `prompt_conversation()` THEN the system SHALL return an async generator that yields individual messages
2. WHEN I iterate through the generator THEN each yield SHALL return a single message (TextMessage, ToolCallMessage, etc.)
3. WHEN I use `yield conversation` THEN I SHALL receive the next message in the conversation flow
4. WHEN the conversation is complete THEN the generator SHALL raise StopIteration
5. WHEN I iterate with a for-loop THEN the system SHALL support `async for` syntax

### Requirement 2

**User Story:** As a developer, I want the prompt_conversation() method to not accept text messages as parameters, so that the API is consistent with the builder pattern and doesn't mix concerns.

#### Acceptance Criteria

1. WHEN I call `prompt_conversation()` THEN the system SHALL NOT accept a text message parameter
2. WHEN I want to add a message THEN I SHALL use the existing `.request()` builder method
3. WHEN I call `prompt_conversation()` without parameters THEN the system SHALL execute the conversation with existing messages
4. WHEN I need to continue a conversation THEN I SHALL use the returned LLMBuilder instance with additional `.request()` calls
5. WHEN I use the old signature THEN the system SHALL raise a clear deprecation or error message

### Requirement 3

**User Story:** As a developer, I want tool calls, their associated assistant messages, and results to be combined into a single ToolCallMessage instance, so that I can handle tool calling as atomic operations.

#### Acceptance Criteria

1. WHEN the AI makes a tool call THEN the system SHALL create a single ToolCallMessage containing the call, arguments, result and potential error (an exception instance)
2. WHEN I receive a ToolCallMessage THEN it SHALL include the tool name, arguments, execution result, and any associated text
3. WHEN a tool call fails THEN the ToolCallMessage SHALL contain error information in the error field
4. WHEN multiple tools are called THEN each SHALL be yielded as a separate ToolCallMessage
5. WHEN I iterate through messages THEN I SHALL NOT receive separate ToolResultMessage instances

### Requirement 4

**User Story:** As a developer, I want to maintain the existing prompt_agentically() method for automatic sequential processing, so that I can choose between manual iteration and automatic processing based on my use case.

#### Acceptance Criteria

1. WHEN I call `prompt_agentically(max_calls)` THEN the system SHALL automatically process all tool calls up to the maximum, relying on the same code path that powers prompt_conversation() internally
2. WHEN I use `prompt_agentically()` THEN the system SHALL return a tuple of (responses, continuation_builder)
3. WHEN I specify max_calls THEN the system SHALL stop after that many iterations to prevent infinite loops
4. WHEN I need automatic processing THEN I SHALL use `prompt_agentically()` instead of manual iteration
5. WHEN both methods exist THEN they SHALL have different names due to different return types

### Requirement 5

**User Story:** As a developer, I want to validate that the async generator approach works correctly with tool calling, so that I can be confident in the new conversation flow.

#### Acceptance Criteria

1. WHEN I run the test suite THEN there SHALL be a unit test demonstrating async generator iteration, replacing the old unit test then obsolete
2. WHEN the test runs THEN it SHALL verify that individual messages are yielded correctly
3. WHEN the test runs THEN it SHALL verify that ToolCallMessage instances contain complete information
4. WHEN the test runs THEN it SHALL verify that for-loop iteration works with `async for`
5. WHEN the test runs THEN it SHALL verify that StopIteration is raised when conversation completes

### Requirement 6

**User Story:** As a developer, I want the async generator approach to be technically sound and follow Python best practices, so that the code is maintainable and follows expected patterns.

#### Acceptance Criteria

1. WHEN I use the async generator THEN it SHALL follow proper Python async generator syntax and semantics
2. WHEN I yield from the generator THEN the system SHALL properly handle async context and state
3. WHEN I use try/except with StopIteration THEN the system SHALL behave according to Python generator protocols
4. WHEN I access the continuation builder THEN it SHALL be available after generator completion
5. WHEN I use the API THEN it SHALL be compatible with standard Python async iteration patterns