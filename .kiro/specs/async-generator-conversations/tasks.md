# Implementation Plan

- [x] 1. Refactor conversation handling to use async generator pattern
  - [x] 1.1 Update ToolCallMessage to separate result and error fields
    - Modify ToolCallMessage dataclass to have separate `result: Any | None` and `error: Exception | None` fields
    - Update `__str__` method to handle error display appropriately
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 1.2 Refactor prompt_conversation() to remove message parameter and return async generator
    - Remove the `message: str | None = None` parameter from `prompt_conversation()` method signature
    - Change return type from tuple to `AsyncGenerator[Message, None]`
    - Update method docstring to reflect new async generator behavior
    - _Requirements: 2.1, 2.2, 2.3, 1.1_

  - [x] 1.3 Implement async generator logic for conversation flow
    - Create the async generator implementation that yields individual messages
    - Handle conversation state initialization using existing `_get_or_create_conversation_state()` method
    - Implement main conversation loop with API calls and response processing
    - _Requirements: 1.2, 1.3, 1.4_

  - [x] 1.4 Update tool execution to return result/error tuple
    - Modify `_execute_tool_call()` method to return `tuple[Any | None, Exception | None]`
    - Update tool call processing to handle the new tuple return format
    - Create ToolCallMessage instances with separate result and error fields
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 1.5 Implement StopAsyncIteration with continuation builder
    - Return continuation builder via StopAsyncIteration exception value when conversation completes
    - Ensure proper async generator protocol compliance
    - Handle generator cleanup and state management
    - _Requirements: 1.4, 6.2, 6.3_

  - [x] 1.6 Add prompt_agentically() method for automatic processing
    - Create new `prompt_agentically(max_calls: int, **kwargs)` method
    - Implement automatic iteration over the async generator up to max_calls limit
    - Return tuple of (complete_message_list, continuation_builder) for backward compatibility
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 1.7 Update conversation state management for generator support
    - Enhance ConversationState dataclass with any needed fields for generator tracking
    - Ensure proper state preservation across generator yields
    - Update internal session management for provider compatibility
    - _Requirements: 1.2, 1.3, 6.2_

  - [x] 1.8 Update provider integration for unified tool call messages
    - Modify provider response processing to work with new ToolCallMessage structure
    - Ensure tool call detection and execution flow works with async generator
    - Update Anthropic provider to handle the new message format
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 1.9 Create comprehensive unit tests for async generator functionality
    - Write test for manual async generator iteration using `__anext__()`
    - Write test for `async for` loop iteration pattern
    - Write test for ToolCallMessage structure with result/error separation
    - Write test for prompt_agentically() compatibility and behavior
    - Write test for StopAsyncIteration handling and continuation builder access
    - Write test to verify prompt_conversation() no longer accepts message parameter
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.4_

  - [x] 1.10 Create live integration test matching the example code
    - Create test file that demonstrates the exact usage pattern from example.py
    - Test both manual iteration and for-loop patterns
    - Verify tool calling works correctly with the new async generator approach
    - Test continuation builder functionality for follow-up conversations
    - _Requirements: 5.1, 5.2, 5.3, 6.5_