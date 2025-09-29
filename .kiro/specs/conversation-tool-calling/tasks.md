# Implementation Plan

- [x] 1. Implement core tool calling infrastructure
  - [x] 1.1 Create tools module with Tool dataclass and schema generation
    - Create `src/fluent_llm/tools.py` module
    - Implement `Tool` dataclass with name, description, function, and schema fields
    - Implement `Tool.from_function()` class method to auto-derive metadata from functions
    - Implement `generate_tool_schema()` function using Pydantic TypeAdapter
    - Handle Optional types and Union types for determining required parameters
    - _Requirements: 1.2, 1.3_

  - [x] 1.2 Extend message types to support tool calling
    - Add `ToolCallMessage` class to `src/fluent_llm/messages.py`
    - Add `ToolResultMessage` class to `src/fluent_llm/messages.py`
    - Ensure both inherit from base Message class with proper role assignments
    - _Requirements: 2.1, 2.2_

  - [x] 1.3 Add tool methods to LLMPromptBuilder
    - Implement `tool()` method to add single tool from function
    - Implement `tools()` method to add multiple tools from function list
    - Add internal `_tools` list to store Tool instances
    - Validate that existing prompt methods fail when tools are present
    - _Requirements: 1.1, 1.4_

  - [x] 1.4 Implement prompt_conversation method in LLMPromptBuilder
    - Add `prompt_conversation()` method that accepts text message and kwargs
    - Return tuple of message list and continuation builder
    - Ensure method only works when tools are defined
    - Handle conversation state management internally
    - _Requirements: 1.5, 2.1, 2.2_

  - [x] 1.5 Implement conversation state management
    - Add conversation state tracking to builder
    - Maintain API-level messages for Anthropic format
    - Maintain abstract message history for return values
    - Handle tool call and result message injection
    - _Requirements: 2.3, 2.4_

- [ ] 2. Implement provider integration and testing
  - [ ] 2.1 Update model selection strategy for tool support
    - Modify `DefaultModelSelectionStrategy` in `src/fluent_llm/model_selector.py`
    - Add logic to prefer Anthropic provider when tools are present
    - Add logic to prefer Anthropic when `prompt_conversation()` is used
    - _Requirements: 4.1, 4.2_

  - [ ] 2.2 Extend Anthropic provider with tool calling support
    - Add `supports_tools()` method to `AnthropicProvider`
    - Implement tool calling in `prompt_via_api()` method
    - Handle tool call parsing from Anthropic responses
    - Handle tool execution and result formatting
    - Support multiple tool calls in single response
    - _Requirements: 4.1, 4.4, 4.5_

  - [ ] 2.3 Add error handling for tool calling
    - Handle tool execution failures gracefully
    - Provide clear errors for unsupported providers
    - Validate tool function signatures
    - Handle API errors during tool calling
    - _Requirements: 2.5, 4.3_

  - [ ] 2.4 Write comprehensive unit test for tool calling conversation
    - Create test with simple tool function (e.g., get_weather)
    - Test complete conversation flow with tool calls
    - Test conversation continuation
    - Verify message types and content
    - Test error scenarios
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_