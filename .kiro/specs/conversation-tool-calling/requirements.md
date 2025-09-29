# Requirements Document

## Introduction

This feature introduces conversation support and tool calling capabilities to the fluent LLM library, initially for the Anthropic provider. The feature enables multi-turn conversations with AI models that can call external tools/functions, with full type safety and a fluent API design that maintains the library's existing patterns.

## Requirements

### Requirement 1

**User Story:** As a developer using the fluent LLM library, I want to define tools that the AI can call, so that I can create interactive applications where the AI can perform actions beyond text generation.

#### Acceptance Criteria

1. WHEN I use a new `tools()` builder method THEN the system SHALL accept a list of tool definitions
2. WHEN I define a tool THEN the system SHALL require a name, description, and function reference
3. WHEN I use typed Python functions for tools THEN the system SHALL automatically derive JSON schemas from type annotations
4. WHEN I have tools defined THEN existing prompt methods SHALL fail with a clear error message
5. WHEN I have tools defined THEN I SHALL be required to use the new `prompt_conversation()` method

### Requirement 2

**User Story:** As a developer, I want to have conversations with AI models that can span multiple turns, so that I can build interactive applications with context preservation.

#### Acceptance Criteria

1. WHEN I call `prompt_conversation()` THEN the system SHALL return a tuple containing responses and a pre-configured builder
2. WHEN I receive the pre-configured builder THEN I SHALL be able to continue the conversation without losing context
3. WHEN the AI makes tool calls THEN the system SHALL automatically handle the tool execution flow
4. WHEN tool calls are made THEN the system SHALL map them against known tools and execute the appropriate functions
5. WHEN a tool call fails THEN the system SHALL handle errors gracefully and continue the conversation

### Requirement 3

**User Story:** As a developer, I want full type safety for tool definitions and conversation flows, so that I can catch errors at development time rather than runtime.

#### Acceptance Criteria

1. WHEN I define tools THEN all tool parameters SHALL be fully typed
2. WHEN I use conversation methods THEN return types SHALL be properly typed tuples
3. WHEN I continue conversations THEN the builder SHALL maintain proper typing
4. WHEN I define tool functions THEN their signatures SHALL be validated against their schemas
5. WHEN I use the API THEN TypeScript-style type hints SHALL be available in Python IDEs

### Requirement 4

**User Story:** As a developer, I want the tool calling feature to work specifically with Anthropic initially, so that I can start using this functionality while the team plans OpenAI support.

#### Acceptance Criteria

1. WHEN I use tools with Anthropic provider THEN the system SHALL properly format tool definitions for Anthropic's API
2. WHEN Anthropic returns tool calls THEN the system SHALL parse and execute them correctly
3. WHEN using other providers with tools THEN the system SHALL raise a clear "not supported" error
4. WHEN tool calls have stop_reason of "tool_use" THEN the system SHALL handle the Anthropic-specific format
5. WHEN continuing conversations THEN the system SHALL maintain Anthropic's conversation format requirements

### Requirement 5

**User Story:** As a developer, I want to validate that tool calling works correctly, so that I can be confident in my implementation.

#### Acceptance Criteria

1. WHEN I run the test suite THEN there SHALL be a unit test demonstrating tool calling with conversations
2. WHEN the test runs THEN it SHALL verify that tools are called correctly
3. WHEN the test runs THEN it SHALL verify that conversation state is maintained
4. WHEN the test runs THEN it SHALL verify that responses are properly formatted
5. WHEN the test runs THEN it SHALL verify error handling for invalid tool calls