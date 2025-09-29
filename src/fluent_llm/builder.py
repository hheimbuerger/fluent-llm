"""Fluent, async builder for Large Language Model (LLM) requests.

This builder lets you fluently construct an *abstract* prompt (vendor-agnostic)
and only converts it to OpenAI-specific message JSON inside ``call()``.

Example
-------
```python
from python_openai_sample import llm, ResponseType

response = await (
    llm()
        .agent("You are an art evaluator.")
        .context("Please review …")
        .file("painting.png")
        .expect(ResponseType.TEXT)
        .call()
)
print(response)
```
"""
from __future__ import annotations

import pathlib
from typing import Any, Sequence, Type
import inspect
from dataclasses import dataclass
from .utils import asyncify

from pydantic import BaseModel
from .messages import TextMessage, AudioMessage, ImageMessage, AgentMessage, ResponseType, MessageList, ToolCallMessage, ToolResultMessage
from .model_selector import ModelSelectionStrategy, DefaultModelSelectionStrategy
from .prompt import Prompt
from .tools import Tool
from fluent_llm import usage_tracker

__all__: Sequence[str] = [
    "llm",
]


_TEMP_last_provider = None


@dataclass
class ConversationState:
    """Internal conversation state for tool calling conversations."""
    api_messages: list[dict]  # Anthropic API format messages (includes user messages)
    tools: list[Tool]
    message_history: MessageList  # Our abstract message format


class LLMPromptBuilder:
    """Fluent, async builder for LLM requests."""

    def __init__(
        self,
        *,
        messages: MessageList | None = None,
        expect: ResponseType | None = None,
        model_selector: ModelSelectionStrategy | None = None,
        preferred_provider: str | None = None,
        preferred_model: str | None = None,
        tools: list[Tool] | None = None,
        conversation_state: ConversationState | None = None
    ) -> None:
        self._messages: MessageList = messages or MessageList()
        self._expect: ResponseType = expect or ResponseType.TEXT
        self._model_selector: ModelSelectionStrategy = model_selector or DefaultModelSelectionStrategy()
        self._preferred_provider: str | None = preferred_provider
        self._preferred_model: str | None = preferred_model
        self._tools: list[Tool] = tools or []
        self._conversation_state: ConversationState | None = conversation_state

    def _copy(self) -> "LLMPromptBuilder":
        """Create a copy of this builder with the same state."""
        new_instance = self.__class__(
            messages=self._messages.copy(),
            expect=self._expect,
            model_selector=self._model_selector,
            preferred_provider=self._preferred_provider,
            preferred_model=self._preferred_model,
            tools=self._tools.copy(),
            conversation_state=self._conversation_state,
        )
        return new_instance

    # ------------------------------------------------------------------
    # Chainable mutators
    # ------------------------------------------------------------------
    def agent(self, prompt: str) -> LLMPromptBuilder:
        """Add a *system* role message describing the agent/persona."""
        new_instance = self._copy()
        new_instance._messages.append(AgentMessage(text=prompt))
        return new_instance

    def context(self, content: str) -> LLMPromptBuilder:
        """Add background context (user role)."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=content))
        return new_instance

    def request(self, content: str) -> LLMPromptBuilder:
        """Add the primary user request message."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=content))
        return new_instance

    def audio(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Add an audio file to the request."""
        path = pathlib.Path(path)
        if path.suffix != '.mp3':
            raise ValueError(f'only .mp3 files are supported, but {path} has extension {path.suffix}')

        new_instance = self._copy()
        new_instance._messages.append(AudioMessage(audio_path=path))
        return new_instance

    def image(self, source: str | pathlib.Path | bytes) -> LLMPromptBuilder:
        """Add an image to the request.

        Args:
            source: Either a file path (str or pathlib.Path) or bytes containing the image data.
        """
        new_instance = self._copy()
        if isinstance(source, bytes):
            new_instance._messages.append(ImageMessage(image_data=source))
        else:
            new_instance._messages.append(ImageMessage(image_path=pathlib.Path(source)))
        return new_instance

    def file(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Attach an image file to the request (basic vision support)."""
        return self.image(path)

    def expect(self, response_type: ResponseType | Type[BaseModel]) -> "LLMPromptBuilder":
        """
        Specify the expected response type.
        Accepts either a ResponseType enum or a Pydantic BaseModel subclass for structured outputs.
        """
        new_instance = self._copy()
        # Accept ResponseType or BaseModel subclass
        if inspect.isclass(response_type) and issubclass(response_type, BaseModel):
            new_instance._expect = response_type
        elif isinstance(response_type, ResponseType):
            new_instance._expect = response_type
        else:
            raise TypeError("expect() argument must be a ResponseType or a Pydantic BaseModel subclass.")
        return new_instance

    def provider(self, provider_name: str) -> "LLMPromptBuilder":
        """Specify a preferred provider for this request.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            
        Returns:
            A new LLMPromptBuilder instance with the provider preference set
        """
        new_instance = self._copy()
        new_instance._preferred_provider = provider_name
        return new_instance

    def model(self, model_name: str) -> "LLMPromptBuilder":
        """Specify a preferred model for this request.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o-mini', 'claude-3-sonnet')
            
        Returns:
            A new LLMPromptBuilder instance with the model preference set
        """
        new_instance = self._copy()
        new_instance._preferred_model = model_name
        return new_instance

    def tool(self, tool_function: Any) -> "LLMPromptBuilder":
        """Add a single tool definition from function, auto-deriving metadata.
        
        Args:
            tool_function: A callable function to be used as a tool
            
        Returns:
            A new LLMPromptBuilder instance with the tool added
        """
        new_instance = self._copy()
        tool = Tool.from_function(tool_function)
        new_instance._tools.append(tool)
        return new_instance

    def tools(self, tool_functions: list[Any]) -> "LLMPromptBuilder":
        """Add multiple tool definitions from functions, auto-deriving metadata.
        
        Args:
            tool_functions: A list of callable functions to be used as tools
            
        Returns:
            A new LLMPromptBuilder instance with the tools added
        """
        new_instance = self._copy()
        for tool_function in tool_functions:
            tool = Tool.from_function(tool_function)
            new_instance._tools.append(tool)
        return new_instance

    # ------------------------------------------------------------------
    # Conversation state management helpers
    # ------------------------------------------------------------------
    def _add_tool_call_to_conversation(self, tool_name: str, tool_call_id: str, arguments: dict) -> None:
        """Add a tool call message to the conversation state."""
        if self._conversation_state is None:
            return
        
        tool_call_msg = ToolCallMessage(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments
        )
        
        self._conversation_state.message_history.append(tool_call_msg)
        self._conversation_state.api_messages.append(tool_call_msg.to_dict())
    
    def _add_tool_result_to_conversation(self, tool_call_id: str, result: Any) -> None:
        """Add a tool result message to the conversation state."""
        if self._conversation_state is None:
            return
        
        tool_result_msg = ToolResultMessage(
            tool_call_id=tool_call_id,
            result=result
        )
        
        self._conversation_state.message_history.append(tool_result_msg)
        self._conversation_state.api_messages.append(tool_result_msg.to_dict())
    
    def _execute_tool_call(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool call and return the result."""
        # Find the tool by name
        tool = None
        for t in self._tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in available tools")
        
        try:
            # Execute the tool function with the provided arguments
            return tool.function(**arguments)
        except Exception as e:
            # Return error information that can be sent back to the model
            return f"Error executing tool '{tool_name}': {str(e)}"

    # ------------------------------------------------------------------
    # Tool validation helpers
    # ------------------------------------------------------------------
    def _validate_no_tools_for_method(self, method_name: str) -> None:
        """Validate that no tools are defined when using non-conversation methods.
        
        Args:
            method_name: Name of the method being called
            
        Raises:
            ValueError: If tools are defined
        """
        if self._tools:
            raise ValueError(
                f"Cannot use {method_name}() when tools are defined. "
                f"Use prompt_conversation() instead for tool calling support."
            )

    # ------------------------------------------------------------------
    # Pricing / stats helpers
    # ------------------------------------------------------------------
    def _gather_token_counts(self, data: Any, path: str = '') -> list[tuple[str, int]]:
        """Recursively gather all token counts from a nested dictionary.

        Args:
            data: The dictionary or value to search for token counts
            path: Dot-separated path to the current location in the dictionary

        Returns:
            List of (token_key, count) tuples where token_key is the full path to the count
        """
        counts = []

        if isinstance(data, dict):
            for key, value in data.items():
                # Skip 'total_tokens' as it's not a billable token type
                if key == 'total_tokens':
                    continue

                current_path = f"{path}.{key}" if path else key
                if key.endswith(('_tokens', '_tokens_details')) or 'token' in key.lower():
                    if isinstance(value, int) and value > 0:
                        counts.append((current_path, value))
                    elif isinstance(value, dict):
                        counts.extend(self._gather_token_counts(value, current_path))
                elif isinstance(value, (dict, list)):
                    counts.extend(self._gather_token_counts(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                counts.extend(self._gather_token_counts(item, f"{path}[{i}]"))

        return counts

    @property
    def usage(self):
        return usage_tracker.tracker

    # ------------------------------------------------------------------
    # Prompt-for-* convenience methods (public API)
    # ------------------------------------------------------------------
    @asyncify
    async def prompt_for_text(self, **kwargs: Any) -> Any:
        """Execute the prompt and return a text response."""
        self._validate_no_tools_for_method("prompt_for_text")
        new_instance = self._copy()
        new_instance._expect = ResponseType.TEXT
        return await new_instance.call(**kwargs)

    @asyncify
    async def prompt_for_image(self, **kwargs: Any) -> Any:
        """Execute the prompt and return an image response."""
        self._validate_no_tools_for_method("prompt_for_image")
        new_instance = self._copy()
        new_instance._expect = ResponseType.IMAGE
        return await new_instance.call(**kwargs)

    @asyncify
    async def prompt_for_audio(self, **kwargs: Any) -> Any:
        """Execute the prompt and return an audio response."""
        self._validate_no_tools_for_method("prompt_for_audio")
        new_instance = self._copy()
        new_instance._expect = ResponseType.AUDIO
        return await new_instance.call(**kwargs)

    @asyncify
    async def prompt_for_type(self, response_type: Type[BaseModel], **kwargs: Any) -> Any:
        """Execute the prompt and return a response parsed into the specified Pydantic model.

        Args:
            response_type: A Pydantic model class to parse the response into.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            An instance of the specified Pydantic model with the parsed response.

        Example:
            ```python
            class EvaluationResult(BaseModel):
                score: int
                reason: str

            result = await llm\
                .agent("You are an art evaluator.")\
                .request("Rate this painting on a scale of 1-10 and explain your rating.")\
                .prompt_for_type(EvaluationResult)
            ```
        """
        self._validate_no_tools_for_method("prompt_for_type")
        new_instance = self._copy()
        new_instance._expect = response_type
        return await new_instance.call(**kwargs)

    @asyncify
    async def prompt(self, **kwargs: Any) -> Any:
        """Alias for prompt_for_text: execute the prompt and return a text response."""
        return await self.prompt_for_text(**kwargs)

    @asyncify
    async def prompt_conversation(self, message: str | None = None, **kwargs: Any) -> tuple[MessageList, "LLMPromptBuilder"]:
        """Execute conversation with tool calling support.
        
        Similar to prompt_for_text but supports tool calling and returns conversation state.
        Only supports text messages - no image, audio, or structured output support.
        
        Args:
            message: Optional text message to add to conversation
            **kwargs: Additional arguments passed to the provider API
        
        Returns:
            Tuple of (message_list, continuation_builder) where message_list
            contains all messages including tool calls and responses.
            
        Raises:
            ValueError: If no tools are defined or if called without tools
        """
        if not self._tools:
            raise ValueError("prompt_conversation() requires tools to be defined. Use tools() method first.")
        
        # Create a new instance for this conversation
        new_instance = self._copy()
        
        # Initialize conversation state if not already present
        if new_instance._conversation_state is None:
            # Convert existing messages to API format for initial state
            api_messages = []
            for msg in new_instance._messages:
                api_messages.append(msg.to_dict())
            
            new_instance._conversation_state = ConversationState(
                api_messages=api_messages,
                tools=new_instance._tools.copy(),
                message_history=new_instance._messages.copy()
            )
        
        # Add the user message if provided
        if message is not None:
            user_message = TextMessage(text=message)
            new_instance._conversation_state.message_history.append(user_message)
            new_instance._conversation_state.api_messages.append(user_message.to_dict())
        
        # For now, this is a basic implementation that doesn't actually call the API
        # The full implementation with tool calling will be completed when provider
        # integration is implemented in task 2
        
        # Create continuation builder with updated conversation state
        continuation_builder = new_instance._copy()
        
        return new_instance._conversation_state.message_history, continuation_builder

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def call(self, **kwargs: Any) -> Any:
        """
        Convert abstract prompt to OpenAI format and execute asynchronously.

        Args:
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from the LLM API. The format depends on the expected type:
            - For text: str
            - For JSON: dict or Pydantic model instance if a model class was specified
            - For other types: depends on the provider's implementation

        Raises:
            ValueError: If the selected model is not valid for the given input/output
            RuntimeError: If there is an error calling the LLM API or processing the response
        """
        # Validate that tools aren't used with call()
        self._validate_no_tools_for_method("call")
        
        # Build Prompt and select model
        p = Prompt(
            messages=self._messages,
            expect_type=self._expect,
            preferred_provider=self._preferred_provider,
            preferred_model=self._preferred_model,
        )
        provider, model = self._model_selector.select_model(p)
        global _TEMP_last_provider
        _TEMP_last_provider = provider

        # Validate the selection
        provider.check_capabilities(model, p)

        # Call the API
        return await provider.prompt_via_api(
            model=model,
            p=p,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Public instance
# ---------------------------------------------------------------------------

# Pre-instantiated LLMPromptBuilder for direct use
llm: LLMPromptBuilder = LLMPromptBuilder()
