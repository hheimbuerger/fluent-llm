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
from typing import Any, Sequence, Type, AsyncGenerator
import inspect
from dataclasses import dataclass
from .utils import asyncify

from pydantic import BaseModel
from .messages import TextMessage, AudioMessage, ImageMessage, AgentMessage, ResponseType, MessageList, ToolCallMessage, Role
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
    message_history: MessageList  # Our abstract message format (returned to user)
    internal_session: list[dict]  # Internal session messages in provider format
    tools: list[Tool]


class ConversationGenerator:
    """Async generator wrapper for conversation handling with continuation builder access."""
    
    def __init__(self, builder: "LLMPromptBuilder", **kwargs):
        self.builder = builder
        self.kwargs = kwargs
        self.llm_continuation: "LLMPromptBuilder" | None = None
        self._generator = None
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._generator is None:
            self._generator = self._create_generator()
        
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            # Generator is exhausted, continuation builder should be available
            raise
    
    async def _create_generator(self):
        """Create the actual async generator for conversation handling."""
        # Initialize conversation state using existing method
        conversation_state = self.builder._get_or_create_conversation_state()
        
        # Main conversation loop
        while True:
            # Build prompt and make API call
            p = Prompt(
                messages=MessageList(conversation_state.message_history),
                expect_type=ResponseType.TEXT,
                preferred_provider=self.builder._preferred_provider,
                preferred_model=self.builder._preferred_model,
                tools=self.builder._tools,
                is_conversation=True,
            )
            
            provider, model = self.builder._model_selector.select_model(p)
            
            # Check if provider supports tools (only if tools are defined)
            if self.builder._tools and (not hasattr(provider, 'supports_tools') or not provider.supports_tools()):
                raise ValueError(
                    f"Provider {type(provider).__name__} does not support tool calling. "
                    f"Tool calling is currently only supported with Anthropic provider."
                )
            
            # Validate capabilities
            provider.check_capabilities(model, p)
            
            # Make API call
            response = await provider.prompt_via_api(
                model=model, 
                p=p, 
                conversation_state=conversation_state,
                **self.kwargs
            )

            # Process response - check if it contains tool calls
            if isinstance(response, dict) and response.get('tool_calls'):
                # Handle tool calls from dict response
                for tool_call in response['tool_calls']:
                    tool_name = tool_call["name"]
                    tool_id = tool_call["id"]
                    tool_args = tool_call.get("input", tool_call.get("arguments", {}))
                    
                    # Execute tool and get result/error tuple
                    result, error = self.builder._execute_tool_call(tool_name, tool_args)
                    
                    # Create unified tool call message
                    tool_message = ToolCallMessage(
                        message=response.get('text', ""),
                        tool_name=tool_name,
                        tool_call_id=tool_id,
                        arguments=tool_args,
                        result=result,
                        error=error
                    )
                    
                    conversation_state.message_history.append(tool_message)
                    conversation_state.internal_session.append(tool_message.to_dict())
                    yield tool_message
                    
                # Continue conversation with tool results
                continue
            else:
                # Regular text response - end of conversation
                if isinstance(response, dict):
                    text_content = response.get('text', str(response))
                else:
                    text_content = str(response)
                    
                text_message = TextMessage(text=text_content, role=Role.ASSISTANT)
                conversation_state.message_history.append(text_message)
                conversation_state.internal_session.append(text_message.to_dict())
                yield text_message
                
                # Create continuation builder and store it
                self.llm_continuation = self.builder._copy()
                self.llm_continuation._conversation_state = conversation_state
                return


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
    def agent(self, text: str) -> LLMPromptBuilder:
        """Add a *system* role message describing the agent/persona."""
        new_instance = self._copy()
        new_instance._messages.append(AgentMessage(text=text.strip()))
        return new_instance

    def context(self, text: str) -> LLMPromptBuilder:
        """Add background context (user role)."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=text.strip()))
        return new_instance

    def request(self, text: str) -> LLMPromptBuilder:
        """Add the primary user request message."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=text.strip()))
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

    def expect(self, response_type: ResponseType | Type[BaseModel]) -> LLMPromptBuilder:
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

    def provider(self, provider_name: str) -> LLMPromptBuilder:
        """Specify a preferred provider for this request.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            
        Returns:
            A new LLMPromptBuilder instance with the provider preference set
        """
        new_instance = self._copy()
        new_instance._preferred_provider = provider_name
        return new_instance

    def model(self, model_name: str) -> LLMPromptBuilder:
        """Specify a preferred model for this request.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o-mini', 'claude-3-sonnet')
            
        Returns:
            A new LLMPromptBuilder instance with the model preference set
        """
        new_instance = self._copy()
        new_instance._preferred_model = model_name
        return new_instance

    def tool(self, tool_function: Any) -> LLMPromptBuilder:
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

    def tools(self, *tool_functions: Any) -> LLMPromptBuilder:
        """Add multiple tool definitions from functions, auto-deriving metadata.
        
        Args:
            *tool_functions: Either individual callable functions as separate arguments,
                           or a single list of callable functions
            
        Returns:
            A new LLMPromptBuilder instance with the tools added
            
        Examples:
            # Pass individual functions as arguments
            builder.tools(func1, func2, func3)
            
            # Pass a list of functions
            builder.tools([func1, func2, func3])
        """
        new_instance = self._copy()
        
        # Handle both individual args and list input
        if len(tool_functions) == 1 and isinstance(tool_functions[0], list):
            # Single argument that is a list - use the list contents
            functions_to_add = tool_functions[0]
        else:
            # Multiple arguments or single non-list argument - use all args
            functions_to_add = tool_functions
        
        for tool_function in functions_to_add:
            tool = Tool.from_function(tool_function)
            new_instance._tools.append(tool)
        return new_instance

    # ------------------------------------------------------------------
    # Conversation state management helpers
    # ------------------------------------------------------------------
    def _get_or_create_conversation_state(self) -> ConversationState:
        """Get existing conversation state or create a new one."""
        if self._conversation_state is None:
            # Convert existing messages to provider format for initial state
            internal_session = []
            for msg in self._messages:
                internal_session.append(msg.to_dict())
            
            self._conversation_state = ConversationState(
                message_history=self._messages.copy(),
                internal_session=internal_session,
                tools=self._tools.copy()
            )
        
        return self._conversation_state

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
    
    def _execute_tool_call(self, tool_name: str, arguments: dict) -> tuple[Any | None, Exception | None]:
        """Execute a tool call and return (result, error) tuple."""
        # Find the tool by name
        tool = None
        for t in self._tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool is None:
            error = ValueError(f"Tool '{tool_name}' not found in available tools")
            return None, error
        
        try:
            # Execute the tool function with the provided arguments
            result = tool.function(**arguments)
            return result, None
        except Exception as e:
            return None, e

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

    def prompt_conversation(self, **kwargs: Any) -> "ConversationGenerator":
        """Execute conversation with tool calling support using async generator.
        
        Returns a ConversationGenerator that yields individual messages as the conversation 
        progresses and provides access to the continuation builder after completion.
        
        Args:
            **kwargs: Additional arguments passed to the provider API
        
        Returns:
            ConversationGenerator that yields messages and provides continuation access
            
        Example:
            ```python
            conversation = llm.agent("...").request("...").prompt_conversation()
            
            # Manual iteration
            response1 = await conversation.__anext__()
            response2 = await conversation.__anext__()
            
            # For loop iteration  
            async for message in conversation:
                print(f"Received: {message}")
                
            # Access continuation builder after completion
            continuation_builder = conversation.llm_continuation
            ```
        """
        return ConversationGenerator(self, **kwargs)

    @asyncify
    async def prompt_agentically(self, max_calls: int, **kwargs: Any) -> tuple[MessageList, "LLMPromptBuilder"]:
        """Execute conversation with automatic tool calling up to max_calls.
        
        Similar to the old prompt_conversation behavior - automatically processes
        all tool calls and returns the complete message list and continuation.
        
        Args:
            max_calls: Maximum number of API calls to prevent infinite loops
            **kwargs: Additional arguments passed to the provider API
        
        Returns:
            Tuple of (complete_message_list, continuation_builder)
        """
        messages = []
        call_count = 0
        
        # Get the conversation generator
        conversation = self.prompt_conversation(**kwargs)
        
        try:
            # Automatically iterate through all messages up to max_calls
            while call_count < max_calls:
                try:
                    message = await conversation.__anext__()
                    messages.append(message)
                    call_count += 1
                except StopAsyncIteration:
                    break
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Error during automatic conversation processing: {str(e)}") from e
        
        # Get continuation builder from the generator
        continuation_builder = conversation.llm_continuation
        if continuation_builder is None:
            # If we hit max_calls, create a continuation builder from current state
            continuation_builder = self._copy()
            if hasattr(self, '_conversation_state'):
                continuation_builder._conversation_state = self._conversation_state
        
        return MessageList(messages), continuation_builder

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
            tools=self._tools if self._tools else None,
            is_conversation=False,
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
            conversation_state=None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Public instance
# ---------------------------------------------------------------------------

# Pre-instantiated LLMPromptBuilder for direct use
llm: LLMPromptBuilder = LLMPromptBuilder()
