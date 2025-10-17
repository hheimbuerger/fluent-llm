"""Fluent, async builder for Large Language Model (LLM) requests.

This builder uses the delta pattern - it's immutable and accumulates changes
that are applied when creating or executing a conversation.
"""
from __future__ import annotations

import pathlib
from typing import Any, Sequence, Type
import inspect
from .utils import asyncify

from pydantic import BaseModel
from .messages import TextMessage, AudioMessage, ImageMessage, AgentMessage, ResponseType, Role, Message
from .model_selector import ModelSelectionStrategy, DefaultModelSelectionStrategy
from .prompt import Prompt
from .tools import Tool
from fluent_llm import usage_tracker
from .conversation import MessageList, LLMConversation, ConversationConfig, DeltaApplicationError

__all__: Sequence[str] = [
    "llm",
]


_TEMP_last_provider = None


class LLMPromptBuilder:
    """Fluent, immutable builder for LLM requests using delta pattern."""

    def __init__(
        self,
        *,
        conversation: LLMConversation | None = None,
        delta_messages: MessageList | None = None,
        delta_config: dict | None = None,
    ) -> None:
        # New delta-based architecture
        self._conversation: LLMConversation = conversation or LLMConversation()
        self._delta_messages: MessageList = delta_messages or MessageList()
        self._delta_config: dict = delta_config or {}

    def _copy_with_delta_message(self, message: Message) -> "LLMPromptBuilder":
        """Create new builder with additional delta message."""
        new_delta_messages = self._delta_messages.copy()
        new_delta_messages.append(message)
        
        # If this is the first operation on a fresh builder (no deltas and empty conversation),
        # create a new conversation to avoid pollution between parallel tests
        conversation = self._conversation
        if (len(self._delta_messages) == 0 and 
            len(self._delta_config) == 0 and 
            len(self._conversation.messages) == 0):
            # This is the first operation on a fresh builder, create new conversation
            conversation = LLMConversation()
        
        return LLMPromptBuilder(
            conversation=conversation,
            delta_messages=new_delta_messages,
            delta_config=self._delta_config.copy(),
        )
    
    def _copy_with_delta_config(self, **config_updates) -> "LLMPromptBuilder":
        """Create new builder with additional config deltas."""
        new_delta_config = {**self._delta_config, **config_updates}
        
        # If this is the first operation on a fresh builder (no deltas and empty conversation),
        # create a new conversation to avoid pollution between parallel tests
        conversation = self._conversation
        if (len(self._delta_messages) == 0 and 
            len(self._delta_config) == 0 and 
            len(self._conversation.messages) == 0):
            # This is the first operation on a fresh builder, create new conversation
            conversation = LLMConversation()
        
        return LLMPromptBuilder(
            conversation=conversation,
            delta_messages=self._delta_messages.copy(),
            delta_config=new_delta_config,
        )

    # ------------------------------------------------------------------
    # Chainable mutators (all return new instances)
    # ------------------------------------------------------------------
    def agent(self, text: str) -> LLMPromptBuilder:
        """Add a *system* role message describing the agent/persona."""
        return self._copy_with_delta_message(AgentMessage(text=text.strip()))

    def assistant(self, text: str) -> LLMPromptBuilder:
        """Add an assistant message to the conversation."""
        return self._copy_with_delta_message(TextMessage(text=text.strip(), role=Role.ASSISTANT))

    def context(self, text: str) -> LLMPromptBuilder:
        """Add background context (user role)."""
        return self._copy_with_delta_message(TextMessage(text=text.strip()))

    def request(self, text: str) -> LLMPromptBuilder:
        """Add the primary user request message."""
        return self._copy_with_delta_message(TextMessage(text=text.strip()))

    def audio(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Add an audio file to the request."""
        path = pathlib.Path(path)
        if path.suffix != '.mp3':
            raise ValueError(f'only .mp3 files are supported, but {path} has extension {path.suffix}')

        return self._copy_with_delta_message(AudioMessage(audio_path=path))

    def image(self, source: str | pathlib.Path | bytes) -> LLMPromptBuilder:
        """Add an image to the request.

        Args:
            source: Either a file path (str or pathlib.Path) or bytes containing the image data.
        """
        if isinstance(source, bytes):
            return self._copy_with_delta_message(ImageMessage(image_data=source))
        else:
            return self._copy_with_delta_message(ImageMessage(image_path=pathlib.Path(source)))

    def file(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Attach an image file to the request (basic vision support)."""
        return self.image(path)

    def provider(self, provider_name: str) -> LLMPromptBuilder:
        """Specify a preferred provider for this request.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            
        Returns:
            A new LLMPromptBuilder instance with the provider preference set
        """
        return self._copy_with_delta_config(preferred_provider=provider_name)

    def model(self, model_name: str) -> LLMPromptBuilder:
        """Specify a preferred model for this request.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o-mini', 'claude-3-sonnet')
            
        Returns:
            A new LLMPromptBuilder instance with the model preference set
        """
        return self._copy_with_delta_config(preferred_model=model_name)

    def tool(self, tool_function: Any) -> LLMPromptBuilder:
        """Add a single tool definition from function, auto-deriving metadata.
        
        Args:
            tool_function: A callable function to be used as a tool
            
        Returns:
            A new LLMPromptBuilder instance with the tool added
        """
        return self.tools(tool_function)

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
        # Handle both individual args and list input
        if len(tool_functions) == 1 and isinstance(tool_functions[0], list):
            # Single argument that is a list - use the list contents
            functions_to_add = tool_functions[0]
        else:
            # Multiple arguments or single non-list argument - use all args
            functions_to_add = tool_functions
        
        new_tools = [Tool.from_function(f) for f in functions_to_add]
        # Get existing tools from delta config or conversation config
        existing_tools = self._delta_config.get("tools", self._conversation._config.tools)
        current_tools = existing_tools + new_tools
        return self._copy_with_delta_config(tools=current_tools)

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------
    def prompt_conversation(self, **kwargs: Any) -> LLMConversation:
        """Apply deltas and return conversation for execution.
        
        Args:
            **kwargs: Additional arguments passed to the provider API
        
        Returns:
            LLMConversation that can be iterated to get messages
            
        Example:
            ```python
            conversation = llm.agent("...").request("...").prompt_conversation()
            
            # Iterate through messages
            async for message in conversation:
                print(f"Received: {message}")
                
            # Access continuation builder
            continuation = conversation.continuation
            ```
        """
        try:
            # Apply delta messages
            self._conversation.messages.extend(self._delta_messages)
            
            # Apply delta config
            if self._delta_config:
                self._conversation.apply_config_deltas(self._delta_config)
            
            # Store execution kwargs
            self._conversation._kwargs.update(kwargs)
            
            return self._conversation
        except Exception as e:
            raise DeltaApplicationError(f"Failed to apply deltas to conversation: {str(e)}") from e

    @asyncify
    async def prompt(self, **kwargs: Any) -> str:
        """Execute the prompt and return final text response.
        
        This is a convenience method that executes the conversation and returns
        the final assistant message text.
        
        Args:
            **kwargs: Additional arguments passed to the provider API
            
        Returns:
            The text content of the final assistant message
        """
        conversation = self.prompt_conversation(**kwargs)
        
        # Execute conversation and extract final text response
        final_message = None
        async for message in conversation:
            final_message = message
        
        if isinstance(final_message, TextMessage) and final_message.role == Role.ASSISTANT:
            return final_message.text
        else:
            return str(final_message) if final_message else ""

    def expect(self, response_type: ResponseType | Type[BaseModel]) -> LLMPromptBuilder:
        """Specify expected response type.
        
        Args:
            response_type: Either a ResponseType enum or a Pydantic BaseModel class
            
        Returns:
            A new builder with the expected response type set
        """
        return self._copy_with_delta_config(expect_type=response_type)
    
    async def call(self, **kwargs: Any) -> Any:
        """Execute the prompt and return the response.
        
        This method creates a temporary conversation, executes it, and returns
        the final result based on the expected response type.
        """
        # Get the expected type from delta config or default to TEXT
        expect_type = self._delta_config.get("expect_type", ResponseType.TEXT)
        
        # For image/audio/structured output, we need special handling
        if expect_type == ResponseType.IMAGE:
            return await self.prompt_for_image(**kwargs)
        elif expect_type == ResponseType.AUDIO:
            return await self.prompt_for_audio(**kwargs)
        elif inspect.isclass(expect_type) and issubclass(expect_type, BaseModel):
            return await self.prompt_for_type(expect_type, **kwargs)
        else:
            return await self.prompt(**kwargs)
    
    @asyncify
    async def prompt_for_image(self, **kwargs: Any) -> Any:
        """Execute the prompt and return an image response.
        
        This creates a conversation with IMAGE response type and executes it.
        """
        # Create a modified builder with IMAGE expect type
        builder = self._copy_with_delta_config(expect_type=ResponseType.IMAGE)
        
        # Create conversation and execute
        conversation = builder.prompt_conversation(**kwargs)
        
        # For image generation, we need to handle the response differently
        # The provider will return image data
        final_message = None
        async for message in conversation:
            final_message = message
        
        # Return the image data (provider-specific format)
        return final_message
    
    @asyncify
    async def prompt_for_audio(self, **kwargs: Any) -> Any:
        """Execute the prompt and return an audio response.
        
        This creates a conversation with AUDIO response type and executes it.
        """
        # Create a modified builder with AUDIO expect type
        builder = self._copy_with_delta_config(expect_type=ResponseType.AUDIO)
        
        # Create conversation and execute
        conversation = builder.prompt_conversation(**kwargs)
        
        # For audio generation, we need to handle the response differently
        final_message = None
        async for message in conversation:
            final_message = message
        
        # Return the audio data (provider-specific format)
        return final_message
    
    @asyncify
    async def prompt_for_type(self, response_type: Type[BaseModel], **kwargs: Any) -> Any:
        """Execute the prompt and return a structured response.
        
        Args:
            response_type: A Pydantic BaseModel class to parse the response into
            
        Returns:
            An instance of the specified Pydantic model
        """
        # Create a modified builder with structured output type
        builder = self._copy_with_delta_config(expect_type=response_type)
        
        # Create conversation and execute
        conversation = builder.prompt_conversation(**kwargs)
        
        # Execute and get final response
        final_message = None
        async for message in conversation:
            final_message = message
        
        # Return the structured data (provider will handle parsing)
        return final_message

    @property
    def usage(self):
        """Access usage tracking."""
        return usage_tracker.tracker


# ---------------------------------------------------------------------------
# Public instance
# ---------------------------------------------------------------------------

# Pre-instantiated LLMPromptBuilder for direct use
llm: LLMPromptBuilder = LLMPromptBuilder()
