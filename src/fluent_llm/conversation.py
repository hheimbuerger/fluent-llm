"""Conversation management for fluent_llm.

This module implements the three-class architecture for conversation handling:
- MessageList: Serializable message container (mutable data)
- LLMConversation: Mutable execution context
- LLMPromptBuilder: Immutable composition tool with delta pattern
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, IO, TYPE_CHECKING
from .messages import Message, TextMessage, AgentMessage, ToolCallMessage, Role, ResponseType

if TYPE_CHECKING:
    from .model_selector import ModelSelectionStrategy, DefaultModelSelectionStrategy
    from .prompt import Prompt
    from .tools import Tool


# Custom exceptions
class MessageListDeserializationError(Exception):
    """Raised when MessageList data is invalid."""
    pass


class DeltaApplicationError(Exception):
    """Raised when deltas cannot be applied to conversation."""
    pass


class ConversationConfigurationError(Exception):
    """Raised when conversation configuration is invalid."""
    pass


class MessageList:
    """Serializable message container - the single source of truth for conversation data."""
    
    def __init__(self, messages: list[Message] | None = None):
        self._messages: list[Message] = messages or []
    
    def append(self, message: Message) -> None:
        """Add message to the list (mutable operation)."""
        self._messages.append(message)
    
    def extend(self, messages: MessageList | list[Message]) -> None:
        """Add multiple messages (mutable operation)."""
        if isinstance(messages, MessageList):
            self._messages.extend(messages._messages)
        else:
            self._messages.extend(messages)
    
    def copy(self) -> 'MessageList':
        """Create a copy of this MessageList."""
        return MessageList([msg for msg in self._messages])
    
    def __iter__(self):
        return iter(self._messages)
    
    def __len__(self):
        return len(self._messages)
    
    def __getitem__(self, index):
        return self._messages[index]
    
    # Helper methods for checking message types
    def has_type(self, msg_type: type) -> bool:
        """Check if the list contains any messages of the specified type."""
        return any(isinstance(msg, msg_type) for msg in self._messages)
    
    @property
    def has_text(self) -> bool:
        """Check if the list contains any TextMessage instances."""
        return self.has_type(TextMessage)
    
    @property
    def has_audio(self) -> bool:
        """Check if the list contains any AudioMessage instances."""
        from .messages import AudioMessage
        return self.has_type(AudioMessage)
    
    @property
    def has_image(self) -> bool:
        """Check if the list contains any ImageMessage instances."""
        from .messages import ImageMessage
        return self.has_type(ImageMessage)
    
    @property
    def has_agent(self) -> bool:
        """Check if the list contains any AgentMessage instances."""
        return self.has_type(AgentMessage)
    
    @property
    def has_tool_call(self) -> bool:
        """Check if the list contains any ToolCallMessage instances."""
        return self.has_type(ToolCallMessage)
    
    def to_dict_list(self) -> list[dict]:
        """Convert all messages to their dictionary representation."""
        return [self._message_to_serializable_dict(msg) for msg in self._messages]
    
    def merge_all_text(self) -> str:
        """Merge all text messages into a single string."""
        return "\n".join(msg.text for msg in self._messages if isinstance(msg, (AgentMessage, TextMessage)))
    
    def merge_all_agent(self) -> str:
        """Merge all agent messages into a single string."""
        return "\n".join(msg.text for msg in self._messages if isinstance(msg, AgentMessage))
    
    # Serialization (only on MessageList)
    def to_dict(self) -> dict:
        """Serialize messages to dictionary."""
        return {
            "messages": [self._message_to_serializable_dict(msg) for msg in self._messages],
            "version": "1.0"
        }
    
    @staticmethod
    def _message_to_serializable_dict(msg: Message) -> dict:
        """Convert a message to a serializable dictionary."""
        if isinstance(msg, TextMessage):
            return {
                "type": "TextMessage",
                "text": msg.text,
                "role": msg.role.value
            }
        elif isinstance(msg, AgentMessage):
            return {
                "type": "AgentMessage",
                "text": msg.text,
                "role": msg.role.value
            }
        elif isinstance(msg, ToolCallMessage):
            return {
                "type": "ToolCallMessage",
                "message": msg.message,
                "tool_name": msg.tool_name,
                "tool_call_id": msg.tool_call_id,
                "arguments": msg.arguments,
                "result": msg.result,
                "error": str(msg.error) if msg.error else None,
                "role": msg.role.value
            }
        else:
            raise MessageListDeserializationError(f"Cannot serialize message type: {type(msg).__name__}")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageList':
        """Deserialize messages from dictionary."""
        if not isinstance(data, dict):
            raise MessageListDeserializationError(f"Expected dict, got {type(data).__name__}")
        
        version = data.get("version")
        if version != "1.0":
            raise MessageListDeserializationError(f"Unsupported version: {version}")
        
        messages_data = data.get("messages")
        if not isinstance(messages_data, list):
            raise MessageListDeserializationError(f"Expected 'messages' to be a list, got {type(messages_data).__name__}")
        
        messages = []
        for i, msg_data in enumerate(messages_data):
            try:
                msg = cls._message_from_serializable_dict(msg_data)
                messages.append(msg)
            except Exception as e:
                raise MessageListDeserializationError(f"Error deserializing message at index {i}: {str(e)}") from e
        
        return cls(messages)
    
    @staticmethod
    def _message_from_serializable_dict(msg_data: dict) -> Message:
        """Convert a serializable dictionary to a message."""
        if not isinstance(msg_data, dict):
            raise MessageListDeserializationError(f"Expected dict, got {type(msg_data).__name__}")
        
        msg_type = msg_data.get("type")
        if not msg_type:
            raise MessageListDeserializationError("Message data missing 'type' field")
        
        if msg_type == "TextMessage":
            return TextMessage(
                text=msg_data["text"],
                role=Role(msg_data["role"])
            )
        elif msg_type == "AgentMessage":
            return AgentMessage(
                text=msg_data["text"],
                role=Role(msg_data.get("role", "system"))
            )
        elif msg_type == "ToolCallMessage":
            error = msg_data.get("error")
            if error:
                error = Exception(error)
            return ToolCallMessage(
                message=msg_data["message"],
                tool_name=msg_data["tool_name"],
                tool_call_id=msg_data["tool_call_id"],
                arguments=msg_data["arguments"],
                result=msg_data.get("result"),
                error=error,
                role=Role(msg_data.get("role", "assistant"))
            )
        else:
            raise MessageListDeserializationError(f"Unknown message type: {msg_type}")


@dataclass
class ConversationConfig:
    """Configuration for conversation execution."""
    preferred_provider: str | None = None
    preferred_model: str | None = None
    model_selector: Any = None  # ModelSelectionStrategy - avoid circular import
    tools: list[Any] = field(default_factory=list)  # list[Tool] - avoid circular import
    expect_type: Any = None  # ResponseType or BaseModel class - avoid circular import
    
    def __post_init__(self):
        """Initialize defaults if not provided."""
        if self.model_selector is None:
            from .model_selector import DefaultModelSelectionStrategy
            self.model_selector = DefaultModelSelectionStrategy()
        if self.expect_type is None:
            self.expect_type = ResponseType.TEXT


class LLMConversation:
    """Mutable execution context - owns MessageList and handles API calls."""
    
    def __init__(self, messages: MessageList | None = None, config: ConversationConfig | None = None):
        # State ownership
        self.messages: MessageList = messages or MessageList()  # Single source of truth
        self._config: ConversationConfig = config or ConversationConfig()
        
        # Execution state
        self._generator: AsyncGenerator | None = None
        self._kwargs: dict = {}
    
    @property
    def continuation(self) -> 'LLMPromptBuilder':
        """Create builder that references this conversation."""
        # Import here to avoid circular dependency
        from .builder import LLMPromptBuilder
        return LLMPromptBuilder(conversation=self)
    
    def apply_config_deltas(self, delta_config: dict) -> None:
        """Apply configuration changes (mutable operation)."""
        for key, value in delta_config.items():
            if key == "tools":
                # For tools, we need to handle list merging
                if not isinstance(value, list):
                    raise ConversationConfigurationError(f"Config key 'tools' must be a list, got {type(value).__name__}")
                self._config.tools = value
            elif hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise ConversationConfigurationError(f"Unknown config key: {key}")
    
    # Async iterator protocol
    def __aiter__(self):
        # Reset generator when starting a new iteration
        self._generator = None
        return self
    
    async def __anext__(self):
        if self._generator is None:
            self._generator = self._create_generator()
        
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            raise
    
    async def _create_generator(self) -> AsyncGenerator[Message, None]:
        """Execute conversation and yield messages."""
        from .prompt import Prompt
        
        while True:
            # Build prompt from current messages
            
            p = Prompt(
                messages=self.messages,
                expect_type=self._config.expect_type,
                preferred_provider=self._config.preferred_provider,
                preferred_model=self._config.preferred_model,
                tools=self._config.tools if self._config.tools else None,
                is_conversation=True,
            )
            
            # Select model and make API call
            provider, model = self._config.model_selector.select_model(p)
            
            # Check if provider supports tools (only if tools are defined)
            if self._config.tools and (not hasattr(provider, 'supports_tools') or not provider.supports_tools()):
                raise ValueError(
                    f"Provider {type(provider).__name__} does not support tool calling. "
                    f"Tool calling is currently only supported with Anthropic provider."
                )
            
            provider.check_capabilities(model, p)
            
            response = await provider.prompt_via_api(
                model=model,
                p=p,
                conversation_state=None,  # We manage our own state
                **self._kwargs
            )
            
            # Process response and update our MessageList
            if isinstance(response, dict) and response.get('tool_calls'):
                # Handle tool calls
                for tool_call in response['tool_calls']:
                    result, error = self._execute_tool_call(
                        tool_call["name"], 
                        tool_call.get("input", tool_call.get("arguments", {}))
                    )
                    
                    tool_message = ToolCallMessage(
                        message=response.get('text', ""),
                        tool_name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        arguments=tool_call.get("input", tool_call.get("arguments", {})),
                        result=result,
                        error=error
                    )
                    
                    self.messages.append(tool_message)  # Mutate our MessageList
                    yield tool_message
                
                continue  # Continue conversation loop
            else:
                # Final text response
                if isinstance(response, dict):
                    text_content = response.get('text', str(response))
                else:
                    text_content = str(response)
                
                text_message = TextMessage(text=text_content, role=Role.ASSISTANT)
                self.messages.append(text_message)  # Mutate our MessageList
                yield text_message
                return  # End conversation
    
    def _execute_tool_call(self, tool_name: str, arguments: dict) -> tuple[Any | None, Exception | None]:
        """Execute tool call using configured tools."""
        for tool in self._config.tools:
            if tool.name == tool_name:
                try:
                    result = tool.function(**arguments)
                    return result, None
                except Exception as e:
                    return None, e
        
        return None, ValueError(f"Tool '{tool_name}' not found")
    
    def save(self, destination: str | pathlib.Path | IO) -> None:
        """Save the conversation's MessageList to a file or stream.
        
        This method serializes only the MessageList (message data) without
        configuration or tools, making it model-agnostic and portable.
        
        Args:
            destination: Can be:
                - str: Path to save the JSON file
                - pathlib.Path: Path object for the output file
                - IO: An open file-like object (stream) to write to
                
        Raises:
            IOError: If there's an error writing to the file/stream
            MessageListDeserializationError: If serialization fails
            
        Examples:
            ```python
            # Save to file path (string)
            conversation.save("conversation.json")
            
            # Save to Path object
            from pathlib import Path
            conversation.save(Path("conversation.json"))
            
            # Save to stream
            with open("conversation.json", "w") as f:
                conversation.save(f)
            
            # Later, restore and continue with any configuration
            restored = llm.load_conversation("conversation.json")
            continuation = restored.continuation \\
                .provider("openai") \\
                .request("Continue")
            ```
        """
        try:
            # Serialize MessageList to dictionary
            data = self.messages.to_dict()
            
            # Handle different output types
            if isinstance(destination, (str, pathlib.Path)):
                # File path output
                path = pathlib.Path(destination)
                # Create parent directories if they don't exist
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif hasattr(destination, 'write'):
                # IO stream output
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                destination.write(json_str)
            else:
                raise ValueError(f"Unsupported destination type: {type(destination).__name__}. "
                               f"Expected str, Path, or IO stream.")
                
        except Exception as e:
            raise IOError(f"Failed to save conversation: {str(e)}") from e
