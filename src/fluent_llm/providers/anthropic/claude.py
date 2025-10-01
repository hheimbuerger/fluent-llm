from typing import Tuple, Any
from decimal import Decimal
import anthropic

from ..provider import LLMProvider, LLMModel
from ...messages import MessageList, TextMessage, ImageMessage, AgentMessage, ToolCallMessage
from ...exceptions import *
from ...usage_tracker import tracker
from ...prompt import Prompt


class AnthropicProvider(LLMProvider):
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return True
    
    def get_token_type_to_price_mapping(self) -> dict:
        """Get Anthropic-specific token type mapping.

        Returns:
            Dictionary mapping Anthropic token types to pricing field base names.
        """
        return {
            'input_tokens': 'price_per_million_text_tokens_input',
            'output_tokens': 'price_per_million_text_tokens_output',
        }

    def get_models(self) -> Tuple[LLMModel]:
        return (
            LLMModel(
                name="claude-3-5-sonnet-20241022",
                text_input=True,
                text_output=True,
                image_input=True,
                image_output=False,
                audio_input=False,
                audio_output=False,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("3.00"),
                price_per_million_text_tokens_output=Decimal("15.00"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
            LLMModel(
                name="claude-3-5-haiku-20241022",
                text_input=True,
                text_output=True,
                image_input=False,
                image_output=False,
                audio_input=False,
                audio_output=False,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("1.00"),
                price_per_million_text_tokens_output=Decimal("5.00"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
            LLMModel(
                name="claude-sonnet-4-20250514",
                text_input=True,
                text_output=True,
                image_input=True,
                image_output=False,
                audio_input=False,
                audio_output=False,
                structured_output=True,
                price_per_million_text_tokens_input=Decimal("3.00"),
                price_per_million_text_tokens_output=Decimal("15.00"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
            LLMModel(
                name="claude-sonnet-4-5-20250929",
                text_input=True,
                text_output=True,
                image_input=True,
                image_output=False,
                audio_input=False,
                audio_output=False,
                structured_output=True,
                price_per_million_text_tokens_input=Decimal("3.00"),
                price_per_million_text_tokens_output=Decimal("15.00"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
        )

    async def prompt_via_api(self, model: str, p: Prompt, conversation_state=None, **kwargs: Any) -> Any:
        client = anthropic.AsyncAnthropic()

        # Prepare the API call parameters
        api_params = {
            "model": model,
            "max_tokens": 4096,  # Safe limit for all Claude models
            "temperature": 1,
            "system": p.messages.merge_all_agent(),
            "messages": tuple(self._convert_messages_to_api_format(p.messages)),
        }

        # Add tools if present
        if p.has_tools:
            api_params["tools"] = self._convert_tools_to_api_format(p.tools)

        try:
            response = await client.messages.create(**api_params)
            tracker.track_usage(self, model, response.usage)
        except anthropic.APIError as e:
            if "tool" in str(e).lower():
                raise RuntimeError(f"Tool calling API error: {str(e)}") from e
            else:
                raise RuntimeError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during API call: {str(e)}") from e

        # Handle different stop reasons
        stop_reason = getattr(response, "stop_reason", None)
        
        if stop_reason == "tool_use":
            # Handle tool calling response
            return await self._handle_tool_use_response(response, p, conversation_state)
        elif stop_reason == "end_turn":
            # Handle regular text response
            return self._handle_text_response(response)
        elif stop_reason == "max_tokens":
            # Continuation logic not yet implemented; treat as failure.
            raise NotImplementedError(
                "Anthropic generation stopped due to max_tokens; continuation not implemented."
            )
        else:
            raise RuntimeError(f"Anthropic API returned unexpected stop_reason: {stop_reason!r}")

    def _convert_tools_to_api_format(self, tools: list) -> list:
        """Convert Tool objects to Anthropic API format."""
        api_tools = []
        for tool in tools:
            api_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema
            })
        return api_tools

    def _handle_text_response(self, response) -> str:
        """Handle a regular text response from Anthropic."""
        # Handle empty content (can happen with some API responses)
        if len(response.content) == 0:
            return ""
        
        # Validate and extract the single text block.
        if len(response.content) != 1:
            raise RuntimeError(f"Expected exactly one content block, got {len(response.content)}")
        response_message = response.content[0]
        if response_message.type != "text":
            raise RuntimeError(f"Unexpected block type {response_message.type!r}, expected 'text'")

        return response_message.text

    async def _handle_tool_use_response(self, response, p: Prompt, conversation_state=None) -> Any:
        """Handle a tool use response from Anthropic.
        
        For the new async generator pattern, this method returns a response object
        that contains tool calls for the generator to process.
        """
        # Extract tool calls from response
        tool_calls = []
        text_content = ""
        
        for content_block in response.content:
            if content_block.type == "text":
                text_content += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append({
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": content_block.input
                })
        
        if not tool_calls:
            raise RuntimeError("Expected tool calls in tool_use response, but found none")
        
        # Return a simple dict that the async generator can process
        return {
            'text': text_content,
            'tool_calls': tool_calls
        }

    async def _continue_conversation_with_tool_results(
        self, 
        original_prompt: Prompt, 
        assistant_response, 
        tool_results: list,
        assistant_text: str,
        conversation_state=None
    ) -> str:
        """Continue the conversation after tool execution."""
        client = anthropic.AsyncAnthropic()
        
        # Build the conversation history including the assistant's response and tool results
        messages = list(self._convert_messages_to_api_format(original_prompt.messages))
        
        # Add the assistant's response (which included tool calls)
        assistant_content = []
        if assistant_text:
            assistant_content.append({"type": "text", "text": assistant_text})
        
        # Add tool use blocks
        for content_block in assistant_response.content:
            if content_block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": content_block.input
                })
        
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        # Add tool results as user message
        user_content = []
        for result in tool_results:
            user_content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_use_id"],
                "content": result["content"]
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Make another API call to get the final response
        api_params = {
            "model": original_prompt.preferred_model or "claude-3-5-haiku-20241022",
            "max_tokens": 4096,  # Safe limit for all Claude models
            "temperature": 1,
            "system": original_prompt.messages.merge_all_agent(),
            "messages": messages,
        }
        
        if original_prompt.has_tools:
            api_params["tools"] = self._convert_tools_to_api_format(original_prompt.tools)
        
        try:
            final_response = await client.messages.create(**api_params)
            tracker.track_usage(self, api_params["model"], final_response.usage)
        except anthropic.APIError as e:
            if "tool" in str(e).lower():
                raise RuntimeError(f"Tool calling continuation API error: {str(e)}") from e
            else:
                raise RuntimeError(f"Anthropic API error during continuation: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during continuation API call: {str(e)}") from e
        
        # Handle the final response (could potentially have more tool calls)
        stop_reason = getattr(final_response, "stop_reason", None)
        
        if stop_reason == "tool_use":
            # Recursive tool calling - handle additional tool calls
            return await self._handle_tool_use_response(final_response, original_prompt, conversation_state)
        elif stop_reason == "end_turn":
            return self._handle_text_response(final_response)
        else:
            raise RuntimeError(f"Unexpected stop_reason in continuation: {stop_reason!r}")

    def _convert_messages_to_api_format(self, messages: MessageList) -> tuple:
        """Generator for converting messages to the format required by the Anthropic API."""
        for msg in messages:
            if isinstance(msg, TextMessage):
                yield {"role": msg.role.value, "content": msg.text}

            elif isinstance(msg, AgentMessage):
                continue   # these are already handled via the system parameter on the API call

            elif isinstance(msg, ToolCallMessage):
                # For the new unified ToolCallMessage, we need to handle both the tool call and result
                content = []
                
                # Add any assistant message text
                if msg.message:
                    content.append({"type": "text", "text": msg.message})
                
                # Add the tool use
                content.append({
                    "type": "tool_use",
                    "id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "input": msg.arguments
                })
                
                yield {
                    "role": "assistant",
                    "content": content
                }
                
                # Add the tool result as a separate user message
                result_content = str(msg.result) if msg.result is not None else str(msg.error)
                yield {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": result_content
                        }
                    ]
                }

            # elif isinstance(msg, AudioMessage):
            #     # In a real implementation, this would encode the audio file
            #     yield {
            #         "role": msg.role.value,
            #         "content": [
            #             {"type": "audio", "audio_url": f"file://{msg.content}"}
            #         ]
            #     }

            elif isinstance(msg, ImageMessage):
                # In a real implementation, this would encode the image file
                yield {
                    "role": msg.role.value,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": msg.media_type,
                                "data": msg.base64_data,
                            }
                        }
                    ]
                }

            else:
                raise ValueError(f"Unsupported message type: {type(msg).__name__}")
