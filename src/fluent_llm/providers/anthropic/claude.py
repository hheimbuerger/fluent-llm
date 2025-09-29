from typing import Tuple, Any
from decimal import Decimal
import anthropic

from ..provider import LLMProvider, LLMModel
from ...messages import MessageList, TextMessage, ImageMessage, AgentMessage, ToolCallMessage, ToolResultMessage
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
        # Validate and extract the single text block.
        if len(response.content) != 1:
            raise RuntimeError(f"Expected exactly one content block, got {len(response.content)}")
        response_message = response.content[0]
        if response_message.type != "text":
            raise RuntimeError(f"Unexpected block type {response_message.type!r}, expected 'text'")

        return response_message.text

    async def _handle_tool_use_response(self, response, p: Prompt, conversation_state=None) -> Any:
        """Handle a tool use response from Anthropic.
        
        This method processes tool calls, executes them, and continues the conversation.
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
        
        # Add any text content that came with tool calls to conversation state
        if text_content.strip() and conversation_state is not None:
            from ...messages import TextMessage, Role
            text_msg = TextMessage(text=text_content.strip(), role=Role.ASSISTANT)
            conversation_state.message_history.append(text_msg)
            conversation_state.internal_session.append(text_msg.to_dict())
        
        # Add tool calls to conversation state
        for tool_call in tool_calls:
            if conversation_state is not None:
                from ...messages import ToolCallMessage
                tool_call_msg = ToolCallMessage(
                    tool_name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    arguments=tool_call["input"]
                )
                conversation_state.message_history.append(tool_call_msg)
                conversation_state.internal_session.append(tool_call_msg.to_dict())
        
        if not tool_calls:
            raise RuntimeError("Expected tool calls in tool_use response, but found none")
        
        # Execute tool calls and collect results
        tool_results = []
        for tool_call in tool_calls:
            try:
                # Find the tool by name
                tool = None
                for t in p.tools:
                    if t.name == tool_call["name"]:
                        tool = t
                        break
                
                if tool is None:
                    result = f"Error: Tool '{tool_call['name']}' not found in available tools: {[t.name for t in p.tools]}"
                else:
                    # Validate tool call arguments
                    try:
                        # Execute the tool function
                        result = tool.function(**tool_call["input"])
                    except TypeError as e:
                        if "unexpected keyword argument" in str(e) or "missing" in str(e):
                            result = f"Error: Invalid arguments for tool '{tool_call['name']}': {str(e)}"
                        else:
                            raise
                
                tool_results.append({
                    "tool_use_id": tool_call["id"],
                    "content": str(result)
                })
                
                # Add tool result to conversation state if provided
                if conversation_state is not None:
                    from ...messages import ToolResultMessage
                    tool_result_msg = ToolResultMessage(
                        tool_call_id=tool_call["id"],
                        result=result
                    )
                    conversation_state.message_history.append(tool_result_msg)
                    conversation_state.internal_session.append(tool_result_msg.to_dict())
                
            except Exception as e:
                # Log the full error for debugging but provide a clean error message to the model
                import traceback
                error_details = traceback.format_exc()
                print(f"Tool execution error for '{tool_call['name']}': {error_details}")
                
                error_result = f"Error executing tool '{tool_call['name']}': {str(e)}"
                tool_results.append({
                    "tool_use_id": tool_call["id"],
                    "content": error_result
                })
                
                # Add error result to conversation state if provided
                if conversation_state is not None:
                    from ...messages import ToolResultMessage
                    tool_result_msg = ToolResultMessage(
                        tool_call_id=tool_call["id"],
                        result=error_result
                    )
                    conversation_state.message_history.append(tool_result_msg)
                    conversation_state.internal_session.append(tool_result_msg.to_dict())
        
        # Continue the conversation with tool results
        return await self._continue_conversation_with_tool_results(
            p, response, tool_results, text_content, conversation_state
        )

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
                # Tool calls are handled as assistant messages with tool_use content
                yield {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": msg.tool_call_id,
                            "name": msg.tool_name,
                            "input": msg.arguments
                        }
                    ]
                }

            elif isinstance(msg, ToolResultMessage):
                # Tool results are handled as user messages with tool_result content
                yield {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": str(msg.result)
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
