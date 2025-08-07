from ..provider import LLMProvider, LLMModel
from typing import Tuple, Any, Type
from decimal import Decimal
from ...messages import Message, MessageList, ResponseType, TextMessage, ImageMessage, AgentMessage
from ...exceptions import *
from ...usage_tracker import tracker
from pydantic import BaseModel
import anthropic


class AnthropicProvider(LLMProvider):
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
                name="claude-sonnet-4",
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
                name="claude-opus-4",
                text_input=True,
                text_output=True,
                image_input=True,
                image_output=False,
                audio_input=False,
                audio_output=False,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("15.00"),
                price_per_million_text_tokens_output=Decimal("75.00"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
        )

    async def prompt_via_api(self, model: str, messages: MessageList, expect_type: ResponseType | Type[BaseModel], **kwargs: Any) -> Any:
        client = anthropic.AsyncAnthropic()

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20000,
            temperature=1,
            system=messages.merge_all_agent(),
            messages=tuple(self._convert_messages_to_api_format(messages)),
        )

        tracker.track_usage(self, model, response.usage)

        # Verify stop_reason – only 'end_turn' is considered success.
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "end_turn":
            if stop_reason == "max_tokens":
                # Continuation logic not yet implemented; treat as failure.
                raise NotImplementedError(
                    "Anthropic generation stopped due to max_tokens; continuation not implemented."
                )
            raise RuntimeError(f"Anthropic API returned unexpected stop_reason: {stop_reason!r}")

        # Validate and extract the single text block.
        if len(response.content) != 1:
            raise RuntimeError(f"Expected exactly one content block, got {len(response.content)}")
        response_message = response.content[0]
        if response_message.type != "text":
            raise RuntimeError(f"Unexpected block type {response_message.type!r}, expected 'text'")

        return response_message.text

    def _convert_messages_to_api_format(self, messages: MessageList) -> tuple:
        """Generator for converting messages to the format required by the Anthropic API."""
        for msg in messages:
            if isinstance(msg, TextMessage):
                yield {"role": msg.role.value, "content": msg.text}

            elif isinstance(msg, AgentMessage):
                continue   # these are already handled via the system parameter on the API call

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
