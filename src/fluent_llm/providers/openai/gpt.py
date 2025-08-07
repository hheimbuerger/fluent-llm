from typing import Any, Type, Tuple, override
from ...usage_tracker import tracker
from ...messages import Message, AudioMessage, ImageMessage, ResponseType, TextMessage, AgentMessage, MessageList
import openai
import base64
from io import BytesIO
import PIL.Image
from pydantic import BaseModel
from ...exceptions import *
from ..provider import LLMProvider, LLMModel
from decimal import Decimal


class OpenAIProvider(LLMProvider):    
    def get_models(self) -> Tuple[LLMModel]:
        return (
            LLMModel(
                name="gpt-4o-mini",
                text_input=True,
                image_input=True,
                audio_input=False,
                text_output=True,
                image_output=False,
                audio_output=False,
                structured_output=True,
                price_per_million_text_tokens_input=Decimal("0.15"),
                price_per_million_text_tokens_output=Decimal("0.60"),
                price_per_million_image_tokens_input=Decimal("1"),
                price_per_million_image_tokens_output=Decimal('NaN'),  # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),  # Not available
                additional_pricing={},
            ),
            LLMModel(
                name="gpt-4.1-mini",
                text_input=True,
                image_input=True,
                audio_input=False,
                text_output=True,
                image_output=True,
                audio_output=False,
                structured_output=True,
                price_per_million_text_tokens_input=Decimal("0.15"),
                price_per_million_text_tokens_output=Decimal("0.60"),
                price_per_million_image_tokens_input=Decimal("1"),
                price_per_million_image_tokens_output=Decimal('NaN'),  # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),  # Not available
                additional_pricing={},
            ),
            LLMModel(
                name="gpt-4o-mini-audio",
                text_input=True,
                image_input=False,
                audio_input=True,
                text_output=True,
                image_output=False,
                audio_output=True,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("0.15"),  # FILL ME
                price_per_million_text_tokens_output=Decimal("0.60"), # FILL ME
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal("10.00"), # FILL ME
                price_per_million_audio_tokens_output=Decimal("20.00"),# FILL ME
                additional_pricing={},
            ),
            LLMModel(
                name="gpt-image-1",
                text_input=True,
                image_input=True,
                audio_input=False,
                text_output=False,
                image_output=True,
                audio_output=False,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("5.00"),  # FILL ME
                price_per_million_text_tokens_output=Decimal('10.00'), # FIXME: actually unknown, not mentioned on https://platform.openai.com/docs/pricing#latest-models
                price_per_million_image_tokens_input=Decimal('10.00'),    # Not available
                price_per_million_image_tokens_output=Decimal('40.00'),   # Not available
                price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
                price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
                additional_pricing={},
            ),
        )

    def get_token_type_to_price_mapping(self) -> dict:
        """Get OpenAI-specific token type mapping.
        
        Returns:
            Dictionary mapping OpenAI token types to pricing field base names.
        """
        return {
            'input_tokens': 'price_per_million_text_tokens_input',
            'output_tokens': 'price_per_million_text_tokens_output',
            'input_tokens_details.image_tokens': 'price_per_million_image_tokens_input',
            'input_tokens_details.audio_tokens': 'price_per_million_audio_tokens_input',
            'input_tokens_details.text_tokens': 'price_per_million_text_tokens_input',
            'output_tokens_details.image_tokens': 'price_per_million_image_tokens_output',
            'output_tokens_details.audio_tokens': 'price_per_million_audio_tokens_output',
        }

    async def prompt_via_api(
        self,
        model: str,
        messages: MessageList,
        expect_type: ResponseType | Type[BaseModel],
        **kwargs: Any
    ) -> Any:
        """
        Make an async call to the OpenAI API with the given messages and return the appropriate response.
        The model is always inferred from the input and expected output.
        Uses the new OpenAI Responses interface.
        """
        # create client
        client = openai.AsyncOpenAI()

        # Prepare messages for the responses API
        openai_messages = [self._convert_to_openai_format(msg) for msg in messages]

        # Prepare API parameters
        api_params = {
            "model": model,
            "input": openai_messages,
            **kwargs,
        }

        # Determine the type of request
        is_structured_output = isinstance(expect_type, type) and issubclass(expect_type, BaseModel)
        is_image_generation = expect_type == ResponseType.IMAGE

        # Handle image generation using the dedicated images API
        if is_image_generation:
            # Call the images API
            response = await client.images.generate(
                model=model,
                prompt=messages.merge_all_text(),
                n=1,
                quality="low",
                size="1024x1024",
                **kwargs
            )
            # Track usage for image generation
            tracker.track_usage(self, model, response.usage)

            if not hasattr(response, 'data') or not response.data:
                raise ValueError("No data in response for image generation")

            # The output should contain the image generation call
            image_output = response.data[0].b64_json

            # Convert base64 to Pillow Image
            image_data = base64.b64decode(image_output)
            return PIL.Image.open(BytesIO(image_data))

        # Handle non-image requests using the responses API
        elif is_structured_output:
            # For structured output, ensure we get JSON
            api_params["text_format"] = expect_type
            response = await client.responses.parse(**api_params)
        else:
            response = await client.responses.create(**api_params)

        # Check response status
        if response.status != 'completed':
            error_msg = f"API call failed with status: {response.status}"
            if hasattr(response, 'error'):
                error_msg += f" - {response.error}"
            raise RuntimeError(error_msg)

        # Track API usage - pass the entire response object
        tracker.track_usage(self, response.model, response.usage)

        # Verify finish_reason – only 'stop' is considered success.
        finish_reason = getattr(response, "finish_reason", "stop")
        if finish_reason != "stop":
            if finish_reason == "length":
                raise NotImplementedError(
                    "OpenAI generation stopped due to length; continuation not implemented."
                )
            raise RuntimeError(f"OpenAI API returned unexpected finish_reason: {finish_reason!r}")

        # Handle TEXT output
        if expect_type == ResponseType.TEXT:
            # The responses API returns a list of choices, each with a message
            return response.output_text

        # Handle JSON/structured output
        if is_structured_output:
            # Check for refusal in the response
            if hasattr(response, 'refusal') and response.refusal is not None:
                raise LLMRefusalError(str(response.refusal))

            if not hasattr(response, 'output_parsed') or response.output_parsed is None:
                raise ValueError("No structured output found in the response")

            return response.output_parsed

        raise NotImplementedError(f"ResponseType {expect_type} not supported yet in call_llm_api.")

    def _convert_to_openai_format(self, message: Message) -> dict:
        """Convert a Message to the OpenAI API format."""
        if isinstance(message, TextMessage) or isinstance(message, AgentMessage):
            return {"role": message.role.value, "content": message.content}

        elif isinstance(message, AudioMessage):
            # encode the audio file
            return {
                "role": message.role.value,
                "content": [
                    {"type": "audio", "audio_url": f"file://{message.content}"}
                ]
            }

        elif isinstance(message, ImageMessage):
            # encode the image file
            return {
                "role": message.role.value,
                "content": [
                    {"type": "input_image", "image_url": message.base64_data_url}
                ]
            }

        raise ValueError(f"Unsupported message type: {type(message).__name__}")

        raise ValueError(f"Unsupported message type: {type(message).__name__}")
