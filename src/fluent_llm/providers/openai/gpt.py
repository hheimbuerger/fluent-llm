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
                name="gpt-4o-mini-audio-preview",
                text_input=True,
                image_input=False,
                audio_input=True,
                text_output=True,
                image_output=False,
                audio_output=True,
                structured_output=False,
                price_per_million_text_tokens_input=Decimal("0.15"),
                price_per_million_text_tokens_output=Decimal("0.60"),
                price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
                price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
                price_per_million_audio_tokens_input=Decimal("10.00"),
                price_per_million_audio_tokens_output=Decimal("20.00"),
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

        # Determine the type of request
        is_structured_output = isinstance(expect_type, type) and issubclass(expect_type, BaseModel)
        is_image_generation = expect_type == ResponseType.IMAGE
        is_audio_involved = any(isinstance(msg, AudioMessage) for msg in messages) or expect_type == ResponseType.AUDIO

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

        elif is_audio_involved:
            # fall back to the Chat Completion API for audio-in :roll:
            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                modalities=['text', 'audio'] if expect_type == ResponseType.AUDIO else ['text'],
                audio={
                    "format": "wav",
                    "voice": "nova",    # TODO: pick voice
                } if expect_type == ResponseType.AUDIO else None,
                **kwargs,
            )

        # Handle non-image requests using the responses API
        else:
            api_params = {
                "model": model,
                "input": openai_messages,
                **kwargs,
            }
            if is_structured_output:
                # For structured output, ensure we get JSON
                api_params["text_format"] = expect_type
                response = await client.responses.parse(**api_params)
            else:
                response = await client.responses.create(**api_params)

        # Track API usage - pass the entire response object
        final_model = getattr(response, 'model', model)   # the Image API doesn't include the model in the response, all others do
        tracker.track_usage(self, final_model, response.usage)

        # Check response status and finish_reason of the different APIs
        if isinstance(response, openai.types.responses.response.Response):   # Responses API
            status = response.status
            finish_reason = None   # I can't find the finish reason anywhere on this object
        elif isinstance(response, openai.types.images_response.ImagesResponse):   # Image API
            status = None   # the ImagesResponse has no status or reason at all
            finish_reason = None
        elif isinstance(response, openai.types.chat.chat_completion.ChatCompletion):   # Chat Completions API
            status = None
            finish_reason = (response.choices[0].finish_reason if len(response.choices) > 0 else None)
        else:
            raise NotImplementedError('Unexpected response type')

        # Verify finish_reason – only 'stop' is considered success.
        if finish_reason is not None and finish_reason != "stop":
            if finish_reason == "length":
                raise NotImplementedError(
                    "OpenAI generation stopped due to length; continuation not implemented."
                )
            raise RuntimeError(f"OpenAI API returned unexpected finish_reason: {finish_reason!r}")
        if status is not None and status != 'completed':
            error_msg = f"API call failed with status: {status}"
            if hasattr(response, 'error'):
                error_msg += f" - {response.error}"
            raise RuntimeError(error_msg)

        # if every check passed, we can now extract and convert the result
        return self._extract_result_from_response(response, expect_type)

    def _extract_result_from_response(
        self,
        response: Any,   # TODO: figure out correct canonical types here: Response API + Chat Completion API + Image API
        expect_type: ResponseType | Type[BaseModel]
    ) -> Any:
        """
        Extract the result from the OpenAI response based on the expected type.

        Args:
            response: The OpenAI response object
            expect_type: The expected type of the response

        Returns:
            The extracted result from the response
        """
        # handle TEXT output
        if expect_type == ResponseType.TEXT:
            # The Chat Completions API returns a list of choices, each with a message
            if isinstance(response, openai.types.chat.chat_completion.ChatCompletion):
                assert len(response.choices) == 1
                return response.choices[0].message.content
            else:
                return response.output_text

        # handle JSON/structured output
        if isinstance(expect_type, type) and issubclass(expect_type, BaseModel):
            # Check for refusal in the response
            if hasattr(response, 'refusal') and response.refusal is not None:
                raise LLMRefusalError(str(response.refusal))

            if not hasattr(response, 'output_parsed') or response.output_parsed is None:
                raise ValueError("No structured output found in the response")

            return response.output_parsed

        # handle IMAGE output
        if expect_type == ResponseType.IMAGE:
            if not hasattr(response, 'data') or not response.data:
                raise ValueError("No data in response for image generation")

            # The output should contain the image generation call
            if hasattr(response, 'data') and len(response.data) == 1:
                image_output = response.data[0].b64_json
                image_data = base64.b64decode(image_output)
                # Convert base64 to Pillow Image
                return PIL.Image.open(BytesIO(image_data))

        # handle AUDIO output
        elif expect_type == ResponseType.AUDIO:
            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("No choices in response for audio generation")

            choices = response.choices
            if not len(choices) > 0 or not hasattr(choices[0], 'message') or not choices[0].message:
                raise ValueError("No message in response for audio generation")

            message = choices[0].message
            if not hasattr(message, 'audio') or not message.audio:
                raise ValueError("No audio in response for audio generation")

            audio_data = base64.b64decode(message.audio.data)
            return (message.audio.transcript, audio_data)

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
                    {"type": "input_audio",
                     "input_audio": {
                        "data": message.content_b64,
                        "format": "mp3"
                     } }
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
