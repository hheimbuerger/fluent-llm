from typing import Any, Type
from ..usage_tracker import tracker
from ..messages import Message, AudioMessage, ImageMessage, ResponseType, TextMessage, AgentMessage, MessageList
import openai
import base64
from io import BytesIO
import PIL.Image
from pydantic import BaseModel

from ..exceptions import LLMRefusalError


def _convert_to_openai_format(message: Message) -> dict:
    """Convert a Message to the OpenAI API format."""
    if isinstance(message, TextMessage) or isinstance(message, AgentMessage):
        return {"role": message.role.value, "content": message.content}

    elif isinstance(message, AudioMessage):
        # In a real implementation, this would encode the audio file
        return {
            "role": message.role.value,
            "content": [
                {"type": "audio", "audio_url": f"file://{message.content}"}
            ]
        }

    elif isinstance(message, ImageMessage):
        # In a real implementation, this would encode the image file
        return {
            "role": message.role.value,
            "content": [
                {"type": "input_image", "image_url": message.base64_data_url}
            ]
        }

    raise ValueError(f"Unsupported message type: {type(message).__name__}")


async def call_llm_api(
    client: Any | None,
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
    # create client if not specified
    client = client or openai.AsyncOpenAI()

    # Prepare messages for the responses API
    openai_messages = [_convert_to_openai_format(msg) for msg in messages]

    # Prepare API parameters
    api_params = {
        "model": model,
        "input": openai_messages,
        **kwargs,
    }

    # Determine if this is a structured output request
    is_structured_output = isinstance(expect_type, type) and issubclass(expect_type, BaseModel)

    # Configure API parameters based on expected output type
    if expect_type == ResponseType.IMAGE:
        api_params["tools"] = [{"type": "image_generation"}]
    elif is_structured_output:
        # For structured output, ensure we get JSON
        api_params["text_format"] = expect_type

    # Call the appropriate OpenAI API method based on the expected output type
    if is_structured_output:
        # For structured output, use parse()
        response = await client.responses.parse(**api_params)
    else:
        # For other types, use create()
        response = await client.responses.create(**api_params)

    # Check response status
    if response.status != 'completed':
        error_msg = f"API call failed with status: {response.status}"
        if hasattr(response, 'error'):
            error_msg += f" - {response.error}"
        raise RuntimeError(error_msg)

    # Track API usage - pass the entire response object
    tracker.track_usage(response)

    # Handle image generation response
    if expect_type == ResponseType.IMAGE:
        if not hasattr(response, 'output') or not response.output:
            raise ValueError("No output in response for image generation")

        # The output should contain the image generation call
        image_output = None
        for output in response.output:
            if hasattr(output, 'type') and output.type == "image_generation_call":
                image_output = output
                break

        if not image_output:
            raise ValueError("No image generation call found in the response")

        # Return the image generation call output
        return base64.b64decode(image_output)

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
