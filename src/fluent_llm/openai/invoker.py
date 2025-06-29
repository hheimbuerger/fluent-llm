from typing import Any, List, Union, get_origin, get_args
from ..usage_tracker import tracker
from ..messages import Message, AudioMessage, ImageMessage, ResponseType, TextMessage, AgentMessage
from .models import select_and_validate_model
import inspect
import openai
import base64


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
    messages: List[Message],
    expect_type: ResponseType,
    **kwargs: Any
) -> Any:
    """
    Make an async call to the OpenAI API with the given messages and return the appropriate response.
    The model is always inferred from the input and expected output.
    Uses the new OpenAI Responses interface.
    """
    # select model
    selected_model = select_and_validate_model(messages, expect_type)

    # create client if not specified
    client = client or openai.AsyncOpenAI()

    # Prepare messages for the responses API
    openai_messages = [_convert_to_openai_format(msg) for msg in messages]

    # Prepare API parameters
    api_params = {
        "model": selected_model,
        "input": openai_messages,
        **kwargs,
    }

    # Add tools parameter if we're expecting an image
    if expect_type == ResponseType.IMAGE:
        api_params["tools"] = [{"type": "image_generation"}]

    # Call the OpenAI responses API (not chat completions)
    response = await client.responses.create(**api_params)

    # Check response status
    if hasattr(response, 'status') and response.status != 'success':
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

    raise NotImplementedError(f"ResponseType {expect_type} not supported yet in call_llm_api.")
