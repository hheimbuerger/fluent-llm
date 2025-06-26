from typing import Any, List, Union, get_origin, get_args
from ..usage_tracker import tracker
from ..messages import Message, AudioMessage, ImageMessage, ResponseType, TextMessage, AgentMessage
from .models import select_and_validate_model
import inspect
import openai

def _has_audio_output(expect_type) -> bool:
    """Check if the expected output type is audio."""
    return expect_type == ResponseType.AUDIO

def _has_image_output(expect_type) -> bool:
    """Check if the expected output type is an image."""
    return expect_type == ResponseType.IMAGE

def _has_audio_input(messages: list[Union[dict, Message]]) -> bool:
    """Check if any message contains audio input."""
    for msg in messages:
        if isinstance(msg, AudioMessage):
            return True
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            if any(part.get("type") == "audio" for part in msg["content"]):
                return True
    return False

def _has_image_input(messages: list[Union[dict, Message]]) -> bool:
    """Check if any message contains image input."""
    for msg in messages:
        if isinstance(msg, ImageMessage):
            return True
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            if any(part.get("type") == "image_url" for part in msg["content"]):
                return True
    return False

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
    # Determine required capabilities
    require_image_input = any(isinstance(m, ImageMessage) or (isinstance(m, dict) and any(part.get("type") == "image_url" for part in m.get("content", []))) for m in messages)
    require_audio_input = any(isinstance(m, AudioMessage) or (isinstance(m, dict) and any(part.get("type") == "audio" for part in m.get("content", []))) for m in messages)
    require_image_output = expect_type == ResponseType.IMAGE
    require_audio_output = expect_type == ResponseType.AUDIO
    require_structured_output = False
    # If expect_type is a Pydantic BaseModel subclass or a structured_output_model kwarg is present, require structured output
    structured_output_model = kwargs.get("structured_output_model")
    if structured_output_model is not None or (inspect.isclass(expect_type) and hasattr(expect_type, "model_validate")):
        require_structured_output = True
    require_text_output = expect_type == ResponseType.TEXT or expect_type == ResponseType.JSON or require_structured_output

    selected_model = select_and_validate_model(messages, expect_type)

    # create client if not specified
    client = client or openai.AsyncOpenAI()

    # Prepare messages for the responses API
    openai_messages = [_convert_to_openai_format(msg) for msg in messages]

    # Call the OpenAI responses API (not chat completions)
    response = await client.responses.create(
        model=selected_model,
        input=openai_messages,
        **kwargs,
    )

    # TODO: check response.status!

    # Track API usage - pass the entire response object
    tracker.track_usage(response)

    # Handle TEXT output
    if expect_type == ResponseType.TEXT:
        # The responses API returns a list of choices, each with a message
        return response.output_text

    # Handle IMAGE output: return raw bytes
    if expect_type == ResponseType.IMAGE:
        # The responses API returns image bytes directly in the response (as per latest API)
        # If bytes are not directly available, check for a file or data property
        if hasattr(response, 'data') and isinstance(response.data, bytes):
            return response.data
        # Fallback: check for a file property
        if hasattr(response, 'file'):
            return response.file.read()
        raise RuntimeError("No image bytes found in OpenAI responses API response.")

    raise NotImplementedError(f"ResponseType {expect_type} not supported yet in call_llm_api.")
