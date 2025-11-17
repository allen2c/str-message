# str_message/utils/messages_to_sharegpt.py
import hashlib
import logging
import pathlib
import typing

import durl
import pydantic
import requests
from openai.types.shared.function_definition import FunctionDefinition
from rich.pretty import pretty_repr

from str_message import (
    CONTENT_AUDIO_TYPE,
    CONTENT_IMAGE_URL_TYPE,
    CONTENT_TEXT_TYPE,
    AssistantMessage,
    DeveloperMessage,
    McpCallMessage,
    McpListToolsMessage,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)

if typing.TYPE_CHECKING:
    from str_message import MessageTypes

logger = logging.getLogger(__name__)

ListFunctionDefinitionAdapter = pydantic.TypeAdapter(typing.List[FunctionDefinition])


def messages_to_sharegpt(
    messages: typing.List["MessageTypes"],
    media_dir: pathlib.Path | str | None = None,
    *,
    tool_definitions: typing.List[FunctionDefinition] | None = None,
    tools_column: str = "tools",
    images_column: str = "images",
    videos_column: str = "videos",  # Not implemented
    audios_column: str = "audios",
    role_tag: str = "role",
    content_tag: str = "content",
    user_tag: str = "user",
    assistant_tag: str = "assistant",
    observation_tag: str = "observation",
    function_tag: str = "function_call",
    system_tag: str = "system",
    messages_tag: str = "messages",
) -> dict:
    """Convert messages to ShareGPT format with media extraction."""
    output = {}
    output_messages: list[dict] = []
    output_tools: list[dict] = []
    output_images: list[str] = []
    output_audios: list[str] = []
    output[messages_tag] = output_messages
    media_dir = pathlib.Path(media_dir) if media_dir else pathlib.Path("./data")
    media_dir.mkdir(parents=True, exist_ok=True)

    # Handle message content
    for msg in messages:
        if isinstance(msg, (SystemMessage, DeveloperMessage)):
            output_messages.append({role_tag: system_tag, content_tag: msg.content})

        elif isinstance(msg, UserMessage):
            for content in msg.content_parts:
                if content.type in (CONTENT_TEXT_TYPE,):
                    output_messages.append(
                        {role_tag: user_tag, content_tag: content.value}
                    )

                elif content.type in (CONTENT_AUDIO_TYPE,):
                    audio_data_url = durl.DataURL.from_url(content.value)
                    audio_bytes = audio_data_url.data_decoded_bytes
                    audio_filename = (
                        f"{hashlib.md5(audio_bytes).hexdigest()}"
                        + f".{durl.ExtensionMIMEType[audio_data_url.mime_type]}"
                    )
                    audio_filepath = media_dir.joinpath(audio_filename)
                    audio_filepath.mkdir(parents=True, exist_ok=True)
                    audio_filepath.write_bytes(audio_bytes)
                    output_messages.append({role_tag: user_tag, content_tag: "<audio>"})
                    output_audios.append(str(audio_filepath))

                elif content.type in (CONTENT_IMAGE_URL_TYPE,):
                    image_data_url = durl.DataURL.from_url(content.value)
                    if image_data_url.data.startswith("http"):
                        response = requests.get(image_data_url.data)
                        response.raise_for_status()
                    else:
                        image_bytes = image_data_url.data_decoded_bytes
                    image_filename = (
                        f"{hashlib.md5(image_bytes).hexdigest()}"
                        + f".{durl.ExtensionMIMEType[image_data_url.mime_type]}"
                    )
                    image_filepath = media_dir.joinpath(image_filename)
                    image_filepath.mkdir(parents=True, exist_ok=True)
                    image_filepath.write_bytes(image_bytes)
                    output_messages.append({role_tag: user_tag, content_tag: "<image>"})
                    output_images.append(str(image_filepath))

                else:
                    raise ValueError(
                        "Unsupported content type: "
                        + f"{content.type}"
                        + f"{pretty_repr(content.value, max_string=100)}"
                    )

        elif isinstance(msg, AssistantMessage):
            output_messages.append({role_tag: assistant_tag, content_tag: msg.content})

        elif isinstance(msg, ToolCallMessage):
            output_messages.append(
                {
                    role_tag: function_tag,
                    content_tag: {
                        "name": msg.tool_name,
                        "arguments": msg.tool_call_arguments,
                    },
                }
            )

        elif isinstance(msg, ToolCallOutputMessage):
            output_messages.append(
                {role_tag: observation_tag, content_tag: msg.content}
            )

        elif isinstance(msg, ReasoningMessage):
            output_messages.append(
                {role_tag: observation_tag, content_tag: msg.content}
            )

        elif isinstance(msg, McpCallMessage):
            output_messages.extend(
                [
                    {
                        role_tag: function_tag,
                        content_tag: {
                            "name": msg.mcp_call_name,
                            "arguments": msg.mcp_call_arguments,
                        },
                    },
                    {role_tag: observation_tag, content_tag: msg.content},
                ]
            )

        elif isinstance(msg, McpListToolsMessage):
            for m in msg.mcp_tools:
                output_tools.append(
                    FunctionDefinition(
                        name=m.name,
                        description=m.description or "",
                        parameters=dict(m.input_schema),  # type: ignore
                    ).model_dump()
                )

        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")

    # Handle tools definitions
    if tool_definitions:
        output_tools.extend(ListFunctionDefinitionAdapter.dump_python(tool_definitions))

    if output_tools:
        output[tools_column] = output_tools
    if output_images:
        output[images_column] = output_images
    if output_audios:
        output[audios_column] = output_audios
    return output
