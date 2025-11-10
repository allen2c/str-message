import json
import logging
import typing

from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_audio_param import (
    InputAudio,
    ResponseInputAudioParam,
)
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_param import (
    Message as ResponseInputMessageParam,
)
from openai.types.responses.response_input_param import (
    ResponseInputParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from openai.types.responses.response_reasoning_item_param import (
    Content as ResponseReasoningContentParam,
)
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)
from openai.types.responses.response_reasoning_item_param import (
    Summary as ResponseReasoningSummaryParam,
)

from str_message import (
    CONTENT_AUDIO_TYPE,
    CONTENT_IMAGE_ID_TYPE,
    CONTENT_IMAGE_URL_TYPE,
    CONTENT_TEXT_TYPE,
    AssistantMessage,
    DeveloperMessage,
    Message,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)


def messages_to_response_input_param(messages: list["Message"]) -> ResponseInputParam:
    return list[ResponseInputItemParam](messages_gen_response_input_param(messages))


def messages_gen_response_input_param(
    messages: list["Message"],
) -> typing.Generator[ResponseInputItemParam, None, None]:
    for message in messages:
        if isinstance(message, UserMessage):
            for content_part in message.content_parts:
                if content_part.type == CONTENT_TEXT_TYPE:
                    yield EasyInputMessageParam(role="user", content=content_part.value)
                elif content_part.type == CONTENT_AUDIO_TYPE:
                    yield ResponseInputMessageParam(
                        role="user",
                        content=[
                            ResponseInputAudioParam(
                                type="input_audio",
                                input_audio=InputAudio(
                                    data=content_part.value, format="wav"
                                ),
                            )  # type: ignore
                        ],
                    )
                elif content_part.type == CONTENT_IMAGE_URL_TYPE:
                    yield ResponseInputMessageParam(
                        role="user",
                        content=[
                            ResponseInputImageParam(
                                detail="auto",
                                type="input_image",
                                image_url=content_part.value,
                            )
                        ],
                    )
                elif content_part.type == CONTENT_IMAGE_ID_TYPE:
                    yield ResponseInputMessageParam(
                        role="user",
                        content=[
                            ResponseInputImageParam(
                                detail="auto",
                                type="input_image",
                                file_id=content_part.value,
                            )
                        ],
                    )
                else:
                    raise ValueError(
                        f"Unsupported content part type: {content_part.type}"
                    )

        elif isinstance(message, SystemMessage):
            yield EasyInputMessageParam(role="system", content=message.content)

        elif isinstance(message, DeveloperMessage):
            yield EasyInputMessageParam(role="developer", content=message.content)

        elif isinstance(message, AssistantMessage):
            for content_part in message.content_parts:
                if content_part.type == CONTENT_TEXT_TYPE:
                    yield ResponseOutputMessageParam(
                        id=message.id,
                        role="assistant",
                        content=[
                            ResponseOutputTextParam(
                                annotations=[],
                                text=content_part.value,
                                type="output_text",
                            )
                        ],
                        status="completed",
                        type="message",
                    )
                else:
                    raise ValueError(
                        f"Unsupported content part type: {content_part.type}"
                    )

        elif isinstance(message, ToolCallMessage):
            yield ResponseFunctionToolCallParam(
                arguments=message.tool_call_arguments,
                call_id=message.tool_call_id,
                name=message.tool_name,
                type="function_call",
            )

        elif isinstance(message, ToolCallOutputMessage):
            yield FunctionCallOutput(
                call_id=message.tool_call_id,
                output=message.content,
                type="function_call_output",
            )

        elif isinstance(message, ReasoningMessage):
            summary: typing.List[ResponseReasoningSummaryParam] = []
            content: typing.List[ResponseReasoningContentParam] = []
            try:
                reasoning_data = json.loads(message.content)
                summary = reasoning_data.get("summary") or []
                content = reasoning_data.get("content") or []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse reasoning content: {message.content}")
            yield ResponseReasoningItemParam(
                id=message.id,
                summary=summary,
                type="reasoning",
                content=content,
            )

        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
