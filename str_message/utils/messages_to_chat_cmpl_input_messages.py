import typing

import durl
from openai.types.chat.chat_completion_assistant_message_param import (
    Audio as ChatCompletionAssistantMessageAudioParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
    InputAudio,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function as ChatCompletionMessageFunctionToolCallFunctionParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from str_message import (
    CONTENT_AUDIO_TYPE,
    CONTENT_FILE_FILENAME_TYPE,
    CONTENT_FILE_ID_TYPE,
    CONTENT_FILE_URL_TYPE,
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


def messages_to_chat_cmpl_input_messages(
    messages: list[Message],
) -> list[ChatCompletionMessageParam]:
    return list[ChatCompletionMessageParam](
        messages_gen_chat_cmpl_input_messages(messages)
    )


def messages_gen_chat_cmpl_input_messages(
    messages: list[Message],
) -> typing.Generator[ChatCompletionMessageParam, None, None]:
    for message in messages:
        if isinstance(message, UserMessage):
            for content_part in message.content_parts:
                if content_part.type == CONTENT_TEXT_TYPE:
                    yield ChatCompletionUserMessageParam(
                        role="user",
                        content=content_part.value,
                    )
                elif content_part.type == CONTENT_AUDIO_TYPE:
                    yield ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartInputAudioParam(
                                type="input_audio",
                                input_audio=InputAudio(
                                    data=durl.DataURL.from_url(content_part.value).data,
                                    format="wav",
                                ),
                            )
                        ],
                    )
                elif content_part.type == CONTENT_IMAGE_URL_TYPE:
                    yield ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url=ImageURL(url=content_part.value),
                            )
                        ],
                    )
                else:
                    raise ValueError(
                        f"Unsupported content part type: {content_part.type}"
                    )

        elif isinstance(message, SystemMessage):
            yield ChatCompletionSystemMessageParam(
                role="system", content=message.content
            )

        elif isinstance(message, DeveloperMessage):
            yield ChatCompletionDeveloperMessageParam(
                role="developer", content=message.content
            )

        elif isinstance(message, AssistantMessage):
            for content_part in message.content_parts:
                if content_part.type == CONTENT_TEXT_TYPE:
                    yield ChatCompletionAssistantMessageParam(
                        role="assistant", content=content_part.value
                    )

                elif content_part.type == CONTENT_AUDIO_TYPE:
                    audio_id = (message.metadata or {}).get("audio_id")
                    if audio_id:
                        yield ChatCompletionAssistantMessageParam(
                            role="assistant",
                            audio=ChatCompletionAssistantMessageAudioParam(id=audio_id),
                        )
                    else:
                        raise ValueError(
                            f"Audio ID is required for audio content part: {content_part}"  # noqa: E501
                        )
                    del audio_id

                elif content_part.type == (
                    CONTENT_IMAGE_URL_TYPE,
                    CONTENT_IMAGE_ID_TYPE,
                    CONTENT_FILE_FILENAME_TYPE,
                    CONTENT_FILE_ID_TYPE,
                    CONTENT_FILE_URL_TYPE,
                ):
                    raise ValueError(
                        "The image and file message type not supported in the chat completion assistant message"  # noqa: E501
                    )

                else:
                    raise ValueError(
                        f"Unsupported content part type: {content_part.type}"
                    )

        elif isinstance(message, ToolCallMessage):
            yield ChatCompletionAssistantMessageParam(
                role="assistant",
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=message.tool_call_id,
                        function=ChatCompletionMessageFunctionToolCallFunctionParam(
                            name=message.tool_name,
                            arguments=message.tool_call_arguments,
                        ),
                        type="function",
                    )
                ],
            )

        elif isinstance(message, ToolCallOutputMessage):
            yield ChatCompletionToolMessageParam(
                role="tool",
                content=message.content,
                tool_call_id=message.tool_call_id,
            )

        elif isinstance(message, ReasoningMessage):
            yield ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message.content,
            )

        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
