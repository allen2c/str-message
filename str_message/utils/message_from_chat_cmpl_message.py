import typing

import uuid_utils as uuid
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from str_message import AssistantMessage, ReasoningMessage, ToolCallMessage


def message_from_chat_cmpl_message(
    data: ChatCompletionMessage, *, msg_id: typing.Optional[str] = None
) -> AssistantMessage | ReasoningMessage | ToolCallMessage:
    msg_id = msg_id or str(uuid.uuid7())

    if data.tool_calls:
        _tool_call = data.tool_calls[0]  # Only one tool call is supported
        if _tool_call.type == "function":
            return ToolCallMessage(
                id=msg_id,
                role="assistant",
                content=(
                    f"[tool_call:{_tool_call.function.name}]"
                    + f"(#{_tool_call.id})"
                    + f":{_tool_call.function.arguments}"
                ),
                tool_call_id=_tool_call.id,
                tool_name=_tool_call.function.name,
                tool_call_arguments=_tool_call.function.arguments,
            )
        else:
            raise ValueError(f"Unsupported tool call type: {_tool_call.type}")

    elif reasoning := getattr(data, "reasoning", None):
        return ReasoningMessage(
            id=msg_id,
            role="assistant",
            content=reasoning,
            channel="analysis",
        )

    else:
        return AssistantMessage(id=msg_id, content=data.content or "")
