import typing

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from str_message import (
    AssistantMessage,
    ReasoningMessage,
    ToolCallMessage,
)


def message_from_chat_cmpl(
    data: ChatCompletion,
) -> AssistantMessage | ReasoningMessage | ToolCallMessage:
    might_choice: typing.Optional[Choice] = next(
        (choice for choice in data.choices),
        None,
    )

    if might_choice is None:
        raise ValueError("No choice found in ChatCompletion")

    choice: Choice = might_choice
    message: ChatCompletionMessage = choice.message

    if message.tool_calls:
        _tool_call = message.tool_calls[0]  # Only one tool call is supported
        if _tool_call.type == "function":
            return ToolCallMessage(
                id=data.id,
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
            id=data.id,
            role="assistant",
            content=reasoning,
            channel="analysis",
        )

    else:
        return AssistantMessage(id=data.id, content=message.content or "")
