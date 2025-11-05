from openai.types.chat.chat_completion_message import ChatCompletionMessage

from str_message import AssistantMessage, ToolCallMessage


def message_from_chat_cmpl_message(
    data: ChatCompletionMessage,
) -> AssistantMessage | ToolCallMessage:
    if data.tool_calls:
        _tool_call = data.tool_calls[0]  # Only one tool call is supported
        if _tool_call.type == "function":
            return ToolCallMessage(
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

    else:
        return AssistantMessage(content=data.content or "")
