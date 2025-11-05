from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from str_message import MessageTypes


def message_from_chat_cmpl_message(data: ChatCompletionMessage) -> MessageTypes:
    if data.tool_calls:
        _tool_call = data.tool_calls[0]  # Only one tool call is supported
        if _tool_call.type == "function":
            return Message(
                role="assistant",
                content=(
                    f"[tool_call:{_tool_call.function.name}]"
                    + f"(#{_tool_call.id})"
                    + f":{_tool_call.function.arguments}"
                ),
                call_id=_tool_call.id,
                tool_name=_tool_call.function.name,
                arguments=_tool_call.function.arguments,
            )
        else:
            raise ValueError(f"Unsupported tool call type: {_tool_call.type}")
    else:
        return Message(
            role="assistant",
            content=data.content or "",
        )
