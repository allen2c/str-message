from str_message import (
    AssistantMessage,
    DeveloperMessage,
    MessageTypes,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)
from str_message.types.chat_completion_messages import (
    ChatCompletionAssistantMessage,
    ChatCompletionDeveloperMessage,
    ChatCompletionFunctionMessage,
)
from str_message.types.chat_completion_messages import (
    ChatCompletionMessage as ChatCompletionInputMessage,
)
from str_message.types.chat_completion_messages import (
    ChatCompletionSystemMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)


def message_from_chat_cmpl_input_message(
    data: ChatCompletionInputMessage,
) -> MessageTypes:
    if isinstance(data, ChatCompletionDeveloperMessage):
        return DeveloperMessage(content=data.str_content())
    elif isinstance(data, ChatCompletionSystemMessage):
        return SystemMessage(content=data.str_content())
    elif isinstance(data, ChatCompletionUserMessage):
        return UserMessage(content=data.str_content())
    elif isinstance(data, ChatCompletionAssistantMessage):
        if data.tool_calls:
            _tool_call = data.tool_calls[0]  # Only one tool call is supported
            if _tool_call.type == "function":
                return ToolCallMessage(
                    content=(
                        f"[tool_call:{_tool_call.function.name}](#{_tool_call.id}):{_tool_call.function.arguments}"  # noqa: E501
                    ),
                    tool_call_id=_tool_call.id,
                    tool_name=_tool_call.function.name,
                    tool_call_arguments=_tool_call.function.arguments,
                )
            else:
                raise ValueError(f"Unsupported tool call type: {_tool_call.type}")
        else:
            return AssistantMessage(content=data.str_content())

    elif isinstance(data, ChatCompletionToolMessage):
        return ToolCallOutputMessage(
            content=data.str_content(), tool_call_id=data.tool_call_id
        )

    elif isinstance(data, ChatCompletionFunctionMessage):
        return ToolCallOutputMessage(
            content=data.str_content(), tool_call_id="__fake_id__"
        )

    else:
        raise ValueError(f"Unsupported message type: {type(data).__name__}")
