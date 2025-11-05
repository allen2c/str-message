from str_message import MessageTypes
from str_message.types.chat_completion_messages import (
    ChatCompletionMessage as ChatCompletionInputMessage,
)


def message_from_chat_cmpl_input_message(
    data: ChatCompletionInputMessage,
) -> MessageTypes:
    raise NotImplementedError
