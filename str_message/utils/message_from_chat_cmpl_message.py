from openai.types.chat.chat_completion_message import ChatCompletionMessage

from str_message import Message


def message_from_chat_cmpl_message(data: ChatCompletionMessage) -> Message:
    raise NotImplementedError
