from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from str_message import Message


def messages_to_chat_cmpl_input_messages(
    messages: list[Message],
) -> list[ChatCompletionMessageParam]:
    pass  # TODO: Implement
