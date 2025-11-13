import typing

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from str_message import MessageTypes


def message_from_chat_cmpl(
    data: ChatCompletion,
) -> typing.List[MessageTypes]:
    from str_message.utils.message_from_chat_cmpl_message import (
        message_from_chat_cmpl_message,
    )

    might_choice: typing.Optional[Choice] = next(
        (choice for choice in data.choices),
        None,
    )

    if might_choice is None:
        raise ValueError("No choice found in ChatCompletion")

    choice: Choice = might_choice
    message: ChatCompletionMessage = choice.message

    return message_from_chat_cmpl_message(message, msg_id=data.id)
