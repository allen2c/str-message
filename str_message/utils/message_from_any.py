import typing

import durl
import pydantic
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.responses.response_output_item import ResponseOutputItem

from str_message import (
    ANY_MESSAGE_TYPES,
    Message,
    MessageTypes,
    ResponseInputItemAdapter,
    ResponseInputItemModels,
    ResponseOutputItemAdapter,
)


def message_from_any(
    data: ANY_MESSAGE_TYPES | ChatCompletion | ResponseOutputItem,
) -> MessageTypes:
    """Convert various data types into Message objects."""
    from str_message.types.chat_completion_messages import (
        ChatCompletionMessage as ChatCompletionInputMessage,
    )
    from str_message.types.chat_completion_messages import (
        ChatCompletionMessageAdapter,
    )
    from str_message.utils.message_from_chat_cmpl import (
        message_from_chat_cmpl,
    )
    from str_message.utils.message_from_chat_cmpl_input_message import (
        message_from_chat_cmpl_input_message,
    )
    from str_message.utils.message_from_chat_cmpl_message import (
        message_from_chat_cmpl_message,
    )
    from str_message.utils.message_from_response_input_item import (
        message_from_response_input_item,
    )
    from str_message.utils.message_from_response_output_item import (
        message_from_response_output_item,
    )

    # Message type
    if isinstance(data, Message):
        return data

    # String type
    if isinstance(data, str):
        return Message(role="user", content=data)

    # Data URL type
    if isinstance(data, durl.DataURL):
        return Message(role="user", content=str(data))

    # Chat completion model type
    if isinstance(data, ChatCompletion):
        return message_from_chat_cmpl(data)

    # Chat completion message model type
    if isinstance(data, ChatCompletionMessage):
        return message_from_chat_cmpl_message(data)

    # Chat completion input message model type
    if isinstance(data, ChatCompletionInputMessage):
        return message_from_chat_cmpl_input_message(
            ChatCompletionMessageAdapter.validate_json(data.model_dump_json())
        )

    # Response output item model type
    if item := (
        _return_response_output_item_model(data)
        if isinstance(data, pydantic.BaseModel)
        else None
    ):
        return message_from_response_output_item(item)

    # Response input item model type
    if isinstance(data, ResponseInputItemModels):
        return message_from_response_input_item(
            ResponseInputItemAdapter.validate_json(data.model_dump_json())
        )

    # Handle dict type
    if isinstance(data, typing.Dict):
        try:
            chat_cmpl = ChatCompletion.model_validate(data)
            return message_from_chat_cmpl(chat_cmpl)
        except pydantic.ValidationError:
            pass  # Not a ChatCompletion

        try:
            chat_cmpl_message = ChatCompletionMessage.model_validate(data)
            return message_from_chat_cmpl_message(chat_cmpl_message)
        except pydantic.ValidationError:
            pass  # Not a ChatCompletionMessage

        try:
            chat_cmpl_input_message = ChatCompletionMessageAdapter.validate_python(data)
            return message_from_chat_cmpl_input_message(chat_cmpl_input_message)
        except pydantic.ValidationError:
            pass  # Not a ChatCompletionInputMessage

        try:
            response_input_item = ResponseInputItemAdapter.validate_python(data)
            return message_from_response_input_item(response_input_item)
        except pydantic.ValidationError:
            pass  # Not a ResponseInputItem

        try:
            response_output_item = ResponseOutputItemAdapter.validate_python(data)
            return message_from_response_output_item(response_output_item)
        except pydantic.ValidationError:
            pass  # Not a ResponseOutputItem

    raise ValueError(f"Unsupported message type: {type(data).__name__}")


def _return_response_output_item_model(
    data: pydantic.BaseModel,
) -> ResponseOutputItem | None:
    try:
        return ResponseOutputItemAdapter.validate_json(data.model_dump_json())
    except pydantic.ValidationError:
        return None
