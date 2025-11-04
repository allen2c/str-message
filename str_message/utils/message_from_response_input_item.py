from openai.types.responses.response_input_item import ResponseInputItem

from str_message import Message


def message_from_response_input_item(data: ResponseInputItem) -> Message:
    raise NotImplementedError
