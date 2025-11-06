from openai.types.responses.easy_input_message import EasyInputMessage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_input_item import (
    FunctionCallOutput,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_input_item import Message as ResponseInputMessage
from openai.types.responses.response_input_item import (
    ResponseInputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from str_message import Message


def message_from_response_input_item(data: ResponseInputItem) -> Message:
    if isinstance(data, EasyInputMessage):
        pass

    elif isinstance(data, ResponseInputMessage):
        pass

    elif isinstance(data, ResponseOutputMessage):
        pass

    elif isinstance(data, ResponseFunctionToolCall):
        pass

    elif isinstance(data, FunctionCallOutput):
        pass

    elif isinstance(data, ResponseReasoningItem):
        pass

    elif isinstance(data, McpListTools):
        pass

    elif isinstance(data, McpCall):
        pass

    else:
        raise ValueError(f"Unsupported response input item: {data}")
