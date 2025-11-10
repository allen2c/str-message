from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
    McpListToolsTool,
    ResponseOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from str_message import MessageTypes


def message_from_response_output_item(data: ResponseOutputItem) -> MessageTypes:
    if isinstance(data, ResponseOutputMessage):
        raise NotImplementedError

    elif isinstance(data, ResponseFileSearchToolCall):
        raise NotImplementedError

    elif isinstance(data, ResponseFunctionToolCall):
        raise NotImplementedError

    elif isinstance(data, ResponseFunctionWebSearch):
        raise NotImplementedError

    elif isinstance(data, ResponseComputerToolCall):
        raise NotImplementedError

    elif isinstance(data, ResponseReasoningItem):
        raise NotImplementedError

    elif isinstance(data, ImageGenerationCall):
        raise NotImplementedError

    elif isinstance(data, ResponseCodeInterpreterToolCall):
        raise NotImplementedError

    elif isinstance(data, LocalShellCall):
        raise NotImplementedError

    elif isinstance(data, McpCall):
        raise NotImplementedError

    elif isinstance(data, McpListTools):
        raise NotImplementedError

    elif isinstance(data, McpApprovalRequest):
        raise NotImplementedError

    elif isinstance(data, ResponseCustomToolCall):
        raise NotImplementedError

    else:
        raise ValueError(f"Unsupported response output item: {data}")
