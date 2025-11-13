import uuid_utils as uuid
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
from openai.types.responses.response_output_item import McpListToolsTool
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from str_message import (
    AssistantMessage,
    DeveloperMessage,
    McpCallMessage,
    McpListToolsMessage,
    Message,
    MessageTypes,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)


def message_from_response_input_item(data: ResponseInputItem) -> MessageTypes:
    from str_message.utils.response_input_content_to_str import (
        response_input_content_to_str,
    )

    if isinstance(data, EasyInputMessage):
        content_str = response_input_content_to_str(data.content)
        if data.role == "user":
            return UserMessage(content=content_str)
        elif data.role == "assistant":
            return AssistantMessage(content=content_str)
        elif data.role == "system":
            return SystemMessage(content=content_str)
        elif data.role == "developer":
            return DeveloperMessage(content=content_str)
        else:
            raise ValueError(f"Unsupported role: {data.role}")

    elif isinstance(data, ResponseInputMessage):
        content_str = response_input_content_to_str(data.content)
        if data.role == "user":
            return UserMessage(content=content_str)
        elif data.role == "system":
            return SystemMessage(content=content_str)
        elif data.role == "developer":
            return DeveloperMessage(content=content_str)
        else:
            raise ValueError(f"Unsupported role: {data.role}")

    elif isinstance(data, ResponseOutputMessage):
        content_str = response_input_content_to_str(data.content)
        if data.role == "assistant":
            return AssistantMessage(id=data.id, content=content_str)
        else:
            raise ValueError(f"Unsupported role: {data.role}")

    elif isinstance(data, ResponseFunctionToolCall):
        Message.set_tool_call(data.call_id, data)
        return ToolCallMessage(
            id=data.id or str(uuid.uuid7()),
            role="assistant",
            content=data.arguments,
            tool_call_id=data.call_id,
            tool_name=data.name,
            tool_call_arguments=data.arguments,
        )

    elif isinstance(data, FunctionCallOutput):
        _tool_call = Message.get_tool_call(data.call_id)
        _tool_name = (
            _tool_call.name
            if _tool_call is not None
            else "__can_not_tracing_tool_call__"
        )
        _tool_call_arguments = (
            _tool_call.arguments
            if _tool_call is not None
            else "__can_not_tracing_tool_call_arguments__"
        )
        return ToolCallOutputMessage(
            id=data.id or str(uuid.uuid7()),
            role="tool",
            content=response_input_content_to_str(data.output),
            tool_call_id=data.call_id,
            tool_name=_tool_name,
            tool_call_arguments=_tool_call_arguments,
        )

    elif isinstance(data, ResponseReasoningItem):
        return ReasoningMessage(
            id=data.id,
            role="assistant",
            content=data.model_dump_json(include={"summary", "content"}),
        )

    elif isinstance(data, McpListTools):
        return McpListToolsMessage(
            id=data.id,
            mcp_server_label=data.server_label,
            mcp_tools=[
                McpListToolsTool(
                    input_schema=t.input_schema,
                    name=t.name,
                    annotations=t.annotations,
                    description=t.description,
                )
                for t in data.tools
            ],
        )

    elif isinstance(data, McpCall):
        return McpCallMessage(
            id=data.id,
            mcp_call_id=data.id,
            mcp_call_server_label=data.server_label,
            mcp_call_name=data.name,
            mcp_call_arguments=data.arguments,
        )

    else:
        raise ValueError(f"Unsupported response input item: {data}")
