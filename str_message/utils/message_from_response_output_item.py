import json
import typing

import pydantic
from openai.types.responses.parsed_response import (
    ParsedResponseFunctionToolCall,
    ParsedResponseOutputItem,
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
)
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
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from str_message import (
    CONTENT_IMAGE_ID_EXPR,
    CONTENT_IMAGE_URL_EXPR,
    AssistantMessage,
    McpCallMessage,
    McpListToolsMessage,
    Message,
    MessageTypes,
    ReasoningMessage,
    ToolCallMessage,
)

McpListToolsToolAdapter = pydantic.TypeAdapter(typing.List[McpListToolsTool])


def message_from_response_output_item(
    data: ResponseOutputItem | ParsedResponseOutputItem,
) -> MessageTypes:

    if isinstance(data, ParsedResponseOutputMessage):
        return AssistantMessage(
            id=data.id,
            content="\n\n".join(
                c.text if isinstance(c, ParsedResponseOutputText) else c.refusal
                for c in data.content
            ),
        )

    elif isinstance(data, ParsedResponseFunctionToolCall):
        Message.set_tool_call(data.call_id, data)
        return ToolCallMessage(
            id=data.call_id,
            tool_call_id=data.call_id,
            tool_name=data.name,
            tool_call_arguments=data.arguments,
        )

    elif isinstance(data, ResponseOutputMessage):
        return AssistantMessage(
            id=data.id,
            content="\n\n".join(
                c.text if isinstance(c, ResponseOutputText) else c.refusal
                for c in data.content
            ),
        )

    elif isinstance(data, ResponseFileSearchToolCall):
        return ToolCallMessage(
            id=data.id,
            tool_call_id=data.id,
            tool_name="file_search",
            tool_call_arguments=json.dumps({"queries": data.queries}),
        )

    elif isinstance(data, ResponseFunctionToolCall):
        Message.set_tool_call(data.call_id, data)
        return ToolCallMessage(
            id=data.call_id,
            tool_call_id=data.call_id,
            tool_name=data.name,
            tool_call_arguments=data.arguments,
        )

    elif isinstance(data, ResponseFunctionWebSearch):
        if data.action.type == "find":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="web_search_find",
                tool_call_arguments=json.dumps({"pattern": data.action.pattern}),
            )
        elif data.action.type == "open_page":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="web_search_open_page",
                tool_call_arguments=json.dumps({"url": data.action.url}),
            )
        elif data.action.type == "search":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="web_search_search",
                tool_call_arguments=json.dumps({"query": data.action.query}),
            )
        else:
            raise ValueError(f"Unsupported action type: {data.action.type}")

    elif isinstance(data, ResponseComputerToolCall):
        if data.action.type == "click":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_click",
                tool_call_arguments=json.dumps(
                    {
                        "button": data.action.button,
                        "x": data.action.x,
                        "y": data.action.y,
                    }
                ),
            )
        elif data.action.type == "double_click":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_double_click",
                tool_call_arguments=json.dumps(
                    {"x": data.action.x, "y": data.action.y}
                ),
            )
        elif data.action.type == "drag":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_drag",
                tool_call_arguments=json.dumps({"path": data.action.path}),
            )
        elif data.action.type == "keypress":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_keypress",
                tool_call_arguments=json.dumps({"keys": data.action.keys}),
            )
        elif data.action.type == "move":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_move",
                tool_call_arguments=json.dumps(
                    {"x": data.action.x, "y": data.action.y}
                ),
            )
        elif data.action.type == "screenshot":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_screenshot",
                tool_call_arguments="{}",
            )
        elif data.action.type == "scroll":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_scroll",
                tool_call_arguments=json.dumps(
                    {"x": data.action.x, "y": data.action.y}
                ),
            )
        elif data.action.type == "type":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_type",
                tool_call_arguments=json.dumps({"text": data.action.text}),
            )
        elif data.action.type == "wait":
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="computer_wait",
                tool_call_arguments="{}",
            )
        else:
            raise ValueError(f"Unsupported action type: {data.action.type}")

    elif isinstance(data, ResponseReasoningItem):
        return ReasoningMessage(
            id=data.id,
            content=data.model_dump_json(include={"summary", "content"}),
        )

    elif isinstance(data, ImageGenerationCall):
        return AssistantMessage(
            id=data.id,
            content=(
                CONTENT_IMAGE_URL_EXPR.format(image_url=data.result)
                if data.result
                else CONTENT_IMAGE_ID_EXPR.format(image_id=data.id)
            ),
        )

    elif isinstance(data, ResponseCodeInterpreterToolCall):
        if data.outputs:
            return AssistantMessage(
                id=data.id,
                content="\n\n".join(
                    (
                        c.logs
                        if c.type == "logs"
                        else CONTENT_IMAGE_URL_EXPR.format(image_url=c.url)
                    )
                    for c in data.outputs
                ),
            )
        else:
            return ToolCallMessage(
                id=data.id,
                tool_call_id=data.id,
                tool_name="code_interpreter",
                tool_call_arguments=json.dumps({"code": data.code}),
            )

    elif isinstance(data, LocalShellCall):
        return ToolCallMessage(
            id=data.id,
            tool_call_id=data.id,
            tool_name="local_shell",
            tool_call_arguments=json.dumps(
                {
                    "command": data.action.command,
                    "env": data.action.env,
                    "timeout_ms": data.action.timeout_ms,
                    "user": data.action.user,
                    "working_directory": data.action.working_directory,
                }
            ),
        )

    elif isinstance(data, McpCall):
        return McpCallMessage(
            id=data.id,
            content=data.output or str(data.error or ""),
            mcp_call_id=data.id,
            mcp_call_server_label=data.server_label,
            mcp_call_name=data.name,
            mcp_call_arguments=data.arguments,
        )

    elif isinstance(data, McpListTools):
        return McpListToolsMessage(
            id=data.id,
            content=str(data.error or ""),
            mcp_server_label=data.server_label,
            mcp_tools=McpListToolsToolAdapter.validate_json(
                McpListToolsToolAdapter.dump_json(data.tools)
            ),
        )

    elif isinstance(data, McpApprovalRequest):
        raise NotImplementedError

    elif isinstance(data, ResponseCustomToolCall):
        return ToolCallMessage(
            id=data.call_id,
            tool_call_id=data.call_id,
            tool_name=data.name,
            tool_call_arguments=data.input,
        )

    else:
        raise ValueError(f"Unsupported response output item: {data}")
