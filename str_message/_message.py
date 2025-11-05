import abc
import datetime
import logging
import textwrap
import time
import typing
import zoneinfo

import durl
import jinja2
import pydantic
import uuid_utils as uuid
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.responses.easy_input_message import EasyInputMessage
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_custom_tool_call_output import (
    ResponseCustomToolCallOutput,
)
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_input_item import (
    ComputerCallOutput,
    FunctionCallOutput,
    ImageGenerationCall,
    ItemReference,
    LocalShellCall,
    LocalShellCallOutput,
    McpApprovalRequest,
    McpApprovalResponse,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_input_item import (
    Message as ResponseInputItemMessage,
)
from openai.types.responses.response_input_item import (
    ResponseInputItem,
)
from openai.types.responses.response_input_item_param import ResponseInputItemParam
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.shared.function_definition import FunctionDefinition
from rich.pretty import pretty_repr

logger = logging.getLogger(__name__)


OPENAI_MESSAGE_PARAM_TYPES: typing.TypeAlias = typing.Union[
    ResponseInputItemParam, ChatCompletionMessageParam
]
OPENAI_MESSAGE_TYPES: typing.TypeAlias = typing.Union[
    ResponseInputItem, ChatCompletionMessage
]

ANY_MESSAGE_TYPES: typing.TypeAlias = typing.Union[
    "Message",
    str,
    durl.DataURL,
    typing.Dict,
    OPENAI_MESSAGE_PARAM_TYPES,
    OPENAI_MESSAGE_TYPES,
]
ListFuncDefAdapter = pydantic.TypeAdapter[typing.List[FunctionDefinition]](
    typing.List[FunctionDefinition]
)
ResponseInputItemModels = (
    EasyInputMessage,
    ResponseInputItemMessage,
    ResponseOutputMessage,
    ResponseFileSearchToolCall,
    ResponseComputerToolCall,
    ComputerCallOutput,
    ResponseFunctionWebSearch,
    ResponseFunctionToolCall,
    FunctionCallOutput,
    ResponseReasoningItem,
    ImageGenerationCall,
    ResponseCodeInterpreterToolCall,
    LocalShellCall,
    LocalShellCallOutput,
    McpListTools,
    McpApprovalRequest,
    McpApprovalResponse,
    McpCall,
    ResponseCustomToolCallOutput,
    ResponseCustomToolCall,
    ItemReference,
)
ResponseInputItemAdapter = pydantic.TypeAdapter[ResponseInputItem](ResponseInputItem)


class MessageUtils(abc.ABC):
    role: typing.Literal["user", "assistant", "system", "developer", "tool"]
    content: str
    created_at: int
    metadata: typing.Optional[typing.Dict[str, str]]

    def to_instructions(
        self,
        *,
        with_datetime: bool = False,
        tz: zoneinfo.ZoneInfo | str | None = None,
        max_string: int = 600,
    ) -> str:
        """Format message as readable instructions."""
        from str_message.utils.ensure_tz import ensure_tz

        _role = self.role
        _content = self.content

        _dt: datetime.datetime | None = None
        if with_datetime:
            _dt = datetime.datetime.fromtimestamp(self.created_at, ensure_tz(tz))
            _dt = _dt.replace(microsecond=0)
        template = jinja2.Template(
            textwrap.dedent(
                """
                [{% if dt %}{{ dt.strftime('%Y-%m-%dT%H:%M:%S') }} {% endif %}{{ role }}] {{ content }}
                """  # noqa: E501
            ).strip()
        )
        return template.render(
            role=_role,
            dt=_dt,
            content=pretty_repr(_content, max_string=max_string),
        ).strip()

    @classmethod
    def from_any(cls, data: ANY_MESSAGE_TYPES) -> "Message":
        """Create message from various input types."""
        from str_message.utils.message_from_any import message_from_any

        return message_from_any(data)


class Message(pydantic.BaseModel, MessageUtils):
    """A universal message format for AI interactions."""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid7()))

    # Required fields
    role: typing.Literal["user", "assistant", "system", "developer", "tool"]
    """
    Role 'user' is for user input.
    Role 'assistant' is for assistant output or assistant tool call.
    Role 'system' is for system instructions.
    Role 'developer' is for developer output.
    Role 'tool' is for tool output.
    """

    content: str  # I love simple definitions
    """The field must be a plain text content or data URL"""

    channel: typing.Optional[typing.Literal["analysis", "commentary", "final"]] = None
    """
    Channel None for user message.
    Channel 'analysis' is for thinking or reasoning.
    Channel 'commentary' is for tool call or tool output.
    Channel 'final' is for final output of the assistant.
    """

    # Optional fields
    tool_call_id: typing.Optional[str] = None
    tool_name: typing.Optional[str] = None
    tool_call_arguments: typing.Optional[str] = None
    created_at: int = pydantic.Field(default_factory=lambda: int(time.time()))
    metadata: typing.Optional[typing.Dict[str, str]] = None

    @pydantic.model_validator(mode="after")
    def warning_empty(self) -> typing.Self:
        if not self.content:
            if self.role == "assistant" and self.channel == "commentary":
                pass
            else:
                logger.warning("Message content is empty")
        return self


class SystemMessage(Message):
    role: typing.Literal["system"] = "system"
    content: str
    channel: typing.Literal[None] = None


class DeveloperMessage(Message):
    role: typing.Literal["developer"] = "developer"
    content: str
    channel: typing.Literal[None] = None


class UserMessage(Message):
    role: typing.Literal["user"] = "user"
    channel: typing.Literal[None] = None


class ReasoningMessage(Message):
    role: typing.Literal["assistant"] = "assistant"
    content: str = ""
    channel: typing.Literal["analysis"] = "analysis"


class ToolCallMessage(Message):
    role: typing.Literal["assistant"] = "assistant"
    content: str = ""
    channel: typing.Literal["commentary"] = "commentary"
    tool_call_id: str = pydantic.Field(default="")
    tool_name: str = pydantic.Field(default="")
    tool_call_arguments: str = "{}"

    @pydantic.model_validator(mode="after")
    def raise_empty(self) -> typing.Self:
        if not self.tool_call_id or not self.tool_name or not self.tool_call_arguments:
            raise ValueError("Tool call id, name, and arguments are required")
        return self


class ToolCallOutputMessage(Message):
    role: typing.Literal["tool"] = "tool"
    content: str = ""
    channel: typing.Literal["commentary"] = "commentary"
    tool_call_id: str = pydantic.Field(default="")
    tool_name: str = pydantic.Field(default="")
    tool_call_arguments: str = "{}"

    @pydantic.model_validator(mode="after")
    def raise_empty(self) -> typing.Self:
        if not self.tool_call_id or not self.tool_name or not self.tool_call_arguments:
            raise ValueError("Tool call id, name, and arguments are required")
        if not self.content:
            logger.warning("Tool call output content is empty")
        return self


MessageTypes: typing.TypeAlias = typing.Union[
    Message,
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
]
ALL_MESSAGE_TYPES = (
    Message,
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
)
