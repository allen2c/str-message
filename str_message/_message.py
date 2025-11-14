import abc
import datetime
import logging
import textwrap
import time
import typing
import zoneinfo

import agents
import cachetools
import durl
import jinja2
import openai_usage
import pydantic
import uuid_utils as uuid
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message import (
    FunctionCall as ChatCompletionFunctionCall,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.easy_input_message import EasyInputMessage
from openai.types.responses.parsed_response import ParsedResponseOutputItem
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
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_output_item import (
    McpListToolsTool,
    ResponseOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_usage import (
    ResponseUsage,
)
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
ResponseOutputItemAdapter = pydantic.TypeAdapter[ResponseOutputItem](ResponseOutputItem)
ParsedResponseOutputItemAdapter = pydantic.TypeAdapter[ParsedResponseOutputItem](
    ParsedResponseOutputItem
)


CONTENT_TEXT_TYPE = "text"
CONTENT_AUDIO_TYPE = "input_audio"
CONTENT_IMAGE_URL_TYPE = "image_url"
CONTENT_IMAGE_ID_TYPE = "image_id"
CONTENT_FILE_ID_TYPE = "file_id"
CONTENT_FILE_URL_TYPE = "file_url"
CONTENT_FILE_FILENAME_TYPE = "file_filename"
ALL_CONTENT_TYPES = (
    CONTENT_TEXT_TYPE,
    CONTENT_AUDIO_TYPE,
    CONTENT_IMAGE_URL_TYPE,
    CONTENT_IMAGE_ID_TYPE,
    CONTENT_FILE_ID_TYPE,
    CONTENT_FILE_URL_TYPE,
    CONTENT_FILE_FILENAME_TYPE,
)
CONTENT_TYPE: typing.TypeAlias = typing.Union[
    typing.Literal[
        "text",
        "input_audio",
        "image_url",
        "image_id",
        "file_id",
        "file_url",
        "file_filename",
    ],
    str,
]

CONTENT_AUDIO_EXPR = r"@![input_audio]({input_audio})"
CONTENT_IMAGE_URL_EXPR = r"@![image_url]({image_url})"
CONTENT_IMAGE_ID_EXPR = r"@![image_id]({image_id})"
CONTENT_FILE_ID_EXPR = r"@![file_id]({file_id})"
CONTENT_FILE_URL_EXPR = r"@![file_url]({file_url})"
CONTENT_FILE_FILENAME_EXPR = r"@![filename]({filename})"


class ContentPart(pydantic.BaseModel):
    type: CONTENT_TYPE
    value: str

    @classmethod
    def from_str(cls, content: str) -> list["ContentPart"]:
        from str_message.utils.content_parts_from_str import content_parts_from_str

        return content_parts_from_str(content)

    @pydantic.model_validator(mode="after")
    def warning_invalid_type(self) -> typing.Self:
        if self.type not in ALL_CONTENT_TYPES:
            logger.warning(f"Invalid content type: {self.type}")
        return self

    def to_str(self) -> str:
        if self.type == CONTENT_TEXT_TYPE:
            return self.value
        elif self.type == CONTENT_AUDIO_TYPE:
            return CONTENT_AUDIO_EXPR.format(input_audio=self.value)
        elif self.type == CONTENT_IMAGE_URL_TYPE:
            return CONTENT_IMAGE_URL_EXPR.format(image_url=self.value)
        elif self.type == CONTENT_IMAGE_ID_TYPE:
            return CONTENT_IMAGE_ID_EXPR.format(image_id=self.value)
        elif self.type == CONTENT_FILE_ID_TYPE:
            return CONTENT_FILE_ID_EXPR.format(file_id=self.value)
        elif self.type == CONTENT_FILE_URL_TYPE:
            return CONTENT_FILE_URL_EXPR.format(file_url=self.value)
        elif self.type == CONTENT_FILE_FILENAME_TYPE:
            return CONTENT_FILE_FILENAME_EXPR.format(filename=self.value)
        else:
            logger.warning(f"Invalid content type: {self.type}")
            return f"@![{self.type}]({self.value})"


ContentParts = pydantic.TypeAdapter[list[ContentPart]](list[ContentPart])


class MessageUtils(abc.ABC):
    role: typing.Literal["user", "assistant", "system", "developer", "tool"]
    content: str
    created_at: int
    metadata: typing.Optional[typing.Dict[str, str]]

    # Class variables
    _call_id_to_tool_calls: cachetools.TTLCache[
        str, ChatCompletionFunctionCall | ResponseFunctionToolCall
    ] = cachetools.TTLCache(
        maxsize=1000, ttl=60 * 60  # 1 hour
    )

    @classmethod
    def from_any(
        cls,
        data: (
            ANY_MESSAGE_TYPES
            | ChatCompletion
            | ResponseOutputItem
            | ParsedResponseOutputItem
        ),
        *,
        verbose: bool = False,
        **kwargs,
    ) -> typing.List["MessageTypes"]:
        """Create message from various input types."""
        from str_message.utils.message_from_any import message_from_any

        final = message_from_any(data)
        if verbose:
            raw_repr = pretty_repr(data, indent_size=0, max_string=16).replace("\n", "")
            final_repr = pretty_repr(final, indent_size=0, max_string=16).replace(
                "\n", ""
            )
            logger.debug(f"Message from {raw_repr} to {final_repr}")
        return final

    @classmethod
    def to_chat_cmpl_input_messages(
        cls, messages: typing.Union[list["Message"], "Message"]
    ) -> list[ChatCompletionMessageParam]:
        """Convert message to list of ChatCompletionMessageParam."""
        from str_message.utils.messages_to_chat_cmpl_input_messages import (
            messages_to_chat_cmpl_input_messages,
        )

        return messages_to_chat_cmpl_input_messages(
            [messages] if isinstance(messages, Message) else messages
        )

    @classmethod
    def to_response_input_param(
        cls,
        messages: typing.Union[list["Message"], "Message"],
        *,
        ignore_reasoning: bool = False,
    ) -> ResponseInputParam:
        """Convert message to ResponseInputParam."""
        from str_message.utils.messages_to_response_input_param import (
            messages_to_response_input_param,
        )

        return messages_to_response_input_param(
            [messages] if isinstance(messages, Message) else messages,
            ignore_reasoning=ignore_reasoning,
        )

    @classmethod
    def content_to_parts(cls, content: str) -> list[ContentPart]:
        return ContentPart.from_str(content)

    @classmethod
    def ensure_reasoning_following_items(
        cls, messages: list["Message"]
    ) -> list["Message"]:
        """
        To solve OpenAI error: Item 'rs_xxx' of type 'reasoning' was provided
        without its required following item.
        """
        if not messages:
            return messages

        remove_ids: list[str] = []

        for idx, msg in enumerate(messages):
            if isinstance(msg, ReasoningMessage):
                if len(messages) < idx + 1:
                    logger.warning(
                        f"Removing reasoning message {msg.id} "
                        + "because it is the last message"
                    )
                    remove_ids.append(msg.id)
                    continue
                # Must be before assistant message else remove it
                if not isinstance(messages[idx + 1], AssistantMessage):
                    logger.warning(
                        f"Removing reasoning message {msg.id} "
                        + "because it is not with correct following item"
                    )
                    remove_ids.append(msg.id)
                    continue

        messages[:] = [msg for msg in messages if msg.id not in remove_ids]
        return messages

    @classmethod
    def set_tool_call(
        cls,
        call_id: str,
        tool_call: ChatCompletionFunctionCall | ResponseFunctionToolCall,
    ) -> None:
        logger.info(
            f"Setting tool call '{call_id}' with value: {tool_call.model_dump_json()}"
        )
        cls._call_id_to_tool_calls[call_id] = tool_call

    @classmethod
    def get_tool_call(
        cls, call_id: str
    ) -> ChatCompletionFunctionCall | ResponseFunctionToolCall | None:
        might_tool_call = cls._call_id_to_tool_calls.get(call_id)
        if might_tool_call is None:
            logger.warning(f"Tool call '{call_id}' not found! Please set it first!")
            return None
        return might_tool_call

    @property
    def content_parts(self) -> list[ContentPart]:
        return self.content_to_parts(self.content)

    def add_audio(
        self, audio: str | bytes, mime_type: durl.AUDIO_MIME_TYPES
    ) -> typing.Self:
        data_url = durl.DataURL.from_data(mime_type, audio)
        self.content += "\n\n" + str(CONTENT_AUDIO_EXPR.format(input_audio=data_url))
        self.content = self.content.strip()
        return self

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


class Message(pydantic.BaseModel, MessageUtils):
    """A universal message format for AI interactions."""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))

    # Required fields
    role: typing.Literal[
        "user", "assistant", "system", "developer", "tool", "mcp", "mcp_list_tools"
    ]
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

    created_at: int = pydantic.Field(default_factory=lambda: int(time.time()))
    metadata: typing.Optional[typing.Dict[str, str]] = None

    @pydantic.model_validator(mode="after")
    def warning_empty(self) -> typing.Self:
        if not self.content:
            if self.role == "assistant" and self.channel == "commentary":
                pass
            elif self.role in ("mcp", "mcp_list_tools"):
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


class AssistantMessage(Message):
    role: typing.Literal["assistant"] = "assistant"
    channel: typing.Literal["final"] = "final"


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


class McpCallMessage(Message):
    role: typing.Literal["mcp"] = "mcp"
    content: str = ""
    channel: typing.Literal["commentary"] = "commentary"
    mcp_call_id: str
    mcp_call_server_label: str
    mcp_call_name: str
    mcp_call_arguments: str = "{}"

    @pydantic.model_validator(mode="after")
    def raise_empty(self) -> typing.Self:
        if (
            not self.mcp_call_id
            or not self.mcp_call_name
            or not self.mcp_call_arguments
        ):
            raise ValueError("Tool call id, name, and arguments are required")
        return self


class McpListToolsMessage(Message):
    role: typing.Literal["mcp_list_tools"] = "mcp_list_tools"
    content: str = ""
    channel: typing.Literal["commentary"] = "commentary"
    mcp_server_label: str
    mcp_tools: typing.List[McpListToolsTool] = pydantic.Field(default_factory=list)


MessageTypes: typing.TypeAlias = typing.Union[
    AssistantMessage,
    DeveloperMessage,
    McpCallMessage,
    McpListToolsMessage,
    Message,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
]
MessageTypesList: typing.TypeAlias = typing.List[MessageTypes]
ALL_MESSAGE_TYPES = (
    AssistantMessage,
    DeveloperMessage,
    McpCallMessage,
    McpListToolsMessage,
    Message,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)


class Conversation(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid7()))
    messages: MessageTypesList = pydantic.Field(default_factory=list)
    usages: typing.List[openai_usage.Usage] = pydantic.Field(default_factory=list)

    @property
    def total_cost(self) -> str:
        import decimal

        total_cost = decimal.Decimal(0)
        for usage in self.usages:
            if usage.cost:
                total_cost += decimal.Decimal(usage.cost)
        return total_cost.to_eng_string()

    @property
    def chat_cmpl_messages(self) -> list[ChatCompletionMessageParam]:
        return Message.to_chat_cmpl_input_messages(self.messages)

    @property
    def response_input_param(self) -> ResponseInputParam:
        return Message.to_response_input_param(self.messages)

    def add_message(self, message: MessageTypes | typing.List[MessageTypes]) -> None:
        if isinstance(message, typing.List):
            self.messages.extend(message)
        else:
            self.messages.append(message)

    def add_usage(
        self,
        usage: (
            openai_usage.Usage
            | ResponseUsage
            | agents.RunContextWrapper
            | agents.Usage
            | CompletionUsage
        ),
        *,
        model: typing.Optional[str] = None,
        annotations: typing.Optional[str] = None,
    ) -> None:
        if isinstance(usage, openai_usage.Usage):
            valid_usage = openai_usage.Usage.model_validate_json(
                usage.model_dump_json()
            )
        else:
            valid_usage = openai_usage.Usage.from_openai(usage)

        if model:
            valid_usage.model = model
            valid_usage.cost = valid_usage.estimate_cost_str()
        else:
            logger.warning(f"Can not find model '{model}' card")

        if annotations:
            valid_usage.annotations = annotations

        self.usages.append(valid_usage)

    def clean_messages(self) -> None:
        self.messages = Message.ensure_reasoning_following_items(self.messages)
