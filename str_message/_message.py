import logging
import time
import typing

import pydantic
import uuid_utils as uuid

logger = logging.getLogger(__name__)


class Message(pydantic.BaseModel):
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
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
]
ALL_MESSAGE_TYPES = (
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
)
