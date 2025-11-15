# https://cookbook.openai.com/articles/openai-harmony
import json
import logging
import typing
from datetime import datetime, timezone

import tiktoken
from openai.types.shared.function_definition import FunctionDefinition
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
)
from openai_harmony import Message as HarmonyMessage
from openai_harmony import (
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

from str_message import (
    AssistantMessage,
    DeveloperMessage,
    McpCallMessage,
    McpListToolsMessage,
    MessageTypes,
    ReasoningMessage,
    SystemMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)

tiktoken_encoding = tiktoken.encoding_for_model("gpt-oss-120b")
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def messages_to_harmony(
    messages: typing.List[MessageTypes],
    *,
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
    conversation_start_date: str | None = None,
    tools: typing.List[FunctionDefinition] | None = None,
) -> typing.Dict:
    harmony_messages: typing.List[HarmonyMessage] = []

    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date(
            conversation_start_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
    )
    harmony_messages.append(
        HarmonyMessage.from_role_and_content(Role.SYSTEM, system_message)
    )

    developer_message = DeveloperContent.new()
    harmony_messages.append(
        HarmonyMessage.from_role_and_content(Role.DEVELOPER, developer_message)
    )

    tools_descriptions: typing.List[ToolDescription] = []
    for tool in tools or []:
        tools_descriptions.append(
            ToolDescription.new(tool.name, tool.description or "", tool.parameters)
        )

    for m in messages:

        if isinstance(m, SystemMessage):
            developer_message.with_instructions(m.content)

        elif isinstance(m, DeveloperMessage):
            developer_message.with_instructions(m.content)

        elif isinstance(m, UserMessage):
            harmony_messages.append(
                HarmonyMessage.from_role_and_content(Role(m.role), m.content)
            )

        elif isinstance(m, AssistantMessage):
            harmony_messages.append(
                HarmonyMessage.from_role_and_content(Role(m.role), m.content)
            )

        elif isinstance(m, ToolCallMessage):
            harmony_messages.append(
                HarmonyMessage.from_role_and_content(
                    Role(m.role), m.tool_call_arguments
                )
                .with_channel(m.channel)
                .with_recipient(f"functions.{m.tool_name}")
                .with_content_type("<|constrain|> json"),
            )

        elif isinstance(m, ToolCallOutputMessage):
            harmony_messages.append(
                HarmonyMessage.from_author_and_content(
                    Author.new(Role(m.role), f"functions.{m.tool_name}"),
                    m.content,
                ).with_channel(m.channel),
            )

        elif isinstance(m, McpListToolsMessage):
            for mcp_tool in m.mcp_tools:
                tools_descriptions.append(
                    ToolDescription.new(
                        mcp_tool.name,
                        mcp_tool.description or "",
                        mcp_tool.input_schema,  # type: ignore
                    )
                )

        elif isinstance(m, McpCallMessage):
            harmony_messages.append(
                HarmonyMessage.from_author_and_content(
                    Author.new(Role(m.role), f"functions.{m.mcp_call_name}"),
                    m.content,
                ).with_channel(m.channel),
            )

        elif isinstance(m, ReasoningMessage):
            harmony_messages.append(
                HarmonyMessage.from_role_and_content(
                    Role.ASSISTANT, m.content
                ).with_channel(m.channel)
            )

        else:
            logger.warning(f"Unsupported message type: {type(m)}")

    if tools_descriptions:
        developer_message.with_function_tools(tools_descriptions)

    harmony_conversation = Conversation.from_messages(harmony_messages)

    return json.loads(json.dumps(harmony_conversation.to_dict()))


def messages_to_harmony_str(
    messages: typing.List[MessageTypes],
    *,
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
    conversation_start_date: str | None = None,
    tools: typing.List[FunctionDefinition] | None = None,
) -> str:

    harmony_conversation_dict = messages_to_harmony(
        messages,
        reasoning_effort=reasoning_effort,
        conversation_start_date=conversation_start_date,
        tools=tools,
    )
    harmony_conversation = Conversation.from_json(json.dumps(harmony_conversation_dict))

    tokens = harmony_encoding.render_conversation(harmony_conversation)

    harmony_prompt = tiktoken_encoding.decode(tokens)

    return (
        harmony_prompt.replace("<|end|>", "<|end|>\n")
        .replace("<|call|>", "<|call|>\n")
        .strip()
    )
