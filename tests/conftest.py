# tests/conftest.py
import datetime
import typing
import zoneinfo

import pytest
import rich.console

import str_message.patches.patch_openai

str_message.patches.patch_openai.patch_openai()


@pytest.fixture(scope="module")
def console():
    return rich.console.Console()


@pytest.fixture(scope="module")
def function_get_current_time():
    """Async function that returns current time in Asia/Taipei timezone."""

    async def get_current_time():
        """Get the current time"""
        dt = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Taipei"))
        dt = dt.replace(microsecond=0)
        return dt.isoformat()

    return get_current_time


@pytest.fixture(scope="module")
def agents_tool_get_current_time(function_get_current_time: typing.Callable[..., str]):
    """Agents function tool wrapper for get_current_time function."""
    import agents

    return agents.function_tool(function_get_current_time)


@pytest.fixture(scope="module")
def chat_cmpl_tool_get_current_time():
    """OpenAI chat completion tool parameter for get_current_time function."""
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    from openai.types.shared_params.function_definition import FunctionDefinition

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name="get_current_time", description="Get the current time", parameters={}
        ),
        type="function",
    )
