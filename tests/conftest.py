# tests/conftest.py
import datetime
import typing
import zoneinfo

import pydantic
import pytest
import rich.console
from openai.types.shared.function_definition import FunctionDefinition

import str_message.patches.patch_openai

str_message.patches.patch_openai.patch_openai()


FuncDefTupleType: typing.TypeAlias = tuple[
    FunctionDefinition,
    typing.Callable[..., typing.Awaitable[str]],
    typing.Type[pydantic.BaseModel],
]


@pytest.fixture(scope="module")
def console():
    return rich.console.Console()


@pytest.fixture(scope="module")
def func_def_get_current_time() -> FuncDefTupleType:
    class Arguments(pydantic.BaseModel):
        timezone: typing.Optional[str] = pydantic.Field(
            default=None,
            description=(
                "The timezone to get the current time in. If not provided, "
                + "the current time in Asia/Taipei timezone will be returned."
            ),
        )

    async def get_current_time(arguments: Arguments | str) -> str:
        """Get the current time"""
        arguments = (
            Arguments.model_validate_json(arguments)
            if not isinstance(arguments, Arguments)
            else arguments
        )
        dt = datetime.datetime.now(
            zoneinfo.ZoneInfo(arguments.timezone or "Asia/Taipei")
        )
        dt = dt.replace(microsecond=0)
        return dt.isoformat()

    func_def = FunctionDefinition(
        name="get_current_time",
        description="Get the current time of optional timezone.",
        parameters=Arguments.model_json_schema(),
    )

    return (func_def, get_current_time, Arguments)


@pytest.fixture(scope="module")
def agents_tool_get_current_time(
    func_def_get_current_time: FuncDefTupleType, console: rich.console.Console
):
    """Agents function tool wrapper for get_current_time function."""
    import agents

    func_def, func, _ = func_def_get_current_time

    async def on_invoke_tool(
        ctx: agents.RunContextWrapper[agents.TContext], arguments: str
    ) -> typing.Any:
        console.print(f"Agent passes context: {ctx}")
        return await func(arguments)

    return agents.FunctionTool(
        name=func_def.name,
        description=func_def.description or "",
        params_json_schema=func_def.parameters or {},
        on_invoke_tool=on_invoke_tool,
    )


@pytest.fixture(scope="module")
def chat_cmpl_tool_get_current_time(func_def_get_current_time: FuncDefTupleType):
    """OpenAI chat completion tool parameter for get_current_time function."""
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    from openai.types.shared_params.function_definition import FunctionDefinition

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name=func_def_get_current_time[0].name,
            description=func_def_get_current_time[0].description or "",
            parameters=func_def_get_current_time[0].parameters or {},
        ),
        type="function",
    )
