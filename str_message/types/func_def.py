import logging
import typing

import agents
import pydantic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared_params.function_definition import (
    FunctionDefinition as FunctionDefinitionParam,
)

logger = logging.getLogger(__name__)


class FuncDef:
    def __init__(
        self,
        func_def: FunctionDefinition,
        callable: typing.Callable[..., typing.Awaitable[str]],
        arguments_type: typing.Type[pydantic.BaseModel],
        *,
        context: typing.Optional[agents.TContext] = None,
        strict: bool = True,
    ):
        self.func_def = func_def
        self.callable = callable
        self.arguments_type = arguments_type
        self.context = context
        self.strict = strict

    @property
    def name(self) -> str:
        return self.func_def.name

    @property
    def description(self) -> str:
        return self.func_def.description or ""

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        param = self.func_def.parameters or {}
        param["additionalProperties"] = False
        if properties := param.get("properties"):
            param["required"] = list(dict(properties).keys())  # type: ignore
        return param

    @property
    def is_context_required(self) -> bool:
        return self.context is not None

    @property
    def chat_cmpl_tool_param(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            function=FunctionDefinitionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
                strict=self.strict,
            ),
            type="function",
        )

    @property
    def agents_tool(self) -> agents.FunctionTool:
        async def on_invoke_tool(
            ctx: agents.RunContextWrapper[agents.TContext], arguments: str
        ) -> typing.Any:
            if self.is_context_required:
                if type(self.context) is not type(ctx.context):
                    logger.error(
                        f"Agent context type {type(ctx.context)} "
                        + f"is not tool expected type {type(self.context)}"
                    )
                return await self.callable(ctx.context, arguments)
            else:
                return await self.callable(arguments)

        return agents.FunctionTool(
            name=self.name,
            description=self.description,
            params_json_schema=self.parameters,
            on_invoke_tool=on_invoke_tool,
            strict_json_schema=self.strict,
        )
