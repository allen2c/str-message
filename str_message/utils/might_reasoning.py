import typing

from openai._types import omit
from openai.types.shared import Reasoning
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning as ReasoningParam

T = typing.TypeVar("T")


def might_reasoning_effort(
    model: str,
    reasoning_effort: ReasoningEffort = "low",
    *,
    default: typing.Union[ReasoningEffort, T] = omit,
) -> ReasoningEffort | T:
    from openai_usage.extra.open_router import get_model

    might_model = get_model(model)
    if might_model is None:
        return default

    if "reasoning" in might_model.supported_parameters:
        return reasoning_effort
    else:
        return default


def might_reasoning(
    model: str,
    reasoning: Reasoning | ReasoningEffort = "low",
    *,
    default: typing.Union[Reasoning, T] = omit,
) -> Reasoning | T:
    from openai_usage.extra.open_router import get_model

    might_model = get_model(model)
    if might_model is None:
        return default

    if "reasoning" in might_model.supported_parameters:
        if isinstance(reasoning, str):
            return Reasoning(effort=reasoning)
        elif reasoning is None:
            return Reasoning(effort="low")
        return reasoning
    else:
        return default


def might_reasoning_param(
    model: str,
    reasoning: ReasoningParam | ReasoningEffort = "low",
    *,
    default: typing.Union[ReasoningParam, T] = omit,
) -> ReasoningParam | T:
    from openai_usage.extra.open_router import get_model

    might_model = get_model(model)
    if might_model is None:
        return default

    if "reasoning" in might_model.supported_parameters:
        if isinstance(reasoning, str):
            return ReasoningParam(effort=reasoning)
        elif reasoning is None:
            return ReasoningParam(effort="low")
        return reasoning
    else:
        return default
