import typing

from openai._types import Omit, omit
from openai.types.shared.reasoning_effort import ReasoningEffort


def might_reasoning_effort(
    model: str,
    reasoning_effort: ReasoningEffort = "low",
    *,
    default: typing.Union[ReasoningEffort, Omit] = omit,
):
    from openai_usage.extra.open_router import get_model

    might_model = get_model(model)
    if might_model is None:
        return default

    if "reasoning" in might_model.supported_parameters:
        return reasoning_effort
    else:
        return default
