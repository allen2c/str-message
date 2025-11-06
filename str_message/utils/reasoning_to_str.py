import typing

from openai.types.responses.response_reasoning_item import Content, Summary


def reasoning_to_str(
    summary: typing.List[Summary], content: typing.Optional[typing.List[Content]] = None
) -> str:
    import itertools

    contents: typing.List[str] = []

    for sum, con in itertools.product(summary + [None], (content or []) + [None]):
        if sum is not None:
            contents.append(f"## {sum.text}")
        if con is not None:
            contents.append(con.text)

    return "\n\n".join(contents)
