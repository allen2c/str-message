import re

from str_message import ContentPart


def content_parts_from_str(content: str) -> list[ContentPart]:
    """Parse content string into ContentPart objects.
    Extracts text and special @![type](value) syntax into structured parts.
    """
    content = content.strip()

    # Pattern to match @![type](value) syntax
    pattern: str = r"@!\[([^\]]+)\]\(([^\)]+)\)"

    parts: list[ContentPart] = []
    last_end: int = 0

    # Find all special syntax matches
    for match in re.finditer(pattern, content):
        start: int = match.start()
        end: int = match.end()

        # Add text before this match (if any)
        if start > last_end:
            text_value: str = content[last_end:start].strip()
            if text_value:
                parts.append(ContentPart(type="text", value=text_value))

        # Add the matched special syntax
        type_value: str = match.group(1)  # Content inside [...]
        value_content: str = match.group(2)  # Content inside (...)
        parts.append(ContentPart(type=type_value, value=value_content))

        last_end = end

    # Add any remaining text after the last match
    if last_end < len(content):
        text_value: str = content[last_end:].strip()
        if text_value:
            parts.append(ContentPart(type="text", value=text_value))

    return parts
