"""
JSON repair utilities for VLM/LLM output.

Provides functions to fix common JSON errors from VLM responses,
including:
- Unterminated strings
- Unclosed brackets/braces
- Trailing commas
- Markdown code blocks
- Truncated responses
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def repair_json(content: str) -> str:
    """
    Attempt to repair common JSON errors from VLM output.

    Strategies:
    1. Fix unterminated strings by finding the last valid position
    2. Close unclosed brackets/braces
    3. Remove trailing commas
    4. Handle truncated responses

    Args:
        content: Raw JSON string that failed to parse

    Returns:
        Repaired JSON string (may still fail to parse)
    """
    original = content

    # Strategy 1: Remove any text after the last valid closing brace
    # Find the last } that could close the root object
    brace_count = 0
    last_valid_pos = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i

    if last_valid_pos > 0 and last_valid_pos < len(content) - 1:
        content = content[:last_valid_pos + 1]
        logger.debug(f"JSON repair: truncated content after position {last_valid_pos}")

    # Strategy 2: Fix unterminated strings
    # Count quotes and if odd, try to find where to close
    quote_count = 0
    last_quote_pos = -1
    in_escape = False
    for i, char in enumerate(content):
        if in_escape:
            in_escape = False
            continue
        if char == '\\':
            in_escape = True
            continue
        if char == '"':
            quote_count += 1
            last_quote_pos = i

    if quote_count % 2 != 0:
        # Odd number of quotes - try to fix
        # Find the last opening quote and close the string
        # Look for patterns like "value without closing quote
        # Try adding a quote before the next structural character
        fixed = []
        in_string = False
        in_escape = False
        for i, char in enumerate(content):
            if in_escape:
                fixed.append(char)
                in_escape = False
                continue
            if char == '\\':
                fixed.append(char)
                in_escape = True
                continue
            if char == '"':
                in_string = not in_string
            elif in_string and char in '\n\r':
                # Newline in string - close the string first
                fixed.append('"')
                in_string = False
            fixed.append(char)

        if in_string:
            fixed.append('"')

        content = ''.join(fixed)
        logger.debug("JSON repair: fixed unterminated string")

    # Strategy 3: Close unclosed brackets/braces
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')

    if open_braces > 0 or open_brackets > 0:
        # Remove trailing comma if present
        content = re.sub(r',\s*$', '', content.rstrip())
        # Add closing characters
        content += ']' * open_brackets + '}' * open_braces
        logger.debug(f"JSON repair: added {open_brackets} ] and {open_braces} }}")

    # Strategy 4: Remove trailing commas before closing brackets/braces
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Strategy 5: Fix common escape issues
    # Replace unescaped control characters in strings
    parts = []
    in_string = False
    current = []
    in_escape = False
    for char in content:
        if in_escape:
            current.append(char)
            in_escape = False
            continue
        if char == '\\':
            current.append(char)
            in_escape = True
            continue
        if char == '"':
            if in_string:
                parts.append(''.join(current))
                current = []
            in_string = not in_string
        current.append(char)

    if current:
        parts.append(''.join(current))
    content = ''.join(parts)

    if content != original:
        logger.debug("JSON repair: modifications applied")

    return content


def strip_markdown(content: str) -> str:
    """
    Remove markdown code block formatting from content.

    Args:
        content: Raw content possibly wrapped in markdown

    Returns:
        Content with markdown formatting removed
    """
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def extract_json_from_response(content: str) -> str:
    """
    Extract JSON from VLM response, handling markdown blocks and extra text.

    Args:
        content: Raw response content

    Returns:
        Cleaned JSON string
    """
    content = strip_markdown(content)

    # Try to find JSON object boundaries if there's extra text
    if not content.startswith('{') and not content.startswith('['):
        # Look for the first { or [
        first_brace = content.find('{')
        first_bracket = content.find('[')
        start = -1
        if first_brace >= 0 and first_bracket >= 0:
            start = min(first_brace, first_bracket)
        elif first_brace >= 0:
            start = first_brace
        elif first_bracket >= 0:
            start = first_bracket

        if start > 0:
            content = content[start:]
            logger.debug(f"JSON extract: removed {start} chars of prefix")

    return content


def safe_json_parse(content: str, repair: bool = True) -> dict:
    """
    Safely parse JSON with optional repair.

    Args:
        content: JSON string to parse
        repair: Whether to attempt repair on failure

    Returns:
        Parsed JSON as dict

    Raises:
        json.JSONDecodeError: If parsing fails even after repair
    """
    # First extract and clean
    content = extract_json_from_response(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if not repair:
            raise

        # Try to repair
        logger.info(f"JSON parse failed, attempting repair: {e}")
        repaired = repair_json(content)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Try more aggressive cleaning
            repaired = extract_json_from_response(repaired)
            return json.loads(repaired)


def recover_partial_json(content: str) -> tuple[Optional[dict], int, Optional[str]]:
    """
    Attempt to recover valid partial JSON from truncated content.

    When VLM output gets cut off mid-JSON, this function tries to
    extract as much valid data as possible.

    Args:
        content: Raw JSON string (possibly truncated)

    Returns:
        Tuple of (recovered_dict, item_count, error_message)
        - recovered_dict: The recovered JSON dict, or None if recovery failed
        - item_count: Number of items in main list (trailing_fees, propositions, etc.)
        - error_message: Description of what was truncated, or None if complete
    """
    # First clean up the content
    cleaned = extract_json_from_response(content)

    # Try direct parse - maybe it's actually valid
    try:
        result = json.loads(cleaned)
        item_count = _count_main_items(result)
        return result, item_count, None
    except json.JSONDecodeError:
        pass

    # Try repair first
    repaired = repair_json(cleaned)
    try:
        result = json.loads(repaired)
        item_count = _count_main_items(result)
        return result, item_count, "JSON repaired successfully"
    except json.JSONDecodeError:
        pass

    # Progressive truncation - find longest valid JSON
    best_result = None
    best_length = 0
    best_item_count = 0

    # Try truncating from the end, step by step
    for end_pos in range(len(repaired), 100, -50):
        try:
            truncated = repaired[:end_pos]

            # Remove incomplete trailing content
            # Find last complete item in array by looking for },
            last_complete = truncated.rfind('},')
            if last_complete > 0:
                truncated = truncated[:last_complete + 1]

            # Count and close brackets/braces
            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')

            # Remove trailing comma if present
            truncated = re.sub(r',\s*$', '', truncated.rstrip())

            # Close all open structures
            truncated += ']' * open_brackets + '}' * open_braces

            result = json.loads(truncated)
            item_count = _count_main_items(result)

            if len(truncated) > best_length:
                best_result = result
                best_length = len(truncated)
                best_item_count = item_count

                # If we have a substantial result, we can stop
                if item_count >= 5:
                    break

        except (json.JSONDecodeError, Exception):
            continue

    if best_result is not None:
        error_msg = f"Recovered {best_item_count} items from truncated JSON"
        logger.warning(error_msg)
        return best_result, best_item_count, error_msg

    return None, 0, "Failed to recover any valid JSON"


def _count_main_items(data: dict) -> int:
    """Count items in main data list (trailing_fees, propositions, activities, etc.)."""
    for key in ['trailing_fees', 'propositions', 'activities', 'commissions', 'records']:
        if key in data and isinstance(data[key], list):
            return len(data[key])
    return 0


def save_debug_json(
    content: str,
    error_type: str,
    error_details: str,
    debug_dir: Optional[Path] = None,
) -> Path:
    """
    Save failed JSON content to debug directory for inspection.

    Args:
        content: Raw content that failed to parse/validate
        error_type: Type of error (json_parse, validation, etc.)
        error_details: Details about the error
        debug_dir: Directory for debug files (defaults to cache/debug)

    Returns:
        Path to the saved debug file
    """
    if debug_dir is None:
        from ..utils.config import get_cache_dir
        debug_dir = get_cache_dir() / "debug"

    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_{error_type}_{timestamp}.json"
    debug_path = debug_dir / filename

    debug_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "error_details": error_details,
        "raw_content": content,
    }

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_entry, f, ensure_ascii=False, indent=2, default=str)

    return debug_path
