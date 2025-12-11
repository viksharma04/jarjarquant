import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def _flatten_dataclass(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a dataclass into a flat dictionary.

    Args:
        obj: The dataclass instance to flatten
        prefix: Prefix to add to field names (for nested dataclasses)

    Returns:
        Flat dictionary with all fields, nested dataclasses flattened with prefixes
    """
    result = {}

    if not is_dataclass(obj) or isinstance(obj, type):
        return result

    for field in fields(obj):
        field_name = field.name
        field_value = getattr(obj, field_name)
        key = f"{prefix}{field_name}" if prefix else field_name

        if is_dataclass(field_value) and not isinstance(field_value, type):
            # Recursively flatten nested dataclass with field name as prefix
            nested = _flatten_dataclass(field_value, prefix=f"{key}_")
            result.update(nested)
        elif isinstance(field_value, Enum):
            # Convert enums to their value
            result[key] = field_value.value
        elif isinstance(field_value, dict):
            # Serialize dicts to JSON
            result[key] = json.dumps(field_value)
        elif isinstance(field_value, (list, tuple)) and any(
            isinstance(item, dict) for item in field_value
        ):
            # Serialize lists/tuples containing dicts to JSON
            result[key] = json.dumps(field_value)
        else:
            result[key] = field_value

    return result
