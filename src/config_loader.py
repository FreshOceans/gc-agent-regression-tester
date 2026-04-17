"""Config loader for loading, validating, and serializing test suite files."""

import json
import os

import yaml
from pydantic import ValidationError

from .models import TestSuite


def load_test_suite(file_path: str) -> TestSuite:
    """Load and validate a test suite from a JSON or YAML file.

    Args:
        file_path: Path to the JSON or YAML test suite file.

    Returns:
        A validated TestSuite object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or content is invalid.
        ValidationError: If the data fails Pydantic validation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test suite file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".yaml", ".yml"):
        fmt = "yaml"
    elif ext == ".json":
        fmt = "json"
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Use .json, .yaml, or .yml"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return load_test_suite_from_string(content, fmt)


def load_test_suite_from_string(content: str, format: str) -> TestSuite:
    """Load and validate a test suite from a string.

    Args:
        content: The raw string content of the test suite file.
        format: The format of the content - "json" or "yaml".

    Returns:
        A validated TestSuite object.

    Raises:
        ValueError: If the format is unsupported or content cannot be parsed.
        ValidationError: If the data fails Pydantic validation.
    """
    fmt = format.lower()
    if fmt not in ("json", "yaml", "yml"):
        raise ValueError(
            f"Unsupported format '{format}'. Use 'json' or 'yaml'"
        )

    try:
        if fmt == "json":
            data = json.loads(content)
        else:
            data = yaml.safe_load(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Test suite content must be a JSON/YAML object (dict)")

    return validate_test_suite(data)


def validate_test_suite(data: dict) -> TestSuite:
    """Validate a raw dictionary against the TestSuite schema.

    Args:
        data: A dictionary representing the test suite data.

    Returns:
        A validated TestSuite object.

    Raises:
        ValidationError: If validation fails, with messages identifying
            the problematic fields.
    """
    return TestSuite.model_validate(data)


def print_test_suite(suite: TestSuite, format: str = "json") -> str:
    """Serialize a TestSuite back to a valid JSON or YAML string.

    Args:
        suite: The TestSuite object to serialize.
        format: Output format - "json" or "yaml". Defaults to "json".

    Returns:
        A string representation of the test suite in the specified format.

    Raises:
        ValueError: If the format is unsupported.
    """
    fmt = format.lower()
    if fmt not in ("json", "yaml", "yml"):
        raise ValueError(
            f"Unsupported format '{format}'. Use 'json' or 'yaml'"
        )

    data = suite.model_dump(exclude_none=True)

    if fmt == "json":
        return json.dumps(data, indent=2)
    else:
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
