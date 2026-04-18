"""Unit tests for config_loader module."""

import json
import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError

from src.config_loader import (
    load_test_suite,
    load_test_suite_from_string,
    print_test_suite,
    validate_test_suite,
)
from src.models import TestSuite


# --- Fixtures ---


VALID_SUITE_DICT = {
    "name": "Basic Suite",
    "scenarios": [
        {
            "name": "Booking Test",
            "persona": "A busy professional",
            "goal": "Book a meeting for Tuesday at 2pm",
            "attempts": 3,
        }
    ],
}

VALID_SUITE_MULTI = {
    "name": "Multi Scenario Suite",
    "scenarios": [
        {
            "name": "Scenario A",
            "persona": "Customer",
            "goal": "Get account balance",
            "attempts": 5,
        },
        {
            "name": "Scenario B",
            "persona": "New user",
            "goal": "Create an account",
        },
    ],
}


# --- validate_test_suite tests ---


class TestValidateTestSuite:
    def test_valid_suite(self):
        suite = validate_test_suite(VALID_SUITE_DICT)
        assert suite.name == "Basic Suite"
        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].name == "Booking Test"
        assert suite.scenarios[0].attempts == 3

    def test_valid_suite_without_attempts(self):
        data = {
            "name": "Suite",
            "scenarios": [
                {"name": "Test", "persona": "User", "goal": "Do something"}
            ],
        }
        suite = validate_test_suite(data)
        assert suite.scenarios[0].attempts is None

    def test_valid_suite_with_expected_intent(self):
        data = {
            "name": "Suite",
            "scenarios": [
                {
                    "name": "Intent Test",
                    "persona": "Traveler",
                    "goal": "Classify intent",
                    "first_message": "I want to cancel my booking",
                    "expected_intent": "flight_cancel",
                }
            ],
        }
        suite = validate_test_suite(data)
        assert suite.scenarios[0].expected_intent == "flight_cancel"

    def test_valid_suite_with_judge_capture_conversation_id(self):
        data = {
            "name": "Suite",
            "scenarios": [
                {
                    "name": "Intent Test",
                    "persona": "Traveler",
                    "goal": "Classify intent",
                    "first_message": "I want to cancel my booking",
                    "expected_intent": "flight_cancel",
                    "judge_capture_conversation_id": True,
                }
            ],
        }
        suite = validate_test_suite(data)
        assert suite.scenarios[0].judge_capture_conversation_id is True

    def test_missing_name(self):
        data = {
            "scenarios": [
                {"name": "Test", "persona": "User", "goal": "Do something"}
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "name" in str(exc_info.value)

    def test_missing_scenarios(self):
        data = {"name": "Suite"}
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "scenarios" in str(exc_info.value)

    def test_empty_scenarios(self):
        data = {"name": "Suite", "scenarios": []}
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "scenarios" in str(exc_info.value)

    def test_scenario_missing_persona(self):
        data = {
            "name": "Suite",
            "scenarios": [{"name": "Test", "goal": "Do something"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "persona" in str(exc_info.value)

    def test_scenario_missing_goal(self):
        data = {
            "name": "Suite",
            "scenarios": [{"name": "Test", "persona": "User"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "goal" in str(exc_info.value)

    def test_negative_attempts(self):
        data = {
            "name": "Suite",
            "scenarios": [
                {
                    "name": "Test",
                    "persona": "User",
                    "goal": "Do something",
                    "attempts": -1,
                }
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "attempts" in str(exc_info.value)

    def test_zero_attempts(self):
        data = {
            "name": "Suite",
            "scenarios": [
                {
                    "name": "Test",
                    "persona": "User",
                    "goal": "Do something",
                    "attempts": 0,
                }
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_test_suite(data)
        assert "attempts" in str(exc_info.value)


# --- load_test_suite_from_string tests ---


class TestLoadTestSuiteFromString:
    def test_load_json(self):
        content = json.dumps(VALID_SUITE_DICT)
        suite = load_test_suite_from_string(content, "json")
        assert suite.name == "Basic Suite"

    def test_load_yaml(self):
        content = yaml.dump(VALID_SUITE_DICT)
        suite = load_test_suite_from_string(content, "yaml")
        assert suite.name == "Basic Suite"

    def test_load_yml_format(self):
        content = yaml.dump(VALID_SUITE_DICT)
        suite = load_test_suite_from_string(content, "yml")
        assert suite.name == "Basic Suite"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            load_test_suite_from_string("{}", "xml")

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_test_suite_from_string("{not valid json", "json")

    def test_invalid_yaml(self):
        # YAML is very permissive, but we can trigger errors with certain content
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_test_suite_from_string(":\n  - :\n    - :", "yaml")

    def test_non_dict_content(self):
        with pytest.raises(ValueError, match="must be a JSON/YAML object"):
            load_test_suite_from_string("[1, 2, 3]", "json")

    def test_validation_error_propagates(self):
        content = json.dumps({"name": "Suite", "scenarios": []})
        with pytest.raises(ValidationError):
            load_test_suite_from_string(content, "json")


# --- load_test_suite tests ---


class TestLoadTestSuite:
    def test_load_json_file(self, tmp_path):
        file_path = tmp_path / "suite.json"
        file_path.write_text(json.dumps(VALID_SUITE_DICT))
        suite = load_test_suite(str(file_path))
        assert suite.name == "Basic Suite"

    def test_load_yaml_file(self, tmp_path):
        file_path = tmp_path / "suite.yaml"
        file_path.write_text(yaml.dump(VALID_SUITE_DICT))
        suite = load_test_suite(str(file_path))
        assert suite.name == "Basic Suite"

    def test_load_yml_file(self, tmp_path):
        file_path = tmp_path / "suite.yml"
        file_path.write_text(yaml.dump(VALID_SUITE_DICT))
        suite = load_test_suite(str(file_path))
        assert suite.name == "Basic Suite"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_test_suite("/nonexistent/path/suite.json")

    def test_unsupported_extension(self, tmp_path):
        file_path = tmp_path / "suite.txt"
        file_path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_test_suite(str(file_path))


# --- print_test_suite tests ---


class TestPrintTestSuite:
    def test_print_json(self):
        suite = TestSuite.model_validate(VALID_SUITE_DICT)
        output = print_test_suite(suite, "json")
        parsed = json.loads(output)
        assert parsed["name"] == "Basic Suite"
        assert len(parsed["scenarios"]) == 1

    def test_print_yaml(self):
        suite = TestSuite.model_validate(VALID_SUITE_DICT)
        output = print_test_suite(suite, "yaml")
        parsed = yaml.safe_load(output)
        assert parsed["name"] == "Basic Suite"
        assert len(parsed["scenarios"]) == 1

    def test_print_unsupported_format(self):
        suite = TestSuite.model_validate(VALID_SUITE_DICT)
        with pytest.raises(ValueError, match="Unsupported format"):
            print_test_suite(suite, "xml")

    def test_excludes_none_values(self):
        """Omitted optional fields (like attempts=None) should not appear in output."""
        data = {
            "name": "Suite",
            "scenarios": [
                {"name": "Test", "persona": "User", "goal": "Do something"}
            ],
        }
        suite = TestSuite.model_validate(data)
        output = print_test_suite(suite, "json")
        parsed = json.loads(output)
        assert "attempts" not in parsed["scenarios"][0]

    def test_round_trip_json(self):
        suite = TestSuite.model_validate(VALID_SUITE_MULTI)
        serialized = print_test_suite(suite, "json")
        reloaded = load_test_suite_from_string(serialized, "json")
        assert reloaded == suite

    def test_round_trip_yaml(self):
        suite = TestSuite.model_validate(VALID_SUITE_MULTI)
        serialized = print_test_suite(suite, "yaml")
        reloaded = load_test_suite_from_string(serialized, "yaml")
        assert reloaded == suite
