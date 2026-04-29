"""Utilities for splitting the English WestJet suite into smaller local suites."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re

import yaml

from .config_loader import load_test_suite
from .models import TestScenario, TestSuite

DEFAULT_SOURCE_SUITE_PATH = Path("local_suites/westjet_test_suite.yaml")
DEFAULT_OUTPUT_DIR = Path("local_suites/westjet_by_intent")
OUTPUT_FILE_PREFIX = "westjet_test_suite_"


def _normalize_group_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "unassigned"


def derive_group_key(scenario: TestScenario) -> str:
    """Resolve the grouping key for a WestJet scenario.

    Primary path is `expected_intent`. When that is missing, fall back to the
    scenario name prefix so unassigned guideline scenarios still get a usable
    split file.
    """

    if scenario.expected_intent:
        return _normalize_group_key(scenario.expected_intent)

    prefix = str(scenario.name or "").split(" - ", 1)[0]
    return _normalize_group_key(prefix)


def split_westjet_suite_by_intent(source_suite: TestSuite) -> dict[str, TestSuite]:
    """Split a source suite into ordered per-intent suites."""

    grouped_scenarios: "OrderedDict[str, list[TestScenario]]" = OrderedDict()
    for scenario in source_suite.scenarios:
        group_key = derive_group_key(scenario)
        grouped_scenarios.setdefault(group_key, []).append(scenario)

    split_suites: dict[str, TestSuite] = OrderedDict()
    for group_key, scenarios in grouped_scenarios.items():
        split_suites[group_key] = TestSuite(
            name=f"{source_suite.name} - {group_key}",
            language=source_suite.language,
            harness_mode=source_suite.harness_mode,
            primary_categories=source_suite.primary_categories,
            scenarios=scenarios,
        )

    return split_suites


def write_split_westjet_suites(
    source_path: str | Path = DEFAULT_SOURCE_SUITE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate local per-intent WestJet suite files and return their paths."""

    source_suite = load_test_suite(str(source_path))
    split_suites = split_westjet_suite_by_intent(source_suite)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = OrderedDict()
    for group_key, suite in split_suites.items():
        destination = output_root / f"{OUTPUT_FILE_PREFIX}{group_key}.yaml"
        destination.write_text(
            yaml.safe_dump(
                suite.model_dump(exclude_none=True),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        written_paths[group_key] = destination

    return written_paths
