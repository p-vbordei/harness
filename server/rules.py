"""Deterministic validation rule engine (Layer 1b).

Validates step submissions against structured extract_requirements
and acceptance_criteria rules BEFORE the LLM evaluator is invoked.

Rule types:
    exists         - field is present
    not_empty      - field is non-empty (string/list/dict)
    count          - array length >= min and/or <= max
    min_length     - string length >= N
    max_length     - string length <= N
    type_check     - field is expected JSON type
    each_has_fields - every array item has required keys
    matches_regex  - string matches pattern
    contains       - string contains substring
    file_exists    - referenced path exists on disk
    any_of         - OR composition (pass if any sub-rule passes)
    llm            - deferred to LLM evaluator (Layer 2, not checked here)

Field paths support dot notation and wildcards:
    "user_stories"           - top-level field
    "user_stories[0].title"  - indexed access
    "user_stories[*].title"  - apply to all elements
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from server.models import CriterionResult, ExtractRequirement

logger = logging.getLogger(__name__)

TYPE_MAP = {
    "string": str,
    "str": str,
    "array": list,
    "list": list,
    "object": dict,
    "dict": dict,
    "number": (int, float),
    "integer": int,
    "float": float,
    "boolean": bool,
    "any": object,
}


# ---------------------------------------------------------------------------
# Field path resolution
# ---------------------------------------------------------------------------

def resolve_path(data: Any, path: str) -> list[Any]:
    """Resolve a dot-path with optional [N] / [*] indexing.

    Returns a list of all matched values. Empty list if path doesn't resolve.
    """
    parts = _parse_path(path)
    current = [data]

    for part in parts:
        next_values = []
        for val in current:
            if part == "*":
                # Wildcard: expand all elements
                if isinstance(val, list):
                    next_values.extend(val)
                elif isinstance(val, dict):
                    next_values.extend(val.values())
            elif part.isdigit():
                # Array index
                idx = int(part)
                if isinstance(val, list) and 0 <= idx < len(val):
                    next_values.append(val[idx])
            else:
                # Dict key
                if isinstance(val, dict) and part in val:
                    next_values.append(val[part])
        current = next_values

    return current


def _parse_path(path: str) -> list[str]:
    """Parse 'a.b[0].c[*].d' into ['a', 'b', '0', 'c', '*', 'd']."""
    parts = []
    for segment in path.split("."):
        # Handle brackets: "items[0]" -> "items", "0"
        bracket_match = re.match(r"(\w+)\[(\*|\d+)\]$", segment)
        if bracket_match:
            parts.append(bracket_match.group(1))
            parts.append(bracket_match.group(2))
        else:
            parts.append(segment)
    return parts


# ---------------------------------------------------------------------------
# Extract requirements validation
# ---------------------------------------------------------------------------

def validate_extract_requirements(
    submission: dict,
    requirements: list[ExtractRequirement],
) -> list[CriterionResult]:
    """Validate that all extract_requirements fields are present and typed correctly.

    This is the Triz-inspired deterministic gate: if a required field is
    missing or wrong type, fail immediately without LLM evaluation.
    """
    results = []
    # Look for structured data in artifacts or at top level
    data = _extract_structured_data(submission)

    for req in requirements:
        values = resolve_path(data, req.field)

        if not values:
            results.append(CriterionResult(
                criterion_id=f"extract:{req.field}",
                passed=False,
                evidence=f"Required field '{req.field}' not found in submission.",
                gap=f"Add '{req.field}' to your submission.",
                rule_type="extract_requirement",
            ))
            continue

        value = values[0]  # Primary value for type/length checks

        # Type check
        if req.type != "any":
            expected = TYPE_MAP.get(req.type, object)
            if not isinstance(value, expected):
                results.append(CriterionResult(
                    criterion_id=f"extract:{req.field}",
                    passed=False,
                    evidence=f"Field '{req.field}' expected type '{req.type}', got '{type(value).__name__}'.",
                    gap=f"Change '{req.field}' to type '{req.type}'.",
                    rule_type="extract_requirement",
                ))
                continue

        # Min items (arrays)
        if req.min_items is not None and isinstance(value, list):
            if len(value) < req.min_items:
                results.append(CriterionResult(
                    criterion_id=f"extract:{req.field}",
                    passed=False,
                    evidence=f"Field '{req.field}' has {len(value)} items, minimum is {req.min_items}.",
                    gap=f"Add at least {req.min_items - len(value)} more items to '{req.field}'.",
                    rule_type="extract_requirement",
                ))
                continue

        # Min length (strings)
        if req.min_length is not None and isinstance(value, str):
            if len(value) < req.min_length:
                results.append(CriterionResult(
                    criterion_id=f"extract:{req.field}",
                    passed=False,
                    evidence=f"Field '{req.field}' is {len(value)} chars, minimum is {req.min_length}.",
                    gap=f"Expand '{req.field}' to at least {req.min_length} characters.",
                    rule_type="extract_requirement",
                ))
                continue

        # Passed
        results.append(CriterionResult(
            criterion_id=f"extract:{req.field}",
            passed=True,
            evidence=f"Field '{req.field}' present with valid type and content.",
            rule_type="extract_requirement",
        ))

    return results


# ---------------------------------------------------------------------------
# Acceptance criteria rule validation
# ---------------------------------------------------------------------------

def validate_criteria_rules(
    submission: dict,
    criteria: list[dict],
) -> tuple[list[CriterionResult], list[dict]]:
    """Validate deterministic acceptance criteria rules.

    Returns:
        (deterministic_results, llm_criteria) - deterministic results and
        criteria that need LLM evaluation (type: llm or plain strings).
    """
    deterministic_results = []
    llm_criteria = []
    data = _extract_structured_data(submission)

    for criterion in criteria:
        # Plain string → defer to LLM
        if isinstance(criterion, str):
            llm_criteria.append({
                "id": f"llm:{_slug(criterion)}",
                "description": criterion,
                "prompt": criterion,
            })
            continue

        crit_id = criterion.get("id", f"crit:{_slug(criterion.get('description', ''))}")
        description = criterion.get("description", "")
        rule = criterion.get("rule", {})

        if not rule:
            llm_criteria.append({"id": crit_id, "description": description, "prompt": description})
            continue

        rule_type = rule.get("type", "llm")

        # LLM rules → defer
        if rule_type == "llm":
            llm_criteria.append({
                "id": crit_id,
                "description": description,
                "prompt": rule.get("prompt", description),
            })
            # Check for llm_supplement on deterministic rules
            if "llm_supplement" in criterion:
                llm_criteria.append({
                    "id": f"{crit_id}:supplement",
                    "description": f"{description} (quality check)",
                    "prompt": criterion["llm_supplement"],
                })
            continue

        # Deterministic rules
        result = _evaluate_rule(crit_id, description, rule, data)
        deterministic_results.append(result)

        # If deterministic rule passed but has llm_supplement, queue it
        if result.passed and "llm_supplement" in criterion:
            llm_criteria.append({
                "id": f"{crit_id}:supplement",
                "description": f"{description} (quality check)",
                "prompt": criterion["llm_supplement"],
            })

    return deterministic_results, llm_criteria


def _evaluate_rule(crit_id: str, description: str, rule: dict, data: dict) -> CriterionResult:
    """Evaluate a single deterministic rule against data."""
    rule_type = rule.get("type", "exists")
    field_path = rule.get("field", "")

    try:
        if rule_type == "exists":
            values = resolve_path(data, field_path)
            passed = len(values) > 0
            return CriterionResult(
                criterion_id=crit_id, passed=passed,
                evidence=f"Field '{field_path}' {'found' if passed else 'not found'}.",
                gap=None if passed else f"Add field '{field_path}'.",
                rule_type=rule_type,
            )

        elif rule_type == "not_empty":
            values = resolve_path(data, field_path)
            if not values:
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found.", f"Add '{field_path}'.", rule_type)
            value = values[0]
            passed = bool(value) and (not isinstance(value, (str, list, dict)) or len(value) > 0)
            return CriterionResult(crit_id, passed, f"Field '{field_path}' is {'non-empty' if passed else 'empty'}.", None if passed else f"Populate '{field_path}'.", rule_type)

        elif rule_type == "count":
            values = resolve_path(data, field_path)
            if not values or not isinstance(values[0], list):
                return CriterionResult(crit_id, False, f"Field '{field_path}' is not an array.", f"Make '{field_path}' an array.", rule_type)
            count = len(values[0])
            min_val = rule.get("min", rule.get("value"))  # Support both "min" and legacy "value"
            max_val = rule.get("max")
            op = rule.get("operator", ">=")
            if min_val is not None:
                passed = count >= int(min_val)
                if not passed:
                    return CriterionResult(crit_id, False, f"'{field_path}' has {count} items, need >= {min_val}.", f"Add {int(min_val) - count} more items.", rule_type)
            if max_val is not None:
                passed = count <= int(max_val)
                if not passed:
                    return CriterionResult(crit_id, False, f"'{field_path}' has {count} items, max is {max_val}.", f"Reduce to <= {max_val} items.", rule_type)
            return CriterionResult(crit_id, True, f"'{field_path}' has {count} items.", rule_type=rule_type)

        elif rule_type in ("min_length", "max_length"):
            values = resolve_path(data, field_path)
            if not values or not isinstance(values[0], str):
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found or not a string.", rule_type=rule_type)
            length = len(values[0])
            threshold = int(rule.get("value", rule.get("min", rule.get("max", 0))))
            if rule_type == "min_length":
                passed = length >= threshold
                return CriterionResult(crit_id, passed, f"'{field_path}' is {length} chars (min {threshold}).", None if passed else f"Expand to >= {threshold} chars.", rule_type)
            else:
                passed = length <= threshold
                return CriterionResult(crit_id, passed, f"'{field_path}' is {length} chars (max {threshold}).", None if passed else f"Shorten to <= {threshold} chars.", rule_type)

        elif rule_type == "type_check":
            values = resolve_path(data, field_path)
            if not values:
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found.", rule_type=rule_type)
            expected_type = rule.get("expected", "string")
            expected_cls = TYPE_MAP.get(expected_type, object)
            passed = isinstance(values[0], expected_cls)
            return CriterionResult(crit_id, passed, f"'{field_path}' is {type(values[0]).__name__}, expected {expected_type}.", rule_type=rule_type)

        elif rule_type == "each_has_fields":
            values = resolve_path(data, field_path)
            if not values or not isinstance(values[0], list):
                return CriterionResult(crit_id, False, f"Field '{field_path}' not an array.", rule_type=rule_type)
            required = rule.get("required", rule.get("required_fields", []))
            for i, item in enumerate(values[0]):
                if not isinstance(item, dict):
                    return CriterionResult(crit_id, False, f"'{field_path}[{i}]' is not an object.", rule_type=rule_type)
                missing = [r for r in required if r not in item]
                if missing:
                    return CriterionResult(crit_id, False, f"'{field_path}[{i}]' missing fields: {missing}.", f"Add {missing} to each item.", rule_type)
            return CriterionResult(crit_id, True, f"All items in '{field_path}' have required fields {required}.", rule_type=rule_type)

        elif rule_type == "matches_regex":
            pattern = rule.get("pattern", "")
            compiled = re.compile(pattern)
            values = resolve_path(data, field_path)
            if not values:
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found.", rule_type=rule_type)
            # Check all resolved values (for wildcard paths)
            for v in values:
                if not isinstance(v, str) or not compiled.search(v):
                    expect_match = rule.get("expect_match", True)
                    if expect_match:
                        return CriterionResult(crit_id, False, f"'{field_path}' value '{str(v)[:50]}' doesn't match /{pattern}/.", rule_type=rule_type)
                    else:
                        return CriterionResult(crit_id, True, f"'{field_path}' correctly doesn't match forbidden pattern.", rule_type=rule_type)
            return CriterionResult(crit_id, True, f"'{field_path}' matches pattern.", rule_type=rule_type)

        elif rule_type == "contains":
            values = resolve_path(data, field_path)
            if not values or not isinstance(values[0], str):
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found or not a string.", rule_type=rule_type)
            substring = rule.get("substring", "")
            any_of = rule.get("any_of", [])
            if substring:
                passed = substring in values[0]
                return CriterionResult(crit_id, passed, f"'{field_path}' {'contains' if passed else 'missing'} '{substring}'.", rule_type=rule_type)
            elif any_of:
                passed = any(s in values[0] for s in any_of)
                return CriterionResult(crit_id, passed, f"'{field_path}' {'contains' if passed else 'missing'} one of {any_of}.", rule_type=rule_type)
            return CriterionResult(crit_id, True, "No substring specified.", rule_type=rule_type)

        elif rule_type == "file_exists":
            values = resolve_path(data, field_path)
            if not values:
                return CriterionResult(crit_id, False, f"Field '{field_path}' not found.", rule_type=rule_type)
            for v in values:
                if isinstance(v, str) and not Path(v).exists():
                    return CriterionResult(crit_id, False, f"File '{v}' does not exist.", f"Create or fix path '{v}'.", rule_type)
            return CriterionResult(crit_id, True, "All referenced files exist.", rule_type=rule_type)

        elif rule_type == "any_of":
            sub_rules = rule.get("rules", [])
            for sub in sub_rules:
                result = _evaluate_rule(crit_id, description, sub, data)
                if result.passed:
                    return CriterionResult(crit_id, True, f"Passed via: {result.evidence}", rule_type="any_of")
            return CriterionResult(crit_id, False, "None of the alternative rules passed.", rule_type="any_of")

        else:
            # Unknown rule type → defer to LLM
            return CriterionResult(crit_id, True, f"Unknown rule type '{rule_type}', skipped.", rule_type="unknown")

    except Exception as exc:
        logger.warning("Rule evaluation error for '%s': %s", crit_id, exc)
        return CriterionResult(crit_id, False, f"Rule evaluation error: {exc}", rule_type="error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_structured_data(submission: dict) -> dict:
    """Extract structured data from submission for rule evaluation.

    Looks in artifacts for json_object type content, or uses the
    submission dict itself as the data source.
    """
    data = dict(submission)  # Start with the full submission

    # Try to extract structured data from JSON artifacts
    artifacts = submission.get("artifacts", [])
    for artifact in artifacts:
        if isinstance(artifact, dict):
            art_type = artifact.get("type", "")
            content = artifact.get("content", "")
            if art_type == "json_object" and isinstance(content, str):
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        data.update(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
            elif art_type == "text" and isinstance(content, str):
                # Try to parse text artifacts as JSON too
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        data.update(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass

    return data


def _slug(text: str) -> str:
    """Convert text to a short slug for criterion IDs."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:40].strip("-")
