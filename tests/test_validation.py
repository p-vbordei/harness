"""Tests for the validation module."""

from server.validation import validate_session_id, validate_sop_id, validate_submission


def test_valid_submission():
    output = {
        "artifacts": [
            {"type": "text", "content": "Hello world"},
        ],
        "self_assessment": "Looks good to me.",
    }
    result = validate_submission(output)
    assert result.is_valid


def test_empty_artifacts_fails():
    output = {"artifacts": [], "self_assessment": "Done."}
    result = validate_submission(output)
    assert not result.is_valid
    assert any("non-empty" in e for e in result.errors)


def test_invalid_artifact_type_fails():
    output = {
        "artifacts": [{"type": "video", "content": "data"}],
        "self_assessment": "Done.",
    }
    result = validate_submission(output)
    assert not result.is_valid
    assert any("invalid type" in e for e in result.errors)


def test_empty_self_assessment_fails():
    output = {
        "artifacts": [{"type": "text", "content": "data"}],
        "self_assessment": "",
    }
    result = validate_submission(output)
    assert not result.is_valid


def test_schema_validation():
    schema = {
        "type": "object",
        "required": ["artifacts", "self_assessment", "custom_field"],
        "properties": {
            "custom_field": {"type": "string"},
        },
    }
    output = {
        "artifacts": [{"type": "text", "content": "data"}],
        "self_assessment": "Done.",
    }
    result = validate_submission(output, schema)
    assert not result.is_valid
    assert any("Schema validation" in e for e in result.errors)


def test_valid_session_id():
    result = validate_session_id("550e8400-e29b-41d4-a716-446655440000")
    assert result.is_valid


def test_invalid_session_id():
    result = validate_session_id("not-a-uuid")
    assert not result.is_valid


def test_valid_sop_id():
    result = validate_sop_id("feature-dev")
    assert result.is_valid


def test_empty_sop_id():
    result = validate_sop_id("")
    assert not result.is_valid
