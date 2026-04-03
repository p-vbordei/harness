"""Tests for the evaluator module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.evaluator import (
    DIMENSIONS,
    SubagentEvaluator,
    AnthropicEvaluator,
    _extract_json,
    parse_evaluation_response,
    create_evaluator,
    build_evaluation_prompt,
)
from server.models import EvaluationResult


# ------------------------------------------------------------------
# _extract_json tests
# ------------------------------------------------------------------

def test_extract_json_raw():
    result = _extract_json('{"completeness": {"score": 4}}')
    assert result["completeness"]["score"] == 4


def test_extract_json_fenced():
    result = _extract_json('```json\n{"completeness": {"score": 4}}\n```')
    assert result["completeness"]["score"] == 4


def test_extract_json_embedded_in_prose():
    result = _extract_json('Here is my evaluation:\n{"completeness": {"score": 3}}\nThank you.')
    assert result["completeness"]["score"] == 3


def test_extract_json_trailing_comma():
    result = _extract_json('{"completeness": {"score": 4,},}')
    assert result["completeness"]["score"] == 4


def test_extract_json_invalid():
    with pytest.raises(ValueError, match="Could not extract"):
        _extract_json("no json here at all")


# ------------------------------------------------------------------
# parse_evaluation_response tests
# ------------------------------------------------------------------

def test_parse_pass():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in DIMENSIONS
    }
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, attempt=1, max_attempts=3)
    assert result.verdict == "PASS"
    assert result.weighted_score == 4.0


def test_parse_fail_below_threshold():
    raw = {
        dim: {"score": 3, "evidence": "OK", "gap": "Minor"}
        for dim in DIMENSIONS
    }
    raw["slop_flags"] = ["filler"]
    raw["top_3_fixes"] = ["Fix 1"]
    result = parse_evaluation_response(raw, attempt=1, max_attempts=3)
    assert result.verdict == "FAIL"
    assert result.weighted_score == 3.0


def test_parse_fail_one_dimension_below_min():
    raw = {dim: {"score": 5, "evidence": "Great", "gap": None} for dim in DIMENSIONS}
    raw["completeness"] = {"score": 2, "evidence": "Missing", "gap": "Add X"}
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, attempt=1, max_attempts=3)
    assert result.verdict == "FAIL"


def test_score_clamping():
    raw = {dim: {"score": 99, "evidence": ".", "gap": None} for dim in DIMENSIONS}
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, attempt=1, max_attempts=3)
    assert all(d.score == 5 for d in result.dimensions.values())


# ------------------------------------------------------------------
# build_evaluation_prompt tests
# ------------------------------------------------------------------

def test_build_evaluation_prompt():
    sys, user = build_evaluation_prompt(
        submission={"artifacts": [{"type": "text", "content": "hello"}]},
        acceptance_criteria=["Criterion 1", "Criterion 2"],
        step_context="Do the thing",
    )
    assert "quality gate evaluator" in sys
    assert "Criterion 1" in user
    assert "Criterion 2" in user


# ------------------------------------------------------------------
# SubagentEvaluator tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_evaluator_returns_prompt():
    evaluator = SubagentEvaluator()
    result = await evaluator.evaluate(
        submission={"artifacts": [{"type": "text", "content": "hello"}]},
        acceptance_criteria=["Criterion 1"],
    )
    assert result["mode"] == "subagent"
    assert "system_prompt" in result
    assert "user_prompt" in result


@pytest.mark.asyncio
async def test_subagent_evaluator_auto_passes_no_criteria():
    evaluator = SubagentEvaluator()
    result = await evaluator.evaluate(
        submission={"artifacts": [{"type": "text", "content": "hello"}]},
        acceptance_criteria=[],
    )
    assert result["mode"] == "auto_pass"
    assert result["result"].verdict == "PASS"


# ------------------------------------------------------------------
# AnthropicEvaluator tests (mocked)
# ------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_evaluator():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        evaluator = AnthropicEvaluator(api_key="test-key")
    return evaluator


@pytest.mark.asyncio
async def test_anthropic_evaluate_pass(mock_anthropic_evaluator):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        dim: {"score": 4, "evidence": "Good work", "gap": None}
        for dim in DIMENSIONS
    } | {"slop_flags": [], "top_3_fixes": []}))]

    mock_anthropic_evaluator._client.messages.create = AsyncMock(return_value=mock_response)

    result = await mock_anthropic_evaluator.evaluate(
        submission={"artifacts": [{"type": "text", "content": "hello"}]},
        acceptance_criteria=["Criterion 1"],
    )
    assert result.verdict == "PASS"
    assert result.weighted_score == 4.0


@pytest.mark.asyncio
async def test_anthropic_evaluate_api_error(mock_anthropic_evaluator):
    import anthropic
    mock_anthropic_evaluator._client.messages.create = AsyncMock(
        side_effect=anthropic.APIError(message="rate limit", request=MagicMock(), body=None)
    )
    result = await mock_anthropic_evaluator.evaluate(
        submission={"artifacts": [{"type": "text", "content": "hello"}]},
        acceptance_criteria=["Criterion 1"],
    )
    assert result.verdict == "FAIL"
    assert result.weighted_score == 0.0


# ------------------------------------------------------------------
# Factory tests
# ------------------------------------------------------------------

def test_create_evaluator_default():
    evaluator = create_evaluator()
    assert isinstance(evaluator, SubagentEvaluator)


def test_create_evaluator_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        evaluator = create_evaluator(backend="anthropic", api_key="test")
    assert isinstance(evaluator, AnthropicEvaluator)


def test_create_evaluator_openai_requires_url():
    with pytest.raises(ValueError, match="base_url"):
        create_evaluator(backend="openai")
