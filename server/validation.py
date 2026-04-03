"""Layer 1 deterministic validation for step submissions.

Validates structure, types, and formats before handing off to
the LLM evaluator (Layer 2). All checks here are fast, deterministic,
and require no model calls.
"""

import uuid
from dataclasses import dataclass, field

import jsonschema

VALID_ARTIFACT_TYPES = frozenset({"file_path", "code_block", "text", "json_object"})
MAX_ARTIFACT_CONTENT_SIZE = 1_000_000  # 1MB per artifact
MAX_ARTIFACTS_PER_SUBMISSION = 50


@dataclass
class ValidationResult:
    """Outcome of a validation check."""
    is_valid: bool
    message: str
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_submission(
    step_output: dict,
    output_schema: dict | None = None,
) -> ValidationResult:
    """Validate a step submission dict.

    Checks performed (in order):
    1. ``artifacts`` list must be present and non-empty.
    2. Each artifact must have a ``type`` in VALID_ARTIFACT_TYPES.
    3. ``self_assessment`` must be a non-empty string.
    4. If *output_schema* is provided, validate *step_output* against it.

    Returns a ``ValidationResult`` that aggregates all errors found.
    """
    errors: list[str] = []

    # --- artifacts ----------------------------------------------------------
    artifacts = step_output.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) == 0:
        errors.append("'artifacts' must be a non-empty list.")
    elif len(artifacts) > MAX_ARTIFACTS_PER_SUBMISSION:
        errors.append(
            f"Too many artifacts ({len(artifacts)}). "
            f"Maximum is {MAX_ARTIFACTS_PER_SUBMISSION}."
        )
    else:
        for idx, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                errors.append(f"artifacts[{idx}]: expected a dict, got {type(artifact).__name__}.")
                continue
            art_type = artifact.get("type")
            if art_type not in VALID_ARTIFACT_TYPES:
                errors.append(
                    f"artifacts[{idx}]: invalid type '{art_type}'. "
                    f"Must be one of {sorted(VALID_ARTIFACT_TYPES)}."
                )
            content = artifact.get("content", "")
            if isinstance(content, str) and len(content) > MAX_ARTIFACT_CONTENT_SIZE:
                errors.append(
                    f"artifacts[{idx}]: content too large "
                    f"({len(content)} bytes, max {MAX_ARTIFACT_CONTENT_SIZE})."
                )

    # --- self_assessment ----------------------------------------------------
    self_assessment = step_output.get("self_assessment")
    if not isinstance(self_assessment, str) or self_assessment.strip() == "":
        errors.append("'self_assessment' must be a non-empty string.")

    # --- JSON Schema (optional) ---------------------------------------------
    if output_schema is not None:
        try:
            jsonschema.validate(instance=step_output, schema=output_schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"Schema validation failed: {exc.message}")
        except jsonschema.SchemaError as exc:
            errors.append(f"Invalid output_schema: {exc.message}")

    if errors:
        return ValidationResult(
            is_valid=False,
            message=f"Submission invalid: {len(errors)} error(s).",
            errors=errors,
        )
    return ValidationResult(is_valid=True, message="Submission valid.")


def validate_session_id(session_id: str) -> ValidationResult:
    """Validate that *session_id* is a well-formed UUID (version 4)."""
    try:
        parsed = uuid.UUID(session_id, version=4)
    except (ValueError, AttributeError):
        return ValidationResult(
            is_valid=False,
            message="Invalid session_id.",
            errors=[f"'{session_id}' is not a valid UUID v4."],
        )
    # Ensure the string round-trips (rejects uppercase / braces / etc.)
    if str(parsed) != session_id:
        return ValidationResult(
            is_valid=False,
            message="Invalid session_id.",
            errors=[f"session_id must be lowercase canonical UUID, got '{session_id}'."],
        )
    return ValidationResult(is_valid=True, message="session_id valid.")


def validate_sop_id(sop_id: str) -> ValidationResult:
    """Validate that *sop_id* is a non-empty identifier.

    Accepts simple slug-style identifiers: alphanumeric, hyphens, underscores,
    dots, and forward slashes (for namespaced IDs like ``org/sop-name``).
    """
    if not isinstance(sop_id, str) or sop_id.strip() == "":
        return ValidationResult(
            is_valid=False,
            message="Invalid sop_id.",
            errors=["sop_id must be a non-empty string."],
        )

    import re
    if not re.fullmatch(r"[A-Za-z0-9._/\-]+", sop_id):
        return ValidationResult(
            is_valid=False,
            message="Invalid sop_id.",
            errors=[
                f"sop_id '{sop_id}' contains invalid characters. "
                "Only alphanumeric, hyphens, underscores, dots, and slashes are allowed."
            ],
        )
    return ValidationResult(is_valid=True, message="sop_id valid.")
