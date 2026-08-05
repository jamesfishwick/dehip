"""Entity schemas and serialization for the HIP cascade and evaluation harness.

File-based model (data-model.md): every entity is either a per-text JSONL
record (Pair, RewriteBundle, JudgeVerdict) or a per-run JSON document (TextSet
manifest, MetricReport). All records carry ``schema_version``; reading an
unknown version raises rather than silently parsing a shape we do not
understand.

Identity rule: texts are joined by ``pair_id`` (stable across corpus, draft,
and rewrite artifacts). Where deduplication matters (embedding cache), a
content SHA-256 provides the key (see :func:`content_sha256`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Enumerations from data-model.md. Kept as frozensets for cheap membership
# checks in validation.
ROLES = frozenset({"human_reference", "instruct_draft", "rewrite"})
DIMENSIONS = frozenset(
    {"overall", "clarity", "coherence", "creativity", "depth", "relevance"}
)
ORDERS = frozenset({"model_first", "human_first"})
CHOICES = frozenset({"A", "B", "invalid"})


class SchemaVersionError(ValueError):
    """Raised when a record's ``schema_version`` is not one we can parse."""


class SchemaValidationError(ValueError):
    """Raised when a record violates a data-model invariant."""


def content_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``, the dedup/cache key.

    UTF-8 encoded so the digest is stable across platforms.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Entity dataclasses ------------------------------------------------------


@dataclass
class Pair:
    """One prompt + human reference (corpus record). JSONL, one per line."""

    pair_id: str
    corpus: str
    prompt: str
    reference_text: str
    source: dict[str, Any]
    register: str
    prompt_generator: str
    word_count: int
    schema_version: int = SCHEMA_VERSION


@dataclass
class TextSet:
    """A named, role-tagged collection referencing Pair ids (JSON manifest).

    The unit the harness scores. ``role`` must be one of :data:`ROLES` and the
    ``corpus`` tag is homogeneous within a set (validated in ``__post_init__``).
    """

    set_id: str
    role: str
    corpus: str
    pair_ids: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)
    round: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SchemaValidationError(
                f"TextSet role {self.role!r} not in {sorted(ROLES)}"
            )
        if not isinstance(self.corpus, str) or not self.corpus:
            raise SchemaValidationError(
                "TextSet corpus must be a non-empty homogeneous tag"
            )
        if self.round is not None and self.role != "rewrite":
            raise SchemaValidationError(
                "TextSet round is only valid for role=rewrite"
            )


@dataclass
class RewriteBundle:
    """Output of one cascade run for one pair (JSONL, one bundle per pair)."""

    run_id: str
    pair_id: str
    prompt: str
    rounds: list[dict[str, Any]]
    final_round: int
    degeneration: dict[str, Any]
    adapter_id: str
    hip_config: dict[str, Any]
    requested_k: int
    draft: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass
class JudgeVerdict:
    """One (pair, dimension) comparison (JSONL). ``order`` is seeded/randomized."""

    pair_id: str
    dimension: str
    judge_model: str
    order: str
    raw_response: str
    choice: str
    retry_count: int
    model_won: bool | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise SchemaValidationError(
                f"JudgeVerdict dimension {self.dimension!r} not in "
                f"{sorted(DIMENSIONS)}"
            )
        if self.order not in ORDERS:
            raise SchemaValidationError(
                f"JudgeVerdict order {self.order!r} not in {sorted(ORDERS)}"
            )
        if self.choice not in CHOICES:
            raise SchemaValidationError(
                f"JudgeVerdict choice {self.choice!r} not in {sorted(CHOICES)}"
            )


@dataclass
class MetricReport:
    """Scored comparison of two TextSets (JSON + rendered markdown)."""

    report_id: str
    compared: dict[str, Any]
    config: dict[str, Any]
    mmd: float
    token_l2: float
    jmq: dict[str, Any]
    timestamps: dict[str, Any]
    caveats: list[Any] = field(default_factory=list)
    benchmark_rows: list[Any] | None = None
    deltas: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION


# --- Serialization core ------------------------------------------------------


def _check_version(raw: dict[str, Any], cls: type) -> None:
    """Reject records whose ``schema_version`` we cannot parse."""
    version = raw.get("schema_version")
    if version is None:
        raise SchemaVersionError(
            f"{cls.__name__} record is missing schema_version"
        )
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{cls.__name__} record has unknown schema_version {version!r}; "
            f"this build only reads version {SCHEMA_VERSION}"
        )


def _from_dict(raw: dict[str, Any], cls: type) -> Any:
    """Build a dataclass from ``raw`` after a version check.

    Unknown keys are rejected so a shape drift under the same version number
    does not parse silently.
    """
    _check_version(raw, cls)
    known = {f.name for f in fields(cls)}
    extra = set(raw) - known
    if extra:
        raise SchemaValidationError(
            f"{cls.__name__} record has unexpected fields: {sorted(extra)}"
        )
    return cls(**raw)


def _to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


# --- JSONL helpers (per-text records) ---------------------------------------


def write_jsonl(records: list[Any], path: str | Path) -> None:
    """Write a list of dataclass records as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_to_dict(record), ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: str | Path, cls: type) -> list[Any]:
    """Read a JSONL file into a list of ``cls`` records.

    Raises :class:`SchemaVersionError` on any line with an unknown version.
    """
    path = Path(path)
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(_from_dict(json.loads(line), cls))
    return records


# --- JSON helpers (per-run documents: manifest + report) --------------------


def write_json(record: Any, path: str | Path) -> None:
    """Write a single dataclass record as a JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_dict(record), fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_json(path: str | Path, cls: type) -> Any:
    """Read a single-document JSON file into a ``cls`` record."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return _from_dict(json.load(fh), cls)
