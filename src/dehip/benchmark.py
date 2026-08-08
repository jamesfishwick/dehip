"""Pinned published benchmark rows for the comparison report (FR-007).

These are the DFT-post numbers a dehip run is measured against: the 14B SFT
"superbaseline" plus the 4B / 8B / 14B DFT results, each with MMD, JMQ, and
token-frequency L2. They are a *code constant*, transcribed once and never
recomputed, so a comparison report can place a dehip run's own scores beside a
fixed published reference.

Source of truth: PLAN.md (repo root), the DFT table under "The DFT numbers to
beat", verified 2026-08-01 against the primary source (the Rosmine "Fixing LLM
writing with Distribution Fine Tuning" post). The four rows below are that
table transcribed verbatim; do not "correct" a value without re-checking
PLAN.md, which is the authority here.

Critical external-protocol caveat (FR-007): these numbers were produced by a
DIFFERENT judge (GPT-5.4-mini in a different harness), on a DIFFERENT corpus
(a 185K-sample cleaned FineWeb subset, 2000 held-out for eval), and the DFT
superbaseline is a *max over a 132-eval-set hyperparameter sweep*, deliberately
hard to beat. JMQ also flatters models (the DFT post's own Appendix 8: judges
favor LLM output). So these rows are NOT apples-to-apples with a local dehip
run's numbers. :data:`BENCHMARK_TABLE` carries ``external_protocol=True`` so any
renderer ALWAYS shows that caveat; a benchmark table shown without it invites a
false apples-to-apples read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BenchmarkRow",
    "BenchmarkTable",
    "BENCHMARK_ROWS",
    "BENCHMARK_TABLE",
    "EXTERNAL_PROTOCOL_CAVEAT",
    "benchmark_table_to_jsonable",
]


# The caveat text every renderer must surface whenever benchmark rows appear.
# Kept as a module constant so the JSON payload and the markdown render carry
# the identical wording (no drift between the two representations).
EXTERNAL_PROTOCOL_CAVEAT = (
    "EXTERNAL PROTOCOL -- these benchmark rows are NOT apples-to-apples with the "
    "dehip run above: they were scored by a different judge in a different "
    "harness, on a different corpus (a 185K-sample cleaned FineWeb subset, 2000 "
    "held out for eval), and the SFT superbaseline is a max over a "
    "132-eval-set hyperparameter sweep. JMQ additionally flatters models. Read "
    "the deltas as directional, never as a head-to-head win. Source: PLAN.md."
)


@dataclass(frozen=True)
class BenchmarkRow:
    """One published benchmark model's MMD / JMQ / token-L2 (transcribed).

    ``mmd`` and ``token_l2`` are lower-is-better; ``jmq`` is higher-is-better
    (the ``2 * win-rate`` JMQ definition, so 1.0 is parity with human). These
    mirror the sign conventions the comparison deltas use.
    """

    model: str
    mmd: float
    jmq: float
    token_l2: float


# --- The pinned rows, transcribed verbatim from PLAN.md ----------------------
#
# PLAN.md "The DFT numbers to beat" table (verified 2026-08-01):
#
#   | Model                  | MMD (lower) | JMQ (higher) | Token L2 (lower) |
#   |------------------------|-------------|--------------|------------------|
#   | 14B SFT superbaseline  | 0.037       | 0.49         | 0.0039           |
#   | 14B DFT                | 0.018       | 0.80         | 0.0036           |
#   | 8B DFT                 | 0.023       | 0.56         | 0.0031           |
#   | 4B DFT                 | 0.025       | 0.40         | 0.0042           |
#
# Order below preserves PLAN.md's row order (superbaseline first, then DFT
# 14B/8B/4B) so a reader can diff the rendered table against the source table
# line for line.
BENCHMARK_ROWS: tuple[BenchmarkRow, ...] = (
    BenchmarkRow(model="14B SFT superbaseline", mmd=0.037, jmq=0.49, token_l2=0.0039),
    BenchmarkRow(model="14B DFT", mmd=0.018, jmq=0.80, token_l2=0.0036),
    BenchmarkRow(model="8B DFT", mmd=0.023, jmq=0.56, token_l2=0.0031),
    BenchmarkRow(model="4B DFT", mmd=0.025, jmq=0.40, token_l2=0.0042),
)


@dataclass(frozen=True)
class BenchmarkTable:
    """The pinned benchmark rows plus the flags that gate how they render.

    ``external_protocol`` is ``True`` and is never a default a caller can flip
    off: the rows always come with the different-judge / different-corpus
    caveat, so no consumer can render them as a clean head-to-head.
    """

    rows: tuple[BenchmarkRow, ...]
    external_protocol: bool = True
    caveat: str = EXTERNAL_PROTOCOL_CAVEAT
    source: str = "PLAN.md"
    metric_directions: dict[str, str] = field(
        default_factory=lambda: {
            "mmd": "lower_is_better",
            "jmq": "higher_is_better",
            "token_l2": "lower_is_better",
        }
    )


# The single pinned table instance the report attaches.
BENCHMARK_TABLE = BenchmarkTable(rows=BENCHMARK_ROWS)


def benchmark_table_to_jsonable(
    table: BenchmarkTable = BENCHMARK_TABLE,
) -> dict[str, Any]:
    """Serialize a :class:`BenchmarkTable` to a strict-JSON-safe dict.

    The ``external_protocol`` flag and ``caveat`` are ALWAYS included so a
    downstream consumer of the JSON cannot drop the caveat when it keeps the
    rows (FR-007). Each row is a plain dict of its four fields.
    """
    return {
        "source": table.source,
        "external_protocol": table.external_protocol,
        "caveat": table.caveat,
        "metric_directions": dict(table.metric_directions),
        "rows": [
            {
                "model": row.model,
                "mmd": row.mmd,
                "jmq": row.jmq,
                "token_l2": row.token_l2,
            }
            for row in table.rows
        ],
    }
