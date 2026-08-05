"""JMQ pairwise-judge orchestration (FR-002, FR-008, FR-009; research.md R6).

JMQ scores a set of model outputs against human references for the same
prompts. For every (pair, dimension) it asks a judge which of two candidates is
better, presenting the model output and the human reference in a seeded,
per-pair-randomized A/B order so positional bias is auditable. The score for a
dimension is ``2 * model_win_rate``: a model that matches human writing wins
about half the comparisons, giving a JMQ near 1.0.

Design invariants
-----------------
- **Prompts are loaded verbatim.** The six templates in ``judge-prompts/`` are
  read byte-for-byte and never edited here; each carries a recorded SHA-256 so a
  drift from the transcribed source is detectable (FR-002).
- **A/B order is seeded and reproducible.** A per-pair order is derived
  deterministically from the base seed and the ``pair_id``, so the exact
  sequence of assignments replays from the seed alone, and the same order is
  used across all six dimensions for that pair (positional placement is a
  property of the pair, not the dimension). The distribution is ~50/50 over many
  pairs (data-model.md, Story 1 scenario 4).
- **Verdicts are persisted before aggregation (FR-008).** Every judge call
  yields a :class:`~dehip.schemas.JudgeVerdict` written to JSONL *before* any
  score is computed. Aggregation then reads *only* that file, so a crash between
  judging and scoring loses nothing: rerun the aggregation over the persisted
  verdicts and JMQ is recomputed without re-querying the judge.
- **Malformed verdicts are excluded-and-counted, never silently scored.** A
  response that does not parse to A or B is retried once; if it still fails it is
  recorded with ``choice="invalid"`` and surfaced in the per-dimension
  ``invalid`` count, not dropped (spec edge-case rule).
- **Cost preflight before spend (FR-009).** :func:`estimate_call_count` and
  :func:`cost_preflight` report the call count (pairs x 6 dimensions) and refuse
  to proceed above a spend threshold unless the caller explicitly confirms.

The OpenAI runtime is expensive and non-deterministic, so it sits behind the
injectable :class:`JudgeClient` protocol. Production uses
:class:`OpenAIJudgeClient`; tests supply a stub that never touches the network.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from dehip.schemas import (
    DIMENSIONS,
    JudgeVerdict,
    content_sha256,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "DIMENSION_ORDER",
    "DEFAULT_JUDGE_MODEL",
    "JudgePair",
    "JudgeClient",
    "OpenAIJudgeClient",
    "JudgePrompts",
    "load_judge_prompts",
    "assign_order",
    "parse_choice",
    "render_prompt",
    "run_judging",
    "aggregate_verdicts",
    "estimate_call_count",
    "CostThresholdError",
    "cost_preflight",
]

# The six dimensions in a fixed, reproducible order. ``DIMENSIONS`` in schemas
# is a frozenset (unordered); JMQ needs a stable sequence so verdict emission
# and iteration are deterministic.
DIMENSION_ORDER: tuple[str, ...] = (
    "overall",
    "clarity",
    "coherence",
    "creativity",
    "depth",
    "relevance",
)
assert set(DIMENSION_ORDER) == set(DIMENSIONS), (
    "DIMENSION_ORDER must cover exactly the schema DIMENSIONS"
)

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"

# Per-pair spend estimate for the preflight. The dollar figure the DFT protocol
# implies is not published; this is a deliberately conservative placeholder the
# caller can override, and it only gates a *confirmation prompt*, never the
# actual API pricing. Recorded so the estimate is auditable, not authoritative.
DEFAULT_COST_PER_CALL_USD = 0.0005


@dataclass(frozen=True)
class JudgePair:
    """One prompt with its model output and human reference to compare.

    ``pair_id`` joins back to the corpus Pair and is the stable identity used
    both to seed the A/B order and to key persisted verdicts.
    """

    pair_id: str
    prompt: str
    model_text: str
    human_text: str


@runtime_checkable
class JudgeClient(Protocol):
    """A single-call judge: rendered prompt -> raw judge response string.

    Implementations must be thread-safe: :func:`run_judging` may call this
    concurrently across pairs and dimensions. The return value is the verbatim
    text of the judge's answer; parsing to A/B/invalid happens in
    :func:`parse_choice`, never in the client.
    """

    def judge(self, rendered_prompt: str, *, model: str) -> str: ...


class OpenAIJudgeClient:
    """Production :class:`JudgeClient` backed by the OpenAI chat API.

    The ``openai`` client is imported lazily so the module (and every test that
    injects a stub client) never requires the SDK or an API key merely to import
    :mod:`dehip.metrics.jmq`.
    """

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            from openai import OpenAI  # lazy: only the real path needs the SDK

            client = OpenAI()
        self._client = client

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": rendered_prompt}],
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


# --- Prompt loading (verbatim, checksummed) ---------------------------------


@dataclass(frozen=True)
class JudgePrompts:
    """The six judge templates, loaded verbatim with recorded checksums.

    ``templates`` maps dimension -> raw template text (with ``{prompt}``,
    ``{candidate_a}``, ``{candidate_b}`` placeholders). ``checksums`` maps
    dimension -> SHA-256 of that template, so a caller can pin the exact
    transcription a run used and detect any later edit to ``judge-prompts/``.
    """

    templates: dict[str, str]
    checksums: dict[str, str]
    source_dir: str


def _find_prompts_dir(start: Path) -> Path:
    """Walk up from ``start`` to the repo root holding ``judge-prompts/``.

    The templates live at the repository root (plan.md: "judge-prompts/ ... stay
    as-is"), outside the installed package, so the module locates them by
    ascending its own path. Raises if no ancestor contains the directory rather
    than silently substituting an empty prompt set.
    """
    for parent in [start, *start.parents]:
        candidate = parent / "judge-prompts"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "could not locate a 'judge-prompts/' directory in any parent of "
        f"{start}; pass prompts_dir explicitly to load_judge_prompts"
    )


def load_judge_prompts(prompts_dir: str | Path | None = None) -> JudgePrompts:
    """Load the six judge templates verbatim, one per dimension.

    Each template is read byte-for-byte (no stripping, no normalization) so it
    is identical to the transcribed ``judge-prompts/<dimension>.txt`` file, and a
    SHA-256 of the exact bytes is recorded per dimension.

    Args:
        prompts_dir: Directory holding ``<dimension>.txt`` files. Defaults to the
            ``judge-prompts/`` directory found by walking up from this module.

    Raises:
        FileNotFoundError: if the directory or any of the six files is missing.
    """
    if prompts_dir is not None:
        base = Path(prompts_dir)
    else:
        base = _find_prompts_dir(Path(__file__).resolve())
    templates: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for dimension in DIMENSION_ORDER:
        path = base / f"{dimension}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"judge prompt for dimension {dimension!r} not found at {path}"
            )
        # Read raw bytes and decode without any newline/whitespace transform so
        # the in-memory template is byte-identical to the file on disk.
        raw = path.read_bytes()
        templates[dimension] = raw.decode("utf-8")
        checksums[dimension] = content_sha256(templates[dimension])
    return JudgePrompts(
        templates=templates, checksums=checksums, source_dir=str(base)
    )


def render_prompt(
    template: str, *, prompt: str, candidate_a: str, candidate_b: str
) -> str:
    """Fill a judge template's placeholders.

    The templates use ``{prompt}``, ``{candidate_a}``, ``{candidate_b}``. We
    substitute by exact replacement rather than ``str.format`` so literal braces
    in any candidate text cannot raise or be misread as format fields.
    """
    return (
        template.replace("{prompt}", prompt)
        .replace("{candidate_a}", candidate_a)
        .replace("{candidate_b}", candidate_b)
    )


# --- Seeded A/B order --------------------------------------------------------


def assign_order(pair_id: str, seed: int) -> str:
    """Return the seeded A/B order for ``pair_id``: ``model_first``/``human_first``.

    The order is derived deterministically from ``(seed, pair_id)`` via a
    per-pair PRNG, so the whole sequence replays from ``seed`` alone (FR-008) and
    does not depend on iteration order or concurrency. Using the pair_id as part
    of the PRNG key means adding or removing pairs does not reshuffle the others.
    """
    rng = random.Random(f"{seed}:{pair_id}")
    return "model_first" if rng.random() < 0.5 else "human_first"


def parse_choice(raw_response: str) -> str:
    """Parse a raw judge response to ``"A"``, ``"B"``, or ``"invalid"``.

    The templates instruct the judge to "Return only A or B". We accept a
    response whose meaningful content is exactly one of the letters, tolerating
    surrounding whitespace and a trailing period, in either case. Anything
    ambiguous (both letters, neither, a sentence) is ``"invalid"`` so it is
    excluded-and-counted rather than guessed at.
    """
    stripped = raw_response.strip().rstrip(".").strip()
    upper = stripped.upper()
    if upper == "A":
        return "A"
    if upper == "B":
        return "B"
    return "invalid"


def _model_won(choice: str, order: str) -> bool | None:
    """Derive whether the model output won from the choice and A/B order.

    ``model_first`` puts the model as candidate A; ``human_first`` puts it as B.
    ``invalid`` yields ``None`` (no winner) so it never counts as a win or loss.
    """
    if choice == "invalid":
        return None
    if order == "model_first":
        return choice == "A"
    return choice == "B"


# --- Judging (one call per pair x dimension, persisted before aggregation) ---


def _judge_one(
    client: JudgeClient,
    prompts: JudgePrompts,
    pair: JudgePair,
    dimension: str,
    order: str,
    model: str,
) -> JudgeVerdict:
    """Run one (pair, dimension) comparison with one retry on a malformed reply.

    Returns a fully-populated :class:`JudgeVerdict`. On a malformed first reply
    the call is retried exactly once; if the retry is also malformed the verdict
    is recorded with ``choice="invalid"`` and ``retry_count=1`` (excluded but
    counted), never silently scored.
    """
    if order == "model_first":
        candidate_a, candidate_b = pair.model_text, pair.human_text
    else:
        candidate_a, candidate_b = pair.human_text, pair.model_text

    rendered = render_prompt(
        prompts.templates[dimension],
        prompt=pair.prompt,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    retry_count = 0
    raw_response = client.judge(rendered, model=model)
    choice = parse_choice(raw_response)
    if choice == "invalid":
        retry_count = 1
        raw_response = client.judge(rendered, model=model)
        choice = parse_choice(raw_response)

    return JudgeVerdict(
        pair_id=pair.pair_id,
        dimension=dimension,
        judge_model=model,
        order=order,
        raw_response=raw_response,
        choice=choice,
        retry_count=retry_count,
        model_won=_model_won(choice, order),
    )


def run_judging(
    pairs: Sequence[JudgePair],
    verdicts_path: str | Path,
    *,
    client: JudgeClient,
    prompts: JudgePrompts | None = None,
    seed: int = 0,
    model: str = DEFAULT_JUDGE_MODEL,
    max_workers: int = 4,
) -> list[JudgeVerdict]:
    """Judge every (pair, dimension), persist all verdicts, and return them.

    One judge call per (pair, dimension) is issued through ``client``,
    concurrency-limited to ``max_workers``. A/B order is assigned per pair from
    ``seed`` and reused across the six dimensions. Every verdict is written to
    ``verdicts_path`` as JSONL **before** this function returns and before any
    scoring, so a crash after this step loses no work: aggregation reads only the
    persisted file (see :func:`aggregate_verdicts`).

    The returned verdicts are ordered deterministically (pair order x
    :data:`DIMENSION_ORDER`) regardless of the concurrent completion order, so
    the JSONL is stable across runs with the same inputs and seed.
    """
    if prompts is None:
        prompts = load_judge_prompts()

    # Freeze the per-pair order once so it is identical across dimensions and
    # independent of scheduling.
    orders = {pair.pair_id: assign_order(pair.pair_id, seed) for pair in pairs}

    # Build the full work list in deterministic order; results are placed back
    # by index so completion order (concurrency) never affects the output order.
    tasks: list[tuple[int, JudgePair, str]] = []
    for pair in pairs:
        for dimension in DIMENSION_ORDER:
            tasks.append((len(tasks), pair, dimension))

    results: list[JudgeVerdict | None] = [None] * len(tasks)

    def _work(item: tuple[int, JudgePair, str]) -> None:
        index, pair, dimension = item
        results[index] = _judge_one(
            client,
            prompts,
            pair,
            dimension,
            orders[pair.pair_id],
            model,
        )

    if tasks:
        # max_workers must be >= 1 for ThreadPoolExecutor.
        workers = max(1, min(max_workers, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Drain the iterator so any exception in a worker propagates.
            list(executor.map(_work, tasks))

    verdicts = [v for v in results if v is not None]
    # Persist BEFORE returning / before any aggregation (FR-008). This is the
    # durability boundary: once written, JMQ is recomputable from this file
    # alone without re-querying the judge.
    write_jsonl(verdicts, verdicts_path)
    return verdicts


# --- Aggregation (reads only persisted verdicts) ----------------------------


def _score_dimension(verdicts: Sequence[JudgeVerdict]) -> dict[str, object]:
    """Score one dimension's verdicts into wins/losses/invalid/n/score.

    ``n`` is the number of *valid* comparisons (invalid excluded); ``score`` is
    ``2 * wins / n`` (the JMQ definition) over that valid n, or ``None`` when no
    valid verdicts exist so a divide-by-zero never silently reads as 0.0.
    """
    wins = sum(1 for v in verdicts if v.model_won is True)
    losses = sum(1 for v in verdicts if v.model_won is False)
    invalid = sum(1 for v in verdicts if v.choice == "invalid")
    n = wins + losses
    win_rate = wins / n if n > 0 else None
    score = 2.0 * win_rate if win_rate is not None else None
    return {
        "score": score,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "invalid": invalid,
        "n": n,
    }


def aggregate_verdicts(
    verdicts: str | Path | Sequence[JudgeVerdict],
) -> dict[str, dict[str, object]]:
    """Aggregate persisted verdicts into per-dimension and overall JMQ scores.

    Accepts either a path to the persisted verdicts JSONL (the recompute path
    that re-reads what :func:`run_judging` wrote, FR-008) or an in-memory
    sequence of verdicts. Returns a mapping ``dimension -> {score, win_rate,
    wins, losses, invalid, n}`` for each of the six dimensions.

    The score is ``2 * model_win_rate`` over the *valid* verdicts for that
    dimension; invalid verdicts are counted and reported but excluded from the
    rate, never scored as wins or losses.
    """
    if isinstance(verdicts, (str, Path)):
        loaded: Sequence[JudgeVerdict] = read_jsonl(verdicts, JudgeVerdict)
    else:
        loaded = verdicts

    by_dimension: dict[str, list[JudgeVerdict]] = {d: [] for d in DIMENSION_ORDER}
    for verdict in loaded:
        by_dimension[verdict.dimension].append(verdict)

    return {
        dimension: _score_dimension(by_dimension[dimension])
        for dimension in DIMENSION_ORDER
    }


# --- Cost preflight (FR-009) -------------------------------------------------


class CostThresholdError(RuntimeError):
    """Raised when estimated spend exceeds the threshold without confirmation."""


def estimate_call_count(num_pairs: int) -> int:
    """Estimated judge call count: ``num_pairs * len(DIMENSION_ORDER)``.

    This is the pre-retry count (one call per pair per dimension). Retries on
    malformed replies add at most one call each and are not included, so the
    estimate is a floor the caller can reason about before spending.
    """
    if num_pairs < 0:
        raise ValueError(f"num_pairs must be non-negative, got {num_pairs}")
    return num_pairs * len(DIMENSION_ORDER)


def cost_preflight(
    num_pairs: int,
    *,
    confirm: bool = False,
    threshold_usd: float = 1.0,
    cost_per_call_usd: float = DEFAULT_COST_PER_CALL_USD,
    printer=print,
) -> dict[str, object]:
    """Report estimated judge cost and gate on a confirmation above threshold.

    Prints the estimated call count (pairs x 6) and estimated spend via
    ``printer``, then, if the estimate is above ``threshold_usd`` and ``confirm``
    is not set, raises :class:`CostThresholdError` so no external calls happen
    without an explicit ``--yes`` (FR-009). Below the threshold the estimate is
    reported and the run proceeds.

    Returns the estimate dict (calls, estimated_usd, threshold_usd) so a caller
    can record it in the run config for audit.
    """
    calls = estimate_call_count(num_pairs)
    estimated_usd = calls * cost_per_call_usd
    printer(
        f"JMQ cost preflight: {num_pairs} pairs x {len(DIMENSION_ORDER)} "
        f"dimensions = {calls} judge calls, "
        f"estimated ${estimated_usd:.2f} (threshold ${threshold_usd:.2f})."
    )
    estimate = {
        "calls": calls,
        "estimated_usd": estimated_usd,
        "threshold_usd": threshold_usd,
    }
    if estimated_usd > threshold_usd and not confirm:
        raise CostThresholdError(
            f"estimated spend ${estimated_usd:.2f} exceeds threshold "
            f"${threshold_usd:.2f}; re-run with confirm=True (--yes) to proceed"
        )
    return estimate
