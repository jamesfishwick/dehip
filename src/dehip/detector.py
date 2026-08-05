"""External AI-text detector scoring -- SC-005's measurement instrument (issue #14).

SC-005 asks whether the cascade's rewrites read more human than the drafts by an
*independent* instrument, not our own MMD/token-L2/JMQ metrics. This module scores
one or more :class:`~dehip.schemas.TextSet` manifests through an external detector
(Pangram by default, GPTZero optional) and reports, per set, the mean/median/
distribution of the detector's human-probability plus every per-text score. The
SC-005 delta is then a single subtraction: rewrite-set mean human-prob minus
draft-set mean human-prob (>= +0.30 passes).

Design mirrors the rest of the harness:

- **The detector client is an injectable seam.** :class:`DetectorClient` is a
  ``Protocol``: one ``score_text(text) -> float`` method returning a
  human-probability in ``[0.0, 1.0]``. Tests inject a mock; the real Pangram /
  GPTZero adapters (:class:`PangramClient`, :class:`GPTZeroClient`) are thin glue
  that import their SDK lazily, so no test needs a real key or the network. The
  CLI checks the required env key and gates spend *before* constructing any
  client (see :mod:`dehip.cli`), so a missing key never reaches a detector call.

- **A detector call failure fails loudly.** A network error, rate-limit, or a
  malformed/out-of-range response is normalized to :class:`DetectorCallError` and
  aborts the whole set. A failed text is never silently dropped or defaulted to
  ``0.0`` -- that would corrupt the mean and read downstream as a real (very
  AI-looking) score. This is the same loud-failure discipline as generate.py's
  :class:`~dehip.generate.EmptyDraftError` and cascade.py's malformed-output
  handling: a real ``0.0`` human-probability (detector is confident the text is
  AI) is a valid datum; a *failed call* is not, and the two are kept distinct.

Persistence: both the per-text scores (a JSONL, one ``{set_id, pair_id, text_sha,
human_prob}`` record per text) and the multi-set summary (one JSON document) land
under ``results/reports/`` so a run is auditable at the row level, not just the
aggregate.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dehip.schemas import content_sha256

__all__ = [
    "DETECTORS",
    "ENV_KEY_BY_DETECTOR",
    "DEFAULT_COST_PER_TEXT_USD",
    "DEFAULT_SPEND_THRESHOLD_USD",
    "HISTOGRAM_BINS",
    "SC005_DELTA_THRESHOLD",
    "DetectorClient",
    "DetectorError",
    "DetectorCallError",
    "MissingApiKeyError",
    "DetectorScore",
    "SetSummary",
    "DetectionReport",
    "PangramClient",
    "GPTZeroClient",
    "estimate_cost",
    "cost_preflight",
    "CostThresholdError",
    "summarize_scores",
    "score_set",
    "score_sets",
    "sc005_delta",
    "scores_path_for",
    "write_detection_artifacts",
]

# Supported detectors and the env var each requires. The CLI reads
# ENV_KEY_BY_DETECTOR to enforce the key BEFORE any client is built (exit 3).
DETECTORS: tuple[str, ...] = ("pangram", "gptzero")
ENV_KEY_BY_DETECTOR: dict[str, str] = {
    "pangram": "PANGRAM_API_KEY",
    "gptzero": "GPTZERO_API_KEY",
}

# Cost-gate constants, mirroring jmq.DEFAULT_COST_PER_CALL_USD / the score
# command's $1.00 threshold. One detector call per text; a run whose estimate is
# above the threshold needs --yes.
#
# ESTIMATE (verify before relying on it for budgeting): real detector spend is
# per-word, not per-call. Pangram's API is ~$0.05 per 1,000 words
# (https://www.pangram.com/pricing); GPTZero is in the same range. A single
# harness text of ~200 words is therefore ~$0.01, and longer texts cost more, so
# $0.001 (0.1 cent) under-gates real spend by 10x-plus and would let a several-
# hundred-text run proceed without --yes. $0.01/text is a conservative floor for
# a typical short text; raise it if the corpus runs long.
DEFAULT_COST_PER_TEXT_USD = 0.01
DEFAULT_SPEND_THRESHOLD_USD = 1.0

# Fixed histogram bins for the human-probability distribution. Ten equal-width
# bins over the closed unit interval; a value of exactly 1.0 lands in the last
# bin. Fixed (not data-derived) so two runs' distributions are directly
# comparable.
HISTOGRAM_BINS = 10

# The SC-005 pass bar: the rewrite set must be at least this many probability
# points (0.30 == 30 percentage points) more human than the draft set.
SC005_DELTA_THRESHOLD = 0.30


# --- Errors ------------------------------------------------------------------


class DetectorError(RuntimeError):
    """Base class for detector failures the CLI maps to a defined exit code."""


class MissingApiKeyError(DetectorError):
    """Raised when the detector's required API key env var is unset.

    The CLI maps this to exit 3 (external-dependency failure) and, critically,
    raises it BEFORE constructing any client or issuing any call, so a missing
    key never causes partial spend.
    """


class DetectorCallError(DetectorError):
    """Raised when a detector call fails or returns an unusable response.

    Covers a network/transport error, a rate-limit, an auth rejection surfaced
    mid-run, and a malformed or out-of-range response (a human-probability that
    is not a finite float in ``[0.0, 1.0]``). The CLI maps this to exit 3.

    Failing loudly here is deliberate: a dropped or zero-defaulted text would
    corrupt the set mean and read downstream as a genuine (AI-looking) score. A
    real ``0.0`` (the detector is confident the text is machine-written) is a
    valid datum and is *not* an error; only a failed/unusable call is.
    """


# --- The injectable seam -----------------------------------------------------


@runtime_checkable
class DetectorClient(Protocol):
    """The detector seam: return one human-probability for a text.

    ``score_text`` returns a float in ``[0.0, 1.0]`` where 1.0 means "certainly
    human-written" and 0.0 means "certainly machine-written". Implementations
    raise (any exception) on a failed call; :func:`score_set` normalizes that to
    :class:`DetectorCallError`. The real implementations are :class:`PangramClient`
    and :class:`GPTZeroClient`; tests inject a mock so no key or network is
    touched.
    """

    def score_text(self, text: str) -> float:
        """Return the detector's human-probability for ``text`` (0.0..1.0)."""
        ...


# --- Records -----------------------------------------------------------------


@dataclass(frozen=True)
class DetectorScore:
    """One text's detector result, persisted as a JSONL row.

    ``text_sha`` is the content SHA-256 (the harness's dedup/identity key, see
    :func:`dehip.schemas.content_sha256`) so a row is traceable to its text
    without persisting the (possibly large) text itself.
    """

    set_id: str
    pair_id: str
    text_sha: str
    human_prob: float


@dataclass(frozen=True)
class SetSummary:
    """Per-set aggregate of the human-probability distribution.

    ``histogram`` is a fixed-bin count (see :data:`HISTOGRAM_BINS`); ``quantiles``
    carries the 25/50/75/90th percentiles. ``n`` is the number of scored texts.
    """

    set_id: str
    role: str
    n: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float
    histogram: list[int]
    quantiles: dict[str, float]


@dataclass
class DetectionReport:
    """The single artifact holding every scored set's summary (SC-005).

    All set summaries live in one document so the SC-005 delta is a single
    subtraction between two of them. ``sc005`` is filled in when exactly a draft
    and a rewrite summary are present (see :func:`sc005_delta`); otherwise it is
    left ``None`` and the raw per-set summaries still stand.
    """

    report_id: str
    detector: str
    seed: int
    n_sets: int
    sets: list[dict[str, Any]]
    timestamps: dict[str, str]
    sc005: dict[str, Any] | None = None
    caveats: list[Any] = field(default_factory=list)


# --- Real SDK adapters (thin glue behind the seam) ---------------------------


class PangramClient:
    """Thin Pangram-SDK glue behind the :class:`DetectorClient` seam.

    Everything Pangram-specific lives here so the rest of the module (and every
    unit test) never imports the SDK. The SDK and client load lazily on first
    :meth:`score_text`; an SDK-import or client-construction failure is
    normalized to :class:`DetectorCallError` (-> CLI exit 3). The key is read
    from ``PANGRAM_API_KEY`` by the SDK itself; the CLI has already enforced its
    presence before this client is ever constructed.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from pangram import Pangram  # type: ignore[import-not-found]
            except Exception as exc:  # SDK not installed / import error
                raise DetectorCallError(
                    f"pangram SDK unavailable: {exc}"
                ) from exc
            try:
                self._client = (
                    Pangram(api_key=self._api_key)
                    if self._api_key
                    else Pangram()
                )
            except Exception as exc:
                raise DetectorCallError(
                    f"pangram client construction failed: {exc}"
                ) from exc
        return self._client

    def score_text(self, text: str) -> float:
        """Return Pangram's human-probability for ``text``.

        Pangram reports an *AI likelihood* in ``[0, 1]``; human-probability is its
        complement (``1 - ai_likelihood``). Any transport/response failure is
        normalized to :class:`DetectorCallError`; :func:`score_set` also range-
        checks the returned value, so an out-of-range reply never becomes a silent
        score.
        """
        client = self._ensure_client()
        try:
            result = client.predict(text)
        except Exception as exc:
            raise DetectorCallError(f"pangram call failed: {exc}") from exc
        try:
            ai_likelihood = float(result["ai_likelihood"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorCallError(
                f"pangram response missing/invalid ai_likelihood: {result!r}"
            ) from exc
        return 1.0 - ai_likelihood


class GPTZeroClient:
    """Thin GPTZero-SDK glue behind the :class:`DetectorClient` seam.

    Same shape as :class:`PangramClient`: lazy SDK import, failures normalized to
    :class:`DetectorCallError`. GPTZero reports a ``completely_generated_prob``
    (probability the text is machine-generated); human-probability is its
    complement. The key is ``GPTZERO_API_KEY``, enforced by the CLI before
    construction.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from gptzero import GPTZero  # type: ignore[import-not-found]
            except Exception as exc:
                raise DetectorCallError(
                    f"gptzero SDK unavailable: {exc}"
                ) from exc
            try:
                self._client = (
                    GPTZero(api_key=self._api_key)
                    if self._api_key
                    else GPTZero()
                )
            except Exception as exc:
                raise DetectorCallError(
                    f"gptzero client construction failed: {exc}"
                ) from exc
        return self._client

    def score_text(self, text: str) -> float:
        client = self._ensure_client()
        try:
            result = client.predict(text)
        except Exception as exc:
            raise DetectorCallError(f"gptzero call failed: {exc}") from exc
        try:
            generated = float(result["completely_generated_prob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorCallError(
                f"gptzero response missing/invalid "
                f"completely_generated_prob: {result!r}"
            ) from exc
        return 1.0 - generated


def build_client(detector: str, *, api_key: str | None = None) -> DetectorClient:
    """Construct the real adapter for ``detector``.

    Called by the CLI only AFTER the key check has passed, so this never runs on
    a keyless machine in a test. Tests inject a mock and never reach here.
    """
    if detector == "pangram":
        return PangramClient(api_key=api_key)
    if detector == "gptzero":
        return GPTZeroClient(api_key=api_key)
    raise ValueError(f"unknown detector {detector!r}; valid: {list(DETECTORS)}")


# --- Cost gate (FR-009) ------------------------------------------------------


class CostThresholdError(RuntimeError):
    """Raised when estimated spend exceeds the threshold without confirmation."""


def estimate_cost(
    num_texts: int, *, cost_per_text_usd: float = DEFAULT_COST_PER_TEXT_USD
) -> float:
    """Estimated detector spend: ``num_texts * cost_per_text_usd`` (one call each)."""
    if num_texts < 0:
        raise ValueError(f"num_texts must be non-negative, got {num_texts}")
    return num_texts * cost_per_text_usd


def cost_preflight(
    text_counts: Sequence[tuple[str, int]],
    *,
    confirm: bool = False,
    threshold_usd: float = DEFAULT_SPEND_THRESHOLD_USD,
    cost_per_text_usd: float = DEFAULT_COST_PER_TEXT_USD,
    printer=print,
) -> dict[str, Any]:
    """Print the text count per set and gate spend above ``threshold_usd``.

    Mirrors :func:`dehip.metrics.jmq.cost_preflight`: reports the per-set text
    counts and total estimated spend via ``printer``, then raises
    :class:`CostThresholdError` if the total is above ``threshold_usd`` and
    ``confirm`` is not set (FR-009). A below-threshold run proceeds; the CLI maps
    the raised error to exit 2. Returns the estimate dict so the caller can record
    it for audit.
    """
    total_texts = sum(count for _set_id, count in text_counts)
    estimated_usd = estimate_cost(total_texts, cost_per_text_usd=cost_per_text_usd)
    for set_id, count in text_counts:
        printer(f"detect cost preflight: set {set_id!r} has {count} texts")
    printer(
        f"detect cost preflight: {total_texts} texts total = "
        f"{total_texts} detector calls, estimated ${estimated_usd:.2f} "
        f"(threshold ${threshold_usd:.2f})."
    )
    estimate = {
        "total_texts": total_texts,
        "estimated_usd": estimated_usd,
        "threshold_usd": threshold_usd,
        "per_set": [{"set_id": s, "texts": c} for s, c in text_counts],
    }
    if estimated_usd > threshold_usd and not confirm:
        raise CostThresholdError(
            f"estimated spend ${estimated_usd:.2f} exceeds threshold "
            f"${threshold_usd:.2f}; re-run with --yes to proceed"
        )
    return estimate


# --- Summary math ------------------------------------------------------------


def _histogram(values: Sequence[float], bins: int = HISTOGRAM_BINS) -> list[int]:
    """Count ``values`` into ``bins`` equal-width bins over ``[0.0, 1.0]``.

    Bin edges are ``i/bins`` for ``i`` in ``0..bins``. A value ``v`` lands in bin
    ``floor(v * bins)``, with the closed upper edge (``v == 1.0``) folded into the
    last bin so it is not lost. Values outside the unit interval are clamped to
    the nearest bin rather than dropped (an out-of-range value never reaches here
    because :func:`score_set` range-checks before summarizing).
    """
    counts = [0] * bins
    for value in values:
        idx = int(value * bins)
        if idx >= bins:
            idx = bins - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1
    return counts


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """Return the 25/50/75/90th percentiles via linear interpolation.

    Uses :func:`statistics.quantiles` (inclusive method) for a robust, dependency-
    free computation. A single value yields that value for every quantile; the
    empty case is handled by :func:`summarize_scores` before this is called.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        only = ordered[0]
        return {"p25": only, "p50": only, "p75": only, "p90": only}
    # quantiles(n=100) gives the 1st..99th percentile cut points (99 values).
    cuts = statistics.quantiles(ordered, n=100, method="inclusive")
    return {
        "p25": cuts[24],
        "p50": cuts[49],
        "p75": cuts[74],
        "p90": cuts[89],
    }


def summarize_scores(
    set_id: str, role: str, human_probs: Sequence[float]
) -> SetSummary:
    """Aggregate a set's human-probabilities into a :class:`SetSummary`.

    Computes mean, median, population stdev, min, max, a fixed-bin histogram, and
    quantiles over ``human_probs``. Raises :class:`ValueError` on an empty set: a
    summary over zero texts has no defensible mean, and silently returning ``0.0``
    would read as a real (very AI-looking) score. The caller (``score_set``)
    guarantees this list contains one entry per scored text with no failed text
    dropped, so the mean is over the true membership.
    """
    values = list(human_probs)
    if not values:
        raise ValueError(
            f"cannot summarize set {set_id!r}: no human-probabilities "
            "(an empty set has no defensible mean)"
        )
    return SetSummary(
        set_id=set_id,
        role=role,
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        histogram=_histogram(values),
        quantiles=_quantiles(values),
    )


# --- Scoring a set through the seam ------------------------------------------


def _valid_human_prob(value: Any) -> bool:
    """True iff ``value`` is a finite float in the closed unit interval."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    v = float(value)
    if v != v:  # NaN
        return False
    return 0.0 <= v <= 1.0


def score_set(
    set_id: str,
    role: str,
    texts: Sequence[tuple[str, str]],
    *,
    client: DetectorClient,
) -> tuple[SetSummary, list[DetectorScore]]:
    """Score one set of ``(pair_id, text)`` texts through the detector seam.

    Calls ``client.score_text`` once per text. Every text contributes exactly one
    score: a call that raises (any exception) or returns a value that is not a
    finite float in ``[0.0, 1.0]`` is normalized to :class:`DetectorCallError` and
    aborts the whole set. This is the loud-failure contract -- a failed text is
    never dropped or zero-defaulted, so the summary mean is always over the full,
    real membership and can never be silently corrupted by a swallowed failure.

    Returns the :class:`SetSummary` and the list of per-text :class:`DetectorScore`
    rows (in input order) for persistence.
    """
    scores: list[DetectorScore] = []
    human_probs: list[float] = []
    for pair_id, text in texts:
        try:
            raw = client.score_text(text)
        except DetectorCallError:
            raise
        except Exception as exc:
            raise DetectorCallError(
                f"detector call failed for set {set_id!r} pair {pair_id!r}: {exc}"
            ) from exc
        if not _valid_human_prob(raw):
            raise DetectorCallError(
                f"detector returned an out-of-range/invalid human-probability "
                f"{raw!r} for set {set_id!r} pair {pair_id!r}; a value outside "
                "[0.0, 1.0] is a failed call, not a real score"
            )
        human_prob = float(raw)
        human_probs.append(human_prob)
        scores.append(
            DetectorScore(
                set_id=set_id,
                pair_id=pair_id,
                text_sha=content_sha256(text),
                human_prob=human_prob,
            )
        )
    summary = summarize_scores(set_id, role, human_probs)
    return summary, scores


def score_sets(
    sets: Sequence[tuple[str, str, Sequence[tuple[str, str]]]],
    *,
    client: DetectorClient,
) -> tuple[list[SetSummary], list[DetectorScore]]:
    """Score several ``(set_id, role, texts)`` sets, aggregating all per-text rows.

    Each set is scored independently by :func:`score_set`. A failure in any set
    propagates as :class:`DetectorCallError` (the whole run fails loudly rather
    than emitting a partial report). Returns the per-set summaries and the flat
    list of every per-text score across all sets.
    """
    summaries: list[SetSummary] = []
    all_scores: list[DetectorScore] = []
    for set_id, role, texts in sets:
        summary, scores = score_set(set_id, role, texts, client=client)
        summaries.append(summary)
        all_scores.extend(scores)
    return summaries, all_scores


# --- SC-005 delta ------------------------------------------------------------


def sc005_delta(summaries: Sequence[SetSummary]) -> dict[str, Any] | None:
    """Compute the SC-005 delta (rewrite mean human-prob minus draft mean).

    Returns ``None`` unless the summaries contain exactly one draft-role set and
    exactly one rewrite-role set (the two-set SC-005 comparison); otherwise the
    raw per-set summaries stand on their own and the delta is left unset rather
    than guessed. When both are present, the delta is a single subtraction and
    ``passed`` records whether it meets :data:`SC005_DELTA_THRESHOLD`.

    Role mapping: ``instruct_draft`` is the draft baseline; ``rewrite`` is the
    cascade output. A set with any other role does not participate.
    """
    drafts = [s for s in summaries if s.role == "instruct_draft"]
    rewrites = [s for s in summaries if s.role == "rewrite"]
    if len(drafts) != 1 or len(rewrites) != 1:
        return None
    draft, rewrite = drafts[0], rewrites[0]
    delta = rewrite.mean - draft.mean
    return {
        "draft_set": draft.set_id,
        "rewrite_set": rewrite.set_id,
        "draft_mean_human_prob": draft.mean,
        "rewrite_mean_human_prob": rewrite.mean,
        "delta": delta,
        "threshold": SC005_DELTA_THRESHOLD,
        "passed": delta >= SC005_DELTA_THRESHOLD,
    }


# --- Report assembly + persistence -------------------------------------------


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing Z, second resolution."""
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def assemble_report(
    *,
    report_id: str,
    detector: str,
    seed: int,
    summaries: Sequence[SetSummary],
    thresholds: dict[str, Any] | None = None,
    started: str | None = None,
    finished: str | None = None,
) -> DetectionReport:
    """Compose per-set summaries into the single :class:`DetectionReport`.

    All summaries land in one document (so the SC-005 delta is one subtraction);
    :func:`sc005_delta` fills ``sc005`` when a draft/rewrite pair is present. A
    ``below_two_sets`` caveat is attached when the delta cannot be computed, so a
    single-set run's report says why the SC-005 field is empty rather than
    reading as a silent omission. The cost estimate (``thresholds``) is recorded
    when supplied so authorized spend is auditable.
    """
    sc005 = sc005_delta(summaries)
    caveats: list[Any] = []
    if sc005 is None:
        caveats.append(
            {
                "kind": "sc005_not_computed",
                "message": (
                    "SC-005 delta needs exactly one draft (instruct_draft) and "
                    "one rewrite set; the delta is left unset and the per-set "
                    "summaries stand on their own."
                ),
            }
        )
    return DetectionReport(
        report_id=report_id,
        detector=detector,
        seed=seed,
        n_sets=len(summaries),
        sets=[asdict(s) for s in summaries],
        timestamps={
            "started": started or _now_iso(),
            "finished": finished or _now_iso(),
        },
        sc005=sc005,
        caveats=(
            caveats
            + ([{"kind": "cost_estimate", **thresholds}] if thresholds else [])
        ),
    )


def scores_path_for(out_path: str | Path) -> Path:
    """Return the per-text scores sibling for a ``.json`` summary ``out_path``.

    The scores JSONL is named from the summary's *full* stem so a multi-dot path
    is not silently mangled: ``a.b.json`` -> ``a.b.scores.jsonl`` (using the whole
    ``a.b`` stem), never ``a.scores.jsonl``. ``out_path`` must end in ``.json``;
    a non-``.json`` path raises :class:`ValueError` (the CLI maps that to exit 2)
    rather than deriving a mangled sibling from an unexpected suffix.
    """
    summary_path = Path(out_path)
    if summary_path.suffix != ".json":
        raise ValueError(
            f"--out must end in .json, got {str(out_path)!r}; a non-.json path "
            "would mangle the derived scores sibling and report_id"
        )
    return summary_path.with_name(f"{summary_path.stem}.scores.jsonl")


def write_detection_artifacts(
    report: DetectionReport,
    scores: Sequence[DetectorScore],
    *,
    out_path: str | Path,
) -> tuple[Path, Path]:
    """Persist BOTH the per-text scores and the summary report under ``out_path``.

    Writes two artifacts beside ``out_path`` and commits them in a fixed order so
    a mid-commit failure never leaves a misleading complete-looking summary with
    no scores:

    - the per-text scores JSONL at ``<full stem>.scores.jsonl`` (the row-level
      audit trail -- committed FIRST because it matters most), and
    - the summary JSON at ``out_path`` (the single SC-005 artifact, committed
      SECOND).

    Both texts are staged to temp files first; if either staging write fails,
    both temps are cleaned up and NEITHER final artifact exists. The commits then
    run scores-first, summary-second, each via :func:`os.replace`. If the second
    commit (the summary) fails, the already-committed scores file is rolled back
    (unlinked) so neither final artifact survives -- an orphaned scores file with
    no summary would otherwise linger, and (worse, before this ordering) an
    orphaned summary with no scores would read downstream as a successful run.
    The guarantee is therefore all-or-nothing on the final pair, with no temp
    debris left behind on any failure path.

    ``out_path`` must end in ``.json`` (see :func:`scores_path_for`); a non-.json
    path raises :class:`ValueError`.
    """
    import os

    summary_path = Path(out_path)
    scores_path = scores_path_for(out_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_text = json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n"
    scores_text = "".join(
        json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in scores
    )

    # Stage both temps first. If either write fails, unlink every temp and leave
    # no final artifact. Order the staged pair scores-first so the commit loop
    # writes the row-level audit trail before the summary that consumers trust.
    staged: list[tuple[Path, Path]] = []
    try:
        for final, text in (
            (scores_path, scores_text),
            (summary_path, summary_text),
        ):
            tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
            tmp.write_text(text, encoding="utf-8")
            staged.append((tmp, final))
    except OSError:
        for tmp, _final in staged:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    # Commit in staged (scores-first) order, tracking what landed. If a commit
    # fails partway, roll back every already-committed final AND drop any temps
    # not yet committed, so neither final artifact survives and no temp lingers.
    committed: list[Path] = []
    try:
        for tmp, final in staged:
            os.replace(tmp, final)
            committed.append(final)
    except OSError:
        for final in committed:
            try:
                final.unlink()
            except OSError:
                pass
        for tmp, _final in staged:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    return summary_path, scores_path
