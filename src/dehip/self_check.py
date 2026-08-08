"""Self-check mode: score a human reference set against itself (FR-003, SC-001).

The self-check is the harness's own smoke test. It takes ONE human reference
set, splits it into two disjoint halves, and scores half A against half B
through the exact same :func:`dehip.report.score` path a real run uses. Because
both halves are drawn from the same human distribution, the metrics must come
back near their identity values: MMD near zero, token-L2 near zero, and a JMQ
win-rate near 50% (JMQ score near 1.0, since JMQ is ``2 * win_rate``). If they
do not, the harness itself is broken -- a metric bug that would otherwise
masquerade as a finding -- and the check fails loudly (exit 4) naming which
bound was exceeded and by how much.

Design invariants the adversarial review probes
-----------------------------------------------
- **Split by pair, seeded, disjoint.** The split is over the reference set's
  pair_ids, shuffled by a seeded PRNG and cut in half. No pair_id lands in both
  halves (:func:`split_pairs` asserts disjointness), so a text can never be
  compared against itself, which would trivially force MMD/token-L2 to a fake
  zero and hide a real metric regression.
- **Scored through the real report path.** :func:`run_self_check` builds a
  :class:`~dehip.report.MetricInputs` from the two halves and calls
  :func:`dehip.report.score` -- the same metric composition #10 built and every
  real run uses. A metric regression therefore surfaces here, because the check
  runs the production code, not a shortcut reimplementation.
- **Odd-N behavior is defined.** An odd-sized set cannot split into two equal
  halves. The pairing is positional (half A pair i vs half B pair i), so the
  extra pair would have no partner; we drop the single leftover pair from the
  larger half after the shuffle, and record it in the result
  (``dropped_pair_id``). The drop is deterministic given the seed. A set with
  fewer than ``2 * DEFAULT_MIN_N`` pairs cannot yield two scorable halves and is
  rejected as a validation error.

The two halves are re-keyed onto a shared synthetic pair-id space
(``sc-0``, ``sc-1``, ...) so :func:`dehip.report.score` -- which requires the
candidate and reference sets to share identical pair_ids (a paired comparison)
-- pairs half-A position i against half-B position i. This is a positional
pairing of two independent halves, not a self-comparison: half A's texts and
half B's texts come from different source pairs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dehip.metrics.bounds import StubInstrumentBounds, jmq_scaled_window
from dehip.report import MetricInputs, score
from dehip.validate import DEFAULT_MIN_N, InputSetValidationError

# The self-check requires at least this many valid JMQ comparisons for the
# win-rate gate to mean anything. A non-skip run that produces fewer (a broken
# judge emitting all-invalid verdicts collapses to n=0) FAILS LOUDLY (CRITICAL
# 2) rather than skipping the window and reporting a spurious pass -- a broken
# judge must be distinguishable from a deliberately-skipped one.
JMQ_MIN_VALID_COMPARISONS = 1

__all__ = [
    "SelfCheckResult",
    "SelfCheckOutOfBounds",
    "SelfCheckIntegrityError",
    "StubInstrumentBounds",
    "split_pairs",
    "load_reference_set",
    "run_self_check",
]


class SelfCheckIntegrityError(AssertionError):
    """Raised when the self-check's own construction is broken (not a metric bound).

    A structural invariant of the check itself failed -- e.g. the seeded split
    leaked a pair_id into both halves, which would let a text be compared against
    itself and fake a zero. This is a fail-loudly case distinct from a metric
    being out of bounds; the CLI maps it to the self-check exit code (4), never a
    bare ``AssertionError`` escaping as exit 1 (IMPORTANT 3).
    """


class SelfCheckOutOfBounds(AssertionError):
    """Raised when a self-check metric falls outside its documented noise bound.

    The message always names which bound was exceeded and by how much, so the
    failure is loud and diagnosable rather than a silent pass (FR-003). Carries
    the machine-readable list of violations for a caller that wants to report
    them structurally.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            "self-check out of bounds: " + "; ".join(violations)
        )


@dataclass(frozen=True)
class SelfCheckResult:
    """Outcome of one self-check run.

    Attributes:
        report: The :class:`~dehip.schemas.MetricReport` produced by scoring the
            two halves through the real report path.
        mmd: The MMD^2 value scored (NaN if MMD did not run, which self-check
            never allows -- MMD always runs).
        token_l2: The token-L2 distance scored.
        jmq_win_rate: The overall JMQ model win-rate, or None when JMQ was
            skipped (``--skip-jmq``). On a non-skip run that produced no valid
            verdicts this is None AND a violation is recorded (CRITICAL 2), so a
            None here on a non-skip run is a loud failure, not a quiet pass.
        jmq_n: The number of valid JMQ comparisons the win-rate was computed over,
            or None when JMQ was skipped. Recorded so the scaled window is
            auditable.
        jmq_window: The effective ``(lo, hi)`` scaled win-rate window the run was
            gated against (centered on 0.5, scaled to ``jmq_n``), or None when JMQ
            was skipped or produced no valid comparisons. Auditable (CRITICAL 1).
        half_size: Number of pairs in each half (the two halves are equal-sized).
        dropped_pair_id: The single pair_id dropped for an odd-N set, or None
            when N was even. Recorded so the drop is auditable.
        violations: Human-readable descriptions of every bound that was
            exceeded; empty when the check passed.
    """

    report: Any
    mmd: float
    token_l2: float
    jmq_win_rate: float | None
    jmq_n: int | None
    jmq_window: tuple[float, float] | None
    half_size: int
    dropped_pair_id: str | None
    violations: list[str]

    @property
    def passed(self) -> bool:
        return not self.violations


def split_pairs(
    pair_ids: Sequence[str], *, seed: int
) -> tuple[list[str], list[str], str | None]:
    """Split ``pair_ids`` into two disjoint, equal-sized halves after a seeded shuffle.

    The shuffle is driven by ``random.Random(seed)`` so the split is fully
    reproducible from the seed alone. For an odd count, the single leftover pair
    (the last one after the shuffle) is dropped so the two halves are equal-sized
    and can be paired positionally; its id is returned as the third tuple element
    so the drop is recorded rather than silent.

    Returns ``(half_a, half_b, dropped_pair_id)`` where ``dropped_pair_id`` is
    ``None`` for an even count. ``half_a`` and ``half_b`` are guaranteed disjoint
    (a text never lands in both halves).

    Raises:
        InputSetValidationError: when fewer than ``2 * DEFAULT_MIN_N`` pairs are
            available, since two halves each need at least ``DEFAULT_MIN_N`` pairs
            to be scorable (MMD needs >= 2 points per set).
    """
    ids = list(pair_ids)
    min_total = 2 * DEFAULT_MIN_N
    if len(ids) < min_total:
        raise InputSetValidationError(
            f"self-check needs at least {min_total} pairs to form two halves of "
            f"{DEFAULT_MIN_N} each, got {len(ids)}"
        )

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)

    dropped: str | None = None
    if len(shuffled) % 2 == 1:
        # Odd N: drop the single leftover so the halves are equal-sized and can
        # be paired positionally. Deterministic given the seed (the shuffle is).
        dropped = shuffled.pop()

    half = len(shuffled) // 2
    half_a = shuffled[:half]
    half_b = shuffled[half:]

    # Disjointness is structural (a partition of a de-duplicated list), but assert
    # it: a pair leaking into both halves would let a text be compared against
    # itself and fake a zero, defeating the whole check.
    if set(half_a) & set(half_b):
        raise SelfCheckIntegrityError(
            "self-check split produced overlapping halves; a pair_id leaked into "
            "both, which would let a text be compared against itself"
        )
    return half_a, half_b, dropped


def _read_pair_texts(path: Path) -> dict[str, str]:
    """Read a ``{pair_id, text}`` JSONL into a mapping, rejecting duplicate ids.

    Mirrors :func:`dehip.report._read_pair_texts` (the texts-file shape is the
    same one score() consumes); kept local so self-check does not reach into a
    private helper.
    """
    import json

    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pair_id = record["pair_id"]
            if pair_id in texts:
                raise InputSetValidationError(
                    f"duplicate pair_id {pair_id!r} in reference texts file {path}"
                )
            texts[pair_id] = record["text"]
    return texts


def load_reference_set(
    reference_manifest: str,
) -> tuple[list[str], dict[str, str], str]:
    """Load a human reference TextSet manifest and its sibling texts JSONL.

    Resolves the texts file the same way :func:`dehip.report.load_scoring_inputs`
    does (an explicit ``provenance.texts_path`` or the ``.manifest.json`` ->
    ``.jsonl`` sibling convention), reads the ``{pair_id, text}`` rows, and
    cross-checks that the manifest's pair_ids exactly equal the texts-file keys so
    a mis-pointed or superset texts file is caught before any scoring.

    Returns ``(pair_ids, texts_by_id, set_id)``.

    Raises:
        InputSetValidationError: on a manifest/texts id mismatch.
    """
    from dehip.report import _texts_path_for
    from dehip.schemas import TextSet, read_json

    manifest_path = Path(reference_manifest)
    ref_set: TextSet = read_json(manifest_path, TextSet)
    texts = _read_pair_texts(_texts_path_for(manifest_path, ref_set.provenance))

    manifest_ids = set(ref_set.pair_ids)
    texts_ids = set(texts.keys())
    if manifest_ids != texts_ids:
        only_manifest = sorted(manifest_ids - texts_ids)
        only_texts = sorted(texts_ids - manifest_ids)
        raise InputSetValidationError(
            "reference manifest and its texts file name different pair_ids; "
            f"only in manifest: {only_manifest}; only in texts: {only_texts}"
        )
    return list(ref_set.pair_ids), texts, ref_set.set_id


def _build_split_inputs(
    half_a: Sequence[str],
    half_b: Sequence[str],
    texts: dict[str, str],
) -> MetricInputs:
    """Re-key two disjoint halves onto a shared synthetic pair-id space.

    ``score`` requires the candidate and reference sets to share identical
    pair_ids (a paired comparison keyed by id). Half A and half B are drawn from
    DIFFERENT source pairs, so they have no shared ids. We relabel them onto a
    common ``sc-0..sc-(k-1)`` space, pairing half-A position i (candidate) against
    half-B position i (reference). Each side's original text is preserved; only
    the id is synthetic. Prompts are supplied (a fixed placeholder per synthetic
    id) so JMQ can run when requested.
    """
    k = len(half_a)
    synthetic_ids = [f"sc-{i}" for i in range(k)]
    candidate_texts = {synthetic_ids[i]: texts[half_a[i]] for i in range(k)}
    reference_texts = {synthetic_ids[i]: texts[half_b[i]] for i in range(k)}
    prompts = {sid: f"self-check pair {sid}" for sid in synthetic_ids}
    return MetricInputs(
        pair_ids=synthetic_ids,
        candidate_texts=candidate_texts,
        reference_texts=reference_texts,
        prompts=prompts,
    )


def _check_bounds(
    mmd: float,
    token_l2: float,
    jmq_win_rate: float | None,
    jmq_n: int | None,
    bounds: StubInstrumentBounds,
    *,
    skip_jmq: bool,
) -> tuple[list[str], tuple[float, float] | None]:
    """Collect every bound violation and the effective JMQ window.

    Each metric is checked against its documented noise bound; a violation string
    names the bound, the observed value, the limit, and the overage so the failure
    is diagnosable, not a bare boolean. Everything is checked (no short-circuit)
    so one run reports every problem at once.

    Returns ``(violations, jmq_window)`` where ``jmq_window`` is the effective
    ``(lo, hi)`` scaled win-rate window this run was gated against (or None when
    JMQ was skipped or produced no valid comparisons), recorded for audit.

    MMD is checked against BOTH an upper and a lower bound. The upper check keeps
    the ``not (x <= max)`` form so a NaN (never <= max) trips a violation. The
    lower check traps a sign-flip/broken-kernel regression driving unbiased MMD^2
    strongly negative (which the upper-only check would pass).

    The JMQ win-rate is gated against a window SCALED to the valid comparison
    count ``jmq_n`` (centered on 0.5), not the fixed [0.45,0.55] target, because
    at the smoke tier that fixed window is statistically unsatisfiable for a fair
    judge (CRITICAL 1). A non-skip run that produced no valid comparisons FAILS
    LOUDLY (CRITICAL 2) rather than skipping the check.
    """
    violations: list[str] = []
    jmq_window: tuple[float, float] | None = None

    # MMD upper bound. `not (mmd <= max)` (not `mmd > max`) so NaN trips it too.
    if not (mmd <= bounds.mmd_max):
        violations.append(
            f"MMD^2 {mmd:.6g} exceeds the documented noise bound "
            f"{bounds.mmd_max:.6g} by {mmd - bounds.mmd_max:.6g}"
        )
    # MMD lower bound: unbiased MMD^2 straddles zero, so a strongly-negative value
    # is a regression, not noise. `not (mmd >= min)` keeps a NaN tripping here too.
    elif not (mmd >= bounds.mmd_min):
        violations.append(
            f"MMD^2 {mmd:.6g} is below the documented noise bound "
            f"{bounds.mmd_min:.6g} by {bounds.mmd_min - mmd:.6g}"
        )
    # token-L2 is a distance (>= 0); the noise bound is an upper limit on it.
    if not (token_l2 <= bounds.token_l2_max):
        violations.append(
            f"token-L2 {token_l2:.6g} exceeds the documented noise bound "
            f"{bounds.token_l2_max:.6g} by {token_l2 - bounds.token_l2_max:.6g}"
        )

    if skip_jmq:
        return violations, jmq_window

    # JMQ was REQUESTED. A run that produced no usable win-rate (broken judge:
    # every verdict invalid -> jmq_n == 0 -> win_rate None) must fail loudly, not
    # skip the window (CRITICAL 2). Distinguish this from --skip-jmq above.
    if jmq_win_rate is None or jmq_n is None or jmq_n < JMQ_MIN_VALID_COMPARISONS:
        violations.append(
            "JMQ was requested but the judge produced no valid win-rate "
            f"(valid comparisons n={jmq_n if jmq_n is not None else 0}, floor "
            f"{JMQ_MIN_VALID_COMPARISONS}); a broken judge must not pass as a "
            "skipped one"
        )
        return violations, jmq_window

    lo, hi = jmq_scaled_window(jmq_n)
    jmq_window = (lo, hi)
    if jmq_win_rate < lo:
        violations.append(
            f"JMQ win-rate {jmq_win_rate:.4f} is below the scaled window "
            f"[{lo:.4f}, {hi:.4f}] (n={jmq_n}, centered on 0.50) by "
            f"{lo - jmq_win_rate:.4f}"
        )
    elif jmq_win_rate > hi:
        violations.append(
            f"JMQ win-rate {jmq_win_rate:.4f} is above the scaled window "
            f"[{lo:.4f}, {hi:.4f}] (n={jmq_n}, centered on 0.50) by "
            f"{jmq_win_rate - hi:.4f}"
        )

    return violations, jmq_window


def run_self_check(
    reference_manifest: str,
    *,
    seed: int = 0,
    skip_jmq: bool = False,
    embed_cache: Any = None,
    tokenizer: Any = None,
    judge_client: Any = None,
    verdicts_path: str | None = None,
    bounds: StubInstrumentBounds | None = None,
    judge_model: str | None = None,
    embedder_id: str | None = None,
    raise_on_violation: bool = True,
) -> SelfCheckResult:
    """Split a human reference set in half and score half vs half (FR-003).

    Loads the reference set, splits its pairs into two disjoint equal halves
    (:func:`split_pairs`), scores half A (candidate) vs half B (reference) through
    the real :func:`dehip.report.score` path, and asserts the results sit inside
    the documented noise bounds. Because both halves are human text from the same
    distribution, the metrics must be near their identity values.

    Args:
        reference_manifest: Path to the human reference TextSet manifest.
        seed: Drives the split shuffle and the JMQ A/B order; a run reproduces
            exactly from it.
        skip_jmq: When True, JMQ is not run and NO judge is constructed or called
            (zero judge spend). Only MMD and token-L2 are checked.
        embed_cache: EmbeddingCache seam for MMD (required unless caller stubs it;
            the CLI builds the real one).
        tokenizer: Tokenizer seam for token-L2 (None builds the real Qwen3
            tokenizer).
        judge_client: JudgeClient seam for JMQ; ignored when ``skip_jmq``.
        verdicts_path: Where JMQ persists verdicts; required when JMQ runs.
        bounds: The noise bounds to assert against; defaults to the documented
            stub-instrument bounds.
        judge_model / embedder_id: Instrument identities recorded in the report.
        raise_on_violation: When True (default), raise
            :class:`SelfCheckOutOfBounds` if any bound is exceeded. When False,
            the violations are only recorded on the result (used by tests that
            want to inspect them without an exception).

    Returns:
        A :class:`SelfCheckResult` with the report, the scored values, the split
        metadata, and any bound violations.

    Raises:
        InputSetValidationError: on a bad reference set (too few pairs, mismatch).
        SelfCheckOutOfBounds: when a metric is out of bounds and
            ``raise_on_violation`` is True.
    """
    # Resolve the embedder that actually ran, not a config default (NIT). Prefer an
    # explicit embedder_id, else read it off the injected cache/embedder, so the
    # report's embedder_id can never silently disagree with the instrument used.
    # This must precede the bounds default: bounds are instrument-specific, so
    # documented() selects the real vs stub set from the embedder identity.
    effective_embedder_id = embedder_id
    if effective_embedder_id is None and embed_cache is not None:
        effective_embedder_id = getattr(embed_cache, "embedder_id", None)

    if bounds is None:
        bounds = StubInstrumentBounds.documented(effective_embedder_id)

    pair_ids, texts, set_id = load_reference_set(reference_manifest)
    half_a, half_b, dropped = split_pairs(pair_ids, seed=seed)
    inputs = _build_split_inputs(half_a, half_b, texts)

    metrics = "mmd,token_l2" if skip_jmq else "mmd,token_l2,jmq"

    score_kwargs: dict[str, Any] = {
        "report_id": f"self-check-{set_id}",
        "candidate_set": f"{set_id}#half-a",
        "reference_set": f"{set_id}#half-b",
        "embed_cache": embed_cache,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "seed": seed,
    }
    if effective_embedder_id is not None:
        score_kwargs["embedder_id"] = effective_embedder_id
    if judge_model is not None:
        score_kwargs["judge_model"] = judge_model
    if not skip_jmq:
        # Only wire the judge seam when JMQ actually runs, so --skip-jmq never
        # constructs a judge client (zero judge spend, proven by a call-counting
        # stub that raises if touched).
        score_kwargs["judge_client"] = judge_client
        score_kwargs["verdicts_path"] = verdicts_path

    report = score(inputs, **score_kwargs)

    mmd = float(report.mmd)
    token_l2 = float(report.token_l2)
    jmq_win_rate: float | None = None
    jmq_n: int | None = None
    if not skip_jmq:
        overall = report.jmq.get("overall") if report.jmq else None
        if isinstance(overall, dict):
            jmq_win_rate = overall.get("win_rate")
            # The count of VALID comparisons the win-rate was computed over.
            # jmq.py sets win_rate=None and n=0 when every verdict is invalid;
            # both surface here so CRITICAL 2 can fail an all-invalid run loudly.
            raw_n = overall.get("n")
            jmq_n = int(raw_n) if isinstance(raw_n, (int, float)) else 0

    violations, jmq_window = _check_bounds(
        mmd, token_l2, jmq_win_rate, jmq_n, bounds, skip_jmq=skip_jmq
    )

    result = SelfCheckResult(
        report=report,
        mmd=mmd,
        token_l2=token_l2,
        jmq_win_rate=jmq_win_rate,
        jmq_n=None if skip_jmq else jmq_n,
        jmq_window=jmq_window,
        half_size=len(half_a),
        dropped_pair_id=dropped,
        violations=violations,
    )

    if violations and raise_on_violation:
        raise SelfCheckOutOfBounds(violations)
    return result
