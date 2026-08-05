"""MetricReport assembly, score orchestration, and rendering (FR-001, FR-004, FR-008).

Two responsibilities live here:

1. **Assembly.** :func:`assemble_report` composes already-computed metric values
   into one :class:`~dehip.schemas.MetricReport` per data-model.md: a ``config``
   block recording the seed, judge/embedder/tokenizer identities and the MMD
   bandwidth actually used; auto-attached caveats (small-N below a documented
   floor, bandwidth comparability, non-default judge); and a bias audit of the
   A/B order distribution so positional bias is auditable (FR-002, Story 1
   scenario 4). :func:`render_markdown` renders a readable table alongside the
   JSON.

2. **Orchestration.** :func:`score` runs the three metrics (MMD, token-L2, JMQ)
   over a candidate set vs a reference set in one invocation (FR-001), through
   injectable seams (embedder, tokenizer, judge client) so tests never touch a
   real model or the network. Validation runs before any scoring spend (FR-009).
   The JMQ half is recomputable from persisted verdicts with zero judge calls
   (FR-008) via :func:`recompute_jmq`.

Every metric selection is honored: the ``metrics`` argument lets a subset run,
so a report can carry MMD-only or JMQ-only results without the others being
computed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dehip.metrics import jmq as jmq_mod
from dehip.metrics import mmd as mmd_mod
from dehip.metrics import token_l2 as token_l2_mod
from dehip.metrics.embeddings import EmbeddingCache
from dehip.metrics.jmq import (
    DEFAULT_JUDGE_MODEL,
    DIMENSION_ORDER,
    JudgePair,
    aggregate_verdicts,
)
from dehip.schemas import JudgeVerdict, MetricReport
from dehip.validate import DEFAULT_MIN_N, validate_input_sets

__all__ = [
    "SMALL_N_FLOOR",
    "ALL_METRICS",
    "MetricInputs",
    "score",
    "recompute_jmq",
    "assemble_report",
    "order_distribution",
    "build_caveats",
    "render_markdown",
    "load_scoring_inputs",
    "report_to_jsonable",
    "PERSONAL_CORPUS",
    "COMPARISON_METRICS",
    "PersonalCorpusBenchmarkError",
    "report_corpus",
    "metric_delta",
    "assemble_comparison",
    "render_comparison",
]

# Documented small-N floor for the caveat. Distinct from validate's DEFAULT_MIN_N
# (2), which is the *hard* gate below which a run is refused outright: this is the
# soft floor below which a run still proceeds but the report carries a warning
# that its numbers are noisy. research R5 leans on the self-check + a baseline run
# to calibrate scale; small samples make MMD/token-L2/JMQ all high-variance, so a
# run under this many pairs is flagged rather than trusted silently.
SMALL_N_FLOOR = 30

# The three metrics score composes, in the fixed order they render.
ALL_METRICS: tuple[str, ...] = ("mmd", "token_l2", "jmq")


@dataclass(frozen=True)
class MetricInputs:
    """Paired candidate/reference texts and prompts for one scoring run.

    ``pair_ids`` is the shared, order-preserving membership of both sets (already
    validated identical). ``candidate_texts`` and ``reference_texts`` map each
    pair_id to its text (model output and human reference respectively).
    ``prompts`` maps each pair_id to the originating prompt, needed by JMQ; it may
    be omitted when JMQ is not requested. ``corpus`` is the homogeneous corpus tag
    of the scored sets (read from the candidate ``TextSet``); it is threaded into
    the report's ``compared["corpus"]`` so the FR-010 benchmark gate has the
    primary corpus signal on real reports, not only the set-id naming fallback.
    """

    pair_ids: Sequence[str]
    candidate_texts: dict[str, str]
    reference_texts: dict[str, str]
    prompts: dict[str, str] | None = None
    corpus: str | None = None


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing Z, second resolution."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


# --- Manifest loading (CLI glue) ---------------------------------------------


def _read_pair_texts(path: Path) -> dict[str, str]:
    """Read a ``{pair_id, text}`` JSONL into a mapping, rejecting duplicate ids."""
    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pair_id = record["pair_id"]
            if pair_id in texts:
                raise ValueError(
                    f"duplicate pair_id {pair_id!r} in texts file {path}"
                )
            texts[pair_id] = record["text"]
    return texts


def _texts_path_for(manifest_path: Path, provenance: dict[str, Any]) -> Path:
    """Resolve the texts JSONL a manifest points at.

    Prefers an explicit ``provenance.texts_path`` (relative to the manifest when
    not absolute); falls back to a sibling with the ``.manifest.json`` suffix
    swapped for ``.jsonl`` (the quickstart convention where
    ``fineweb-smoke.manifest.json`` sits beside ``fineweb-smoke.jsonl``).
    """
    raw = provenance.get("texts_path")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return manifest_path.parent / candidate
    name = manifest_path.name
    if name.endswith(".manifest.json"):
        stem = name[: -len(".manifest.json")]
    else:
        stem = manifest_path.stem
    return manifest_path.parent / f"{stem}.jsonl"


def _assert_ids_match(
    left_name: str,
    left_ids: Sequence[str],
    right_name: str,
    right_ids: Sequence[str],
) -> None:
    """Raise :class:`InputSetValidationError` unless the two id sets are identical.

    Reports the asymmetric difference (only-in-left / only-in-right) so a
    hand-broken or mis-pointed manifest is pinpointed, not merely rejected. Runs
    before any scoring spend so a mismatched reference is caught at load time
    (exit 2), never as a later KeyError while embedding or judging.
    """
    from dehip.validate import InputSetValidationError

    left_set, right_set = set(left_ids), set(right_ids)
    if left_set != right_set:
        only_left = sorted(left_set - right_set)
        only_right = sorted(right_set - left_set)
        raise InputSetValidationError(
            f"pair_id mismatch between {left_name} and {right_name}; "
            f"only in {left_name}: {only_left}; "
            f"only in {right_name}: {only_right}"
        )


def load_scoring_inputs(
    candidate_manifest: str,
    reference_manifest: str,
    *,
    prompts_path: str | None = None,
) -> tuple[MetricInputs, str, str]:
    """Load candidate/reference manifests and their texts into :class:`MetricInputs`.

    Each manifest is a TextSet JSON pointing at a ``{pair_id, text}`` JSONL (see
    data-model.md, TextSet). Prompts (needed only by JMQ) come from
    ``prompts_path`` (a ``{pair_id, prompt}`` or corpus ``{pair_id, prompt, ...}``
    JSONL); when omitted, JMQ cannot run.

    Both manifests' ``pair_ids`` are read and cross-validated here (FR-009):
    the candidate ids, the reference ids, and each side's texts-file keys must
    all name the same set. A reference whose ids diverge from the candidate, or a
    texts file that is a superset (or subset) of its manifest, is rejected with
    :class:`~dehip.validate.InputSetValidationError` (exit 2) *before* any
    embedder or judge is touched -- never surfacing later as a KeyError.

    When a ``prompts_path`` is given (JMQ requested), the prompt ids get the same
    exact-set gate: they must equal the pair ids. A prompts file that is a
    superset (stray prompt id) or subset (missing prompt id) is rejected here at
    exit 2 before any spend, rather than the superset being silently ignored or
    the subset failing late as a KeyError. When no prompts file is passed (a
    metrics subset that omits JMQ), no prompt check runs.

    Returns the inputs plus the two ``set_id`` strings for the report identity.
    """
    from dehip.schemas import TextSet, read_json

    cand_path = Path(candidate_manifest)
    ref_path = Path(reference_manifest)
    cand_set: TextSet = read_json(cand_path, TextSet)
    ref_set: TextSet = read_json(ref_path, TextSet)

    candidate_texts = _read_pair_texts(
        _texts_path_for(cand_path, cand_set.provenance)
    )
    reference_texts = _read_pair_texts(
        _texts_path_for(ref_path, ref_set.provenance)
    )

    # Cross-validate all four id sources before any spend: candidate manifest ==
    # reference manifest, and each manifest == its own texts-file keys. This is
    # the real pairing gate; score() no longer needs a separate one. Checking the
    # texts keys exactly (not just membership) rejects a texts file that is a
    # superset of the manifest, so a subset is never silently scored.
    cand_ids = list(cand_set.pair_ids)
    ref_ids = list(ref_set.pair_ids)
    _assert_ids_match("candidate", cand_ids, "reference", ref_ids)
    _assert_ids_match(
        "candidate manifest", cand_ids, "candidate texts", candidate_texts.keys()
    )
    _assert_ids_match(
        "reference manifest", ref_ids, "reference texts", reference_texts.keys()
    )

    prompts: dict[str, str] | None = None
    if prompts_path is not None:
        prompts = {}
        with Path(prompts_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompts[record["pair_id"]] = record["prompt"]
        # Prompts are a JMQ input the judge spend depends on, so they get the same
        # exact-set gate as the texts files: the prompt ids must equal the pair
        # ids, not merely cover them. A superset (an extra prompt id) is caught
        # here rather than silently ignored; a subset (a missing prompt id) is
        # caught here as exit-2 validation rather than late as a KeyError inside
        # _judge_pairs. The check runs only when a prompts file was passed, so a
        # metrics-subset run with no prompts (e.g. --metrics token_l2) is
        # untouched and cannot false-positive.
        _assert_ids_match("pair_ids", cand_ids, "prompts", prompts.keys())

    # The candidate TextSet's homogeneous corpus tag is the run's corpus; carry
    # it so score() can stamp compared["corpus"] and the FR-010 gate has its
    # primary signal on real reports (not only the set-id naming fallback).
    return (
        MetricInputs(
            pair_ids=cand_ids,
            candidate_texts=candidate_texts,
            reference_texts=reference_texts,
            prompts=prompts,
            corpus=cand_set.corpus,
        ),
        cand_set.set_id,
        ref_set.set_id,
    )


# --- Metric composition ------------------------------------------------------


def _ordered_texts(
    pair_ids: Sequence[str], texts: dict[str, str]
) -> list[str]:
    """Texts in pair_id order, raising on a missing id rather than embedding a hole."""
    ordered: list[str] = []
    for pair_id in pair_ids:
        if pair_id not in texts:
            raise KeyError(f"no text for pair_id {pair_id!r}")
        ordered.append(texts[pair_id])
    return ordered


def _compute_mmd(
    inputs: MetricInputs, cache: EmbeddingCache
) -> mmd_mod.MMDResult:
    """Embed both sets through the cache, then unbiased MMD^2 with the median heuristic.

    Embedding both sets in one ``cache.embed`` call per set means a text shared
    between sets (or a warm cache) is embedded at most once; the bandwidth is the
    median heuristic over the pooled sample and is returned so the report records
    it (R5).
    """
    cand = cache.embed(_ordered_texts(inputs.pair_ids, inputs.candidate_texts))
    ref = cache.embed(_ordered_texts(inputs.pair_ids, inputs.reference_texts))
    return mmd_mod.mmd2_unbiased(cand, ref)


def _compute_token_l2(
    inputs: MetricInputs, tokenizer: Any
) -> token_l2_mod.TokenL2Result:
    cand = _ordered_texts(inputs.pair_ids, inputs.candidate_texts)
    ref = _ordered_texts(inputs.pair_ids, inputs.reference_texts)
    return token_l2_mod.token_l2(cand, ref, tokenizer=tokenizer)


def _judge_pairs(inputs: MetricInputs) -> list[JudgePair]:
    """Build the JMQ work list: one JudgePair per shared pair_id."""
    if inputs.prompts is None:
        raise ValueError(
            "JMQ requires prompts for each pair; pass MetricInputs.prompts when "
            "the 'jmq' metric is requested"
        )
    pairs: list[JudgePair] = []
    for pair_id in inputs.pair_ids:
        if pair_id not in inputs.prompts:
            raise KeyError(f"no prompt for pair_id {pair_id!r}")
        pairs.append(
            JudgePair(
                pair_id=pair_id,
                prompt=inputs.prompts[pair_id],
                model_text=inputs.candidate_texts[pair_id],
                human_text=inputs.reference_texts[pair_id],
            )
        )
    return pairs


# --- Bias audit (A/B order distribution) ------------------------------------


def order_distribution(verdicts: Sequence[JudgeVerdict]) -> dict[str, Any]:
    """A/B assignment stats over ``verdicts`` so positional bias is auditable.

    Counts how often the model output was placed as candidate A (``model_first``)
    vs B (``human_first``). Because the A/B order is a per-pair property frozen
    from the seed (see :func:`~dehip.metrics.jmq.assign_order`), this distribution
    reproduces exactly from the seed and the pair_ids, independent of judge
    behavior. A ~50/50 split over many pairs is the expected shape (Story 1
    scenario 4); a skew is surfaced here rather than hidden inside the win rate.

    The fraction is computed over *distinct pairs* (each pair's order is shared
    across all six dimensions), not over raw verdict rows, so six dimensions do
    not inflate the count.
    """
    order_by_pair: dict[str, str] = {}
    for verdict in verdicts:
        order_by_pair.setdefault(verdict.pair_id, verdict.order)
    model_first = sum(1 for o in order_by_pair.values() if o == "model_first")
    human_first = sum(1 for o in order_by_pair.values() if o == "human_first")
    total = model_first + human_first
    return {
        "pairs": total,
        "model_first": model_first,
        "human_first": human_first,
        "model_first_fraction": (model_first / total) if total else None,
    }


# --- Caveats -----------------------------------------------------------------


def build_caveats(
    *,
    n: int,
    judge_model: str,
    ran_jmq: bool,
    ran_mmd: bool,
) -> list[dict[str, Any]]:
    """Auto-attach the report's caveats per data-model.md.

    Three conditions, each a self-describing dict so the JSON and markdown carry
    the same explanation:

    - **small-N** when ``n`` is below :data:`SMALL_N_FLOOR`: the metrics are
      high-variance at this sample size and should not be over-read.
    - **bandwidth comparability** whenever MMD ran: the DFT post never published
      its kernel bandwidth, so absolute MMD comparability to the benchmark rows
      is best-effort even with the identical embedder (R5).
    - **non-default judge** when ``judge_model`` differs from the exact-protocol
      default: JMQ absolute values are only comparable within one judge, so a
      swapped judge breaks comparability to the benchmark JMQ column.
    """
    caveats: list[dict[str, Any]] = []
    if n < SMALL_N_FLOOR:
        caveats.append(
            {
                "kind": "small_n",
                "n": n,
                "floor": SMALL_N_FLOOR,
                "message": (
                    f"small sample: N={n} is below the documented floor of "
                    f"{SMALL_N_FLOOR}; MMD, token-L2, and JMQ are all "
                    "high-variance at this size, so treat absolute values as "
                    "indicative, not conclusive."
                ),
            }
        )
    if ran_mmd:
        caveats.append(
            {
                "kind": "mmd_bandwidth_comparability",
                "message": (
                    "the reference benchmark never published its MMD kernel "
                    "bandwidth, so absolute MMD comparability to the benchmark "
                    "rows is best-effort even with the identical embedder; the "
                    "bandwidth used here is recorded in config.mmd_bandwidth."
                ),
            }
        )
    if ran_jmq and judge_model != DEFAULT_JUDGE_MODEL:
        caveats.append(
            {
                "kind": "non_default_judge",
                "judge_model": judge_model,
                "default_judge": DEFAULT_JUDGE_MODEL,
                "message": (
                    f"JMQ ran with judge {judge_model!r}, not the exact-protocol "
                    f"default {DEFAULT_JUDGE_MODEL!r}; JMQ absolute values are only "
                    "comparable within one judge, so comparability to the "
                    "benchmark JMQ column is broken."
                ),
            }
        )
    return caveats


# --- Assembly ----------------------------------------------------------------


def assemble_report(
    *,
    report_id: str,
    candidate_set: str,
    reference_set: str,
    n: int,
    seed: int,
    judge_model: str,
    embedder_id: str,
    tokenizer_id: str | None,
    mmd_result: mmd_mod.MMDResult | None,
    token_l2_result: token_l2_mod.TokenL2Result | None,
    jmq_scores: dict[str, Any] | None,
    verdicts: Sequence[JudgeVerdict] | None,
    thresholds: dict[str, Any] | None = None,
    started: str | None = None,
    finished: str | None = None,
    corpus: str | None = None,
) -> MetricReport:
    """Compose computed metric values into one :class:`MetricReport`.

    Only the metrics actually computed contribute a value; a metric not run is
    left as its schema default (``mmd``/``token_l2`` at ``float('nan')`` so the
    absence is not misread as a real zero, ``jmq`` an empty dict). The ``config``
    block records the seed, all three instrument identities, and the MMD
    bandwidth actually used (``None`` when MMD did not run). Caveats are attached
    by :func:`build_caveats`; the JMQ block carries a ``bias_audit`` sub-object
    from :func:`order_distribution` when verdicts are present.

    Args:
        report_id: Stable id for the report.
        candidate_set / reference_set: Set ids being compared (recorded in
            ``compared`` with their shared size ``n``).
        n: Number of paired texts scored (stated per the edge-case N rule).
        seed: Global seed, recorded in config for reproducibility (FR-004).
        judge_model / embedder_id / tokenizer_id: Instrument identities.
        mmd_result: MMD result, or None if MMD did not run.
        token_l2_result: Token-L2 result, or None if it did not run.
        jmq_scores: Per-dimension JMQ aggregate, or None if JMQ did not run.
        verdicts: The JMQ verdicts, used only for the bias audit; may be None.
        thresholds: Optional threshold record for config (e.g. cost preflight).
        started / finished: ISO timestamps; defaulted to now when omitted.
        corpus: The corpus tag of the scored sets (e.g. ``"fineweb"`` /
            ``"personal"``), recorded in ``compared["corpus"]`` when given so a
            downstream comparison can honor FR-010 (exclude the benchmark from a
            personal-corpus run). Omitted (``None``) leaves the key absent, which
            existing callers do; the comparison falls back to the set-id naming.
    """
    ran_mmd = mmd_result is not None
    ran_token_l2 = token_l2_result is not None
    ran_jmq = jmq_scores is not None

    config: dict[str, Any] = {
        "seed": seed,
        "judge_model": judge_model,
        "embedder_id": embedder_id,
        "tokenizer_id": tokenizer_id,
        "mmd_bandwidth": mmd_result.bandwidth if ran_mmd else None,
        "thresholds": thresholds or {},
    }

    jmq_block: dict[str, Any] = dict(jmq_scores) if ran_jmq else {}
    if ran_jmq and verdicts is not None:
        jmq_block["bias_audit"] = order_distribution(verdicts)

    compared: dict[str, Any] = {
        "candidate_set": candidate_set,
        "reference_set": reference_set,
        "n": n,
    }
    if corpus is not None:
        compared["corpus"] = corpus

    return MetricReport(
        report_id=report_id,
        compared=compared,
        config=config,
        # NaN, not 0.0, for an un-run metric: a real zero (identical sets) must
        # never be confused with "did not run".
        mmd=mmd_result.mmd2 if ran_mmd else float("nan"),
        token_l2=token_l2_result.distance if ran_token_l2 else float("nan"),
        jmq=jmq_block,
        timestamps={
            "started": started or _now_iso(),
            "finished": finished or _now_iso(),
        },
        caveats=build_caveats(
            n=n, judge_model=judge_model, ran_jmq=ran_jmq, ran_mmd=ran_mmd
        ),
    )


# --- Recompute JMQ from persisted verdicts (FR-008) --------------------------


def recompute_jmq(
    verdicts_path: str,
) -> tuple[dict[str, Any], list[JudgeVerdict]]:
    """Re-aggregate JMQ from a persisted verdicts JSONL with zero judge calls.

    Reads only the file (FR-008): no judge client is constructed and no network
    call is made. Returns the per-dimension aggregate (identical to what the
    original run produced, since aggregation groups by pair and dimension and is
    order-independent) plus the loaded verdicts so the caller can attach the bias
    audit.
    """
    from dehip.schemas import read_jsonl

    verdicts = read_jsonl(verdicts_path, JudgeVerdict)
    scores = aggregate_verdicts(verdicts)
    return scores, verdicts


# --- Score orchestration (FR-001, FR-009) ------------------------------------


def _parse_metrics(metrics: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize the metric-selection argument to a validated ordered tuple."""
    if metrics is None:
        return ALL_METRICS
    if isinstance(metrics, str):
        requested = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        requested = [m.strip() for m in metrics if m.strip()]
    unknown = [m for m in requested if m not in ALL_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; valid metrics are {list(ALL_METRICS)}"
        )
    # Preserve the canonical order regardless of request order.
    return tuple(m for m in ALL_METRICS if m in requested)


def score(
    inputs: MetricInputs,
    *,
    report_id: str,
    candidate_set: str,
    reference_set: str,
    embed_cache: EmbeddingCache | None = None,
    tokenizer: Any = None,
    judge_client: Any = None,
    verdicts_path: str | None = None,
    metrics: str | Sequence[str] | None = None,
    seed: int = 0,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedder_id: str = "nvidia/llama-embed-nemotron-8b",
    thresholds: dict[str, Any] | None = None,
    min_n: int = DEFAULT_MIN_N,
    corpus: str | None = None,
) -> MetricReport:
    """Score a candidate set vs a reference set into one MetricReport (FR-001).

    Validation runs first, before any scoring spend (FR-009): the pairing, count,
    length, and minimum-N gates on ``inputs``. Only after they pass is any metric
    computed. Each requested metric runs through its injectable seam:

    - **mmd**: embeds both sets via ``embed_cache`` and computes unbiased MMD^2.
    - **token_l2**: 1-gram token-frequency L2 via ``tokenizer``.
    - **jmq**: one judge call per (pair, dimension) via ``judge_client``, each
      verdict persisted to ``verdicts_path`` before aggregation.

    The three seams are what keep tests off real models and the network: pass a
    stub embed cache, a stub tokenizer, and a mock judge client.

    Args:
        inputs: The paired texts (and prompts, for JMQ).
        report_id / candidate_set / reference_set: Report identity.
        embed_cache: EmbeddingCache wrapping the embedder; required for ``mmd``.
        tokenizer: Tokenizer seam for ``token_l2`` (None builds the real Qwen3
            tokenizer, which tests never do).
        judge_client: JudgeClient for ``jmq``; required for ``jmq``.
        verdicts_path: Where JMQ persists verdicts before aggregating; required
            for ``jmq``.
        metrics: Which metrics to run (subset of :data:`ALL_METRICS`); default all.
        seed / judge_model / embedder_id: Recorded in config; ``seed`` also drives
            the JMQ A/B order.
        thresholds: Optional threshold record for config.
        min_n: Hard minimum-N gate passed to validation.
        corpus: Explicit corpus tag override for ``compared["corpus"]``; when
            ``None`` (the default) the corpus is taken from ``inputs.corpus``
            (the candidate TextSet's tag, wired by :func:`load_scoring_inputs`),
            so a real score run carries its corpus for the FR-010 gate.

    Raises:
        InputSetValidationError: on any failed input gate (before spend).
        ValueError: on unknown metrics or a missing seam for a requested metric.
    """
    selected = _parse_metrics(metrics)

    # Validation before any scoring spend (FR-009). The candidate and reference
    # text maps are validated against the shared pair_ids independently: their
    # key sets must each equal pair_ids (so neither a missing nor an extra id
    # slips through) and every referenced text must be non-empty. Passing the
    # candidate keys as candidate and the reference keys as reference -- not the
    # same list twice -- means a mismatched side is actually caught here, not
    # later as a KeyError while embedding or judging.
    pair_ids = list(inputs.pair_ids)
    validate_input_sets(
        list(inputs.candidate_texts.keys()),
        list(inputs.reference_texts.keys()),
        texts=None,
        min_n=min_n,
    )
    _assert_ids_match(
        "pair_ids", pair_ids, "candidate texts", inputs.candidate_texts.keys()
    )
    for side_name, side in (
        ("candidate", inputs.candidate_texts),
        ("reference", inputs.reference_texts),
    ):
        for pair_id in pair_ids:
            text = side.get(pair_id)
            if text is None or not text.strip():
                from dehip.validate import InputSetValidationError

                raise InputSetValidationError(
                    f"{side_name} text for pair_id {pair_id!r} is missing or empty"
                )

    n = len(pair_ids)
    started = _now_iso()

    mmd_result: mmd_mod.MMDResult | None = None
    token_l2_result: token_l2_mod.TokenL2Result | None = None
    jmq_scores: dict[str, Any] | None = None
    verdicts: list[JudgeVerdict] | None = None
    tokenizer_id: str | None = None

    if "mmd" in selected:
        if embed_cache is None:
            raise ValueError("the 'mmd' metric requires an embed_cache seam")
        mmd_result = _compute_mmd(inputs, embed_cache)

    if "token_l2" in selected:
        token_l2_result = _compute_token_l2(inputs, tokenizer)
        tokenizer_id = token_l2_result.tokenizer_id

    if "jmq" in selected:
        if judge_client is None:
            raise ValueError("the 'jmq' metric requires a judge_client seam")
        if verdicts_path is None:
            raise ValueError(
                "the 'jmq' metric requires a verdicts_path to persist verdicts "
                "before aggregation (FR-008)"
            )
        pairs = _judge_pairs(inputs)
        verdicts = jmq_mod.run_judging(
            pairs,
            verdicts_path,
            client=judge_client,
            seed=seed,
            model=judge_model,
        )
        jmq_scores = aggregate_verdicts(verdicts)

    return assemble_report(
        report_id=report_id,
        candidate_set=candidate_set,
        reference_set=reference_set,
        n=n,
        seed=seed,
        judge_model=judge_model,
        embedder_id=embedder_id,
        tokenizer_id=tokenizer_id,
        mmd_result=mmd_result,
        token_l2_result=token_l2_result,
        jmq_scores=jmq_scores,
        verdicts=verdicts,
        thresholds=thresholds,
        started=started,
        corpus=corpus if corpus is not None else inputs.corpus,
    )


# --- JSON serialization boundary ---------------------------------------------


def _sanitize_nan(value: Any) -> Any:
    """Recursively replace ``NaN`` floats with ``None`` for strict-JSON output.

    Bare ``NaN`` is invalid per ECMA-404: ``JSON.parse`` and Go's ``encoding/json``
    reject it, and ``json.loads(..., parse_constant=...)`` or a strict parser will
    trip on it. The in-memory :class:`~dehip.schemas.MetricReport` keeps ``NaN``
    for an un-run ``mmd``/``token_l2`` (so ``np.isnan`` stays meaningful and an
    un-run metric is never confused with a real ``0.0``); this converts that
    sentinel to JSON ``null`` only at the serialization boundary, leaving ``0.0``
    untouched so the two remain distinguishable on disk.
    """
    if isinstance(value, float):
        return None if value != value else value  # value != value is True for NaN
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nan(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_nan(v) for v in value]
    return value


def report_to_jsonable(report: MetricReport) -> dict[str, Any]:
    """Turn a :class:`MetricReport` into a strict-JSON-safe dict.

    ``asdict`` then :func:`_sanitize_nan`, so an un-run metric's ``NaN`` becomes
    ``null`` while a real ``0.0`` stays ``0.0``. Use this at every write boundary
    (file and stdout) instead of a bare ``asdict`` so nothing emits bare ``NaN``.
    """
    from dataclasses import asdict

    return _sanitize_nan(asdict(report))


# --- Rendering ---------------------------------------------------------------


def _fmt_number(value: float) -> str:
    """Render a metric number; an un-run (NaN) metric shows as a plain marker."""
    if value != value:  # NaN
        return "not run"
    return f"{value:.6g}"


def render_markdown(report: MetricReport) -> str:
    """Render a MetricReport as a readable markdown document (JSON's companion).

    Sections: a header with the report id and compared sets, a config table, a
    metric-values table, the per-dimension JMQ table with its bias audit, and the
    caveats as a bulleted list. Deterministic given the report.
    """
    lines: list[str] = []
    compared = report.compared
    lines.append(f"# MetricReport {report.report_id}")
    lines.append("")
    lines.append(
        f"Candidate `{compared.get('candidate_set')}` vs reference "
        f"`{compared.get('reference_set')}` (N={compared.get('n')})."
    )
    lines.append("")

    cfg = report.config
    lines.append("## Config")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key in (
        "seed",
        "judge_model",
        "embedder_id",
        "tokenizer_id",
        "mmd_bandwidth",
    ):
        lines.append(f"| {key} | {cfg.get(key)} |")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| MMD^2 (unbiased) | {_fmt_number(report.mmd)} |")
    lines.append(f"| token-L2 (1-gram) | {_fmt_number(report.token_l2)} |")
    overall = report.jmq.get("overall") if report.jmq else None
    overall_score = overall.get("score") if isinstance(overall, dict) else None
    lines.append(
        f"| JMQ overall | {overall_score if overall_score is not None else 'not run'} |"
    )
    lines.append("")

    if report.jmq:
        lines.append("## JMQ per dimension")
        lines.append("")
        lines.append("| Dimension | Score | Wins | Losses | Invalid | N |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for dimension in DIMENSION_ORDER:
            row = report.jmq.get(dimension)
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {dimension} | {row.get('score')} | {row.get('wins')} | "
                f"{row.get('losses')} | {row.get('invalid')} | {row.get('n')} |"
            )
        lines.append("")

        bias = report.jmq.get("bias_audit")
        if isinstance(bias, dict):
            lines.append("## Bias audit (A/B order)")
            lines.append("")
            frac = bias.get("model_first_fraction")
            frac_str = f"{frac:.4f}" if isinstance(frac, float) else str(frac)
            lines.append(
                f"model-first {bias.get('model_first')} / human-first "
                f"{bias.get('human_first')} over {bias.get('pairs')} pairs "
                f"(model-first fraction {frac_str})."
            )
            lines.append("")

    lines.append("## Caveats")
    lines.append("")
    if not report.caveats:
        lines.append("None.")
    else:
        for caveat in report.caveats:
            if isinstance(caveat, dict):
                lines.append(f"- **{caveat.get('kind')}**: {caveat.get('message')}")
            else:
                lines.append(f"- {caveat}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Comparison assembly (FR-007, FR-010, Story 3) -- ADDITIVE.
#
# Everything below composes ALREADY-SCORED MetricReports (draft vs one-or-more
# rewrites) into one comparison artifact: per-metric deltas, a k-round
# trajectory when several rewrite reports are given, and (opt-in) the pinned
# published benchmark rows. It never re-scores and never touches a model or the
# network; it reads MetricReport dataclasses via the existing schemas. The score
# path above (score / assemble_report / recompute_jmq / render_markdown) is
# untouched.
# =============================================================================

# The corpus tag that must never be placed beside the published benchmark rows
# (FR-010): the benchmark was measured on a public FineWeb subset, so attaching
# it to a run scored on James's private posts would invite a false comparison
# across incomparable corpora.
PERSONAL_CORPUS = "personal"

# The three metrics compared, each with its "closer to human is..." direction.
# MMD and token-L2 measure distance from the human distribution (lower = closer
# to human); JMQ is 2*win-rate vs human (higher = closer to parity with human).
# "improved" in a delta means "moved closer to human", disambiguating the raw
# sign so a reader never has to remember which way each metric points.
COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("mmd", "lower_is_better"),
    ("token_l2", "lower_is_better"),
    ("jmq", "higher_is_better"),
)


class PersonalCorpusBenchmarkError(ValueError):
    """Raised when ``--benchmark`` is asked for over a personal-corpus report.

    A hard, loud refusal (mapped to exit 2 by the CLI), per FR-010: the pinned
    benchmark rows come from a public corpus, so they must never be attached to
    a comparison whose inputs were scored on the personal corpus. The alternative
    -- silently dropping the benchmark rows, or silently scoring the personal set
    into a benchmark comparison -- is exactly the false-comparison failure FR-010
    exists to prevent, so this refuses rather than degrades.
    """


def report_corpus(report: MetricReport) -> str | None:
    """Best-effort corpus tag for a MetricReport, for the FR-010 gate.

    Primary source: ``compared["corpus"]`` when the scoring run recorded it.
    Fallback (defense-in-depth): the set-id naming convention -- a personal
    corpus is built with ``set_id`` prefixed ``personal-`` (see
    :func:`dehip.corpus.build_personal_corpus`), so a report whose candidate or
    reference set id starts with ``personal-`` is treated as personal even if the
    explicit corpus tag was never stamped. Returning :data:`PERSONAL_CORPUS` for
    either signal means the benchmark refusal cannot be dodged by an
    older report that predates the explicit tag.
    """
    compared = report.compared or {}
    corpus = compared.get("corpus")
    if isinstance(corpus, str) and corpus:
        return corpus
    for key in ("candidate_set", "reference_set"):
        value = compared.get(key)
        if isinstance(value, str) and value.startswith(f"{PERSONAL_CORPUS}-"):
            return PERSONAL_CORPUS
    return None


def _is_personal(report: MetricReport) -> bool:
    return report_corpus(report) == PERSONAL_CORPUS


def _jmq_overall_score(report: MetricReport) -> float | None:
    """The single JMQ number a comparison uses: ``jmq.overall.score``.

    ``None`` when JMQ did not run (empty jmq block) or when the overall
    dimension had no valid verdicts (aggregation yields ``score=None``). Kept
    distinct from a real ``0.0`` so a null is never folded into a 0 delta.
    """
    jmq = report.jmq or {}
    overall = jmq.get("overall")
    if isinstance(overall, dict):
        score = overall.get("score")
        if isinstance(score, (int, float)) and not _is_nan(score):
            return float(score)
    return None


def _is_nan(value: Any) -> bool:
    """True only for a float NaN (an un-run mmd/token_l2 sentinel)."""
    return isinstance(value, float) and value != value


def _metric_value(report: MetricReport, metric: str) -> float | None:
    """Pull one comparison metric from a report, or ``None`` if un-run/absent.

    ``mmd``/``token_l2`` carry ``NaN`` when un-run (mapped to ``None`` here so a
    downstream delta is marked not-comparable, never a 0). ``jmq`` resolves to
    the overall score.
    """
    if metric == "jmq":
        return _jmq_overall_score(report)
    value = getattr(report, metric)
    if value is None or _is_nan(value):
        return None
    return float(value)


def metric_delta(
    metric: str,
    direction: str,
    draft_value: float | None,
    rewrite_value: float | None,
) -> dict[str, Any]:
    """Per-metric delta = rewrite - draft, with an unambiguous direction read.

    The raw ``delta`` is ``rewrite - draft`` (signed). ``status`` interprets that
    against the metric's direction as a TRI-STATE so a metric that did not move is
    never misreported as a regression:

    - ``"improved"`` -- moved closer to human (a negative delta for a
      lower-is-better metric, a positive delta for a higher-is-better one).
    - ``"regressed"`` -- moved farther from human.
    - ``"unchanged"`` -- an EXACTLY zero delta: neither closer nor farther.

    ``improved`` keeps its original boolean meaning for a NON-zero delta (``True``
    when the status is ``"improved"``, ``False`` when ``"regressed"``) so existing
    callers reading the boolean stay valid; for an unchanged (zero-delta) metric
    ``improved`` is ``None`` (a no-change is not a regression), and the ``status``
    field is the reliable read. Render via :func:`_fmt_delta`.

    Null handling (CRITICAL): if EITHER value is ``None`` (metric un-run in one
    of the two reports), the delta is ``None``, ``comparable`` is ``False``,
    ``status`` is ``None`` and ``improved`` is ``None`` -- a null NEVER becomes a
    silent 0 delta.
    """
    if draft_value is None or rewrite_value is None:
        return {
            "metric": metric,
            "direction": direction,
            "draft": draft_value,
            "rewrite": rewrite_value,
            "delta": None,
            "improved": None,
            "status": None,
            "comparable": False,
        }
    delta = rewrite_value - draft_value
    if direction == "lower_is_better":
        moved_closer = delta < 0
    elif direction == "higher_is_better":
        moved_closer = delta > 0
    else:  # pragma: no cover - COMPARISON_METRICS only uses the two above
        raise ValueError(f"unknown metric direction {direction!r}")
    # Tri-state: an exactly-zero delta is "unchanged", never a regression. Only a
    # genuinely non-zero delta gets the improved/regressed boolean.
    if delta == 0:
        status = "unchanged"
        improved: bool | None = None
    elif moved_closer:
        status = "improved"
        improved = True
    else:
        status = "regressed"
        improved = False
    return {
        "metric": metric,
        "direction": direction,
        "draft": draft_value,
        "rewrite": rewrite_value,
        "delta": delta,
        "improved": improved,
        "status": status,
        "comparable": True,
    }


def _report_round(report: MetricReport) -> int | None:
    """The k-round a rewrite report scored, if recorded.

    Prefers an explicit ``compared["round"]`` (always emitted by the rewrite
    stage, so it is the reliable primary); falls back only to ``config["round"]``.
    A ``config["k"]`` fallback is deliberately NOT consulted: ``k`` is overloaded
    (the rewrite state machine uses ``k`` as the current-round index), so reading
    it here could silently misorder the trajectory. ``None`` when no round is
    recorded (a single rewrite with no trajectory needs none).
    """
    for source, key in (
        (report.compared, "round"),
        (report.config, "round"),
    ):
        if isinstance(source, dict):
            value = source.get(key)
            if isinstance(value, int):
                return value
    return None


def _deltas_for(
    draft: MetricReport, rewrite: MetricReport
) -> dict[str, dict[str, Any]]:
    """All per-metric deltas between one draft and one rewrite report."""
    return {
        metric: metric_delta(
            metric,
            direction,
            _metric_value(draft, metric),
            _metric_value(rewrite, metric),
        )
        for metric, direction in COMPARISON_METRICS
    }


def _trajectory_row(rewrite: MetricReport, index: int) -> dict[str, Any]:
    """One k-trajectory row: the round (or its input index) plus each metric."""
    round_no = _report_round(rewrite)
    return {
        "round": round_no,
        "input_index": index,
        "report_id": rewrite.report_id,
        "mmd": _metric_value(rewrite, "mmd"),
        "token_l2": _metric_value(rewrite, "token_l2"),
        "jmq": _metric_value(rewrite, "jmq"),
    }


def assemble_comparison(
    *,
    draft: MetricReport,
    rewrites: Sequence[MetricReport],
    comparison_id: str,
    attach_benchmark: bool = False,
) -> dict[str, Any]:
    """Assemble the comparison artifact (FR-007, FR-010) from scored reports.

    Places the draft scores, each rewrite's scores, and per-metric deltas side by
    side. When more than one rewrite report is given, adds a k-trajectory table
    ORDERED BY ROUND (ascending) even when the inputs arrive in mixed round order
    -- a rewrite with no recorded round sorts after all rounded rewrites, by its
    input index, so a mixed batch is never scrambled. When ``attach_benchmark``
    is set, the pinned published rows are attached WITH their external-protocol
    caveat.

    FR-010 refusal (HARD): if ``attach_benchmark`` is set, this raises
    :class:`PersonalCorpusBenchmarkError` when ANY input report (draft or any
    rewrite) is on the personal corpus, AND ALSO when any input's corpus cannot be
    CONFIRMED non-personal -- i.e. its corpus tag is absent and its set-id carries
    no positive signal (:func:`report_corpus` returns ``None``). Under a benchmark
    run the safe default is refuse-on-uncertainty: a report of unknown provenance
    might be personal, and placing an unverified corpus beside the public
    benchmark rows is exactly the false comparison FR-010 exists to prevent. It
    never silently drops the benchmark rows and never silently scores an
    unconfirmed set into a benchmark comparison. Without ``attach_benchmark``, a
    personal-corpus OR unknown-corpus comparison assembles normally (deltas +
    trajectory, no benchmark rows).

    Returns a plain, strict-JSON-safe dict (no ``NaN``: un-run metrics are already
    ``None`` here). Rendering is :func:`render_comparison`.
    """
    if not rewrites:
        raise ValueError("assemble_comparison requires at least one rewrite report")

    all_reports = [draft, *rewrites]

    if attach_benchmark:
        personal = [r for r in all_reports if _is_personal(r)]
        if personal:
            ids = ", ".join(sorted(r.report_id for r in personal))
            raise PersonalCorpusBenchmarkError(
                "refusing --benchmark: the published benchmark rows are measured "
                "on a public corpus, but these input report(s) were scored on the "
                f"personal corpus and must not be placed beside them (FR-010): {ids}. "
                "Re-run without --benchmark to assemble the non-benchmark comparison."
            )
        # Refuse-on-uncertainty: an input whose corpus cannot be CONFIRMED
        # non-personal (no explicit tag and no positive set-id signal) might be
        # personal, so it must not be attached to the public benchmark either.
        unconfirmed = [r for r in all_reports if report_corpus(r) is None]
        if unconfirmed:
            ids = ", ".join(sorted(r.report_id for r in unconfirmed))
            raise PersonalCorpusBenchmarkError(
                "refusing --benchmark: the published benchmark rows are measured "
                "on a public corpus, but these input report(s) carry no corpus tag "
                "and cannot be confirmed non-personal, so they must not be placed "
                f"beside the benchmark (FR-010): {ids}. Re-run without --benchmark, "
                "or re-score so the report records its corpus."
            )

    draft_scores = {
        "report_id": draft.report_id,
        "corpus": report_corpus(draft),
        "mmd": _metric_value(draft, "mmd"),
        "token_l2": _metric_value(draft, "token_l2"),
        "jmq": _metric_value(draft, "jmq"),
    }

    rewrite_blocks: list[dict[str, Any]] = []
    for rewrite in rewrites:
        rewrite_blocks.append(
            {
                "report_id": rewrite.report_id,
                "corpus": report_corpus(rewrite),
                "round": _report_round(rewrite),
                "mmd": _metric_value(rewrite, "mmd"),
                "token_l2": _metric_value(rewrite, "token_l2"),
                "jmq": _metric_value(rewrite, "jmq"),
                "deltas": _deltas_for(draft, rewrite),
            }
        )

    comparison: dict[str, Any] = {
        "comparison_id": comparison_id,
        "metric_directions": {m: d for m, d in COMPARISON_METRICS},
        "draft": draft_scores,
        "rewrites": rewrite_blocks,
    }

    # k-trajectory only when several rewrite reports are scored. Ordered by
    # (round, input_index): rounded rewrites ascend by round; an unrounded
    # rewrite sorts last by its arrival index, so a mixed-order batch (k3, k1,
    # k2) renders k1, k2, k3.
    if len(rewrites) > 1:
        rows = [_trajectory_row(rw, i) for i, rw in enumerate(rewrites)]
        rows.sort(
            key=lambda row: (
                row["round"] is None,
                row["round"] if row["round"] is not None else 0,
                row["input_index"],
            )
        )
        comparison["trajectory"] = rows

    if attach_benchmark:
        from dehip.benchmark import benchmark_table_to_jsonable

        comparison["benchmark"] = benchmark_table_to_jsonable()

    return comparison


# --- Comparison rendering ----------------------------------------------------


def _fmt_opt(value: float | None) -> str:
    """Render an optional metric number; ``None`` (un-run) shows as a marker."""
    if value is None:
        return "not run"
    return f"{value:.6g}"


def _fmt_delta(cell: dict[str, Any]) -> str:
    """Render one delta cell: signed delta + a closer/farther/no-change note.

    Tri-state on ``status`` so an exactly-zero delta reads as "no change" rather
    than being folded into "farther from human": ``improved`` -> "closer to
    human", ``regressed`` -> "farther from human", ``unchanged`` -> "no change".
    """
    if not cell.get("comparable"):
        return "n/a (metric un-run in one report)"
    delta = cell["delta"]
    status = cell.get("status")
    if status == "unchanged":
        note = "no change"
    elif status == "improved":
        note = "closer to human"
    else:  # "regressed"
        note = "farther from human"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6g} ({note})"


def render_comparison(comparison: dict[str, Any]) -> str:
    """Render a comparison dict as readable markdown (JSON's companion).

    Sections: header, a side-by-side draft/rewrite/delta table per rewrite, the
    k-trajectory table (when present), and the benchmark rows (when attached)
    ALWAYS preceded by the external-protocol caveat. Deterministic given the
    comparison dict.
    """
    lines: list[str] = []
    lines.append(f"# Comparison {comparison.get('comparison_id')}")
    lines.append("")

    draft = comparison.get("draft", {})
    lines.append(
        f"Draft `{draft.get('report_id')}` (corpus `{draft.get('corpus')}`) vs "
        f"{len(comparison.get('rewrites', []))} rewrite report(s)."
    )
    lines.append("")

    # Per-rewrite side-by-side: draft, rewrite, delta for each metric.
    for block in comparison.get("rewrites", []):
        header = f"## Rewrite `{block.get('report_id')}`"
        if block.get("round") is not None:
            header += f" (round {block['round']})"
        lines.append(header)
        lines.append("")
        lines.append("| Metric | Draft | Rewrite | Delta |")
        lines.append("| --- | --- | --- | --- |")
        deltas = block.get("deltas", {})
        for metric, _direction in COMPARISON_METRICS:
            cell = deltas.get(metric, {})
            lines.append(
                f"| {metric} | {_fmt_opt(cell.get('draft'))} | "
                f"{_fmt_opt(cell.get('rewrite'))} | {_fmt_delta(cell)} |"
            )
        lines.append("")

    trajectory = comparison.get("trajectory")
    if trajectory:
        lines.append("## k-round trajectory")
        lines.append("")
        lines.append("| Round | Report | MMD | token-L2 | JMQ overall |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in trajectory:
            round_label = row.get("round")
            round_str = "k?" if round_label is None else f"k{round_label}"
            lines.append(
                f"| {round_str} | {row.get('report_id')} | "
                f"{_fmt_opt(row.get('mmd'))} | {_fmt_opt(row.get('token_l2'))} | "
                f"{_fmt_opt(row.get('jmq'))} |"
            )
        lines.append("")

    benchmark = comparison.get("benchmark")
    if benchmark:
        # The external-protocol caveat ALWAYS renders before the rows, so a
        # benchmark table is never shown as a clean apples-to-apples comparison
        # (FR-007). The caveat text comes from the attached payload, not a literal
        # here, so JSON and markdown never drift.
        lines.append("## Published benchmark (external protocol)")
        lines.append("")
        lines.append(f"> {benchmark.get('caveat')}")
        lines.append("")
        lines.append(f"Source: {benchmark.get('source')}.")
        lines.append("")
        lines.append("| Model | MMD (lower) | JMQ (higher) | token-L2 (lower) |")
        lines.append("| --- | --- | --- | --- |")
        for row in benchmark.get("rows", []):
            lines.append(
                f"| {row.get('model')} | {row.get('mmd')} | {row.get('jmq')} | "
                f"{row.get('token_l2')} |"
            )
        lines.append("")

    return "\n".join(lines)
