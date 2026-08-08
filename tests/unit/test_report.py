"""Tests for MetricReport assembly and score orchestration (issue #10).

Every test uses stubs: a stub embedder (deterministic vectors, counts its calls),
a stub tokenizer (whitespace split), and a mock judge client (fixed replies,
counts its calls). No real model is loaded and no network call is made. The
suite locks the composition contract the adversarial review will probe:

- score composes all three metrics into one MetricReport in one invocation (FR-001).
- Deterministic metrics (MMD, token-L2) reproduce exactly given the same inputs
  and seed (FR-008).
- --recompute-jmq-from re-aggregates JMQ from persisted verdicts with zero judge
  calls, matching the original aggregation (FR-008).
- The report records every config field and auto-attaches the right caveats.
- The bias audit is present and reproduces from the seed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dehip.metrics.embeddings import EmbeddingCache
from dehip.metrics.jmq import DEFAULT_JUDGE_MODEL, DIMENSION_ORDER
from dehip.report import (
    SMALL_N_FLOOR,
    MetricInputs,
    assemble_report,
    build_caveats,
    order_distribution,
    recompute_jmq,
    render_markdown,
    report_to_jsonable,
    score,
)
from dehip.schemas import JudgeVerdict, MetricReport, read_json, write_json
from dehip.validate import InputSetValidationError

# --- Stubs -------------------------------------------------------------------


class StubEmbedder:
    """Deterministic text -> vector embedder that counts its calls.

    Maps each text to a fixed 4-D vector derived from a hash of its bytes, so
    identical text always embeds to the identical vector (needed for the
    determinism test) and two different texts get different vectors (needed for a
    non-degenerate MMD bandwidth). Never touches the network.
    """

    embedder_id = "stub-embedder"

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, texts):
        self.call_count += 1
        rows = []
        for text in texts:
            seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(4).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)


class StubTokenizer:
    """Whitespace tokenizer with a stable id. No download."""

    tokenizer_id = "stub-tokenizer"

    def tokenize(self, text: str):
        return text.split()


class ScriptedJudge:
    """Judge returning replies from a per-render lookup, counting calls.

    ``model_wins`` picks whichever candidate slot holds the marker; here we just
    always return "A" but count calls so the recompute test can assert zero
    additional calls.
    """

    def __init__(self, reply: str = "A") -> None:
        self._reply = reply
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        return self._reply


# --- Fixtures ----------------------------------------------------------------


def _inputs(n: int, *, with_prompts: bool = True) -> MetricInputs:
    pair_ids = [f"fineweb-{i}" for i in range(n)]
    candidate = {pid: f"model output number {i}" for i, pid in enumerate(pair_ids)}
    reference = {pid: f"human reference number {i}" for i, pid in enumerate(pair_ids)}
    prompts = (
        {pid: f"prompt {i}" for i, pid in enumerate(pair_ids)} if with_prompts else None
    )
    return MetricInputs(
        pair_ids=pair_ids,
        candidate_texts=candidate,
        reference_texts=reference,
        prompts=prompts,
    )


def _cache(tmp_path, embedder=None) -> EmbeddingCache:
    return EmbeddingCache(embedder or StubEmbedder(), cache_dir=tmp_path / "emb")


# --- FR-001: one invocation composes all three metrics -----------------------


def test_score_composes_all_three_metrics(tmp_path):
    inputs = _inputs(4)
    embedder = StubEmbedder()
    judge = ScriptedJudge("A")
    report = score(
        inputs,
        report_id="r1",
        candidate_set="cand",
        reference_set="ref",
        embed_cache=_cache(tmp_path, embedder),
        tokenizer=StubTokenizer(),
        judge_client=judge,
        verdicts_path=str(tmp_path / "verdicts.jsonl"),
        seed=7,
    )
    # All three metrics carry a real value in one report.
    assert not np.isnan(report.mmd)
    assert not np.isnan(report.token_l2)
    assert report.jmq["overall"]["n"] == 4  # 4 pairs judged on 'overall'
    # Judge was called once per pair per dimension.
    assert judge.calls == 4 * len(DIMENSION_ORDER)


def test_metric_subset_runs_only_requested(tmp_path):
    inputs = _inputs(4)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="r",
        tokenizer=StubTokenizer(),
        metrics="token_l2",
        seed=1,
    )
    # token_l2 ran; mmd and jmq did not (mmd is NaN, jmq empty).
    assert not np.isnan(report.token_l2)
    assert np.isnan(report.mmd)
    assert report.jmq == {}
    # The non-run mmd caveat is absent; token-only run attaches no bandwidth note.
    kinds = {c["kind"] for c in report.caveats}
    assert "mmd_bandwidth_comparability" not in kinds


# --- Config records every field ----------------------------------------------


def test_config_records_all_identities(tmp_path):
    inputs = _inputs(4)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=ScriptedJudge(),
        verdicts_path=str(tmp_path / "v.jsonl"),
        seed=42,
        judge_model="gpt-5.4-mini",
        embedder_id="stub-embedder",
    )
    cfg = report.config
    assert cfg["seed"] == 42
    assert cfg["judge_model"] == "gpt-5.4-mini"
    assert cfg["embedder_id"] == "stub-embedder"
    assert cfg["tokenizer_id"] == "stub-tokenizer"
    # The MMD bandwidth actually used is recorded (R5), not hidden.
    assert cfg["mmd_bandwidth"] is not None
    assert cfg["mmd_bandwidth"] > 0


# --- Caveats -----------------------------------------------------------------


def test_small_n_caveat_below_floor():
    caveats = build_caveats(
        n=SMALL_N_FLOOR - 1, judge_model=DEFAULT_JUDGE_MODEL, ran_jmq=True, ran_mmd=True
    )
    kinds = {c["kind"] for c in caveats}
    assert "small_n" in kinds


def test_no_small_n_caveat_at_or_above_floor():
    caveats = build_caveats(
        n=SMALL_N_FLOOR, judge_model=DEFAULT_JUDGE_MODEL, ran_jmq=True, ran_mmd=True
    )
    kinds = {c["kind"] for c in caveats}
    assert "small_n" not in kinds


def test_bandwidth_caveat_when_mmd_ran():
    caveats = build_caveats(
        n=100, judge_model=DEFAULT_JUDGE_MODEL, ran_jmq=False, ran_mmd=True
    )
    kinds = {c["kind"] for c in caveats}
    assert "mmd_bandwidth_comparability" in kinds


def test_non_default_judge_caveat():
    caveats = build_caveats(
        n=100, judge_model="some-other-judge", ran_jmq=True, ran_mmd=False
    )
    kinds = {c["kind"] for c in caveats}
    assert "non_default_judge" in kinds


def test_default_judge_no_cross_judge_caveat():
    caveats = build_caveats(
        n=100, judge_model=DEFAULT_JUDGE_MODEL, ran_jmq=True, ran_mmd=False
    )
    kinds = {c["kind"] for c in caveats}
    assert "non_default_judge" not in kinds


def test_score_attaches_all_three_caveats_when_applicable(tmp_path):
    # Small N + MMD + a non-default judge -> all three caveats.
    inputs = _inputs(3)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=ScriptedJudge(),
        verdicts_path=str(tmp_path / "v.jsonl"),
        judge_model="custom-judge",
    )
    kinds = {c["kind"] for c in report.caveats}
    assert kinds == {"small_n", "mmd_bandwidth_comparability", "non_default_judge"}


# --- Determinism (FR-008) ----------------------------------------------------


def test_deterministic_metrics_reproduce_exactly(tmp_path):
    inputs = _inputs(5)
    kwargs = dict(
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        tokenizer=StubTokenizer(),
        metrics="mmd,token_l2",
        seed=99,
    )
    r1 = score(inputs, embed_cache=_cache(tmp_path / "a"), **kwargs)
    r2 = score(inputs, embed_cache=_cache(tmp_path / "b"), **kwargs)
    # Bit-for-bit identical for the deterministic metrics.
    assert r1.mmd == r2.mmd
    assert r1.token_l2 == r2.token_l2
    assert r1.config["mmd_bandwidth"] == r2.config["mmd_bandwidth"]


# --- Recompute JMQ from persisted verdicts, zero judge calls (FR-008) --------


def test_recompute_jmq_makes_no_judge_calls(tmp_path):
    inputs = _inputs(4)
    verdicts_path = str(tmp_path / "verdicts.jsonl")
    judge = ScriptedJudge("A")

    original = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        judge_client=judge,
        verdicts_path=verdicts_path,
        metrics="jmq",
        seed=3,
    )
    calls_after_run = judge.calls
    assert calls_after_run == 4 * len(DIMENSION_ORDER)

    # Recompute reads only the file: no judge is even passed in.
    recomputed, verdicts = recompute_jmq(verdicts_path)

    # No further judge calls were made (the same judge object is untouched).
    assert judge.calls == calls_after_run

    # And the recomputed aggregation matches the original exactly.
    for dimension in DIMENSION_ORDER:
        assert recomputed[dimension] == original.jmq[dimension]


def test_recompute_matches_direct_aggregation(tmp_path):
    # A recompute over verdicts written by run_judging equals the score()'s jmq.
    inputs = _inputs(6)
    verdicts_path = str(tmp_path / "v.jsonl")
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        judge_client=ScriptedJudge("B"),
        verdicts_path=verdicts_path,
        metrics="jmq",
        seed=11,
    )
    recomputed, _ = recompute_jmq(verdicts_path)
    for dimension in DIMENSION_ORDER:
        assert recomputed[dimension] == report.jmq[dimension]


# --- Bias audit reproduces from the seed -------------------------------------


def test_bias_audit_present_in_report(tmp_path):
    inputs = _inputs(8)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        judge_client=ScriptedJudge(),
        verdicts_path=str(tmp_path / "v.jsonl"),
        metrics="jmq",
        seed=5,
    )
    bias = report.jmq["bias_audit"]
    assert bias["pairs"] == 8
    assert bias["model_first"] + bias["human_first"] == 8
    assert 0.0 <= bias["model_first_fraction"] <= 1.0


def test_bias_audit_reproducible_from_seed(tmp_path):
    inputs = _inputs(10)

    def run(subdir):
        return score(
            inputs,
            report_id="r",
            candidate_set="c",
            reference_set="ref",
            judge_client=ScriptedJudge(),
            verdicts_path=str(tmp_path / subdir / "v.jsonl"),
            metrics="jmq",
            seed=123,
        ).jmq["bias_audit"]

    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    b1 = run("one")
    b2 = run("two")
    # Same seed -> identical A/B distribution, independent of scheduling.
    assert b1 == b2


def test_order_distribution_counts_pairs_not_rows():
    # Six dimensions per pair must not inflate the pair count.
    verdicts = []
    for pair_id, order in (("p0", "model_first"), ("p1", "human_first")):
        for dimension in DIMENSION_ORDER:
            verdicts.append(
                JudgeVerdict(
                    pair_id=pair_id,
                    dimension=dimension,
                    judge_model="j",
                    order=order,
                    raw_response="A",
                    choice="A",
                    retry_count=0,
                    model_won=(order == "model_first"),
                )
            )
    dist = order_distribution(verdicts)
    assert dist["pairs"] == 2
    assert dist["model_first"] == 1
    assert dist["human_first"] == 1
    assert dist["model_first_fraction"] == 0.5


# --- Validation runs before spend (FR-009) -----------------------------------


def test_validation_rejects_below_min_n_before_any_metric(tmp_path):
    inputs = _inputs(1)  # below DEFAULT_MIN_N of 2
    embedder = StubEmbedder()
    judge = ScriptedJudge()
    with pytest.raises(InputSetValidationError):
        score(
            inputs,
            report_id="r",
            candidate_set="c",
            reference_set="ref",
            embed_cache=_cache(tmp_path, embedder),
            tokenizer=StubTokenizer(),
            judge_client=judge,
            verdicts_path=str(tmp_path / "v.jsonl"),
        )
    # No embedding and no judging happened: validation fired first.
    assert embedder.call_count == 0
    assert judge.calls == 0


def test_validation_rejects_empty_text_before_spend(tmp_path):
    inputs = _inputs(3)
    inputs.candidate_texts["fineweb-1"] = "   "  # whitespace-only
    embedder = StubEmbedder()
    with pytest.raises(InputSetValidationError):
        score(
            inputs,
            report_id="r",
            candidate_set="c",
            reference_set="ref",
            embed_cache=_cache(tmp_path, embedder),
            tokenizer=StubTokenizer(),
            metrics="mmd",
        )
    assert embedder.call_count == 0


def test_unknown_metric_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        score(
            _inputs(3),
            report_id="r",
            candidate_set="c",
            reference_set="ref",
            metrics="mmd,bogus",
        )


# --- assemble_report semantics -----------------------------------------------


def test_assemble_marks_unrun_metrics_as_nan():
    report = assemble_report(
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        n=50,
        seed=0,
        judge_model=DEFAULT_JUDGE_MODEL,
        embedder_id="e",
        tokenizer_id=None,
        mmd_result=None,
        token_l2_result=None,
        jmq_scores=None,
        verdicts=None,
    )
    # An un-run metric is NaN, never a real 0.0 that reads as identical sets.
    assert np.isnan(report.mmd)
    assert np.isnan(report.token_l2)
    assert report.jmq == {}
    assert report.config["mmd_bandwidth"] is None


# --- Rendering ---------------------------------------------------------------


def test_render_markdown_includes_config_metrics_and_caveats(tmp_path):
    inputs = _inputs(4)
    report = score(
        inputs,
        report_id="my-report",
        candidate_set="cand",
        reference_set="ref",
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=ScriptedJudge(),
        verdicts_path=str(tmp_path / "v.jsonl"),
        judge_model="custom",  # forces the non-default judge caveat
    )
    md = render_markdown(report)
    assert "my-report" in md
    assert "## Config" in md
    assert "stub-tokenizer" in md
    assert "## Metrics" in md
    assert "## JMQ per dimension" in md
    assert "## Bias audit" in md
    assert "## Caveats" in md
    # The three caveats render as bullets.
    assert "small_n" in md
    assert "mmd_bandwidth_comparability" in md
    assert "non_default_judge" in md


# --- IMPORTANT 1: NaN sanitized to null at the JSON boundary ------------------


def test_report_to_jsonable_maps_nan_to_null_keeps_zero():
    """An un-run metric's NaN becomes null; a real 0.0 stays 0.0 and distinguishable."""
    report = assemble_report(
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        n=50,
        seed=0,
        judge_model=DEFAULT_JUDGE_MODEL,
        embedder_id="e",
        tokenizer_id=None,
        mmd_result=None,  # un-run -> NaN in memory
        token_l2_result=None,  # un-run -> NaN in memory
        jmq_scores=None,
        verdicts=None,
    )
    # In memory the sentinel is still NaN (existing np.isnan tests stay valid).
    assert np.isnan(report.mmd)
    jsonable = report_to_jsonable(report)
    # At the boundary, un-run metrics are null, not NaN.
    assert jsonable["mmd"] is None
    assert jsonable["token_l2"] is None
    # Strict JSON: no bare NaN token survives; parse_constant rejects NaN/Infinity.
    text = json.dumps(jsonable)
    assert "NaN" not in text

    def _reject(_c):
        raise AssertionError("non-strict JSON constant present")

    round_tripped = json.loads(text, parse_constant=_reject)
    assert round_tripped["mmd"] is None


def test_report_to_jsonable_preserves_real_zero():
    """A real 0.0 metric (identical sets) is preserved, never collapsed to null."""

    class _Zero:
        mmd2 = 0.0
        distance = 0.0
        bandwidth = 1.0

    report = assemble_report(
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        n=50,
        seed=0,
        judge_model=DEFAULT_JUDGE_MODEL,
        embedder_id="e",
        tokenizer_id="t",
        mmd_result=_Zero(),
        token_l2_result=_Zero(),
        jmq_scores=None,
        verdicts=None,
    )
    jsonable = report_to_jsonable(report)
    assert jsonable["mmd"] == 0.0
    assert jsonable["mmd"] is not None
    assert jsonable["token_l2"] == 0.0


# --- NIT 2: invalid-verdict path, seed-driven bias, JSON round-trip -----------


def test_score_counts_and_excludes_invalid_verdicts(tmp_path):
    """An invalid judge reply is excluded-and-counted: invalid>0, out of the win rate.

    A judge that returns a non-letter drives every verdict to choice='invalid'.
    That must surface as jmq[dim]['invalid'] > 0 and be excluded from wins/losses
    (n == 0), never scored as a win.
    """
    inputs = _inputs(4)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        judge_client=ScriptedJudge("not-a-letter"),
        verdicts_path=str(tmp_path / "v.jsonl"),
        metrics="jmq",
        seed=2,
    )
    for dimension in DIMENSION_ORDER:
        row = report.jmq[dimension]
        assert row["invalid"] == 4  # all four pairs invalid on every dimension
        assert row["wins"] == 0
        assert row["losses"] == 0
        assert row["n"] == 0  # excluded from the valid comparison count
        assert row["score"] is None  # no valid verdicts -> no win rate scored


def test_render_markdown_invalid_column_nonzero(tmp_path):
    """The rendered JMQ table's Invalid column reflects the excluded verdicts."""
    inputs = _inputs(3)
    report = score(
        inputs,
        report_id="r",
        candidate_set="c",
        reference_set="ref",
        judge_client=ScriptedJudge("???"),
        verdicts_path=str(tmp_path / "v.jsonl"),
        metrics="jmq",
        seed=4,
    )
    md = render_markdown(report)
    # The 'overall' row shows Invalid=3 (all three pairs invalid).
    assert "| overall |" in md
    overall_line = next(li for li in md.splitlines() if li.startswith("| overall |"))
    # Columns: | dim | score | wins | losses | invalid | n |
    cells = [c.strip() for c in overall_line.strip("|").split("|")]
    assert cells[4] == "3"  # invalid column


def test_bias_audit_unchanged_when_judge_varies_at_fixed_seed(tmp_path):
    """Vary the judge at a FIXED seed: the bias-audit distribution is identical.

    Proves the A/B order distribution comes from the seed and pair_ids, not from
    the judge's output. A judge answering all 'A' and one answering all 'B' at
    the same seed yield the same model_first/human_first split.
    """
    inputs = _inputs(10)

    def run(reply, subdir):
        (tmp_path / subdir).mkdir()
        return score(
            inputs,
            report_id="r",
            candidate_set="c",
            reference_set="ref",
            judge_client=ScriptedJudge(reply),
            verdicts_path=str(tmp_path / subdir / "v.jsonl"),
            metrics="jmq",
            seed=77,
        ).jmq["bias_audit"]

    audit_a = run("A", "judge_a")
    audit_b = run("B", "judge_b")
    assert audit_a == audit_b


def test_emitted_report_round_trips_through_read_json(tmp_path):
    """Round-trip the emitted JSON through read_json(MetricReport).

    Catches a renamed or extra field: _from_dict rejects unknown keys and a
    missing schema_version, so a drift in the emitted shape fails loudly here.
    """
    inputs = _inputs(4)
    report = score(
        inputs,
        report_id="rt",
        candidate_set="c",
        reference_set="ref",
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=ScriptedJudge("A"),
        verdicts_path=str(tmp_path / "v.jsonl"),
        seed=8,
    )
    path = tmp_path / "report.json"
    write_json(report, path)
    restored = read_json(path, MetricReport)
    assert isinstance(restored, MetricReport)
    assert restored.report_id == "rt"
    assert restored.config["seed"] == 8
    assert restored.jmq["overall"]["n"] == 4
