"""Tests for JMQ judge orchestration (FR-002, FR-008, FR-009; research.md R6).

All tests use a mocked judge client. No real OpenAI calls are made, and the
module is imported without the SDK or an API key being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dehip.metrics.jmq import (
    DIMENSION_ORDER,
    CostThresholdError,
    JudgePair,
    aggregate_verdicts,
    assign_order,
    cost_preflight,
    estimate_call_count,
    load_judge_prompts,
    parse_choice,
    render_prompt,
    run_judging,
)
from dehip.schemas import JudgeVerdict, read_jsonl

# --- Mock judge clients ------------------------------------------------------


class ScriptedJudge:
    """Judge client returning a fixed sequence of replies, recording calls.

    Never touches the network. Each ``judge`` call pops the next scripted reply;
    ``calls`` records every rendered prompt seen so retry behavior is testable.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls.append(rendered_prompt)
        if not self._replies:
            raise AssertionError("ScriptedJudge ran out of scripted replies")
        return self._replies.pop(0)


class ModelAlwaysWinsJudge:
    """Judge that always picks whichever candidate is the model output.

    Reads the rendered prompt to find which candidate slot holds the model text,
    so it exercises the real A/B-order-to-winner derivation rather than a fixed
    letter. Marker strings are embedded by the test's pairs.
    """

    def __init__(self, model_marker: str) -> None:
        self._marker = model_marker

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        a_idx = rendered_prompt.index("=== CANDIDATE A ===")
        b_idx = rendered_prompt.index("=== CANDIDATE B ===")
        candidate_a = rendered_prompt[a_idx:b_idx]
        return "A" if self._marker in candidate_a else "B"


class CountingJudge:
    """Judge returning a fixed valid reply and counting every call.

    Used to snapshot the judge-call count so a later aggregation can be proven to
    issue zero additional calls.
    """

    def __init__(self, reply: str = "A") -> None:
        self._reply = reply
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        return self._reply


class RaisesOnKthCallJudge:
    """Judge that returns a valid reply until the Kth call, which raises always.

    Models a judge that dies partway through a run. Every call is counted; the
    Kth and every later call raise a transient-looking error, so with bounded
    retries that (pair, dimension) ends up a counted failure marker while the
    verdicts completed before it stay on disk.
    """

    def __init__(self, fail_from_call: int, reply: str = "A") -> None:
        self._fail_from_call = fail_from_call
        self._reply = reply
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        if self.calls >= self._fail_from_call:
            raise ConnectionError(f"simulated transient failure at call {self.calls}")
        return self._reply


class FlakyThenOkJudge:
    """Judge that raises for the first ``fail_times`` calls, then returns a reply.

    Exercises the retry path: a transient error that clears on a later attempt
    must yield a valid verdict, not a failure marker.
    """

    def __init__(self, fail_times: int, reply: str = "A") -> None:
        self._remaining_failures = fail_times
        self._reply = reply
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise TimeoutError("simulated transient timeout")
        return self._reply


class AlwaysRaisesJudge:
    """Judge whose every call raises a transient-looking error."""

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        raise ConnectionError("simulated persistent transient failure")


def _pairs(n: int) -> list[JudgePair]:
    return [
        JudgePair(
            pair_id=f"fineweb-{i}",
            prompt=f"prompt {i}",
            model_text=f"MODEL_MARKER model output {i}",
            human_text=f"human reference {i}",
        )
        for i in range(n)
    ]


# --- Templates load byte-identical to the files ------------------------------


def test_templates_load_byte_identical_to_files() -> None:
    prompts = load_judge_prompts()
    source = Path(prompts.source_dir)
    assert source.is_dir()
    for dimension in DIMENSION_ORDER:
        on_disk = (source / f"{dimension}.txt").read_bytes()
        # In-memory template must decode to exactly the file's bytes: no
        # stripping, newline translation, or normalization.
        assert prompts.templates[dimension].encode("utf-8") == on_disk
        # And the recorded checksum must match a fresh hash of those bytes.
        import hashlib

        assert prompts.checksums[dimension] == hashlib.sha256(on_disk).hexdigest()


def test_load_judge_prompts_missing_file_raises(tmp_path: Path) -> None:
    (tmp_path / "overall.txt").write_text("only one file")
    with pytest.raises(FileNotFoundError):
        load_judge_prompts(tmp_path)


def test_render_prompt_tolerates_literal_braces() -> None:
    # A candidate containing braces must not be treated as a format field.
    rendered = render_prompt(
        "P={prompt} A={candidate_a} B={candidate_b}",
        prompt="the {task}",
        candidate_a="{not_a_field}",
        candidate_b="plain",
    )
    assert rendered == "P=the {task} A={not_a_field} B=plain"


# --- Seeded A/B order: ~50/50 and exactly reproducible -----------------------


def test_ab_order_is_roughly_balanced_over_200_pairs() -> None:
    seed = 12345
    pairs = _pairs(200)
    orders = [assign_order(p.pair_id, seed) for p in pairs]
    model_first = orders.count("model_first")
    human_first = orders.count("human_first")
    assert model_first + human_first == 200
    # ~50/50: allow a generous band so the test is not flaky, while still
    # catching a stuck-on-one-side bug (which would land at 0 or 200).
    assert 80 <= model_first <= 120, (model_first, human_first)


def test_ab_order_is_exactly_reproducible_from_seed() -> None:
    pairs = _pairs(200)
    first = [assign_order(p.pair_id, 999) for p in pairs]
    second = [assign_order(p.pair_id, 999) for p in pairs]
    assert first == second
    # A different seed produces a different sequence (not identical), proving the
    # seed actually drives the assignment.
    other = [assign_order(p.pair_id, 1000) for p in pairs]
    assert other != first


def test_ab_order_is_stable_when_pairs_added() -> None:
    # Adding a pair must not reshuffle the others (per-pair-keyed PRNG).
    seed = 7
    base = {p.pair_id: assign_order(p.pair_id, seed) for p in _pairs(10)}
    extended = {p.pair_id: assign_order(p.pair_id, seed) for p in _pairs(20)}
    for pair_id, order in base.items():
        assert extended[pair_id] == order


def test_run_judging_records_order_and_replays_from_seed(tmp_path: Path) -> None:
    pairs = _pairs(50)
    prompts = load_judge_prompts()

    def run(path: Path) -> list[JudgeVerdict]:
        return run_judging(
            pairs,
            path,
            client=ModelAlwaysWinsJudge("MODEL_MARKER"),
            prompts=prompts,
            seed=42,
            max_workers=4,
        )

    v1 = run(tmp_path / "a.jsonl")
    v2 = run(tmp_path / "b.jsonl")
    order_by_pair_1 = {(v.pair_id, v.dimension): v.order for v in v1}
    order_by_pair_2 = {(v.pair_id, v.dimension): v.order for v in v2}
    # Identical seed -> identical recorded order assignments across full reruns.
    assert order_by_pair_1 == order_by_pair_2
    # The same pair gets the same order across all six dimensions.
    for pair in pairs:
        orders = {v.order for v in v1 if v.pair_id == pair.pair_id}
        assert len(orders) == 1


# --- Invalid verdict: retried once, then excluded-and-counted ----------------


def test_invalid_verdict_retried_then_excluded_and_counted(tmp_path: Path) -> None:
    prompts = load_judge_prompts()
    pair = JudgePair(
        pair_id="p-invalid",
        prompt="q",
        model_text="MODEL_MARKER m",
        human_text="h",
    )
    # First reply malformed, retry also malformed -> invalid after one retry.
    judge = ScriptedJudge(["I cannot decide", "still garbage"] * len(DIMENSION_ORDER))
    verdicts = run_judging(
        [pair], tmp_path / "v.jsonl", client=judge, prompts=prompts, seed=1
    )

    assert len(verdicts) == len(DIMENSION_ORDER)
    for v in verdicts:
        assert v.choice == "invalid"
        assert v.retry_count == 1
        assert v.model_won is None
    # Exactly two calls per (pair, dimension): the original plus one retry.
    assert len(judge.calls) == 2 * len(DIMENSION_ORDER)

    scores = aggregate_verdicts(verdicts)
    for dimension in DIMENSION_ORDER:
        # Invalid is surfaced as a count, never dropped and never scored.
        assert scores[dimension]["invalid"] == 1
        assert scores[dimension]["n"] == 0
        assert scores[dimension]["wins"] == 0
        assert scores[dimension]["losses"] == 0
        assert scores[dimension]["score"] is None


def test_valid_after_one_retry_is_scored_not_invalid(tmp_path: Path) -> None:
    prompts = load_judge_prompts()
    pair = JudgePair(
        pair_id="p-recover",
        prompt="q",
        model_text="MODEL_MARKER m",
        human_text="h",
    )
    # First reply malformed, retry valid ("A"): counts as a real verdict.
    judge = ScriptedJudge(["???", "A"] * len(DIMENSION_ORDER))
    verdicts = run_judging(
        [pair], tmp_path / "v.jsonl", client=judge, prompts=prompts, seed=1
    )
    for v in verdicts:
        assert v.choice == "A"
        assert v.retry_count == 1
        assert v.model_won is not None


def test_parse_choice_variants() -> None:
    assert parse_choice("A") == "A"
    assert parse_choice(" b ") == "B"
    assert parse_choice("A.") == "A"
    assert parse_choice("B.\n") == "B"
    assert parse_choice("") == "invalid"
    assert parse_choice("The answer is A") == "invalid"
    assert parse_choice("A or B") == "invalid"


# --- Aggregation reads only persisted verdicts (crash-safety, FR-008) --------


def test_aggregation_reads_only_persisted_verdicts(tmp_path: Path) -> None:
    """A crash between judging and scoring must lose nothing.

    run_judging persists verdicts to disk before returning. We then throw away
    the in-memory return value entirely and recompute JMQ from the JSONL alone,
    proving aggregation depends only on the persisted file (no re-querying).
    """
    pairs = _pairs(20)
    prompts = load_judge_prompts()
    path = tmp_path / "verdicts.jsonl"

    returned = run_judging(
        pairs,
        path,
        client=ModelAlwaysWinsJudge("MODEL_MARKER"),
        prompts=prompts,
        seed=3,
    )
    assert path.exists()

    # Simulate a crash: discard the return value, reload from disk, aggregate.
    from_disk = read_jsonl(path, JudgeVerdict)
    assert len(from_disk) == 20 * len(DIMENSION_ORDER)

    scores_from_path = aggregate_verdicts(path)
    scores_from_returned = aggregate_verdicts(returned)
    # Recomputing from the persisted file matches scoring the in-memory verdicts.
    assert scores_from_path == scores_from_returned

    # ModelAlwaysWins -> model wins every valid comparison -> win_rate 1.0,
    # score 2.0, for every dimension, regardless of A/B order.
    for dimension in DIMENSION_ORDER:
        row = scores_from_path[dimension]
        assert row["invalid"] == 0
        assert row["n"] == 20
        assert row["wins"] == 20
        assert row["win_rate"] == 1.0
        assert row["score"] == 2.0


def test_persisted_verdicts_written_before_return(tmp_path: Path) -> None:
    # The file must exist and be complete the moment run_judging returns.
    pairs = _pairs(5)
    prompts = load_judge_prompts()
    path = tmp_path / "v.jsonl"
    run_judging(
        pairs, path, client=ModelAlwaysWinsJudge("MODEL_MARKER"), prompts=prompts
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 5 * len(DIMENSION_ORDER)


def test_win_rate_reflects_split_outcomes(tmp_path: Path) -> None:
    """A half-win/half-loss judge yields win_rate ~0.5 and JMQ ~1.0."""
    prompts = load_judge_prompts()
    pairs = _pairs(40)
    path = tmp_path / "v.jsonl"

    # Judge picks candidate A always. With seeded ~50/50 order, model is A in
    # about half the pairs, so model win-rate lands near 0.5.
    class AlwaysA:
        def judge(self, rendered_prompt: str, *, model: str) -> str:
            return "A"

    run_judging(pairs, path, client=AlwaysA(), prompts=prompts, seed=99)
    scores = aggregate_verdicts(path)
    overall = scores["overall"]
    assert overall["n"] == 40
    assert 0.3 <= overall["win_rate"] <= 0.7
    assert abs(overall["score"] - 2.0 * overall["win_rate"]) < 1e-9


# --- Durability: incremental persistence + retry / failure marker ------------


@pytest.fixture
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize retry backoff so failure/retry tests run instantly."""
    import dehip.metrics.jmq as jmq_mod

    monkeypatch.setattr(jmq_mod.time, "sleep", lambda _seconds: None)


def test_completed_verdicts_survive_a_mid_run_judge_failure(
    tmp_path: Path, _no_backoff_sleep: None
) -> None:
    """A judge dying mid-run must not discard the verdicts already computed.

    With single workers the calls are serialized, so the first (K-1) calls
    succeed and land on disk before the Kth begins failing. The run must COMPLETE
    (not abort): the failed (pair, dimension) becomes a counted failure marker and
    aggregation surfaces the shortfall as an ``invalid`` count.
    """
    prompts = load_judge_prompts()
    pairs = _pairs(4)  # 4 pairs x 6 dims = 24 tasks
    path = tmp_path / "v.jsonl"

    # Succeed on the first 5 calls, then fail on every call from the 6th on. With
    # bounded retries each failing task consumes several calls, but every task
    # still resolves to a verdict (valid or counted-failure); the run completes.
    judge = RaisesOnKthCallJudge(fail_from_call=6, reply="A")

    verdicts = run_judging(
        pairs,
        path,
        client=judge,
        prompts=prompts,
        seed=5,
        max_workers=1,
    )

    # The run completed: one verdict per (pair, dimension), all persisted.
    assert len(verdicts) == 4 * len(DIMENSION_ORDER)
    from_disk = read_jsonl(path, JudgeVerdict)
    assert len(from_disk) == 4 * len(DIMENSION_ORDER)

    # At least the verdicts completed before the failure are durably on disk and
    # scored (not lost), and at least one is a counted failure marker.
    valid = [v for v in from_disk if v.choice != "invalid"]
    failures = [v for v in from_disk if v.choice == "invalid"]
    assert valid, "verdicts completed before the failure must survive on disk"
    assert failures, "the failed calls must be recorded as counted markers"
    # Failure markers carry the exception text for audit and never score.
    for v in failures:
        assert v.model_won is None
        assert "judge-call-failed" in v.raw_response

    # Aggregation surfaces the shortfall: the total invalid count across
    # dimensions equals the number of failure markers, and none are scored.
    scores = aggregate_verdicts(path)
    total_invalid = sum(scores[d]["invalid"] for d in DIMENSION_ORDER)
    assert total_invalid == len(failures)


def test_transient_failure_then_success_is_retried_to_a_valid_verdict(
    tmp_path: Path, _no_backoff_sleep: None
) -> None:
    """A judge that fails transiently then succeeds yields a valid verdict."""
    prompts = load_judge_prompts()
    pair = JudgePair(
        pair_id="p-flaky",
        prompt="q",
        model_text="MODEL_MARKER m",
        human_text="h",
    )
    # Fail once per attempt-sequence, then succeed: within the bounded retries.
    judge = FlakyThenOkJudge(fail_times=1, reply="A")

    verdicts = run_judging(
        [pair], tmp_path / "v.jsonl", client=judge, prompts=prompts, seed=1,
        max_workers=1,
    )

    assert len(verdicts) == len(DIMENSION_ORDER)
    # First dimension retried past the single transient failure to a real "A".
    assert verdicts[0].choice == "A"
    assert verdicts[0].model_won is not None
    assert "judge-call-failed" not in verdicts[0].raw_response


def test_persistent_failure_yields_counted_marker_and_run_continues(
    tmp_path: Path, _no_backoff_sleep: None
) -> None:
    """A judge that always fails yields counted markers; the run still completes."""
    prompts = load_judge_prompts()
    pairs = _pairs(2)
    path = tmp_path / "v.jsonl"
    judge = AlwaysRaisesJudge()

    verdicts = run_judging(
        pairs, path, client=judge, prompts=prompts, seed=1, max_workers=1
    )

    # Run completed with a verdict per task, all counted failure markers.
    assert len(verdicts) == 2 * len(DIMENSION_ORDER)
    assert all(v.choice == "invalid" and v.model_won is None for v in verdicts)
    # Bounded retries: more than one call per task, but a finite, bounded number.
    assert judge.calls > len(verdicts)

    scores = aggregate_verdicts(path)
    for dimension in DIMENSION_ORDER:
        row = scores[dimension]
        assert row["invalid"] == 2
        assert row["n"] == 0
        assert row["score"] is None


# --- Nit: aggregation issues zero judge calls --------------------------------


def test_aggregation_issues_zero_judge_calls(tmp_path: Path) -> None:
    """aggregate_verdicts must read the file only -- never query the judge."""
    prompts = load_judge_prompts()
    pairs = _pairs(5)
    path = tmp_path / "v.jsonl"

    judge = CountingJudge()
    run_judging(pairs, path, client=judge, prompts=prompts, seed=2, max_workers=1)
    calls_after_judging = judge.calls

    # Aggregating from the persisted path issues no further judge calls.
    aggregate_verdicts(path)
    assert judge.calls == calls_after_judging


# --- Nit: deterministic mid-range win-rate known-answer test -----------------


def test_known_answer_mid_range_win_rate_and_score() -> None:
    """Four verdicts (3 model-wins, 1 loss) -> win_rate 0.75, JMQ score 1.5."""
    verdicts = [
        JudgeVerdict(
            pair_id=f"kat-{i}",
            dimension="overall",
            judge_model="test",
            order="model_first",
            raw_response="A" if model_won else "B",
            choice="A" if model_won else "B",
            retry_count=0,
            model_won=model_won,
        )
        for i, model_won in enumerate([True, True, True, False])
    ]

    scores = aggregate_verdicts(verdicts)
    row = scores["overall"]
    assert row["wins"] == 3
    assert row["losses"] == 1
    assert row["n"] == 4
    assert row["win_rate"] == 0.75
    assert row["score"] == 1.5


# --- Cost preflight (FR-009) -------------------------------------------------


def test_estimate_call_count() -> None:
    assert estimate_call_count(0) == 0
    assert estimate_call_count(100) == 600
    with pytest.raises(ValueError):
        estimate_call_count(-1)


def test_cost_preflight_blocks_above_threshold_without_confirm() -> None:
    printed: list[str] = []
    with pytest.raises(CostThresholdError):
        cost_preflight(
            10_000,
            confirm=False,
            threshold_usd=1.0,
            cost_per_call_usd=0.001,
            printer=printed.append,
        )
    # It reports the estimate before refusing (so the operator sees the count).
    assert printed and "judge calls" in printed[0]


def test_cost_preflight_proceeds_with_confirm() -> None:
    estimate = cost_preflight(
        10_000,
        confirm=True,
        threshold_usd=1.0,
        cost_per_call_usd=0.001,
        printer=lambda _msg: None,
    )
    assert estimate["calls"] == 60_000
    assert estimate["estimated_usd"] == pytest.approx(60.0)


def test_cost_preflight_proceeds_below_threshold_without_confirm() -> None:
    estimate = cost_preflight(
        1,
        confirm=False,
        threshold_usd=1.0,
        cost_per_call_usd=0.0001,
        printer=lambda _msg: None,
    )
    # 6 calls * $0.0001 = $0.0006, below $1.00 -> no confirmation needed.
    assert estimate["calls"] == 6
