"""Comparison-report assembly tests (issue #15, FR-007 / FR-010, Story 3).

Locks the DoD of `dehip report`: per-metric delta math with an unambiguous
"closer to human" sign, the HARD personal-corpus benchmark refusal (both
directions), k-trajectory ordering from deliberately-mixed input, the
always-present external-protocol benchmark caveat, and null-metric handling
that never folds an un-run metric into a 0 delta.

Everything runs in-process on fixture MetricReports built through the existing
schemas (no model, no network). The CLI paths run through ``cli.main`` and
assert the documented exit codes.
"""

from __future__ import annotations

import json
import math

from dehip import benchmark as benchmark_mod
from dehip import cli
from dehip import report as report_mod
from dehip.schemas import MetricReport, TextSet, read_json, write_json

# --- Fixtures ----------------------------------------------------------------


def _report(
    report_id,
    *,
    mmd,
    token_l2,
    jmq_overall,
    corpus="fineweb",
    candidate_set="cand",
    reference_set="ref",
    round_no=None,
):
    """Build a MetricReport fixture with the given metric values.

    ``mmd``/``token_l2`` may be ``None`` (rendered here as the NaN sentinel a
    real un-run metric carries), or a float. ``jmq_overall`` may be ``None`` (no
    jmq block / no valid verdicts) or a float, wrapped in the per-dimension
    aggregate shape ``jmq.overall.score``.
    """
    compared = {
        "candidate_set": candidate_set,
        "reference_set": reference_set,
        "n": 40,
        "corpus": corpus,
    }
    if round_no is not None:
        compared["round"] = round_no
    jmq: dict = {}
    if jmq_overall is not None:
        jmq = {"overall": {"score": jmq_overall, "win_rate": jmq_overall / 2}}
    return MetricReport(
        report_id=report_id,
        compared=compared,
        config={"seed": 0},
        mmd=float("nan") if mmd is None else mmd,
        token_l2=float("nan") if token_l2 is None else token_l2,
        jmq=jmq,
        timestamps={"started": "t", "finished": "t"},
    )


# --- Delta math (rewrite - draft, direction-aware) ---------------------------


def test_delta_math_mmd_and_token_l2_lower_is_closer_to_human():
    draft = _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.80)
    rewrite = _report("r", mmd=0.04, token_l2=0.006, jmq_overall=0.95)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    deltas = comparison["rewrites"][0]["deltas"]

    # MMD: 0.04 - 0.10 = -0.06, lower is better -> improved (closer to human).
    assert math.isclose(deltas["mmd"]["delta"], -0.06)
    assert deltas["mmd"]["improved"] is True
    assert deltas["mmd"]["comparable"] is True
    # token-L2: 0.006 - 0.010 = -0.004, lower is better -> improved.
    assert math.isclose(deltas["token_l2"]["delta"], -0.004)
    assert deltas["token_l2"]["improved"] is True


def test_delta_math_jmq_higher_is_closer_to_human():
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.60)
    rewrite = _report("r", mmd=0.05, token_l2=0.005, jmq_overall=0.90)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    jmq_delta = comparison["rewrites"][0]["deltas"]["jmq"]
    # 0.90 - 0.60 = +0.30, higher is better -> improved.
    assert math.isclose(jmq_delta["delta"], 0.30)
    assert jmq_delta["improved"] is True


def test_delta_regression_marks_farther_from_human():
    draft = _report("d", mmd=0.02, token_l2=0.003, jmq_overall=0.90)
    # Rewrite got WORSE on every metric.
    rewrite = _report("r", mmd=0.08, token_l2=0.009, jmq_overall=0.40)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    deltas = comparison["rewrites"][0]["deltas"]
    assert deltas["mmd"]["improved"] is False  # went up, lower-is-better
    assert deltas["token_l2"]["improved"] is False
    assert deltas["jmq"]["improved"] is False  # went down, higher-is-better


def test_zero_delta_renders_as_no_change_not_regression():
    """IMPORTANT 1(a): an EXACTLY-zero delta is "no change", not a regression.

    A metric that did not move (delta == 0) must not be rendered as "farther from
    human" -- that would report a no-change as a false regression. The tri-state
    ``status`` is "unchanged", ``improved`` is None (a no-change is not a
    regression), and the rendered cell says "no change".
    """
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80)
    # MMD identical (delta 0); the others move so the row still assembles.
    rewrite = _report("r", mmd=0.05, token_l2=0.004, jmq_overall=0.85)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    mmd_cell = comparison["rewrites"][0]["deltas"]["mmd"]
    assert mmd_cell["delta"] == 0
    assert mmd_cell["status"] == "unchanged"
    assert mmd_cell["improved"] is None  # not folded into a regression
    assert mmd_cell["comparable"] is True

    rendered = report_mod.render_comparison(comparison)
    # The rendered MMD row says "no change", never "farther from human".
    mmd_row = next(ln for ln in rendered.splitlines() if ln.startswith("| mmd |"))
    assert "no change" in mmd_row
    assert "farther from human" not in mmd_row
    assert "closer to human" not in mmd_row


def test_rendered_arrow_direction_matches_improvement():
    """IMPORTANT 1(b): the RENDERED arrow direction is asserted, not just the dict.

    A JMQ gain must render "closer to human" and an MMD regression "farther from
    human". Asserting the rendered markdown (not only the delta dict boolean)
    means a flipped _fmt_delta ternary is caught here, where a dict-only assertion
    would leave it green.
    """
    draft = _report("d", mmd=0.02, token_l2=0.005, jmq_overall=0.60)
    # JMQ up (closer to human), MMD up (farther from human, lower-is-better).
    rewrite = _report("r", mmd=0.09, token_l2=0.005, jmq_overall=0.90)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    rendered = report_mod.render_comparison(comparison)
    jmq_row = next(ln for ln in rendered.splitlines() if ln.startswith("| jmq |"))
    mmd_row = next(ln for ln in rendered.splitlines() if ln.startswith("| mmd |"))
    # A gain renders closer, a regression renders farther -- and never the reverse.
    assert "closer to human" in jmq_row and "farther from human" not in jmq_row
    assert "farther from human" in mmd_row and "closer to human" not in mmd_row


# --- Null / un-run metric handling (never a silent 0 delta) ------------------


def test_null_metric_in_one_report_is_not_comparable_not_zero():
    # Rewrite never ran MMD (un-run -> NaN sentinel).
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80)
    rewrite = _report("r", mmd=None, token_l2=0.004, jmq_overall=0.85)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    mmd_delta = comparison["rewrites"][0]["deltas"]["mmd"]
    # A null in one report must NOT become a 0 delta.
    assert mmd_delta["delta"] is None
    assert mmd_delta["improved"] is None
    assert mmd_delta["comparable"] is False
    assert mmd_delta["rewrite"] is None
    # The comparable metrics still resolve.
    assert comparison["rewrites"][0]["deltas"]["token_l2"]["comparable"] is True


def test_null_jmq_when_no_overall_dimension():
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=None)
    rewrite = _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85)
    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    jmq_delta = comparison["rewrites"][0]["deltas"]["jmq"]
    assert jmq_delta["draft"] is None
    assert jmq_delta["delta"] is None
    assert jmq_delta["comparable"] is False


def test_null_on_draft_side_of_distance_metric_is_not_comparable():
    """NIT 2(1): the DRAFT being null (rewrite present) is also not-comparable.

    The either-side None guard must fire regardless of WHICH side is null. Here
    the draft never ran token-L2 (NaN sentinel) while the rewrite has a value; the
    delta must be None / not-comparable, never a silent 0. A mutation narrowing
    the guard to only the rewrite side (dropping the draft-None branch) fails this.
    """
    draft = _report("d", mmd=0.05, token_l2=None, jmq_overall=0.80)
    rewrite = _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    token_delta = comparison["rewrites"][0]["deltas"]["token_l2"]
    assert token_delta["draft"] is None
    assert token_delta["rewrite"] == 0.004  # the present side is preserved
    assert token_delta["delta"] is None
    assert token_delta["improved"] is None
    assert token_delta["comparable"] is False
    # The comparable metrics on the same row still resolve.
    assert comparison["rewrites"][0]["deltas"]["mmd"]["comparable"] is True


# --- FR-010: personal-corpus benchmark refusal (BOTH directions) -------------


def test_personal_corpus_with_benchmark_refuses_loudly():
    draft = _report(
        "d", mmd=0.05, token_l2=0.005, jmq_overall=0.80, corpus="personal"
    )
    rewrite = _report(
        "r", mmd=0.04, token_l2=0.004, jmq_overall=0.85, corpus="personal"
    )

    try:
        report_mod.assemble_comparison(
            draft=draft,
            rewrites=[rewrite],
            comparison_id="c",
            attach_benchmark=True,
        )
    except report_mod.PersonalCorpusBenchmarkError as exc:
        assert "FR-010" in str(exc)
        assert "personal" in str(exc)
    else:
        raise AssertionError("expected PersonalCorpusBenchmarkError, got none")


def test_personal_corpus_refusal_triggers_on_any_input():
    # Draft is fineweb, but ONE rewrite is personal -> still refuse.
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80, corpus="fineweb")
    ok = _report("r1", mmd=0.04, token_l2=0.004, jmq_overall=0.85, corpus="fineweb")
    personal = _report(
        "r2", mmd=0.03, token_l2=0.003, jmq_overall=0.90, corpus="personal"
    )

    try:
        report_mod.assemble_comparison(
            draft=draft,
            rewrites=[ok, personal],
            comparison_id="c",
            attach_benchmark=True,
        )
    except report_mod.PersonalCorpusBenchmarkError as exc:
        assert "r2" in str(exc)
    else:
        raise AssertionError("expected refusal on a personal rewrite input")


def test_personal_corpus_detected_by_set_id_naming_fallback():
    # No explicit corpus tag, but the set id names the personal corpus.
    draft = _report(
        "d",
        mmd=0.05,
        token_l2=0.005,
        jmq_overall=0.80,
        corpus="",  # blank explicit tag
        candidate_set="personal-human",
        reference_set="personal-human",
    )
    # Blank corpus dropped so the fallback (set-id) is what fires.
    draft.compared.pop("corpus")
    rewrite = _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85)

    try:
        report_mod.assemble_comparison(
            draft=draft,
            rewrites=[rewrite],
            comparison_id="c",
            attach_benchmark=True,
        )
    except report_mod.PersonalCorpusBenchmarkError:
        pass
    else:
        raise AssertionError("set-id fallback should catch a personal corpus")


def test_personal_corpus_without_benchmark_assembles_non_benchmark_comparison():
    # Contract: without --benchmark, a personal-corpus comparison still assembles
    # (deltas, no benchmark rows), it does not refuse.
    draft = _report(
        "d", mmd=0.05, token_l2=0.005, jmq_overall=0.80, corpus="personal"
    )
    rewrite = _report(
        "r", mmd=0.04, token_l2=0.004, jmq_overall=0.85, corpus="personal"
    )

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c", attach_benchmark=False
    )
    assert "benchmark" not in comparison
    assert comparison["rewrites"][0]["deltas"]["mmd"]["comparable"] is True


# --- k-trajectory ordering from MIXED input ----------------------------------


def test_trajectory_orders_by_round_from_mixed_input():
    draft = _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60)
    # Deliberately mixed: k3, k1, k2.
    k3 = _report("k3", mmd=0.03, token_l2=0.003, jmq_overall=0.90, round_no=3)
    k1 = _report("k1", mmd=0.07, token_l2=0.007, jmq_overall=0.70, round_no=1)
    k2 = _report("k2", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=2)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[k3, k1, k2], comparison_id="c"
    )
    rounds = [row["round"] for row in comparison["trajectory"]]
    assert rounds == [1, 2, 3]
    report_ids = [row["report_id"] for row in comparison["trajectory"]]
    assert report_ids == ["k1", "k2", "k3"]


def test_trajectory_round_order_in_rendered_markdown():
    """NIT 2(2): the RENDERED trajectory rows are in round order (k1, k2, k3).

    Asserting the markdown row order (not only the JSON list) catches a rendering
    path that iterates the trajectory in a different order than it was sorted.
    """
    draft = _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60)
    # Mixed input order; the rendered rows must still be k1 < k2 < k3.
    k3 = _report("k3", mmd=0.03, token_l2=0.003, jmq_overall=0.90, round_no=3)
    k1 = _report("k1", mmd=0.07, token_l2=0.007, jmq_overall=0.70, round_no=1)
    k2 = _report("k2", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=2)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[k3, k1, k2], comparison_id="c"
    )
    rendered = report_mod.render_comparison(comparison)
    # Locate the trajectory table's data rows (round labels k1/k2/k3).
    idx_k1 = rendered.index("| k1 |")
    idx_k2 = rendered.index("| k2 |")
    idx_k3 = rendered.index("| k3 |")
    assert idx_k1 < idx_k2 < idx_k3


def test_single_rewrite_has_no_trajectory():
    draft = _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60)
    rewrite = _report("r", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=1)
    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c"
    )
    assert "trajectory" not in comparison


def test_round_comes_from_compared_not_config_k():
    """NIT 1: the round is read from compared["round"], never config["k"].

    ``k`` is overloaded (the rewrite state machine's current-round index), so a
    stray ``config["k"]`` must NOT be treated as the trajectory round. The primary
    signal, ``compared["round"]``, still orders the trajectory; a config ``k`` of
    a different meaning is ignored (round reads None on the ordering-only report).
    """
    draft = _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60)
    # compared["round"] present -> used. A misleading config["k"] is also stamped
    # and must be ignored (would have flipped ordering if consulted).
    k1 = _report("k1", mmd=0.07, token_l2=0.007, jmq_overall=0.70, round_no=1)
    k1.config["k"] = 99
    k2 = _report("k2", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=2)
    k2.config["k"] = 0

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[k2, k1], comparison_id="c"
    )
    # Ordered by compared["round"] (1, 2), unaffected by the stray config["k"].
    assert [row["round"] for row in comparison["trajectory"]] == [1, 2]
    assert [row["report_id"] for row in comparison["trajectory"]] == ["k1", "k2"]

    # A report with ONLY config["k"] (no compared/config round) reads round None,
    # proving the k fallback was dropped.
    only_k = _report("ok", mmd=0.04, token_l2=0.004, jmq_overall=0.88)
    only_k.compared.pop("round", None)
    only_k.config["k"] = 5
    assert report_mod._report_round(only_k) is None


# --- Benchmark rows + always-present external-protocol caveat -----------------


def test_benchmark_rows_transcribed_from_plan_md():
    # Guards the exact PLAN.md values against a silent typo.
    by_model = {row.model: row for row in benchmark_mod.BENCHMARK_ROWS}
    assert by_model["14B SFT superbaseline"].mmd == 0.037
    assert by_model["14B SFT superbaseline"].jmq == 0.49
    assert by_model["14B SFT superbaseline"].token_l2 == 0.0039
    assert by_model["14B DFT"].mmd == 0.018
    assert by_model["14B DFT"].jmq == 0.80
    assert by_model["14B DFT"].token_l2 == 0.0036
    assert by_model["8B DFT"].mmd == 0.023
    assert by_model["8B DFT"].jmq == 0.56
    assert by_model["8B DFT"].token_l2 == 0.0031
    assert by_model["4B DFT"].mmd == 0.025
    assert by_model["4B DFT"].jmq == 0.40
    assert by_model["4B DFT"].token_l2 == 0.0042


def test_benchmark_table_flagged_external_protocol():
    assert benchmark_mod.BENCHMARK_TABLE.external_protocol is True
    payload = benchmark_mod.benchmark_table_to_jsonable()
    assert payload["external_protocol"] is True
    assert payload["caveat"]


def test_benchmark_caveat_always_rendered_when_rows_present():
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80)
    rewrite = _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85)

    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c", attach_benchmark=True
    )
    # JSON payload carries the caveat and the external_protocol flag.
    assert comparison["benchmark"]["external_protocol"] is True
    assert comparison["benchmark"]["caveat"]

    # Markdown always shows the caveat wherever the rows appear.
    rendered = report_mod.render_comparison(comparison)
    assert benchmark_mod.EXTERNAL_PROTOCOL_CAVEAT in rendered
    assert "14B SFT superbaseline" in rendered
    # The rows never appear without the caveat: caveat text precedes the table.
    assert rendered.index("EXTERNAL PROTOCOL") < rendered.index("14B SFT superbaseline")


def test_no_benchmark_no_rows_no_caveat():
    draft = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80)
    rewrite = _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85)
    comparison = report_mod.assemble_comparison(
        draft=draft, rewrites=[rewrite], comparison_id="c", attach_benchmark=False
    )
    assert "benchmark" not in comparison
    rendered = report_mod.render_comparison(comparison)
    assert benchmark_mod.EXTERNAL_PROTOCOL_CAVEAT not in rendered


# --- CLI wiring (dehip report) -----------------------------------------------


def _write_report(path, report):
    write_json(report, path)
    return str(path)


def test_cli_report_writes_json_and_md(tmp_path):
    draft = _write_report(
        tmp_path / "draft.json",
        _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60),
    )
    r1 = _write_report(
        tmp_path / "r1.json",
        _report("r1", mmd=0.07, token_l2=0.007, jmq_overall=0.70, round_no=1),
    )
    r2 = _write_report(
        tmp_path / "r2.json",
        _report("r2", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=2),
    )
    out = tmp_path / "cmp.json"

    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            r2,  # deliberately out of order
            "--rewrite-report",
            r1,
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    # Trajectory rendered in round order despite mixed CLI arg order.
    assert [row["round"] for row in written["trajectory"]] == [1, 2]
    md = (tmp_path / "cmp.md").read_text()
    assert "k-round trajectory" in md


def test_cli_report_md_render_failure_leaves_neither_artifact(tmp_path, monkeypatch):
    """IMPORTANT 3: a .md render/write failure leaves NEITHER .json nor .md.

    _emit_comparison stages both artifacts to temp files and only commits once
    BOTH stage successfully, mirroring _emit_report's all-or-nothing pair. When
    render_comparison raises OSError after the .json temp is staged, the branch
    must discard the staged .json, exit EXIT_IO (5), leave no orphaned .json, no
    .md, and no .tmp debris.
    """
    draft = _write_report(
        tmp_path / "draft.json",
        _report("d", mmd=0.10, token_l2=0.010, jmq_overall=0.60),
    )
    rewrite = _write_report(
        tmp_path / "r.json",
        _report("r", mmd=0.05, token_l2=0.005, jmq_overall=0.80, round_no=1),
    )

    def _boom(_comparison):
        raise OSError("simulated md render failure")

    monkeypatch.setattr(report_mod, "render_comparison", _boom)

    out = tmp_path / "cmp.json"
    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            rewrite,
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_IO
    # All-or-nothing: neither final artifact exists, no temp debris.
    assert not out.exists()
    assert not (tmp_path / "cmp.md").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_report_benchmark_refusal_exits_2(tmp_path):
    draft = _write_report(
        tmp_path / "draft.json",
        _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80, corpus="personal"),
    )
    rewrite = _write_report(
        tmp_path / "r.json",
        _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85, corpus="personal"),
    )
    out = tmp_path / "cmp.json"

    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            rewrite,
            "--benchmark",
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    # HARD refusal: no partial artifact written.
    assert not out.exists()
    assert not (tmp_path / "cmp.md").exists()


def test_cli_report_benchmark_attaches_for_fineweb(tmp_path):
    draft = _write_report(
        tmp_path / "draft.json",
        _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80, corpus="fineweb"),
    )
    rewrite = _write_report(
        tmp_path / "r.json",
        _report("r", mmd=0.04, token_l2=0.004, jmq_overall=0.85, corpus="fineweb"),
    )
    out = tmp_path / "cmp.json"

    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            rewrite,
            "--benchmark",
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    assert written["benchmark"]["external_protocol"] is True
    assert len(written["benchmark"]["rows"]) == 4


def test_cli_report_missing_file_exits_2(tmp_path):
    draft = _write_report(
        tmp_path / "draft.json",
        _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80),
    )
    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            str(tmp_path / "nope.json"),
        ]
    )
    assert rc == cli.EXIT_VALIDATION


def test_cli_report_round_trips_through_schema_reader(tmp_path):
    # The CLI reads reports via the existing schema reader; a report written by
    # write_json round-trips back through read_json without a parallel reader.
    report = _report("d", mmd=0.05, token_l2=0.005, jmq_overall=0.80)
    path = tmp_path / "d.json"
    write_json(report, path)
    loaded = read_json(path, MetricReport)
    assert loaded.report_id == "d"
    assert report_mod.report_corpus(loaded) == "fineweb"


# --- IMPORTANT 2(b): end-to-end wired corpus drives the FR-010 refusal --------


class _StubTokenizer:
    """Whitespace tokenizer stub for the token_l2 score path (no real model)."""

    tokenizer_id = "stub-tokenizer"

    def tokenize(self, text: str):
        return text.split()


def _score_personal_report(tmp_path, monkeypatch, name):
    """Produce ONE report through the REAL score path over a personal corpus.

    Builds candidate/reference TextSet manifests tagged ``corpus="personal"`` and
    runs ``dehip score --metrics token_l2`` so the corpus is wired from the
    manifest through MetricInputs -> score() -> compared["corpus"] the same way
    production emits it -- no hand-stamped fixture. Returns the report path.
    """
    monkeypatch.setattr(
        "dehip.metrics.token_l2.Qwen3Tokenizer", lambda *a, **k: _StubTokenizer()
    )
    pair_ids = [f"personal-{i}" for i in range(4)]
    for role, tag, texts in (
        ("instruct_draft", "cand", "model output"),
        ("human_reference", "ref", "human reference"),
    ):
        manifest = TextSet(
            set_id=f"{tag}-set",
            role=role,
            corpus="personal",
            pair_ids=pair_ids,
            provenance={"texts_path": f"{name}-{tag}.jsonl"},
        )
        write_json(manifest, tmp_path / f"{name}-{tag}.manifest.json")
        with (tmp_path / f"{name}-{tag}.jsonl").open("w", encoding="utf-8") as fh:
            for i, pid in enumerate(pair_ids):
                fh.write(
                    json.dumps({"pair_id": pid, "text": f"{texts} {i} words here"})
                    + "\n"
                )
    out = tmp_path / f"{name}-report.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            str(tmp_path / f"{name}-cand.manifest.json"),
            "--reference",
            str(tmp_path / f"{name}-ref.manifest.json"),
            "--metrics",
            "token_l2",
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    return str(out)


def test_cli_report_benchmark_refuses_on_wired_personal_corpus(tmp_path, monkeypatch):
    """IMPORTANT 2(b): a report with ONLY the production-wired corpus tag refuses.

    The FR-010 refusal must fire on the PRIMARY signal (compared["corpus"] wired
    by the real score path), not merely the set-id naming fallback. Here the set
    ids are ``cand-set``/``ref-set`` (no ``personal-`` prefix, so the fallback
    does NOT fire); only the wired ``compared["corpus"] == "personal"`` can catch
    it. --benchmark over such a report must exit EXIT_VALIDATION with no artifact.
    """
    draft = _score_personal_report(tmp_path, monkeypatch, "draft")
    # Confirm the primary signal is present and the fallback is NOT the trigger.
    loaded = read_json(draft, MetricReport)
    assert loaded.compared["corpus"] == "personal"
    assert not loaded.compared["candidate_set"].startswith("personal-")
    rewrite = _score_personal_report(tmp_path, monkeypatch, "rw")

    out = tmp_path / "cmp.json"
    rc = cli.main(
        [
            "report",
            "--draft-report",
            draft,
            "--rewrite-report",
            rewrite,
            "--benchmark",
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    assert not out.exists()
    assert not (tmp_path / "cmp.md").exists()
