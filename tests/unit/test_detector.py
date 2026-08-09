"""Unit tests for external detector scoring (issue #14, SC-005).

Every test injects a MOCK detector client, so nothing here needs a real API key
or the network. The mocks let each behavior be pinned exactly:

- summary math against known inputs (mean, median, stdev, histogram, quantiles);
- per-text + summary persistence under results/reports/;
- the missing-key gate exiting 3 with ZERO detector calls (proved by a client
  that raises if its score_text is ever reached);
- the cost gate blocking above threshold without --yes, with zero calls;
- loud failure on a detector call error / out-of-range score (no dropped text,
  no fake 0.0), and the distinction between a real 0.0 and a failed call.
"""

from __future__ import annotations

import json
import math

import pytest

from dehip import cli
from dehip import detector as det
from dehip.schemas import TextSet, content_sha256, write_json

# --- Mock clients ------------------------------------------------------------


class MappingClient:
    """Returns a preset human-probability per text; records every call."""

    def __init__(self, by_text: dict[str, float]) -> None:
        self._by_text = by_text
        self.calls: list[str] = []

    def score_text(self, text: str) -> float:
        self.calls.append(text)
        return self._by_text[text]


class ExplodingClient:
    """Raises if ever called -- proves a code path makes ZERO detector calls."""

    def __init__(self) -> None:
        self.calls = 0

    def score_text(self, text: str) -> float:  # pragma: no cover - must not run
        self.calls += 1
        raise AssertionError("detector was called despite a pre-call gate")


class FailingClient:
    """Raises a transport-style error on the Nth text (1-based)."""

    def __init__(self, fail_on: int = 1, value: float = 0.9) -> None:
        self.fail_on = fail_on
        self.value = value
        self.calls = 0

    def score_text(self, text: str) -> float:
        self.calls += 1
        if self.calls == self.fail_on:
            raise ConnectionError("simulated network failure")
        return self.value


class OutOfRangeClient:
    """Returns an out-of-range value (a malformed response), not a valid prob."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def score_text(self, text: str) -> float:
        self.calls += 1
        return self.value


# --- Fixtures ----------------------------------------------------------------


def _write_set(tmp_path, name, set_id, role, texts_by_id):
    """Write a TextSet manifest + sibling {pair_id, text} JSONL (score's convention)."""
    pair_ids = list(texts_by_id)
    manifest = TextSet(
        set_id=set_id,
        role=role,
        corpus="fineweb",
        pair_ids=pair_ids,
        provenance={"texts_path": f"{name}.jsonl"},
    )
    manifest_path = tmp_path / f"{name}.manifest.json"
    write_json(manifest, manifest_path)
    with (tmp_path / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for pid in pair_ids:
            fh.write(json.dumps({"pair_id": pid, "text": texts_by_id[pid]}) + "\n")
    return str(manifest_path)


# --- DoD 1: summary math against known inputs --------------------------------


def test_summarize_mean_median_over_known_inputs():
    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    summary = det.summarize_scores("s", "rewrite", probs)
    assert summary.n == 5
    assert summary.mean == pytest.approx(0.5)
    assert summary.median == pytest.approx(0.5)
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    # Population stdev of an even spread about 0.5.
    assert summary.stdev == pytest.approx(math.sqrt(0.125))


def test_summarize_median_even_count_is_midpoint():
    # Even count: median is the mean of the two central values. Use an ASYMMETRIC
    # set so median != mean -- 0.2 and 0.6 are the central pair (median 0.4) while
    # the mean is 0.45. A median-becomes-mean mutation would then fail here (a
    # symmetric set would let both mutants pass).
    summary = det.summarize_scores("s", "rewrite", [0.1, 0.2, 0.6, 0.9])
    assert summary.median == pytest.approx(0.4)
    assert summary.mean == pytest.approx(0.45)
    assert summary.median != pytest.approx(summary.mean)


def test_summarize_histogram_bins_are_fixed_and_total_n():
    # 0.0 -> bin 0, 0.05 -> bin 0, 0.5 -> bin 5, 0.95 -> bin 9, 1.0 -> bin 9.
    probs = [0.0, 0.05, 0.5, 0.95, 1.0]
    summary = det.summarize_scores("s", "instruct_draft", probs)
    assert len(summary.histogram) == det.HISTOGRAM_BINS
    assert sum(summary.histogram) == len(probs)  # no text lost from the histogram
    assert summary.histogram[0] == 2  # 0.0 and 0.05
    assert summary.histogram[5] == 1  # 0.5
    assert summary.histogram[9] == 2  # 0.95 and the closed upper edge 1.0


def test_summarize_quantiles_present_and_ordered():
    summary = det.summarize_scores("s", "rewrite", [0.1, 0.2, 0.3, 0.4, 0.5])
    q = summary.quantiles
    assert set(q) == {"p25", "p50", "p75", "p90"}
    assert q["p25"] <= q["p50"] <= q["p75"] <= q["p90"]
    assert q["p50"] == pytest.approx(summary.median, abs=0.05)


def test_summarize_empty_set_raises_not_zero():
    # An empty set has no defensible mean; summarizing it must raise, never
    # silently return a 0.0 that reads as a real (AI-looking) score.
    with pytest.raises(ValueError):
        det.summarize_scores("s", "rewrite", [])


# --- score_set: a real 0.0 is kept; a failed call is not ---------------------


def test_score_set_real_zero_is_a_valid_datum():
    # A detector confident the text is AI returns 0.0 -- that is a real score and
    # must be counted, not treated as a failure.
    texts = [("p0", "aaa"), ("p1", "bbb")]
    client = MappingClient({"aaa": 0.0, "bbb": 0.2})
    summary, scores = det.score_set("s", "rewrite", texts, client=client)
    assert summary.n == 2
    assert summary.mean == pytest.approx(0.1)
    assert [sc.human_prob for sc in scores] == [0.0, 0.2]
    # text_sha traces each row to its content.
    assert scores[0].text_sha == content_sha256("aaa")


def test_score_set_call_failure_aborts_loudly_no_drop():
    # DoD: a failed call is normalized to DetectorCallError and aborts the set --
    # the failed text is never dropped, so the mean can't be silently corrupted.
    texts = [("p0", "a"), ("p1", "b"), ("p2", "c")]
    client = FailingClient(fail_on=2)
    with pytest.raises(det.DetectorCallError):
        det.score_set("s", "rewrite", texts, client=client)
    assert client.calls == 2  # stopped at the failure, did not silently continue


def test_score_set_out_of_range_response_is_a_failed_call():
    # A malformed response (prob > 1.0) is a failed call, not a real score.
    client = OutOfRangeClient(1.7)
    with pytest.raises(det.DetectorCallError):
        det.score_set("s", "rewrite", [("p0", "a")], client=client)


def test_score_set_nan_response_is_a_failed_call():
    client = OutOfRangeClient(float("nan"))
    with pytest.raises(det.DetectorCallError):
        det.score_set("s", "rewrite", [("p0", "a")], client=client)


def test_score_set_bool_response_is_a_failed_call():
    # A bool is an int subclass, so True would pass a naive 0<=v<=1 check and
    # silently count as human_prob 1.0. The _valid_human_prob bool guard rejects
    # it as a malformed response -> DetectorCallError, never a fake 1.0 score.
    client = OutOfRangeClient(True)  # noqa: FBT003 - deliberately a bool
    with pytest.raises(det.DetectorCallError):
        det.score_set("s", "rewrite", [("p0", "a")], client=client)


# --- SC-005 delta ------------------------------------------------------------


def test_sc005_delta_single_subtraction_and_pass_bar():
    draft = det.summarize_scores("draft", "instruct_draft", [0.1, 0.2, 0.3])  # mean 0.2
    rewrite = det.summarize_scores("rw", "rewrite", [0.6, 0.7, 0.8])  # mean 0.7
    sc005 = det.sc005_delta([draft, rewrite])
    assert sc005 is not None
    assert sc005["delta"] == pytest.approx(0.5)
    assert sc005["passed"] is True  # 0.5 >= 0.30
    assert sc005["draft_set"] == "draft"
    assert sc005["rewrite_set"] == "rw"


def test_sc005_delta_below_threshold_fails():
    draft = det.summarize_scores("draft", "instruct_draft", [0.4])
    rewrite = det.summarize_scores("rw", "rewrite", [0.5])  # delta 0.1 < 0.30
    sc005 = det.sc005_delta([draft, rewrite])
    assert sc005["delta"] == pytest.approx(0.1)
    assert sc005["passed"] is False


def test_sc005_delta_needs_exactly_one_draft_and_one_rewrite():
    only_draft = det.summarize_scores("d", "instruct_draft", [0.2])
    assert det.sc005_delta([only_draft]) is None


def test_sc005_delta_two_rewrites_no_draft_returns_none():
    # Two rewrite-role sets and zero drafts: the draft/rewrite comparison is
    # undefined, so the delta is left unset (None) rather than guessed from an
    # arbitrary pairing.
    rw_a = det.summarize_scores("a", "rewrite", [0.6])
    rw_b = det.summarize_scores("b", "rewrite", [0.7])
    assert det.sc005_delta([rw_a, rw_b]) is None


def test_sc005_delta_counts_degenerate_drops_as_failures():
    # 4 drafts (all AI), but only 2 rewrites survived the cascade -- the other 2
    # degenerated and were dropped. Survivor-only mean (0.5) would "pass" the raw
    # delta, but the 2 collapses are failures to humanize: over the full universe
    # the effective mean is 0.5 * 2/4 = 0.25, below the 0.30 bar.
    draft = det.summarize_scores("d", "instruct_draft", [0.0, 0.0, 0.0, 0.0])
    rewrite = det.summarize_scores("r", "rewrite", [0.0, 1.0])  # 2 survivors
    sc005 = det.sc005_delta([draft, rewrite])
    assert sc005["dropped_degenerate"] == 2
    assert sc005["delta"] == pytest.approx(0.5)  # raw survivor-only
    assert sc005["delta_effective"] == pytest.approx(0.25)  # counts the 2 drops
    assert sc005["passed"] is False  # 0.25 < 0.30, not the spurious 0.5 pass


def test_sc005_envelope_robust_fail_when_upper_below_threshold():
    # 30 drafts all AI; 19 survivors with 2 flips (mean ~0.105), 11 degenerated.
    draft = det.summarize_scores("d", "instruct_draft", [0.0] * 30)
    rewrite = det.summarize_scores("r", "rewrite", [1.0, 1.0] + [0.0] * 17)
    sc = det.sc005_delta([draft, rewrite], imprecision_s=2.0)
    assert sc["verdict"] == "robust_fail"
    assert sc["delta_upper"] < det.SC005_DELTA_THRESHOLD
    # the adversarial "drops might be human" bound crosses the bar -> the
    # degeneration assumption is what the verdict hinges on, made explicit.
    assert sc["delta_upper_if_drops_human"] > det.SC005_DELTA_THRESHOLD


def test_sc005_envelope_indeterminate_when_band_straddles():
    # High point flip but tiny n -> the band straddles 0.30, so "get more data".
    draft = det.summarize_scores("d", "instruct_draft", [0.0] * 3)
    rewrite = det.summarize_scores("r", "rewrite", [1.0, 1.0])  # n=2, 1 dropped
    sc = det.sc005_delta([draft, rewrite], imprecision_s=2.0)
    assert sc["verdict"] == "indeterminate"
    assert sc["delta_lower"] < det.SC005_DELTA_THRESHOLD <= sc["delta_upper"]


def test_sc005_envelope_robust_pass_needs_lower_above_threshold():
    # Large n, strong flip -> even the lower bound clears the bar.
    draft = det.summarize_scores("d", "instruct_draft", [0.0] * 50)
    rewrite = det.summarize_scores("r", "rewrite", [1.0] * 40 + [0.0] * 10)
    sc = det.sc005_delta([draft, rewrite], imprecision_s=2.0)
    assert sc["verdict"] == "robust_pass"
    assert sc["delta_lower"] >= det.SC005_DELTA_THRESHOLD


# --- DoD 2: per-text + summary persistence -----------------------------------


def test_write_artifacts_persists_per_text_and_summary(tmp_path):
    draft = det.summarize_scores("draft", "instruct_draft", [0.1, 0.2])
    rewrite = det.summarize_scores("rw", "rewrite", [0.6, 0.8])
    report = det.assemble_report(
        report_id="r", detector="pangram", seed=0, summaries=[draft, rewrite]
    )
    scores = [
        det.DetectorScore("draft", "p0", content_sha256("a"), 0.1),
        det.DetectorScore("draft", "p1", content_sha256("b"), 0.2),
        det.DetectorScore("rw", "p0", content_sha256("c"), 0.6),
        det.DetectorScore("rw", "p1", content_sha256("d"), 0.8),
    ]
    out = tmp_path / "results" / "reports" / "pangram-detect.json"
    summary_path, scores_path = det.write_detection_artifacts(
        report, scores, out_path=out
    )

    # Summary JSON: both set summaries in ONE artifact so SC-005 is one subtraction.
    written = json.loads(summary_path.read_text())
    assert written["n_sets"] == 2
    assert {s["set_id"] for s in written["sets"]} == {"draft", "rw"}
    # draft mean 0.15, rewrite mean 0.7 -> delta 0.55.
    assert written["sc005"]["delta"] == pytest.approx(0.55)

    # Per-text scores JSONL: one row per text, not just the summary.
    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(rows) == 4
    assert rows[0]["pair_id"] == "p0"
    assert rows[0]["human_prob"] == pytest.approx(0.1)
    assert rows[0]["text_sha"] == content_sha256("a")


def test_assemble_report_single_set_carries_sc005_not_computed_caveat():
    # A single-set (or role-mismatched) run cannot compute the SC-005 delta, so
    # the report's sc005 is None AND a sc005_not_computed caveat explains why --
    # the empty delta must not read as a silent omission.
    single = det.summarize_scores("s", "rewrite", [0.5])
    report = det.assemble_report(
        report_id="r", detector="pangram", seed=0, summaries=[single]
    )
    assert report.sc005 is None
    kinds = {c.get("kind") for c in report.caveats if isinstance(c, dict)}
    assert "sc005_not_computed" in kinds


def test_write_artifacts_scores_path_is_sibling_jsonl(tmp_path):
    report = det.assemble_report(
        report_id="r",
        detector="pangram",
        seed=0,
        summaries=[det.summarize_scores("s", "rewrite", [0.5])],
    )
    out = tmp_path / "x-detect.json"
    summary_path, scores_path = det.write_detection_artifacts(
        report, [], out_path=out
    )
    assert summary_path.name == "x-detect.json"
    assert scores_path.name == "x-detect.scores.jsonl"


def test_write_artifacts_scores_path_uses_full_stem_multi_dot(tmp_path):
    # A multi-dot --out must derive the scores sibling from the FULL stem, not a
    # truncated one: a.b.json -> a.b.scores.jsonl, never a.scores.jsonl.
    report = det.assemble_report(
        report_id="r",
        detector="pangram",
        seed=0,
        summaries=[det.summarize_scores("s", "rewrite", [0.5])],
    )
    out = tmp_path / "a.b.json"
    summary_path, scores_path = det.write_detection_artifacts(
        report, [], out_path=out
    )
    assert summary_path.name == "a.b.json"
    assert scores_path.name == "a.b.scores.jsonl"


def test_write_artifacts_rejects_non_json_out(tmp_path):
    report = det.assemble_report(
        report_id="r",
        detector="pangram",
        seed=0,
        summaries=[det.summarize_scores("s", "rewrite", [0.5])],
    )
    with pytest.raises(ValueError, match="must end in .json"):
        det.write_detection_artifacts(report, [], out_path=tmp_path / "report.tar.gz")


def test_write_artifacts_summary_commit_failure_leaves_no_artifact(
    tmp_path, monkeypatch
):
    # IMPORTANT: the two final artifacts are an all-or-nothing pair. If the SECOND
    # commit (the summary os.replace) fails after the FIRST (the scores) landed,
    # the already-committed scores file is rolled back so NEITHER final artifact
    # survives, and no .tmp debris lingers. An orphaned complete-looking summary
    # (or a lone scores file) would otherwise mislead a downstream reader.
    import os as _os

    report = det.assemble_report(
        report_id="r",
        detector="pangram",
        seed=0,
        summaries=[det.summarize_scores("s", "rewrite", [0.5])],
    )
    scores = [det.DetectorScore("s", "p0", content_sha256("a"), 0.5)]
    out = tmp_path / "results" / "reports" / "pangram-detect.json"

    real_replace = _os.replace
    calls = {"n": 0}

    def _replace_fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # the summary commit (scores commit is #1)
            raise OSError("simulated summary-commit failure")
        return real_replace(src, dst)

    monkeypatch.setattr(_os, "replace", _replace_fail_second)

    with pytest.raises(OSError):
        det.write_detection_artifacts(report, scores, out_path=out)

    # Neither final artifact survives.
    assert not out.exists()
    assert not out.with_name("pangram-detect.scores.jsonl").exists()
    # No temp debris left behind.
    debris = [p.name for p in out.parent.iterdir() if p.name.endswith(".tmp")]
    assert debris == []


# --- DoD 3: missing-key exit 3 with ZERO detector calls ----------------------


def test_cli_missing_key_exits_3_before_any_call(tmp_path, monkeypatch, capsys):
    # No PANGRAM_API_KEY set. The key check must fire before any manifest read or
    # client construction -> exit 3, and build_client must never run.
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)

    called = {"build": False}

    def _boom_build(*a, **k):  # pragma: no cover - must not run
        called["build"] = True
        raise AssertionError("build_client was reached despite a missing key")

    monkeypatch.setattr(det, "build_client", _boom_build)

    draft = _write_set(tmp_path, "draft", "draft", "instruct_draft", {"p0": "a"})
    rewrite = _write_set(tmp_path, "rw", "rw", "rewrite", {"p0": "b"})

    out = tmp_path / "o.json"
    rc = cli.main(["detect", "--sets", draft, rewrite, "--out", str(out)])
    assert rc == cli.EXIT_EXTERNAL_DEP == 3
    assert called["build"] is False
    # No artifact written.
    assert not (tmp_path / "o.json").exists()
    err = capsys.readouterr().err
    assert "PANGRAM_API_KEY" in err


def test_cli_gptzero_missing_key_names_its_env_var(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
    s = _write_set(tmp_path, "s", "s", "rewrite", {"p0": "a"})
    rc = cli.main(["detect", "--sets", s, "--detector", "gptzero"])
    assert rc == cli.EXIT_EXTERNAL_DEP
    assert "GPTZERO_API_KEY" in capsys.readouterr().err


# --- DoD 4: cost gate blocks above threshold without --yes -------------------


def _force_low_threshold(monkeypatch):
    """Make cost_preflight gate at $0.00 so any non-empty run trips it without --yes."""
    import functools

    real = det.cost_preflight
    monkeypatch.setattr(
        det, "cost_preflight", functools.partial(real, threshold_usd=0.0)
    )


def test_cli_cost_gate_blocks_without_yes_zero_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    _force_low_threshold(monkeypatch)

    exploding = ExplodingClient()
    monkeypatch.setattr(det, "build_client", lambda *a, **k: exploding)

    draft = _write_set(tmp_path, "draft", "draft", "instruct_draft", {"p0": "a"})
    rewrite = _write_set(tmp_path, "rw", "rw", "rewrite", {"p0": "b"})
    out = tmp_path / "o.json"

    rc = cli.main(["detect", "--sets", draft, rewrite, "--out", str(out)])
    assert rc == cli.EXIT_VALIDATION == 2
    assert exploding.calls == 0  # gate fired before any detector call
    assert not out.exists()
    err = capsys.readouterr().err
    # The per-set text count was printed before the block.
    assert "1 texts" in err or "has 1 texts" in err


def test_cli_below_threshold_proceeds(tmp_path, monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")

    draft = _write_set(tmp_path, "draft", "draft", "instruct_draft", {"p0": "aaa"})
    rewrite = _write_set(tmp_path, "rw", "rw", "rewrite", {"p0": "bbb"})

    client = MappingClient({"aaa": 0.1, "bbb": 0.9})
    monkeypatch.setattr(det, "build_client", lambda *a, **k: client)

    out = tmp_path / "results" / "reports" / "detect.json"
    # No --yes, but the default threshold ($1.00) is far above a 2-text estimate.
    rc = cli.main(
        ["--seed", "5", "detect", "--sets", draft, rewrite, "--out", str(out)]
    )
    assert rc == cli.EXIT_SUCCESS == 0
    assert client.calls == ["aaa", "bbb"]

    written = json.loads(out.read_text())
    assert written["seed"] == 5
    assert written["sc005"]["delta"] == pytest.approx(0.8)  # 0.9 - 0.1
    assert written["sc005"]["passed"] is True
    # Per-text scores persisted beside the summary.
    scores_path = out.with_suffix(".scores.jsonl")
    rows = scores_path.read_text().splitlines()
    assert len(rows) == 2


# --- CLI loud failure on a detector call error -------------------------------


def test_cli_detector_failure_exits_3_no_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    monkeypatch.setattr(det, "build_client", lambda *a, **k: FailingClient(fail_on=1))

    s = _write_set(tmp_path, "s", "s", "rewrite", {"p0": "a", "p1": "b"})
    out = tmp_path / "o.json"
    rc = cli.main(["detect", "--sets", s, "--out", str(out)])
    assert rc == cli.EXIT_EXTERNAL_DEP == 3
    # A failed call must not leave a report that reads as real.
    assert not out.exists()
    assert not out.with_suffix(".scores.jsonl").exists()
    assert "detector call failed" in capsys.readouterr().err


def test_cli_bad_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    monkeypatch.setattr(det, "build_client", lambda *a, **k: ExplodingClient())
    rc = cli.main(["detect", "--sets", str(tmp_path / "missing.json")])
    assert rc == cli.EXIT_VALIDATION == 2


def test_cli_default_out_path_under_results_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    s = _write_set(tmp_path, "s", "s", "rewrite", {"p0": "aaa"})
    monkeypatch.setattr(
        det, "build_client", lambda *a, **k: MappingClient({"aaa": 0.7})
    )
    rc = cli.main(["detect", "--sets", s])
    assert rc == cli.EXIT_SUCCESS
    # Default lands under results/reports/.
    assert (tmp_path / "results" / "reports" / "pangram-detect.json").exists()


# --- IMPORTANT 2: SC-005 uncomputable on a run that requested it -------------


def test_cli_two_rewrites_exits_non_ok_naming_reason(tmp_path, monkeypatch, capsys):
    # Two sets were given, so a delta was requested; both are rewrites, so it
    # cannot be computed. The command must NOT report ok after paid spend: a
    # non-ok status naming the reason AND a distinct non-zero exit code, while the
    # per-set artifacts are still written.
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    a = _write_set(tmp_path, "a", "a", "rewrite", {"p0": "aaa"})
    b = _write_set(tmp_path, "b", "b", "rewrite", {"p0": "bbb"})
    monkeypatch.setattr(
        det, "build_client", lambda *a, **k: MappingClient({"aaa": 0.6, "bbb": 0.7})
    )
    out = tmp_path / "o.json"
    rc = cli.main(["detect", "--sets", a, b, "--out", str(out)])

    assert rc != cli.EXIT_SUCCESS
    assert rc == cli.EXIT_SC005_NOT_COMPUTED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "sc005_not_computed"
    assert payload["sc005"] is None
    # Artifacts are still written (the per-set summaries remain useful).
    assert out.exists()
    assert out.with_suffix(".scores.jsonl").exists()


def test_cli_single_set_null_sc005_still_ok(tmp_path, monkeypatch, capsys):
    # A single-set run never requested a delta, so its null sc005 stays ok (0).
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    s = _write_set(tmp_path, "s", "s", "rewrite", {"p0": "aaa"})
    monkeypatch.setattr(
        det, "build_client", lambda *a, **k: MappingClient({"aaa": 0.7})
    )
    out = tmp_path / "o.json"
    rc = cli.main(["detect", "--sets", s, "--out", str(out)])
    assert rc == cli.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["sc005"] is None


# --- IMPORTANT 3: duplicate pair_id fails loudly before any spend ------------


def test_cli_duplicate_pair_id_exits_2_zero_calls(tmp_path, monkeypatch, capsys):
    # A manifest listing the same pair_id twice would double-score and double-pay.
    # Detect it before counting/scoring and fail loudly (exit 2) with ZERO
    # detector calls.
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    exploding = ExplodingClient()
    monkeypatch.setattr(det, "build_client", lambda *a, **k: exploding)

    # Hand-write a manifest with a duplicated pair_id (bypasses _write_set, which
    # dedups via a dict).
    manifest = TextSet(
        set_id="s",
        role="rewrite",
        corpus="fineweb",
        pair_ids=["p0", "p0"],
        provenance={"texts_path": "s.jsonl"},
    )
    manifest_path = tmp_path / "s.manifest.json"
    write_json(manifest, manifest_path)
    with (tmp_path / "s.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pair_id": "p0", "text": "aaa"}) + "\n")

    out = tmp_path / "o.json"
    rc = cli.main(["detect", "--sets", str(manifest_path), "--out", str(out)])
    assert rc == cli.EXIT_VALIDATION == 2
    assert exploding.calls == 0  # no detector call before the validation failure
    assert not out.exists()
    err = capsys.readouterr().err
    assert "duplicate pair_id" in err
    assert "p0" in err


# --- NIT 2: --out must end in .json ------------------------------------------


def test_cli_non_json_out_exits_2_zero_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    exploding = ExplodingClient()
    monkeypatch.setattr(det, "build_client", lambda *a, **k: exploding)
    s = _write_set(tmp_path, "s", "s", "rewrite", {"p0": "aaa"})
    rc = cli.main(["detect", "--sets", s, "--out", str(tmp_path / "report.tar.gz")])
    assert rc == cli.EXIT_VALIDATION == 2
    assert exploding.calls == 0
    assert "must end in .json" in capsys.readouterr().err


# --- IMPORTANT 4(5): multi-set, later set fails -> exit 3, NO artifacts -------


def test_cli_later_set_failure_exits_3_no_artifacts(tmp_path, monkeypatch, capsys):
    # An earlier set scores clean, then a later set's call fails. The whole run
    # must fail loudly (exit 3) and leave NO artifacts -- a partial report over
    # only the clean set would read as a real, complete run. Locks the current
    # no-leak behavior.
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")

    class _FailOnSecondSet:
        # Clean for the first set's text, then raises on the second set's text.
        def __init__(self) -> None:
            self.calls: list[str] = []

        def score_text(self, text: str) -> float:
            self.calls.append(text)
            if text == "bbb":
                raise ConnectionError("simulated failure on the second set")
            return 0.7

    client = _FailOnSecondSet()
    monkeypatch.setattr(det, "build_client", lambda *a, **k: client)

    draft = _write_set(tmp_path, "draft", "draft", "instruct_draft", {"p0": "aaa"})
    rewrite = _write_set(tmp_path, "rw", "rw", "rewrite", {"p0": "bbb"})
    out = tmp_path / "o.json"

    rc = cli.main(["detect", "--sets", draft, rewrite, "--out", str(out)])
    assert rc == cli.EXIT_EXTERNAL_DEP == 3
    # The clean set was scored, then the failure aborted the run.
    assert client.calls == ["aaa", "bbb"]
    # No artifact from the partially-scored run.
    assert not out.exists()
    assert not out.with_suffix(".scores.jsonl").exists()
    assert "detector call failed" in capsys.readouterr().err


# --- PangramClient response parsing (real SDK shape, surfaced by issue #16) ----


class _FakePangramSDK:
    """Stand-in for the real pangram SDK client: predict() returns the documented
    fraction_* dict, so we lock the response parsing without a network call."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def predict(self, text, model=None):
        self.calls.append((text, model))
        return self._response


def test_pangram_client_reads_fraction_human():
    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK({"fraction_human": 0.83, "fraction_ai": 0.17})
    assert c.score_text("x") == 0.83
    # model is passed explicitly (deprecation: omitting it is rejected after
    # 2026-09-30) and defaults to the web-app model, not the older "default".
    assert c._client.calls == [("x", detector.DEFAULT_PANGRAM_MODEL)]
    assert detector.DEFAULT_PANGRAM_MODEL == "pangram-4"


def test_pangram_client_honours_explicit_model():
    from dehip import detector

    c = detector.PangramClient(model="default")
    c._client = _FakePangramSDK({"fraction_human": 1.0})
    c.score_text("x")
    assert c._client.calls == [("x", "default")]


def test_build_client_threads_model_to_pangram():
    from dehip import detector

    c = detector.build_client("pangram", api_key="k", model="default")
    assert isinstance(c, detector.PangramClient)
    assert c._model == "default"
    # None -> the web-app default
    assert detector.build_client("pangram", api_key="k")._model == "pangram-4"


def test_pangram_client_falls_back_to_one_minus_fraction_ai():
    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK({"fraction_ai": 0.30})  # no fraction_human
    assert abs(c.score_text("x") - 0.70) < 1e-9


def test_pangram_client_missing_fields_fails_loudly():
    import pytest

    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK({"unexpected": 1})
    with pytest.raises(detector.DetectorCallError):
        c.score_text("x")


class _AttrResponse:
    """An object (not a dict) exposing fraction_* as attributes -- the shape the
    real pangram SDK returns, which exercises the getattr branch of _field."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_pangram_client_reads_object_response():
    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK(_AttrResponse(fraction_human=0.62, fraction_ai=0.38))
    assert c.score_text("x") == 0.62


def test_pangram_client_object_response_fraction_ai_fallback():
    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK(_AttrResponse(fraction_ai=0.25))  # no fraction_human
    assert abs(c.score_text("x") - 0.75) < 1e-9


def test_pangram_client_invalid_fraction_human_fails():
    import pytest

    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK({"fraction_human": "not-a-number"})
    with pytest.raises(detector.DetectorCallError):
        c.score_text("x")


def test_pangram_client_invalid_fraction_ai_fails():
    import pytest

    from dehip import detector

    c = detector.PangramClient()
    c._client = _FakePangramSDK({"fraction_ai": "not-a-number"})
    with pytest.raises(detector.DetectorCallError):
        c.score_text("x")


def test_pangram_client_predict_error_is_wrapped():
    import pytest

    from dehip import detector

    class _Boom:
        def predict(self, text, model=None):
            raise RuntimeError("transport blew up")

    c = detector.PangramClient()
    c._client = _Boom()
    with pytest.raises(detector.DetectorCallError):
        c.score_text("x")
