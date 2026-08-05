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
    # Even count: median is the mean of the two central values (0.4, 0.6 -> 0.5).
    summary = det.summarize_scores("s", "rewrite", [0.1, 0.4, 0.6, 0.9])
    assert summary.median == pytest.approx(0.5)
    assert summary.mean == pytest.approx(0.5)


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
