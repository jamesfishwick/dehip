"""Integration tests for `dehip self-check` (issue #11, FR-003, SC-001).

Every test runs the REAL self-check path (dehip.self_check.run_self_check ->
dehip.report.score) with stub seams: a deterministic stub embedder (hash-seeded
4-D standard normal, identical to test_report's), a whitespace stub tokenizer,
and a call-counting stub judge. No model is downloaded and no network call is
made -- the same stub instruments the noise bounds were derived over.

The suite locks the invariants the adversarial review probes:

- The self-check PASSES on the stub smoke corpus (SC-001): MMD/token-L2 within
  the documented bounds, JMQ win-rate in [0.45, 0.55].
- It EXITS 4 (raises SelfCheckOutOfBounds) when the bounds are artificially
  tightened, naming which bound was exceeded -- fail loudly, never a silent pass.
- ``--skip-jmq`` spends ZERO judge calls: a judge stub that RAISES if called
  proves no judge is even constructed on that path.
- The half-vs-half split is seeded, reproducible, and disjoint (no pair leaks
  into both halves). Odd-N behavior is defined and tested (one pair dropped).
- The scoring goes through the real report.py path, so a metric regression is
  actually caught (asserted via a monkeypatched MMD that returns a large value).
"""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest

from dehip import cli
from dehip.metrics.bounds import (
    STUB_INSTRUMENT_BOUNDS,
    StubInstrumentBounds,
    jmq_scaled_window,
)
from dehip.metrics.embeddings import EmbeddingCache
from dehip.schemas import TextSet, write_json
from dehip.self_check import (
    SelfCheckIntegrityError,
    SelfCheckOutOfBounds,
    load_reference_set,
    run_self_check,
    split_pairs,
)
from dehip.validate import InputSetValidationError

# --- Stubs (mirror test_report / test_score_cli) -----------------------------


class StubEmbedder:
    """Hash-seeded 4-D standard-normal embedder; the derivation stub. No network."""

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
    tokenizer_id = "stub-tokenizer"

    def tokenize(self, text: str):
        return text.split()


class TargetedJudge:
    """A fair judge that makes the model win on exactly the even-indexed pairs.

    A fair judge on two same-distribution halves wins about half the comparisons
    (SC-001). At 25 pairs a *random* judge has too much binomial noise to land
    inside 45-55% every run, so this stub makes the outcome deterministic and
    in-window without faking the metric: it reads the synthetic pair index the
    self-check embeds in each prompt (``self-check pair sc-N``), reconstructs the
    seeded A/B order via the public :func:`assign_order`, and answers so the model
    wins iff the index is even. Over 25 pairs that is 13 wins -> 0.52, inside the
    window, on every seed. Counts its calls so the recompute/skip tests can assert
    call counts.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        from dehip.metrics.jmq import assign_order

        self.calls += 1
        index = int(re.search(r"sc-(\d+)", rendered_prompt).group(1))
        model_should_win = index % 2 == 0
        order = assign_order(f"sc-{index}", self.seed)
        # The model is candidate A when model_first, B when human_first. Return
        # its slot to make it win, the other slot to make it lose.
        if order == "model_first":
            return "A" if model_should_win else "B"
        return "B" if model_should_win else "A"


class ExplodingJudge:
    """A judge that RAISES if judged. Proves --skip-jmq constructs no judge."""

    def judge(self, rendered_prompt: str, *, model: str) -> str:  # pragma: no cover
        raise AssertionError(
            "judge was called on a --skip-jmq run; no judge spend is allowed"
        )


class RandomFairJudge:
    """A genuinely random p=0.5 judge (indifferent between the two candidates).

    Unlike TargetedJudge, this does NOT rig the outcome: it flips a seeded coin
    per (pair, dimension) with no reference to the model/human slot, so the model
    win-rate is a real Binomial(n, 0.5) sample. It proves the SCALED window is
    reachable by an honest fair judge (not just a rigged 0.52), across several
    seeds, at n=25 -- the reachability CRITICAL 1 requires.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        return "A" if self._rng.random() < 0.5 else "B"


class AlwaysModelJudge:
    """A broken judge that ALWAYS makes the model win (win-rate 1.0).

    Reconstructs the model's slot from the seeded A/B order and always picks it,
    so every valid comparison is a model win. Trips the scaled window from above.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        from dehip.metrics.jmq import assign_order

        index = int(re.search(r"sc-(\d+)", rendered_prompt).group(1))
        order = assign_order(f"sc-{index}", self.seed)
        return "A" if order == "model_first" else "B"


class AlwaysHumanJudge:
    """A broken judge that ALWAYS makes the human win (model win-rate 0.0).

    The inverse of AlwaysModelJudge: always picks the human's slot, so the model
    never wins. Trips the scaled window from below.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        from dehip.metrics.jmq import assign_order

        index = int(re.search(r"sc-(\d+)", rendered_prompt).group(1))
        order = assign_order(f"sc-{index}", self.seed)
        # The human is the OTHER slot from the model.
        return "B" if order == "model_first" else "A"


class AllInvalidJudge:
    """A broken judge whose every reply fails to parse (choice='invalid').

    Every verdict is excluded-and-counted, so the valid-comparison count collapses
    to n=0 and the win-rate is None. A non-skip run over this judge must FAIL
    LOUDLY (CRITICAL 2): a broken judge that produced zero usable verdicts must
    not be indistinguishable from a --skip-jmq run.
    """

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        return "the model is clearly better in every dimension"


# --- Fixtures ----------------------------------------------------------------


def _write_reference(tmp_path, n, *, name="ref", set_id="human-smoke"):
    """Write an n-pair human_reference manifest + its sibling {pair_id, text} JSONL.

    The texts imitate the smoke corpus: varied multi-sentence human-like prose so
    the two halves have overlapping-but-not-identical vocabulary (a realistic
    same-distribution split, not a trivial identical-text case).
    """
    subjects = [
        "the river", "a small town", "the old library", "her grandmother",
        "the market", "the mountain trail", "the winter storm", "the harbor",
        "an empty theater", "the garden", "the night sky", "the train station",
    ]
    verbs = [
        "changed slowly over the years", "held a strange quiet",
        "drew people from far away", "kept its secrets well",
        "smelled of rain and dust", "seemed larger at dusk",
    ]
    pairs = []
    for i in range(n):
        subj = subjects[i % len(subjects)]
        verb = verbs[i % len(verbs)]
        extra = " ".join(f"word{(i * 7 + j) % 40}" for j in range((i % 9) + 4))
        text = (
            f"{subj.capitalize()} {verb}. It was a place many remembered long "
            f"after they left. {extra}. Nothing about it was ordinary."
        )
        pairs.append((f"{set_id}-{i}", text))

    manifest = TextSet(
        set_id=set_id,
        role="human_reference",
        corpus="fineweb",
        pair_ids=[pid for pid, _ in pairs],
        provenance={"texts_path": f"{name}.jsonl", "count": n},
    )
    manifest_path = tmp_path / f"{name}.manifest.json"
    write_json(manifest, manifest_path)
    with (tmp_path / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for pid, text in pairs:
            fh.write(json.dumps({"pair_id": pid, "text": text}) + "\n")
    return str(manifest_path)


def _cache(tmp_path, embedder=None):
    return EmbeddingCache(embedder or StubEmbedder(), cache_dir=tmp_path / "emb")


# --- DoD: PASSES on the stub smoke corpus (SC-001) ---------------------------


def test_self_check_passes_on_stub_smoke_corpus(tmp_path):
    """SC-001: MMD/token-L2 within bounds and a 45-55% JMQ win-rate, no raise."""
    manifest = _write_reference(tmp_path, 50)
    judge = TargetedJudge(seed=0)
    result = run_self_check(
        manifest,
        seed=0,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=judge,
        verdicts_path=str(tmp_path / "verdicts.jsonl"),
    )
    assert result.passed
    assert result.violations == []
    # Within the documented stub-instrument bounds.
    assert result.mmd <= STUB_INSTRUMENT_BOUNDS.mmd_max
    assert result.token_l2 <= STUB_INSTRUMENT_BOUNDS.token_l2_max
    # JMQ win-rate in the 45-55% window (a fair judge on same-distribution halves).
    assert result.jmq_win_rate is not None
    assert 0.45 <= result.jmq_win_rate <= 0.55
    # The judge was actually consulted (one call per pair per dimension).
    assert judge.calls == result.half_size * 6


def test_self_check_scores_through_report_path(tmp_path):
    """The result carries a real MetricReport from report.score (not a shortcut)."""
    from dehip.schemas import MetricReport

    manifest = _write_reference(tmp_path, 20)
    result = run_self_check(
        manifest,
        seed=1,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
    )
    assert isinstance(result.report, MetricReport)
    # MMD and token-L2 both actually ran (real values, not the NaN sentinel).
    assert not np.isnan(result.report.mmd)
    assert not np.isnan(result.report.token_l2)
    # The report's compared block names the two halves of one set.
    assert result.report.compared["candidate_set"].endswith("#half-a")
    assert result.report.compared["reference_set"].endswith("#half-b")


# --- DoD: EXITS 4 when bounds are artificially tightened ---------------------


def test_tightened_bounds_raise_out_of_bounds(tmp_path):
    """Tightening the bounds to ~0 forces a violation -> SelfCheckOutOfBounds."""
    manifest = _write_reference(tmp_path, 40)
    impossible = StubInstrumentBounds(mmd_max=-1.0, token_l2_max=0.0)
    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            skip_jmq=True,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
            bounds=impossible,
        )
    # Fail loudly: the message names which bound was exceeded and by how much.
    message = str(excinfo.value)
    assert "token-L2" in message
    assert "exceeds" in message
    assert excinfo.value.violations  # machine-readable list present


def test_cli_self_check_exits_4_on_tightened_bounds(tmp_path, monkeypatch):
    """DoD: the CLI exits 4 when bounds are tightened, with no silent pass.

    The self-check runs through cli.main. The embedder/tokenizer are stubbed and
    the bounds are monkeypatched to impossible values so a real (in-bounds) run is
    forced out of bounds, proving the exit-4 path end to end.
    """
    manifest = _write_reference(tmp_path, 40)
    _patch_stub_instruments(monkeypatch)
    # Force the in-force bounds to impossible values. This run uses the stub
    # embedder, so documented() selects STUB_INSTRUMENT_BOUNDS (not the real set);
    # patch that constant to drive the run out of bounds and prove the exit-4 path.
    monkeypatch.setattr(
        "dehip.metrics.bounds.STUB_INSTRUMENT_BOUNDS",
        StubInstrumentBounds(mmd_max=-1.0, token_l2_max=0.0),
    )

    rc = cli.main(["self-check", "--reference", manifest, "--skip-jmq"])
    assert rc == cli.EXIT_SELF_CHECK == 4


def test_cli_self_check_passes_exits_0(tmp_path, monkeypatch):
    """The happy path through the CLI: in-bounds run exits 0 with a JSON summary."""
    manifest = _write_reference(tmp_path, 50)
    _patch_stub_instruments(monkeypatch)

    rc = cli.main(["self-check", "--reference", manifest, "--skip-jmq"])
    assert rc == cli.EXIT_SUCCESS == 0


def test_metric_regression_is_caught(tmp_path, monkeypatch):
    """A metric regression through the real path trips the bound (exit 4 territory).

    Monkeypatch the MMD computation report.score reaches to return a large value
    (a stand-in for a metric bug). Because self-check scores through report.score,
    the regression surfaces as an out-of-bounds violation rather than passing.
    """
    from dehip.metrics import mmd as mmd_mod

    manifest = _write_reference(tmp_path, 30)

    def _broken_mmd(x, y, bandwidth=None):
        return mmd_mod.MMDResult(mmd2=0.9, bandwidth=1.0)  # far above mmd_max

    monkeypatch.setattr("dehip.report.mmd_mod.mmd2_unbiased", _broken_mmd)

    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            skip_jmq=True,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
        )
    assert any("MMD" in v for v in excinfo.value.violations)


# --- DoD: --skip-jmq spends ZERO judge calls ---------------------------------


def test_skip_jmq_makes_zero_judge_calls(tmp_path):
    """A judge that RAISES if called proves --skip-jmq never touches a judge."""
    manifest = _write_reference(tmp_path, 30)
    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=ExplodingJudge(),  # would raise if judged
    )
    # No judge was called; the run still produced MMD + token-L2.
    assert result.jmq_win_rate is None
    assert not np.isnan(result.report.mmd)
    assert not np.isnan(result.report.token_l2)


def test_skip_jmq_report_has_no_jmq_block(tmp_path):
    manifest = _write_reference(tmp_path, 20)
    result = run_self_check(
        manifest,
        seed=2,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
    )
    assert result.report.jmq == {}


def test_cli_skip_jmq_constructs_no_judge(tmp_path, monkeypatch):
    """Through the CLI, --skip-jmq must not construct the OpenAI judge at all.

    OpenAIJudgeClient is patched to a factory that raises on construction; a
    --skip-jmq run must exit 0 without ever calling it.
    """
    manifest = _write_reference(tmp_path, 40)
    _patch_stub_instruments(monkeypatch)

    def _no_judge(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("a judge was constructed on a --skip-jmq run")

    monkeypatch.setattr("dehip.metrics.jmq.OpenAIJudgeClient", _no_judge)

    rc = cli.main(["self-check", "--reference", manifest, "--skip-jmq"])
    assert rc == cli.EXIT_SUCCESS


# --- Split: seeded, reproducible, disjoint, odd-N defined --------------------


def test_split_is_disjoint():
    """No pair_id leaks into both halves (would let a text score against itself)."""
    ids = [f"p{i}" for i in range(20)]
    half_a, half_b, dropped = split_pairs(ids, seed=0)
    assert set(half_a).isdisjoint(set(half_b))
    assert dropped is None
    assert len(half_a) == len(half_b) == 10
    # The two halves partition the whole set (even N).
    assert set(half_a) | set(half_b) == set(ids)


def test_split_is_reproducible_from_seed():
    ids = [f"p{i}" for i in range(20)]
    a1, b1, d1 = split_pairs(ids, seed=42)
    a2, b2, d2 = split_pairs(ids, seed=42)
    assert (a1, b1, d1) == (a2, b2, d2)


def test_split_differs_across_seeds():
    ids = [f"p{i}" for i in range(20)]
    a1, _, _ = split_pairs(ids, seed=1)
    a2, _, _ = split_pairs(ids, seed=2)
    assert a1 != a2  # different seed -> different partition


def test_odd_n_drops_one_pair_deterministically():
    """Odd N: exactly one pair dropped, halves equal-sized, drop reproducible."""
    ids = [f"p{i}" for i in range(21)]
    half_a, half_b, dropped = split_pairs(ids, seed=7)
    assert dropped is not None
    assert len(half_a) == len(half_b) == 10
    # The dropped pair is in neither half.
    assert dropped not in set(half_a)
    assert dropped not in set(half_b)
    # All 21 ids are accounted for: 10 + 10 + 1 dropped, no overlap.
    assert set(half_a) | set(half_b) | {dropped} == set(ids)
    # Deterministic drop given the seed.
    _, _, dropped2 = split_pairs(ids, seed=7)
    assert dropped2 == dropped


def test_odd_n_self_check_records_dropped_pair(tmp_path):
    """An odd-sized reference set self-checks and records the dropped pair id."""
    manifest = _write_reference(tmp_path, 41)
    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
    )
    assert result.half_size == 20  # 41 -> drop 1 -> 20 + 20
    assert result.dropped_pair_id is not None


def test_too_few_pairs_rejected(tmp_path):
    """Fewer than 2 * min_n pairs cannot form two scorable halves -> validation."""
    manifest = _write_reference(tmp_path, 3)  # < 2 * DEFAULT_MIN_N (4)
    with pytest.raises(InputSetValidationError):
        split_pairs([f"p{i}" for i in range(3)], seed=0)
    with pytest.raises(InputSetValidationError):
        run_self_check(
            manifest,
            seed=0,
            skip_jmq=True,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
        )


# --- Reference loading -------------------------------------------------------


def test_load_reference_set_reads_manifest_and_texts(tmp_path):
    manifest = _write_reference(tmp_path, 10)
    pair_ids, texts, set_id = load_reference_set(manifest)
    assert len(pair_ids) == 10
    assert set(texts.keys()) == set(pair_ids)
    assert set_id == "human-smoke"


def test_load_reference_set_rejects_manifest_texts_mismatch(tmp_path):
    """A manifest whose ids differ from its texts file is rejected before scoring."""
    manifest = TextSet(
        set_id="s",
        role="human_reference",
        corpus="fineweb",
        pair_ids=["a", "b", "c"],
        provenance={"texts_path": "t.jsonl"},
    )
    write_json(manifest, tmp_path / "m.manifest.json")
    with (tmp_path / "t.jsonl").open("w", encoding="utf-8") as fh:
        # Texts file is missing 'c' and carries an extra 'z'.
        for pid in ("a", "b", "z"):
            fh.write(json.dumps({"pair_id": pid, "text": "some human text"}) + "\n")
    with pytest.raises(InputSetValidationError):
        load_reference_set(str(tmp_path / "m.manifest.json"))


# --- No downloads / no network (verified structurally) -----------------------


def test_no_real_model_constructed(tmp_path, monkeypatch):
    """Guard: a real embedder/tokenizer construction would fail this test loudly.

    Patch the real instruments to raise; the stub-seam run must still succeed,
    proving the self-check path never falls through to a download.
    """
    manifest = _write_reference(tmp_path, 20)

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("a real model was constructed in an integration test")

    monkeypatch.setattr("dehip.metrics.embeddings.TransformersEmbedder", _boom)
    monkeypatch.setattr("dehip.metrics.token_l2.Qwen3Tokenizer", _boom)

    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),  # stub embedder
        tokenizer=StubTokenizer(),  # stub tokenizer
    )
    assert result.passed


# --- Shared CLI stub-instrument patch ----------------------------------------


def _patch_stub_instruments(monkeypatch):
    """Swap the real embedder/tokenizer constructors for deterministic stubs.

    self-check's CLI handler builds TransformersEmbedder() and score() builds the
    Qwen3 tokenizer lazily; patch both so the CLI path runs with no download.
    """
    monkeypatch.setattr(
        "dehip.metrics.embeddings.TransformersEmbedder", lambda *a, **k: StubEmbedder()
    )
    monkeypatch.setattr(
        "dehip.metrics.token_l2.Qwen3Tokenizer", lambda *a, **k: StubTokenizer()
    )


# --- CRITICAL 1: JMQ window scaled to n, reachable by an honest fair judge ----


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_random_fair_judge_passes_scaled_window(tmp_path, seed):
    """A genuinely random p=0.5 judge lands inside the scaled window at n=25.

    CRITICAL 1: the fixed [0.45, 0.55] window is unsatisfiable at the smoke tier
    (~0.31 pass under Binomial(25, 0.5)). The scaled window 0.5 +/- Z_JMQ *
    sqrt(0.25/n) must PASS an honest fair judge -- not just the rigged 0.52 of
    TargetedJudge -- across several seeds. This proves reachability, not rigging.
    """
    manifest = _write_reference(tmp_path, 50)  # 25 vs 25 -> n=25 comparisons
    judge = RandomFairJudge(seed=seed)
    result = run_self_check(
        manifest,
        seed=seed,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        judge_client=judge,
        verdicts_path=str(tmp_path / f"verdicts-{seed}.jsonl"),
    )
    assert result.passed, result.violations
    # The window was scaled to the actual valid-comparison count and recorded.
    assert result.jmq_n == 25
    lo, hi = jmq_scaled_window(25)
    assert result.jmq_window == (lo, hi)
    assert lo <= result.jmq_win_rate <= hi


def test_scaled_window_widens_at_small_n_and_narrows_at_large_n():
    """The window is centered on 0.5, wide at n=25, narrowing toward 45-55%."""
    lo25, hi25 = jmq_scaled_window(25)
    # At n=25 the window is [0.20, 0.80] -- far wider than [0.45, 0.55].
    assert lo25 == pytest.approx(0.20, abs=1e-9)
    assert hi25 == pytest.approx(0.80, abs=1e-9)
    # As n grows the window narrows toward the documented [0.45, 0.55] target.
    lo200, hi200 = jmq_scaled_window(200)
    assert 0.45 > lo200 > lo25  # tighter than n=25, still looser than 0.45
    assert 0.55 < hi200 < hi25
    # Centered on 0.5 for any n.
    assert (lo200 + hi200) / 2 == pytest.approx(0.5, abs=1e-9)
    # An empty n has no defined window -> loud error, never a silent pass.
    with pytest.raises(ValueError):
        jmq_scaled_window(0)


def test_always_model_judge_trips_window_high(tmp_path):
    """A judge with win-rate 1.0 trips exit 4 with a win-rate violation."""
    manifest = _write_reference(tmp_path, 50)
    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
            judge_client=AlwaysModelJudge(seed=0),
            verdicts_path=str(tmp_path / "verdicts.jsonl"),
        )
    # The violation names the JMQ win-rate window (loud on real breakage).
    assert any(
        "win-rate" in v and "above" in v for v in excinfo.value.violations
    ), excinfo.value.violations


def test_always_human_judge_trips_window_low(tmp_path):
    """A judge with model win-rate 0.0 trips exit 4 with a win-rate violation."""
    manifest = _write_reference(tmp_path, 50)
    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
            judge_client=AlwaysHumanJudge(seed=0),
            verdicts_path=str(tmp_path / "verdicts.jsonl"),
        )
    assert any(
        "win-rate" in v and "below" in v for v in excinfo.value.violations
    ), excinfo.value.violations


# --- CRITICAL 2: all-invalid verdicts on a non-skip run fail loudly -----------


def test_all_invalid_verdicts_fail_loudly(tmp_path):
    """A judge producing ZERO valid verdicts on a non-skip run must NOT pass.

    CRITICAL 2: jmq.py returns win_rate None when every verdict is invalid. That
    must be a loud non-zero failure naming the reason, never a silent exit-0 pass
    indistinguishable from --skip-jmq.
    """
    manifest = _write_reference(tmp_path, 50)
    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
            judge_client=AllInvalidJudge(),
            verdicts_path=str(tmp_path / "verdicts.jsonl"),
        )
    assert any(
        "no valid win-rate" in v for v in excinfo.value.violations
    ), excinfo.value.violations


def test_cli_all_invalid_verdicts_exits_nonzero(tmp_path, monkeypatch):
    """Through the CLI, an all-invalid non-skip run exits 4 (not 0)."""
    manifest = _write_reference(tmp_path, 50)
    _patch_stub_instruments(monkeypatch)
    # A non-skip run whose judge always returns an unparseable reply.
    monkeypatch.setattr(
        "dehip.metrics.jmq.OpenAIJudgeClient", lambda *a, **k: AllInvalidJudge()
    )

    rc = cli.main(["self-check", "--reference", manifest])
    assert rc == cli.EXIT_SELF_CHECK == 4


def test_skip_jmq_is_not_flagged_as_all_invalid(tmp_path):
    """--skip-jmq (jmq_win_rate None by design) is a pass, NOT an all-invalid fail.

    Distinguishes skip-requested from ran-but-all-invalid: a skipped run has
    jmq_win_rate None and MUST pass; only a REQUESTED-but-empty run fails.
    """
    manifest = _write_reference(tmp_path, 50)
    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
    )
    assert result.passed
    assert result.jmq_win_rate is None
    assert result.jmq_n is None  # skipped, not zero-valid


# --- IMPORTANT 1: MMD lower bound traps a negative-driving regression ---------


def test_negative_mmd_regression_trips_lower_bound(tmp_path, monkeypatch):
    """A strongly-negative MMD (sign flip / broken kernel) trips a violation.

    IMPORTANT 1: unbiased MMD^2 straddles zero. Without a lower bound, a
    regression driving MMD to -5.0 passes (`-5.0 <= 0.10`). The lower bound traps
    it.
    """
    from dehip.metrics import mmd as mmd_mod

    manifest = _write_reference(tmp_path, 30)

    def _negative_mmd(x, y, bandwidth=None):
        return mmd_mod.MMDResult(mmd2=-5.0, bandwidth=1.0)  # far below mmd_min

    monkeypatch.setattr("dehip.report.mmd_mod.mmd2_unbiased", _negative_mmd)

    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            skip_jmq=True,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
        )
    assert any(
        "MMD" in v and "below" in v for v in excinfo.value.violations
    ), excinfo.value.violations


def test_nan_mmd_trips_upper_bound(tmp_path, monkeypatch):
    """A NaN MMD trips a violation, locking the `not (x <= max)` form.

    IMPORTANT 1: `not (nan <= max)` is True (NaN comparisons are False), so NaN
    trips the upper check. A refactor to `mmd > max` would let NaN pass; this
    test forbids that.
    """
    from dehip.metrics import mmd as mmd_mod

    manifest = _write_reference(tmp_path, 30)

    def _nan_mmd(x, y, bandwidth=None):
        return mmd_mod.MMDResult(mmd2=float("nan"), bandwidth=1.0)

    monkeypatch.setattr("dehip.report.mmd_mod.mmd2_unbiased", _nan_mmd)

    with pytest.raises(SelfCheckOutOfBounds) as excinfo:
        run_self_check(
            manifest,
            seed=0,
            skip_jmq=True,
            embed_cache=_cache(tmp_path),
            tokenizer=StubTokenizer(),
        )
    assert any("MMD" in v for v in excinfo.value.violations), excinfo.value.violations


# --- IMPORTANT 2: bounds provenance is internally consistent ------------------


def test_derivation_seeds_match_script_range():
    """DERIVATION_SEEDS is 0..23, matching scripts/derive_bounds.py's range(24)."""
    from dehip.metrics.bounds import DERIVATION_SEEDS

    assert DERIVATION_SEEDS == tuple(range(24))


def test_documented_bounds_clear_observed_max_over_derivation_seeds(tmp_path):
    """Re-derive over 0..23 and confirm the 0.10 bounds clear the true max.

    IMPORTANT 2: the documented ranges (MMD abs-max 0.02948646, token-L2 max
    0.04379749) were derived over seeds 0..23. This re-runs the real self-check
    path over those seeds with the stub instruments and asserts the chosen 0.10
    bounds still bracket the observed noise with margin -- so the constant, the
    docstring, and the script agree on the seed set actually used.
    """
    from dehip.metrics.bounds import DERIVATION_SEEDS

    manifest = _write_reference(tmp_path, 50, set_id="stub-smoke-50")
    open_bounds = StubInstrumentBounds(
        mmd_max=float("inf"), mmd_min=float("-inf"), token_l2_max=float("inf")
    )
    mmds = []
    token_l2s = []
    for seed in DERIVATION_SEEDS:
        result = run_self_check(
            manifest,
            seed=seed,
            skip_jmq=True,
            embed_cache=EmbeddingCache(StubEmbedder(), cache_dir=tmp_path / f"e{seed}"),
            tokenizer=StubTokenizer(),
            bounds=open_bounds,
        )
        mmds.append(result.mmd)
        token_l2s.append(result.token_l2)

    # The observed ranges reproduce the documented provenance numbers.
    assert max(abs(m) for m in mmds) == pytest.approx(0.02948646, abs=1e-6)
    assert max(token_l2s) == pytest.approx(0.04379749, abs=1e-6)
    # The documented 0.10 bounds clear the true observed noise (both directions).
    assert max(mmds) < STUB_INSTRUMENT_BOUNDS.mmd_max
    assert min(mmds) > STUB_INSTRUMENT_BOUNDS.mmd_min
    assert max(token_l2s) < STUB_INSTRUMENT_BOUNDS.token_l2_max


# --- IMPORTANT 3: self-check integrity failure maps to a defined exit code ----


def test_integrity_error_is_typed_not_bare_assertion():
    """A leaked-pair split raises the typed SelfCheckIntegrityError (exit-mappable).

    IMPORTANT 3: the disjointness guard must raise a typed error the CLI maps to a
    defined exit code, not a bare AssertionError that surfaces as exit 1 (which
    the exit-code contract has no case for).
    """
    import dehip.self_check as sc

    # Force split_pairs past its shuffle with a stub that returns overlapping
    # halves, so the disjointness guard fires and must raise the typed error.
    ids = [f"p{i}" for i in range(10)]

    class _OverlapShuffle:
        def __init__(self, *a, **k):
            pass

        def shuffle(self, seq):
            # Leave duplicated ids so the two halves overlap.
            seq[:] = ids[:5] + ids[:5]

    original = sc.random.Random
    sc.random.Random = _OverlapShuffle
    try:
        with pytest.raises(SelfCheckIntegrityError):
            split_pairs(ids, seed=0)
    finally:
        sc.random.Random = original


def test_cli_maps_integrity_failure_to_self_check_exit(tmp_path, monkeypatch):
    """The CLI maps a self-check integrity failure to exit 4, never exit 1.

    IMPORTANT 3: even though the disjointness branch is hard to reach in normal
    operation, the CLI must map the integrity failure to the defined self-check
    exit code rather than letting a bare AssertionError escape as exit 1.
    """
    manifest = _write_reference(tmp_path, 40)
    _patch_stub_instruments(monkeypatch)

    def _raise_integrity(*a, **k):
        raise SelfCheckIntegrityError("split leaked a pair into both halves")

    monkeypatch.setattr("dehip.self_check.run_self_check", _raise_integrity)

    rc = cli.main(["self-check", "--reference", manifest, "--skip-jmq"])
    assert rc == cli.EXIT_SELF_CHECK == 4
    assert rc != 1


# --- NIT: embedder_id recorded from the injected instrument -------------------


def test_embedder_id_recorded_from_injected_cache(tmp_path):
    """The report's embedder_id comes from the instrument that ran, not a default.

    NIT: the CLI does not pass embedder_id, so without this the report would
    record the default nvidia/llama-embed-nemotron-8b regardless of what ran. The
    injected stub cache carries embedder_id='stub-embedder'; the report must show
    it.
    """
    manifest = _write_reference(tmp_path, 20)
    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),  # StubEmbedder -> embedder_id 'stub-embedder'
        tokenizer=StubTokenizer(),
    )
    assert result.report.config["embedder_id"] == "stub-embedder"


def test_explicit_embedder_id_overrides_injected(tmp_path):
    """An explicit embedder_id wins over the injected cache's id."""
    manifest = _write_reference(tmp_path, 20)
    result = run_self_check(
        manifest,
        seed=0,
        skip_jmq=True,
        embed_cache=_cache(tmp_path),
        tokenizer=StubTokenizer(),
        embedder_id="explicit-embedder",
    )
    assert result.report.config["embedder_id"] == "explicit-embedder"
