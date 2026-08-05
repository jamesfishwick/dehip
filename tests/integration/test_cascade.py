"""Integration tests for the k-round HIP cascade (issue #13).

Every test injects a *stub* :class:`~dehip.cascade.HipRunner` so nothing shells
out, downloads a model, or touches the HIP checkout. The stub is the seam the
CRITICAL notes require: it drives the round-loop state machine, the degeneration
gating, resumability, and the subprocess-boundary failure modes without a real
``hip-run``.

Test map (each locks one DoD / adversarial-design item):

- ``test_per_round_capture_every_round_in_bundle`` -- per-round capture: every
  requested round is recorded in the bundle, in order, with its text + flags.
- ``test_hard_trip_stops_at_last_good_round_and_flags`` -- stop-and-flag on a
  degenerate round: final_round is the last GOOD round (k-1), hard_tripped true,
  the degenerate round is still kept but marked, and sibling pairs are
  unaffected (one degenerate pair never aborts the run).
- ``test_hard_trip_on_round_one_yields_final_round_zero`` -- the boundary: a hard
  trip on round 1 with no good round yet sets final_round 0 (the draft is the
  last good output) and keeps the one degenerate round.
- ``test_repetition_flag_does_not_stop_iteration`` -- a repetition FLAG is
  recorded but never stops iteration (flag-only, not hard).
- ``test_bundle_round_trips_through_read_jsonl`` -- bundle schema validity: the
  completed bundle round-trips through read_jsonl(RewriteBundle).
- ``test_draft_file_mode_matches_run_continuation_shape`` -- draft-file mode
  produces the identical bundle shape as run-continuation mode.
- ``test_missing_checkout_exits_3_before_any_subprocess_inference`` -- a missing
  sibling checkout fails fast with exit 3 before the seam ever runs a round.
- ``test_resume_reruns_only_incomplete_pairs`` -- resumability: an interrupt
  mid-cascade resumes without re-running completed pairs or losing rounds.
- ``test_malformed_hip_output_fails_loudly`` /
  ``test_empty_rewrite_fails_loudly`` -- the subprocess boundary: a malformed or
  empty hip-run result raises HipRunError (-> exit 3), never a silent blank round.
- ``test_requested_k_over_max_is_validation_error`` -- rounds > 4 is exit 2.
- ``test_per_round_manifest_emitted_per_round`` -- one role=rewrite TextSet
  manifest per round (round=k), matching data-model.md.
"""

from __future__ import annotations

import json

import pytest

from dehip import cascade as cascade_mod
from dehip import cli
from dehip.cascade import (
    HipRunError,
    RewriteBundle,
    run_cascade,
)
from dehip.schemas import TextSet, read_json, read_jsonl

# --- Stubs -------------------------------------------------------------------


class ScriptedHipRunner:
    """A :class:`HipRunner` stub that rewrites via a per-round transform.

    Never shells out. ``transform(text, k, pair_id)`` returns the round's
    rewrite text, so a test can script degeneration (an empty string, a length
    blow-up, a repetition burst) at a chosen round for a chosen pair while other
    pairs sail through. Records every ``run_round`` call for the resumability
    call-count proof.
    """

    def __init__(self, transform=None) -> None:
        self.calls: list[tuple[str, int]] = []  # (pair_id, round_k) per call
        # Default: append a round marker so the text grows a little each round
        # without tripping the length-ratio gate ([0.5, 2.0]).
        self._transform = transform or (
            lambda text, k, pair_id: f"{text} [r{k}]"
        )

    def config_for(self, *, round_k, adapter_id):
        return {"round": round_k, "adapter_id": adapter_id, "rounds": 1}

    def run_round(self, inputs, *, round_k, adapter_id):
        out: dict[str, str] = {}
        for pair_id, text in inputs.items():
            self.calls.append((pair_id, round_k))
            out[pair_id] = self._transform(text, round_k, pair_id)
        return out


class BadOutputHipRunner:
    """Stub whose round omits a pair (a malformed/incomplete hip-run result)."""

    def config_for(self, *, round_k, adapter_id):
        return {"round": round_k}

    def run_round(self, inputs, *, round_k, adapter_id):
        return {}  # rewrote nothing: the pair is missing from the output


# --- Fixtures ----------------------------------------------------------------


def _nascent(pair_id: str, draft_text: str, *, run_id: str = "RUN1") -> RewriteBundle:
    """A nascent bundle as generate.py would have written (draft, empty rounds)."""
    return RewriteBundle(
        run_id=run_id,
        pair_id=pair_id,
        prompt=f"prompt for {pair_id}",
        rounds=[],
        final_round=0,
        degeneration={"hard_tripped": False},
        adapter_id="",
        hip_config={},
        requested_k=0,
        draft={
            "text": draft_text,
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "sampling": {"temperature": 0.7, "top_p": 0.95, "seed": 7},
        },
    )


def _bundles(n: int, *, corpus: str = "fineweb") -> list[RewriteBundle]:
    return [
        _nascent(f"{corpus}-{i:05d}", f"This is draft number {i} with plenty of words.")
        for i in range(n)
    ]


def _run(runner, bundles, run_dir, *, requested_k=2, adapter_id="ADP"):
    return run_cascade(
        bundles,
        runner=runner,
        run_dir=run_dir,
        run_id="RUN1",
        requested_k=requested_k,
        adapter_id=adapter_id,
        printer=lambda *_: None,
    )


# --- Per-round capture -------------------------------------------------------


def test_per_round_capture_every_round_in_bundle(tmp_path):
    bundles = _bundles(3)
    runner = ScriptedHipRunner()
    summary = _run(runner, bundles, tmp_path, requested_k=3)

    assert summary["pairs"] == 3
    assert summary["rewritten"] == 3
    assert summary["requested_k"] == 3

    completed = {
        b.pair_id: b
        for b in read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)
    }
    assert len(completed) == 3
    for bundle in completed.values():
        # Every requested round is captured, in order, each carrying its text.
        assert [r["k"] for r in bundle.rounds] == [1, 2, 3]
        assert bundle.final_round == 3
        assert bundle.requested_k == 3
        assert bundle.adapter_id == "ADP"
        # The rewrite text is the draft grown one marker per round (no degeneration).
        assert bundle.rounds[-1]["text"].endswith("[r3]")
        # hip_config inlines every round's emitted config for audit.
        assert len(bundle.hip_config["rounds"]) == 3


# --- Stop-and-flag with correct final_round ----------------------------------


def test_hard_trip_stops_at_last_good_round_and_flags(tmp_path):
    """A hard trip on round 2 stops at final_round 1; the sibling pair is unaffected.

    The degenerate pair blows its length up 5x on round 2 (length ratio > 2.0 =
    hard). final_round must be the LAST GOOD round (1), hard_tripped true, and the
    degenerate round 2 still KEPT in bundle.rounds but marked. A second, healthy
    pair completes all rounds -- one degenerate pair never aborts the run.
    """
    degenerate_id = "fineweb-00000"
    healthy_id = "fineweb-00001"
    bundles = [
        _nascent(degenerate_id, "short draft text here for the ratio."),
        _nascent(healthy_id, "another healthy draft with plenty of words to keep."),
    ]

    def transform(text, k, pair_id):
        if pair_id == degenerate_id and k == 2:
            return text * 6  # length ratio ~6x -> hard length_ratio trip
        return f"{text} [r{k}]"

    runner = ScriptedHipRunner(transform)
    summary = _run(runner, bundles, tmp_path, requested_k=3)

    completed = {
        b.pair_id: b
        for b in read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)
    }

    degen = completed[degenerate_id]
    # Stopped at the last GOOD round (1); round 2 is the degenerate one.
    assert degen.final_round == 1
    assert degen.degeneration["hard_tripped"] is True
    # The degenerate round is still KEPT (every intermediate kept) but marked, and
    # NO round 3 ran (iteration stopped).
    assert [r["k"] for r in degen.rounds] == [1, 2]
    assert degen.rounds[0]["hard_tripped"] is False
    assert degen.rounds[1]["hard_tripped"] is True
    assert "length_ratio" in degen.rounds[1]["flags"]

    # The healthy sibling completed all 3 rounds and is NOT flagged.
    healthy = completed[healthy_id]
    assert healthy.final_round == 3
    assert healthy.degeneration["hard_tripped"] is False
    assert [r["k"] for r in healthy.rounds] == [1, 2, 3]

    # The run-level count reflects exactly one flagged pair.
    assert summary["flagged_degenerate"] == 1


def test_hard_trip_on_round_one_yields_final_round_zero(tmp_path):
    """Boundary: a hard trip on round 1 (no good round yet) -> final_round 0.

    With no prior good round, the draft (round 0) IS the last good output, so
    final_round is 0. The one degenerate round is still kept and marked, and no
    round 2 runs. A length-ratio blow-up drives the hard trip -- a genuine
    degenerate (but non-blank) rewrite; a *blank* round-1 result is a different
    concern (the loud subprocess-boundary failure, tested separately) and never
    reaches the degeneration gate.
    """
    target = "fineweb-00000"
    bundles = [_nascent(target, "a compact draft with several words in it here.")]

    def transform(text, k, pair_id):
        if k == 1:
            return text * 6  # length ratio ~6x -> hard length_ratio trip round 1
        return f"{text} [r{k}]"

    runner = ScriptedHipRunner(transform)
    _run(runner, bundles, tmp_path, requested_k=2)

    bundle = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)[0]
    assert bundle.final_round == 0  # the draft was the last good output
    assert bundle.degeneration["hard_tripped"] is True
    # The one degenerate round is kept and marked; round 2 never ran.
    assert [r["k"] for r in bundle.rounds] == [1]
    assert bundle.rounds[0]["hard_tripped"] is True
    assert "length_ratio" in bundle.rounds[0]["flags"]
    # The seam was asked for round 1 only, never round 2.
    assert runner.calls == [(target, 1)]


# --- Repetition flag is not a stop -------------------------------------------


def test_repetition_flag_does_not_stop_iteration(tmp_path):
    """A repetition FLAG is recorded but never stops iteration (flag-only)."""
    target = "fineweb-00000"
    bundles = [_nascent(target, "Draft opening sentence one here. Draft body two.")]

    def transform(text, k, pair_id):
        if k == 1:
            # 3+ consecutive sentences sharing a start word -> repetition flag,
            # but a length within [0.5, 2.0] so it is NOT a hard trip.
            return "The cat sat. The cat ran. The cat ate. The cat slept."
        return f"{text} more words to keep the ratio sane across the round."

    runner = ScriptedHipRunner(transform)
    _run(runner, bundles, tmp_path, requested_k=2)

    bundle = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)[0]
    # Iteration continued through both rounds despite the round-1 repetition flag.
    assert bundle.final_round == 2
    assert [r["k"] for r in bundle.rounds] == [1, 2]
    assert "repetition" in bundle.rounds[0]["flags"]
    assert bundle.rounds[0]["hard_tripped"] is False
    assert bundle.degeneration["hard_tripped"] is False


# --- Bundle schema validity --------------------------------------------------


def test_bundle_round_trips_through_read_jsonl(tmp_path):
    """The completed bundle round-trips through read_jsonl(RewriteBundle)."""
    bundles = _bundles(2)
    _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=2)

    # read_jsonl is strict: unknown fields / bad version raise. A clean parse of
    # every completed line is the schema-validity proof.
    completed = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)
    assert len(completed) == 2
    for bundle in completed:
        assert isinstance(bundle, RewriteBundle)
        assert bundle.draft is not None
        assert bundle.requested_k == 2
        assert bundle.final_round <= bundle.requested_k  # data-model invariant


# --- Draft-file parity -------------------------------------------------------


def test_draft_file_mode_matches_run_continuation_shape(tmp_path):
    """Draft-file mode produces the identical bundle shape as run-continuation.

    Same drafts fed two ways -- as generate's nascent bundles vs a draft JSONL --
    must yield byte-identical completed-bundle rounds/final_round/degeneration
    (only draft-provenance metadata differs, which is expected).
    """
    drafts = {
        "fineweb-00000": "First draft with enough words to rewrite cleanly here.",
        "fineweb-00001": "Second draft also carrying a fair number of words along.",
    }

    # Run-continuation shape: nascent bundles.
    cont_dir = tmp_path / "cont"
    cont_bundles = [_nascent(pid, text) for pid, text in drafts.items()]
    _run(ScriptedHipRunner(), cont_bundles, cont_dir, requested_k=2)
    cont = {
        b.pair_id: b
        for b in read_jsonl(cont_dir / "rewrite-bundles.jsonl", RewriteBundle)
    }

    # Draft-file mode: synthesize nascent bundles from a draft JSONL.
    draft_path = tmp_path / "drafts.jsonl"
    with draft_path.open("w", encoding="utf-8") as fh:
        for pid, text in drafts.items():
            fh.write(json.dumps({"pair_id": pid, "text": text}))
            fh.write("\n")
    file_dir = tmp_path / "file"
    file_bundles = cascade_mod.bundles_from_draft_file(draft_path, run_id="RUN1")
    _run(ScriptedHipRunner(), file_bundles, file_dir, requested_k=2)
    filed = {
        b.pair_id: b
        for b in read_jsonl(file_dir / "rewrite-bundles.jsonl", RewriteBundle)
    }

    assert set(cont) == set(filed)
    for pair_id in cont:
        # The rewrite trajectory is identical: same rounds, final_round, and
        # degeneration verdict regardless of how the draft got there.
        assert cont[pair_id].rounds == filed[pair_id].rounds
        assert cont[pair_id].final_round == filed[pair_id].final_round
        assert cont[pair_id].degeneration == filed[pair_id].degeneration
        assert cont[pair_id].requested_k == filed[pair_id].requested_k


# --- Missing checkout -> exit 3 before any subprocess inference ---------------


def test_missing_checkout_exits_3_before_any_subprocess_inference(
    tmp_path, monkeypatch
):
    """A missing HIP sibling checkout fails fast with exit 3, before any round.

    The precondition runs before the seam is ever asked to rewrite. We point
    --hip-repo at a nonexistent dir and assert the CLI returns exit 3 -- and, to
    prove no subprocess inference was reached, we make the seam explode if ever
    constructed/called.
    """
    # If the round loop were reached, this would raise instead of returning 3.
    def _boom(*a, **k):
        raise AssertionError("hip-run seam must not be reached on a bad precondition")

    monkeypatch.setattr(cascade_mod, "SubprocessHipRunner", _boom)

    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        json.dumps({"pair_id": "fineweb-00000", "text": "a draft with words here."})
        + "\n",
        encoding="utf-8",
    )
    missing_repo = tmp_path / "no-such-hip-checkout"
    code = cli.main(
        [
            "rewrite",
            "--draft-file",
            str(draft_path),
            "--hip-repo",
            str(missing_repo),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == cli.EXIT_EXTERNAL_DEP == 3


def test_precondition_directly_raises_for_missing_checkout(tmp_path):
    """check_hip_precondition raises HipPreconditionError for an absent checkout."""
    with pytest.raises(cascade_mod.HipPreconditionError):
        cascade_mod.check_hip_precondition(tmp_path / "absent")


# --- Resumability ------------------------------------------------------------


def test_resume_reruns_only_incomplete_pairs(tmp_path):
    """An interrupt mid-cascade resumes without re-running completed pairs.

    A first pass is killed after the first pair's bundle is flushed. The re-run
    must run exactly the remaining pairs (proved by the seam's per-pair call
    count), keep the first pair's captured rounds, and leave no duplicate bundle.
    """
    bundles = _bundles(3)

    class KillAfterOnePair(ScriptedHipRunner):
        """Completes pair 0's rounds, then raises when a NEW pair starts."""

        def __init__(self) -> None:
            super().__init__()
            self._done: set[str] = set()

        def run_round(self, inputs, *, round_k, adapter_id):
            (pair_id,) = inputs  # one pair per round in the cascade
            if pair_id != bundles[0].pair_id and pair_id not in self._done:
                raise KeyboardInterrupt("simulated interrupt at a new pair")
            self._done.add(pair_id)
            return super().run_round(inputs, round_k=round_k, adapter_id=adapter_id)

    killer = KillAfterOnePair()
    with pytest.raises(KeyboardInterrupt):
        _run(killer, bundles, tmp_path, requested_k=2)

    # Exactly pair 0 is durable after the interrupt.
    done, skipped = cascade_mod._load_done_rewrite_ids(
        tmp_path / "rewrite-bundles.jsonl"
    )
    assert done == {bundles[0].pair_id}
    assert skipped == 0

    # Re-run with a fresh runner: it must run only the 2 remaining pairs.
    resumer = ScriptedHipRunner()
    summary = _run(resumer, bundles, tmp_path, requested_k=2)
    rerun_pairs = {pid for pid, _ in resumer.calls}
    assert rerun_pairs == {bundles[1].pair_id, bundles[2].pair_id}
    assert summary["rewritten"] == 2
    assert summary["already_done"] == 1

    # No duplicate bundles: exactly 3 pair_ids, each once, first pair's rounds kept.
    completed = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)
    ids = [b.pair_id for b in completed]
    assert sorted(ids) == sorted(b.pair_id for b in bundles)
    assert len(ids) == len(set(ids))
    first = next(b for b in completed if b.pair_id == bundles[0].pair_id)
    assert [r["k"] for r in first.rounds] == [1, 2]


def test_idempotent_rerun_makes_no_calls(tmp_path):
    """Re-running a fully complete cascade runs zero rounds and appends nothing."""
    bundles = _bundles(3)
    _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=2)
    path = tmp_path / "rewrite-bundles.jsonl"
    before = len(path.read_text(encoding="utf-8").splitlines())

    rerun = ScriptedHipRunner()
    summary = _run(rerun, bundles, tmp_path, requested_k=2)
    assert rerun.calls == []
    assert summary["rewritten"] == 0
    assert summary["already_done"] == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == before


# --- Subprocess boundary: fail loudly on malformed/empty output --------------


def test_malformed_hip_output_fails_loudly(tmp_path):
    """A hip-run round missing a pair raises HipRunError, never a silent skip."""
    bundles = _bundles(1)
    with pytest.raises(HipRunError):
        _run(BadOutputHipRunner(), bundles, tmp_path, requested_k=2)


def test_empty_rewrite_via_cli_maps_to_exit_3(tmp_path, monkeypatch):
    """A mid-run HipRunError maps to CLI exit 3 (external dep), not exit 1.

    The precondition is stubbed to pass so the failure is reached in the round
    loop, where the seam returns an empty rewrite -- exactly the silent-blank
    corruption the guard rejects.
    """
    monkeypatch.setattr(cascade_mod, "check_hip_precondition", lambda repo: None)

    class EmptyRewriteRunner:
        def config_for(self, *, round_k, adapter_id):
            return {"round": round_k}

        def run_round(self, inputs, *, round_k, adapter_id):
            return {pid: "   " for pid in inputs}  # whitespace-only rewrite

    monkeypatch.setattr(
        cascade_mod, "SubprocessHipRunner", lambda *a, **k: EmptyRewriteRunner()
    )

    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        json.dumps({"pair_id": "fineweb-00000", "text": "a draft with words here."})
        + "\n",
        encoding="utf-8",
    )
    code = cli.main(
        [
            "rewrite",
            "--draft-file",
            str(draft_path),
            "--hip-repo",
            str(tmp_path),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == cli.EXIT_EXTERNAL_DEP == 3


# --- Rounds validation -------------------------------------------------------


def test_requested_k_over_max_is_validation_error(tmp_path):
    """requested_k > MAX_ROUNDS raises RoundsValidationError (-> exit 2)."""
    bundles = _bundles(1)
    with pytest.raises(cascade_mod.RoundsValidationError):
        _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=5)


def test_requested_k_over_max_via_cli_exits_2(tmp_path, monkeypatch):
    """--rounds 5 is rejected exit 2 -- before the precondition subprocess runs."""

    def _boom(repo):
        raise AssertionError("precondition must not run for a bad --rounds")

    monkeypatch.setattr(cascade_mod, "check_hip_precondition", _boom)
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        json.dumps({"pair_id": "fineweb-00000", "text": "a draft here."}) + "\n",
        encoding="utf-8",
    )
    code = cli.main(
        ["rewrite", "--draft-file", str(draft_path), "--rounds", "5"]
    )
    assert code == cli.EXIT_VALIDATION == 2


def test_run_and_draft_file_mutually_exclusive(tmp_path):
    """Passing neither --run nor --draft-file (or both) is exit 2."""
    assert cli.main(["rewrite"]) == cli.EXIT_VALIDATION
    assert (
        cli.main(["rewrite", "--run", str(tmp_path), "--draft-file", "d.jsonl"])
        == cli.EXIT_VALIDATION
    )


# --- Per-round TextSet manifests ---------------------------------------------


def test_per_round_manifest_emitted_per_round(tmp_path):
    """One role=rewrite TextSet manifest per round (round=k), per data-model.md.

    A pair that hard-trips at round 2 (final_round 1) appears in the k1 manifest
    but not the k2 manifest, so the per-round sets reflect exactly the surviving
    good outputs.
    """
    degenerate_id = "fineweb-00000"
    healthy_id = "fineweb-00001"
    bundles = [
        _nascent(degenerate_id, "short draft text here for the ratio check."),
        _nascent(healthy_id, "healthy draft with plenty of good words to keep going."),
    ]

    def transform(text, k, pair_id):
        if pair_id == degenerate_id and k == 2:
            return text * 6  # hard length trip at round 2
        return f"{text} [r{k}]"

    _run(ScriptedHipRunner(transform), bundles, tmp_path, requested_k=2)

    k1: TextSet = read_json(tmp_path / "rewrite-k1.manifest.json", TextSet)
    k2: TextSet = read_json(tmp_path / "rewrite-k2.manifest.json", TextSet)

    assert k1.role == "rewrite" and k1.round == 1
    assert k2.role == "rewrite" and k2.round == 2
    # Both pairs have a good round-1 output; only the healthy pair reaches round 2.
    assert sorted(k1.pair_ids) == sorted([degenerate_id, healthy_id])
    assert k2.pair_ids == [healthy_id]

    # The manifest points at its texts JSONL via provenance.texts_path (the
    # convention report._texts_path_for resolves).
    from dehip.report import _texts_path_for

    resolved = _texts_path_for(
        tmp_path / "rewrite-k1.manifest.json", k1.provenance
    )
    assert resolved == tmp_path / "rewrite-k1-texts.jsonl"
    texts = {
        json.loads(ln)["pair_id"]: json.loads(ln)["text"]
        for ln in resolved.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    assert set(texts) == {degenerate_id, healthy_id}
