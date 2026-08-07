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
- ``test_rounds_at_max_completes_successfully`` -- rounds == MAX_ROUNDS (4)
  completes; the inclusive upper edge, not just the k=5 rejection.
- ``test_per_round_manifest_emitted_per_round`` -- one role=rewrite TextSet
  manifest per round (round=k), matching data-model.md.
- ``test_hip_config_records_requested_seed`` /
  ``test_cli_threads_global_seed_into_hip_config`` -- hip_config carries the
  harness-requested seed (top level + per round), wired from the CLI --seed.
- ``test_foreign_pair_id_in_rewrite_file_fails_loudly`` /
  ``test_corpus_drift_maps_to_exit_2_via_cli`` -- a stale foreign pair_id in the
  rewrite file fails loudly (CorpusDriftError -> exit 2), no contaminated manifest.
- ``test_parseable_duplicate_does_not_inflate_skipped`` -- a parseable duplicate
  pair_id keeps skipped at 0 (skipped counts real parse failures, not dedup).
- ``test_shrink_resume_prunes_stale_higher_k_artifacts`` -- completing at rounds=3
  then resuming at rounds=2 removes the stale k3 manifest + texts.
- ``test_hard_trip_on_last_round_yields_final_round_k_minus_one`` -- a hard trip on
  the LAST requested round sets final_round == requested_k - 1 (mutation catcher).
- ``test_draft_file_*_is_validation_error`` -- bundles_from_draft_file guards
  (unparseable line, blank text, duplicate pair_id, empty file) each -> exit 2.
- ``test_torn_final_line_repaired_on_resume`` /
  ``test_torn_tail_with_corrupt_record_counts_in_skipped`` -- a torn/truncated
  final line (no trailing newline) repairs on resume; skipped reflects the real
  torn line; the torn pair regenerates cleanly with no duplicate.
- ``test_subprocess_work_dirs_are_per_pair`` -- each pair's subprocess work dir is
  keyed on the pair_id so per-pair audit artifacts persist (NIT 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml as _yaml

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
        # Mirrors the real seam's audit shape: only fields hip-run applies
        # (adapter, num_rounds, sampling), no dead ``round`` key and no ``seed``.
        return {
            "adapter_path": adapter_id,
            "num_rounds": 1,
            "temperature": 1.0,
            "top_p": 0.95,
        }

    def run_round(self, inputs, *, round_k, adapter_id, seed=0):
        out: dict[str, str] = {}
        for pair_id, text in inputs.items():
            self.calls.append((pair_id, round_k))
            out[pair_id] = self._transform(text, round_k, pair_id)
        return out


class BadOutputHipRunner:
    """Stub whose round omits a pair (a malformed/incomplete hip-run result)."""

    def config_for(self, *, round_k, adapter_id):
        return {"adapter_path": adapter_id, "num_rounds": 1}

    def run_round(self, inputs, *, round_k, adapter_id, seed=0):
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
    """Draft-file mode produces the same round + degeneration shape as run-continuation.

    Same drafts fed two ways -- as generate's nascent bundles vs a draft JSONL --
    must yield the same completed-bundle rounds/final_round/degeneration. The two
    modes are NOT byte-identical by design: draft provenance (model_id/sampling)
    differs, so this parity check excludes the draft field and compares only the
    rewrite trajectory.
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

        def run_round(self, inputs, *, round_k, adapter_id, seed=0):
            (pair_id,) = inputs  # one pair per round in the cascade
            if pair_id != bundles[0].pair_id and pair_id not in self._done:
                raise KeyboardInterrupt("simulated interrupt at a new pair")
            self._done.add(pair_id)
            return super().run_round(
                inputs, round_k=round_k, adapter_id=adapter_id, seed=seed
            )

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
            return {"adapter_path": adapter_id, "num_rounds": 1}

        def run_round(self, inputs, *, round_k, adapter_id, seed=0):
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


# --- hip_config records the harness-controlled seed (IMPORTANT 2) -------------


def test_hip_config_records_requested_seed(tmp_path):
    """hip_config records the seed as advisory (not-applied) and the applied sampling.

    ``hip-run`` has no seed flag and never seeds the RNG (inference.py), so the
    seed is bookkeeping, not a reproducibility guarantee: it is recorded top-level
    but flagged ``seed_applied: False`` so it is never mistaken for one. The
    fields that ACTUALLY govern the rewrite -- ``temperature``/``top_p`` -- are the
    ones inlined per round, so a resumed/reproduced run has an honest trail of the
    sampling that produced each round. No round carries a bare ``seed``.
    """
    bundles = _bundles(1)
    run_cascade(
        bundles,
        runner=ScriptedHipRunner(),
        run_dir=tmp_path,
        run_id="RUN1",
        requested_k=2,
        adapter_id="ADP",
        seed=4242,
        printer=lambda *_: None,
    )

    bundle = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)[0]
    assert bundle.hip_config["seed"] == 4242
    # The seed is recorded but explicitly marked not-applied (advisory only).
    assert bundle.hip_config["seed_applied"] is False
    # Every round's requested config carries the applied sampling, not a seed.
    assert [rc["temperature"] for rc in bundle.hip_config["rounds"]] == [1.0, 1.0]
    assert [rc["top_p"] for rc in bundle.hip_config["rounds"]] == [0.95, 0.95]
    assert all("seed" not in rc for rc in bundle.hip_config["rounds"])
    assert all("round" not in rc for rc in bundle.hip_config["rounds"])


def test_cli_threads_global_seed_into_hip_config(tmp_path, monkeypatch):
    """The global --seed reaches hip_config through the rewrite CLI path.

    Proves the seam is wired end to end (CLI --seed -> run_cascade -> config_for),
    not merely that run_cascade accepts a seed kwarg.
    """
    monkeypatch.setattr(cascade_mod, "check_hip_precondition", lambda repo: None)
    monkeypatch.setattr(
        cascade_mod, "SubprocessHipRunner", lambda *a, **k: ScriptedHipRunner()
    )

    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        json.dumps({"pair_id": "fineweb-00000", "text": "a draft with words here."})
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    code = cli.main(
        [
            "--seed",
            "99",
            "rewrite",
            "--draft-file",
            str(draft_path),
            "--hip-repo",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--rounds",
            "2",
        ]
    )
    assert code == cli.EXIT_SUCCESS
    bundle = read_jsonl(out_dir / "rewrite-bundles.jsonl", RewriteBundle)[0]
    assert bundle.hip_config["seed"] == 99


# --- Corpus-drift guard (IMPORTANT 1) ----------------------------------------


def test_foreign_pair_id_in_rewrite_file_fails_loudly(tmp_path):
    """A stale foreign pair_id in the rewrite file -> loud exit 2, no manifest.

    A run dir carrying a completed rewrite bundle from a DIFFERENT/larger corpus
    must not silently merge that foreign pair_id into the emitted manifests +
    texts, mislabeled and counted, exiting 0. Mirroring generate.py, the persisted
    pair_ids must be a subset of the input pair_ids; a stray fails loudly with
    CorpusDriftError (-> exit 2), naming the stray, before any manifest is written.
    """
    bundles = _bundles(1)  # input corpus is exactly {fineweb-00000}

    # Seed the rewrite file with a completed bundle for a FOREIGN pair the input
    # does not carry (as a stale larger-corpus run dir would).
    foreign = _nascent("otherset-00042", "a foreign draft from a stale run dir here.")
    completed_foreign = cascade_mod._run_rounds_for_pair(
        foreign,
        runner=ScriptedHipRunner(),
        requested_k=2,
        adapter_id="ADP",
        seed=0,
    )
    rewrite_path = tmp_path / "rewrite-bundles.jsonl"
    cascade_mod._append_bundle(completed_foreign, rewrite_path)

    with pytest.raises(cascade_mod.CorpusDriftError) as exc_info:
        _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=2)
    assert "otherset-00042" in str(exc_info.value)

    # No contaminated manifest was written for the foreign pair.
    assert not (tmp_path / "rewrite-k1.manifest.json").exists()
    assert not (tmp_path / "rewrite-k2.manifest.json").exists()


def test_corpus_drift_maps_to_exit_2_via_cli(tmp_path, monkeypatch):
    """CorpusDriftError (a ValueError) maps to CLI exit 2 through the rewrite path."""
    monkeypatch.setattr(cascade_mod, "check_hip_precondition", lambda repo: None)
    monkeypatch.setattr(
        cascade_mod, "SubprocessHipRunner", lambda *a, **k: ScriptedHipRunner()
    )

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    foreign = _nascent("otherset-00042", "a foreign draft from a stale run dir here.")
    completed_foreign = cascade_mod._run_rounds_for_pair(
        foreign,
        runner=ScriptedHipRunner(),
        requested_k=2,
        adapter_id="ADP",
        seed=0,
    )
    cascade_mod._append_bundle(
        completed_foreign, out_dir / "rewrite-bundles.jsonl"
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
            str(out_dir),
        ]
    )
    assert code == cli.EXIT_VALIDATION == 2


# --- skipped-count does not mislabel a parseable duplicate (IMPORTANT 3) ------


def test_parseable_duplicate_does_not_inflate_skipped(tmp_path):
    """A durable file with a duplicate PARSEABLE pair_id -> skipped stays 0.

    _read_all_bundles de-dupes a duplicate parseable pair_id (last writer wins),
    so ``raw_nonempty - len(bundles)`` would wrongly count the duplicate as a
    "skipped/truncated" line and claim the pair "will be re-run". skipped must be
    computed from the actual parse-FAILURE count, so a parseable duplicate keeps
    skipped at 0 and the pair is treated as done (not re-run).
    """
    bundles = _bundles(2)
    rewrite_path = tmp_path / "rewrite-bundles.jsonl"

    # Write two completed bundles, then append a DUPLICATE (parseable) line for
    # pair 0 -- as a regenerated-over-a-stale-parseable-line resume would produce.
    for b in bundles:
        completed = cascade_mod._run_rounds_for_pair(
            b, runner=ScriptedHipRunner(), requested_k=2, adapter_id="ADP", seed=0
        )
        cascade_mod._append_bundle(completed, rewrite_path)
    dup = cascade_mod._run_rounds_for_pair(
        bundles[0], runner=ScriptedHipRunner(), requested_k=2, adapter_id="ADP", seed=0
    )
    cascade_mod._append_bundle(dup, rewrite_path)

    done, skipped = cascade_mod._load_done_rewrite_ids(rewrite_path)
    assert done == {bundles[0].pair_id, bundles[1].pair_id}
    assert skipped == 0  # a parseable duplicate is NOT a skipped/truncated line

    # And a resume re-runs NEITHER pair (both are durable/done despite the dup).
    resumer = ScriptedHipRunner()
    summary = _run(resumer, bundles, tmp_path, requested_k=2)
    assert resumer.calls == []
    assert summary["rewritten"] == 0
    assert summary["already_done"] == 2


# --- stale manifest pruned on shrink-resume (IMPORTANT 4) --------------------


def test_shrink_resume_prunes_stale_higher_k_artifacts(tmp_path):
    """Complete at rounds=3, resume at rounds=2 -> k3 artifacts are gone.

    Higher-k manifests + texts from the prior larger run must not linger for a
    downstream report/score to consume as if round 3 were part of this run. On the
    shorter resume the k3 manifest and its texts file are removed; the k1/k2
    artifacts this run emits survive.
    """
    bundles = _bundles(2)

    _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=3)
    assert (tmp_path / "rewrite-k3.manifest.json").exists()
    assert (tmp_path / "rewrite-k3-texts.jsonl").exists()

    # Resume at a SMALLER rounds. The already-done pairs are not re-run, but the
    # manifests are re-derived (from the completed bundles) at the new requested_k,
    # so the stale k3 artifacts must be pruned.
    _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=2)
    assert not (tmp_path / "rewrite-k3.manifest.json").exists()
    assert not (tmp_path / "rewrite-k3-texts.jsonl").exists()
    # The rounds this run emits are intact.
    assert (tmp_path / "rewrite-k1.manifest.json").exists()
    assert (tmp_path / "rewrite-k2.manifest.json").exists()


# --- hard trip on the LAST requested round (IMPORTANT 5.1) --------------------


def test_hard_trip_on_last_round_yields_final_round_k_minus_one(tmp_path):
    """A hard trip on the LAST requested round (k=2 of 2) -> final_round == 1.

    Distinct from a clean completion's final_round == requested_k (== 2). This
    catches a mutation that sets final_round = requested_k at loop end regardless
    of a trip on the final round: here the final round hard-trips, so final_round
    must be requested_k - 1 (the last good round), not requested_k.
    """
    target = "fineweb-00000"
    bundles = [_nascent(target, "a compact draft with several words in it here.")]

    def transform(text, k, pair_id):
        if k == 2:
            return text * 6  # hard length trip on the LAST requested round
        return f"{text} [r{k}]"

    _run(ScriptedHipRunner(transform), bundles, tmp_path, requested_k=2)

    bundle = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)[0]
    assert bundle.requested_k == 2
    assert bundle.final_round == 1  # requested_k - 1, NOT requested_k
    assert bundle.degeneration["hard_tripped"] is True
    assert [r["k"] for r in bundle.rounds] == [1, 2]
    assert bundle.rounds[1]["hard_tripped"] is True


# --- bundles_from_draft_file guard tests (IMPORTANT 5.2) ----------------------


def test_draft_file_unparseable_line_is_validation_error(tmp_path):
    """An unparseable draft-file line -> ValueError (-> exit 2)."""
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cascade_mod.bundles_from_draft_file(draft_path, run_id="RUN1")


def test_draft_file_blank_text_is_validation_error(tmp_path):
    """A blank draft text -> ValueError (nothing to rewrite; -> exit 2)."""
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        json.dumps({"pair_id": "fineweb-00000", "text": "   "}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        cascade_mod.bundles_from_draft_file(draft_path, run_id="RUN1")


def test_draft_file_duplicate_pair_id_is_validation_error(tmp_path):
    """A duplicate pair_id in the draft file -> ValueError (-> exit 2)."""
    draft_path = tmp_path / "drafts.jsonl"
    with draft_path.open("w", encoding="utf-8") as fh:
        for text in ("first draft.", "dup draft."):
            fh.write(json.dumps({"pair_id": "fineweb-00000", "text": text}) + "\n")
    with pytest.raises(ValueError):
        cascade_mod.bundles_from_draft_file(draft_path, run_id="RUN1")


def test_draft_file_empty_is_validation_error(tmp_path):
    """An empty draft file (no records) -> ValueError (-> exit 2)."""
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError):
        cascade_mod.bundles_from_draft_file(draft_path, run_id="RUN1")


def test_draft_file_guards_map_to_exit_2_via_cli(tmp_path, monkeypatch):
    """A draft-file guard failure maps to CLI exit 2 (input error), not exit 1."""
    monkeypatch.setattr(cascade_mod, "check_hip_precondition", lambda repo: None)
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text("{not valid json\n", encoding="utf-8")
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
    assert code == cli.EXIT_VALIDATION == 2


# --- torn/truncated final line on resume (IMPORTANT 5.3) ----------------------


def test_torn_final_line_repaired_on_resume(tmp_path):
    """A torn final rewrite-bundle line (NO trailing newline) repairs on resume.

    A crash between an append's record write and its newline leaves the final
    record with no trailing newline. On resume the torn tail is repaired BEFORE
    the first append (so a regenerated record never concatenates onto the
    fragment), skipped reflects the real torn line, and the torn pair regenerates
    cleanly -- no duplicate, no lost round.
    """
    bundles = _bundles(2)
    rewrite_path = tmp_path / "rewrite-bundles.jsonl"

    # pair 0 is durably complete (newline-terminated).
    completed0 = cascade_mod._run_rounds_for_pair(
        bundles[0], runner=ScriptedHipRunner(), requested_k=2, adapter_id="ADP", seed=0
    )
    cascade_mod._append_bundle(completed0, rewrite_path)

    # pair 1's record was written but its trailing newline never landed: a torn
    # tail. Append the JSON WITHOUT a newline to simulate the crash.
    completed1 = cascade_mod._run_rounds_for_pair(
        bundles[1], runner=ScriptedHipRunner(), requested_k=2, adapter_id="ADP", seed=0
    )
    from dehip.schemas import _to_dict

    torn = json.dumps(_to_dict(completed1))
    with rewrite_path.open("a", encoding="utf-8") as fh:
        fh.write(torn)  # NO trailing newline -> torn tail

    # Before resume, the torn line still parses on its own (it is complete JSON,
    # just unterminated), so it counts as done. The repair + resume path is what
    # we assert: the boundary is fixed so a later append is not corrupted.
    done_before, skipped_before = cascade_mod._load_done_rewrite_ids(rewrite_path)

    # Resume: the torn tail is repaired (newline appended) before any new append.
    resumer = ScriptedHipRunner()
    summary = _run(resumer, bundles, tmp_path, requested_k=2)

    # No duplicate bundle; both pairs present exactly once with their rounds kept.
    completed = read_jsonl(rewrite_path, RewriteBundle)
    ids = [b.pair_id for b in completed]
    assert sorted(ids) == sorted(b.pair_id for b in bundles)
    assert len(ids) == len(set(ids))
    for b in completed:
        assert [r["k"] for r in b.rounds] == [1, 2]
    # The final rewrite file is newline-terminated after the repair (no torn tail).
    assert rewrite_path.read_bytes().endswith(b"\n")
    assert summary["already_done"] >= 1


def test_torn_tail_with_corrupt_record_counts_in_skipped(tmp_path):
    """A truly unparseable torn tail is counted in skipped and the pair re-runs.

    Distinct from the complete-but-unterminated tail above: here the final line is
    a genuinely corrupt fragment (partial JSON). It parses in NEITHER reader, so
    skipped counts exactly that one torn line and the corresponding pair is re-run
    on resume, with no duplicate.
    """
    bundles = _bundles(2)
    rewrite_path = tmp_path / "rewrite-bundles.jsonl"

    completed0 = cascade_mod._run_rounds_for_pair(
        bundles[0], runner=ScriptedHipRunner(), requested_k=2, adapter_id="ADP", seed=0
    )
    cascade_mod._append_bundle(completed0, rewrite_path)
    # A corrupt partial-JSON fragment as the torn tail (no newline, unparseable).
    with rewrite_path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair_id": "fineweb-00001", "rounds": [{"k": 1,')

    _done, skipped = cascade_mod._load_done_rewrite_ids(rewrite_path)
    assert skipped == 1  # exactly the one corrupt torn line

    resumer = ScriptedHipRunner()
    _run(resumer, bundles, tmp_path, requested_k=2)
    # The corrupt fragment stays on its own skippable line (repaired boundary), so
    # read through the tolerant reader that drops it -- exactly what resume uses.
    completed = cascade_mod._read_all_bundles(rewrite_path)
    ids = [b.pair_id for b in completed]
    assert sorted(ids) == sorted(b.pair_id for b in bundles)
    assert len(ids) == len(set(ids))
    # pair 1 (whose torn record was corrupt) was re-run.
    assert bundles[1].pair_id in {pid for pid, _ in resumer.calls}


# --- rounds == MAX_ROUNDS completes (IMPORTANT 5.4) ---------------------------


def test_rounds_at_max_completes_successfully(tmp_path):
    """rounds == 4 (MAX_ROUNDS) completes; only k=5 is rejected.

    The prior suite tested only the k=5 rejection boundary. This locks the
    inclusive upper edge: requested_k == MAX_ROUNDS runs all four rounds and
    records final_round == 4.
    """
    assert cascade_mod.MAX_ROUNDS == 4
    bundles = _bundles(1)
    summary = _run(ScriptedHipRunner(), bundles, tmp_path, requested_k=4)
    assert summary["requested_k"] == 4

    bundle = read_jsonl(tmp_path / "rewrite-bundles.jsonl", RewriteBundle)[0]
    assert [r["k"] for r in bundle.rounds] == [1, 2, 3, 4]
    assert bundle.final_round == 4


# --- SubprocessHipRunner against the REAL hip-run config + parquet contract ---
#
# The tests below drive the REAL SubprocessHipRunner against a FAKE hip-run
# instead of the mocked HipRunner seam the round-loop tests inject. The old
# suite mocked the whole seam, which is exactly why the config/CLI/output-format
# mismatch with the real hip-run was invisible until an issue #16 real run. A
# fake hip-run reproduces the real I/O contract WITHOUT the model:
#   - reads the --config YAML (the only CLI arg the real hip-run accepts),
#   - asserts the REAL config keys (adapter_path, input_jsonl, output_parquet,
#     text_field, num_rounds) are present,
#   - reads input_jsonl (one {"text", "pair_id"} per line, IN ORDER),
#   - writes an output PARQUET matching the real schema (source_row_index, round,
#     original_text, input_text, output_text + word counts).
# It is installed by monkeypatching cascade_mod.subprocess.run so `uv run
# hip-run --config <cfg>` never shells out to the real CLI.


def _install_fake_hip_run(
    monkeypatch,
    *,
    rewrite=lambda text, idx: f"{text} [rewritten]",
    drop_indices=(),
    returncode=0,
    stderr="",
):
    """Monkeypatch subprocess.run with a fake hip-run; return captured invocations.

    ``rewrite(text, source_row_index) -> output_text`` scripts each row's
    rewrite (return "" or "   " to script a blank). ``drop_indices`` omits those
    source rows from the parquet (a pair the round did not rewrite).
    ``returncode``/``stderr`` script a non-zero exit. Every fake run records the
    config path + parsed config into the returned ``calls`` list so a test can
    assert the exact CLI shape (``--config`` only) and config schema.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        # The real hip-run argparse accepts ONLY --config: assert the CLI shape.
        assert cmd[:3] == ["uv", "run", "hip-run"], cmd
        assert cmd[3] == "--config", cmd
        assert "--input" not in cmd and "--output" not in cmd, cmd
        assert len(cmd) == 5, cmd  # uv run hip-run --config <path>, nothing else
        config_path = cmd[4]
        cfg = _yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        calls.append({"cmd": list(cmd), "config": cfg, "cwd": kwargs.get("cwd")})

        class _Proc:
            pass

        proc = _Proc()
        proc.returncode = returncode
        proc.stdout = ""
        proc.stderr = stderr
        if returncode != 0:
            return proc

        # The REAL config keys must be present (this is the contract under test).
        real_keys = (
            "adapter_path",
            "input_jsonl",
            "output_parquet",
            "text_field",
            "num_rounds",
        )
        for key in real_keys:
            assert key in cfg, f"config missing real hip-run key {key!r}: {cfg}"
        text_field = cfg["text_field"]
        rows = [
            json.loads(ln)
            for ln in Path(cfg["input_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        records = []
        for source_index, row in enumerate(rows):
            if source_index in drop_indices:
                continue
            # The real hip-run strips input text (inference.py:112); mirror it so
            # the fake's schema-fidelity claim stays honest.
            original = str(row.get(text_field) or "").strip()
            out = rewrite(original, source_index)
            records.append(
                {
                    "source_row_index": source_index,
                    "round": 1,
                    "original_text": original,
                    "input_text": original,
                    "output_text": out,
                    "original_word_count": len(original.split()),
                    "input_word_count": len(original.split()),
                    "output_word_count": len(out.split()),
                }
            )
        out_path = Path(cfg["output_parquet"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), out_path)
        return proc

    monkeypatch.setattr(cascade_mod.subprocess, "run", fake_run)
    return calls


def test_subprocess_config_emits_real_hip_run_schema(tmp_path, monkeypatch):
    """config_for emits the REAL hip-run keys; run_round writes them to the YAML.

    Locks the config schema against hip/inference.py: adapter_path (NOT
    adapter_id), num_rounds 1 (one round per invocation for inter-round
    degeneration), text_field "text", the applied sampling temperature/top_p
    (inference.py:118-120), and -- written by run_round -- input_jsonl /
    output_parquet / metadata_json / trust_remote_code. The dead ``round`` key and
    the never-applied ``seed`` are NOT emitted. The exclusivity assertions run on
    the on-disk cfg, not just the audit dict, so a stray ``adapter_id``/``round``/
    ``seed`` written to disk would fail here.
    """
    calls = _install_fake_hip_run(monkeypatch)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")

    # config_for is the path-independent audit view recorded in hip_config.
    audit = runner.config_for(round_k=1, adapter_id="an-adapter-id")
    assert audit["adapter_path"] == "an-adapter-id"
    assert "adapter_id" not in audit  # the real key is adapter_path
    assert audit["num_rounds"] == 1
    assert audit["text_field"] == "text"
    assert audit["temperature"] == 1.0
    assert audit["top_p"] == 0.95
    assert "round" not in audit  # hip-run never reads a bare round key
    assert "seed" not in audit  # hip-run has no seed flag
    # The audit dict is exactly the fields hip-run reads (path keys are added
    # only to the on-disk cfg in run_round).
    assert set(audit) == {
        "adapter_path",
        "num_rounds",
        "text_field",
        "temperature",
        "top_p",
    }

    runner.run_round(
        {"fineweb-00000": "a draft with words"},
        round_k=1,
        adapter_id="an-adapter-id",
    )
    cfg = calls[0]["config"]
    assert cfg["adapter_path"] == "an-adapter-id"
    assert cfg["num_rounds"] == 1
    assert cfg["text_field"] == "text"
    assert cfg["temperature"] == 1.0
    assert cfg["top_p"] == 0.95
    assert cfg["trust_remote_code"] is True
    for key in ("input_jsonl", "output_parquet", "metadata_json"):
        assert key in cfg
    # Exclusivity on the EMITTED on-disk cfg (not just the audit dict): the stray
    # adapter_id / round / seed keys must be absent from what hip-run actually
    # reads, and the on-disk keyset is exactly the audit fields plus path/flag keys.
    assert "adapter_id" not in cfg
    assert "round" not in cfg
    assert "seed" not in cfg
    assert set(cfg) == {
        "adapter_path",
        "num_rounds",
        "text_field",
        "temperature",
        "top_p",
        "input_jsonl",
        "output_parquet",
        "metadata_json",
        "trust_remote_code",
    }


def test_subprocess_invokes_config_only(tmp_path, monkeypatch):
    """run_round invokes `uv run hip-run --config <cfg>` and NOTHING else.

    The real hip-run argparse defines only --config; --input/--output would fail
    with "unrecognized arguments". The fake asserts the CLI shape internally, and
    here we re-assert on the captured cmd so the invariant is explicit.
    """
    calls = _install_fake_hip_run(monkeypatch)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    runner.run_round({"fineweb-00000": "text a"}, round_k=1, adapter_id="ADP")

    cmd = calls[0]["cmd"]
    assert cmd[:4] == ["uv", "run", "hip-run", "--config"]
    assert len(cmd) == 5
    assert "--input" not in cmd and "--output" not in cmd
    assert calls[0]["cwd"] == tmp_path  # invoked in the HIP checkout


def test_subprocess_parses_parquet_and_maps_source_row_index_to_pair_id(
    tmp_path, monkeypatch
):
    """The parquet's source_row_index maps back to pair_id by input write order.

    hip-run carries no pair_id; the mapping is positional. The runner writes the
    input in a stable (sorted) pair_id order, so source_row_index i is
    ordered_ids[i]. A rewrite keyed on its index proves each pair got ITS own
    row, not a shuffled one.
    """
    # rewrite embeds the source index so a mis-map would surface as wrong text.
    calls = _install_fake_hip_run(
        monkeypatch, rewrite=lambda text, idx: f"{text} :: idx={idx}"
    )
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")

    inputs = {
        "fineweb-00002": "gamma draft",
        "fineweb-00000": "alpha draft",
        "fineweb-00001": "beta draft",
    }
    out = runner.run_round(inputs, round_k=1, adapter_id="ADP")

    # Stable sorted order: 00000 -> idx 0, 00001 -> idx 1, 00002 -> idx 2.
    assert out["fineweb-00000"] == "alpha draft :: idx=0"
    assert out["fineweb-00001"] == "beta draft :: idx=1"
    assert out["fineweb-00002"] == "gamma draft :: idx=2"
    # The input JSONL carried {"text", "pair_id"} rows in that stable order.
    input_path = Path(calls[0]["config"]["input_jsonl"])
    written = [
        json.loads(ln)
        for ln in input_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["pair_id"] for r in written] == [
        "fineweb-00000",
        "fineweb-00001",
        "fineweb-00002",
    ]
    assert all("text" in r for r in written)


def test_subprocess_emitted_paths_are_absolute(tmp_path, monkeypatch):
    """input_jsonl / output_parquet / metadata_json in the config are ABSOLUTE.

    hip-run resolves relative config paths against ITS repo root, not dehip's
    work dir, so relative paths would land in the wrong place. Every path key
    must be absolute.
    """
    calls = _install_fake_hip_run(monkeypatch)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    runner.run_round({"fineweb-00000": "a draft"}, round_k=1, adapter_id="ADP")

    cfg = calls[0]["config"]
    for key in ("input_jsonl", "output_parquet", "metadata_json"):
        assert Path(cfg[key]).is_absolute(), f"{key} is not absolute: {cfg[key]}"


def test_subprocess_config_omits_base_model_by_default(tmp_path, monkeypatch):
    """config_for OMITS base_model by default so the adapter self-resolves.

    hip-run infers the base from the adapter's PeftConfig when base_model is
    absent, so the 0.6B adapter resolves to Qwen/Qwen3-0.6B-Base and the 4B to
    its own base. Emitting a fixed base_model would force a wrong base for any
    other-sized adapter.
    """
    calls = _install_fake_hip_run(monkeypatch)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    audit = runner.config_for(round_k=1, adapter_id="ADP")
    assert "base_model" not in audit

    runner.run_round({"fineweb-00000": "a draft"}, round_k=1, adapter_id="ADP")
    assert "base_model" not in calls[0]["config"]


def test_subprocess_config_includes_base_model_when_overridden(tmp_path, monkeypatch):
    """An explicit base_model override IS emitted (the only case it appears)."""
    calls = _install_fake_hip_run(monkeypatch)
    runner = cascade_mod.SubprocessHipRunner(
        tmp_path, work_dir=tmp_path / "work", base_model="Qwen/Qwen3-4B-Base"
    )
    audit = runner.config_for(round_k=1, adapter_id="ADP")
    assert audit["base_model"] == "Qwen/Qwen3-4B-Base"

    runner.run_round({"fineweb-00000": "a draft"}, round_k=1, adapter_id="ADP")
    assert calls[0]["config"]["base_model"] == "Qwen/Qwen3-4B-Base"


def test_subprocess_missing_pair_row_fails_loudly(tmp_path, monkeypatch):
    """A parquet omitting a requested pair's row -> HipRunError (never a skip)."""
    _install_fake_hip_run(monkeypatch, drop_indices=(1,))
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    with pytest.raises(HipRunError):
        runner.run_round(
            {"fineweb-00000": "alpha", "fineweb-00001": "beta"},
            round_k=1,
            adapter_id="ADP",
        )


def test_subprocess_blank_output_text_fails_loudly(tmp_path, monkeypatch):
    """A blank/whitespace output_text -> HipRunError, never a silent blank round."""
    _install_fake_hip_run(monkeypatch, rewrite=lambda text, idx: "   ")
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    with pytest.raises(HipRunError):
        runner.run_round({"fineweb-00000": "alpha"}, round_k=1, adapter_id="ADP")


def test_subprocess_nonzero_exit_fails_loudly(tmp_path, monkeypatch):
    """A non-zero hip-run exit -> HipRunError carrying the stderr."""
    _install_fake_hip_run(
        monkeypatch, returncode=1, stderr="hip-run: boom"
    )
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    with pytest.raises(HipRunError) as exc_info:
        runner.run_round({"fineweb-00000": "alpha"}, round_k=1, adapter_id="ADP")
    assert "boom" in str(exc_info.value)


def test_subprocess_missing_parquet_fails_loudly(tmp_path, monkeypatch):
    """A clean exit that writes NO parquet -> HipRunError (missing output)."""

    def fake_run(cmd, **kwargs):
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Proc()  # exits 0 but writes nothing

    monkeypatch.setattr(cascade_mod.subprocess, "run", fake_run)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    with pytest.raises(HipRunError):
        runner.run_round({"fineweb-00000": "alpha"}, round_k=1, adapter_id="ADP")


def _install_fake_hip_run_with_records(monkeypatch, *, records):
    """Monkeypatch subprocess.run to write EXACTLY ``records`` to the parquet.

    Unlike :func:`_install_fake_hip_run`, this lets a test control the raw rows --
    including arbitrary ``round`` stamps and repeated ``source_row_index`` values --
    so the round-filter and duplicate-index behavior can be exercised directly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def fake_run(cmd, **kwargs):
        cfg = _yaml.safe_load(Path(cmd[4]).read_text(encoding="utf-8"))

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        out_path = Path(cfg["output_parquet"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), out_path)
        return _Proc()

    monkeypatch.setattr(cascade_mod.subprocess, "run", fake_run)


def test_subprocess_reads_the_single_round_the_parquet_holds(tmp_path, monkeypatch):
    """A round_k=2 invocation resolves the pair even though hip-run stamps round 1.

    The cascade runs hip-run once per round with num_rounds=1, and hip-run's loop
    is range(1, num_rounds+1) (inference.py:124,161), so EVERY output row is
    stamped round==1 no matter which cascade round produced it. The runner must
    read the round the file actually contains, never filter on the cascade's
    round_k. This parquet carries a round-1 AND a round-2 row for the one source
    (distinguishable text); at round_k=2 the runner must still resolve the pair --
    a round_k filter would drop every row and raise instead.
    """
    records = [
        {"source_row_index": 0, "round": 1, "output_text": "the round-one rewrite"},
        {"source_row_index": 0, "round": 2, "output_text": "the round-two rewrite"},
    ]
    _install_fake_hip_run_with_records(monkeypatch, records=records)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")

    # A single source with a duplicate source_row_index across two rounds is the
    # real hip-run's shape for num_rounds>1; here the point is that round_k=2 does
    # not filter to empty. (This parquet has a repeated index, so the loud
    # duplicate guard fires -- proving the filter is gone AND the guard is honest.)
    with pytest.raises(HipRunError) as exc_info:
        runner.run_round({"fineweb-00000": "a draft"}, round_k=2, adapter_id="ADP")
    # It failed on the DUPLICATE index (rows were read), not on an empty result
    # from a round_k filter -- the message names the duplicate, not a missing pair.
    assert "duplicate" in str(exc_info.value)

    # And a clean single-round parquet (what num_rounds=1 actually writes: one row,
    # always stamped round 1) resolves at round_k=2 without any filter dropping it.
    single = [
        {"source_row_index": 0, "round": 1, "output_text": "resolved at round two"},
    ]
    _install_fake_hip_run_with_records(monkeypatch, records=single)
    out = runner.run_round({"fineweb-00000": "a draft"}, round_k=2, adapter_id="ADP")
    assert out == {"fineweb-00000": "resolved at round two"}


def test_subprocess_round_k_2_succeeds_end_to_end(tmp_path, monkeypatch):
    """A full round_k=2 run through the real seam resolves the pair (regression).

    Every other subprocess test drives round_k=1, which masks the round-filter bug
    because hip-run's stamped round (1) then matches round_k (1). This drives a
    round_k=2 invocation end to end through the standard fake (one row, stamped
    round 1) and asserts the pair is resolved -- the exact case the old filter
    turned into an empty result on the default 2-round run.
    """
    _install_fake_hip_run(monkeypatch, rewrite=lambda text, idx: f"{text} [k2]")
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    out = runner.run_round(
        {"fineweb-00000": "a draft with words"}, round_k=2, adapter_id="ADP"
    )
    assert out == {"fineweb-00000": "a draft with words [k2]"}


def test_subprocess_duplicate_source_row_index_fails_loudly(tmp_path, monkeypatch):
    """A repeated source_row_index in the kept rows -> HipRunError, not clobber.

    Two rows sharing a source_row_index would map to the same pair_id; a
    last-write-wins assignment would silently attribute one pair's rewrite to
    another and slip past the coverage/blank checks. The runner must fail loudly,
    naming the duplicate.
    """
    records = [
        {"source_row_index": 0, "round": 1, "output_text": "first rewrite"},
        {"source_row_index": 0, "round": 1, "output_text": "clobbering rewrite"},
    ]
    _install_fake_hip_run_with_records(monkeypatch, records=records)
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=tmp_path / "work")
    with pytest.raises(HipRunError) as exc_info:
        runner.run_round({"fineweb-00000": "a draft"}, round_k=1, adapter_id="ADP")
    assert "duplicate" in str(exc_info.value)


# --- per-pair work dir keyed on pair_id (NIT 1) ------------------------------


def test_subprocess_work_dirs_are_per_pair(tmp_path, monkeypatch):
    """Two pairs at the same round get DISTINCT work dirs (NIT 1).

    The SubprocessHipRunner's per-round work dir includes the pair_id, so one
    pair's subprocess input/config/output survive rather than being overwritten by
    the next pair at the same round. We drive the seam through a fake hip-run so
    no real hip-run is invoked, and assert each pair produced its own audit dir.
    """
    work_dir = tmp_path / "hip-work"
    runner = cascade_mod.SubprocessHipRunner(tmp_path, work_dir=work_dir)
    calls = _install_fake_hip_run(monkeypatch)

    runner.run_round({"fineweb-00000": "text a"}, round_k=1, adapter_id="ADP")
    runner.run_round({"fineweb-00001": "text b"}, round_k=1, adapter_id="ADP")

    # Two DISTINCT per-pair round-1 work dirs, both surviving on disk.
    out_dirs = {Path(c["config"]["output_parquet"]).parent for c in calls}
    assert len(out_dirs) == 2
    surviving = {p.name for p in work_dir.iterdir() if p.is_dir()}
    assert len(surviving) == 2
    for d in out_dirs:
        assert d.exists()
