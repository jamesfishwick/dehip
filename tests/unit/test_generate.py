"""Unit tests for stage-1 instruct-draft generation (issue #12).

Every test here injects a *stub* :class:`~dehip.generate.DraftModel` so nothing
loads transformers, torch, or the network. The stub counts its ``generate``
calls, which is exactly the instrument the resumability DoD requires: a first
run writes N drafts, is killed after k, and the re-run must make exactly N-k
further calls with no duplicate bundles.

Test map (each locks one DoD / adversarial-design item):

- ``test_smoke_run_one_draft_per_pair_with_full_metadata`` -- smoke run produces
  one draft per corpus pair with model id + full sampling settings in every
  bundle record.
- ``test_resume_generates_only_missing_drafts_by_call_count`` -- the call-count
  resumability proof: kill after k, re-run makes exactly N-k more generate calls,
  no duplicate pair_ids.
- ``test_truncated_final_bundle_line_does_not_corrupt_resume`` -- a partially
  written final JSONL line is skipped-and-counted; its pair is regenerated, and
  the resume neither crashes nor duplicates.
- ``test_torn_tail_without_trailing_newline_regenerates_pair_once`` -- a torn
  final record with NO trailing newline is repaired on resume, so the
  regenerated pair appears exactly once, no unparseable concatenated line
  survives, and the manifest has every pair (CRITICAL 1).
- ``test_empty_draft_is_not_written_and_not_marked_done`` -- an empty draft
  raises EmptyDraftError, is NOT persisted as a finished bundle, and is
  regenerated on a later resume (CRITICAL 2).
- ``test_generation_runtime_error_maps_to_exit_3`` /
  ``test_corpus_dir_maps_to_exit_2`` -- exit-code discipline (IMPORTANT 1).
- ``test_per_pair_seed_is_distinct_and_reproducible`` -- per-pair seeds differ,
  a re-generated pair reproduces its draft, recorded seed == effective seed
  (IMPORTANT 2).
- ``test_stale_bundle_pair_id_fails_loudly`` -- bundle drift fails exit 2
  (IMPORTANT 3).
- ``test_device_is_surfaced_in_summary`` -- the model device is recorded
  (IMPORTANT 4).
- ``test_every_bundle_records_model_id_and_sampling`` -- model id + sampling
  present in every persisted bundle record (redundant belt-and-suspenders over
  the smoke test, asserted directly on disk).
- ``test_sampling_seen_records_exact_effective_values`` /
  ``test_empty_pairs_raises`` / ``test_heterogeneous_corpus_raises`` /
  ``test_idempotent_rerun_makes_no_calls_and_appends_nothing`` -- test-gap
  closures (IMPORTANT 5).
- ``test_manifest_points_at_draft_texts_and_is_scorable_shape`` -- the
  role=instruct_draft TextSet manifest points at the {pair_id, text} JSONL so a
  scorer can consume it.
- ``test_model_load_failure_maps_to_exit_3`` -- a ModelLoadError from the seam
  maps to CLI exit 3 (external-dependency), not a bare traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dehip import cli
from dehip import generate as generate_mod
from dehip.generate import (
    ModelLoadError,
    _derive_pair_seed,
    generate_drafts,
    load_done_pair_ids,
)
from dehip.schemas import (
    Pair,
    RewriteBundle,
    TextSet,
    read_json,
    read_jsonl,
    write_jsonl,
)

# --- Stubs -------------------------------------------------------------------


class CallCountingModel:
    """A :class:`DraftModel` stub that records every generate call.

    Never touches transformers or the network. Each draft is a deterministic
    function of the prompt AND the effective seed, so a re-generated pair (same
    derived seed) reproduces its exact draft while different pairs (different
    derived seeds) get different drafts -- letting the per-pair-seed contract be
    proven by content, not just by the recorded value.

    Exposes ``device`` so the device-surfacing contract (IMPORTANT 4) can be
    exercised through the same seam without loading torch.
    """

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[str] = []  # prompts, in call order
        self.sampling_seen: list[tuple[float, float, int]] = []

    def generate(self, prompt, *, temperature, top_p, seed):
        self.calls.append(prompt)
        self.sampling_seen.append((temperature, top_p, seed))
        return f"draft::{prompt}::seed={seed}"


class EmptyDraftModel(CallCountingModel):
    """Stub that returns an empty draft for one target prompt, real drafts else.

    Simulates a degenerate generation for a single pair so the empty-draft guard
    (CRITICAL 2) can be exercised without the others tripping.
    """

    def __init__(self, empty_prompt: str) -> None:
        super().__init__()
        self._empty_prompt = empty_prompt

    def generate(self, prompt, *, temperature, top_p, seed):
        super().generate(prompt, temperature=temperature, top_p=top_p, seed=seed)
        if prompt == self._empty_prompt:
            return "   \n  "  # whitespace-only: a degenerate/empty generation
        return f"draft::{prompt}::seed={seed}"


class RuntimeFailingModel:
    """Stub whose generate raises GenerationError (a mid-run model blowup).

    Stands in for the real seam normalizing a torch OOM / device RuntimeError /
    chat-template KeyError to GenerationError (IMPORTANT 1a).
    """

    def generate(self, prompt, *, temperature, top_p, seed):
        raise generate_mod.GenerationError("simulated mid-run OOM")


class KillAfterKModel(CallCountingModel):
    """Stub that raises after ``k`` successful generate calls, simulating a crash.

    The bundles written before the raise are already flushed to disk (per-record
    append), so a resume must pick up from exactly there.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self._k = k

    def generate(self, prompt, *, temperature, top_p, seed):
        if len(self.calls) >= self._k:
            raise KeyboardInterrupt("simulated interrupt")
        return super().generate(prompt, temperature=temperature, top_p=top_p, seed=seed)


class LoadFailingModel:
    """Stub whose generate always raises ModelLoadError (bad repo id / no weights)."""

    def generate(self, prompt, *, temperature, top_p, seed):
        raise ModelLoadError("failed to load instruct model 'nope/nonexistent'")


# --- Fixtures ----------------------------------------------------------------


def _pairs(n: int, corpus: str = "fineweb") -> list[Pair]:
    return [
        Pair(
            pair_id=f"{corpus}-{i:05d}",
            corpus=corpus,
            prompt=f"prompt {i}",
            reference_text=f"human reference {i}",
            source={"dataset": "fineweb", "doc": str(i)},
            register="blog",
            prompt_generator="gpt-5.4-mini",
            word_count=200,
        )
        for i in range(n)
    ]


def _run(model, pairs, run_dir, **kw):
    return generate_drafts(
        pairs,
        model=model,
        run_dir=run_dir,
        run_id="RUN1",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.7,
        top_p=0.95,
        seed=1234,
        printer=lambda *_: None,
        **kw,
    )


# --- Smoke: one draft per pair, full metadata --------------------------------


def test_smoke_run_one_draft_per_pair_with_full_metadata(tmp_path):
    pairs = _pairs(5)
    model = CallCountingModel()
    summary = _run(model, pairs, tmp_path)

    assert summary["pairs"] == 5
    assert summary["generated"] == 5
    assert len(model.calls) == 5

    bundles = read_jsonl(tmp_path / "bundles.jsonl", RewriteBundle)
    assert len(bundles) == 5
    assert {b.pair_id for b in bundles} == {p.pair_id for p in pairs}
    for bundle in bundles:
        assert bundle.draft is not None
        # The recorded seed is the effective per-pair seed derived from the base
        # seed (1234) and the pair_id, NOT the run-wide base seed (IMPORTANT 2).
        effective = _derive_pair_seed(1234, bundle.pair_id)
        assert bundle.draft["text"] == f"draft::{bundle.prompt}::seed={effective}"
        assert bundle.draft["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert bundle.draft["sampling"] == {
            "temperature": 0.7,
            "top_p": 0.95,
            "seed": effective,
        }


# --- Resumability: the call-count proof --------------------------------------


def test_resume_generates_only_missing_drafts_by_call_count(tmp_path):
    pairs = _pairs(6)  # N = 6

    # First run is killed after k = 4 successful drafts.
    killer = KillAfterKModel(k=4)
    with pytest.raises(KeyboardInterrupt):
        _run(killer, pairs, tmp_path)
    assert len(killer.calls) == 4  # 4 drafts flushed to disk before the interrupt

    done, skipped = load_done_pair_ids(tmp_path / "bundles.jsonl")
    assert len(done) == 4
    assert skipped == 0  # clean interrupt boundary (raise before append)

    # Re-run with a fresh counting model: it must make exactly N - k = 2 calls.
    resumer = CallCountingModel()
    summary = _run(resumer, pairs, tmp_path)
    assert len(resumer.calls) == 2  # exactly the two missing pairs
    assert summary["generated"] == 2
    assert summary["already_done"] == 4

    # No duplicate bundles: exactly N pair_ids, each once.
    bundles = read_jsonl(tmp_path / "bundles.jsonl", RewriteBundle)
    pair_ids = [b.pair_id for b in bundles]
    assert len(pair_ids) == 6
    assert len(set(pair_ids)) == 6
    assert set(pair_ids) == {p.pair_id for p in pairs}


# --- Truncated tail robustness -----------------------------------------------


def test_truncated_final_bundle_line_does_not_corrupt_resume(tmp_path):
    pairs = _pairs(4)

    # First run drafts all 4, then we simulate a crash mid-flush by corrupting
    # the final line into a truncated (unparseable) JSON fragment. This variant
    # keeps a trailing newline on the fragment (an already-terminated but
    # unparseable tail); the torn-boundary variant below drops it. Both must
    # regenerate the pair exactly once.
    _run(CallCountingModel(), pairs, tmp_path)
    bundles_path = tmp_path / "bundles.jsonl"
    lines = bundles_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    # Truncate the last line to a partial JSON object (crash mid-write).
    lines[-1] = lines[-1][: len(lines[-1]) // 2]
    bundles_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The truncated tail is skipped-and-counted, not fatal.
    done, skipped = load_done_pair_ids(bundles_path)
    assert skipped == 1
    assert len(done) == 3  # only the 3 intact bundles are "done"

    # Re-run: exactly the one truncated pair is regenerated (1 generate call),
    # no crash, and the final state has all 4 pairs exactly once.
    resumer = CallCountingModel()
    summary = _run(resumer, pairs, tmp_path)
    assert len(resumer.calls) == 1
    assert summary["generated"] == 1
    assert summary["skipped_truncated"] == 1

    # The manifest assembly tolerated the partial tail and the texts file reflects
    # all 4 pairs exactly once.
    texts = _read_texts(tmp_path / "draft-texts.jsonl")
    assert len(texts) == 4
    assert set(texts) == {p.pair_id for p in pairs}


def test_torn_tail_without_trailing_newline_regenerates_pair_once(tmp_path):
    """CRITICAL 1: a torn final record with NO trailing newline is repaired.

    A real crash between an append's record write and its newline write leaves
    the last record with no ``\\n``. Without the repair, the next append would
    concatenate onto the fragment into one unparseable line, silently dropping
    the regenerated pair from bundles/manifest/draft-texts while exiting 0. The
    resume must repair the boundary, regenerate the torn pair exactly once, leave
    no unparseable concatenated line, and produce a manifest with every pair.
    """
    pairs = _pairs(4)

    _run(CallCountingModel(), pairs, tmp_path)
    bundles_path = tmp_path / "bundles.jsonl"
    lines = bundles_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    torn_pair_id = json.loads(lines[-1])["pair_id"]
    # Truncate the last line AND drop the trailing newline: a genuine torn tail.
    lines[-1] = lines[-1][: len(lines[-1]) // 2]
    bundles_path.write_text("\n".join(lines), encoding="utf-8")  # no final "\n"
    assert not bundles_path.read_text(encoding="utf-8").endswith("\n")

    # Only the 3 intact records are "done"; the torn one is regenerable.
    done, skipped = load_done_pair_ids(bundles_path)
    assert skipped == 1
    assert done == {p.pair_id for p in pairs} - {torn_pair_id}

    torn_prompt = next(p.prompt for p in pairs if p.pair_id == torn_pair_id)
    resumer = CallCountingModel()
    summary = _run(resumer, pairs, tmp_path)
    assert resumer.calls == [torn_prompt]  # exactly the one torn pair regenerated
    assert summary["generated"] == 1

    # The torn fragment survives on its OWN line (skippable), and the regenerated
    # record is a clean parseable line -- crucially, the repair prevented the
    # regenerated JSON from concatenating onto the fragment into one line. Split
    # every line into parseable vs fragment: exactly one fragment (the torn tail),
    # and the fragment carries no valid JSON tail (no concatenation happened).
    raw_lines = [
        ln for ln in bundles_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    fragments = []
    for line in raw_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            fragments.append(line)
    assert len(fragments) == 1  # only the original torn fragment, nothing appended
    # The regenerated pair is one of the clean, parseable lines exactly once.
    parseable_ids = [
        json.loads(ln)["pair_id"] for ln in raw_lines if ln not in fragments
    ]
    assert parseable_ids.count(torn_pair_id) == 1

    # _read_all_bundles tolerates the fragment; the manifest has every pair once.
    manifest: TextSet = read_json(tmp_path / "draft.manifest.json", TextSet)
    assert sorted(manifest.pair_ids) == sorted(p.pair_id for p in pairs)


# --- Metadata on disk --------------------------------------------------------


def test_every_bundle_records_model_id_and_sampling(tmp_path):
    pairs = _pairs(3)
    _run(CallCountingModel(), pairs, tmp_path)

    for raw in (tmp_path / "bundles.jsonl").read_text().splitlines():
        record = json.loads(raw)
        draft = record["draft"]
        assert draft["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert set(draft["sampling"]) == {"temperature", "top_p", "seed"}
        # The persisted seed is the effective per-pair seed, not the base seed.
        assert draft["sampling"]["seed"] == _derive_pair_seed(1234, record["pair_id"])


# --- Manifest is scorable shape ----------------------------------------------


def test_manifest_points_at_draft_texts_and_is_scorable_shape(tmp_path):
    pairs = _pairs(3)
    _run(CallCountingModel(), pairs, tmp_path)

    manifest: TextSet = read_json(tmp_path / "draft.manifest.json", TextSet)
    assert manifest.role == "instruct_draft"
    assert manifest.corpus == "fineweb"
    assert sorted(manifest.pair_ids) == sorted(p.pair_id for p in pairs)
    # The manifest points at the texts JSONL a scorer resolves via texts_path.
    assert manifest.provenance["texts_path"] == "draft-texts.jsonl"

    texts = _read_texts(tmp_path / "draft-texts.jsonl")
    assert set(texts) == set(manifest.pair_ids)
    # Report's resolver uses exactly this convention (provenance.texts_path).
    from dehip.report import _texts_path_for

    resolved = _texts_path_for(tmp_path / "draft.manifest.json", manifest.provenance)
    assert resolved == tmp_path / "draft-texts.jsonl"


# --- Model-load failure -> exit 3 --------------------------------------------


def test_model_load_failure_maps_to_exit_3(tmp_path, monkeypatch):
    """A ModelLoadError from the seam maps to CLI exit 3, not a bare traceback."""
    pairs = _pairs(2)
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(pairs, corpus_path)

    # Inject the load-failing model as the seam so the CLI's real transformers
    # class is never constructed and no weights/network are touched.
    monkeypatch.setattr(
        generate_mod, "TransformersDraftModel", lambda model_id: LoadFailingModel()
    )

    code = cli.main(
        ["generate", "--corpus", str(corpus_path), "--out", str(tmp_path / "run")]
    )
    assert code == cli.EXIT_EXTERNAL_DEP == 3


def test_missing_corpus_maps_to_exit_2(tmp_path):
    """A missing corpus file is bad input -> exit 2, before any model touch."""
    code = cli.main(["generate", "--corpus", str(tmp_path / "absent.jsonl")])
    assert code == cli.EXIT_VALIDATION == 2


def test_cli_generate_end_to_end_with_stub_seam(tmp_path, monkeypatch, capsys):
    """The full CLI path writes a bundle per pair and echoes an ok JSON summary."""
    pairs = _pairs(3)
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(pairs, corpus_path)
    run_dir = tmp_path / "run"

    monkeypatch.setattr(
        generate_mod, "TransformersDraftModel", lambda model_id: CallCountingModel()
    )

    code = cli.main(
        ["generate", "--corpus", str(corpus_path), "--out", str(run_dir)]
    )
    assert code == cli.EXIT_SUCCESS == 0

    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["command"] == "generate"
    assert summary["status"] == "ok"
    assert summary["pairs"] == 3
    assert summary["generated"] == 3

    bundles = read_jsonl(run_dir / "bundles.jsonl", RewriteBundle)
    assert len(bundles) == 3


# --- CRITICAL 2: empty draft never enters the scored set ---------------------


def test_empty_draft_is_not_written_and_not_marked_done(tmp_path):
    """An empty/degenerate draft raises, is not persisted, and regenerates later.

    A blank draft must never be written as a finished bundle (it would be
    counted, manifested, and read downstream as a real blank draft, and read as
    done on resume). The chosen behavior is a loud failure: EmptyDraftError, no
    bundle for that pair, so a later re-run (with a healthy model) regenerates it.
    """
    pairs = _pairs(3)
    target = pairs[1]

    model = EmptyDraftModel(empty_prompt=target.prompt)
    with pytest.raises(generate_mod.EmptyDraftError):
        _run(model, pairs, tmp_path)

    bundles_path = tmp_path / "bundles.jsonl"
    # The empty pair was NOT persisted; only the pairs drafted before it are on
    # disk (generation is in input order, so pair[0] only).
    done, skipped = load_done_pair_ids(bundles_path)
    assert target.pair_id not in done  # never marked done
    persisted = read_jsonl(bundles_path, RewriteBundle)
    assert all(b.pair_id != target.pair_id for b in persisted)

    # A later resume with a healthy model regenerates the missing pair(s),
    # including the previously-empty one, and it enters the scored set exactly
    # once with a non-blank draft.
    healthy = CallCountingModel()
    summary = _run(healthy, pairs, tmp_path)
    assert target.prompt in healthy.calls
    final = read_jsonl(bundles_path, RewriteBundle)
    assert sorted(b.pair_id for b in final) == sorted(p.pair_id for p in pairs)
    target_bundle = next(b for b in final if b.pair_id == target.pair_id)
    assert target_bundle.draft is not None
    assert target_bundle.draft["text"].strip()  # non-blank
    assert summary["generated"] >= 1


# --- IMPORTANT 1: exit-code discipline ---------------------------------------


def test_generation_runtime_error_maps_to_exit_3(tmp_path, monkeypatch):
    """A mid-run GenerationError (OOM-like) maps to CLI exit 3, not exit 1."""
    pairs = _pairs(2)
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(pairs, corpus_path)

    monkeypatch.setattr(
        generate_mod, "TransformersDraftModel", lambda model_id: RuntimeFailingModel()
    )
    code = cli.main(
        ["generate", "--corpus", str(corpus_path), "--out", str(tmp_path / "run")]
    )
    assert code == cli.EXIT_EXTERNAL_DEP == 3


def test_corpus_dir_maps_to_exit_2(tmp_path):
    """--corpus pointing at a directory is bad input -> exit 2, not exit 1."""
    a_dir = tmp_path / "corpus_dir"
    a_dir.mkdir()
    code = cli.main(["generate", "--corpus", str(a_dir)])
    assert code == cli.EXIT_VALIDATION == 2


# --- IMPORTANT 2: per-pair seed ----------------------------------------------


def test_per_pair_seed_is_distinct_and_reproducible(tmp_path):
    """Distinct effective seeds per pair; a re-generated pair reproduces its draft.

    Two pairs get different effective seeds (decorrelated), the recorded seed is
    the effective per-pair seed (not the base seed), and re-generating a single
    pair reproduces its original draft byte-for-byte (determinism preserved).
    """
    pairs = _pairs(2)
    run_a = tmp_path / "a"
    _run(CallCountingModel(), pairs, run_a)
    bundles_a = {
        b.pair_id: b for b in read_jsonl(run_a / "bundles.jsonl", RewriteBundle)
    }

    seeds = {pid: b.draft["sampling"]["seed"] for pid, b in bundles_a.items()}
    # Distinct per pair.
    assert len(set(seeds.values())) == 2
    # Recorded seed equals the effective per-pair seed (base seed 1234).
    for pid, seed in seeds.items():
        assert seed == _derive_pair_seed(1234, pid)

    # Re-generating a single pair (in a fresh run dir) reproduces its draft.
    single = [pairs[0]]
    run_b = tmp_path / "b"
    _run(CallCountingModel(), single, run_b)
    bundle_b = read_jsonl(run_b / "bundles.jsonl", RewriteBundle)[0]
    assert bundle_b.draft["text"] == bundles_a[pairs[0].pair_id].draft["text"]
    assert bundle_b.draft["sampling"]["seed"] == seeds[pairs[0].pair_id]


# --- IMPORTANT 3: manifest/bundle drift --------------------------------------


def test_stale_bundle_pair_id_fails_loudly(tmp_path):
    """A bundle pair_id absent from the current corpus fails loudly (exit 2)."""
    pairs = _pairs(3)
    # First run over all 3 builds a bundles file.
    _run(CallCountingModel(), pairs, tmp_path)

    # Resume with a SMALLER corpus (a different/narrower run): the persisted
    # bundles now carry pair_ids the current corpus lacks -> loud drift error.
    narrower = pairs[:1]
    with pytest.raises(generate_mod.CorpusDriftError) as exc:
        _run(CallCountingModel(), narrower, tmp_path)
    # The stray ids are named.
    assert pairs[1].pair_id in str(exc.value)
    assert pairs[2].pair_id in str(exc.value)

    # And through the CLI it is exit 2 (CorpusDriftError subclasses ValueError).
    corpus_path = tmp_path / "narrow.jsonl"
    write_jsonl(narrower, corpus_path)
    import dehip.generate as gm

    class _NoopModel:
        def generate(self, prompt, *, temperature, top_p, seed):
            return f"draft::{prompt}"

    import unittest.mock as mock

    with mock.patch.object(gm, "TransformersDraftModel", lambda mid: _NoopModel()):
        code = cli.main(
            ["generate", "--corpus", str(corpus_path), "--out", str(tmp_path)]
        )
    assert code == cli.EXIT_VALIDATION == 2


# --- IMPORTANT 4: device surfaced --------------------------------------------


def test_device_is_surfaced_in_summary(tmp_path):
    """The selected model device is recorded in the run summary (IMPORTANT 4)."""
    pairs = _pairs(2)
    summary = _run(CallCountingModel(), pairs, tmp_path)
    assert summary["device"] == "cpu"  # the stub's device, surfaced


def test_device_is_printed(tmp_path):
    """A silent device fallback is visible: the device is printed via the printer."""
    pairs = _pairs(1)
    lines: list[str] = []
    generate_drafts(
        pairs,
        model=CallCountingModel(),
        run_dir=tmp_path,
        run_id="RUN1",
        model_id="m",
        seed=1234,
        printer=lines.append,
    )
    assert any("device" in line and "cpu" in line for line in lines)


# --- IMPORTANT 5: test-gap closures ------------------------------------------


def test_sampling_seen_records_exact_effective_values(tmp_path):
    """The stub's sampling_seen must equal the expected (temp, top_p, seed) per call.

    Guards against a record-metadata-but-pass-a-different-value divergence: the
    seed the model is CALLED with must be the effective per-pair seed, and the
    temperature/top_p must be exactly what was requested.
    """
    pairs = _pairs(3)
    model = CallCountingModel()
    _run(model, pairs, tmp_path)

    # Calls happen in input (remaining) order over all 3 pairs.
    expected = [(0.7, 0.95, _derive_pair_seed(1234, p.pair_id)) for p in pairs]
    assert model.sampling_seen == expected


def test_empty_pairs_raises(tmp_path):
    """The zero-pairs guard is non-deletable (generate.py:419)."""
    with pytest.raises(ValueError, match="zero pairs"):
        _run(CallCountingModel(), [], tmp_path)


def test_heterogeneous_corpus_raises(tmp_path):
    """The homogeneity check is non-deletable (generate.py:424)."""
    mixed = _pairs(2, corpus="fineweb") + _pairs(2, corpus="personal")
    with pytest.raises(ValueError, match="homogeneous corpus"):
        _run(CallCountingModel(), mixed, tmp_path)


def test_idempotent_rerun_makes_no_calls_and_appends_nothing(tmp_path):
    """Re-running a fully complete run makes zero generate calls and zero appends."""
    pairs = _pairs(4)
    _run(CallCountingModel(), pairs, tmp_path)
    bundles_path = tmp_path / "bundles.jsonl"
    line_count_before = len(bundles_path.read_text(encoding="utf-8").splitlines())

    rerun = CallCountingModel()
    summary = _run(rerun, pairs, tmp_path)
    assert rerun.calls == []  # zero generate calls
    assert summary["generated"] == 0
    assert summary["already_done"] == 4
    line_count_after = len(bundles_path.read_text(encoding="utf-8").splitlines())
    assert line_count_after == line_count_before  # zero appended bundles


# --- Helpers -----------------------------------------------------------------


def _read_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        out[record["pair_id"]] = record["text"]
    return out
