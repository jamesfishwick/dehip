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
- ``test_every_bundle_records_model_id_and_sampling`` -- model id + sampling
  present in every persisted bundle record (redundant belt-and-suspenders over
  the smoke test, asserted directly on disk).
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
    function of the prompt so a re-run over the same pair would produce the same
    text (letting a duplicate be spotted by content as well as by id).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []  # prompts, in call order
        self.sampling_seen: list[tuple[float, float, int]] = []

    def generate(self, prompt, *, temperature, top_p, seed):
        self.calls.append(prompt)
        self.sampling_seen.append((temperature, top_p, seed))
        return f"draft::{prompt}"


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
        assert bundle.draft["text"] == f"draft::{bundle.prompt}"
        assert bundle.draft["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert bundle.draft["sampling"] == {
            "temperature": 0.7,
            "top_p": 0.95,
            "seed": 1234,
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
    # the final line into a truncated (unparseable) JSON fragment.
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


# --- Metadata on disk --------------------------------------------------------


def test_every_bundle_records_model_id_and_sampling(tmp_path):
    pairs = _pairs(3)
    _run(CallCountingModel(), pairs, tmp_path)

    for raw in (tmp_path / "bundles.jsonl").read_text().splitlines():
        record = json.loads(raw)
        draft = record["draft"]
        assert draft["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert set(draft["sampling"]) == {"temperature", "top_p", "seed"}
        assert draft["sampling"]["seed"] == 1234


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
