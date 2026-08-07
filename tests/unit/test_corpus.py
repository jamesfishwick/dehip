"""Unit tests for the corpus builder (issue #5, FR-010, R8).

All tests mock the two external dependencies: the FineWeb stream is injected as
a plain iterable of fake documents, and prompt generation is a stub client that
returns canned prompts and counts its calls. No real dataset download and no
real OpenAI call happens.
"""

from __future__ import annotations

import pytest

from dehip import corpus
from dehip.schemas import Pair, read_json, read_jsonl


class StubPromptClient:
    """Injectable :class:`corpus.PromptClient` that never touches the network.

    Returns a deterministic canned prompt derived from the call index and
    records every ``(document, instruction, model)`` it saw, so a test can assert
    both the generated content and the number of generation calls.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def generate_prompt(self, document: str, *, instruction: str, model: str) -> str:
        self.calls.append((document, instruction, model))
        return f"reverse-prompt #{len(self.calls)}"


def _fake_docs(n: int, *, words: int = 300) -> list[dict]:
    """A stream of ``n`` fake FineWeb records that pass the qualification filter."""
    body = " ".join(f"word{i}" for i in range(words))
    return [
        {
            "text": f"Human blog document number {i}. {body}",
            "language": "en",
            "id": f"doc-{i}",
            "url": f"https://example.com/blog/{i}",
        }
        for i in range(n)
    ]


# --- Smoke tier: schema-valid fineweb Pairs ---------------------------------


def test_smoke_tier_produces_schema_valid_pairs(tmp_path):
    out = tmp_path / "fineweb.jsonl"
    client = StubPromptClient()
    docs = _fake_docs(60)  # > smoke target (50) so sampling has a surplus

    pairs = corpus.build_fineweb_corpus(
        tier="smoke",
        out_path=out,
        client=client,
        seed=7,
        docs=docs,
        confirm=True,
    )

    assert len(pairs) == corpus.TIER_SIZES["smoke"] == 50
    # Every pair is schema-complete with the FR-010 required fields populated.
    for pair in pairs:
        assert isinstance(pair, Pair)
        assert pair.corpus == "fineweb"
        assert pair.prompt.startswith("reverse-prompt #")
        assert pair.reference_text
        assert pair.source["dataset"] == corpus.FINEWEB_DATASET
        assert pair.register in {"blog", "news"}
        assert pair.prompt_generator == corpus.DEFAULT_PROMPT_GENERATOR
        assert corpus.WORD_COUNT_MIN <= pair.word_count <= corpus.WORD_COUNT_MAX
    # One generation call per sampled doc.
    assert len(client.calls) == 50
    # Round-trip through the on-disk JSONL parses as valid Pairs.
    on_disk = read_jsonl(out, Pair)
    assert {p.pair_id for p in on_disk} == {p.pair_id for p in pairs}


def test_prompt_variant_rotation_diversifies_instructions():
    # The rotation covers every variant across enough documents.
    seen = {corpus._render_instruction(i) for i in range(len(corpus.PROMPT_VARIANTS))}
    assert seen == set(corpus.PROMPT_VARIANTS)


def test_seeded_sample_is_reproducible(tmp_path):
    docs = _fake_docs(60)
    client_a = StubPromptClient()
    client_b = StubPromptClient()
    pairs_a = corpus.build_fineweb_corpus(
        tier="smoke", out_path=tmp_path / "a.jsonl", client=client_a,
        seed=123, docs=docs, confirm=True,
    )
    pairs_b = corpus.build_fineweb_corpus(
        tier="smoke", out_path=tmp_path / "b.jsonl", client=client_b,
        seed=123, docs=docs, confirm=True,
    )
    assert [p.reference_text for p in pairs_a] == [p.reference_text for p in pairs_b]


def test_sample_documents_bounds_an_unbounded_stream():
    # The real FineWeb stream is effectively endless. list(qualified) would hang
    # (and OOM on the real dataset); islice(pool_cap) must bound it. This test
    # would hang forever if sample_documents materialized the whole stream.
    def endless():
        i = 0
        while True:
            yield {
                "text": f"doc {i}",
                "word_count": 200,
                "source": {"seq": i},
                "register": "blog",
            }
            i += 1

    sampled = corpus.sample_documents(endless(), target=5, seed=42, pool_cap=100)
    assert len(sampled) == 5
    # Reproducible over the deterministic prefix.
    again = corpus.sample_documents(endless(), target=5, seed=42, pool_cap=100)
    assert [d["text"] for d in sampled] == [d["text"] for d in again]
    # Every pick came from within the cap (the first pool_cap qualified docs).
    seqs = [d["source"]["seq"] for d in sampled]
    assert all(0 <= s < 100 for s in seqs)


def test_pool_cap_scales_with_target():
    assert corpus.pool_cap_for(50) == 500  # MIN_POOL_SIZE floor
    assert corpus.pool_cap_for(2000) == 20_000  # target * factor


# --- Personal side corpus ----------------------------------------------------


def test_personal_pairs_are_tagged_and_complete(tmp_path):
    out = tmp_path / "personal.jsonl"
    client = StubPromptClient()
    documents = [
        ("My first published post. " + " ".join(f"w{i}" for i in range(200)),
         {"url": "https://jamesfishwick.com/post-1"}),
        ("A second essay of mine. " + " ".join(f"w{i}" for i in range(200)),
         {"path": "posts/essay-2.md"}),
    ]

    pairs = corpus.build_personal_corpus(
        sources=[],  # ignored when documents are injected
        out_path=out,
        client=client,
        documents=documents,
        confirm=True,
    )

    assert len(pairs) == 2
    for pair in pairs:
        assert pair.corpus == "personal"  # the FR-010 exclusion tag
        assert pair.prompt  # prompt present (reconstructed)
        assert pair.prompt_generator == corpus.DEFAULT_PROMPT_GENERATOR
        assert pair.reference_text
        assert pair.word_count > 0
    assert len(client.calls) == 2


def test_manifest_excludes_personal_via_corpus_tag(tmp_path):
    client = StubPromptClient()
    documents = [("A post. " + " ".join(f"w{i}" for i in range(200)), {"url": "u"})]
    pairs = corpus.build_personal_corpus(
        sources=[], out_path=tmp_path / "p.jsonl", client=client,
        documents=documents, confirm=True,
    )
    manifest = corpus.write_human_reference_manifest(
        pairs, set_id="personal-human", manifest_path=tmp_path / "p.manifest.json"
    )
    # The manifest carries corpus="personal" so the report layer can exclude it.
    assert manifest.corpus == "personal"
    assert manifest.role == "human_reference"
    reloaded = read_json(tmp_path / "p.manifest.json", type(manifest))
    assert reloaded.corpus == "personal"
    assert reloaded.pair_ids == [p.pair_id for p in pairs]


# --- Resumability ------------------------------------------------------------


def test_interrupted_run_resumes_without_duplication(tmp_path):
    out = tmp_path / "fineweb.jsonl"
    docs = _fake_docs(60)

    # First run is "interrupted" after producing a partial file: simulate by
    # building the smoke corpus, then truncating the JSONL to its first 10 pairs.
    first_client = StubPromptClient()
    corpus.build_fineweb_corpus(
        tier="smoke", out_path=out, client=first_client,
        seed=7, docs=docs, confirm=True,
    )
    all_lines = out.read_text(encoding="utf-8").splitlines()
    assert len(all_lines) == 50
    out.write_text("\n".join(all_lines[:10]) + "\n", encoding="utf-8")

    # Resume with the SAME seed and docs so the plan (and pair_ids) matches.
    resume_client = StubPromptClient()
    pairs = corpus.build_fineweb_corpus(
        tier="smoke", out_path=out, client=resume_client,
        seed=7, docs=docs, confirm=True,
    )

    # The 10 already-done pairs are kept and NOT regenerated.
    assert len(resume_client.calls) == 40  # only the remaining 40
    # No duplicate pair_ids on disk, and the full set is restored.
    on_disk = read_jsonl(out, Pair)
    ids = [p.pair_id for p in on_disk]
    assert len(ids) == len(set(ids)) == 50
    assert len(pairs) == 50


def test_fully_complete_run_makes_no_generation_calls(tmp_path):
    out = tmp_path / "fineweb.jsonl"
    docs = _fake_docs(60)
    corpus.build_fineweb_corpus(
        tier="smoke", out_path=out, client=StubPromptClient(),
        seed=7, docs=docs, confirm=True,
    )
    # A second run over the complete file regenerates nothing.
    resume_client = StubPromptClient()
    pairs = corpus.build_fineweb_corpus(
        tier="smoke", out_path=out, client=resume_client,
        seed=7, docs=docs, confirm=True,
    )
    assert resume_client.calls == []
    assert len(pairs) == 50


# --- Doc shortage after filtering -------------------------------------------


def test_doc_shortage_reports_shortfall(tmp_path):
    # Only 5 qualified docs but smoke needs 50.
    docs = _fake_docs(5)
    with pytest.raises(corpus.DocShortageError) as exc:
        corpus.build_fineweb_corpus(
            tier="smoke", out_path=tmp_path / "x.jsonl",
            client=StubPromptClient(), docs=docs, confirm=True,
        )
    assert exc.value.requested == 50
    assert exc.value.available == 5
    assert "shortage" in str(exc.value)


def test_doc_shortage_maps_to_nonzero_cli_exit(tmp_path, monkeypatch):
    # The CLI handler turns a shortage into a non-zero exit without a real
    # dataset download: patch stream_fineweb to yield too few docs.
    from dehip import cli

    monkeypatch.setattr(corpus, "stream_fineweb", lambda **_: _fake_docs(3))
    args = _build_corpus_args(cli, tier="smoke", corpus="fineweb")
    rc = cli._run_build_corpus(args)
    assert rc != 0


# --- Cost gate ---------------------------------------------------------------


def test_cost_gate_blocks_above_threshold_without_confirm(tmp_path):
    docs = _fake_docs(60)
    with pytest.raises(corpus.CostThresholdError):
        corpus.build_fineweb_corpus(
            tier="smoke", out_path=tmp_path / "x.jsonl",
            client=StubPromptClient(), docs=docs,
            confirm=False, threshold_usd=0.0,  # force the gate
        )


def test_cost_gate_reports_estimate(capsys):
    lines: list[str] = []
    estimate = corpus.cost_preflight(50, confirm=True, printer=lines.append)
    assert estimate["calls"] == 50
    assert any("50 calls" in line for line in lines)


def _build_corpus_args(cli_module, **overrides):
    """Parse a build-corpus argv into a Namespace for direct handler tests."""
    parser = cli_module._build_parser()
    argv = ["build-corpus"]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    return parser.parse_args(argv)
