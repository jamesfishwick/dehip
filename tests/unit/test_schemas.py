"""Unit tests for dehip entity schemas and serialization."""

import json

import pytest

from dehip.schemas import (
    SCHEMA_VERSION,
    JudgeVerdict,
    MetricReport,
    Pair,
    RewriteBundle,
    SchemaValidationError,
    SchemaVersionError,
    TextSet,
    content_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _pair(pair_id: str = "fineweb-1") -> Pair:
    return Pair(
        pair_id=pair_id,
        corpus="fineweb",
        prompt="Write a short blog post about tea.",
        reference_text="Tea is a drink made by steeping leaves in hot water.",
        source={"dataset": "fineweb", "doc_id": "abc123"},
        register="blog",
        prompt_generator="gpt-5.4-mini",
        word_count=250,
    )


def _text_set() -> TextSet:
    return TextSet(
        set_id="fineweb-400-rewrite-k2",
        role="rewrite",
        corpus="fineweb",
        pair_ids=["fineweb-1", "fineweb-2"],
        provenance={"run_id": "run-2026"},
        round=2,
    )


def _bundle() -> RewriteBundle:
    return RewriteBundle(
        run_id="run-2026",
        pair_id="fineweb-1",
        prompt="Write a short blog post about tea.",
        draft={
            "text": "draft body",
            "model_id": "instruct-8b",
            "sampling": {"temperature": 0.7, "top_p": 0.9, "seed": 42},
        },
        rounds=[{"k": 1, "text": "round one", "flags": []}],
        final_round=1,
        degeneration={
            "empty": False,
            "length_ratio": 1.1,
            "repetition": False,
            "non_ascii_burst": False,
            "hard_tripped": False,
        },
        adapter_id="hip-adapter-v1",
        hip_config={"rounds": 2},
        requested_k=2,
    )


def _verdict() -> JudgeVerdict:
    return JudgeVerdict(
        pair_id="fineweb-1",
        dimension="overall",
        judge_model="gpt-5.4-mini",
        order="model_first",
        raw_response="A is better.",
        choice="A",
        model_won=True,
        retry_count=0,
    )


def _report() -> MetricReport:
    return MetricReport(
        report_id="report-1",
        compared={
            "candidate_set": "fineweb-400-rewrite-k2",
            "reference_set": "fineweb-400-human",
            "candidate_n": 400,
            "reference_n": 400,
        },
        config={
            "seed": 42,
            "judge_model": "gpt-5.4-mini",
            "embedder_id": "e5-base",
            "tokenizer_id": "gpt2",
            "mmd_bandwidth": 1.0,
            "thresholds": {"min_n": 30},
        },
        mmd=0.0123,
        token_l2=0.045,
        jmq={
            "overall": {"score": 0.52, "wins": 52, "losses": 48, "invalid": 0, "n": 100}
        },
        caveats=["small-N warning"],
        timestamps={
            "started": "2026-08-01T00:00:00Z",
            "finished": "2026-08-01T00:05:00Z",
        },
    )


# --- content SHA-256 ---------------------------------------------------------


def test_content_sha256_is_stable_and_utf8():
    assert content_sha256("hello") == content_sha256("hello")
    # Known digest for "hello" (UTF-8) guards against encoding drift.
    assert content_sha256("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert content_sha256("a") != content_sha256("b")


# --- round-trip: JSONL per-text records --------------------------------------


@pytest.mark.parametrize(
    "factory, cls",
    [(_pair, Pair), (_bundle, RewriteBundle), (_verdict, JudgeVerdict)],
)
def test_jsonl_roundtrip_byte_exact(tmp_path, factory, cls):
    records = [factory()]
    path = tmp_path / "records.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path, cls)
    assert loaded == records
    # Re-serializing the loaded record yields byte-identical output.
    rewritten = tmp_path / "again.jsonl"
    write_jsonl(loaded, rewritten)
    assert path.read_bytes() == rewritten.read_bytes()


def test_jsonl_multiple_records_roundtrip(tmp_path):
    records = [_pair("fineweb-1"), _pair("fineweb-2")]
    path = tmp_path / "pairs.jsonl"
    write_jsonl(records, path)
    assert read_jsonl(path, Pair) == records


# --- round-trip: JSON per-run documents --------------------------------------


@pytest.mark.parametrize(
    "factory, cls", [(_text_set, TextSet), (_report, MetricReport)]
)
def test_json_roundtrip_byte_exact(tmp_path, factory, cls):
    record = factory()
    path = tmp_path / "doc.json"
    write_json(record, path)
    loaded = read_json(path, cls)
    assert loaded == record
    rewritten = tmp_path / "again.json"
    write_json(loaded, rewritten)
    assert path.read_bytes() == rewritten.read_bytes()


# --- unknown / missing schema_version raises ---------------------------------


def test_read_jsonl_unknown_version_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    raw = {
        "pair_id": "fineweb-1",
        "corpus": "fineweb",
        "prompt": "p",
        "reference_text": "r",
        "source": {},
        "register": "blog",
        "prompt_generator": "m",
        "word_count": 200,
        "schema_version": 2,
    }
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        read_jsonl(path, Pair)


def test_read_json_missing_version_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"set_id": "s", "role": "rewrite"}), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        read_json(path, TextSet)


def test_read_jsonl_unexpected_field_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    raw = {"schema_version": SCHEMA_VERSION, "surprise": 1}
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        read_jsonl(path, Pair)


# --- TextSet role / corpus validation ----------------------------------------


def test_textset_rejects_bad_role():
    with pytest.raises(SchemaValidationError):
        TextSet(set_id="s", role="not_a_role", corpus="fineweb", pair_ids=["a"])


def test_textset_rejects_empty_corpus():
    with pytest.raises(SchemaValidationError):
        TextSet(set_id="s", role="rewrite", corpus="", pair_ids=["a"])


def test_textset_round_only_for_rewrite():
    with pytest.raises(SchemaValidationError):
        TextSet(
            set_id="s",
            role="human_reference",
            corpus="fineweb",
            pair_ids=["a"],
            round=2,
        )


def test_read_json_bad_manifest_role_raises(tmp_path):
    path = tmp_path / "manifest.json"
    raw = {
        "set_id": "s",
        "role": "bogus",
        "corpus": "fineweb",
        "pair_ids": ["a"],
        "provenance": {},
        "round": None,
        "schema_version": SCHEMA_VERSION,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        read_json(path, TextSet)


# --- JudgeVerdict enum validation --------------------------------------------


def test_judgeverdict_rejects_bad_dimension():
    with pytest.raises(SchemaValidationError):
        JudgeVerdict(
            pair_id="p",
            dimension="vibes",
            judge_model="m",
            order="model_first",
            raw_response="x",
            choice="A",
            retry_count=0,
        )
