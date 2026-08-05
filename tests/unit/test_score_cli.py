"""Tests for the `dehip score` CLI wiring (issue #10).

The metric-computing paths run in-process with stub seams monkeypatched onto
dehip.report / dehip.metrics so no real model or network is touched. The
recompute path (--recompute-jmq-from) needs no seams at all, so it also runs
cleanly through argparse dispatch.
"""

from __future__ import annotations

import json

import numpy as np

from dehip import cli
from dehip import report as report_mod
from dehip.metrics.jmq import DIMENSION_ORDER, run_judging
from dehip.schemas import TextSet, write_json

# --- Stubs (mirrors of test_report's) ----------------------------------------


class StubEmbedder:
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


class ScriptedJudge:
    def __init__(self, reply: str = "A") -> None:
        self._reply = reply
        self.calls = 0

    def judge(self, rendered_prompt: str, *, model: str) -> str:
        self.calls += 1
        return self._reply


# --- Manifest / texts fixtures -----------------------------------------------


def _write_manifest(tmp_path, name, set_id, role, pair_ids, texts_by_id):
    """Write a TextSet manifest + its sibling {pair_id, text} JSONL."""
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


def _corpus(tmp_path, n):
    pair_ids = [f"fineweb-{i}" for i in range(n)]
    cand = {pid: f"model output {i} words here" for i, pid in enumerate(pair_ids)}
    ref = {pid: f"human reference {i} words here" for i, pid in enumerate(pair_ids)}
    prompts = {pid: f"prompt {i}" for i, pid in enumerate(pair_ids)}
    cand_manifest = _write_manifest(
        tmp_path, "cand", "cand-set", "instruct_draft", pair_ids, cand
    )
    ref_manifest = _write_manifest(
        tmp_path, "ref", "ref-set", "human_reference", pair_ids, ref
    )
    prompts_path = tmp_path / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as fh:
        for pid in pair_ids:
            fh.write(json.dumps({"pair_id": pid, "prompt": prompts[pid]}) + "\n")
    return cand_manifest, ref_manifest, str(prompts_path)


def _patch_seams(monkeypatch, embedder=None, judge=None):
    """Swap the real embedder/judge/tokenizer constructors for stubs."""
    embedder = embedder or StubEmbedder()
    judge = judge or ScriptedJudge()
    monkeypatch.setattr(
        "dehip.metrics.embeddings.TransformersEmbedder", lambda *a, **k: embedder
    )
    monkeypatch.setattr(
        "dehip.metrics.jmq.OpenAIJudgeClient", lambda *a, **k: judge
    )
    # token_l2 default tokenizer -> stub. score() passes tokenizer=None from the
    # CLI path, so patch the resolver's default.
    monkeypatch.setattr(
        "dehip.metrics.token_l2.Qwen3Tokenizer", lambda *a, **k: StubTokenizer()
    )
    return embedder, judge


# --- Full score through the CLI (all three metrics, one invocation) ----------


def test_cli_score_composes_report(tmp_path, monkeypatch):
    cand, ref, prompts = _corpus(tmp_path, 4)
    embedder, judge = _patch_seams(monkeypatch)
    out = tmp_path / "report.json"

    rc = cli.main(
        [
            "--seed",
            "7",
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--judge",
            "gpt-5.4-mini",
            "--out",
            str(out),
            "--yes",
        ]
    )
    assert rc == cli.EXIT_SUCCESS

    written = json.loads(out.read_text())
    assert not np.isnan(written["mmd"])
    assert not np.isnan(written["token_l2"])
    assert written["jmq"]["overall"]["n"] == 4
    assert written["config"]["seed"] == 7
    assert written["config"]["embedder_id"] == "nvidia/llama-embed-nemotron-8b"
    # A markdown companion was written next to the JSON.
    assert (tmp_path / "report.md").exists()
    # Bias audit present.
    assert "bias_audit" in written["jmq"]


def test_cli_metric_subset(tmp_path, monkeypatch):
    cand, ref, _ = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)
    out = tmp_path / "r.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--metrics",
            "token_l2",
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    assert not np.isnan(written["token_l2"])
    assert np.isnan(written["mmd"])
    assert written["jmq"] == {}


# --- Recompute path: zero judge calls, through CLI dispatch ------------------


def test_cli_recompute_jmq_no_judge_calls(tmp_path, monkeypatch):
    # First produce verdicts via a direct run (stub judge), then recompute via CLI.
    pair_ids = [f"fineweb-{i}" for i in range(5)]
    from dehip.metrics.jmq import JudgePair

    pairs = [
        JudgePair(
            pair_id=pid,
            prompt=f"p{i}",
            model_text=f"model {i}",
            human_text=f"human {i}",
        )
        for i, pid in enumerate(pair_ids)
    ]
    verdicts_path = tmp_path / "verdicts.jsonl"
    judge = ScriptedJudge("A")
    run_judging(pairs, verdicts_path, client=judge, seed=3)
    calls_before = judge.calls
    assert calls_before == 5 * len(DIMENSION_ORDER)

    # If the CLI recompute path ever constructed a judge, this stub would count
    # calls; patch it so any accidental construction is observable.
    accidental = ScriptedJudge("A")
    monkeypatch.setattr(
        "dehip.metrics.jmq.OpenAIJudgeClient", lambda *a, **k: accidental
    )

    out = tmp_path / "recomputed.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            "cand-set",
            "--reference",
            "ref-set",
            "--recompute-jmq-from",
            str(verdicts_path),
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    # No judge was constructed or called on the recompute path.
    assert accidental.calls == 0

    written = json.loads(out.read_text())
    # Recomputed aggregation matches a direct aggregation of the same file.
    direct, _ = report_mod.recompute_jmq(str(verdicts_path))
    for dimension in DIMENSION_ORDER:
        assert written["jmq"][dimension] == direct[dimension]


def test_cli_missing_input_exits_validation(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    rc = cli.main(
        [
            "score",
            "--candidate",
            str(tmp_path / "nope.manifest.json"),
            "--reference",
            str(tmp_path / "also-nope.manifest.json"),
            "--metrics",
            "token_l2",
        ]
    )
    assert rc == cli.EXIT_VALIDATION
