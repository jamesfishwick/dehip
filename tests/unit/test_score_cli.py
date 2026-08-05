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
    # IMPORTANT 1: an un-run metric serializes to strict-JSON null, not bare NaN.
    assert written["mmd"] is None
    assert written["jmq"] == {}


def test_cli_written_json_is_strict_and_unrun_is_null(tmp_path, monkeypatch):
    """IMPORTANT 1: on-disk JSON re-parses under strict json.loads; un-run mmd is null.

    A --metrics token_l2 run leaves mmd un-run. The file must be valid ECMA-404
    JSON (no bare NaN that JSON.parse / Go would reject), and the un-run mmd must
    read back as null -- not 0.0 (would be misread as identical sets) and not the
    string 'NaN'. token_l2 (which ran) keeps its real numeric value.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)
    out = tmp_path / "strict.json"
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

    raw = out.read_text()
    assert "NaN" not in raw  # no bare NaN token anywhere on disk

    # Strict parse: reject the non-standard NaN/Infinity constants outright.
    def _reject(_const):
        raise AssertionError("non-strict JSON constant present")

    written = json.loads(raw, parse_constant=_reject)
    assert written["mmd"] is None  # un-run -> null, not 0.0, not 'NaN'
    assert written["mmd"] != 0.0
    assert isinstance(written["token_l2"], float)  # ran -> a real number


# --- IMPORTANT 2: cost estimate threaded into config + --yes gate ------------


def _force_low_threshold(monkeypatch):
    """Make cost_preflight gate at a threshold any real run exceeds.

    Wraps the real cost_preflight with threshold_usd=0.0 so the gate fires unless
    --yes is given, exercising the confirmation path without needing thousands of
    pairs.
    """
    import functools

    from dehip.metrics import jmq as jmq_mod

    real = jmq_mod.cost_preflight
    patched = functools.partial(real, threshold_usd=0.0)
    monkeypatch.setattr("dehip.metrics.jmq.cost_preflight", patched)


def test_cli_cost_gate_blocks_without_yes_no_judge_calls(tmp_path, monkeypatch):
    """Above threshold without --yes -> exit 2 and ZERO judge calls."""
    cand, ref, prompts = _corpus(tmp_path, 4)
    embedder, judge = _patch_seams(monkeypatch)
    _force_low_threshold(monkeypatch)

    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--metrics",
            "jmq",
            "--out",
            str(tmp_path / "r.json"),
            # no --yes
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    assert judge.calls == 0  # the gate blocked before any spend


def test_cli_cost_estimate_recorded_in_config(tmp_path, monkeypatch):
    """With --yes, the cost estimate/threshold lands in report config.thresholds."""
    cand, ref, prompts = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)
    _force_low_threshold(monkeypatch)
    out = tmp_path / "r.json"

    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--metrics",
            "jmq",
            "--out",
            str(out),
            "--yes",
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    thresholds = written["config"]["thresholds"]
    # The estimate dict cost_preflight returned is threaded through, not discarded.
    assert thresholds["calls"] == 4 * len(DIMENSION_ORDER)
    assert "estimated_usd" in thresholds
    assert "threshold_usd" in thresholds


# --- IMPORTANT 3: recompute derives judge_model from the verdict rows ---------


def test_cli_recompute_derives_judge_model_and_caveat(tmp_path):
    """Recompute stamps the verdicts' actual judge_model, not the CLI default.

    Verdicts written with a non-default judge_model must drive
    report.config.judge_model AND fire the non-default-judge caveat, rather than
    the default gpt-5.4-mini the CLI would otherwise pass.
    """
    from dehip.metrics.jmq import JudgePair, run_judging

    pairs = [
        JudgePair(pair_id=f"fineweb-{i}", prompt="p", model_text="m", human_text="h")
        for i in range(3)
    ]
    verdicts_path = tmp_path / "verdicts.jsonl"

    class _NamedJudge:
        def judge(self, rendered_prompt, *, model):
            return "A"

    # run_judging stamps model= into each verdict row.
    run_judging(
        pairs, verdicts_path, client=_NamedJudge(), seed=1, model="claude-experiment"
    )

    out = tmp_path / "recomputed.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            "cand-set",
            "--reference",
            "ref-set",
            "--judge",
            "gpt-5.4-mini",  # the CLI default, which must NOT win
            "--recompute-jmq-from",
            str(verdicts_path),
            "--out",
            str(out),
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    # config.judge_model reflects the verdict rows, not the --judge default.
    assert written["config"]["judge_model"] == "claude-experiment"
    # And the non-default-judge caveat fires off the derived value.
    kinds = {c["kind"] for c in written["caveats"]}
    assert "non_default_judge" in kinds


# --- IMPORTANT 4: atomic + guarded report write ------------------------------


def test_cli_unwritable_report_path_exits_io(tmp_path, monkeypatch):
    """A write to an unwritable path fails loudly with EXIT_IO, no half file.

    The report json target lives under a path whose parent is a regular file, so
    mkdir/write raises OSError. That must map to a clear non-zero exit, not a
    bare exit-1 traceback with a half-written artifact.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)
    # Make a file where a directory is needed, so writing under it raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    out = blocker / "report.json"  # blocker is a file, not a dir

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
    assert rc == cli.EXIT_IO
    # No half-written report exists.
    assert not out.exists()


def test_cli_md_write_failure_leaves_neither_artifact(tmp_path, monkeypatch):
    """NIT 3: when the .md render/write fails, NEITHER final .json nor .md exists.

    The two artifacts are an all-or-nothing pair: both are staged to temp files
    before either is committed, so a failure while producing the .md must leave
    no orphaned .json (and no .md). Also asserts exit EXIT_IO and that no .tmp
    debris survives in the output dir.

    Failure is injected by making render_markdown raise OSError, so the .json has
    already been staged to a temp when the .md stage fails -- the exact "json
    succeeds, md fails" ordering this fix protects.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)

    def _boom(_report):
        raise OSError("simulated md render failure")

    monkeypatch.setattr(report_mod, "render_markdown", _boom)

    out = tmp_path / "report.json"
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
    assert rc == cli.EXIT_IO
    # All-or-nothing: neither final artifact exists.
    assert not out.exists()
    assert not out.with_suffix(".md").exists()
    # No temp debris left behind anywhere in the output dir.
    assert not list(tmp_path.glob("*.tmp"))


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


# --- CRITICAL 1: pairing gate catches a mismatched reference before spend -----


def test_cli_mismatched_reference_exits_validation_no_spend(tmp_path, monkeypatch):
    """A reference manifest whose ids differ from the candidate -> exit 2, no spend.

    The pairing gate must fire at load time (before any embedder or judge is
    constructed or called), listing the asymmetric difference, rather than
    surfacing later as a KeyError. Asserts zero embedder and judge calls.
    """
    pair_ids = [f"fineweb-{i}" for i in range(4)]
    cand_texts = {pid: f"model output {i} words here" for i, pid in enumerate(pair_ids)}
    cand = _write_manifest(
        tmp_path, "cand", "cand-set", "instruct_draft", pair_ids, cand_texts
    )
    # Reference covers every candidate id AND carries an extra (fineweb-99). The
    # old tautology (validate pair_ids against itself, pair_ids taken from the
    # candidate only) would find non-empty reference text for all four candidate
    # ids and silently score a subset. The real pairing gate rejects the
    # asymmetric difference before any spend.
    ref_ids = pair_ids + ["fineweb-99"]
    ref_texts = {pid: f"human ref {i} words here" for i, pid in enumerate(ref_ids)}
    ref = _write_manifest(
        tmp_path, "ref", "ref-set", "human_reference", ref_ids, ref_texts
    )

    embedder, judge = _patch_seams(monkeypatch)
    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--metrics",
            "mmd",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    # Validation fired before any spend: nothing embedded, nothing judged.
    assert embedder.call_count == 0
    assert judge.calls == 0
    # No report was written (validation aborted before emit).
    assert not (tmp_path / "r.json").exists()


def test_cli_texts_superset_of_manifest_not_scored(tmp_path, monkeypatch):
    """A texts file that is a superset of the manifest must not silently score a subset.

    The manifest names 3 ids; the texts JSONL carries 4 (an extra id). Scoring
    the 3-id subset silently would hide a data-prep bug, so the exact-key check
    rejects it with exit 2 and no spend.
    """
    manifest_ids = [f"fineweb-{i}" for i in range(3)]
    texts_ids = manifest_ids + ["fineweb-extra"]
    cand_texts = {pid: f"model {i} words here" for i, pid in enumerate(texts_ids)}
    ref_texts = {pid: f"human {i} words here" for i, pid in enumerate(texts_ids)}

    # Candidate: manifest lists 3 ids but the sibling JSONL holds 4 (superset).
    cand_manifest = TextSet(
        set_id="cand-set",
        role="instruct_draft",
        corpus="fineweb",
        pair_ids=manifest_ids,
        provenance={"texts_path": "cand.jsonl"},
    )
    write_json(cand_manifest, tmp_path / "cand.manifest.json")
    with (tmp_path / "cand.jsonl").open("w", encoding="utf-8") as fh:
        for pid in texts_ids:  # 4 rows, one more than the manifest
            fh.write(json.dumps({"pair_id": pid, "text": cand_texts[pid]}) + "\n")

    ref = _write_manifest(
        tmp_path, "ref", "ref-set", "human_reference", manifest_ids, ref_texts
    )

    embedder, judge = _patch_seams(monkeypatch)
    rc = cli.main(
        [
            "score",
            "--candidate",
            str(tmp_path / "cand.manifest.json"),
            "--reference",
            ref,
            "--metrics",
            "mmd",
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    assert embedder.call_count == 0
    assert judge.calls == 0


def _write_prompts(tmp_path, prompt_ids):
    """Write a {pair_id, prompt} JSONL for the given ids; return its path str."""
    path = tmp_path / "prompts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, pid in enumerate(prompt_ids):
            fh.write(json.dumps({"pair_id": pid, "prompt": f"prompt {i}"}) + "\n")
    return str(path)


def test_cli_prompts_superset_exits_validation_no_spend(tmp_path, monkeypatch):
    """A prompts JSONL that is a SUPERSET of the pair set -> exit 2, no spend.

    The manifests name 4 ids; the prompts file carries a 5th (an extra prompt
    id). Because JMQ's spend depends on prompts, a stray prompt id must be
    rejected at load time with exit 2 (like the texts superset gate), not
    silently ignored while the run proceeds and pays for judging. Asserts zero
    judge calls and no report written.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    pair_ids = [f"fineweb-{i}" for i in range(4)]
    prompts = _write_prompts(tmp_path, pair_ids + ["fineweb-99"])  # superset

    embedder, judge = _patch_seams(monkeypatch)
    out = tmp_path / "r.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--metrics",
            "jmq",
            "--out",
            str(out),
            "--yes",
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    assert judge.calls == 0  # rejected before any judge spend
    assert embedder.call_count == 0
    assert not out.exists()  # no report written


def test_cli_prompts_subset_exits_validation_not_keyerror(tmp_path, monkeypatch):
    """A prompts JSONL that is a SUBSET of the pair set -> exit 2, not a late KeyError.

    The manifests name 4 ids; the prompts file covers only 3 (a missing prompt
    id). This must fail as exit-2 input validation at load time, not surface late
    as a KeyError inside _judge_pairs. Asserts zero judge calls and no report.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    pair_ids = [f"fineweb-{i}" for i in range(4)]
    prompts = _write_prompts(tmp_path, pair_ids[:3])  # subset (missing fineweb-3)

    embedder, judge = _patch_seams(monkeypatch)
    out = tmp_path / "r.json"
    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--metrics",
            "jmq",
            "--out",
            str(out),
            "--yes",
        ]
    )
    assert rc == cli.EXIT_VALIDATION
    assert judge.calls == 0
    assert not out.exists()


def test_cli_token_l2_no_prompts_still_succeeds(tmp_path, monkeypatch):
    """False-positive guard: a --metrics token_l2 run with NO --prompts succeeds.

    The prompts exact-set gate must fire ONLY when a prompts file is passed. A
    metrics subset that omits JMQ and passes no prompts must NOT start failing
    validation just because the prompts check exists.
    """
    cand, ref, _ = _corpus(tmp_path, 4)
    _patch_seams(monkeypatch)
    out = tmp_path / "ok.json"
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
            # no --prompts
        ]
    )
    assert rc == cli.EXIT_SUCCESS
    written = json.loads(out.read_text())
    assert not np.isnan(written["token_l2"])


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


# --- CRITICAL 2: exit-code integrity for external-dep and data failures --------


def test_cli_missing_openai_key_jmq_exits_external_dep(tmp_path, monkeypatch):
    """(a) A jmq run with no OPENAI_API_KEY -> exit 3 (external dependency).

    The judge client is NOT stubbed here: the real OpenAIJudgeClient() is
    constructed, which raises openai.OpenAIError when the key is absent. That
    must map to EXIT_EXTERNAL_DEP with a message, not a bare exit-1 traceback.
    Inputs are well-formed so validation passes and the code actually reaches
    judge construction.
    """
    cand, ref, prompts = _corpus(tmp_path, 4)
    # Stub only the embedder (not needed for a jmq-only run, but harmless); leave
    # the judge real so its keyless construction is exercised.
    monkeypatch.setattr(
        "dehip.metrics.embeddings.TransformersEmbedder", lambda *a, **k: StubEmbedder()
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            prompts,
            "--metrics",
            "jmq",
            "--out",
            str(tmp_path / "r.json"),
            "--yes",
        ]
    )
    assert rc == cli.EXIT_EXTERNAL_DEP


def test_cli_bad_input_beats_missing_key(tmp_path, monkeypatch):
    """Bad input on a keyless machine reports exit 2 (bad input), not exit 3.

    Because the judge is constructed lazily AFTER validation, a mismatched
    reference is caught first: the missing key never gets a chance to raise.
    """
    pair_ids = [f"fineweb-{i}" for i in range(4)]
    cand_texts = {pid: f"model {i} words here" for i, pid in enumerate(pair_ids)}
    cand = _write_manifest(
        tmp_path, "cand", "cand-set", "instruct_draft", pair_ids, cand_texts
    )
    ref_ids = pair_ids[:3] + ["fineweb-99"]
    ref_texts = {pid: f"human {i} words here" for i, pid in enumerate(ref_ids)}
    ref = _write_manifest(
        tmp_path, "ref", "ref-set", "human_reference", ref_ids, ref_texts
    )
    prompts_path = tmp_path / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as fh:
        for pid in pair_ids:
            fh.write(json.dumps({"pair_id": pid, "prompt": "p"}) + "\n")

    monkeypatch.setattr(
        "dehip.metrics.embeddings.TransformersEmbedder", lambda *a, **k: StubEmbedder()
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rc = cli.main(
        [
            "score",
            "--candidate",
            cand,
            "--reference",
            ref,
            "--prompts",
            str(prompts_path),
            "--metrics",
            "jmq",
            "--yes",
        ]
    )
    assert rc == cli.EXIT_VALIDATION


def test_cli_corrupt_verdicts_recompute_exits_validation(tmp_path):
    """(b) A corrupt verdicts.jsonl on --recompute-jmq-from -> exit 2 (not 1).

    A truncated / non-JSON line is an input/data failure and must map to
    EXIT_VALIDATION, never escape as a bare exit-1 traceback.
    """
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text('{"pair_id": "p0", "dimension": "overall",\n', encoding="utf-8")

    rc = cli.main(
        [
            "score",
            "--candidate",
            "cand-set",
            "--reference",
            "ref-set",
            "--recompute-jmq-from",
            str(verdicts),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == cli.EXIT_VALIDATION
