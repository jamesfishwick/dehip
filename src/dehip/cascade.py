"""Cascade rewrite stage: wrap the HIP repo's ``hip-run`` CLI (issue #13, R2).

The second cascade stage. For every :class:`~dehip.schemas.RewriteBundle` a
prior ``dehip generate`` wrote (a *nascent* bundle carrying a ``draft`` and an
empty ``rounds`` list) -- or, in draft-file mode, a bundle synthesized directly
from a draft text -- apply the HIP paraphraser for ``requested_k`` rounds and
fill in the bundle's ``rounds``, ``final_round``, ``degeneration``,
``adapter_id``, and ``hip_config``. Every intermediate round is kept.

Four design commitments drive the module shape, all aimed at an adversarial
review:

- **The ``hip-run`` invocation is an injectable seam.** :class:`HipRunner` is a
  ``Protocol`` with a single :meth:`run_round` method, mirroring
  ``generate.py``'s :class:`~dehip.generate.DraftModel`. Tests inject a stub, so
  the round-loop state machine, degeneration gating, resumability, and manifest
  logic are exercised *without* shelling out, downloading models, or touching
  the HIP checkout. The real subprocess adapter (:class:`SubprocessHipRunner`)
  is thin glue behind the seam: emit the real ``hip-run`` YAML config, run one
  ``uv run hip-run --config`` round (input JSONL in, output parquet out), and
  map the parquet's ``source_row_index`` back to each pair_id.

- **The round loop stops at the LAST GOOD round on a hard degeneration trip.**
  Round 0 is the draft. Each round ``k`` (1-indexed) runs one ``hip-run``
  invocation and then :func:`~dehip.validate.detect_degeneration` compares the
  result against the *prior good round's* length. A HARD trip (empty, length
  ratio outside [0.5, 2.0], non-ASCII burst) records the degenerate round
  (still kept, but marked ``hard_tripped``) and stops that pair: ``final_round``
  becomes ``k - 1`` (the last good round; ``0`` means the draft was the last
  good output -- a hard trip on round 1). A repetition FLAG is recorded but
  never stops iteration. One degenerate pair never aborts the run; the loop
  moves on to the next pair.

- **Resumability reuses ``generate.py``'s persistence discipline verbatim.**
  Completed bundles are appended one JSON object per line to a *rewrite* bundles
  file, each fsync'd, with a torn-tail repair before the first resume append --
  exactly the append+fsync+torn-tail+resume machinery
  :mod:`dehip.generate` hardened under adversarial review. A partially-completed
  multi-round run resumes without re-running any pair whose completed bundle is
  already durable, and without losing captured rounds.

- **A malformed / empty ``hip-run`` result fails loudly.** A ``hip-run`` round
  that returns no text for a pair (or empty/whitespace text) must never be
  persisted as a real rewrite that scores downstream -- that is exactly the
  silent-empty-draft corruption ``generate.py`` guards against. The seam
  normalizes such a result to :class:`HipRunError` (-> CLI exit 3) rather than
  emitting a blank round.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dehip.generate import (
    BUNDLES_FILENAME,
    CorpusDriftError,
    _append_bundle,
    _read_all_bundles,
    _repair_torn_tail,
)
from dehip.schemas import RewriteBundle, TextSet, write_json
from dehip.validate import DegenerationReport, detect_degeneration

__all__ = [
    "DEFAULT_ADAPTER",
    "DEFAULT_HIP_REPO",
    "DEFAULT_ROUNDS",
    "MAX_ROUNDS",
    "REWRITE_BUNDLES_FILENAME",
    "HipPreconditionError",
    "HipRunError",
    "RoundsValidationError",
    "CorpusDriftError",
    "HipRunner",
    "SubprocessHipRunner",
    "check_hip_precondition",
    "run_cascade",
]

# Defaults mirror the CLI contract (contracts/cli.md rewrite section).
DEFAULT_ADAPTER = "YixuanEvenXu/Qwen3-4B-Base-HIP-adapter"
DEFAULT_HIP_REPO = "../humanization-by-iterative-paraphrasing"
DEFAULT_ROUNDS = 2
MAX_ROUNDS = 4

# The cascade CONTINUES the nascent bundles generate.py wrote. It reads that
# same bundles.jsonl (BUNDLES_FILENAME) but persists COMPLETED bundles to a
# distinct rewrite file so a resume can tell a not-yet-rewritten nascent bundle
# apart from a finished one without re-running rounds. In draft-file mode there
# is no prior generate file, so the rewrite file is the sole persistence target.
REWRITE_BUNDLES_FILENAME = "rewrite-bundles.jsonl"
REWRITE_TEXTS_TEMPLATE = "rewrite-k{k}-texts.jsonl"
REWRITE_MANIFEST_TEMPLATE = "rewrite-k{k}.manifest.json"


class HipPreconditionError(RuntimeError):
    """Raised when the HIP sibling checkout is absent or ``hip-run`` won't run.

    The CLI maps this to exit 3 (external-dependency failure), matching the
    exit-code discipline the ``generate``/``score`` commands use, so a missing
    sibling checkout reports a clean exit-3 diagnostic rather than a bare
    subprocess traceback -- and it is checked BEFORE any inference/subprocess-
    inference work so a weightless machine fails fast.
    """


class HipRunError(RuntimeError):
    """Raised when a ``hip-run`` round fails or returns malformed/empty output.

    Covers a non-zero subprocess exit, an unreadable/missing output parquet, a
    result missing a pair the round was asked to rewrite, and an
    empty/whitespace-only rewrite for any pair. Every one of these must fail
    loudly: a blank or missing
    rewrite silently persisted as a finished round would score downstream as a
    real (blank) rewrite -- the exact silent corruption ``generate.py``'s
    empty-draft guard exists to prevent. The CLI maps this to exit 3 (external
    dependency), not a bare exit-1 traceback.
    """


class RoundsValidationError(ValueError):
    """Raised when ``requested_k`` is outside the allowed range (1..MAX_ROUNDS).

    Subclasses ``ValueError`` so the CLI's existing input-error handler maps it
    to exit 2 (bad input), matching how ``generate`` classifies input errors.
    """


# --- The hip-run seam --------------------------------------------------------


@runtime_checkable
class HipRunner(Protocol):
    """The rewrite seam: run ONE ``hip-run`` round over a batch of texts.

    One round maps ``{pair_id: input_text}`` to ``{pair_id: rewrite_text}`` for
    exactly the same pair_ids. Running rounds explicitly (k single-round
    invocations) rather than one k-round call is what lets the degeneration
    checks run between rounds and the bundle record every intermediate (R2).

    The real implementation is :class:`SubprocessHipRunner`; tests inject a stub
    so no subprocess, model download, or HIP checkout is touched.
    """

    def run_round(
        self,
        inputs: dict[str, str],
        *,
        round_k: int,
        adapter_id: str,
        seed: int = 0,
    ) -> dict[str, str]:
        """Return the rewrite of every input text for round ``round_k``.

        The returned mapping must cover exactly the input pair_ids. ``seed`` is the
        harness-controlled seed passed through to ``hip-run`` for reproducibility.
        Implementations raise :class:`HipRunError` on any failure so the caller
        need not know how the round was produced.
        """
        ...

    def config_for(
        self, *, round_k: int, adapter_id: str, seed: int
    ) -> dict[str, Any]:
        """Return the REQUESTED config for ``round_k``, inlined into the bundle audit.

        Recorded verbatim in each bundle's ``hip_config`` (data-model.md), so a
        review can see exactly what config was HANDED TO ``hip-run`` for every
        round -- including the ``seed`` the harness controls. This is the
        requested config, advisory for any field ``hip-run`` may override at run
        time; it is not a readback of what ``hip-run`` actually applied.
        """
        ...


class SubprocessHipRunner:
    """Thin ``uv run hip-run`` glue behind the :class:`HipRunner` seam.

    Everything subprocess- and YAML-specific lives here so the round loop (and
    every test) never shells out. For each round it writes the REAL ``hip-run``
    YAML config plus a JSONL input file into a work dir, invokes ``uv run hip-run
    --config <cfg>`` (and NOTHING else -- ``hip-run``'s argparse defines only
    ``--config``) in the HIP checkout, then reads the output PARQUET ``hip-run``
    writes back into ``{pair_id: text}``. Any failure -- non-zero exit,
    unreadable parquet, a missing/blank rewrite -- is normalized to
    :class:`HipRunError` (-> CLI exit 3).

    The config schema matches ``hip/inference.py`` verbatim: ``adapter_path``
    (the HF adapter id, NOT ``adapter_id``), ``input_jsonl``, ``text_field``,
    ``output_parquet``, ``metadata_json``, ``num_rounds`` (1 per invocation so
    the cascade's inter-round degeneration checks run one round at a time), plus
    sampling knobs. ``hip-run`` resolves relative config paths against the HIP
    repo root, so ``input_jsonl``/``output_parquet``/``metadata_json`` are
    written as ABSOLUTE paths (``dehip``'s work dir is in the dehip repo, not the
    HIP repo). ``base_model`` is OMITTED by default so any adapter self-resolves
    to its correct base via its ``PeftConfig``; it is only emitted when the
    caller passes a ``base_model`` override.

    The precondition (:func:`check_hip_precondition`) must have passed before a
    real round runs; this class does not re-check it per round.
    """

    def __init__(
        self,
        hip_repo: str | Path,
        *,
        work_dir: str | Path,
        base_model: str | None = None,
        timeout_s: float = 7200.0,
    ) -> None:
        self.hip_repo = Path(hip_repo)
        self.work_dir = Path(work_dir)
        # base_model None => let hip-run infer the base from the adapter's
        # PeftConfig (base_model_name_or_path). Only an explicit override is
        # emitted into the config so an adapter never resolves to a wrong base.
        self.base_model = base_model
        # CPU inference in float32 (hip-run's choose_dtype has no CUDA/MPS path
        # on this machine) is slow, so the per-round timeout is generous.
        self.timeout_s = timeout_s

    def config_for(
        self, *, round_k: int, adapter_id: str, seed: int
    ) -> dict[str, Any]:
        """Build the per-round hip-run AUDIT config (one round, one adapter, one seed).

        Recorded verbatim in each bundle's ``hip_config`` (data-model.md). It is
        the path-independent audit view of what ``hip-run`` was handed for
        ``round_k``: ``adapter_path`` (the real key ``hip-run`` reads, mapped from
        the harness ``adapter_id``), ``num_rounds`` 1, ``text_field`` ``"text"``,
        and the harness-controlled ``seed``. ``base_model`` appears ONLY when an
        override was passed to the constructor -- omitting it lets the adapter
        self-resolve its base. The ``seed`` is recorded for the reproducibility
        audit trail even though ``hip-run`` ignores it (its argparse has no seed
        flag; it is advisory, like every field ``hip-run`` may override).

        The full run-time config emitted to disk in :meth:`run_round` extends
        this with the ABSOLUTE ``input_jsonl``/``output_parquet``/``metadata_json``
        paths for the round; those are per-round file locations, not part of the
        cross-round audit shape, so they live only in the on-disk config.
        """
        config: dict[str, Any] = {
            "adapter_path": adapter_id,
            "num_rounds": 1,
            "round": round_k,
            "text_field": "text",
            "seed": seed,
        }
        if self.base_model is not None:
            config["base_model"] = self.base_model
        return config

    def run_round(
        self,
        inputs: dict[str, str],
        *,
        round_k: int,
        adapter_id: str,
        seed: int = 0,
    ) -> dict[str, str]:
        """Emit config + input JSONL, run one ``hip-run`` round, parse the parquet.

        Writes ``inputs`` as ``{"text": <draft>, "pair_id": <pid>}`` one JSON
        object per line in a STABLE (pair_id) order -- ``hip-run`` reads the rows
        in order and keys each output row back by ``source_row_index`` (its 0-based
        input position), so the write order IS the mapping. Emits the real
        ``hip-run`` YAML config with ABSOLUTE ``input_jsonl``/``output_parquet``/
        ``metadata_json`` paths (``hip-run`` resolves relative config paths against
        the HIP repo root, not dehip's work dir), invokes ``uv run hip-run
        --config <cfg>`` (``--config`` ONLY -- passing ``--input``/``--output``
        would make ``hip-run``'s argparse fail with "unrecognized arguments") in
        the HIP checkout, then reads the output PARQUET into ``{pair_id: text}``.

        Every failure mode -- process error, non-zero exit, unreadable/missing
        parquet, a pair the round did not rewrite, or a blank rewrite -- raises
        :class:`HipRunError` so nothing blank is ever returned as real text.

        The per-round work dir is keyed on both the pair_id(s) and the round so
        each pair's subprocess input/config/output persist for audit rather than
        being overwritten by the next pair at the same round (NIT 1). The cascade
        runs one pair per round, so the single pair_id namespaces the dir.
        """
        import yaml  # local import: only the real subprocess path needs pyyaml

        pair_tag = "-".join(_slug(pid) for pid in inputs) or "batch"
        round_dir = self.work_dir / f"pair-{pair_tag}" / f"round-{round_k}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # hip-run resolves relative config paths against its own repo root, so
        # every path handed to it must be absolute (dehip's work dir is not under
        # the HIP repo). resolve() makes them absolute + canonical.
        config_path = round_dir / "config.yaml"
        input_path = (round_dir / "input.jsonl").resolve()
        output_path = (round_dir / "output.parquet").resolve()
        metadata_path = (round_dir / "output.metadata.json").resolve()

        # A STABLE pair_id order fixes source_row_index -> pair_id: row i of the
        # input JSONL is the pair at ordered_ids[i], and hip-run stamps each
        # output row with source_row_index == its input position.
        ordered_ids = sorted(inputs)
        with input_path.open("w", encoding="utf-8") as fh:
            for pair_id in ordered_ids:
                fh.write(json.dumps({"text": inputs[pair_id], "pair_id": pair_id}))
                fh.write("\n")

        config = self.config_for(
            round_k=round_k, adapter_id=adapter_id, seed=seed
        )
        config["input_jsonl"] = str(input_path)
        config["output_parquet"] = str(output_path)
        config["metadata_json"] = str(metadata_path)
        config["trust_remote_code"] = True
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        try:
            proc = subprocess.run(
                ["uv", "run", "hip-run", "--config", str(config_path.resolve())],
                cwd=self.hip_repo,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HipRunError(
                f"hip-run round {round_k} failed to execute in {self.hip_repo}: {exc}"
            ) from exc
        if proc.returncode != 0:
            raise HipRunError(
                f"hip-run round {round_k} exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        return _parse_round_output(
            output_path, ordered_ids=ordered_ids, round_k=round_k
        )


def _parse_round_output(
    output_path: Path, *, ordered_ids: list[str], round_k: int
) -> dict[str, str]:
    """Parse a hip-run output PARQUET into ``{pair_id: text}``, failing loudly.

    ``hip-run`` writes a parquet (NOT a JSONL) with one row per input example per
    round. The columns this reads: ``source_row_index`` (the 0-based position of
    the row in the input JSONL -- the key back to the pair_id, since the caller
    wrote the input in ``ordered_ids`` order), ``round`` (1-indexed; the cascade
    runs ``num_rounds`` 1 so it reads the ``round == round_k`` rows), and
    ``output_text`` (the rewrite). ``hip-run`` does NOT carry ``pair_id``, so the
    mapping is purely positional: ``ordered_ids[source_row_index]``.

    Raises :class:`HipRunError` if the parquet is missing/unreadable, lacks the
    expected columns, a ``source_row_index`` is out of range, a pair the round
    was asked to rewrite is absent from this round's rows, or any rewrite is
    empty/whitespace-only. A blank or absent rewrite is treated exactly like
    ``generate.py``'s empty-draft guard: loud failure, never a silently-persisted
    blank.
    """
    import pyarrow.parquet as pq  # local import: only the real path reads parquet

    if not output_path.exists():
        raise HipRunError(
            f"hip-run round {round_k} produced no output parquet at {output_path}"
        )
    try:
        table = pq.read_table(output_path)
    except Exception as exc:  # noqa: BLE001 -- any parquet read failure is loud
        raise HipRunError(
            f"hip-run round {round_k} produced an unreadable output parquet "
            f"at {output_path}: {exc}"
        ) from exc

    columns = set(table.column_names)
    required = {"source_row_index", "round", "output_text"}
    missing_cols = required - columns
    if missing_cols:
        raise HipRunError(
            f"hip-run round {round_k} output parquet lacks column(s) "
            f"{sorted(missing_cols)}; got {sorted(columns)}"
        )

    result: dict[str, str] = {}
    for record in table.to_pylist():
        if record["round"] != round_k:
            continue  # hip-run emits one row per source row per round; take ours
        source_index = record["source_row_index"]
        if not isinstance(source_index, int) or not (
            0 <= source_index < len(ordered_ids)
        ):
            raise HipRunError(
                f"hip-run round {round_k} output parquet has an out-of-range "
                f"source_row_index {source_index!r} (input had {len(ordered_ids)} rows)"
            )
        pair_id = ordered_ids[source_index]
        result[pair_id] = record["output_text"]

    return _validate_round_result(
        result, expected=set(ordered_ids), round_k=round_k
    )


def _validate_round_result(
    result: dict[str, str], *, expected: set[str], round_k: int
) -> dict[str, str]:
    """Assert a round result covers every expected pair with non-blank text.

    Shared by the subprocess adapter and any seam whose result must be checked
    before it enters a bundle. A missing pair or a blank rewrite is a loud
    :class:`HipRunError`, never a silently-dropped or blank round.
    """
    missing = expected - set(result)
    if missing:
        raise HipRunError(
            f"hip-run round {round_k} did not rewrite pair(s) {sorted(missing)}; "
            "refusing to persist an incomplete round"
        )
    for pair_id, text in result.items():
        if not isinstance(text, str) or not text.strip():
            raise HipRunError(
                f"hip-run round {round_k} returned an empty/whitespace rewrite for "
                f"pair {pair_id!r}; refusing to persist it as a real rewrite"
            )
    return result


# --- Precondition ------------------------------------------------------------


def check_hip_precondition(hip_repo: str | Path) -> None:
    """Verify the HIP sibling checkout resolves and ``uv run hip-run`` responds.

    Two checks, cheapest first, both mapping to :class:`HipPreconditionError`
    (-> CLI exit 3): the checkout directory must exist, and ``uv run hip-run
    --help`` must exit cleanly there. Called BEFORE any inference or per-round
    subprocess work so a missing checkout fails fast on a machine that could
    never run the paraphraser (contracts/cli.md rewrite precondition).
    """
    repo = Path(hip_repo)
    if not repo.is_dir():
        raise HipPreconditionError(
            f"HIP sibling checkout not found at {repo} (pass --hip-repo to override); "
            "clone humanization-by-iterative-paraphrasing beside this repo"
        )
    try:
        proc = subprocess.run(
            ["uv", "run", "hip-run", "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HipPreconditionError(
            f"could not run `uv run hip-run --help` in {repo}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise HipPreconditionError(
            f"`uv run hip-run --help` failed in {repo} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


# --- Round-loop state machine ------------------------------------------------


def _degeneration_to_dict(report: DegenerationReport) -> dict[str, Any]:
    """Serialize a :class:`DegenerationReport` to the bundle degeneration object.

    Matches data-model.md's degeneration object (per-check results +
    ``hard_tripped``). The per-round ``flags`` list is what gets stamped onto the
    round record; this whole-report view is the bundle-level summary (the last
    round's report, so ``hard_tripped`` reflects whether the pair tripped).
    """
    return {
        "empty": report.empty,
        "length_ratio": report.length_ratio,
        "length_ratio_tripped": report.length_ratio_tripped,
        "repetition_flagged": report.repetition_flagged,
        "non_ascii_fraction": report.non_ascii_fraction,
        "non_ascii_burst_tripped": report.non_ascii_burst_tripped,
        "hard_tripped": report.hard_tripped,
        "flags": list(report.flags),
    }


def _run_rounds_for_pair(
    bundle: RewriteBundle,
    *,
    runner: HipRunner,
    requested_k: int,
    adapter_id: str,
    seed: int,
) -> RewriteBundle:
    """Run the k-round state machine for ONE pair; return the completed bundle.

    The heart of the ticket. Round 0 is the draft. For each round ``k`` in
    ``1..requested_k``:

    1. Run one ``hip-run`` round (via the seam) to rewrite the *prior good
       round's* text.
    2. :func:`detect_degeneration` compares the result against the prior good
       round's length.
    3. On a HARD trip, RECORD the degenerate round (kept, flagged) and STOP:
       ``final_round`` is ``k - 1`` (the last good round; ``0`` = the draft was
       the last good output, i.e. a hard trip on round 1). No further rounds run.
    4. On no hard trip (including a repetition-only flag), record the round as a
       new good round, advance ``final_round`` to ``k``, and continue.

    Every intermediate round -- good or degenerate -- is kept in
    ``bundle.rounds``. The bundle's ``hip_config`` inlines the REQUESTED config
    handed to ``hip-run`` for each round (including the harness-controlled
    ``seed``), advisory for any field ``hip-run`` may override -- not a readback
    of what it applied. ``degeneration`` records the last round's report so
    ``hard_tripped`` flags the bundle.
    """
    assert bundle.draft is not None, "cascade requires a draft to rewrite from"
    draft_text = bundle.draft["text"]

    rounds: list[dict[str, Any]] = []
    hip_configs: list[dict[str, Any]] = []
    # Round 0 is the draft. prior_good_text/length track the last GOOD output,
    # which is what the next round rewrites and what the length ratio compares
    # against -- never a degenerate round.
    prior_good_text = draft_text
    prior_good_length = len(draft_text)
    final_round = 0
    last_report = DegenerationReport()  # a clean (no-op) report if k == 0

    for k in range(1, requested_k + 1):
        config = runner.config_for(round_k=k, adapter_id=adapter_id, seed=seed)
        hip_configs.append(config)
        result = runner.run_round(
            {bundle.pair_id: prior_good_text},
            round_k=k,
            adapter_id=adapter_id,
            seed=seed,
        )
        # The seam guarantees a non-blank text for the pair, but re-validate the
        # single-pair result so a stub seam is held to the same contract as the
        # subprocess adapter (no silent blank round). A blank hip-run result is a
        # LOUD failure here (HipRunError -> exit 3), matching generate.py's
        # empty-draft discipline -- distinct from detect_degeneration's R10 empty
        # hard-check, which the cascade never reaches because a genuinely blank
        # round is treated as a subprocess malfunction, not a stop-and-flag.
        result = _validate_round_result(
            result, expected={bundle.pair_id}, round_k=k
        )
        rewrite_text = result[bundle.pair_id]

        report = detect_degeneration(rewrite_text, prior_length=prior_good_length)
        last_report = report
        rounds.append(
            {
                "k": k,
                "text": rewrite_text,
                "flags": list(report.flags),
                "hard_tripped": report.hard_tripped,
            }
        )
        if report.hard_tripped:
            # Stop at the LAST GOOD round: final_round stays k-1. The degenerate
            # round is kept above (marked) but never becomes the new prior-good
            # text, so a later round would never rewrite a degenerate output.
            final_round = k - 1
            break
        # A good round (repetition-only flags do not stop iteration): advance.
        final_round = k
        prior_good_text = rewrite_text
        prior_good_length = len(rewrite_text)

    return RewriteBundle(
        run_id=bundle.run_id,
        pair_id=bundle.pair_id,
        prompt=bundle.prompt,
        rounds=rounds,
        final_round=final_round,
        degeneration=_degeneration_to_dict(last_report),
        adapter_id=adapter_id,
        # hip_config is the REQUESTED config passed to hip-run (advisory for
        # fields hip-run may override), not a readback of what it applied. The
        # harness-controlled seed is recorded at the top level so the audit trail
        # is non-empty on the reproducibility field that matters even for a bundle
        # with zero rounds, and mirrored per-round in ``rounds`` (IMPORTANT 2).
        hip_config={"seed": seed, "rounds": hip_configs},
        requested_k=requested_k,
        draft=bundle.draft,
    )


# --- Round TextSet manifests -------------------------------------------------


def _good_text_at_round(bundle: RewriteBundle, k: int) -> str | None:
    """Return the pair's text for round ``k`` iff round ``k`` is a *good* round.

    A round appears in a per-round TextSet only if the pair actually produced a
    good (non-hard-tripped) output at that round: ``k <= final_round``. A pair
    whose cascade stopped at final_round 1 contributes to the k1 set but not the
    k2 set. The draft (round 0) is not a rewrite round, so it never appears here.
    """
    if k < 1 or k > bundle.final_round:
        return None
    for record in bundle.rounds:
        if record["k"] == k:
            return record["text"]
    return None


def _prune_stale_round_artifacts(run_dir: Path, *, requested_k: int) -> None:
    """Remove ``rewrite-k*`` manifest + texts artifacts for ``k > requested_k``.

    A prior run at a larger ``requested_k`` leaves per-round artifacts this
    shorter run never rewrites; without pruning they linger and a downstream
    consumer would read stale higher-k rewrites as part of this run (IMPORTANT 4).
    Only artifacts strictly above the current ``requested_k`` are removed, so the
    rounds this run does emit are never touched. Non-numeric or malformed
    ``rewrite-k*`` names are left alone (not ours to interpret).
    """
    for manifest_path in run_dir.glob("rewrite-k*.manifest.json"):
        k = _round_of_artifact(manifest_path.name, suffix=".manifest.json")
        if k is not None and k > requested_k:
            manifest_path.unlink()
    for texts_path in run_dir.glob("rewrite-k*-texts.jsonl"):
        k = _round_of_artifact(texts_path.name, suffix="-texts.jsonl")
        if k is not None and k > requested_k:
            texts_path.unlink()


def _round_of_artifact(name: str, *, suffix: str) -> int | None:
    """Parse the round ``k`` out of a ``rewrite-k{k}{suffix}`` artifact name.

    Returns ``None`` for any name that does not match the exact
    ``rewrite-k<int>{suffix}`` shape, so an unrelated or malformed file is left
    untouched by the pruner rather than mis-parsed.
    """
    prefix = "rewrite-k"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : len(name) - len(suffix)]
    if not middle.isdigit():
        return None
    return int(middle)


def _write_round_manifests(
    bundles: list[RewriteBundle],
    *,
    run_dir: Path,
    corpus: str,
    run_id: str,
    requested_k: int,
    adapter_id: str,
) -> list[TextSet]:
    """Emit one role=rewrite TextSet manifest per round (round=k), + a texts JSONL.

    For each round ``k`` in ``1..requested_k``, gather every pair whose cascade
    produced a good output at round ``k`` (``k <= final_round``), write a
    ``{pair_id, text}`` JSONL, and a role=rewrite TextSet manifest tagged with
    ``round=k`` pointing at it via ``provenance.texts_path`` (matching the
    draft-manifest convention ``report._texts_path_for`` resolves). A round with
    no surviving pairs (every pair hard-tripped before reaching it) is skipped,
    so no empty manifest is written.

    Before writing, any stale higher-k artifacts from a prior run at a LARGER
    ``requested_k`` are removed (IMPORTANT 4): completing at ``--rounds 3`` then
    resuming at ``--rounds 2`` must not leave a ``rewrite-k3.manifest.json`` +
    ``rewrite-k3-texts.jsonl`` that a downstream report/score could consume as if
    round 3 were part of this run. Only ``k > requested_k`` artifacts are pruned;
    the artifacts this run (re)writes are untouched.
    """
    _prune_stale_round_artifacts(run_dir, requested_k=requested_k)
    manifests: list[TextSet] = []
    for k in range(1, requested_k + 1):
        members = [
            (b.pair_id, _good_text_at_round(b, k))
            for b in bundles
            if _good_text_at_round(b, k) is not None
        ]
        if not members:
            continue
        texts_name = REWRITE_TEXTS_TEMPLATE.format(k=k)
        texts_path = run_dir / texts_name
        texts_path.parent.mkdir(parents=True, exist_ok=True)
        with texts_path.open("w", encoding="utf-8") as fh:
            for pair_id, text in members:
                fh.write(json.dumps({"pair_id": pair_id, "text": text}))
                fh.write("\n")
        manifest = TextSet(
            set_id=f"{corpus}-{len(members)}-rewrite-k{k}",
            role="rewrite",
            corpus=corpus,
            round=k,
            pair_ids=[pair_id for pair_id, _ in members],
            provenance={
                "builder": "dehip rewrite",
                "run_id": run_id,
                "adapter_id": adapter_id,
                "round": k,
                "count": len(members),
                "texts_path": texts_name,
            },
        )
        write_json(manifest, run_dir / REWRITE_MANIFEST_TEMPLATE.format(k=k))
        manifests.append(manifest)
    return manifests


# --- Cascade driver ----------------------------------------------------------


def run_cascade(
    nascent_bundles: list[RewriteBundle],
    *,
    runner: HipRunner,
    run_dir: str | Path,
    run_id: str,
    requested_k: int = DEFAULT_ROUNDS,
    adapter_id: str = DEFAULT_ADAPTER,
    seed: int = 0,
    printer=print,
) -> dict[str, Any]:
    """Run the k-round cascade over every nascent bundle, resumable per pair.

    For each nascent :class:`~dehip.schemas.RewriteBundle` (a bundle with a
    ``draft`` and empty ``rounds``, as ``generate`` wrote, or as draft-file mode
    synthesizes) whose pair does not already have a *completed* rewrite bundle
    on disk, runs :func:`_run_rounds_for_pair` and appends the completed bundle
    to ``rewrite-bundles.jsonl`` (fsync'd per record). A HARD degeneration trip
    stops only that pair at its last good round and flags its bundle; the run
    continues to the remaining pairs. After the cascade, one role=rewrite
    TextSet manifest per round is (re)written from the full completed-bundle list.

    Resumability (the testable core): re-invoking with the same ``run_dir``
    continues an interrupted cascade. Pairs whose completed bundle is already
    durable are detected and NOT re-run -- so a first pass that finishes ``m`` of
    ``M`` pairs before a kill produces exactly ``M - m`` further per-pair cascades
    on re-run, with no duplicate bundles and no lost captured rounds. This reuses
    ``generate.py``'s append+fsync + torn-tail-repair discipline verbatim.

    Args:
        nascent_bundles: Bundles to rewrite (each carries a ``draft``).
        runner: The injected :class:`HipRunner` seam (real or stub).
        run_dir: Output run directory (``results/runs/{run_id}/``).
        run_id: Run identifier recorded in every completed bundle + manifest.
        requested_k: Configured round count (validated 1..MAX_ROUNDS).
        adapter_id: HIP adapter id recorded in every bundle.
        seed: Harness-controlled seed passed to ``hip-run`` and recorded in each
            bundle's ``hip_config`` for the reproducibility audit trail.
        printer: Progress sink (stderr in the CLI).

    Returns:
        A summary dict: pair counts (total / already-done / rewritten), the
        skipped-truncated-line count, the number of pairs flagged for
        degeneration, and the written artifact paths.
    """
    if requested_k < 1 or requested_k > MAX_ROUNDS:
        raise RoundsValidationError(
            f"requested rounds {requested_k} out of range; must be 1..{MAX_ROUNDS}"
        )
    if not nascent_bundles:
        raise ValueError("cannot run the cascade over zero bundles")

    corpora = {_corpus_of(b) for b in nascent_bundles}
    if len(corpora) != 1:
        raise ValueError(
            f"cascade expects a homogeneous corpus, got {sorted(corpora)}"
        )
    corpus = next(iter(corpora))

    run_dir = Path(run_dir)
    rewrite_path = run_dir / REWRITE_BUNDLES_FILENAME

    done_ids, skipped = _load_done_rewrite_ids(rewrite_path)
    if skipped:
        printer(
            f"dehip rewrite: skipped {skipped} truncated/corrupt rewrite-bundle "
            "line(s) from a prior interrupted run; those pairs will be re-run"
        )
    # Repair a torn tail BEFORE the first resume append (CRITICAL 1, verbatim
    # from generate.py) so a regenerated record never concatenates onto a
    # fragment into one silently-dropped line.
    if _repair_torn_tail(rewrite_path):
        printer(
            "dehip rewrite: repaired a torn final rewrite-bundle line (no trailing "
            "newline) from a prior interrupted run before resuming"
        )

    remaining = [b for b in nascent_bundles if b.pair_id not in done_ids]
    printer(
        f"dehip rewrite: {len(nascent_bundles)} pairs, {len(done_ids)} already "
        f"rewritten, {len(remaining)} to run ({requested_k} rounds)"
    )

    rewritten = 0
    for bundle in remaining:
        completed = _run_rounds_for_pair(
            bundle,
            runner=runner,
            requested_k=requested_k,
            adapter_id=adapter_id,
            seed=seed,
        )
        _append_bundle(completed, rewrite_path)
        rewritten += 1

    # Re-read the authoritative append-only file (tolerating a prior-run torn
    # tail) to assemble manifests deterministically and count flagged pairs.
    completed_bundles = _read_all_bundles(rewrite_path)

    # Guard against corpus drift, mirroring generate.py exactly (IMPORTANT 1): a
    # run dir holding rewrite bundles from a DIFFERENT/larger corpus carries
    # pair_ids the current (nascent) input does not, which would otherwise be
    # silently merged into the emitted manifests + texts, mislabeled and counted,
    # exiting 0. The persisted rewrite-bundle pair_ids MUST be a subset of the
    # nascent input pair_ids; a stray id fails loudly (CorpusDriftError -> exit 2),
    # naming the strays, before any manifest is written.
    input_ids = {b.pair_id for b in nascent_bundles}
    stray = sorted({b.pair_id for b in completed_bundles} - input_ids)
    if stray:
        raise CorpusDriftError(
            "rewrite bundles carry pair_ids absent from the input corpus "
            f"(stale run dir?): {stray}. Refusing to merge them into the "
            "manifest."
        )

    # Order manifests by the input bundle order for determinism.
    order = {b.pair_id: i for i, b in enumerate(nascent_bundles)}
    completed_bundles.sort(key=lambda b: order.get(b.pair_id, len(order)))

    flagged = sum(
        1 for b in completed_bundles if b.degeneration.get("hard_tripped")
    )
    manifests = _write_round_manifests(
        completed_bundles,
        run_dir=run_dir,
        corpus=corpus,
        run_id=run_id,
        requested_k=requested_k,
        adapter_id=adapter_id,
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pairs": len(nascent_bundles),
        "already_done": len(done_ids),
        "rewritten": rewritten,
        "skipped_truncated": skipped,
        "requested_k": requested_k,
        "adapter_id": adapter_id,
        "flagged_degenerate": flagged,
        "rewrite_bundles": str(rewrite_path),
        "round_manifests": [m.set_id for m in manifests],
    }


def _slug(text: str) -> str:
    """Filesystem-safe slug of a pair_id for a per-pair work dir (NIT 1).

    Keeps alphanumerics, dash, and underscore; replaces every other character
    with ``_`` so an arbitrary pair_id becomes a safe path segment without
    colliding two distinct ids onto one dir under normal ``{corpus}-{seq}`` ids.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def _corpus_of(bundle: RewriteBundle) -> str:
    """Derive a bundle's corpus tag from its pair_id (``{corpus}-{seq}``).

    Pair ids are ``{corpus}-{seq}`` (data-model.md). The corpus is everything
    before the final ``-``; a pair_id without a ``-`` falls back to the whole id
    so a hand-built bundle still groups deterministically.
    """
    pid = bundle.pair_id
    return pid.rsplit("-", 1)[0] if "-" in pid else pid


def _load_done_rewrite_ids(rewrite_path: Path) -> tuple[set[str], int]:
    """Return pair_ids with a COMPLETED rewrite bundle, plus skipped bad lines.

    Reuses ``generate.py``'s tolerant :func:`_read_all_bundles` reader (skips a
    truncated final line from a crash mid-flush) so a completed bundle is
    "done": its pair is not re-run on resume. A pair whose bundle was torn is
    absent here and gets re-run, overwriting nothing (append-only).

    ``skipped`` is the count of lines that FAIL to parse in a tolerant pass, not
    ``raw_nonempty - len(bundles)`` (IMPORTANT 3): ``_read_all_bundles`` de-dupes
    a duplicate PARSEABLE pair_id (a regenerated pair whose prior line also
    parsed), so the subtraction would wrongly inflate ``skipped`` and the CLI
    message would claim a de-duped-but-durable pair "will be re-run" when it will
    not. Counting real parse failures keeps ``skipped`` at exactly the
    truncated/corrupt lines that force a re-run.
    """
    if not rewrite_path.exists():
        return set(), 0
    bundles = _read_all_bundles(rewrite_path)
    done = {b.pair_id for b in bundles}
    skipped = _count_unparseable_lines(rewrite_path)
    return done, skipped


def _count_unparseable_lines(rewrite_path: Path) -> int:
    """Count non-empty rewrite-bundle lines that fail to parse (IMPORTANT 3).

    A tolerant pass mirroring :func:`_read_all_bundles`'s per-line parse (same
    JSON + schema-version + shape gate), counting exactly the lines that DO NOT
    yield a valid bundle -- a truncated tail, a corrupt record, a wrong-version
    line. A parseable duplicate pair_id is NOT counted (it parses fine; it is
    merely de-duped downstream), so ``skipped`` never over-reports.
    """
    from dehip.generate import _bundle_from_raw
    from dehip.schemas import SchemaValidationError, SchemaVersionError

    unparseable = 0
    with rewrite_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                _bundle_from_raw(json.loads(line))
            except (
                json.JSONDecodeError,
                SchemaVersionError,
                SchemaValidationError,
                TypeError,
                KeyError,
            ):
                unparseable += 1
    return unparseable


# --- Draft-file mode ---------------------------------------------------------


def bundles_from_draft_file(
    draft_path: str | Path,
    *,
    run_id: str,
) -> list[RewriteBundle]:
    """Synthesize nascent bundles from a draft JSONL, for rewrite-only mode.

    Draft-file mode skips ``generate`` but must produce the same round and
    degeneration shape as run-continuation mode: each ``{pair_id, text[, prompt]}``
    record becomes a nascent :class:`~dehip.schemas.RewriteBundle` carrying that
    text as its ``draft`` with an empty ``rounds`` list, so :func:`run_cascade`
    treats both modes the same. The two modes are NOT byte-identical by design:
    the draft's ``model_id``/``sampling`` provenance is recorded as ``draft-file``
    (no model produced these), so draft provenance differs while the rewrite
    trajectory (rounds, ``final_round``, ``degeneration``) matches.

    Raises ``ValueError`` (-> CLI exit 2) on a missing/unreadable file, an
    unparseable line, or a record lacking ``pair_id``/``text``.
    """
    path = Path(draft_path)
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"draft file not readable: {exc}") from exc

    bundles: list[RewriteBundle] = []
    seen: set[str] = set()
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"draft file has an unparseable line: {exc}") from exc
        pair_id = record.get("pair_id")
        text = record.get("text")
        if not pair_id or text is None:
            raise ValueError(
                f"draft file record lacks pair_id/text: {record!r}"
            )
        if not str(text).strip():
            raise ValueError(
                f"draft file text for pair {pair_id!r} is empty; nothing to rewrite"
            )
        if pair_id in seen:
            raise ValueError(f"draft file has a duplicate pair_id {pair_id!r}")
        seen.add(pair_id)
        bundles.append(
            RewriteBundle(
                run_id=run_id,
                pair_id=pair_id,
                prompt=record.get("prompt", ""),
                rounds=[],
                final_round=0,
                degeneration={"hard_tripped": False},
                adapter_id="",
                hip_config={},
                requested_k=0,
                draft={
                    "text": text,
                    "model_id": "draft-file",
                    "sampling": {"source": "draft-file"},
                },
            )
        )
    if not bundles:
        raise ValueError(f"draft file {path} contained no records")
    return bundles


def load_nascent_bundles_from_run(run_dir: str | Path) -> list[RewriteBundle]:
    """Load the nascent bundles ``generate`` wrote into ``run_dir``.

    Run-continuation mode reads ``generate``'s ``bundles.jsonl`` (the nascent
    bundles carrying a ``draft`` and empty ``rounds``) via the tolerant reader,
    so a truncated final line from an interrupted ``generate`` does not abort the
    cascade. A bundle missing its ``draft`` is a corrupt/unfinished generate
    record; it is a ``ValueError`` (-> exit 2) so the cascade never rewrites a
    draftless bundle.
    """
    run_dir = Path(run_dir)
    bundles_path = run_dir / BUNDLES_FILENAME
    if not bundles_path.exists():
        raise ValueError(
            f"no generate bundles at {bundles_path}; run `dehip generate` first "
            "or use --draft-file for rewrite-only mode"
        )
    bundles = _read_all_bundles(bundles_path)
    if not bundles:
        raise ValueError(f"generate bundles at {bundles_path} are empty")
    draftless = [b.pair_id for b in bundles if b.draft is None]
    if draftless:
        raise ValueError(
            f"generate bundles carry draftless records: {sorted(draftless)}"
        )
    return bundles
