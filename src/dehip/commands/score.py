"""`dehip score` handler, its recompute/full paths, and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_IO,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _commit_staged,
    _discard_staged,
    _external_dep_exc_types,
    _progress,
    _stage_text,
)


def _verdicts_derived_judge_model(verdicts, default: str) -> str:
    """Derive the judge_model to record from persisted verdicts (IMPORTANT 3).

    Each :class:`~dehip.schemas.JudgeVerdict` row carries the ``judge_model`` that
    actually produced it, so a recompute must reflect that value rather than
    stamp the CLI's ``--judge`` default (``gpt-5.4-mini``). When every row shares
    one model, that shared value is returned. When the rows are mixed (or empty),
    the false default is still avoided: a ``mixed:{a,b,...}`` marker records the
    fact so config.judge_model and the non-default-judge caveat both reflect
    reality instead of a made-up single judge.
    """
    models = {v.judge_model for v in verdicts}
    if not models:
        return default
    if len(models) == 1:
        return next(iter(models))
    return "mixed:" + ",".join(sorted(models))


def _run_score(args: argparse.Namespace) -> int:
    """Real `dehip score` handler (issue #10): compose the metrics into a report.

    Localized to this command. Imports of dehip.report and the metric seams are
    done here (not at module top) so the other stub subcommands stay import-cheap
    and the change stays contained to the score path.

    Exit-code contract (cli.md): external-dependency failures (missing/invalid
    OPENAI_API_KEY, missing judge-prompts/, embedder model-load, torch OOM,
    embedding-cache corruption) map to EXIT_EXTERNAL_DEP; input/data-class
    failures (id mismatch, metric-selection or MMD-degeneracy ValueError, a
    corrupt/truncated/wrong-version verdicts file on the recompute path) map to
    EXIT_VALIDATION; a report-write I/O failure maps to EXIT_IO. The judge client
    is constructed lazily *after* validation so bad input on a keyless machine
    still reports exit 2 (bad input), not exit 3.
    """
    from pathlib import Path

    from dehip import report as report_mod

    if args.recompute_jmq_from:
        return _run_recompute(args, report_mod, Path)
    return _run_full_score(args, report_mod, Path)


def _run_recompute(args: argparse.Namespace, report_mod, Path) -> int:
    """--recompute-jmq-from branch: re-aggregate JMQ from persisted verdicts.

    Reads only the file (FR-008): no judge/API calls, no embedder. A corrupt,
    truncated, or wrong-version verdicts file (json.JSONDecodeError,
    SchemaVersionError, SchemaValidationError) is an input/data failure -> exit
    2, never a bare exit-1 traceback. The recorded judge_model is derived from
    the verdict rows, not the CLI default (IMPORTANT 3).
    """
    from dehip.schemas import SchemaValidationError, SchemaVersionError

    _progress(f"dehip score: recomputing JMQ from {args.recompute_jmq_from}")
    try:
        jmq_scores, verdicts = report_mod.recompute_jmq(args.recompute_jmq_from)
    except (json.JSONDecodeError, SchemaVersionError, SchemaValidationError) as exc:
        _progress(f"dehip score: corrupt verdicts file: {exc}")
        return EXIT_VALIDATION
    except FileNotFoundError as exc:
        # A missing verdicts file the docs told the user to pass is an external
        # dependency (the artifact) that is absent -> exit 3.
        _progress(f"dehip score: verdicts file not found: {exc}")
        return EXIT_EXTERNAL_DEP

    judge_model = _verdicts_derived_judge_model(verdicts, args.judge)
    # Best-effort corpus for the FR-010 gate: read the candidate manifest's
    # homogeneous corpus tag when the manifest is a readable TextSet. A recompute
    # run may point --candidate at a set id rather than a manifest path, in which
    # case the tag is simply absent (None) and the set-id naming fallback still
    # protects the gate; a bad manifest is never fatal to a recompute.
    corpus = _best_effort_corpus(args.candidate)
    report = report_mod.assemble_report(
        report_id=Path(args.out).stem if args.out else "recompute",
        candidate_set=args.candidate,
        reference_set=args.reference,
        n=len({v.pair_id for v in verdicts}),
        seed=args.seed,
        judge_model=judge_model,
        embedder_id=args.embedder,
        tokenizer_id=None,
        mmd_result=None,
        token_l2_result=None,
        jmq_scores=jmq_scores,
        verdicts=verdicts,
        corpus=corpus,
    )
    return _emit_report(report, args.out, report_mod)


def _best_effort_corpus(candidate_manifest: str | None) -> str | None:
    """Read the candidate TextSet's corpus tag, or ``None`` if unavailable.

    Used by the recompute path, where no MetricInputs is loaded, to still stamp
    ``compared["corpus"]`` for the FR-010 gate. Any failure (the argument is a set
    id not a path, the file is missing or not a valid TextSet) degrades to ``None``
    -- the set-id naming fallback in the gate still applies -- and never aborts a
    recompute over an otherwise-valid verdicts file.
    """
    if not candidate_manifest:
        return None
    try:
        from pathlib import Path

        from dehip.schemas import TextSet, read_json

        path = Path(candidate_manifest)
        if not path.is_file():
            return None
        text_set = read_json(path, TextSet)
    except Exception:
        return None
    corpus = text_set.corpus
    return corpus if isinstance(corpus, str) and corpus else None


def _run_full_score(args: argparse.Namespace, report_mod, Path) -> int:
    """Main score path: validate, preflight, then compute the requested metrics."""
    from dehip.metrics.embeddings import EmbeddingCache, TransformersEmbedder
    from dehip.metrics.jmq import CostThresholdError, OpenAIJudgeClient, cost_preflight
    from dehip.validate import InputSetValidationError

    # Load + cross-validate the two manifests before any spend (FR-009). A
    # mismatched reference (or a texts file that is a superset of its manifest)
    # raises InputSetValidationError here -> exit 2, with zero embedder/judge
    # calls, never a later KeyError.
    try:
        inputs, cand_set_id, ref_set_id = report_mod.load_scoring_inputs(
            args.candidate, args.reference, prompts_path=args.prompts
        )
    except (InputSetValidationError, OSError, ValueError, KeyError) as exc:
        # A bad input path, a mismatched/corrupt manifest, or a duplicate/missing
        # id are all the user's input being wrong -> exit 2 (matches the contract
        # test for a missing manifest). Distinct from a missing *dependency*
        # artifact (judge-prompts/, verdicts file) handled downstream as exit 3.
        _progress(f"dehip score: input validation failed: {exc}")
        return EXIT_VALIDATION

    selected = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # JMQ cost preflight before any spend (FR-009), gated on --yes. Capture the
    # estimate so authorized spend is recorded in the report config (IMPORTANT 2)
    # rather than discarded.
    thresholds: dict | None = None
    if "jmq" in selected:
        try:
            thresholds = cost_preflight(
                len(inputs.pair_ids), confirm=args.yes, printer=_progress
            )
        except CostThresholdError as exc:
            _progress(f"dehip score: {exc}")
            return EXIT_VALIDATION

    # Build the seams. The embedder and judge are constructed only for the
    # metrics selected, and only AFTER validation + preflight have passed, so a
    # missing OPENAI_API_KEY on otherwise-bad input reports exit 2, not exit 3.
    external_dep = _external_dep_exc_types()
    try:
        embed_cache = None
        if "mmd" in selected:
            embedder = TransformersEmbedder(args.embedder)
            embed_cache = EmbeddingCache(embedder)

        judge_client = OpenAIJudgeClient() if "jmq" in selected else None
    except external_dep as exc:
        _progress(f"dehip score: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP

    # Verdicts live beside the report at <out-dir>/verdicts.jsonl, matching the
    # documented --recompute-jmq-from example. `score` has no run_id concept; the
    # results/runs/{run_id}/ layout in data-model belongs to the rewrite/run
    # pipeline (a later ticket), not this command.
    verdicts_path = None
    if "jmq" in selected:
        if args.out:
            verdicts_path = str(Path(args.out).parent / "verdicts.jsonl")
        else:
            verdicts_path = "results/verdicts.jsonl"

    default_report_id = f"{cand_set_id}-vs-{ref_set_id}"
    try:
        report = report_mod.score(
            inputs,
            report_id=Path(args.out).stem if args.out else default_report_id,
            candidate_set=cand_set_id,
            reference_set=ref_set_id,
            embed_cache=embed_cache,
            judge_client=judge_client,
            verdicts_path=verdicts_path,
            metrics=args.metrics,
            seed=args.seed,
            judge_model=args.judge,
            embedder_id=args.embedder,
            thresholds=thresholds,
        )
    except InputSetValidationError as exc:
        _progress(f"dehip score: input validation failed: {exc}")
        return EXIT_VALIDATION
    except external_dep as exc:
        # Metric internals reaching a real dependency (embedder model-load,
        # torch OOM, missing judge-prompts/, cache corruption) -> exit 3.
        _progress(f"dehip score: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except (ValueError, KeyError) as exc:
        # Input/data-class seams: unknown metric, id mismatch surfacing as a
        # KeyError, MMD median-heuristic on all-coincident points -> exit 2. The
        # classification is by exception TYPE (data-class errors, not external
        # deps), deliberately broad so a real input error reports exit 2 rather
        # than escaping as a bare exit-1 traceback.
        _progress(f"dehip score: input error: {exc}")
        return EXIT_VALIDATION

    return _emit_report(report, args.out, report_mod)


def _emit_report(report, out_path, report_mod) -> int:
    """Write the report JSON and a sibling .md as an all-or-nothing pair, echo JSON.

    The on-disk JSON and the stdout echo both go through
    :func:`~dehip.report.report_to_jsonable`, which converts an un-run metric's
    ``NaN`` to strict-JSON ``null`` (IMPORTANT 1) while leaving a real ``0.0``.

    The two artifacts are written as one atomic pair (NIT 3): both are staged to
    temp files in their target dir first, and only after BOTH temp writes succeed
    are they ``os.replace``d into place. A failure on either write leaves NEITHER
    final artifact (no orphaned ``.json`` with a missing ``.md``) and removes any
    temp debris. Any I/O failure maps to EXIT_IO with the path and artifact named,
    rather than a half-written pair and a bare exit-1 traceback. Returns the
    process exit code.
    """
    from pathlib import Path

    jsonable = report_mod.report_to_jsonable(report)

    if out_path:
        json_path = Path(out_path)
        md_path = json_path.with_suffix(".md")
        staged: list[tuple[Path, Path]] = []  # (tmp, final) pairs to commit
        try:
            # Stage BOTH artifacts to temp files before committing either, so a
            # failure while rendering/writing the .md cannot leave a lone .json.
            staged.append(
                (
                    _stage_text(
                        json_path,
                        json.dumps(jsonable, ensure_ascii=False, indent=2) + "\n",
                    ),
                    json_path,
                )
            )
            staged.append(
                (_stage_text(md_path, report_mod.render_markdown(report)), md_path)
            )
        except OSError as exc:
            # Clean up any temp already staged; commit nothing.
            _discard_staged(staged)
            failed = md_path if staged else json_path
            _progress(f"dehip score: failed to write report {failed}: {exc}")
            return EXIT_IO

        # Both temps written: commit them. os.replace is atomic per file; if the
        # second replace somehow fails, the discard drops the second temp (the
        # first is already committed, matching per-file atomicity).
        try:
            _commit_staged(staged)
        except OSError as exc:
            _discard_staged(staged)
            _progress(f"dehip score: failed to finalize report {out_path}: {exc}")
            return EXIT_IO
        _progress(f"dehip score: wrote {json_path} and {md_path}")

    json.dump(jsonable, sys.stdout)
    sys.stdout.write("\n")
    return EXIT_SUCCESS


def _add_score(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "score", help="The harness (FR-001..004, FR-008, FR-009)."
    )
    parser.add_argument("--candidate", required=True, help="Candidate set manifest.")
    parser.add_argument("--reference", required=True, help="Reference set manifest.")
    parser.add_argument("--metrics", default="mmd,token_l2,jmq")
    parser.add_argument("--judge", default="gpt-5.4-mini")
    parser.add_argument("--embedder", default="nvidia/llama-embed-nemotron-8b")
    parser.add_argument("--out", help="Output report path.")
    parser.add_argument(
        "--recompute-jmq-from",
        help="Re-aggregate from a verdicts.jsonl without API calls (FR-008).",
    )
    parser.add_argument(
        "--prompts",
        help="Prompts JSONL ({pair_id, prompt}) for JMQ; required when jmq runs.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm JMQ spend above the cost threshold (FR-009).",
    )
    parser.set_defaults(func=_run_score)
