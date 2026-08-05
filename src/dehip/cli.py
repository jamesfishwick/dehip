"""Command-line entry point for the dehip harness.

Subcommand dispatch (build-corpus, generate, rewrite, score, self-check,
detect, report) per specs/001-hip-cascade-harness/contracts/cli.md.

Every command is currently a stub: it parses its documented flags, echoes a
JSON result summary to stdout, writes human-readable progress to stderr, and
exits cleanly. Real behavior lands in later tickets (#5, #10, #12, #13, #14,
#15). Exit-code contract (all commands):

    0  success
    2  validation failure (bad or unknown args)
    3  external dependency failure (HIP checkout, API auth)
    4  self-check out of bounds

argparse already exits 2 on unknown or malformed flags, matching the contract.
"""

import argparse
import json
import sys

# Exit codes shared across commands (see module docstring / cli.md).
EXIT_SUCCESS = 0
EXIT_VALIDATION = 2
EXIT_EXTERNAL_DEP = 3
EXIT_SELF_CHECK = 4
# Report-write I/O failure. Distinct from EXIT_EXTERNAL_DEP so a failure to
# persist the report (unwritable path, full disk) is not confused with a judge
# or embedder dependency failure; a dedicated code makes the artifact I/O
# failure unambiguous to a caller.
EXIT_IO = 5

# Every subcommand name in the CLI contract, in pipeline order.
COMMANDS = (
    "build-corpus",
    "generate",
    "rewrite",
    "score",
    "self-check",
    "detect",
    "report",
)


def _progress(message: str) -> None:
    """Write a human-readable progress line to stderr."""
    print(message, file=sys.stderr)


def _emit_summary(command: str, args: argparse.Namespace) -> None:
    """Emit the JSON result summary for a stub command to stdout.

    Records the command, the global ``--seed``, a ``stub`` marker so callers can
    tell real output apart from placeholder output, and the parsed arguments.
    Real commands will replace this with their own machine-readable report.
    """
    summary = {
        "command": command,
        "seed": args.seed,
        "status": "stub",
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"seed", "func", "command"}
        },
    }
    json.dump(summary, sys.stdout)
    sys.stdout.write("\n")


def _run_stub(command: str, args: argparse.Namespace) -> int:
    """Shared stub body: progress to stderr, JSON summary to stdout, exit 0."""
    _progress(f"dehip {command}: stub (real behavior lands in a later ticket)")
    _emit_summary(command, args)
    return EXIT_SUCCESS


def _add_build_corpus(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build-corpus", help="Build the primary or personal corpus (FR-010)."
    )
    parser.add_argument("--tier", choices=("smoke", "judged", "full"), default="smoke")
    parser.add_argument("--corpus", choices=("fineweb", "personal"), default="fineweb")
    parser.add_argument("--source", help="Path or URLs; required for personal corpus.")
    parser.add_argument("--out", help="Output path for the corpus JSONL.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm spend above the cost threshold (FR-009).",
    )
    parser.set_defaults(func=lambda a: _run_stub("build-corpus", a))


def _add_generate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate", help="Stage-1 instruct drafts for a corpus (R3)."
    )
    parser.add_argument("--corpus", required=True, help="Corpus JSONL to draft from.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", help="Output run directory.")
    parser.set_defaults(func=lambda a: _run_stub("generate", a))


def _add_rewrite(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "rewrite", help="Cascade stage: k rounds of hip-run (FR-005, FR-006, R2)."
    )
    parser.add_argument("--run", help="Run directory to continue from generate.")
    parser.add_argument(
        "--draft-file", help="Rewrite-only mode: draft file, skips generate."
    )
    parser.add_argument("--adapter", default="YixuanEvenXu/Qwen3-4B-Base-HIP-adapter")
    parser.add_argument("--rounds", type=int, default=2, help="Max 4.")
    parser.add_argument(
        "--hip-repo", default="../humanization-by-iterative-paraphrasing"
    )
    parser.set_defaults(func=lambda a: _run_stub("rewrite", a))


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


# Exceptions that signal an external dependency failed (map to EXIT_EXTERNAL_DEP).
# Kept as a helper so both score paths classify identically. Imports are lazy
# (openai is optional) and any missing module degrades gracefully.
def _external_dep_exc_types() -> tuple[type[BaseException], ...]:
    # OSError covers FileNotFoundError and embedder model-load OSError.
    types: list[type[BaseException]] = [OSError]
    try:
        from dehip.metrics.embeddings import CacheIntegrityError

        types.append(CacheIntegrityError)
    except Exception:  # pragma: no cover - embeddings always importable here
        pass
    try:
        import openai

        types.append(openai.OpenAIError)  # incl. missing OPENAI_API_KEY at construct
    except Exception:  # openai SDK not installed: nothing to add
        pass
    return tuple(types)


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
    )
    return _emit_report(report, args.out, report_mod)


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

    # Verdicts live in the run directory alongside the report, matching
    # data-model's results/runs/{run_id}/verdicts.jsonl and the documented
    # --recompute-jmq-from example (NIT 1): <out-dir>/verdicts.jsonl.
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
        # KeyError, MMD median-heuristic on all-coincident points -> exit 2.
        _progress(f"dehip score: input error: {exc}")
        return EXIT_VALIDATION

    return _emit_report(report, args.out, report_mod)


def _emit_report(report, out_path, report_mod) -> int:
    """Write the report JSON (and a sibling .md) and echo the JSON to stdout.

    The on-disk JSON and the stdout echo both go through
    :func:`~dehip.report.report_to_jsonable`, which converts an un-run metric's
    ``NaN`` to strict-JSON ``null`` (IMPORTANT 1) while leaving a real ``0.0``.
    The file write is atomic (temp file + ``os.replace``, IMPORTANT 4); any I/O
    failure is caught and mapped to EXIT_IO with the path and artifact named,
    rather than leaving a half-written file and a bare exit-1 traceback. Returns
    the process exit code.
    """
    from pathlib import Path

    jsonable = report_mod.report_to_jsonable(report)

    if out_path:
        try:
            _atomic_write_text(
                Path(out_path),
                json.dumps(jsonable, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            _progress(f"dehip score: failed to write report json {out_path}: {exc}")
            return EXIT_IO
        md_path = Path(out_path).with_suffix(".md")
        try:
            _atomic_write_text(md_path, report_mod.render_markdown(report))
        except OSError as exc:
            _progress(f"dehip score: failed to write report md {md_path}: {exc}")
            return EXIT_IO
        _progress(f"dehip score: wrote {out_path} and {md_path}")

    json.dump(jsonable, sys.stdout)
    sys.stdout.write("\n")
    return EXIT_SUCCESS


def _atomic_write_text(path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: temp file in the same dir, then replace.

    Mirrors the embedding cache's approach (temp file + ``os.replace``, which is
    atomic on one filesystem): a reader never sees a partially-written report,
    and a mid-write failure leaves the previous file (if any) intact rather than
    a truncated one. Raises ``OSError`` on any failure (unwritable dir, full
    disk) so the caller can map it to a clear exit code.
    """
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # Clean up the partial temp file; never leave debris on a failed write.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


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


def _add_self_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "self-check", help="FR-003: score a human set against itself."
    )
    parser.add_argument("--reference", required=True, help="Human set manifest.")
    parser.add_argument("--skip-jmq", action="store_true")
    parser.set_defaults(func=lambda a: _run_stub("self-check", a))


def _add_detect(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "detect", help="External AI-text detector scoring (SC-005)."
    )
    parser.add_argument(
        "--sets", nargs="+", required=True, help="One or more set manifests."
    )
    parser.add_argument("--detector", choices=("pangram", "gptzero"), default="pangram")
    parser.add_argument("--out", help="Output report path.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm spend above the cost threshold (FR-009).",
    )
    parser.set_defaults(func=lambda a: _run_stub("detect", a))


def _add_report(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report", help="Comparison assembly (FR-007, Story 3)."
    )
    parser.add_argument("--draft-report", required=True, help="Draft MetricReport.")
    parser.add_argument(
        "--rewrite-report",
        action="append",
        required=True,
        dest="rewrite_reports",
        metavar="REWRITE_REPORT",
        help="Rewrite MetricReport; repeat for a k-trajectory table.",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Attach pinned benchmark rows."
    )
    parser.add_argument("--out", help="Output report path.")
    parser.set_defaults(func=lambda a: _run_stub("report", a))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dehip",
        description="HIP cascade and evaluation harness.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed recorded in every command's output summary.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_build_corpus(subparsers)
    _add_generate(subparsers)
    _add_rewrite(subparsers)
    _add_score(subparsers)
    _add_self_check(subparsers)
    _add_detect(subparsers)
    _add_report(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_VALIDATION
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
