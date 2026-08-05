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


def _run_score(args: argparse.Namespace) -> int:
    """Real `dehip score` handler (issue #10): compose the metrics into a report.

    Localized to this command. Imports of dehip.report and the metric seams are
    done here (not at module top) so the other stub subcommands stay import-cheap
    and the change stays contained to the score path.
    """
    from pathlib import Path

    from dehip import report as report_mod
    from dehip.metrics.embeddings import EmbeddingCache, TransformersEmbedder
    from dehip.metrics.jmq import OpenAIJudgeClient, cost_preflight
    from dehip.validate import InputSetValidationError

    # Recompute path (FR-008): re-aggregate JMQ from persisted verdicts with no
    # judge/API calls, then re-emit a report. No validation spend, no embedder.
    if args.recompute_jmq_from:
        _progress(f"dehip score: recomputing JMQ from {args.recompute_jmq_from}")
        jmq_scores, verdicts = report_mod.recompute_jmq(args.recompute_jmq_from)
        report = report_mod.assemble_report(
            report_id=Path(args.out).stem if args.out else "recompute",
            candidate_set=args.candidate,
            reference_set=args.reference,
            n=len({v.pair_id for v in verdicts}),
            seed=args.seed,
            judge_model=args.judge,
            embedder_id=args.embedder,
            tokenizer_id=None,
            mmd_result=None,
            token_l2_result=None,
            jmq_scores=jmq_scores,
            verdicts=verdicts,
        )
        _emit_report(report, args.out, report_mod)
        return EXIT_SUCCESS

    try:
        inputs, cand_set_id, ref_set_id = report_mod.load_scoring_inputs(
            args.candidate, args.reference, prompts_path=args.prompts
        )
    except (OSError, ValueError, KeyError) as exc:
        _progress(f"dehip score: could not load input sets: {exc}")
        return EXIT_VALIDATION

    selected = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # JMQ cost preflight before any spend (FR-009). Gated on --yes.
    if "jmq" in selected:
        try:
            cost_preflight(
                len(inputs.pair_ids), confirm=args.yes, printer=_progress
            )
        except Exception as exc:  # CostThresholdError and friends -> validation exit
            _progress(f"dehip score: {exc}")
            return EXIT_VALIDATION

    # Build the seams. The embedder and judge are lazy, so constructing them here
    # costs nothing until a metric actually calls them.
    embed_cache = None
    if "mmd" in selected:
        embedder = TransformersEmbedder(args.embedder)
        embed_cache = EmbeddingCache(embedder)

    judge_client = OpenAIJudgeClient() if "jmq" in selected else None
    verdicts_path = None
    if "jmq" in selected:
        verdicts_path = (
            str(Path(args.out).with_suffix(".verdicts.jsonl"))
            if args.out
            else "results/verdicts.jsonl"
        )

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
        )
    except InputSetValidationError as exc:
        _progress(f"dehip score: input validation failed: {exc}")
        return EXIT_VALIDATION

    _emit_report(report, args.out, report_mod)
    return EXIT_SUCCESS


def _emit_report(report, out_path, report_mod) -> None:
    """Write the report JSON (and a sibling .md) and echo the JSON to stdout."""
    from dataclasses import asdict
    from pathlib import Path

    from dehip.schemas import write_json

    if out_path:
        write_json(report, out_path)
        md_path = str(Path(out_path).with_suffix(".md"))
        Path(md_path).write_text(
            report_mod.render_markdown(report), encoding="utf-8"
        )
        _progress(f"dehip score: wrote {out_path} and {md_path}")
    json.dump(asdict(report), sys.stdout)
    sys.stdout.write("\n")


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
