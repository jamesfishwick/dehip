"""`dehip report` handler, comparison emitter, and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_IO,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _commit_staged,
    _discard_staged,
    _progress,
    _stage_text,
)


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
    parser.set_defaults(func=_run_report)


def _run_report(args: argparse.Namespace) -> int:
    """Real `dehip report` handler (issue #15): assemble the comparison (FR-007).

    Localized to this command. Reads the draft and rewrite MetricReports through
    the existing schema reader, assembles per-metric deltas (+ a k-trajectory when
    several rewrite reports are given), and -- only when ``--benchmark`` is set and
    no input scored the personal corpus -- attaches the pinned benchmark rows with
    their external-protocol caveat.

    Exit-code contract (cli.md): a bad/missing/corrupt report file or a
    schema-version mismatch is an input/data failure -> exit 2; the FR-010
    personal-corpus benchmark refusal is ALSO exit 2 (a loud validation failure,
    never a silent drop); a report-write I/O failure -> exit 5. Nothing here
    reaches a model or the network.
    """
    from pathlib import Path

    from dehip import report as report_mod
    from dehip.schemas import (
        MetricReport,
        SchemaValidationError,
        SchemaVersionError,
        read_json,
    )

    # Load every input report before any assembly, so a bad path/shape fails at
    # exit 2 with the offending file named, never a later AttributeError.
    try:
        draft = read_json(args.draft_report, MetricReport)
        rewrites = [
            read_json(path, MetricReport) for path in args.rewrite_reports
        ]
    except FileNotFoundError as exc:
        _progress(f"dehip report: report file not found: {exc}")
        return EXIT_VALIDATION
    except (
        json.JSONDecodeError,
        SchemaVersionError,
        SchemaValidationError,
        OSError,
        KeyError,
    ) as exc:
        _progress(f"dehip report: bad report file: {exc}")
        return EXIT_VALIDATION

    comparison_id = (
        Path(args.out).stem if args.out else f"{draft.report_id}-comparison"
    )
    try:
        comparison = report_mod.assemble_comparison(
            draft=draft,
            rewrites=rewrites,
            comparison_id=comparison_id,
            attach_benchmark=args.benchmark,
        )
    except report_mod.PersonalCorpusBenchmarkError as exc:
        # FR-010: a HARD, LOUD refusal at exit 2. Never a silent benchmark drop.
        _progress(f"dehip report: {exc}")
        return EXIT_VALIDATION
    except ValueError as exc:
        _progress(f"dehip report: {exc}")
        return EXIT_VALIDATION

    return _emit_comparison(comparison, args.out, report_mod)


def _emit_comparison(comparison, out_path, report_mod) -> int:
    """Write the comparison JSON and a sibling .md atomically, then echo the JSON.

    Mirrors :func:`_emit_report`'s all-or-nothing pair discipline: both artifacts
    are staged to temp files first and only committed once BOTH temp writes
    succeed, so a failure while rendering the .md never leaves a lone .json. Any
    I/O failure maps to EXIT_IO with the path named.
    """
    from pathlib import Path

    if out_path:
        json_path = Path(out_path)
        md_path = json_path.with_suffix(".md")
        staged: list[tuple[Path, Path]] = []
        try:
            staged.append(
                (
                    _stage_text(
                        json_path,
                        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                    ),
                    json_path,
                )
            )
            staged.append(
                (
                    _stage_text(md_path, report_mod.render_comparison(comparison)),
                    md_path,
                )
            )
        except OSError as exc:
            _discard_staged(staged)
            failed = md_path if staged else json_path
            _progress(f"dehip report: failed to write comparison {failed}: {exc}")
            return EXIT_IO
        try:
            _commit_staged(staged)
        except OSError as exc:
            _discard_staged(staged)
            _progress(f"dehip report: failed to finalize {out_path}: {exc}")
            return EXIT_IO
        _progress(f"dehip report: wrote {json_path} and {md_path}")

    json.dump(comparison, sys.stdout)
    sys.stdout.write("\n")
    return EXIT_SUCCESS
