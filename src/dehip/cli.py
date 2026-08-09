"""Command-line entry point for the dehip harness.

Subcommand dispatch (build-corpus, generate, rewrite, score, self-check,
detect, report) per specs/001-hip-cascade-harness/contracts/cli.md.

Every subcommand is real; each reads and writes the file formats in
data-model.md, gates spend before any judge/detector call, prints a JSON result
summary to stdout with human-readable progress on stderr, and returns the exit
code below. Real-instrument validation on live models, judges, and detectors is
tracked separately (issue #16) and is not what these handlers assert. Exit-code
contract (all commands):

    0  success
    2  validation failure (bad or unknown args)
    3  external dependency failure (HIP checkout, API auth)
    4  self-check out of bounds
    5  report-write I/O failure
    6  detect: SC-005 delta requested (two sets) but not computable

argparse already exits 2 on unknown or malformed flags, matching the contract.
"""

import argparse
import sys

from dehip.commands._shared import (
    COMMANDS,
    EXIT_EXTERNAL_DEP,
    EXIT_IO,
    EXIT_SC005_NOT_COMPUTED,
    EXIT_SELF_CHECK,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
)
from dehip.commands.build_corpus import _add_build_corpus, _run_build_corpus
from dehip.commands.detect import _add_detect
from dehip.commands.generate import _add_generate
from dehip.commands.report import _add_report
from dehip.commands.rewrite import _add_rewrite
from dehip.commands.score import _add_score
from dehip.commands.self_check import _add_self_check

# Re-exported for callers/tests that reference them as cli.<name> (the exit
# codes, the COMMANDS tuple, and the _run_build_corpus handler).
__all__ = [
    "COMMANDS",
    "EXIT_EXTERNAL_DEP",
    "EXIT_IO",
    "EXIT_SC005_NOT_COMPUTED",
    "EXIT_SELF_CHECK",
    "EXIT_SUCCESS",
    "EXIT_VALIDATION",
    "_add_build_corpus",
    "_add_detect",
    "_add_generate",
    "_add_report",
    "_add_rewrite",
    "_add_score",
    "_add_self_check",
    "_build_parser",
    "_run_build_corpus",
    "main",
]


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
