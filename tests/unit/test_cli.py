"""Unit tests for the dehip CLI dispatch (issue #4).

These lock the CLI contract that later tickets build on: the seven
subcommands exist, each rejects unknown flags with exit 2, and a stub run
writes a JSON summary (carrying the seed) to stdout with progress on stderr.
"""

import json
import subprocess
import sys

import pytest

from dehip import cli

# One minimal invocation per subcommand that satisfies its required flags.
# argparse never touches the filesystem for these stubs, so placeholder paths
# are fine.
VALID_INVOCATIONS = {
    "build-corpus": ["build-corpus"],
    "generate": ["generate", "--corpus", "corpus.jsonl"],
    "rewrite": ["rewrite"],
    "score": ["score", "--candidate", "cand.json", "--reference", "ref.json"],
    "self-check": ["self-check", "--reference", "ref.json"],
    "detect": ["detect", "--sets", "a.json", "b.json"],
    "report": [
        "report",
        "--draft-report",
        "draft.json",
        "--rewrite-report",
        "rw.json",
    ],
}

ALL_COMMANDS = sorted(cli.COMMANDS)

# Every subcommand is now a real command (build-corpus #5, score #10,
# generate #12, self-check #11, rewrite #13, detect #14, report #15): none emits
# the {"status": "stub"} summary any more, so their behavior is covered by their
# own test files (test_corpus.py, test_score_cli.py, test_generate.py,
# test_self_check.py, test_cascade.py, test_detector.py, test_report_comparison.py).
# The stub-shape assertions that used to parametrize over STUB_COMMANDS are
# retired; the flag-rejection test below still covers every command via
# ALL_COMMANDS. No stub commands remain.
STUB_COMMANDS: list[str] = sorted(
    set(cli.COMMANDS)
    - {
        "build-corpus",
        "score",
        "generate",
        "self-check",
        "rewrite",
        "detect",
        "report",
    }
)  # == [] ; kept as an explicit assertion that nothing is a stub.

def _run_cli(argv):
    """Run the CLI as a subprocess so stdout/stderr are cleanly separated."""
    return subprocess.run(
        [sys.executable, "-m", "dehip.cli", *argv],
        capture_output=True,
        text=True,
    )


def test_command_list_matches_contract():
    assert ALL_COMMANDS == sorted(
        [
            "build-corpus",
            "generate",
            "rewrite",
            "score",
            "self-check",
            "detect",
            "report",
        ]
    )
    # Every listed command has a valid invocation exercised below.
    assert set(VALID_INVOCATIONS) == set(cli.COMMANDS)


def test_help_lists_every_subcommand():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    for command in cli.COMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_unknown_flag_rejected_with_exit_2(command):
    argv = [*VALID_INVOCATIONS[command], "--definitely-not-a-flag"]
    result = _run_cli(argv)
    assert result.returncode == cli.EXIT_VALIDATION == 2


def test_no_stub_commands_remain():
    # Every subcommand is real now; none emits the {"status": "stub"} summary.
    # Per-command behavior (including seed recording) is covered by each
    # command's own test file. This replaces the old stub-shape assertions.
    assert STUB_COMMANDS == []


def test_no_subcommand_exits_2():
    result = _run_cli([])
    assert result.returncode == cli.EXIT_VALIDATION


def _run_json(argv):
    result = _run_cli(argv)
    assert result.returncode == 0
    return json.loads(result.stdout)
