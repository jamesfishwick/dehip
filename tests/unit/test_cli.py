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

# Commands still served by the shared stub. ``score`` became a real command in
# issue #10, so its handler no longer emits the {"status": "stub"} summary and
# is excluded from the stub-shape assertions below (its behavior is covered by
# test_score_cli.py). Every other command remains a stub until its own ticket.
STUB_COMMANDS = sorted(set(cli.COMMANDS) - {"score"})


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


@pytest.mark.parametrize("command", STUB_COMMANDS)
def test_stdout_is_json_on_its_own(command):
    result = _run_cli(VALID_INVOCATIONS[command])
    assert result.returncode == cli.EXIT_SUCCESS == 0
    # stdout must parse as JSON with nothing else mixed in (progress on stderr).
    summary = json.loads(result.stdout)
    assert summary["command"] == command
    assert summary["status"] == "stub"
    # Progress output goes to stderr, not stdout.
    assert result.stderr.strip() != ""


@pytest.mark.parametrize("command", STUB_COMMANDS)
def test_seed_appears_in_summary(command):
    argv = ["--seed", "4242", *VALID_INVOCATIONS[command]]
    result = _run_cli(argv)
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["seed"] == 4242


def test_default_seed_recorded():
    summary = _run_json(["build-corpus"])
    assert summary["seed"] == 0


def test_no_subcommand_exits_2():
    result = _run_cli([])
    assert result.returncode == cli.EXIT_VALIDATION


def _run_json(argv):
    result = _run_cli(argv)
    assert result.returncode == 0
    return json.loads(result.stdout)
