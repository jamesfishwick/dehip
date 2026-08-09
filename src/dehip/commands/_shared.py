"""Shared low-level helpers and constants for dehip command handlers.

This is the DAG root: it imports nothing from cli.py or any command module.
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
# detect: the run requested an SC-005 delta (two sets given) but it could not be
# computed (not exactly one instruct_draft + one rewrite). Distinct from success
# so a caller can tell a computed delta from a paid run that produced no decision;
# distinct from EXIT_VALIDATION because the inputs were structurally accepted and
# scored (the per-set summaries are still written) -- only the comparison is
# missing. The artifacts are kept; the JSON status is non-ok and this code is
# returned so the null sc005 is loud, not silent.
EXIT_SC005_NOT_COMPUTED = 6

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


# Exceptions that signal an external dependency failed (map to EXIT_EXTERNAL_DEP).
# Kept as a helper so both score paths classify identically. Imports are lazy
# (openai is optional) and any missing module degrades gracefully.
def _external_dep_exc_types() -> tuple[type[BaseException], ...]:
    # OSError covers FileNotFoundError and embedder model-load OSError.
    types: list[type[BaseException]] = [OSError]
    try:
        from dehip.metrics.embeddings import CacheIntegrityError

        types.append(CacheIntegrityError)
    except Exception:  # noqa: S110  # pragma: no cover - embeddings always importable
        pass  # optional import probe: absence just means no extra error type to add
    try:
        import openai

        types.append(openai.OpenAIError)  # incl. missing OPENAI_API_KEY at construct
    except Exception:  # noqa: S110  # openai SDK not installed: nothing to add
        pass  # optional import probe: absence just means no extra error type to add
    return tuple(types)


def _stage_text(path, text: str):
    """Write ``text`` to a temp file beside ``path``; return the temp Path.

    Ensures the target directory exists and writes the content to a sibling temp
    file, but does NOT replace ``path`` -- that is the commit step. Raises
    ``OSError`` on any failure (unwritable dir, full disk) with the temp file
    already removed, so a caller can stage several artifacts and only commit once
    all of them are on disk.
    """
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return tmp


def _commit_staged(staged) -> None:
    """``os.replace`` each staged ``(tmp, final)`` pair into place.

    ``os.replace`` is atomic on one filesystem, so a reader never sees a
    partially-written artifact. Called only after every temp is confirmed on
    disk, so the common failure (render/write of the second artifact) has already
    been ruled out before any final file is touched.
    """
    import os

    for tmp, final in staged:
        os.replace(tmp, final)


def _discard_staged(staged) -> None:
    """Remove any staged temp files; never leave debris on a failed write."""
    for tmp, _final in staged:
        try:
            tmp.unlink()
        except OSError:
            pass
