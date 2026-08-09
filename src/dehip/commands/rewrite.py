"""`dehip rewrite` handler and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _progress,
)


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
    # Sampling knobs hip-run actually applies (inference.py:118-120). Defaults
    # match the HIP inference protocol (temperature 1.0, top_p 0.95).
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument(
        "--hip-repo", default="../humanization-by-iterative-paraphrasing"
    )
    parser.add_argument(
        "--out",
        help="Output run directory for draft-file mode (default results/runs/).",
    )
    parser.set_defaults(func=_run_rewrite)


def _run_rewrite(args: argparse.Namespace) -> int:
    """Real `dehip rewrite` handler (issue #13): the k-round HIP cascade.

    Two input modes producing the identical bundle structure: run-continuation
    (``--run results/runs/{run_id}/`` continues from generate's nascent bundles)
    and draft-file (``--draft-file PATH`` skips generation). Localized to this
    command; imports of dehip.cascade live here (not at module top) so the other
    stub subcommands stay import-cheap and the change stays contained.

    Exit-code contract (cli.md rewrite section): the HIP precondition (checkout
    resolves + `uv run hip-run` responds) is checked BEFORE any inference and a
    failure maps to EXIT_EXTERNAL_DEP (3); a mid-run hip-run failure (non-zero
    exit, malformed/empty output) is also EXIT_EXTERNAL_DEP; bad inputs
    (rounds out of range, missing/unreadable draft, malformed manifest, missing
    generate run) map to EXIT_VALIDATION (2). A subprocess RuntimeError from the
    seam is normalized to HipRunError so it never escapes as a bare exit-1
    traceback.
    """
    import time
    from pathlib import Path

    from dehip import cascade as cascade_mod

    if bool(args.run) == bool(args.draft_file):
        _progress(
            "dehip rewrite: exactly one of --run or --draft-file is required"
        )
        return EXIT_VALIDATION

    # Validate rounds first (exit 2) so a bad --rounds fails before the
    # precondition subprocess is even attempted.
    if args.rounds < 1 or args.rounds > cascade_mod.MAX_ROUNDS:
        _progress(
            f"dehip rewrite: --rounds {args.rounds} out of range "
            f"(1..{cascade_mod.MAX_ROUNDS})"
        )
        return EXIT_VALIDATION

    # Precondition BEFORE any inference/subprocess-inference work (cli.md): the
    # HIP sibling checkout must resolve and `uv run hip-run` must respond, else
    # exit 3 -- checked even on a machine that could never run the paraphraser.
    try:
        cascade_mod.check_hip_precondition(args.hip_repo)
    except cascade_mod.HipPreconditionError as exc:
        _progress(f"dehip rewrite: HIP precondition failed: {exc}")
        return EXIT_EXTERNAL_DEP

    if args.run:
        run_dir = args.run
        run_id = Path(run_dir).name or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        try:
            nascent = cascade_mod.load_nascent_bundles_from_run(run_dir)
        except ValueError as exc:
            _progress(f"dehip rewrite: {exc}")
            return EXIT_VALIDATION
    else:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = args.out or f"results/runs/{run_id}/"
        try:
            nascent = cascade_mod.bundles_from_draft_file(
                args.draft_file, run_id=run_id
            )
        except ValueError as exc:
            _progress(f"dehip rewrite: {exc}")
            return EXIT_VALIDATION

    # The real subprocess adapter is the injectable seam: it is constructed only
    # after the precondition passed, and its per-round subprocess is the only
    # place that shells out.
    runner = cascade_mod.SubprocessHipRunner(
        args.hip_repo,
        work_dir=Path(run_dir) / "hip-work",
        temperature=args.temperature,
        top_p=args.top_p,
    )
    try:
        summary = cascade_mod.run_cascade(
            nascent,
            runner=runner,
            run_dir=run_dir,
            run_id=run_id,
            requested_k=args.rounds,
            adapter_id=args.adapter,
            seed=args.seed,
            printer=_progress,
        )
    except cascade_mod.HipRunError as exc:
        # A mid-run hip-run failure (non-zero exit, malformed/empty output) is an
        # external-dependency failure (exit 3), not a bare exit-1 traceback.
        _progress(f"dehip rewrite: hip-run failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except cascade_mod.RoundsValidationError as exc:
        _progress(f"dehip rewrite: input error: {exc}")
        return EXIT_VALIDATION
    except ValueError as exc:
        # Zero-bundle / heterogeneous-corpus input errors -> exit 2.
        _progress(f"dehip rewrite: input error: {exc}")
        return EXIT_VALIDATION

    json.dump(
        {
            "command": "rewrite",
            "seed": args.seed,
            "status": "ok",
            "adapter": args.adapter,
            **{
                k: summary[k]
                for k in (
                    "run_id",
                    "pairs",
                    "rewritten",
                    "requested_k",
                    "flagged_degenerate",
                    "round_manifests",
                )
            },
            "out": str(Path(run_dir)),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return EXIT_SUCCESS
