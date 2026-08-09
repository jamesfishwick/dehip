"""`dehip self-check` handler and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_SELF_CHECK,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _external_dep_exc_types,
    _progress,
)


def _run_self_check(args: argparse.Namespace) -> int:
    """Real `dehip self-check` handler (issue #11): human set scored vs itself.

    Splits the reference set in half, scores half-A vs half-B through the real
    :func:`dehip.report.score` path (issue #10), and asserts the results sit
    inside the documented noise bounds. Localized to this command: the imports of
    the metric seams and self_check live here (not at module top) so the other
    stub subcommands stay import-cheap and the change stays contained.

    Exit-code contract (cli.md): a bad reference set (too few pairs, a
    manifest/texts id mismatch, a missing/corrupt manifest) is input/data failure
    -> exit 2; an embedder model-load / cache-corruption / (JMQ) missing key or
    judge-prompts is an external dependency -> exit 3; a metric out of bounds OR a
    self-check integrity failure (a split that leaked a pair into both halves) is
    the self-check's own failure -> exit 4, naming the reason, never a silent pass
    and never a bare exit-1 traceback. ``--skip-jmq`` constructs NO judge (zero
    judge spend).
    """
    from dehip.metrics.embeddings import EmbeddingCache, TransformersEmbedder
    from dehip.schemas import SchemaValidationError, SchemaVersionError
    from dehip.self_check import (
        SelfCheckIntegrityError,
        SelfCheckOutOfBounds,
        run_self_check,
    )
    from dehip.validate import InputSetValidationError

    external_dep = _external_dep_exc_types()

    # Build the embedder/cache before the try so a construction failure classifies
    # as external-dep. The judge is built lazily inside score() only when JMQ runs;
    # --skip-jmq never reaches it.
    try:
        embedder = TransformersEmbedder()
        embed_cache = EmbeddingCache(embedder)
        judge_client = None
        verdicts_path = None
        if not args.skip_jmq:
            from dehip.metrics.jmq import OpenAIJudgeClient

            judge_client = OpenAIJudgeClient()
            verdicts_path = "results/self-check-verdicts.jsonl"
    except external_dep as exc:
        _progress(f"dehip self-check: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP

    try:
        result = run_self_check(
            args.reference,
            seed=args.seed,
            skip_jmq=args.skip_jmq,
            embed_cache=embed_cache,
            judge_client=judge_client,
            verdicts_path=verdicts_path,
        )
    except SelfCheckOutOfBounds as exc:
        # Fail loudly: name every exceeded bound and the overage on stderr, and
        # exit 4 so a metric bug can never masquerade as a passing self-check.
        _progress("dehip self-check: FAILED -- results outside documented bounds:")
        for violation in exc.violations:
            _progress(f"  - {violation}")
        return EXIT_SELF_CHECK
    except SelfCheckIntegrityError as exc:
        # The check's own construction is broken (e.g. a split leaked a pair into
        # both halves). This is a fail-loudly integrity failure mapped to the
        # self-check exit code, never a bare AssertionError escaping as exit 1.
        _progress(f"dehip self-check: FAILED -- self-check integrity broken: {exc}")
        return EXIT_SELF_CHECK
    except (
        InputSetValidationError,
        SchemaValidationError,
        SchemaVersionError,
        ValueError,
        KeyError,
    ) as exc:
        _progress(f"dehip self-check: input validation failed: {exc}")
        return EXIT_VALIDATION
    except FileNotFoundError as exc:
        _progress(f"dehip self-check: reference manifest not found: {exc}")
        return EXIT_VALIDATION
    except external_dep as exc:
        _progress(f"dehip self-check: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP

    _progress(
        f"dehip self-check: PASSED -- MMD={result.mmd:.6g}, "
        f"token_l2={result.token_l2:.6g}, "
        f"jmq_win_rate={result.jmq_win_rate} "
        f"(n={result.jmq_n}, window={result.jmq_window})"
    )
    json.dump(
        {
            "command": "self-check",
            "seed": args.seed,
            "status": "ok",
            "reference": args.reference,
            "half_size": result.half_size,
            "dropped_pair_id": result.dropped_pair_id,
            "mmd": result.mmd,
            "token_l2": result.token_l2,
            "jmq_win_rate": result.jmq_win_rate,
            # The effective scaled window and the valid-comparison count the
            # win-rate was gated against, so the gate is auditable (CRITICAL 1).
            "jmq_n": result.jmq_n,
            "jmq_window": list(result.jmq_window) if result.jmq_window else None,
            "skip_jmq": args.skip_jmq,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return EXIT_SUCCESS


def _add_self_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "self-check", help="FR-003: score a human set against itself."
    )
    parser.add_argument("--reference", required=True, help="Human set manifest.")
    parser.add_argument("--skip-jmq", action="store_true")
    parser.set_defaults(func=_run_self_check)
