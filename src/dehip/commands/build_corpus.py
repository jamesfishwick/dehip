"""`dehip build-corpus` handler and parser wiring."""

import argparse
import json
import sys

from dehip import corpus as corpus_mod
from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _progress,
)


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
    parser.set_defaults(func=_run_build_corpus)


def _run_build_corpus(args: argparse.Namespace) -> int:
    """Build the fineweb or personal corpus (FR-010, R8; issue #5)."""
    default_name = args.corpus
    out_path = args.out or f"data/corpus/{default_name}.jsonl"
    manifest_path = f"{out_path.rsplit('.', 1)[0]}.manifest.json"
    # The OpenAI client is built lazily inside the builder call so a validation
    # or cost-gate rejection never requires an API key. build_client is invoked
    # only after the preflight passes and generation is about to start.
    build_client = corpus_mod.OpenAIPromptClient
    try:
        if args.corpus == "personal":
            if not args.source:
                _progress("build-corpus: --source is required for personal corpus")
                return EXIT_VALIDATION
            sources = [s for s in args.source.split(",") if s.strip()]
            pairs = corpus_mod.build_personal_corpus(
                sources=sources,
                out_path=out_path,
                client=build_client,
                confirm=args.yes,
                printer=_progress,
            )
            set_id = "personal-human"
        else:
            pairs = corpus_mod.build_fineweb_corpus(
                tier=args.tier,
                out_path=out_path,
                client=build_client,
                seed=args.seed,
                confirm=args.yes,
                printer=_progress,
            )
            set_id = f"fineweb-{corpus_mod.TIER_SIZES[args.tier]}-human"
    except corpus_mod.DocShortageError as exc:
        _progress(f"build-corpus: {exc}")
        return EXIT_VALIDATION
    except corpus_mod.CostThresholdError as exc:
        _progress(f"build-corpus: {exc}")
        return EXIT_EXTERNAL_DEP
    corpus_mod.write_human_reference_manifest(
        pairs, set_id=set_id, manifest_path=manifest_path
    )
    json.dump(
        {
            "command": "build-corpus",
            "seed": args.seed,
            "status": "ok",
            "corpus": args.corpus,
            "pairs": len(pairs),
            "out": out_path,
            "manifest": manifest_path,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return EXIT_SUCCESS
