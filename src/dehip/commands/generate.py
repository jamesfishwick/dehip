"""`dehip generate` handler and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _progress,
)


def _add_generate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate", help="Stage-1 instruct drafts for a corpus (R3)."
    )
    parser.add_argument("--corpus", required=True, help="Corpus JSONL to draft from.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", help="Output run directory.")
    parser.set_defaults(func=_run_generate)


def _run_generate(args: argparse.Namespace) -> int:
    """Real `dehip generate` handler (issue #12): stage-1 instruct drafts (R3).

    Localized to this command; imports of dehip.generate and dehip.schemas are
    done here (not at module top) so the other stub subcommands stay
    import-cheap and the change stays contained to the generate path.

    Exit-code contract (cli.md): a bad/missing corpus file or a bad-shape corpus
    record (the user's input being wrong) maps to EXIT_VALIDATION; a model-load
    failure (bad repo id, no weights) maps to EXIT_EXTERNAL_DEP, mirroring the
    exit-3 discipline the score command uses. The real transformers model is
    constructed as a lazy seam and only its generation is reached after the
    corpus loads, so a bad corpus on a weightless machine still reports exit 2.
    """
    import time
    from pathlib import Path

    from dehip import generate as generate_mod
    from dehip.schemas import (
        Pair,
        SchemaValidationError,
        SchemaVersionError,
        read_jsonl,
    )

    # Load + validate the corpus first (exit 2 on a bad path or bad-shape record),
    # before any model is built, so a weightless machine still reports exit 2 on
    # bad input rather than exit 3.
    try:
        pairs = read_jsonl(args.corpus, Pair)
    except OSError as exc:
        # OSError covers FileNotFoundError plus IsADirectoryError / PermissionError
        # (--corpus pointing at a directory or an unreadable file) -- all the
        # user's input being wrong -> exit 2, mirroring the score command's OSError
        # handling, rather than escaping as a bare exit-1 traceback.
        _progress(f"dehip generate: corpus not readable: {exc}")
        return EXIT_VALIDATION
    except (json.JSONDecodeError, SchemaVersionError, SchemaValidationError) as exc:
        _progress(f"dehip generate: corrupt corpus file: {exc}")
        return EXIT_VALIDATION
    if not pairs:
        _progress("dehip generate: corpus is empty")
        return EXIT_VALIDATION

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out or f"results/runs/{run_id}/"

    # The model is an injectable seam. The real transformers model is built here
    # but loads its weights lazily on first generate(), so its only failure mode
    # (ModelLoadError -> exit 3) is reached during generation, not construction.
    model = generate_mod.TransformersDraftModel(args.model)
    try:
        summary = generate_mod.generate_drafts(
            pairs,
            model=model,
            run_dir=run_dir,
            run_id=run_id,
            model_id=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            printer=_progress,
        )
    except generate_mod.ModelLoadError as exc:
        # A bad --model repo id or missing weights is an external-dependency
        # failure (exit 3), matching the score command's discipline -- not a bare
        # traceback.
        _progress(f"dehip generate: model load failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except generate_mod.GenerationError as exc:
        # A mid-run generation failure that the seam normalized to GenerationError
        # (torch OOM, a device-side RuntimeError, a missing-chat-template
        # KeyError, or an empty/degenerate draft) is an external-dependency
        # failure (exit 3), not a bare exit-1 traceback. Scoped to the seam's
        # normalized type so a genuine logic bug in generate_drafts is NOT
        # swallowed as exit 3.
        _progress(f"dehip generate: generation failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except ValueError as exc:
        # Zero-pairs / heterogeneous-corpus / corpus-drift input errors -> exit 2.
        # (CorpusDriftError subclasses ValueError.)
        _progress(f"dehip generate: input error: {exc}")
        return EXIT_VALIDATION

    json.dump(
        {
            "command": "generate",
            "seed": args.seed,
            "status": "ok",
            "model": args.model,
            **{k: summary[k] for k in ("run_id", "pairs", "generated", "manifest")},
            "out": str(Path(run_dir)),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return EXIT_SUCCESS
