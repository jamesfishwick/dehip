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
import json
import sys

from dehip import corpus as corpus_mod

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


def _verdicts_derived_judge_model(verdicts, default: str) -> str:
    """Derive the judge_model to record from persisted verdicts (IMPORTANT 3).

    Each :class:`~dehip.schemas.JudgeVerdict` row carries the ``judge_model`` that
    actually produced it, so a recompute must reflect that value rather than
    stamp the CLI's ``--judge`` default (``gpt-5.4-mini``). When every row shares
    one model, that shared value is returned. When the rows are mixed (or empty),
    the false default is still avoided: a ``mixed:{a,b,...}`` marker records the
    fact so config.judge_model and the non-default-judge caveat both reflect
    reality instead of a made-up single judge.
    """
    models = {v.judge_model for v in verdicts}
    if not models:
        return default
    if len(models) == 1:
        return next(iter(models))
    return "mixed:" + ",".join(sorted(models))


def _run_score(args: argparse.Namespace) -> int:
    """Real `dehip score` handler (issue #10): compose the metrics into a report.

    Localized to this command. Imports of dehip.report and the metric seams are
    done here (not at module top) so the other stub subcommands stay import-cheap
    and the change stays contained to the score path.

    Exit-code contract (cli.md): external-dependency failures (missing/invalid
    OPENAI_API_KEY, missing judge-prompts/, embedder model-load, torch OOM,
    embedding-cache corruption) map to EXIT_EXTERNAL_DEP; input/data-class
    failures (id mismatch, metric-selection or MMD-degeneracy ValueError, a
    corrupt/truncated/wrong-version verdicts file on the recompute path) map to
    EXIT_VALIDATION; a report-write I/O failure maps to EXIT_IO. The judge client
    is constructed lazily *after* validation so bad input on a keyless machine
    still reports exit 2 (bad input), not exit 3.
    """
    from pathlib import Path

    from dehip import report as report_mod

    if args.recompute_jmq_from:
        return _run_recompute(args, report_mod, Path)
    return _run_full_score(args, report_mod, Path)


# Exceptions that signal an external dependency failed (map to EXIT_EXTERNAL_DEP).
# Kept as a helper so both score paths classify identically. Imports are lazy
# (openai is optional) and any missing module degrades gracefully.
def _external_dep_exc_types() -> tuple[type[BaseException], ...]:
    # OSError covers FileNotFoundError and embedder model-load OSError.
    types: list[type[BaseException]] = [OSError]
    try:
        from dehip.metrics.embeddings import CacheIntegrityError

        types.append(CacheIntegrityError)
    except Exception:  # pragma: no cover - embeddings always importable here
        pass
    try:
        import openai

        types.append(openai.OpenAIError)  # incl. missing OPENAI_API_KEY at construct
    except Exception:  # openai SDK not installed: nothing to add
        pass
    return tuple(types)


def _run_recompute(args: argparse.Namespace, report_mod, Path) -> int:
    """--recompute-jmq-from branch: re-aggregate JMQ from persisted verdicts.

    Reads only the file (FR-008): no judge/API calls, no embedder. A corrupt,
    truncated, or wrong-version verdicts file (json.JSONDecodeError,
    SchemaVersionError, SchemaValidationError) is an input/data failure -> exit
    2, never a bare exit-1 traceback. The recorded judge_model is derived from
    the verdict rows, not the CLI default (IMPORTANT 3).
    """
    from dehip.schemas import SchemaValidationError, SchemaVersionError

    _progress(f"dehip score: recomputing JMQ from {args.recompute_jmq_from}")
    try:
        jmq_scores, verdicts = report_mod.recompute_jmq(args.recompute_jmq_from)
    except (json.JSONDecodeError, SchemaVersionError, SchemaValidationError) as exc:
        _progress(f"dehip score: corrupt verdicts file: {exc}")
        return EXIT_VALIDATION
    except FileNotFoundError as exc:
        # A missing verdicts file the docs told the user to pass is an external
        # dependency (the artifact) that is absent -> exit 3.
        _progress(f"dehip score: verdicts file not found: {exc}")
        return EXIT_EXTERNAL_DEP

    judge_model = _verdicts_derived_judge_model(verdicts, args.judge)
    # Best-effort corpus for the FR-010 gate: read the candidate manifest's
    # homogeneous corpus tag when the manifest is a readable TextSet. A recompute
    # run may point --candidate at a set id rather than a manifest path, in which
    # case the tag is simply absent (None) and the set-id naming fallback still
    # protects the gate; a bad manifest is never fatal to a recompute.
    corpus = _best_effort_corpus(args.candidate)
    report = report_mod.assemble_report(
        report_id=Path(args.out).stem if args.out else "recompute",
        candidate_set=args.candidate,
        reference_set=args.reference,
        n=len({v.pair_id for v in verdicts}),
        seed=args.seed,
        judge_model=judge_model,
        embedder_id=args.embedder,
        tokenizer_id=None,
        mmd_result=None,
        token_l2_result=None,
        jmq_scores=jmq_scores,
        verdicts=verdicts,
        corpus=corpus,
    )
    return _emit_report(report, args.out, report_mod)


def _best_effort_corpus(candidate_manifest: str | None) -> str | None:
    """Read the candidate TextSet's corpus tag, or ``None`` if unavailable.

    Used by the recompute path, where no MetricInputs is loaded, to still stamp
    ``compared["corpus"]`` for the FR-010 gate. Any failure (the argument is a set
    id not a path, the file is missing or not a valid TextSet) degrades to ``None``
    -- the set-id naming fallback in the gate still applies -- and never aborts a
    recompute over an otherwise-valid verdicts file.
    """
    if not candidate_manifest:
        return None
    try:
        from pathlib import Path

        from dehip.schemas import TextSet, read_json

        path = Path(candidate_manifest)
        if not path.is_file():
            return None
        text_set = read_json(path, TextSet)
    except Exception:
        return None
    corpus = text_set.corpus
    return corpus if isinstance(corpus, str) and corpus else None


def _run_full_score(args: argparse.Namespace, report_mod, Path) -> int:
    """Main score path: validate, preflight, then compute the requested metrics."""
    from dehip.metrics.embeddings import EmbeddingCache, TransformersEmbedder
    from dehip.metrics.jmq import CostThresholdError, OpenAIJudgeClient, cost_preflight
    from dehip.validate import InputSetValidationError

    # Load + cross-validate the two manifests before any spend (FR-009). A
    # mismatched reference (or a texts file that is a superset of its manifest)
    # raises InputSetValidationError here -> exit 2, with zero embedder/judge
    # calls, never a later KeyError.
    try:
        inputs, cand_set_id, ref_set_id = report_mod.load_scoring_inputs(
            args.candidate, args.reference, prompts_path=args.prompts
        )
    except (InputSetValidationError, OSError, ValueError, KeyError) as exc:
        # A bad input path, a mismatched/corrupt manifest, or a duplicate/missing
        # id are all the user's input being wrong -> exit 2 (matches the contract
        # test for a missing manifest). Distinct from a missing *dependency*
        # artifact (judge-prompts/, verdicts file) handled downstream as exit 3.
        _progress(f"dehip score: input validation failed: {exc}")
        return EXIT_VALIDATION

    selected = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # JMQ cost preflight before any spend (FR-009), gated on --yes. Capture the
    # estimate so authorized spend is recorded in the report config (IMPORTANT 2)
    # rather than discarded.
    thresholds: dict | None = None
    if "jmq" in selected:
        try:
            thresholds = cost_preflight(
                len(inputs.pair_ids), confirm=args.yes, printer=_progress
            )
        except CostThresholdError as exc:
            _progress(f"dehip score: {exc}")
            return EXIT_VALIDATION

    # Build the seams. The embedder and judge are constructed only for the
    # metrics selected, and only AFTER validation + preflight have passed, so a
    # missing OPENAI_API_KEY on otherwise-bad input reports exit 2, not exit 3.
    external_dep = _external_dep_exc_types()
    try:
        embed_cache = None
        if "mmd" in selected:
            embedder = TransformersEmbedder(args.embedder)
            embed_cache = EmbeddingCache(embedder)

        judge_client = OpenAIJudgeClient() if "jmq" in selected else None
    except external_dep as exc:
        _progress(f"dehip score: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP

    # Verdicts live beside the report at <out-dir>/verdicts.jsonl, matching the
    # documented --recompute-jmq-from example. `score` has no run_id concept; the
    # results/runs/{run_id}/ layout in data-model belongs to the rewrite/run
    # pipeline (a later ticket), not this command.
    verdicts_path = None
    if "jmq" in selected:
        if args.out:
            verdicts_path = str(Path(args.out).parent / "verdicts.jsonl")
        else:
            verdicts_path = "results/verdicts.jsonl"

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
            thresholds=thresholds,
        )
    except InputSetValidationError as exc:
        _progress(f"dehip score: input validation failed: {exc}")
        return EXIT_VALIDATION
    except external_dep as exc:
        # Metric internals reaching a real dependency (embedder model-load,
        # torch OOM, missing judge-prompts/, cache corruption) -> exit 3.
        _progress(f"dehip score: external dependency failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except (ValueError, KeyError) as exc:
        # Input/data-class seams: unknown metric, id mismatch surfacing as a
        # KeyError, MMD median-heuristic on all-coincident points -> exit 2. The
        # classification is by exception TYPE (data-class errors, not external
        # deps), deliberately broad so a real input error reports exit 2 rather
        # than escaping as a bare exit-1 traceback.
        _progress(f"dehip score: input error: {exc}")
        return EXIT_VALIDATION

    return _emit_report(report, args.out, report_mod)


def _emit_report(report, out_path, report_mod) -> int:
    """Write the report JSON and a sibling .md as an all-or-nothing pair, echo JSON.

    The on-disk JSON and the stdout echo both go through
    :func:`~dehip.report.report_to_jsonable`, which converts an un-run metric's
    ``NaN`` to strict-JSON ``null`` (IMPORTANT 1) while leaving a real ``0.0``.

    The two artifacts are written as one atomic pair (NIT 3): both are staged to
    temp files in their target dir first, and only after BOTH temp writes succeed
    are they ``os.replace``d into place. A failure on either write leaves NEITHER
    final artifact (no orphaned ``.json`` with a missing ``.md``) and removes any
    temp debris. Any I/O failure maps to EXIT_IO with the path and artifact named,
    rather than a half-written pair and a bare exit-1 traceback. Returns the
    process exit code.
    """
    from pathlib import Path

    jsonable = report_mod.report_to_jsonable(report)

    if out_path:
        json_path = Path(out_path)
        md_path = json_path.with_suffix(".md")
        staged: list[tuple[Path, Path]] = []  # (tmp, final) pairs to commit
        try:
            # Stage BOTH artifacts to temp files before committing either, so a
            # failure while rendering/writing the .md cannot leave a lone .json.
            staged.append(
                (
                    _stage_text(
                        json_path,
                        json.dumps(jsonable, ensure_ascii=False, indent=2) + "\n",
                    ),
                    json_path,
                )
            )
            staged.append(
                (_stage_text(md_path, report_mod.render_markdown(report)), md_path)
            )
        except OSError as exc:
            # Clean up any temp already staged; commit nothing.
            _discard_staged(staged)
            failed = md_path if staged else json_path
            _progress(f"dehip score: failed to write report {failed}: {exc}")
            return EXIT_IO

        # Both temps written: commit them. os.replace is atomic per file; if the
        # second replace somehow fails, the discard drops the second temp (the
        # first is already committed, matching per-file atomicity).
        try:
            _commit_staged(staged)
        except OSError as exc:
            _discard_staged(staged)
            _progress(f"dehip score: failed to finalize report {out_path}: {exc}")
            return EXIT_IO
        _progress(f"dehip score: wrote {json_path} and {md_path}")

    json.dump(jsonable, sys.stdout)
    sys.stdout.write("\n")
    return EXIT_SUCCESS


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


def _run_detect(args: argparse.Namespace) -> int:
    """Real `dehip detect` handler (issue #14): external AI-text detector scoring.

    SC-005's independent instrument. Scores one or more TextSet manifests through
    an external detector (Pangram default, GPTZero optional) and writes a per-set
    human-probability summary plus every per-text score under results/reports/;
    the SC-005 delta is the rewrite-set mean minus the draft-set mean. Localized
    to this command: imports of dehip.detector and dehip.report live here (not at
    module top) so the other stub subcommands stay import-cheap.

    Exit-code contract (cli.md): the required API key is checked FIRST -- a
    missing PANGRAM_API_KEY (or GPTZERO_API_KEY for gptzero) is an external-
    dependency failure (exit 3) raised BEFORE any manifest is read and BEFORE any
    detector client is constructed, so a keyless machine never spends. The cost
    gate then prints the text count per set and, above the spend threshold without
    --yes, exits 2 (validation) with zero detector calls (mirrors the score
    command). A mid-run detector call failure (network, rate-limit, malformed
    response) is normalized to DetectorCallError and maps to exit 3 -- never a
    silent report with missing/zero scores. Bad inputs (unreadable/mismatched
    manifest, empty set) map to exit 2. A report-write I/O failure maps to
    EXIT_IO.
    """
    import os
    from pathlib import Path

    from dehip import detector as detector_mod
    from dehip import report as report_mod
    from dehip.schemas import (
        SchemaValidationError,
        SchemaVersionError,
        TextSet,
        read_json,
    )

    # CRITICAL: enforce the required key BEFORE reading any manifest or building
    # any client, so a missing key never causes partial spend. os.environ is read
    # directly here (not inside the seam) so this gate is unambiguously ahead of
    # every detector call.
    env_key = detector_mod.ENV_KEY_BY_DETECTOR[args.detector]
    if not os.environ.get(env_key):
        _progress(
            f"dehip detect: {env_key} is not set; the {args.detector!r} detector "
            "requires it. Set the key and re-run (no calls were made)."
        )
        return EXIT_EXTERNAL_DEP

    # Validate a user-supplied --out ends in .json BEFORE any spend: a non-.json
    # path would mangle the derived scores sibling and report_id (see
    # detector.scores_path_for). The default path is always .json. Pure input
    # validation -> exit 2 with zero detector calls.
    out_path = args.out or f"results/reports/{args.detector}-detect.json"
    if not str(out_path).endswith(".json"):
        _progress(
            f"dehip detect: --out must end in .json, got {out_path!r} "
            "(a non-.json path would mangle the scores sibling and report_id)."
        )
        return EXIT_VALIDATION

    # Load every manifest + its texts via the SAME conventions the score command
    # uses (report._texts_path_for / report._read_pair_texts) -- no parallel
    # manifest reader. A bad path, bad-version/shape manifest, or duplicate id is
    # the user's input being wrong -> exit 2, before any client or spend.
    loaded: list[tuple[str, str, list[tuple[str, str]]]] = []
    text_counts: list[tuple[str, int]] = []
    try:
        for manifest in args.sets:
            manifest_path = Path(manifest)
            text_set: TextSet = read_json(manifest_path, TextSet)
            # Reject a manifest listing the same pair_id twice BEFORE counting or
            # scoring: a duplicate would score and pay for that text twice and
            # skew the set mean. Fail loudly (exit 2) naming the duplicate, in
            # keeping with the validation-before-spend discipline -- never
            # silently de-duplicate a paid run.
            seen: set[str] = set()
            duplicates: list[str] = []
            for pair_id in text_set.pair_ids:
                if pair_id in seen:
                    duplicates.append(pair_id)
                seen.add(pair_id)
            if duplicates:
                raise ValueError(
                    f"manifest {manifest!r} lists duplicate pair_id(s) "
                    f"{sorted(set(duplicates))!r}; each pair_id must appear once "
                    "(a duplicate would double-score, double-pay, and skew the "
                    "set mean)"
                )
            texts_by_id = report_mod._read_pair_texts(
                report_mod._texts_path_for(manifest_path, text_set.provenance)
            )
            # Score in the manifest's pair_id order; a missing text is a broken
            # manifest -> exit 2 (raised as KeyError, caught below), never a hole
            # in the scored set.
            ordered: list[tuple[str, str]] = []
            for pair_id in text_set.pair_ids:
                if pair_id not in texts_by_id:
                    raise KeyError(
                        f"manifest {manifest!r} references pair_id {pair_id!r} "
                        "absent from its texts file"
                    )
                ordered.append((pair_id, texts_by_id[pair_id]))
            if not ordered:
                _progress(
                    f"dehip detect: set {text_set.set_id!r} ({manifest}) is empty"
                )
                return EXIT_VALIDATION
            loaded.append((text_set.set_id, text_set.role, ordered))
            text_counts.append((text_set.set_id, len(ordered)))
    except (
        OSError,
        json.JSONDecodeError,
        SchemaVersionError,
        SchemaValidationError,
        ValueError,
        KeyError,
    ) as exc:
        _progress(f"dehip detect: input validation failed: {exc}")
        return EXIT_VALIDATION

    # Cost gate (FR-009): print the text count per set and require --yes above the
    # spend threshold. A below-threshold run proceeds; above without --yes exits 2
    # with ZERO detector calls (the client is built only after this passes).
    try:
        thresholds = detector_mod.cost_preflight(
            text_counts, confirm=args.yes, printer=_progress
        )
    except detector_mod.CostThresholdError as exc:
        _progress(f"dehip detect: {exc}")
        return EXIT_VALIDATION

    # Build the real adapter only now -- after the key check and cost gate both
    # passed. It is the injectable seam; tests inject a mock and never reach here.
    client = detector_mod.build_client(args.detector)

    started = detector_mod._now_iso()
    try:
        summaries, scores = detector_mod.score_sets(loaded, client=client)
    except detector_mod.DetectorCallError as exc:
        # A failed/malformed detector call fails loudly (exit 3) rather than
        # emitting a report with a hole or a fake 0.0.
        _progress(f"dehip detect: detector call failed: {exc}")
        return EXIT_EXTERNAL_DEP
    except ValueError as exc:
        # An empty set reaching summarization -> input error (exit 2).
        _progress(f"dehip detect: input error: {exc}")
        return EXIT_VALIDATION

    report = detector_mod.assemble_report(
        report_id=Path(out_path).stem,
        detector=args.detector,
        seed=args.seed,
        summaries=summaries,
        thresholds=thresholds,
        started=started,
    )
    try:
        summary_path, scores_path = detector_mod.write_detection_artifacts(
            report, scores, out_path=out_path
        )
    except OSError as exc:
        _progress(f"dehip detect: failed to write report {out_path}: {exc}")
        return EXIT_IO
    _progress(f"dehip detect: wrote {summary_path} and {scores_path}")

    # A run that GAVE two or more sets requested an SC-005 delta; if the delta
    # could not be computed (not exactly one instruct_draft + one rewrite -- e.g.
    # two rewrites, or a mislabeled role), surface it loudly instead of reporting
    # ok with sc005 null after paid spend. Keep the artifacts (the per-set
    # summaries are still useful) but report a non-ok status and return a distinct
    # non-zero code so a caller can tell a computed delta from a paid run that
    # produced no decision. A single-set run never requested a delta, so its null
    # sc005 stays ok (exit 0).
    sc005_requested = len(args.sets) >= 2
    if sc005_requested and report.sc005 is None:
        status = "sc005_not_computed"
        exit_code = EXIT_SC005_NOT_COMPUTED
        _progress(
            "dehip detect: SC-005 delta could not be computed (need exactly one "
            "instruct_draft and one rewrite set); artifacts written but the run "
            "produced no decision."
        )
    else:
        status = "ok"
        exit_code = EXIT_SUCCESS

    json.dump(
        {
            "command": "detect",
            "seed": args.seed,
            "status": status,
            "detector": args.detector,
            "n_sets": report.n_sets,
            "out": str(summary_path),
            "scores": str(scores_path),
            "sc005": report.sc005,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return exit_code


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
    parser.set_defaults(func=_run_detect)


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
