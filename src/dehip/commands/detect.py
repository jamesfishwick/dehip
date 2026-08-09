"""`dehip detect` handler and parser wiring."""

import argparse
import json
import sys

from dehip.commands._shared import (
    EXIT_EXTERNAL_DEP,
    EXIT_IO,
    EXIT_SC005_NOT_COMPUTED,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    _progress,
)


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
