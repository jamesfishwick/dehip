"""Command-line entry point for the dehip harness.

Subcommand dispatch (build-corpus, generate, rewrite, score, self-check,
detect, report) is issue #4. For now this exposes a top-level parser so
`dehip --help` succeeds.
"""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dehip",
        description="HIP cascade and evaluation harness.",
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
