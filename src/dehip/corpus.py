"""Corpus builder: prompt + human-reference Pairs (FR-010, research.md R8).

The corpus is the harness's ground truth. Every scored comparison starts from a
:class:`~dehip.schemas.Pair`, so this module is where "where do the human
references come from" is answered, once, in an auditable way.

Two sources, one pair schema
----------------------------
- **FineWeb (primary, benchmark).** Stream ``HuggingFaceFW/fineweb`` (sample-10BT
  config) via ``datasets``, keep English blog/news-register documents of
  150-1200 words, drop boilerplate-heavy docs, then draw a *seeded* sample to
  the tier size. For every kept document a prompt is reverse-generated with
  ``gpt-5.4-mini`` (the judge model, to avoid a third API dependency), mirroring
  the published protocol: human document first, prompt derived from it. The
  generating model id is recorded per pair in ``prompt_generator``.
- **Personal (side corpus, spot checks only).** Ingest James's published posts
  from local paths or URLs into the *same* pair schema with ``corpus="personal"``
  and reverse-generate a prompt the same way. Personal pairs are never a
  benchmark reference: the ``corpus`` tag is the exclusion key the report layer
  reads (FR-010).

Design invariants
-----------------
- **Human document first, prompt reverse-generated.** The reference is the real
  human text; the prompt is derived from it (FR-010). Never the other way round.
- **Prompt generation is injected, not hard-wired.** The expensive,
  non-deterministic OpenAI call sits behind the :class:`PromptClient` protocol
  (mirroring :class:`~dehip.metrics.jmq.JudgeClient`). Production uses
  :class:`OpenAIPromptClient`; tests inject a stub that never touches the
  network. The FineWeb stream is likewise injectable so tests never download the
  dataset.
- **Resumable (FR-010 failure mode).** Prompt generation is the paid, failure-
  prone step. Each finished Pair is appended to the output JSONL the moment it is
  generated, so an interrupted run keeps every pair it already produced. A resume
  reads the partial file, skips pairs whose ``pair_id`` is already present, and
  never re-calls generation for them.
- **Cost gate before spend (FR-009).** :func:`estimate_call_count` reports how
  many prompt-generation calls a run will make (one per not-yet-done document);
  :func:`cost_preflight` prints the estimate and refuses to proceed above a
  spend threshold unless ``--yes`` confirms it.
- **Prompt-variant rotation for diversity.** Documents rotate through a small set
  of reverse-generation instructions (mirroring the protocol's diversity move)
  so the corpus is not a monoculture of one prompt phrasing.
- **Doc shortage is loud (FR-010 failure mode).** If filtering yields fewer
  documents than the tier requires, :func:`build_fineweb_corpus` raises
  :class:`DocShortageError` reporting the shortfall; the CLI maps it to a
  non-zero exit.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dehip.schemas import (
    Pair,
    TextSet,
    read_jsonl,
    write_json,
)

__all__ = [
    "DEFAULT_PROMPT_GENERATOR",
    "FINEWEB_DATASET",
    "FINEWEB_CONFIG",
    "TIER_SIZES",
    "WORD_COUNT_MIN",
    "WORD_COUNT_MAX",
    "PROMPT_VARIANTS",
    "DEFAULT_COST_PER_CALL_USD",
    "CorpusError",
    "DocShortageError",
    "CostThresholdError",
    "PromptClient",
    "OpenAIPromptClient",
    "estimate_call_count",
    "cost_preflight",
    "stream_fineweb",
    "iter_qualified_docs",
    "sample_documents",
    "build_fineweb_corpus",
    "build_personal_corpus",
    "write_human_reference_manifest",
]

# --- Constants (research.md R8, data-model.md Pair) --------------------------

FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"

# The judge model doubles as the prompt reverse-generator (R8: avoids a third
# API dependency). Recorded per pair so the choice can be revisited later.
DEFAULT_PROMPT_GENERATOR = "gpt-5.4-mini"

# Tier -> target pair count (contracts/cli.md: 50 / 400 / 2000).
TIER_SIZES: dict[str, int] = {"smoke": 50, "judged": 400, "full": 2000}

# Word-count band for the fineweb tier (data-model.md Pair.word_count).
WORD_COUNT_MIN = 150
WORD_COUNT_MAX = 1200

# Bounded sampling pool. ``sample_documents`` draws the tier's sample from at most
# this many qualified docs rather than materializing the whole (effectively
# unbounded) FineWeb stream, which OOMs the process on the real dataset. The pool
# is a seeded oversample of the target so the draw stays representative and
# reproducible; the cap keeps memory O(pool), not O(dataset).
POOL_OVERSAMPLE_FACTOR = 10
MIN_POOL_SIZE = 500


def pool_cap_for(target: int) -> int:
    """Bounded qualified-doc pool size for a tier ``target``."""
    return max(target * POOL_OVERSAMPLE_FACTOR, MIN_POOL_SIZE)

# Prompt-variant rotation: each document draws one of these reverse-generation
# instructions (round-robin by index) so the corpus carries a spread of prompt
# phrasings rather than one template (R8 diversity). The human document text is
# appended after the instruction by :meth:`_render_generation_request`.
PROMPT_VARIANTS: tuple[str, ...] = (
    "Write the single instruction a writer was given that would produce the "
    "text below. Reply with only the instruction.",
    "Reconstruct the writing prompt this passage answers. Output just the "
    "prompt, one or two sentences, no preamble.",
    "What task or question was this piece written to address? Give only the "
    "task as a direct instruction.",
    "Infer the assignment behind this text and state it as a prompt. Return "
    "the prompt alone, nothing else.",
)

# Per-call spend estimate for the cost preflight. A deliberately conservative
# placeholder (the protocol's dollar figure is unpublished); it only gates a
# confirmation prompt, never actual pricing, and is recorded for audit.
DEFAULT_COST_PER_CALL_USD = 0.0005

# Boilerplate quality screen. A document is rejected when too large a share of
# its lines look like navigation/legal boilerplate (cookie banners, "all rights
# reserved", subscribe/sign-in chrome) rather than prose.
_BOILERPLATE_MARKERS = (
    "cookie",
    "privacy policy",
    "all rights reserved",
    "subscribe",
    "sign in",
    "sign up",
    "terms of service",
    "©",
    "copyright",
    "read more",
    "click here",
)
_BOILERPLATE_LINE_FRACTION = 0.30

_WORD_RE = re.compile(r"\S+")


class CorpusError(RuntimeError):
    """Base class for corpus-builder failures."""


class DocShortageError(CorpusError):
    """Raised when filtered documents fall short of the requested tier size."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"document shortage after filtering: need {requested}, "
            f"only {available} qualified"
        )


class CostThresholdError(CorpusError):
    """Raised when estimated spend exceeds the threshold without confirmation."""


# --- Prompt-generation client (injectable) ----------------------------------


@runtime_checkable
class PromptClient(Protocol):
    """Reverse-generate one prompt from one human document.

    ``document`` is the human reference text; ``instruction`` is the rotated
    prompt-variant that tells the model how to derive a prompt. The return value
    is the generated prompt text. Mirrors
    :class:`~dehip.metrics.jmq.JudgeClient`: parsing and record assembly happen
    in this module, never in the client, so a test stub is a one-method mock.
    """

    def generate_prompt(
        self, document: str, *, instruction: str, model: str
    ) -> str: ...


class OpenAIPromptClient:
    """Production :class:`PromptClient` backed by the OpenAI chat API.

    The ``openai`` SDK is imported lazily so importing this module (and every
    test that injects a stub) never requires the SDK or an API key.
    """

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            from openai import OpenAI  # lazy: only the real path needs the SDK

            client = OpenAI()
        self._client = client

    def generate_prompt(
        self, document: str, *, instruction: str, model: str
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{instruction}\n\n{document}"},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


# --- Document qualification (filter + quality screen) ------------------------


def word_count(text: str) -> int:
    """Whitespace-delimited word count."""
    return len(_WORD_RE.findall(text))


def _looks_english(text: str) -> bool:
    """Cheap English-register screen: mostly ASCII letters.

    FineWeb's ``language`` field already carries the language label when present;
    this is a defensive fallback for documents that lack it. A high non-ASCII
    fraction (accented scripts, CJK) means the doc is not the English blog/news
    register the benchmark story needs.
    """
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text) <= 0.10


def _is_boilerplate_heavy(text: str) -> bool:
    """Whether too large a share of lines look like navigation/legal boilerplate."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    hits = 0
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
            hits += 1
    return hits / len(lines) > _BOILERPLATE_LINE_FRACTION


def _doc_text(doc: Any) -> str:
    """Extract the document text from a FineWeb record (or a plain string)."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(doc.get("text", ""))
    return str(getattr(doc, "text", ""))


def _doc_language(doc: Any) -> str | None:
    """FineWeb language label if present, else None."""
    if isinstance(doc, dict):
        lang = doc.get("language")
        return str(lang) if lang is not None else None
    lang = getattr(doc, "language", None)
    return str(lang) if lang is not None else None


def _doc_source(doc: Any, index: int) -> dict[str, Any]:
    """Provenance object for a FineWeb record (data-model.md Pair.source)."""
    source: dict[str, Any] = {"dataset": FINEWEB_DATASET, "config": FINEWEB_CONFIG}
    doc_id = None
    if isinstance(doc, dict):
        doc_id = doc.get("id") or doc.get("url")
    else:
        doc_id = getattr(doc, "id", None) or getattr(doc, "url", None)
    source["doc_id"] = str(doc_id) if doc_id is not None else f"index-{index}"
    return source


def _doc_register(doc: Any) -> str:
    """Register tag: ``news`` if the URL smells like news, else ``blog``."""
    url = ""
    if isinstance(doc, dict):
        url = str(doc.get("url", ""))
    else:
        url = str(getattr(doc, "url", "") or "")
    return "news" if "news" in url.lower() else "blog"


def stream_fineweb(*, streaming: bool = True) -> Iterable[Any]:
    """Load the FineWeb sample-10BT stream via ``datasets`` (lazy import).

    Returns an iterable of raw dataset records. Isolated so tests inject a plain
    iterable of fake docs in its place and never download the dataset.
    """
    from datasets import load_dataset  # lazy: only the real path needs datasets

    return load_dataset(
        FINEWEB_DATASET, name=FINEWEB_CONFIG, split="train", streaming=streaming
    )


def iter_qualified_docs(
    docs: Iterable[Any],
    *,
    word_min: int = WORD_COUNT_MIN,
    word_max: int = WORD_COUNT_MAX,
) -> Iterator[dict[str, Any]]:
    """Yield qualified documents from a raw FineWeb stream.

    A document qualifies when it is English register, within the word band, and
    not boilerplate-heavy. Each yielded item carries the extracted ``text`` and
    its computed ``word_count`` so downstream code does not recompute them.
    """
    for index, doc in enumerate(docs):
        text = _doc_text(doc).strip()
        if not text:
            continue
        wc = word_count(text)
        if wc < word_min or wc > word_max:
            continue
        language = _doc_language(doc)
        if language is not None:
            if language != "en":
                continue
        elif not _looks_english(text):
            continue
        if _is_boilerplate_heavy(text):
            continue
        yield {
            "text": text,
            "word_count": wc,
            "source": _doc_source(doc, index),
            "register": _doc_register(doc),
        }


def sample_documents(
    qualified: Iterable[dict[str, Any]],
    *,
    target: int,
    seed: int,
    pool_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Seeded sample of ``target`` qualified documents.

    Materializes at most ``pool_cap`` qualified docs (default
    :func:`pool_cap_for` of ``target``) via :func:`itertools.islice`, then draws a
    reproducible sample with a seeded RNG. Bounding the pool is essential: the real
    FineWeb stream is effectively unbounded, so ``list(qualified)`` would exhaust
    memory. The seeded draw over a deterministic prefix stays reproducible. Raises
    :class:`DocShortageError` when fewer than ``target`` documents qualified within
    the cap (FR-010 failure mode).
    """
    if pool_cap is None:
        pool_cap = pool_cap_for(target)
    pool = list(itertools.islice(qualified, pool_cap))
    if len(pool) < target:
        raise DocShortageError(requested=target, available=len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, target)


# --- Cost gate (FR-009) ------------------------------------------------------


def estimate_call_count(num_documents: int) -> int:
    """Prompt-generation call count: one call per document."""
    if num_documents < 0:
        raise ValueError(f"num_documents must be non-negative, got {num_documents}")
    return num_documents


def cost_preflight(
    num_documents: int,
    *,
    confirm: bool = False,
    threshold_usd: float = 1.0,
    cost_per_call_usd: float = DEFAULT_COST_PER_CALL_USD,
    printer=print,
) -> dict[str, object]:
    """Report estimated prompt-generation cost and gate above a threshold (FR-009).

    Prints the estimated call count (one per document) and estimated spend, then,
    if the estimate is above ``threshold_usd`` and ``confirm`` is not set, raises
    :class:`CostThresholdError` so no external calls happen without an explicit
    ``--yes``. Below the threshold the estimate is reported and the run proceeds.
    Returns the estimate dict so a caller can record it for audit.
    """
    calls = estimate_call_count(num_documents)
    estimated_usd = calls * cost_per_call_usd
    printer(
        f"prompt-generation preflight: {calls} calls "
        f"(~${estimated_usd:.4f} at ${cost_per_call_usd}/call), "
        f"threshold ${threshold_usd:.2f}"
    )
    estimate: dict[str, object] = {
        "calls": calls,
        "estimated_usd": estimated_usd,
        "threshold_usd": threshold_usd,
    }
    if estimated_usd > threshold_usd and not confirm:
        raise CostThresholdError(
            f"estimated spend ${estimated_usd:.4f} exceeds threshold "
            f"${threshold_usd:.2f}; pass --yes to confirm"
        )
    return estimate


# --- Resumability helpers ----------------------------------------------------


def _load_existing_pairs(out_path: Path) -> list[Pair]:
    """Read already-generated Pairs from a partial output file, if any."""
    if not out_path.exists():
        return []
    return read_jsonl(out_path, Pair)


def _append_pair(pair: Pair, out_path: Path) -> None:
    """Append one Pair to the output JSONL, creating the file/parents as needed.

    Each pair is flushed as it is generated so an interrupted run keeps its work
    (FR-010 resumability). Reuses the schema serializer for a single record.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    from dehip.schemas import _to_dict  # single-record append; reuse serializer

    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_dict(pair), ensure_ascii=False))
        fh.write("\n")


def _render_instruction(index: int) -> str:
    """Prompt-variant for the ``index``-th document (round-robin rotation)."""
    return PROMPT_VARIANTS[index % len(PROMPT_VARIANTS)]


def _resolve_client(client: PromptClient | Any) -> PromptClient:
    """Resolve a client argument that may be an instance or a zero-arg factory.

    Tests inject a ready :class:`PromptClient` instance. The CLI injects the
    :class:`OpenAIPromptClient` *class* (a factory) so the real client (and its
    API-key requirement) is constructed only after the cost preflight passes and
    generation is about to start, never on a validation or gate rejection.
    """
    # A class (the CLI's factory) must be checked BEFORE the isinstance below:
    # PromptClient is @runtime_checkable, so isinstance(cls, PromptClient) is True
    # for the class object itself (it has a generate_prompt attribute). Without
    # this branch the class is returned unconstructed and calls bind self to the
    # first positional argument.
    if isinstance(client, type):
        return client()
    if isinstance(client, PromptClient):
        return client
    if callable(client):
        return client()
    raise TypeError(
        "client must be a PromptClient or a zero-arg factory returning one"
    )


# --- Corpus builders ---------------------------------------------------------


def build_fineweb_corpus(
    *,
    tier: str,
    out_path: str | Path,
    client: PromptClient | Any,
    seed: int = 0,
    model: str = DEFAULT_PROMPT_GENERATOR,
    docs: Iterable[Any] | None = None,
    confirm: bool = False,
    threshold_usd: float = 1.0,
    cost_per_call_usd: float = DEFAULT_COST_PER_CALL_USD,
    printer=print,
) -> list[Pair]:
    """Build the FineWeb corpus for a tier, resumable and cost-gated (FR-010, R8).

    Streams (or accepts injected) FineWeb docs, filters + quality-screens them,
    seeded-samples to the tier size, reverse-generates one prompt per doc, and
    writes schema-valid :class:`Pair` records to ``out_path`` (appended as each is
    generated). A resume skips pairs already present in ``out_path`` and never
    re-calls generation for them.

    Args:
        tier: One of ``smoke`` / ``judged`` / ``full`` (sets the target size).
        out_path: Pair JSONL path; also the resume source.
        client: Injected :class:`PromptClient` for reverse-generation.
        seed: Sampling seed (recorded implicitly via the reproducible sample).
        model: Prompt-generator model id, recorded per pair.
        docs: Injected raw doc stream; when ``None``, streams real FineWeb.
        confirm: Confirms spend above ``threshold_usd`` (``--yes``).
        threshold_usd: Spend threshold for the cost gate.
        cost_per_call_usd: Per-call spend estimate for the preflight.
        printer: Sink for preflight/progress lines (stderr in the CLI).

    Returns:
        The full list of Pairs (already-done + newly generated) for the corpus.

    Raises:
        DocShortageError: Fewer qualified docs than the tier requires.
        CostThresholdError: Estimated spend above threshold without ``confirm``.
        ValueError: Unknown ``tier``.
    """
    if tier not in TIER_SIZES:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_SIZES)}")
    target = TIER_SIZES[tier]
    out_path = Path(out_path)

    raw = docs if docs is not None else stream_fineweb()
    sampled = sample_documents(
        iter_qualified_docs(raw), target=target, seed=seed
    )

    # Assign a stable pair_id per sampled document (index-based, deterministic
    # given the seeded sample) so a resume can match already-done work.
    planned: list[tuple[str, dict[str, Any], str]] = []
    for index, doc in enumerate(sampled):
        pair_id = f"fineweb-{index:05d}"
        instruction = _render_instruction(index)
        planned.append((pair_id, doc, instruction))

    existing = _load_existing_pairs(out_path)
    done_ids = {pair.pair_id for pair in existing}
    pairs_by_id = {pair.pair_id: pair for pair in existing}

    remaining = [item for item in planned if item[0] not in done_ids]
    cost_preflight(
        len(remaining),
        confirm=confirm,
        threshold_usd=threshold_usd,
        cost_per_call_usd=cost_per_call_usd,
        printer=printer,
    )

    resolved = _resolve_client(client) if remaining else None
    for pair_id, doc, instruction in remaining:
        prompt = resolved.generate_prompt(
            doc["text"], instruction=instruction, model=model
        )
        pair = Pair(
            pair_id=pair_id,
            corpus="fineweb",
            prompt=prompt,
            reference_text=doc["text"],
            source=doc["source"],
            register=doc["register"],
            prompt_generator=model,
            word_count=doc["word_count"],
        )
        _append_pair(pair, out_path)
        pairs_by_id[pair_id] = pair

    # Return in planned order so the manifest is deterministic.
    return [pairs_by_id[pair_id] for pair_id, _, _ in planned]


def _read_personal_document(spec: str) -> tuple[str, dict[str, Any]]:
    """Read one personal-post document from a local path or a URL.

    Returns ``(text, source)``. A ``http://`` / ``https://`` spec is fetched;
    anything else is treated as a local file path.
    """
    if spec.startswith("http://") or spec.startswith("https://"):
        from urllib.request import urlopen  # lazy: only the URL path needs it

        with urlopen(spec) as response:  # noqa: S310 - user-supplied trusted URL
            text = response.read().decode("utf-8", errors="replace")
        return text, {"url": spec}
    path = Path(spec)
    text = path.read_text(encoding="utf-8")
    return text, {"path": str(path)}


def build_personal_corpus(
    *,
    sources: Iterable[str],
    out_path: str | Path,
    client: PromptClient | Any,
    model: str = DEFAULT_PROMPT_GENERATOR,
    documents: Iterable[tuple[str, dict[str, Any]]] | None = None,
    confirm: bool = False,
    threshold_usd: float = 1.0,
    cost_per_call_usd: float = DEFAULT_COST_PER_CALL_USD,
    printer=print,
) -> list[Pair]:
    """Build the personal side corpus from James's published posts (FR-010).

    Reads each post (local path or URL) into the same Pair schema, reverse-
    generates a prompt via ``client``, and tags every pair ``corpus="personal"``
    so the report layer excludes it from benchmark rows. Resumable and cost-gated
    like the FineWeb path.

    Args:
        sources: Local paths or URLs for the published posts.
        out_path: Pair JSONL path; also the resume source.
        client: Injected :class:`PromptClient` for reverse-generation.
        model: Prompt-generator model id, recorded per pair.
        documents: Injected ``(text, source)`` pairs; when given, ``sources`` is
            ignored and no path/URL reading happens (test seam).
        confirm / threshold_usd / cost_per_call_usd / printer: Cost gate, as for
            :func:`build_fineweb_corpus`.

    Returns:
        The full list of personal Pairs (already-done + newly generated).
    """
    out_path = Path(out_path)

    if documents is not None:
        loaded = list(documents)
    else:
        loaded = [_read_personal_document(spec) for spec in sources]

    planned: list[tuple[str, str, dict[str, Any], int, str]] = []
    for index, (text, source) in enumerate(loaded):
        text = text.strip()
        pair_id = f"personal-{index:05d}"
        instruction = _render_instruction(index)
        planned.append((pair_id, text, source, word_count(text), instruction))

    existing = _load_existing_pairs(out_path)
    done_ids = {pair.pair_id for pair in existing}
    pairs_by_id = {pair.pair_id: pair for pair in existing}

    remaining = [item for item in planned if item[0] not in done_ids]
    cost_preflight(
        len(remaining),
        confirm=confirm,
        threshold_usd=threshold_usd,
        cost_per_call_usd=cost_per_call_usd,
        printer=printer,
    )

    resolved = _resolve_client(client) if remaining else None
    for pair_id, text, source, wc, instruction in remaining:
        prompt = resolved.generate_prompt(text, instruction=instruction, model=model)
        pair = Pair(
            pair_id=pair_id,
            corpus="personal",
            prompt=prompt,
            reference_text=text,
            source=source,
            register="blog",
            prompt_generator=model,
            word_count=wc,
        )
        _append_pair(pair, out_path)
        pairs_by_id[pair_id] = pair

    return [pairs_by_id[pair_id] for pair_id, *_ in planned]


# --- Manifest ----------------------------------------------------------------


def write_human_reference_manifest(
    pairs: list[Pair],
    *,
    set_id: str,
    manifest_path: str | Path,
) -> TextSet:
    """Write a role=human_reference TextSet manifest over ``pairs`` (contracts/cli.md).

    All pairs must share one ``corpus`` tag (the manifest's corpus is homogeneous;
    enforced by :class:`~dehip.schemas.TextSet`). Returns the written manifest.
    """
    if not pairs:
        raise CorpusError("cannot build a human_reference manifest over zero pairs")
    corpora = {pair.corpus for pair in pairs}
    if len(corpora) != 1:
        raise CorpusError(
            f"manifest corpus must be homogeneous, got {sorted(corpora)}"
        )
    manifest = TextSet(
        set_id=set_id,
        role="human_reference",
        corpus=next(iter(corpora)),
        pair_ids=[pair.pair_id for pair in pairs],
        provenance={"builder": "dehip build-corpus", "count": len(pairs)},
    )
    write_json(manifest, manifest_path)
    return manifest
