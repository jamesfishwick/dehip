"""Stage-1 instruct-draft generation for the HIP cascade (issue #12, R3).

The first cascade stage: for every corpus :class:`~dehip.schemas.Pair`, run an
instruct model on the pair's ``prompt`` to produce a *draft* text, and persist
that draft inside a nascent :class:`~dehip.schemas.RewriteBundle` (one bundle
per pair, ``draft: {text, model_id, sampling: {temperature, top_p, seed}}``).
The rewrite stage (a later ticket) fills in the ``rounds``.

Two design commitments drive the module shape, both aimed at an adversarial
review:

- **The model is an injectable seam.** :class:`DraftModel` is a ``Protocol``
  with a single ``generate`` method. Tests inject a stub (a call-counting
  callable), so the resumability, metadata, and manifest logic are exercised
  *without* transformers, torch, MPS/CUDA, or any network. The real
  transformers path (:class:`TransformersDraftModel`) is thin glue behind the
  seam and is constructed lazily, only when a real run is about to start.

- **Resumability keys on ``pair_id`` and survives a truncated tail.** Bundles
  are appended one JSON object per line to ``bundles.jsonl`` and flushed as each
  completes, mirroring how ``jmq.py`` persists verdicts and ``corpus.py``
  appends pairs. On resume, :func:`load_done_pair_ids` reads that file line by
  line and **skips any unparseable line** (a partially-written final record left
  by a crash mid-flush) rather than aborting, counting the skips so the caller
  can report them. A pair whose bundle was truncated is simply regenerated -- no
  duplicate, no corruption.

Seeds (torch + python ``random`` + numpy) are recorded in every bundle's
``draft.sampling.seed`` so a run is reproducible. The real model seeds all three
RNGs before each generation call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dehip.schemas import (
    RewriteBundle,
    SchemaValidationError,
    SchemaVersionError,
    TextSet,
    _to_dict,
    write_json,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "ModelLoadError",
    "DraftModel",
    "TransformersDraftModel",
    "load_done_pair_ids",
    "generate_drafts",
]

# Verified 2026-08-05 that the HF repo id resolves (200 from the HF model API).
# The 8B variant is configurable via --model; Qwen/Qwen3-8B is the valid 8B repo
# (there is no Qwen3-8B-Instruct-2507 -- that id 401s).
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Draft records for pair p live at bundles.jsonl (one RewriteBundle per line).
# The draft texts a scorer consumes live at a sibling {pair_id, text} JSONL the
# draft TextSet manifest points at via provenance.texts_path.
BUNDLES_FILENAME = "bundles.jsonl"
DRAFT_TEXTS_FILENAME = "draft-texts.jsonl"
DRAFT_MANIFEST_FILENAME = "draft.manifest.json"


class ModelLoadError(RuntimeError):
    """Raised when the instruct model fails to load (bad repo id, no weights).

    The CLI maps this to exit 3 (external-dependency failure), matching the
    exit-code discipline the ``score`` command uses, so a bad ``--model`` or a
    missing local checkout reports a clean exit-3 diagnostic rather than a bare
    transformers traceback.
    """


@runtime_checkable
class DraftModel(Protocol):
    """The generation seam: produce one draft text for a prompt.

    Implementations are seeded per call so a run is reproducible from the
    recorded seed. The real implementation is :class:`TransformersDraftModel`;
    tests inject a stub so no weights or network are touched.
    """

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        """Return the model's draft continuation for ``prompt``."""
        ...


class TransformersDraftModel:
    """Thin transformers glue behind the :class:`DraftModel` seam.

    Everything transformers-specific lives here so the rest of the module (and
    every unit test) never imports torch or transformers. The model and
    tokenizer load lazily on first :meth:`generate`, and any load failure is
    normalized to :class:`ModelLoadError` (-> CLI exit 3). Device is autodetected
    MPS -> CUDA -> CPU.
    """

    def __init__(self, model_id: str, *, max_new_tokens: int = 1024) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    @staticmethod
    def _autodetect_device() -> str:
        """Pick the best available torch device: MPS, then CUDA, then CPU."""
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_loaded(self) -> None:
        """Load the tokenizer + model once; normalize failures to ModelLoadError.

        A bad repo id, missing weights, or a transformers/torch import problem all
        surface here as :class:`ModelLoadError` so the CLI maps them to exit 3
        rather than leaking a raw traceback.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = self._autodetect_device()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32 if self._device == "cpu" else "auto",
            )
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - normalize every load failure
            raise ModelLoadError(
                f"failed to load instruct model {self.model_id!r}: {exc}"
            ) from exc

    def _seed_everything(self, seed: int) -> None:
        """Seed torch, python ``random``, and numpy so the draft is reproducible."""
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self._device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        """Generate one draft for ``prompt`` with the recorded sampling settings."""
        self._ensure_loaded()
        self._seed_everything(seed)

        import torch

        # Use the chat template so the instruct model sees a proper user turn.
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        # Strip the prompt tokens; decode only the newly generated continuation.
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


# --- Resumability ------------------------------------------------------------


def load_done_pair_ids(bundles_path: str | Path) -> tuple[set[str], int]:
    """Return the pair_ids already drafted, plus a count of skipped bad lines.

    Reads ``bundles.jsonl`` line by line. A line that is empty, is not valid
    JSON, or fails the :class:`~dehip.schemas.RewriteBundle` version/shape check
    is a partially-written record from a crash mid-flush: it is **skipped** (and
    counted), not fatal, so the truncated final record of an interrupted run does
    not corrupt the resume. The pair it belonged to is simply regenerated, which
    overwrites nothing (bundles are append-only and that pair is absent from the
    returned done-set).

    Returns ``(done_pair_ids, skipped_line_count)``. A missing file is a fresh
    run: ``(set(), 0)``.
    """
    path = Path(bundles_path)
    if not path.exists():
        return set(), 0

    done: set[str] = set()
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line (crash mid-flush). Skip and count.
                skipped += 1
                continue
            try:
                # Reuse the schema deserializer so a version/shape drift is caught
                # rather than trusting a bare dict. A malformed-but-parseable line
                # (wrong version, missing field) is also treated as regenerable.
                bundle = _bundle_from_raw(raw)
            except (SchemaVersionError, SchemaValidationError, TypeError, KeyError):
                skipped += 1
                continue
            done.add(bundle.pair_id)
    return done, skipped


def _bundle_from_raw(raw: dict[str, Any]) -> RewriteBundle:
    """Deserialize one RewriteBundle dict through the schema version/shape gate."""
    # read_jsonl is the public reader but takes a path; reuse its private core so
    # a single-record parse goes through the same version + unknown-field checks.
    from dehip.schemas import _from_dict

    return _from_dict(raw, RewriteBundle)


def _append_bundle(bundle: RewriteBundle, bundles_path: Path) -> None:
    """Append one RewriteBundle to bundles.jsonl, flushing so a crash keeps it.

    Mirrors ``corpus._append_pair`` / ``jmq``'s single-writer persistence: each
    record is written and flushed to the OS as it completes, so an interrupt
    keeps every finished draft on disk.
    """
    bundles_path.parent.mkdir(parents=True, exist_ok=True)
    with bundles_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_dict(bundle), ensure_ascii=False))
        fh.write("\n")
        fh.flush()


# --- Draft-texts manifest ----------------------------------------------------


def _rewrite_texts_file(run_dir: Path) -> Path:
    return run_dir / DRAFT_TEXTS_FILENAME


def _write_draft_texts_and_manifest(
    bundles: list[RewriteBundle],
    *,
    run_dir: Path,
    set_id: str,
    corpus: str,
    run_id: str,
    model_id: str,
) -> TextSet:
    """Rewrite the draft texts JSONL + the role=instruct_draft TextSet manifest.

    Both are regenerated wholesale from the authoritative ``bundles`` list (the
    resume-safe append-only file), so they always reflect exactly the pairs that
    have a persisted draft -- never a stale or partial view. The manifest points
    at the texts JSONL through ``provenance.texts_path`` so ``dehip score
    --candidate <this manifest>`` resolves the draft texts (see
    ``report._texts_path_for``).
    """
    texts_path = _rewrite_texts_file(run_dir)
    texts_path.parent.mkdir(parents=True, exist_ok=True)
    with texts_path.open("w", encoding="utf-8") as fh:
        for bundle in bundles:
            assert bundle.draft is not None  # every generate bundle carries a draft
            fh.write(
                json.dumps(
                    {"pair_id": bundle.pair_id, "text": bundle.draft["text"]},
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

    manifest = TextSet(
        set_id=set_id,
        role="instruct_draft",
        corpus=corpus,
        pair_ids=[bundle.pair_id for bundle in bundles],
        provenance={
            "builder": "dehip generate",
            "run_id": run_id,
            "model_id": model_id,
            "count": len(bundles),
            "texts_path": DRAFT_TEXTS_FILENAME,
        },
    )
    write_json(manifest, run_dir / DRAFT_MANIFEST_FILENAME)
    return manifest


# --- Draft generation --------------------------------------------------------


def _make_bundle(
    *,
    run_id: str,
    pair,
    draft_text: str,
    model_id: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> RewriteBundle:
    """Build a nascent RewriteBundle carrying the draft + full sampling metadata.

    ``rounds`` is empty (the rewrite stage fills it), ``final_round`` is 0, and
    ``degeneration`` is a not-yet-run marker. The model id and every sampling
    setting (temperature, top_p, seed) are recorded in ``draft`` so a review can
    confirm reproducibility metadata is present in every record.
    """
    return RewriteBundle(
        run_id=run_id,
        pair_id=pair.pair_id,
        prompt=pair.prompt,
        rounds=[],
        final_round=0,
        degeneration={"hard_tripped": False},
        adapter_id="",
        hip_config={},
        requested_k=0,
        draft={
            "text": draft_text,
            "model_id": model_id,
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
        },
    )


def generate_drafts(
    pairs,
    *,
    model: DraftModel,
    run_dir: str | Path,
    run_id: str,
    model_id: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = 0,
    printer=print,
) -> dict[str, Any]:
    """Generate one instruct draft per corpus pair, seeded and resumable.

    For each :class:`~dehip.schemas.Pair` in ``pairs`` whose ``pair_id`` does not
    already have a persisted bundle, calls ``model.generate`` on the pair's
    prompt and appends a nascent :class:`~dehip.schemas.RewriteBundle` (carrying
    the draft text, ``model_id``, and the ``{temperature, top_p, seed}`` sampling
    block) to ``bundles.jsonl``, flushed per record.

    Resumability (the testable core): an interrupted run is continued by simply
    re-invoking with the same ``run_dir``. Already-drafted pairs are detected via
    :func:`load_done_pair_ids` and **not** re-generated -- so a first run that
    writes ``k`` of ``N`` before a kill produces exactly ``N - k`` further
    ``model.generate`` calls on re-run, with no duplicate bundles.

    After generation, the draft-texts JSONL and the role=instruct_draft TextSet
    manifest are rewritten from the full bundle list so a scorer can consume the
    drafts directly.

    Args:
        pairs: Corpus :class:`~dehip.schemas.Pair` records (homogeneous corpus).
        model: The injected :class:`DraftModel` seam (real or stub).
        run_dir: Output run directory (``results/runs/{run_id}/``).
        run_id: Timestamped run identifier recorded in every bundle.
        model_id: Model id recorded in every bundle's ``draft.model_id``.
        temperature / top_p / seed: Sampling settings recorded per bundle.
        printer: Progress sink (stderr in the CLI).

    Returns:
        A summary dict: pair counts (total / already-done / generated), the
        skipped-truncated-line count, and the written artifact paths.
    """
    run_dir = Path(run_dir)
    bundles_path = run_dir / BUNDLES_FILENAME

    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("cannot generate drafts over zero pairs")

    corpora = {pair.corpus for pair in pair_list}
    if len(corpora) != 1:
        raise ValueError(
            f"generate expects a homogeneous corpus, got {sorted(corpora)}"
        )
    corpus = next(iter(corpora))

    done_ids, skipped = load_done_pair_ids(bundles_path)
    if skipped:
        printer(
            f"dehip generate: skipped {skipped} truncated/corrupt bundle line(s) "
            "from a prior interrupted run; those pairs will be regenerated"
        )

    remaining = [pair for pair in pair_list if pair.pair_id not in done_ids]
    printer(
        f"dehip generate: {len(pair_list)} pairs, {len(done_ids)} already drafted, "
        f"{len(remaining)} to generate"
    )

    generated = 0
    for pair in remaining:
        draft_text = model.generate(
            pair.prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        bundle = _make_bundle(
            run_id=run_id,
            pair=pair,
            draft_text=draft_text,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        _append_bundle(bundle, bundles_path)
        generated += 1

    # Re-read the authoritative append-only file to assemble the manifest in a
    # deterministic, corpus-order-independent way, and to double-count against the
    # pair list. read_jsonl is strict (skips nothing), so at this point every line
    # must parse: the only bad lines were truncated *prior*-run tails, which are
    # not rewritten by an append, so a partial tail could still be present. Read
    # defensively through load_done_pair_ids' tolerant path instead.
    bundles = _read_all_bundles(bundles_path)

    # Order the manifest by the input pair order for determinism.
    order = {pair.pair_id: index for index, pair in enumerate(pair_list)}
    bundles.sort(key=lambda b: order.get(b.pair_id, len(order)))

    set_id = f"{corpus}-{len(bundles)}-draft"
    manifest = _write_draft_texts_and_manifest(
        bundles,
        run_dir=run_dir,
        set_id=set_id,
        corpus=corpus,
        run_id=run_id,
        model_id=model_id,
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pairs": len(pair_list),
        "already_done": len(done_ids),
        "generated": generated,
        "skipped_truncated": skipped,
        "bundles": str(bundles_path),
        "draft_texts": str(_rewrite_texts_file(run_dir)),
        "manifest": str(run_dir / DRAFT_MANIFEST_FILENAME),
        "set_id": manifest.set_id,
    }


def _read_all_bundles(bundles_path: Path) -> list[RewriteBundle]:
    """Read every well-formed bundle, tolerating a truncated final line.

    Unlike a bare :func:`~dehip.schemas.read_jsonl`, this skips an unparseable
    tail (a crash mid-flush from a prior run whose pair was just regenerated as a
    fresh appended record) so the manifest assembly never dies on a partial line.
    A regenerated pair appears once here (its good record); the truncated old line
    is dropped.
    """
    good: list[RewriteBundle] = []
    seen: set[str] = set()
    with bundles_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                bundle = _bundle_from_raw(json.loads(line))
            except (
                json.JSONDecodeError,
                SchemaVersionError,
                SchemaValidationError,
                TypeError,
                KeyError,
            ):
                continue
            # Last writer wins for a duplicated pair_id (a regenerated pair whose
            # prior truncated line happened to parse). Keep the latest record.
            if bundle.pair_id in seen:
                good = [b for b in good if b.pair_id != bundle.pair_id]
            seen.add(bundle.pair_id)
            good.append(bundle)
    return good
