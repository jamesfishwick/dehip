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

- **Resumability keys on ``pair_id`` and survives a torn tail two ways.**
  Bundles are appended one JSON object per line to ``bundles.jsonl``, each
  fsync'd as it completes, mirroring how ``jmq.py`` persists verdicts and
  ``corpus.py`` appends pairs. On resume, :func:`load_done_pair_ids` reads that
  file line by line and **skips any unparseable line** (a partially-written
  final record left by a crash mid-flush) rather than aborting, counting the
  skips. It also terminates a *torn* tail -- a final record with no trailing
  newline, left by a crash between the record and newline writes -- via
  :func:`_repair_torn_tail` before the first resume append, so the regenerated
  record lands on its own line instead of concatenating onto the fragment into
  one unparseable line that would silently drop the pair. Either way a pair
  whose bundle was torn is simply regenerated -- no duplicate, no corruption.

Seeds are recorded in every bundle's ``draft.sampling.seed`` so a run is
reproducible. The recorded seed is the *effective per-pair seed*, derived
deterministically from ``(base_seed, pair_id)`` via :func:`_derive_pair_seed`,
not the run-wide base seed: resetting the same base seed before every call makes
every pair sample from an identical RNG state (correlated, low-diversity, even
byte-identical drafts when prompts coincide), so drafts are decorrelated per
pair while a single regenerated pair still reproduces its original draft. The
real model seeds all three RNGs (torch + python ``random`` + numpy) with that
effective seed before each generation call.
"""

from __future__ import annotations

import hashlib
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
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "ModelLoadError",
    "GenerationError",
    "EmptyDraftError",
    "CorpusDriftError",
    "DraftModel",
    "TransformersDraftModel",
    "load_done_pair_ids",
    "generate_drafts",
]

# Verified 2026-08-05 that the HF repo id resolves (200 from the HF model API).
# The 8B variant is configurable via --model; Qwen/Qwen3-8B is the valid 8B repo
# (there is no Qwen3-8B-Instruct-2507 -- that id 401s).
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Pin the draft model's commit so the drafts a run produces are reproducible (the
# whole cascade is seeded, and the smoke-run results assume this exact revision).
# No trust_remote_code here -- Qwen is a standard architecture -- so this is a
# reproducibility pin, not a code-trust one. Applies only to the default model; a
# custom --model uses whatever revision that id resolves to. Bump deliberately.
DEFAULT_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
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


class GenerationError(RuntimeError):
    """Raised when a mid-run generation fails against an external dependency.

    Covers a degenerate/empty generation (:class:`EmptyDraftError`) and a
    runtime failure inside ``model.generate`` that the seam did not normalize to
    :class:`ModelLoadError` (torch OOM, a chat-template ``KeyError``, a
    device-side ``RuntimeError``). The CLI maps this to exit 3 (external
    dependency), so a mid-run model blowup reports a clean exit-3 diagnostic
    rather than escaping as a bare exit-1 traceback.
    """


class EmptyDraftError(GenerationError):
    """Raised when ``model.generate`` returns an empty / whitespace-only draft.

    A blank draft must never be persisted as a finished bundle: it would be
    counted, listed in the manifest, and read downstream by ``score`` as a real
    (blank) draft -- silent corruption -- and on resume the pair would read as
    done, so the blank would never be regenerated. Failing loudly instead means
    the offending pair keeps NO bundle, so a later re-run regenerates it.
    """


class CorpusDriftError(ValueError):
    """Raised when persisted bundles carry pair_ids absent from the input corpus.

    Resuming a run dir built from a different (or larger) corpus would otherwise
    silently merge stale bundles into the new manifest. Subclasses ``ValueError``
    so the CLI's existing input-error handler maps it to exit 2.
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

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        # Auto-pin the default model to its known-good revision; a custom --model
        # stays unpinned unless the caller passes an explicit revision.
        if revision is None and model_id == DEFAULT_MODEL:
            revision = DEFAULT_MODEL_REVISION
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    @property
    def device(self) -> str | None:
        """The selected torch device (``mps``/``cuda``/``cpu``), or None if unloaded.

        Surfaced so a silent MPS/CUDA->CPU fallback is visible: the generation
        driver reads this after the first :meth:`generate` and records it in the
        run summary (IMPORTANT 4).
        """
        return self._device

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
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
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
        """Generate one draft for ``prompt`` with the recorded sampling settings.

        A runtime failure inside the actual generation (torch OOM, a device-side
        ``RuntimeError``, a missing-chat-template ``KeyError``) is normalized to
        :class:`GenerationError` so the CLI maps it to exit 3 (external
        dependency) rather than letting it escape as a bare exit-1 traceback.
        Loading failures stay :class:`ModelLoadError` (also exit 3) via
        :meth:`_ensure_loaded`.
        """
        self._ensure_loaded()
        self._seed_everything(seed)

        import torch

        try:
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
            # Strip the prompt tokens; decode only the new generated continuation.
            generated = output_ids[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001 - normalize mid-run generation failure
            raise GenerationError(
                f"instruct model {self.model_id!r} failed mid-generation: {exc}"
            ) from exc


# --- Seeds -------------------------------------------------------------------


def _derive_pair_seed(base_seed: int, pair_id: str) -> int:
    """Derive a deterministic per-pair seed from ``(base_seed, pair_id)``.

    Distinct per pair (so drafts are decorrelated) yet stable (so re-generating
    one pair reproduces its original draft). A blake2b digest over
    ``{base_seed}:{pair_id}`` gives a well-distributed value; it is masked to 32
    bits so the result is a valid seed for torch/numpy (both reject seeds outside
    ``[0, 2**32)``), and ``random.seed`` accepts any int.
    """
    digest = hashlib.blake2b(
        f"{base_seed}:{pair_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


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


def _repair_torn_tail(bundles_path: Path) -> bool:
    """Terminate a torn last line before the first resume append; return if repaired.

    A crash between an append's record write and its newline write (or before an
    ``os.fsync`` completed the newline) leaves the final record with NO trailing
    newline. The next append would then concatenate its JSON onto that fragment,
    producing a single unparseable line that :func:`load_done_pair_ids` /
    :func:`_read_all_bundles` skip -- silently dropping the freshly regenerated
    pair from the bundles, manifest, and draft-texts while the run still exits 0.
    Repair the boundary by appending a lone newline so the torn record stays on
    its own (skippable) line and the regenerated record lands cleanly on the next.

    Returns ``True`` if a newline was appended (the file existed, was non-empty,
    and did not end in ``\\n``), ``False`` otherwise (missing/empty/already
    newline-terminated).
    """
    import os

    if not bundles_path.exists():
        return False
    with bundles_path.open("rb") as fh:
        try:
            fh.seek(-1, 2)  # last byte
        except OSError:
            return False  # empty file
        last = fh.read(1)
    if last in (b"", b"\n"):
        return False
    with bundles_path.open("ab") as fh:
        fh.write(b"\n")
        fh.flush()
        os.fsync(fh.fileno())
    return True


def _append_bundle(bundle: RewriteBundle, bundles_path: Path) -> None:
    """Append one RewriteBundle to bundles.jsonl, fsync'd so a crash keeps it.

    Mirrors ``corpus._append_pair`` / ``jmq``'s single-writer persistence: each
    record is written, flushed, and ``os.fsync``'d as it completes, so even a
    host crash (not just a process kill) keeps every finished draft durably on
    disk. The record and its trailing newline are written before the fsync, so a
    completed append is always a whole newline-terminated line; a crash *before*
    the fsync can still leave a torn tail, which :func:`_repair_torn_tail`
    terminates on the next resume.
    """
    import os

    bundles_path.parent.mkdir(parents=True, exist_ok=True)
    with bundles_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_dict(bundle), ensure_ascii=False))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


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
        skipped-truncated-line count, the selected model ``device`` (so a silent
        MPS/CUDA->CPU fallback is visible; ``None`` for a seam without one), and
        the written artifact paths.
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

    # Repair a torn tail (a crash between an append's record and its newline)
    # BEFORE the first resume append, so the regenerated record lands on its own
    # line instead of concatenating onto the fragment into one unparseable line
    # that would silently drop the pair (CRITICAL 1).
    if _repair_torn_tail(bundles_path):
        printer(
            "dehip generate: repaired a torn final bundle line (no trailing "
            "newline) from a prior interrupted run before resuming"
        )

    remaining = [pair for pair in pair_list if pair.pair_id not in done_ids]
    printer(
        f"dehip generate: {len(pair_list)} pairs, {len(done_ids)} already drafted, "
        f"{len(remaining)} to generate"
    )

    generated = 0
    for pair in remaining:
        pair_seed = _derive_pair_seed(seed, pair.pair_id)
        draft_text = model.generate(
            pair.prompt,
            temperature=temperature,
            top_p=top_p,
            seed=pair_seed,
        )
        # A blank draft must NOT be persisted as a finished bundle: it would be
        # counted, manifested, and read downstream as a real (blank) draft, and
        # on resume the pair would read as done. Fail loudly so the pair keeps no
        # bundle and a later re-run regenerates it (CRITICAL 2).
        if not draft_text or not draft_text.strip():
            raise EmptyDraftError(
                f"model produced an empty/whitespace-only draft for pair "
                f"{pair.pair_id!r} (seed {pair_seed}); refusing to persist it as "
                "a finished bundle. Re-run to regenerate this pair."
            )
        bundle = _make_bundle(
            run_id=run_id,
            pair=pair,
            draft_text=draft_text,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            seed=pair_seed,
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

    # Guard against manifest/bundle drift: a run dir built from a different or
    # larger corpus carries pair_ids the current corpus does not, which would
    # otherwise be silently merged into this manifest. Fail loudly (exit 2),
    # naming the stray ids (IMPORTANT 3).
    input_ids = {pair.pair_id for pair in pair_list}
    stray = sorted({b.pair_id for b in bundles} - input_ids)
    if stray:
        raise CorpusDriftError(
            "persisted bundles carry pair_ids absent from the input corpus "
            f"(stale run dir?): {stray}. Refusing to merge them into the "
            "manifest."
        )

    # Order the manifest by the input pair order for determinism.
    order = {pair.pair_id: index for index, pair in enumerate(pair_list)}
    bundles.sort(key=lambda b: order.get(b.pair_id, len(order)))

    device = getattr(model, "device", None)
    if device is not None:
        printer(f"dehip generate: model device = {device}")

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
        "device": device,
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
