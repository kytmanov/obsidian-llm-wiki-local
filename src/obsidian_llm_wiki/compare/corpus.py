"""
Corpus + queries loader and mode detector for olw compare.

Three input modes:
  curated       — directory with corpus.toml (+ optional queries.toml);
                  full ground-truth metrics
  byo           — directory of raw .md files, no ground truth
  baseline_vault — user's existing vault; compare runs on its raw/ notes
                   while preserving the vault read-only for provenance

Mode is chosen at runner-start time by inspecting the inputs; this
module exposes the detection + validation primitives so the CLI can
make the decision explicitly.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class CorpusMode(StrEnum):
    CURATED = "curated"
    BYO = "byo"
    BASELINE = "baseline"


class CorpusError(Exception):
    """Raised when corpus input is malformed or unusable."""


@dataclass(frozen=True)
class Note:
    file: str
    path: Path  # absolute
    category: str = "uncategorized"
    weight: float = 1.0
    known_concepts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    expected_pages: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_refusal: bool = False


@dataclass
class Corpus:
    mode: CorpusMode
    version: str
    language: str
    description: str
    notes: list[Note]
    queries: list[Query]
    notes_set_hash: str  # SHA256 of sorted (filename, content) pairs
    root: Path

    @property
    def has_queries(self) -> bool:
        return len(self.queries) > 0

    @property
    def has_ground_truth_concepts(self) -> bool:
        """True if at least one note has non-empty known_concepts."""
        return any(n.known_concepts for n in self.notes)


# ── Mode detection ────────────────────────────────────────────────────────────


def detect_mode(
    corpus_path: Path | None,
    notes_path: Path | None,
    baseline_vault_path: Path | None,
) -> CorpusMode:
    """Choose input mode from CLI-supplied paths.

    Caller must ensure at most one of (corpus_path, notes_path,
    baseline_vault_path) is set; no-arg case (built-in corpus) resolves
    to CURATED as well.
    """
    provided = [p for p in (corpus_path, notes_path, baseline_vault_path) if p is not None]
    if len(provided) > 1:
        raise CorpusError("Pass at most one of --corpus / --notes / --baseline-vault")
    if baseline_vault_path is not None:
        return CorpusMode.BASELINE
    if notes_path is not None:
        return CorpusMode.BYO
    return CorpusMode.CURATED


# ── Hashing ───────────────────────────────────────────────────────────────────


def notes_set_hash(notes: list[Note]) -> str:
    """SHA256 of sorted (filename, content) pairs.

    Embedded in report.json so re-running on the same-but-renamed
    notes produces a mismatch the user can see.
    """
    h = hashlib.sha256()
    for n in sorted(notes, key=lambda n: n.file):
        h.update(n.file.encode("utf-8"))
        h.update(b"\x00")
        h.update(n.path.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_corpus(
    corpus_path: Path | None = None,
    notes_path: Path | None = None,
    baseline_vault_path: Path | None = None,
    queries_path: Path | None = None,
    sample_n: int | None = None,
) -> Corpus:
    """Dispatch to the right loader for the chosen mode.

    The runner calls this once at start; the returned Corpus is the
    canonical input spec for that run.
    """
    mode = detect_mode(corpus_path, notes_path, baseline_vault_path)

    if mode == CorpusMode.CURATED:
        root = corpus_path or (Path(__file__).resolve().parents[3] / "tests" / "compare_corpus")
        if not root.exists():
            raise CorpusError(f"Curated corpus not found at: {root}")
        return _load_curated(root, sample_n=sample_n)

    if mode == CorpusMode.BYO:
        assert notes_path is not None
        return _load_byo(notes_path, queries_path=queries_path, sample_n=sample_n)

    assert baseline_vault_path is not None
    return _load_baseline(baseline_vault_path, sample_n=sample_n)


def _load_curated(root: Path, sample_n: int | None) -> Corpus:
    corpus_toml = root / "corpus.toml"
    if not corpus_toml.exists():
        raise CorpusError(f"Missing corpus.toml in {root}")
    data = tomllib.loads(corpus_toml.read_text())

    version = data.get("version", "1.0")
    language = data.get("language", "en")
    description = data.get("description", "")

    notes_dir = root / "notes"
    if not notes_dir.is_dir():
        raise CorpusError(f"Missing notes/ dir in {root}")

    notes: list[Note] = []
    for entry in data.get("note", []):
        fname = entry.get("file")
        if not fname:
            raise CorpusError(f"Corpus entry missing 'file': {entry}")
        fpath = notes_dir / fname
        if not fpath.exists():
            raise CorpusError(f"Corpus references missing note file: {fpath}")
        notes.append(
            Note(
                file=fname,
                path=fpath.resolve(),
                category=entry.get("category", "uncategorized"),
                weight=float(entry.get("weight", 1.0)),
                known_concepts=[c.lower() for c in entry.get("known_concepts", [])],
            )
        )

    if not notes:
        raise CorpusError("Curated corpus has zero notes")

    if sample_n is not None and sample_n > 0:
        notes = notes[:sample_n]

    queries_toml = root / "queries.toml"
    queries = _load_queries(queries_toml) if queries_toml.exists() else []

    return Corpus(
        mode=CorpusMode.CURATED,
        version=version,
        language=language,
        description=description,
        notes=notes,
        queries=queries,
        notes_set_hash=notes_set_hash(notes),
        root=root,
    )


def _load_byo(notes_path: Path, queries_path: Path | None, sample_n: int | None) -> Corpus:
    if not notes_path.is_dir():
        raise CorpusError(f"--notes must be a directory: {notes_path}")

    md_files = sorted(p for p in notes_path.rglob("*.md") if not p.name.startswith("."))
    if len(md_files) < 3:
        raise CorpusError(
            f"BYO mode requires at least 3 .md notes in {notes_path}; found {len(md_files)}. "
            "Add more notes or run without --notes to use the built-in corpus."
        )

    if sample_n is not None and sample_n > 0:
        md_files = md_files[:sample_n]

    notes = [
        Note(
            file=str(p.relative_to(notes_path)),
            path=p.resolve(),
            category="byo",
            weight=1.0,
            known_concepts=[],
        )
        for p in md_files
    ]

    queries = _load_queries(queries_path) if queries_path and queries_path.exists() else []

    return Corpus(
        mode=CorpusMode.BYO,
        version="byo",
        language="",  # auto-detect downstream
        description=f"User-supplied notes from {notes_path}",
        notes=notes,
        queries=queries,
        notes_set_hash=notes_set_hash(notes),
        root=notes_path,
    )


def _load_baseline(vault_path: Path, sample_n: int | None) -> Corpus:
    """Load user's existing vault as a corpus.

    Only the raw/ directory is treated as notes. The full vault is hashed
    for provenance in the report, but no current wiki/state agreement
    metrics are computed yet.
    """
    if not vault_path.is_dir():
        raise CorpusError(f"--baseline-vault must be a directory: {vault_path}")
    raw_dir = vault_path / "raw"
    if not raw_dir.is_dir():
        raise CorpusError(
            f"Baseline vault missing raw/ dir: {vault_path}. "
            "Point --baseline-vault at a mature olw vault."
        )
    if not (vault_path / "wiki.toml").exists():
        raise CorpusError(
            f"Baseline vault missing wiki.toml: {vault_path}. "
            "Point --baseline-vault at an existing olw vault."
        )

    md_files = sorted(p for p in raw_dir.rglob("*.md") if not p.name.startswith("."))
    if len(md_files) < 3:
        raise CorpusError(
            f"Baseline vault's raw/ has only {len(md_files)} note(s); "
            "need at least 3 for meaningful compare."
        )

    if sample_n is not None and sample_n > 0:
        md_files = md_files[:sample_n]

    notes = [
        Note(
            file=str(p.relative_to(raw_dir)),
            path=p.resolve(),
            category="baseline",
            weight=1.0,
            known_concepts=[],
        )
        for p in md_files
    ]

    return Corpus(
        mode=CorpusMode.BASELINE,
        version="baseline",
        language="",
        description=f"Baseline vault at {vault_path}",
        notes=notes,
        queries=[],
        notes_set_hash=notes_set_hash(notes),
        root=vault_path,
    )


def _load_queries(queries_toml: Path) -> list[Query]:
    if not queries_toml.exists():
        return []
    data = tomllib.loads(queries_toml.read_text())
    out: list[Query] = []
    seen_ids: set[str] = set()
    for entry in data.get("query", []):
        qid = entry.get("id")
        if not qid:
            raise CorpusError(f"Query missing id: {entry}")
        if qid in seen_ids:
            raise CorpusError(f"Duplicate query id: {qid}")
        seen_ids.add(qid)
        question = entry.get("question")
        if not question:
            raise CorpusError(f"Query {qid} missing 'question'")
        out.append(
            Query(
                id=qid,
                question=question,
                expected_pages=list(entry.get("expected_pages", [])),
                expected_contains=list(entry.get("expected_contains", [])),
                expected_refusal=bool(entry.get("expected_refusal", False)),
            )
        )
    return out
