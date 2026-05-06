"""Tests for V6 Pydantic schemas added in Phase 0.

These schemas are not yet used by any code path. These tests just
verify they instantiate correctly, serialize round-trip, and enforce
their constraints.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from obsidian_llm_wiki.models import (
    BibliographicMetadata,
    Paper,
    PaperCitation,
    PipelineVersion,
    RelationCandidate,
    SourceDocument,
    SourceSegment,
    TermRecord,
    Theorem,
)


def test_bibliographic_metadata_minimal() -> None:
    md = BibliographicMetadata(title="OSTEP")
    assert md.title == "OSTEP"
    assert md.authors == []


def test_bibliographic_metadata_full() -> None:
    md = BibliographicMetadata(
        title="Operating Systems: Three Easy Pieces",
        authors=["Remzi Arpaci-Dusseau", "Andrea Arpaci-Dusseau"],
        year=2018,
        doi="10.1234/abc",
        arxiv_id=None,
        bibtex_key="arpaci2018ostep",
    )
    assert md.year == 2018
    assert "Remzi Arpaci-Dusseau" in md.authors


def test_source_document_defaults() -> None:
    sd = SourceDocument(id="ostep")
    assert sd.source_type == "notes"
    assert sd.redistribution == "unknown"
    assert sd.bibliographic_metadata is None


def test_source_document_with_bibliographic() -> None:
    sd = SourceDocument(
        id="ostep",
        source_type="textbook",
        bibliographic_metadata=BibliographicMetadata(title="OSTEP"),
    )
    assert sd.bibliographic_metadata is not None
    assert sd.bibliographic_metadata.title == "OSTEP"


def test_source_document_invalid_source_type() -> None:
    with pytest.raises(ValidationError):
        SourceDocument(id="x", source_type="not_a_real_type")  # type: ignore[arg-type]


def test_source_segment_required_fields() -> None:
    seg = SourceSegment(
        id="ostep:p218-vector-clocks:7a3f2b1c",
        identity="ostep:p218-vector-clocks",
        ordinal=42,
        source_id="ostep",
        structural_locator="p218-vector-clocks",
        content_hash="7a3f2b1c",
        text="Vector clocks provide...",
    )
    assert seg.id != seg.identity
    assert seg.id.endswith(seg.content_hash)


def test_term_record_provenance() -> None:
    term = TermRecord(
        name="Lamport timestamp",
        definition="A scalar logical clock value.",
        aliases=["LT"],
        source_segment_id="ostep:p217:5b2d",
        provenance="extracted",
        confidence=0.92,
    )
    assert term.provenance == "extracted"
    assert 0.0 <= term.confidence <= 1.0


def test_term_record_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        TermRecord(
            name="X",
            definition="Y",
            source_segment_id="x:y:z",
            provenance="extracted",
            confidence=1.5,
        )


def test_relation_candidate_predicate_constrained() -> None:
    rel = RelationCandidate(
        subject="Vector Clocks",
        predicate="implemented_by",
        object="Causal Consistency",
        evidence="...",
        source_segment_id="ostep:p218:7a3f",
        provenance="extracted",
        confidence=0.85,
    )
    assert rel.predicate == "implemented_by"

    with pytest.raises(ValidationError):
        RelationCandidate(
            subject="A",
            predicate="not_a_real_predicate",  # type: ignore[arg-type]
            object="B",
            evidence="",
            source_segment_id="x:y:z",
            provenance="extracted",
            confidence=0.5,
        )


def test_paper_minimal() -> None:
    p = Paper(
        bibliographic=BibliographicMetadata(title="Attention Is All You Need"),
        abstract="The dominant sequence transduction models...",
    )
    assert p.bibliographic.title == "Attention Is All You Need"
    assert p.keywords == []


def test_theorem_types() -> None:
    for theorem_type in (
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "definition",
        "axiom",
    ):
        thm = Theorem(
            id=f"paper:{theorem_type}-1",
            name=f"{theorem_type.title()} 1",
            type=theorem_type,  # type: ignore[arg-type]
            statement="...",
            source_segment_id="paper:s1:abc",
        )
        assert thm.type == theorem_type


def test_paper_citation() -> None:
    citation = PaperCitation(
        citing_segment_id="paper-a:s2:xxx",
        cited_paper_id="paper-b",
        cited_title="Earlier Work",
        cited_authors=["Smith"],
        cited_year=2010,
    )
    assert citation.cited_year == 2010
    assert citation.bibtex_key is None


def test_pipeline_version_fingerprint_deterministic() -> None:
    pv1 = PipelineVersion(
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
        prompt_versions={"textbook": "v3"},
        extractor_versions={"extract_pdf": "1.2.0"},
    )
    pv2 = PipelineVersion(
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
        prompt_versions={"textbook": "v3"},
        extractor_versions={"extract_pdf": "1.2.0"},
    )
    assert pv1.fingerprint() == pv2.fingerprint()


def test_pipeline_version_fingerprint_changes_with_model() -> None:
    pv1 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="qwen2.5:14b")
    pv2 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="claude-opus-4-7")
    assert pv1.fingerprint() != pv2.fingerprint()


def test_round_trip_serialization() -> None:
    """Every new schema should serialize -> deserialize round-trip."""
    sd = SourceDocument(id="ostep", source_type="textbook")
    restored = SourceDocument.model_validate(sd.model_dump())
    assert restored == sd

    seg = SourceSegment(
        id="ostep:p1:abc",
        identity="ostep:p1",
        ordinal=0,
        source_id="ostep",
        structural_locator="p1",
        content_hash="abc",
        text="hello",
    )
    restored_seg = SourceSegment.model_validate(seg.model_dump())
    assert restored_seg == seg
