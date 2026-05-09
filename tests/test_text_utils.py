"""Tests for text_utils.py."""

from __future__ import annotations

from obsidian_llm_wiki.text_utils import (
    canonicalize_concept,
    fingerprint_concept,
    hash_concepts,
    normalize_concept,
)


class TestNormalizeConcept:
    def test_level_1_case_folding(self):
        assert normalize_concept("Machine Learning", level=1) == "machine learning"

    def test_level_2_strips_separators(self):
        assert normalize_concept("machine-learning", level=2) == "machine learning"
        assert normalize_concept("machine_learning", level=2) == "machine learning"
        assert normalize_concept("api/gateway", level=2) == "api gateway"

    def test_level_3_removes_brackets(self):
        result = normalize_concept("Kubernetes (k8s)", level=3)
        assert "k8s" not in result
        assert "kubernetes" in result

    def test_level_3_removes_punctuation(self):
        assert normalize_concept("hello, world!", level=3) == "hello world"

    def test_level_4_folds_whitespace(self):
        assert normalize_concept("machine learning", level=4) == "machinelearning"
        assert normalize_concept("state   machine", level=4) == "statemachine"


class TestCanonicalizeConcept:
    def test_lowercases_and_strips_punctuation(self):
        result = canonicalize_concept("Machine-Learning")
        assert result == "machine learning"

    def test_removes_brackets(self):
        result = canonicalize_concept("Extreme Programming (XP)")
        assert result == "extreme programming"


class TestFingerprintConcept:
    def test_equivalent_variants_produce_same_fingerprint(self):
        variants = [
            "machine learning",
            "machine-learning",
            "machine_learning",
            "Machine Learning",
            "machine  learning",
        ]
        fingerprints = {fingerprint_concept(v) for v in variants}
        assert len(fingerprints) == 1
        assert fingerprints == {"machinelearning"}

    def test_bracket_content_removed(self):
        assert fingerprint_concept("Kubernetes (k8s)") == fingerprint_concept("kubernetes")

    def test_separators_equivalent_to_spaces(self):
        assert fingerprint_concept("api/gateway") == fingerprint_concept("api gateway")

    def test_punctuation_removed(self):
        assert fingerprint_concept("Hello, World!") == fingerprint_concept("hello world")

    def test_leading_trailing_whitespace_ignored(self):
        assert fingerprint_concept("  concept  ") == fingerprint_concept("concept")


class TestHashConcepts:
    def test_deterministic(self):
        concepts = ["Machine Learning", "Deep Learning"]
        h1 = hash_concepts(concepts)
        h2 = hash_concepts(list(reversed(concepts)))
        assert h1 == h2
        assert len(h1) == 64

    def test_different_concepts_produce_different_hash(self):
        h1 = hash_concepts(["A", "B"])
        h2 = hash_concepts(["A", "C"])
        assert h1 != h2