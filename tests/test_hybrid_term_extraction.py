"""
Regression tests for hybrid term extraction module
Tests term dictionary and fuzzy matching functionality
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_term_extractor import (
    TermDictionary,
    TechnicalTerm,
    HybridTermExtractor,
    LLMQuestionAnalyzer
)


class TestTermDictionary:
    """Tests for TermDictionary class"""

    @pytest.fixture
    def term_dict(self):
        """Create a TermDictionary instance"""
        td = TermDictionary()
        td.add_term(TechnicalTerm(
            name="chf",
            spec_id="TS 32.240",
            section="5.2.1",
            synonyms=["charging function", "charging system"],
            category="charging"
        ))
        td.add_term(TechnicalTerm(
            name="amf",
            spec_id="TS 23.501",
            section="4.2.2",
            synonyms=["access and mobility management function"],
            category="core"
        ))
        return td

    def test_add_term_stores_term(self, term_dict):
        """add_term should store the term in dictionary"""
        assert "chf" in term_dict.terms
        assert "amf" in term_dict.terms

    def test_add_term_creates_synonym_map(self, term_dict):
        """add_term should create synonym mappings"""
        assert "charging function" in term_dict.synonym_map
        assert term_dict.synonym_map["charging function"] == "chf"

    def test_find_matches_exact(self, term_dict):
        """find_matches should find exact matches"""
        matches = term_dict.find_matches(["chf"])
        assert "chf" in matches

    def test_find_matches_synonym(self, term_dict):
        """find_matches should find synonym matches"""
        matches = term_dict.find_matches(["charging function"])
        assert "chf" in matches

    def test_find_matches_case_insensitive(self, term_dict):
        """find_matches should be case insensitive"""
        matches = term_dict.find_matches(["CHF", "AMF"])
        assert "chf" in matches
        assert "amf" in matches

    def test_find_matches_no_match(self, term_dict):
        """find_matches should return empty for no matches with low threshold"""
        matches = term_dict.find_matches(["xyz123"], threshold=100)
        assert len(matches) == 0


class TestHybridTermExtractor:
    """Tests for HybridTermExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create a HybridTermExtractor instance"""
        try:
            return HybridTermExtractor()
        except Exception:
            pytest.skip("HybridTermExtractor requires JSON data files")

    def test_enhance_query_returns_dict(self, extractor):
        """enhance_query should return a dictionary"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("What is the CHF?")
        assert isinstance(result, dict)
        assert 'original_question' in result
        assert 'enhanced_terms' in result

    def test_enhance_query_includes_cypher(self, extractor):
        """enhance_query should include Cypher query"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("What is the CHF?")
        assert 'cypher_query' in result
        assert 'MATCH' in result['cypher_query']

    def test_enhance_query_extracts_terms(self, extractor):
        """enhance_query should extract LLM terms"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("What is the CHF in 5G?")
        assert 'llm_terms' in result
        assert len(result['llm_terms']) > 0

    def test_generate_cypher_query_valid_structure(self, extractor):
        """Generated Cypher query should have valid structure"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("Test query")
        query = result['cypher_query']

        assert 'MATCH' in query
        assert 'WHERE' in query
        assert 'RETURN' in query
        assert 'LIMIT' in query


class TestLLMQuestionAnalyzer:
    """Tests for LLMQuestionAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create an LLMQuestionAnalyzer instance"""
        return LLMQuestionAnalyzer()

    def test_extract_terms_returns_list(self, analyzer):
        """extract_terms should return a list"""
        result = analyzer.extract_terms("What is CHF?")
        assert isinstance(result, list)

    def test_extract_terms_not_empty(self, analyzer):
        """extract_terms should return non-empty list for valid question"""
        result = analyzer.extract_terms("What is the AMF function?")
        assert len(result) > 0

    def test_extract_terms_recognizes_patterns(self, analyzer):
        """extract_terms should recognize 3GPP patterns"""
        result = analyzer.extract_terms("How does the SMF handle sessions?")
        assert any('smf' in term.lower() for term in result)


class TestTechnicalTerm:
    """Tests for TechnicalTerm dataclass"""

    def test_technical_term_creation(self):
        """TechnicalTerm should be creatable"""
        term = TechnicalTerm(
            name="amf",
            spec_id="TS 23.501",
            section="4.2.2",
            synonyms=["access management function"],
            category="core"
        )

        assert term.name == "amf"
        assert term.spec_id == "TS 23.501"
        assert len(term.synonyms) == 1

    def test_technical_term_empty_synonyms(self):
        """TechnicalTerm should work with empty synonyms"""
        term = TechnicalTerm(
            name="test",
            spec_id="TS 00.000",
            section="1.0",
            synonyms=[],
            category="test"
        )

        assert term.synonyms == []


class TestCypherQueryGeneration:
    """Regression tests for Cypher query generation"""

    @pytest.fixture
    def extractor(self):
        try:
            return HybridTermExtractor()
        except Exception:
            pytest.skip("HybridTermExtractor requires JSON data files")

    def test_query_handles_special_characters(self, extractor):
        """Query generation should handle special characters safely"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("What's the AMF's role?")
        assert 'cypher_query' in result

    def test_query_handles_empty_terms(self, extractor):
        """Query generation should handle when no terms match"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("xyz abc 123")
        assert 'cypher_query' in result

    def test_query_limits_results(self, extractor):
        """Query should include result limit"""
        if extractor is None:
            pytest.skip("Extractor not available")
        result = extractor.enhance_query("What is AMF?")
        assert 'LIMIT' in result['cypher_query']
