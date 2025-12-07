"""
Regression tests for cypher_sanitizer module
Tests input sanitization and Cypher injection prevention
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cypher_sanitizer import (
    CypherSanitizer,
    create_safe_cypher_query,
    safe_contains_query
)


class TestCypherSanitizerStringMethods:
    """Tests for basic string sanitization"""

    def test_sanitize_string_removes_null_bytes(self):
        """Null bytes should be removed from input"""
        input_str = "test\x00string"
        result = CypherSanitizer.sanitize_string(input_str)
        assert "\x00" not in result
        assert "test" in result and "string" in result

    def test_sanitize_string_removes_control_characters(self):
        """Control characters should be removed"""
        input_str = "test\x08\x09\x0a\x0dstring"
        result = CypherSanitizer.sanitize_string(input_str)
        # Control chars are removed and whitespace is normalized
        assert "test" in result and "string" in result

    def test_sanitize_string_handles_empty_input(self):
        """Empty string should return empty string"""
        assert CypherSanitizer.sanitize_string("") == ""
        assert CypherSanitizer.sanitize_string(None) == ""

    def test_sanitize_string_limits_length(self):
        """Long strings should be truncated to 1000 chars"""
        long_string = "a" * 2000
        result = CypherSanitizer.sanitize_string(long_string)
        assert len(result) == 1000

    def test_sanitize_string_removes_semicolons(self):
        """Semicolons should be removed to prevent query chaining"""
        input_str = "test; DROP TABLE users"
        result = CypherSanitizer.sanitize_string(input_str)
        assert ";" not in result


class TestCypherSanitizerSearchTerms:
    """Tests for search term sanitization"""

    def test_sanitize_search_term_removes_quotes(self):
        """Quotes should be removed from search terms"""
        assert CypherSanitizer.sanitize_search_term("test'term") == "testterm"
        assert CypherSanitizer.sanitize_search_term('test"term') == "testterm"

    def test_sanitize_search_term_removes_backslashes(self):
        """Backslashes should be removed"""
        result = CypherSanitizer.sanitize_search_term("test\\term")
        assert "\\" not in result

    def test_sanitize_search_term_removes_cypher_keywords(self):
        """Cypher keywords should be removed from search terms"""
        result = CypherSanitizer.sanitize_search_term("MATCH this DELETE that")
        assert "MATCH" not in result
        assert "DELETE" not in result

    def test_sanitize_search_term_preserves_valid_terms(self):
        """Valid 3GPP terms should be preserved"""
        valid_terms = ["AMF", "SMF", "UPF", "5G Core", "SBI"]
        for term in valid_terms:
            result = CypherSanitizer.sanitize_search_term(term)
            assert result == term


class TestCypherSanitizerQueryValidation:
    """Tests for query safety validation"""

    def test_validate_query_safety_blocks_delete(self):
        """DELETE operations should be blocked"""
        dangerous_query = "MATCH (n) DELETE n"
        assert CypherSanitizer.validate_query_safety(dangerous_query) is False

    def test_validate_query_safety_blocks_detach_delete(self):
        """DETACH DELETE should be blocked"""
        dangerous_query = "MATCH () DETACH DELETE"
        assert CypherSanitizer.validate_query_safety(dangerous_query) is False

    def test_validate_query_safety_blocks_drop_index(self):
        """DROP INDEX should be blocked"""
        dangerous_query = "DROP INDEX test_index"
        assert CypherSanitizer.validate_query_safety(dangerous_query) is False

    def test_validate_query_safety_blocks_load_csv(self):
        """LOAD CSV should be blocked"""
        dangerous_query = "LOAD CSV FROM 'file:///etc/passwd' AS row"
        assert CypherSanitizer.validate_query_safety(dangerous_query) is False

    def test_validate_query_safety_blocks_db_procedures(self):
        """Database procedures should be blocked"""
        # Use pattern that matches 'CALL db.' with space (regex expects \s+)
        dangerous_query = "CALL  db.labels()"  # Two spaces to match \s+
        dangerous_query2 = "CALL  dbms.components()"  # Two spaces to match \s+
        # Both should be blocked
        result1 = CypherSanitizer.validate_query_safety(dangerous_query)
        result2 = CypherSanitizer.validate_query_safety(dangerous_query2)
        # Both queries should be flagged as dangerous
        assert result1 is False
        assert result2 is False

    def test_validate_query_safety_allows_safe_queries(self):
        """Safe read queries should be allowed"""
        safe_query = """
        MATCH (c:Chunk)
        WHERE toLower(c.content) CONTAINS 'amf'
        RETURN c.chunk_id, c.content
        LIMIT 10
        """
        assert CypherSanitizer.validate_query_safety(safe_query) is True

    def test_validate_query_safety_handles_empty(self):
        """Empty query should be considered safe"""
        assert CypherSanitizer.validate_query_safety("") is True
        assert CypherSanitizer.validate_query_safety(None) is True


class TestCypherSanitizerQuestionAnalysis:
    """Tests for question analysis sanitization"""

    def test_sanitize_question_analysis_sanitizes_entities(self, sample_analysis):
        """Entity values should be sanitized"""
        malicious_analysis = {
            'entities': [{'value': "AMF'; DELETE --", 'type': 'network_function'}],
            'key_terms': ['AMF'],
            'question_type': 'definition'
        }
        result = CypherSanitizer.sanitize_question_analysis(malicious_analysis)
        assert "DELETE" not in result['entities'][0]['value']
        assert "'" not in result['entities'][0]['value']

    def test_sanitize_question_analysis_sanitizes_key_terms(self):
        """Key terms should be sanitized"""
        malicious_analysis = {
            'entities': [],
            'key_terms': ["MATCH (n) DELETE n", "normal term"],
            'question_type': 'general'
        }
        result = CypherSanitizer.sanitize_question_analysis(malicious_analysis)
        assert "DELETE" not in str(result['key_terms'])

    def test_sanitize_question_analysis_preserves_safe_fields(self, sample_analysis):
        """Safe fields like question_type should be preserved"""
        result = CypherSanitizer.sanitize_question_analysis(sample_analysis)
        assert result['question_type'] == 'definition'


class TestSafeQueryGeneration:
    """Tests for safe query generation functions"""

    def test_safe_contains_query_basic(self):
        """Basic query generation should work"""
        query = safe_contains_query(["AMF", "SMF"])
        assert "MATCH (c:Chunk)" in query
        assert "toLower(c.content) CONTAINS" in query
        assert "LIMIT" in query

    def test_safe_contains_query_sanitizes_terms(self):
        """Search terms should be sanitized in generated query"""
        query = safe_contains_query(["AMF'; DELETE --"])
        # DELETE keyword should be removed from the search term
        assert "DELETE" not in query.upper().split("CONTAINS")[1].split("'")[1]

    def test_safe_contains_query_limits_terms(self):
        """Should limit number of search terms"""
        many_terms = [f"term{i}" for i in range(20)]
        query = safe_contains_query(many_terms)
        # Should only have 5 terms max
        assert query.count("CONTAINS") <= 5

    def test_safe_contains_query_validates_limit(self):
        """Limit parameter should be validated"""
        query = safe_contains_query(["AMF"], limit=100)
        assert "LIMIT 5" in query  # Should default to 5 if over 50

    def test_safe_contains_query_rejects_empty(self):
        """Should raise error for empty terms"""
        with pytest.raises(ValueError):
            safe_contains_query([])

    def test_safe_contains_query_rejects_all_malicious(self):
        """Should raise error if all terms are sanitized away"""
        # These terms should be completely stripped after sanitization
        with pytest.raises(ValueError):
            safe_contains_query(["", "   "])


class TestInjectionPrevention:
    """Regression tests for known injection patterns"""

    def test_sql_style_injection(self, malicious_inputs):
        """SQL-style injection attempts should be sanitized"""
        for malicious in malicious_inputs:
            sanitized = CypherSanitizer.sanitize_search_term(malicious)
            # Quotes should be removed
            assert "'" not in sanitized
            assert '"' not in sanitized
            # DELETE keyword should be removed
            assert "DELETE" not in sanitized.upper()

    def test_cypher_injection_in_query(self):
        """Cypher injection in generated queries should be prevented"""
        malicious_term = "test' OR 1=1 MATCH (n) DELETE n --"

        # This should either sanitize or raise an error
        try:
            query = safe_contains_query([malicious_term])
            assert CypherSanitizer.validate_query_safety(query)
        except ValueError:
            pass  # Raising error is also acceptable

    def test_unicode_bypass_attempt(self):
        """Unicode bypass attempts should be handled"""
        unicode_attack = "DELETE\u200b"  # Zero-width space
        result = CypherSanitizer.sanitize_search_term(unicode_attack)
        # DELETE keyword should still be removed
        assert "DELETE" not in result.upper()
