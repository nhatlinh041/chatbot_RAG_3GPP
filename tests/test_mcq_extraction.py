"""
Test suite for MCQ answer extraction from LLM responses.

Tests the extract_choice_from_response() method in run_tele_qna_benchmark.py
to ensure it correctly handles various response formats.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tele_qna_benchmark import TeleQnABenchmark


class TestMCQExtraction:
    """Test MCQ answer extraction logic"""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance for testing"""
        return TeleQnABenchmark(model="test-model")

    @pytest.fixture
    def sample_choices(self):
        """Sample MCQ choices"""
        return ["7", "12", "14", "28"]

    def test_answer_colon_format(self, benchmark, sample_choices):
        """Test 'Answer: C' format"""
        response = "Answer: C. 14 symbols"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_answer_colon_lowercase(self, benchmark, sample_choices):
        """Test 'answer: c' format (lowercase)"""
        response = "answer: c"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_answer_is_format(self, benchmark, sample_choices):
        """Test 'The answer is C' format"""
        response = "The answer is C"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_option_format(self, benchmark, sample_choices):
        """Test 'Option C is correct' format"""
        response = "Option C is correct"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_choice_format(self, benchmark, sample_choices):
        """Test 'Choice C' format"""
        response = "Choice C: 14 symbols per slot"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_letter_at_start_of_line(self, benchmark, sample_choices):
        """Test 'C. [explanation]' format"""
        response = "C. According to TS 38.211 Section 4.3.2"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_letter_at_start_multiline(self, benchmark, sample_choices):
        """Test letter at start of line in multiline response"""
        response = """Based on the context provided:

C. 14 symbols

According to TS 38.211, a slot consists of 14 OFDM symbols with normal cyclic prefix."""
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_answer_with_period(self, benchmark, sample_choices):
        """Test 'Answer: C.' with period"""
        response = "Answer: C."
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_answer_with_comma(self, benchmark, sample_choices):
        """Test 'Answer: C,' with comma"""
        response = "Answer: C, based on spec"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_answer_with_newline(self, benchmark, sample_choices):
        """Test 'Answer: C\\n' with newline"""
        response = "Answer: C\n\nReasoning: According to TS 38.211..."
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_text_matching_fallback(self, benchmark, sample_choices):
        """Test fallback to text matching when no letter found"""
        response = "Based on the spec, the answer is 14 symbols per slot"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (14), got {result}"

    def test_first_choice_text_wins(self, benchmark, sample_choices):
        """Test that first mentioned choice text is selected"""
        # Response mentions "12" before "14"
        response = "Extended CP has 12 symbols, but normal CP has 14 symbols"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        # Should match first occurrence (12 = index 1)
        assert result == 1, f"Expected index 1 (12), got {result}"

    def test_choice_a(self, benchmark, sample_choices):
        """Test extracting choice A"""
        response = "Answer: A"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 0, f"Expected index 0 (A), got {result}"

    def test_choice_b(self, benchmark, sample_choices):
        """Test extracting choice B"""
        response = "Answer: B"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 1, f"Expected index 1 (B), got {result}"

    def test_choice_d(self, benchmark, sample_choices):
        """Test extracting choice D"""
        response = "Answer: D"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 3, f"Expected index 3 (D), got {result}"

    def test_no_match_returns_none(self, benchmark, sample_choices):
        """Test that no match returns None"""
        response = "I don't know the answer"
        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result is None, f"Expected None, got {result}"

    def test_ambiguous_text_uses_letter(self, benchmark):
        """Test that letter pattern takes precedence over text matching"""
        # All choices contain "AMF"
        choices = ["AMF only", "AMF and SMF", "AMF, SMF, UPF", "AMF with PCF"]
        response = "Answer: C. AMF, SMF, UPF are correct"

        result = benchmark.extract_choice_from_response(response, choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_real_world_deepseek_response(self, benchmark, sample_choices):
        """Test real DeepSeek R1 response format"""
        response = """Answer: C. 14

Reasoning:
According to TS 38.211 Section 4.3.2 (OFDM symbols), a slot in New Radio (NR) consists of 14 consecutive OFDM symbols when using normal cyclic prefix configuration. This is explicitly stated in the specification and confirmed in Table 4.3.2-1.

Note: Option B (12 symbols) applies to extended cyclic prefix, not normal CP."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_new_mcq_prompt_format(self, benchmark, sample_choices):
        """Test new MCQ prompt format: 'Answer: X. [text]' at start"""
        response = """Answer: C. 14

According to TS 38.211, a slot consists of 14 OFDM symbols with normal cyclic prefix."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_new_mcq_format_with_full_text(self, benchmark):
        """Test new MCQ format with full option text"""
        choices = ["Access and mobility management", "Authentication services", "Session management", "User plane handling"]
        response = """Answer: B. Authentication services

The AUSF is responsible for authentication services in the 5G core network."""

        result = benchmark.extract_choice_from_response(response, choices)
        assert result == 1, f"Expected index 1 (B), got {result}"

    def test_new_mcq_format_with_period(self, benchmark, sample_choices):
        """Test new MCQ format with period after letter"""
        response = """Answer: C. 14

Brief explanation here."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_new_mcq_format_lowercase(self, benchmark, sample_choices):
        """Test new MCQ format lowercase"""
        response = """answer: c. 14

Explanation..."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_ausf_question_format(self, benchmark):
        """Test AUSF question with expected format"""
        choices = ["Access and mobility management", "Authentication services", "Session management", "User plane handling"]
        response = """Answer: B. Authentication services

The AUSF (Authentication Server Function) is a critical network function in the 5G Core Network (5GC) responsible for providing authentication services."""

        result = benchmark.extract_choice_from_response(response, choices)
        assert result == 1, f"Expected index 1 (B), got {result}"

    def test_real_world_claude_response(self, benchmark, sample_choices):
        """Test real Claude response format"""
        response = """Based on the 3GPP specifications provided in the context:

The answer is C. 14 symbols.

TS 38.211 Section 4.3.2 specifies that a slot consists of 14 OFDM symbols with normal cyclic prefix."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"

    def test_case_insensitive(self, benchmark, sample_choices):
        """Test case insensitivity"""
        responses = [
            "ANSWER: C",
            "answer: c",
            "Answer: C",
            "aNsWeR: c"
        ]

        for resp in responses:
            result = benchmark.extract_choice_from_response(resp, sample_choices)
            assert result == 2, f"Expected index 2 for '{resp}', got {result}"

    def test_extra_whitespace(self, benchmark, sample_choices):
        """Test handling of extra whitespace"""
        responses = [
            "Answer:  C",
            "Answer:C",
            "Answer :C",
            "Answer : C",
        ]

        for resp in responses:
            result = benchmark.extract_choice_from_response(resp, sample_choices)
            assert result == 2, f"Expected index 2 for '{resp}', got {result}"

    def test_long_explanation(self, benchmark, sample_choices):
        """Test extraction from long explanation"""
        response = """The question asks about the number of OFDM symbols in one NR slot with normal cyclic prefix.

According to 3GPP TS 38.211 Section 4.3.2, which defines the frame structure and OFDM parameters:

Answer: C. 14

This is because:
1. Normal cyclic prefix configuration uses 14 OFDM symbols per slot
2. Extended cyclic prefix configuration uses 12 OFDM symbols per slot
3. The slot duration is 0.5ms for 30kHz subcarrier spacing
4. Each symbol includes the cyclic prefix overhead

Therefore, option C (14 symbols) is correct for normal CP."""

        result = benchmark.extract_choice_from_response(response, sample_choices)
        assert result == 2, f"Expected index 2 (C), got {result}"


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def benchmark(self):
        return TeleQnABenchmark(model="test-model")

    def test_empty_response(self, benchmark):
        """Test empty response"""
        result = benchmark.extract_choice_from_response("", ["A", "B", "C"])
        assert result is None

    def test_empty_choices(self, benchmark):
        """Test empty choices list"""
        result = benchmark.extract_choice_from_response("Answer: A", [])
        assert result is None

    def test_single_choice(self, benchmark):
        """Test single choice"""
        result = benchmark.extract_choice_from_response("Answer: A", ["Only option"])
        assert result == 0

    def test_many_choices(self, benchmark):
        """Test more than 4 choices (A-F)"""
        choices = ["Opt1", "Opt2", "Opt3", "Opt4", "Opt5", "Opt6"]
        result = benchmark.extract_choice_from_response("Answer: E", choices)
        assert result == 4  # E = index 4

    def test_special_characters_in_choices(self, benchmark):
        """Test choices with special characters"""
        choices = ["5G-NR", "LTE/LTE-A", "UMTS (3G)", "GSM/2G"]
        result = benchmark.extract_choice_from_response("Answer: C", choices)
        assert result == 2


class TestMCQDetection:
    """Test MCQ intent detection in fallback analysis"""

    def test_detect_mcq_format_with_period(self):
        """Test MCQ detection with A. B. C. D. format"""
        from hybrid_retriever import SemanticQueryAnalyzer

        # Mock analyzer with minimal setup
        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()

        # Bind both methods to mock
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)

        query = """What is AUSF responsible for in 5G core network?

A. Access and mobility management
B. Authentication services
C. Session management
D. User plane handling"""

        result = analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'multiple_choice', f"Expected 'multiple_choice', got {result['primary_intent']}"

    def test_detect_mcq_format_with_paren(self):
        """Test MCQ detection with A) B) C) D) format"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)

        query = """How many symbols in NR slot?

A) 7
B) 12
C) 14
D) 28"""

        result = analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'multiple_choice', f"Expected 'multiple_choice', got {result['primary_intent']}"

    def test_non_mcq_definition(self):
        """Test that definition questions are not detected as MCQ"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)

        query = "What does AMF stand for in 5G architecture?"

        result = analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'definition', f"Expected 'definition', got {result['primary_intent']}"

    def test_detect_mcq_json_choices_format(self):
        """Test MCQ detection with JSON choices format"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        # Bind both methods
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)

        # JSON-style with choices array
        query = '''{"question": "What is AUSF responsible for?", "choices": ["Access management", "Authentication", "Session", "User plane"]}'''

        result = analyzer._detect_mcq_format(query)
        assert result is True, "Should detect JSON choices format as MCQ"

    def test_detect_mcq_inline_format(self):
        """Test MCQ detection with inline (A) (B) (C) (D) format"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)

        query = "Which function handles authentication? (A) AMF (B) AUSF (C) SMF (D) UPF"

        result = analyzer._detect_mcq_format(query)
        assert result is True, "Should detect inline format as MCQ"

    def test_detect_mcq_colon_format(self):
        """Test MCQ detection with A: B: C: D: format"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)

        query = "What is 5G? A: First generation B: Second generation C: Third generation D: Fifth generation"

        result = analyzer._detect_mcq_format(query)
        assert result is True, "Should detect colon format as MCQ"

    def test_detect_mcq_which_following_keyword(self):
        """Test MCQ detection with 'which of the following' keyword"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)

        query = "Which of the following is correct? A. Option 1 B. Option 2"

        result = analyzer._detect_mcq_format(query)
        assert result is True, "Should detect 'which of the following' pattern as MCQ"

    def test_non_mcq_simple_question(self):
        """Test that simple questions without options are not detected as MCQ"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)

        queries = [
            "What is AMF?",
            "How does registration procedure work?",
            "Compare AMF and SMF",
            "Explain the role of UPF in 5G",
        ]

        for query in queries:
            result = analyzer._detect_mcq_format(query)
            assert result is False, f"'{query}' should NOT be detected as MCQ"

    def test_detect_mcq_benchmark_format(self):
        """Test MCQ detection with benchmark question format (choices list)"""
        from hybrid_retriever import SemanticQueryAnalyzer

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {}
                import logging
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)

        # Format from tele_qna benchmark
        query = '''question: What is AUSF responsible for?
choices: ["Access and mobility management", "Authentication services", "Session management", "User plane handling"]'''

        result = analyzer._detect_mcq_format(query)
        assert result is True, "Should detect benchmark format with 'choices:' as MCQ"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
