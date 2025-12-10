"""
Integration tests for MCQ (Multiple Choice Question) flow with actual LLM.

Tests the full pipeline:
1. MCQ intent detection
2. MCQ prompt template generation
3. LLM response generation
4. Answer extraction from LLM response

Usage:
    # Run all tests (requires LLM server running)
    pytest tests/test_mcq_llm_integration.py -v

    # Run specific test
    pytest tests/test_mcq_llm_integration.py::TestMCQLLMIntegration::test_ausf_mcq -v

    # Skip if LLM not available
    pytest tests/test_mcq_llm_integration.py -v -m "not llm_required"

Requirements:
    - Local LLM server running at http://192.168.1.14:11434
    - Neo4j database running with vector index
"""

import pytest
import os
import sys
import re
import requests
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_retriever import SemanticQueryAnalyzer
from prompt_templates import PromptTemplates
from run_tele_qna_benchmark import TeleQnABenchmark


def is_llm_available(url: str = "http://192.168.1.14:11434") -> bool:
    """Check if LLM server is available"""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


# Mark for tests requiring LLM
llm_required = pytest.mark.skipif(
    not is_llm_available(),
    reason="LLM server not available at http://192.168.1.14:11434"
)


class TestMCQIntentDetection:
    """Test MCQ intent detection without LLM"""

    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer for testing fallback analysis"""
        import logging

        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {
                    'AMF': {'full_name': 'Access and Mobility Management Function'},
                    'AUSF': {'full_name': 'Authentication Server Function'},
                    'SMF': {'full_name': 'Session Management Function'},
                }
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        # Bind both methods
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)
        return analyzer

    def test_detect_ausf_mcq(self, mock_analyzer):
        """Test AUSF MCQ question detection"""
        query = """What is AUSF responsible for in 5G core network?

A. Access and mobility management
B. Authentication services
C. Session management
D. User plane handling"""

        result = mock_analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'multiple_choice', \
            f"Expected 'multiple_choice', got '{result['primary_intent']}'"

    def test_detect_nr_slot_mcq(self, mock_analyzer):
        """Test NR slot MCQ question detection"""
        query = """How many symbols are in one NR slot with normal cyclic prefix?

A. 7
B. 12
C. 14
D. 28"""

        result = mock_analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'multiple_choice', \
            f"Expected 'multiple_choice', got '{result['primary_intent']}'"

    def test_detect_mcq_with_parentheses(self, mock_analyzer):
        """Test MCQ with A) B) C) D) format"""
        query = """Which network function handles authentication?

A) AMF
B) AUSF
C) SMF
D) UPF"""

        result = mock_analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'multiple_choice', \
            f"Expected 'multiple_choice', got '{result['primary_intent']}'"

    def test_non_mcq_definition(self, mock_analyzer):
        """Test that plain definition is NOT detected as MCQ"""
        query = "What does AMF stand for in 5G architecture?"

        result = mock_analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'definition', \
            f"Expected 'definition', got '{result['primary_intent']}'"

    def test_non_mcq_comparison(self, mock_analyzer):
        """Test that comparison is NOT detected as MCQ"""
        query = "Compare AMF and SMF in 5G core network"

        result = mock_analyzer._fallback_analysis(query)
        assert result['primary_intent'] == 'comparison', \
            f"Expected 'comparison', got '{result['primary_intent']}'"


class TestMCQPromptTemplate:
    """Test MCQ prompt template generation"""

    def test_mcq_prompt_has_answer_format(self):
        """Test that MCQ prompt includes Answer: format instruction"""
        query = """What is AUSF responsible for?

A. Access and mobility management
B. Authentication services
C. Session management
D. User plane handling"""

        context = "The AUSF handles authentication in 5G core network."

        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        assert "Answer:" in prompt, "Prompt should include 'Answer:' format"
        assert "FIRST LINE MUST BE" in prompt, "Prompt should enforce first line format"
        assert "EXACT TEXT OF THE CHOSEN OPTION" in prompt, "Prompt should require option text"

    def test_mcq_prompt_extracts_options(self):
        """Test that MCQ prompt extracts and displays options"""
        query = """Test question?

A. Option one
B. Option two
C. Option three
D. Option four"""

        context = "Test context"

        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        assert "Available Options" in prompt, "Prompt should show available options"
        assert "A. Option one" in prompt, "Option A should be in prompt"
        assert "D. Option four" in prompt, "Option D should be in prompt"

    def test_mcq_prompt_uses_dynamic_example(self):
        """Test that MCQ prompt uses option from question as example"""
        query = """What is correct?

A. First choice
B. Second choice
C. Third choice
D. Fourth choice"""

        context = "Context here"

        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        # Example should use option C (index 2)
        assert "C. Third choice" in prompt, "Example should use option C"


class TestMCQAnswerExtraction:
    """Test answer extraction from various LLM response formats"""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance for extraction testing"""
        return TeleQnABenchmark(model="test-model")

    @pytest.fixture
    def ausf_choices(self):
        return [
            "Access and mobility management",
            "Authentication services",
            "Session management",
            "User plane handling"
        ]

    def test_extract_new_format_with_text(self, benchmark, ausf_choices):
        """Test extraction: 'Answer: B. Authentication services'"""
        response = """Answer: B. Authentication services

The AUSF is responsible for authentication services in 5G core."""

        result = benchmark.extract_choice_from_response(response, ausf_choices)
        assert result == 1, f"Expected 1 (B), got {result}"

    def test_extract_letter_only(self, benchmark, ausf_choices):
        """Test extraction: 'Answer: B' (backward compatible)"""
        response = """Answer: B

Authentication services."""

        result = benchmark.extract_choice_from_response(response, ausf_choices)
        assert result == 1, f"Expected 1 (B), got {result}"

    def test_extract_from_verbose_response(self, benchmark, ausf_choices):
        """Test extraction from verbose response mentioning the answer"""
        response = """Based on the context, the answer is B. Authentication services.

The AUSF (Authentication Server Function) handles authentication."""

        result = benchmark.extract_choice_from_response(response, ausf_choices)
        assert result == 1, f"Expected 1 (B), got {result}"

    def test_extract_with_lowercase(self, benchmark, ausf_choices):
        """Test extraction with lowercase"""
        response = """answer: b. authentication services

Explanation here."""

        result = benchmark.extract_choice_from_response(response, ausf_choices)
        assert result == 1, f"Expected 1 (B), got {result}"


@llm_required
class TestMCQLLMIntegration:
    """Integration tests with actual LLM - requires LLM server running"""

    @pytest.fixture
    def llm_url(self):
        return os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')

    @pytest.fixture
    def model(self):
        return "deepseek-r1:14b"

    @pytest.fixture
    def benchmark(self):
        return TeleQnABenchmark(model="deepseek-r1:14b")

    def call_llm(self, prompt: str, model: str, llm_url: str) -> str:
        """Call local LLM and get response"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.1
        }

        response = requests.post(llm_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def test_ausf_mcq_llm_response(self, llm_url, model, benchmark):
        """Test AUSF MCQ with actual LLM response"""
        query = """What is AUSF responsible for in 5G core network?

A. Access and mobility management
B. Authentication services
C. Session management
D. User plane handling"""

        context = """The AUSF (Authentication Server Function) is the network entity in the 5G Core network
supporting authentication functionalities. According to TS 29.509, the AUSF authenticates the UE
for the requesting NF, provides keying material, and protects steering information lists."""

        # Generate prompt using MCQ template
        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        # Call LLM
        response = self.call_llm(prompt, model, llm_url)
        print(f"\n=== LLM Response ===\n{response}\n===================")

        # Extract answer
        choices = [
            "Access and mobility management",
            "Authentication services",
            "Session management",
            "User plane handling"
        ]
        extracted = benchmark.extract_choice_from_response(response, choices)

        # Verify
        assert extracted is not None, f"Failed to extract answer from: {response[:200]}"
        assert extracted == 1, f"Expected B (index 1), got {chr(65 + extracted) if extracted else 'None'}"

        # Check format compliance
        assert response.strip().startswith("Answer:"), \
            f"Response should start with 'Answer:', got: {response[:50]}"

    def test_nr_slot_mcq_llm_response(self, llm_url, model, benchmark):
        """Test NR slot MCQ with actual LLM response"""
        query = """How many symbols are in one NR slot with normal cyclic prefix?

A. 7
B. 12
C. 14
D. 28"""

        context = """According to TS 38.211 Section 4.3.2 (OFDM symbols), a slot consists of
14 consecutive OFDM symbols with normal cyclic prefix and 12 symbols with extended cyclic prefix.
Table 4.3.2-1 defines: Normal CP = 14 symbols per slot, Extended CP = 12 symbols per slot."""

        # Generate prompt
        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        # Call LLM
        response = self.call_llm(prompt, model, llm_url)
        print(f"\n=== LLM Response ===\n{response}\n===================")

        # Extract answer
        choices = ["7", "12", "14", "28"]
        extracted = benchmark.extract_choice_from_response(response, choices)

        # Verify
        assert extracted is not None, f"Failed to extract answer from: {response[:200]}"
        assert extracted == 2, f"Expected C (index 2), got {chr(65 + extracted) if extracted else 'None'}"

    def test_amf_definition_mcq_llm_response(self, llm_url, model, benchmark):
        """Test AMF definition MCQ with actual LLM response"""
        query = """What does AMF stand for in 5G architecture?

A. Access Management Function
B. Authentication and Mobility Function
C. Access and Mobility Management Function
D. Application Management Function"""

        context = """The AMF (Access and Mobility Management Function) is a key network function
in the 5G Core (5GC) architecture. It handles access and mobility management, including
registration, connection management, reachability management, and mobility management."""

        # Generate prompt
        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)

        # Call LLM
        response = self.call_llm(prompt, model, llm_url)
        print(f"\n=== LLM Response ===\n{response}\n===================")

        # Extract answer
        choices = [
            "Access Management Function",
            "Authentication and Mobility Function",
            "Access and Mobility Management Function",
            "Application Management Function"
        ]
        extracted = benchmark.extract_choice_from_response(response, choices)

        # Verify
        assert extracted is not None, f"Failed to extract answer from: {response[:200]}"
        assert extracted == 2, f"Expected C (index 2), got {chr(65 + extracted) if extracted else 'None'}"

    def test_llm_response_format_compliance(self, llm_url, model):
        """Test that LLM response follows the required format"""
        query = """Which protocol is used for N2 interface?

A. HTTP/2
B. NGAP
C. PFCP
D. Diameter"""

        context = """The N2 interface connects the AMF to the gNB. It uses NGAP (NG Application Protocol)
for control plane signaling. NGAP is defined in TS 38.413."""

        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)
        response = self.call_llm(prompt, model, llm_url)

        print(f"\n=== Format Compliance Test ===")
        print(f"Response:\n{response}")
        print(f"==============================")

        # Check format requirements
        errors = []

        # 1. Should start with "Answer:"
        if not response.strip().startswith("Answer:"):
            errors.append(f"Response doesn't start with 'Answer:', starts with: {response[:30]}")

        # 2. First line should have format "Answer: X. text" or "Answer: X"
        first_line = response.strip().split('\n')[0]
        answer_match = re.match(r'^Answer:\s*([A-Da-d])[\.\s]', first_line, re.IGNORECASE)
        if not answer_match:
            errors.append(f"First line doesn't match 'Answer: X.' format: {first_line}")

        # 3. Should NOT have markdown headers before answer
        if response.strip().startswith("**") or response.strip().startswith("##"):
            errors.append("Response starts with markdown formatting")

        # Report all errors
        if errors:
            pytest.fail("\n".join(errors))


class TestMCQEndToEnd:
    """End-to-end tests for MCQ flow (without actual LLM)"""

    def test_full_mcq_flow_mock(self):
        """Test full MCQ flow with mocked LLM response"""
        import logging

        # 1. Query
        query = """What is AUSF responsible for?

A. Access and mobility management
B. Authentication services
C. Session management
D. User plane handling"""

        # 2. Intent detection
        class MockAnalyzer:
            def __init__(self):
                self.term_dict = {'AUSF': {'full_name': 'Authentication Server Function'}}
                self.logger = logging.getLogger('test')

        analyzer = MockAnalyzer()
        analyzer._detect_mcq_format = lambda q: SemanticQueryAnalyzer._detect_mcq_format(analyzer, q)
        analyzer._fallback_analysis = lambda q: SemanticQueryAnalyzer._fallback_analysis(analyzer, q)

        analysis = analyzer._fallback_analysis(query)
        assert analysis['primary_intent'] == 'multiple_choice', "Should detect MCQ"

        # 3. Prompt generation
        context = "AUSF handles authentication in 5G."
        prompt = PromptTemplates.get_multiple_choice_prompt(query, context)
        assert "Answer:" in prompt, "Prompt should have Answer format"

        # 4. Mock LLM response (simulating correct format)
        mock_response = """Answer: B. Authentication services

The AUSF is responsible for authentication services in the 5G core network."""

        # 5. Extraction
        benchmark = TeleQnABenchmark(model="test")
        choices = [
            "Access and mobility management",
            "Authentication services",
            "Session management",
            "User plane handling"
        ]
        extracted = benchmark.extract_choice_from_response(mock_response, choices)

        assert extracted == 1, f"Expected B (index 1), got {extracted}"

        # 6. Verify correctness (expected answer is B = index 1)
        expected_index = 1
        assert extracted == expected_index, "Answer should be correct"

        print("✅ Full MCQ flow test passed!")


if __name__ == '__main__':
    # Run with verbose output
    pytest.main([__file__, '-v', '-s'])
