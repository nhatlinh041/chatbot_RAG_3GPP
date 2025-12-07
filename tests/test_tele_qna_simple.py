"""
Simple pytest integration for tele_qna benchmark
Can be run with: pytest tests/test_tele_qna_simple.py -v
"""

import json
import os
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system_v2 import create_rag_system_v2


# Load questions once for all tests
QUESTIONS_FILE = Path(__file__).parent / 'tele_qna_representative_set.json'

with open(QUESTIONS_FILE, 'r') as f:
    TELE_QNA_QUESTIONS = json.load(f)


@pytest.fixture(scope="module")
def rag_system():
    """Create RAG system for testing - uses local LLM by default"""
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    deepseek_api_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')

    rag = create_rag_system_v2(
        claude_api_key=claude_api_key,
        deepseek_api_url=deepseek_api_url
    )

    yield rag

    rag.close()


@pytest.fixture(scope="module")
def model():
    """Get model from environment or use deepseek-r1:14b as default"""
    return os.getenv('TEST_MODEL', 'deepseek-r1:14b')


class TestTeleQNABenchmark:
    """Test RAG system against tele_qna questions"""

    def test_questions_loaded(self):
        """Verify questions file is loaded correctly"""
        assert len(TELE_QNA_QUESTIONS) > 0
        assert all('question' in q for q in TELE_QNA_QUESTIONS)
        assert all('answer' in q for q in TELE_QNA_QUESTIONS)
        assert all('choices' in q for q in TELE_QNA_QUESTIONS)

    @pytest.mark.parametrize("question_data", TELE_QNA_QUESTIONS[:5])
    def test_first_five_questions(self, rag_system, model, question_data):
        """Test RAG system on first 5 questions (quick smoke test)"""
        question = question_data['question']
        expected_idx = question_data['answer']
        expected_text = question_data['choices'][expected_idx]

        # Query RAG system
        response = rag_system.query(question, model=model)

        # Basic validation
        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0
        assert len(response.sources) > 0

        # Check if answer contains expected text (loose matching)
        # This is a simple heuristic - more sophisticated evaluation could be added
        answer_lower = response.answer.lower()
        expected_lower = expected_text.lower()

        # Log for manual review
        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_text}")
        print(f"RAG Answer: {response.answer[:200]}")
        print(f"Sources: {len(response.sources)}")
        print(f"Strategy: {response.retrieval_strategy}")

        # Soft assertion - we want to know but not fail the test
        if expected_lower not in answer_lower:
            print(f"WARNING: Expected answer not found in response")


class TestTeleQNABySubject:
    """Test questions grouped by subject"""

    @pytest.fixture(scope="class")
    def questions_by_subject(self):
        """Group questions by subject"""
        by_subject = {}
        for q in TELE_QNA_QUESTIONS:
            subject = q['subject']
            if subject not in by_subject:
                by_subject[subject] = []
            by_subject[subject].append(q)
        return by_subject

    def test_lexicon_questions(self, rag_system, model, questions_by_subject):
        """Test Lexicon questions (definitions and acronyms)"""
        lexicon_questions = questions_by_subject.get('Lexicon', [])

        if not lexicon_questions:
            pytest.skip("No Lexicon questions found")

        for q in lexicon_questions[:3]:  # Test first 3
            response = rag_system.query(q['question'], model=model)
            assert response is not None
            assert len(response.sources) > 0

    def test_standards_specifications(self, rag_system, model, questions_by_subject):
        """Test Standards specifications questions"""
        spec_questions = questions_by_subject.get('Standards specifications', [])

        if not spec_questions:
            pytest.skip("No Standards specifications questions found")

        for q in spec_questions[:3]:  # Test first 3
            response = rag_system.query(q['question'], model=model)
            assert response is not None
            assert len(response.sources) > 0


class TestTeleQNAByCategory:
    """Test questions grouped by category"""

    @pytest.fixture(scope="class")
    def questions_by_category(self):
        """Group questions by category"""
        by_category = {}
        for q in TELE_QNA_QUESTIONS:
            category = q.get('category', 'Unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)
        return by_category

    def test_definition_questions(self, rag_system, model, questions_by_category):
        """Test Definition questions"""
        def_questions = [q for q in TELE_QNA_QUESTIONS
                        if 'Definition' in q.get('category', '')]

        if not def_questions:
            pytest.skip("No Definition questions found")

        for q in def_questions[:3]:  # Test first 3
            response = rag_system.query(q['question'], model=model)
            assert response is not None
            assert 'definition' in response.retrieval_strategy.lower() or \
                   'network_function' in response.retrieval_strategy.lower()

    def test_procedure_questions(self, rag_system, model, questions_by_category):
        """Test Procedure questions"""
        proc_questions = [q for q in TELE_QNA_QUESTIONS
                         if 'Procedure' in q.get('category', '')]

        if not proc_questions:
            pytest.skip("No Procedure questions found")

        for q in proc_questions[:3]:  # Test first 3
            response = rag_system.query(q['question'], model=model)
            assert response is not None
            # Procedure questions should retrieve relevant sources
            assert len(response.sources) > 0


@pytest.mark.slow
class TestTeleQNAFull:
    """Full benchmark test (marked as slow)"""

    def test_all_questions(self, rag_system, model):
        """Test all questions in the benchmark set"""
        results = {'correct': 0, 'incorrect': 0, 'failed': 0}

        for idx, q in enumerate(TELE_QNA_QUESTIONS, 1):
            try:
                response = rag_system.query(q['question'], model=model)
                expected_text = q['choices'][q['answer']].lower()

                is_correct = expected_text in response.answer.lower()

                if is_correct:
                    results['correct'] += 1
                else:
                    results['incorrect'] += 1

            except Exception as e:
                print(f"Failed on question {idx}: {e}")
                results['failed'] += 1

        # Log results
        total = len(TELE_QNA_QUESTIONS)
        accuracy = results['correct'] / total * 100 if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Full Benchmark Results:")
        print(f"  Total:      {total}")
        print(f"  Correct:    {results['correct']} ({accuracy:.1f}%)")
        print(f"  Incorrect:  {results['incorrect']}")
        print(f"  Failed:     {results['failed']}")
        print(f"{'='*60}")

        # Assert minimum accuracy threshold
        assert accuracy >= 50.0, f"Accuracy {accuracy:.1f}% below 50% threshold"
