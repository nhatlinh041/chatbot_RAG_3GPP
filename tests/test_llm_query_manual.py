"""
Manual test script for LocalLM-based question analysis and Cypher query generation.
This test requires Neo4j and LocalLM to be running.

Run manually with: pytest tests/test_llm_query_manual.py -v
NOT included in regular test suite (requires external services).
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment
load_dotenv()


# Skip all tests in this module unless explicitly requested
pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_LLM_TESTS') != '1',
    reason="LLM tests require RUN_LLM_TESTS=1 environment variable"
)


@pytest.fixture(scope="module")
def neo4j_driver():
    """Create Neo4j driver for tests"""
    from neo4j import GraphDatabase

    neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    yield driver
    driver.close()


@pytest.fixture(scope="module")
def cypher_generator(neo4j_driver):
    """Create CypherQueryGenerator with Neo4j driver"""
    from rag_system_v2 import CypherQueryGenerator

    local_llm_url = os.getenv('DEEPSEEK_API_URL', 'http://192.168.1.14:11434/api/chat')
    return CypherQueryGenerator(neo4j_driver=neo4j_driver, local_llm_url=local_llm_url)


class TestTermLoading:
    """Test Term node loading from KG"""

    def test_terms_loaded(self, cypher_generator):
        """Verify terms are loaded from KG"""
        assert len(cypher_generator.all_terms) > 0
        print(f"\nLoaded {len(cypher_generator.all_terms)} terms from KG")

    def test_scp_term_exists(self, cypher_generator):
        """Verify SCP term is loaded"""
        assert 'SCP' in cypher_generator.all_terms
        print(f"\nSCP: {cypher_generator.all_terms['SCP']}")

    def test_sepp_term_exists(self, cypher_generator):
        """Verify SEPP term is loaded"""
        assert 'SEPP' in cypher_generator.all_terms
        print(f"\nSEPP: {cypher_generator.all_terms['SEPP']}")


class TestRuleBasedAnalysis:
    """Test rule-based question analysis"""

    def test_comparison_question(self, cypher_generator):
        """Test comparison question detection"""
        question = "What is the different between SCP and SEPP?"
        analysis = cypher_generator.analyze_question(question, use_llm=False)

        assert analysis['question_type'] == 'comparison'
        assert len(analysis['entities']) >= 2
        print(f"\nAnalysis: {analysis}")

    def test_definition_question(self, cypher_generator):
        """Test definition question detection"""
        question = "What is AMF?"
        analysis = cypher_generator.analyze_question(question, use_llm=False)

        assert analysis['question_type'] == 'definition'
        assert len(analysis['entities']) >= 1
        print(f"\nAnalysis: {analysis}")

    def test_procedure_question(self, cypher_generator):
        """Test procedure question detection"""
        question = "How does the registration procedure work?"
        analysis = cypher_generator.analyze_question(question, use_llm=False)

        assert analysis['question_type'] == 'procedure'
        print(f"\nAnalysis: {analysis}")


class TestRuleBasedQueryGeneration:
    """Test rule-based Cypher query generation"""

    def test_comparison_query(self, cypher_generator):
        """Test comparison query generation"""
        question = "What is the different between SCP and SEPP?"
        analysis = cypher_generator.analyze_question(question, use_llm=False)
        query = cypher_generator.generate_cypher_query(question, analysis, use_llm=False)

        assert 'MATCH' in query
        assert 'RETURN' in query
        print(f"\nGenerated query (first 300 chars):\n{query[:300]}")

    def test_definition_query(self, cypher_generator):
        """Test definition query generation"""
        question = "What is AMF?"
        analysis = cypher_generator.analyze_question(question, use_llm=False)
        query = cypher_generator.generate_cypher_query(question, analysis, use_llm=False)

        assert 'MATCH' in query
        assert 'RETURN' in query
        print(f"\nGenerated query (first 300 chars):\n{query[:300]}")


class TestLLMBasedAnalysis:
    """Test LLM-based question analysis"""

    def test_llm_comparison_analysis(self, cypher_generator):
        """Test LLM-based comparison question analysis"""
        question = "What is the different between SCP and SEPP?"
        analysis = cypher_generator.analyze_question(question, use_llm=True, llm_model="deepseek-r1:7b")

        assert analysis.get('question_type') == 'comparison'
        print(f"\nLLM Analysis: {analysis}")

    def test_llm_definition_analysis(self, cypher_generator):
        """Test LLM-based definition question analysis"""
        question = "What is AMF?"
        analysis = cypher_generator.analyze_question(question, use_llm=True, llm_model="deepseek-r1:7b")

        assert analysis.get('question_type') == 'definition'
        print(f"\nLLM Analysis: {analysis}")


class TestLLMBasedQueryGeneration:
    """Test LLM-based Cypher query generation"""

    def test_llm_comparison_query(self, cypher_generator):
        """Test LLM-based comparison query generation"""
        question = "What is the different between SCP and SEPP?"
        analysis = cypher_generator.analyze_question(question, use_llm=True, llm_model="deepseek-r1:7b")
        query = cypher_generator.generate_cypher_query(question, analysis, use_llm=True, llm_model="deepseek-r1:7b")

        assert 'MATCH' in query or 'OPTIONAL' in query
        assert 'RETURN' in query
        print(f"\nLLM Generated query (first 300 chars):\n{query[:300]}")

    def test_llm_definition_query(self, cypher_generator):
        """Test LLM-based definition query generation"""
        question = "What is AMF?"
        analysis = cypher_generator.analyze_question(question, use_llm=True, llm_model="deepseek-r1:7b")
        query = cypher_generator.generate_cypher_query(question, analysis, use_llm=True, llm_model="deepseek-r1:7b")

        assert 'MATCH' in query or 'OPTIONAL' in query
        assert 'RETURN' in query
        print(f"\nLLM Generated query (first 300 chars):\n{query[:300]}")


if __name__ == "__main__":
    # Run with: RUN_LLM_TESTS=1 python -m pytest tests/test_llm_query_manual.py -v -s
    pytest.main([__file__, "-v", "-s"])
