"""
Regression tests for rag_system_v2 module
Tests question analysis, query generation, and retrieval components
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCypherQueryGenerator:
    """Tests for CypherQueryGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create a CypherQueryGenerator instance"""
        from rag_system_v2 import CypherQueryGenerator
        return CypherQueryGenerator()

    def test_analyze_question_returns_dict(self, generator, sample_question):
        """analyze_question should return a dictionary"""
        result = generator.analyze_question(sample_question)
        assert isinstance(result, dict)
        assert 'question_type' in result
        assert 'entities' in result
        assert 'key_terms' in result

    def test_analyze_question_definition_type(self, generator):
        """Questions with 'what is' should be classified as definition"""
        question = "What is the AMF in 5G?"
        result = generator.analyze_question(question)
        assert result['question_type'] == 'definition'

    def test_analyze_question_comparison_type(self, generator):
        """Questions with comparison words should be classified as comparison"""
        # 'compare' is the keyword that triggers comparison type
        question = "Compare the AMF and SMF functions"
        result = generator.analyze_question(question)
        assert result['question_type'] == 'comparison'

    def test_analyze_question_procedure_type(self, generator):
        """Questions with 'how' should be classified as procedure"""
        question = "How does the registration procedure work?"
        result = generator.analyze_question(question)
        assert result['question_type'] == 'procedure'

    def test_analyze_question_extracts_entities(self, generator):
        """analyze_question should extract network function entities"""
        question = "What is the role of AMF in 5G Core?"
        result = generator.analyze_question(question)

        entity_values = [e['value'] for e in result['entities']]
        assert 'AMF' in entity_values

    def test_analyze_question_extracts_key_terms(self, generator, sample_question):
        """analyze_question should extract key terms"""
        result = generator.analyze_question(sample_question)
        assert len(result['key_terms']) > 0

    def test_generate_cypher_query_returns_string(self, generator, sample_question, sample_analysis):
        """generate_cypher_query should return a Cypher query string"""
        query = generator.generate_cypher_query(sample_question, sample_analysis)
        assert isinstance(query, str)
        assert 'MATCH' in query
        assert 'RETURN' in query

    def test_generate_cypher_query_includes_limit(self, generator, sample_question, sample_analysis):
        """Generated queries should include LIMIT clause"""
        query = generator.generate_cypher_query(sample_question, sample_analysis)
        assert 'LIMIT' in query

    def test_generate_definition_query(self, generator):
        """Definition queries should search in definition sections"""
        question = "What is AMF?"
        analysis = {
            'question_type': 'definition',
            'entities': [{'value': 'AMF', 'type': 'network_function'}],
            'key_terms': ['AMF']
        }
        query = generator.generate_cypher_query(question, analysis)
        assert 'definition' in query.lower() or 'overview' in query.lower()

    def test_generate_comparison_query(self, generator):
        """Comparison queries should include both entities"""
        question = "Compare AMF and SMF"
        analysis = {
            'question_type': 'comparison',
            'entities': [
                {'value': 'AMF', 'type': 'network_function'},
                {'value': 'SMF', 'type': 'network_function'}
            ],
            'key_terms': ['AMF', 'SMF']
        }
        query = generator.generate_cypher_query(question, analysis)
        assert 'amf' in query.lower()
        assert 'smf' in query.lower()


class TestEnhancedKnowledgeRetriever:
    """Tests for EnhancedKnowledgeRetriever class"""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever with patched Neo4j"""
        with patch('rag_system_v2.GraphDatabase') as mock_db:
            mock_driver = Mock()
            mock_db.driver.return_value = mock_driver

            from rag_system_v2 import EnhancedKnowledgeRetriever
            retriever = EnhancedKnowledgeRetriever(
                "neo4j://localhost:7687",
                "neo4j",
                "password"
            )
            yield retriever

    def test_retriever_initialization(self, mock_retriever):
        """Retriever should initialize with Neo4j connection"""
        assert mock_retriever is not None
        assert mock_retriever.driver is not None

    def test_retriever_has_cypher_generator(self, mock_retriever):
        """Retriever should have a CypherQueryGenerator"""
        assert hasattr(mock_retriever, 'cypher_generator')
        assert mock_retriever.cypher_generator is not None


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass"""

    def test_retrieved_chunk_creation(self):
        """RetrievedChunk should be creatable with required fields"""
        from rag_system_v2 import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id="test_123",
            spec_id="TS_23.501",
            section_id="4.2.1",
            section_title="AMF Overview",
            content="Test content",
            chunk_type="definition",
            complexity_score=0.5,
            key_terms=["AMF"],
            reference_path=[]
        )

        assert chunk.chunk_id == "test_123"
        assert chunk.spec_id == "TS_23.501"
        assert chunk.content == "Test content"

    def test_retrieved_chunk_defaults(self):
        """RetrievedChunk should have sensible defaults"""
        from rag_system_v2 import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id="test",
            spec_id="test",
            section_id="test",
            section_title="test",
            content="test",
            chunk_type="test",
            complexity_score=0.0,
            key_terms=[],
            reference_path=[]
        )

        assert chunk.key_terms == []
        assert chunk.reference_path == []


class TestRAGResponse:
    """Tests for RAGResponse dataclass"""

    def test_rag_response_creation(self):
        """RAGResponse should be creatable with required fields"""
        from rag_system_v2 import RAGResponse, RetrievedChunk
        from datetime import datetime

        chunk = RetrievedChunk(
            chunk_id="test",
            spec_id="test",
            section_id="test",
            section_title="test",
            content="test",
            chunk_type="test",
            complexity_score=0.0,
            key_terms=[],
            reference_path=[]
        )

        response = RAGResponse(
            answer="Test answer",
            sources=[chunk],
            query="What is AMF?",
            cypher_query="MATCH (c:Chunk) RETURN c",
            retrieval_strategy="definition",
            timestamp=datetime.now()
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.query == "What is AMF?"


class TestLLMIntegrator:
    """Tests for unified LLMIntegrator class"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('rag_system_v2.anthropic') as mock:
            mock_client = Mock()
            mock.Anthropic.return_value = mock_client
            yield mock

    def test_integrator_initialization_with_claude(self, mock_anthropic):
        """Integrator should initialize with Claude API key"""
        from rag_system_v2 import LLMIntegrator

        integrator = LLMIntegrator(claude_api_key="test_api_key")
        assert integrator.claude_client is not None

    def test_integrator_initialization_with_local_llm(self):
        """Integrator should initialize with local LLM URL"""
        from rag_system_v2 import LLMIntegrator

        integrator = LLMIntegrator(local_llm_url="http://localhost:11434/api/chat")
        assert integrator.local_llm_url == "http://localhost:11434/api/chat"

    def test_generate_answer_with_claude(self, mock_anthropic):
        """generate_answer should work with Claude model"""
        from rag_system_v2 import LLMIntegrator, RetrievedChunk

        mock_message = Mock()
        mock_message.content = [Mock(text="Test response")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

        integrator = LLMIntegrator(claude_api_key="test_api_key")
        chunk = RetrievedChunk(
            chunk_id="test",
            spec_id="test",
            section_id="test",
            section_title="test",
            content="test content",
            chunk_type="test",
            complexity_score=0.0,
            key_terms=[],
            reference_path=[]
        )

        result = integrator.generate_answer("test question", [chunk], "MATCH (c) RETURN c", model="claude")
        assert isinstance(result, str)

    @patch('rag_system_v2.requests.post')
    def test_generate_answer_with_local_llm(self, mock_post):
        """generate_answer should work with local LLM"""
        from rag_system_v2 import LLMIntegrator, RetrievedChunk

        mock_response = Mock()
        mock_response.json.return_value = {'message': {'content': 'Test response'}}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        integrator = LLMIntegrator(local_llm_url="http://localhost:11434/api/chat")
        chunk = RetrievedChunk(
            chunk_id="test",
            spec_id="test",
            section_id="test",
            section_title="test",
            content="test content",
            chunk_type="test",
            complexity_score=0.0,
            key_terms=[],
            reference_path=[]
        )

        result = integrator.generate_answer("test question", [chunk], "MATCH (c) RETURN c", model="deepseek-r1:7b")
        assert isinstance(result, str)


class TestRAGOrchestratorV2:
    """Tests for RAGOrchestratorV2 class"""

    @pytest.fixture
    def mock_orchestrator_deps(self):
        """Mock all orchestrator dependencies"""
        with patch('rag_system_v2.GraphDatabase') as mock_db, \
             patch('rag_system_v2.anthropic') as mock_anthropic:

            mock_driver = Mock()
            mock_db.driver.return_value = mock_driver

            mock_message = Mock()
            mock_message.content = [Mock(text="Test response")]
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

            yield mock_db, mock_anthropic

    def test_create_rag_system_v2(self, mock_orchestrator_deps):
        """create_rag_system_v2 should return an orchestrator"""
        from rag_system_v2 import create_rag_system_v2

        rag = create_rag_system_v2("test_api_key", "http://localhost:11434/api/chat")
        assert rag is not None

    def test_orchestrator_has_query_method(self, mock_orchestrator_deps):
        """Orchestrator should have a query method"""
        from rag_system_v2 import create_rag_system_v2

        rag = create_rag_system_v2("test_api_key", "http://localhost:11434/api/chat")
        assert hasattr(rag, 'query')
        assert callable(rag.query)

    def test_orchestrator_has_close_method(self, mock_orchestrator_deps):
        """Orchestrator should have a close method"""
        from rag_system_v2 import create_rag_system_v2

        rag = create_rag_system_v2("test_api_key", "http://localhost:11434/api/chat")
        assert hasattr(rag, 'close')
        assert callable(rag.close)


class TestQuestionTypeClassification:
    """Regression tests for question type classification"""

    @pytest.fixture
    def generator(self):
        from rag_system_v2 import CypherQueryGenerator
        return CypherQueryGenerator()

    @pytest.mark.parametrize("question,expected_type", [
        ("What is AMF?", "definition"),
        ("Define the meaning of SMF", "definition"),
        ("Compare the difference between AMF and SMF", "comparison"),
        ("Compare UPF and SMF", "comparison"),
        ("How does registration work?", "procedure"),
        ("How to perform handover?", "procedure"),
        ("What does the specification document say about AMF?", "specification"),
        ("Describe the role of AMF in the network", "network_function"),
    ])
    def test_question_classification(self, generator, question, expected_type):
        """Question types should be correctly classified"""
        result = generator.analyze_question(question)
        assert result['question_type'] == expected_type


class TestEntityExtraction:
    """Regression tests for entity extraction"""

    @pytest.fixture
    def generator(self):
        from rag_system_v2 import CypherQueryGenerator
        return CypherQueryGenerator()

    @pytest.mark.parametrize("question,expected_entity", [
        ("What is AMF?", "AMF"),
        ("Explain the SMF role", "SMF"),
        ("How does UPF handle packets?", "UPF"),
        ("What is NRF used for?", "NRF"),
        ("Describe PCF functionality", "PCF"),
    ])
    def test_entity_extraction(self, generator, question, expected_entity):
        """Network function entities should be extracted"""
        result = generator.analyze_question(question)
        entity_values = [e['value'] for e in result['entities']]
        assert expected_entity in entity_values
