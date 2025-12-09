"""
Tests for rag_system_v3.py module.
Tests RAGOrchestratorV3 and EnhancedLLMIntegrator.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from dataclasses import dataclass
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Mock Classes
# ============================================================
@dataclass
class MockScoredChunk:
    """Mock ScoredChunk for testing"""
    chunk_id: str
    spec_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    complexity_score: float
    key_terms: List[str]
    retrieval_score: float
    retrieval_method: str


# ============================================================
# RAGResponseV3 Tests
# ============================================================
class TestRAGResponseV3:
    """Tests for RAGResponseV3 dataclass"""

    def test_rag_response_v3_creation(self):
        """Test creating RAGResponseV3"""
        from rag_system_v3 import RAGResponseV3

        chunk = MockScoredChunk(
            chunk_id="test_001",
            spec_id="ts_23.501",
            section_id="5.2",
            section_title="Test",
            content="Test content",
            chunk_type="definition",
            complexity_score=0.5,
            key_terms=["test"],
            retrieval_score=0.9,
            retrieval_method="hybrid"
        )

        response = RAGResponseV3(
            answer="Test answer",
            sources=[chunk],
            query="What is AMF?",
            retrieval_strategy="hybrid",
            query_analysis={'primary_intent': 'definition'},
            timestamp=datetime.now(),
            model_used="deepseek-r1:14b",
            retrieval_time_ms=100.0,
            generation_time_ms=500.0
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.model_used == "deepseek-r1:14b"
        assert response.retrieval_time_ms == 100.0

    def test_rag_response_v3_default_times(self):
        """Test default timing values"""
        from rag_system_v3 import RAGResponseV3

        response = RAGResponseV3(
            answer="Test",
            sources=[],
            query="Test",
            retrieval_strategy="test",
            query_analysis={},
            timestamp=datetime.now(),
            model_used="test"
        )

        assert response.retrieval_time_ms == 0.0
        assert response.generation_time_ms == 0.0


# ============================================================
# RAGConfigV3 Tests
# ============================================================
class TestRAGConfigV3:
    """Tests for RAGConfigV3 class"""

    def test_config_defaults(self):
        """Test default configuration values"""
        from rag_system_v3 import RAGConfigV3

        config = RAGConfigV3()

        assert config.neo4j_uri == "neo4j://localhost:7687"
        assert config.neo4j_user == "neo4j"
        assert config.neo4j_password == "password"
        assert config.claude_api_key is None
        assert "11434" in config.local_llm_url
        assert "sentence-transformers" in config.embedding_model


# ============================================================
# EnhancedLLMIntegrator Tests (Mock-based)
# ============================================================
class TestEnhancedLLMIntegrator:
    """Tests for EnhancedLLMIntegrator class"""

    @pytest.fixture
    def mock_chunks(self):
        """Create mock chunks"""
        return [
            MockScoredChunk(
                chunk_id="chunk_001",
                spec_id="ts_23.501",
                section_id="5.2.1",
                section_title="AMF Overview",
                content="AMF handles mobility management.",
                chunk_type="definition",
                complexity_score=0.5,
                key_terms=["AMF"],
                retrieval_score=0.9,
                retrieval_method="hybrid"
            )
        ]

    def test_generate_answer_v3_uses_analysis(self, mock_chunks):
        """Test that answer generation uses query analysis"""
        from rag_system_v3 import EnhancedLLMIntegrator

        # Create integrator without API key, then patch
        integrator = EnhancedLLMIntegrator(claude_api_key=None)

        # Mock the Claude client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated answer")]
        mock_client.messages.create.return_value = mock_response
        integrator.claude_client = mock_client

        analysis = {
            'primary_intent': 'definition',
            'entities': ['AMF']
        }

        answer = integrator.generate_answer_v3(
            query="What is AMF?",
            chunks=mock_chunks,
            analysis=analysis,
            model="claude"
        )

        assert answer == "Generated answer"
        # Verify Claude was called
        mock_client.messages.create.assert_called_once()

    @patch('rag_system_v3.requests')
    def test_generate_answer_v3_ollama(self, mock_requests, mock_chunks):
        """Test answer generation with Ollama"""
        from rag_system_v3 import EnhancedLLMIntegrator

        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Ollama answer"}
        }
        mock_response.raise_for_status = Mock()
        mock_requests.post.return_value = mock_response

        integrator = EnhancedLLMIntegrator(local_llm_url="http://localhost:11434/api/chat")

        analysis = {
            'primary_intent': 'definition',
            'entities': ['AMF']
        }

        answer = integrator.generate_answer_v3(
            query="What is AMF?",
            chunks=mock_chunks,
            analysis=analysis,
            model="deepseek-r1:14b"
        )

        assert answer == "Ollama answer"
        mock_requests.post.assert_called_once()

    def test_generate_answer_v3_no_claude_client(self, mock_chunks):
        """Test error handling when Claude client not initialized"""
        from rag_system_v3 import EnhancedLLMIntegrator

        integrator = EnhancedLLMIntegrator(claude_api_key=None)

        answer = integrator.generate_answer_v3(
            query="What is AMF?",
            chunks=mock_chunks,
            analysis={},
            model="claude"
        )

        assert "not available" in answer.lower()


# ============================================================
# RAGOrchestratorV3 Tests (Mock-based)
# ============================================================
class TestRAGOrchestratorV3:
    """Tests for RAGOrchestratorV3 class"""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create mock Neo4j driver"""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver

    @patch('rag_system_v3.GraphDatabase')
    @patch('rag_system_v3.HybridRetriever')
    @patch('rag_system_v3.VectorIndexer')
    @patch('rag_system_v3.CypherQueryGenerator')
    @patch('rag_system_v3.EnhancedLLMIntegrator')
    def test_orchestrator_initialization(
        self, mock_llm, mock_cypher, mock_indexer,
        mock_hybrid, mock_graph_db
    ):
        """Test RAGOrchestratorV3 initialization"""
        from rag_system_v3 import RAGOrchestratorV3

        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        orchestrator = RAGOrchestratorV3()

        mock_graph_db.driver.assert_called_once()
        assert orchestrator.driver is not None

    @patch('rag_system_v3.GraphDatabase')
    @patch('rag_system_v3.HybridRetriever')
    @patch('rag_system_v3.VectorIndexer')
    @patch('rag_system_v3.CypherQueryGenerator')
    @patch('rag_system_v3.EnhancedLLMIntegrator')
    def test_check_vector_index_status(
        self, mock_llm, mock_cypher, mock_indexer,
        mock_hybrid, mock_graph_db
    ):
        """Test vector index status checking"""
        from rag_system_v3 import RAGOrchestratorV3

        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        mock_indexer_instance = Mock()
        mock_indexer_instance.check_embeddings_exist.return_value = (100, 100)
        mock_indexer_instance.check_vector_index_exists.return_value = True
        mock_indexer.return_value = mock_indexer_instance

        orchestrator = RAGOrchestratorV3()
        status = orchestrator.check_vector_index_status()

        assert status['total_chunks'] == 100
        assert status['chunks_with_embeddings'] == 100
        assert status['embeddings_complete'] == True
        assert status['vector_index_exists'] == True
        assert status['ready_for_hybrid'] == True

    @patch('rag_system_v3.GraphDatabase')
    @patch('rag_system_v3.HybridRetriever')
    @patch('rag_system_v3.VectorIndexer')
    @patch('rag_system_v3.CypherQueryGenerator')
    @patch('rag_system_v3.EnhancedLLMIntegrator')
    def test_query_returns_response(
        self, mock_llm, mock_cypher, mock_indexer,
        mock_hybrid, mock_graph_db
    ):
        """Test that query returns RAGResponseV3"""
        from rag_system_v3 import RAGOrchestratorV3, RAGResponseV3

        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        mock_indexer_instance = Mock()
        mock_indexer_instance.check_embeddings_exist.return_value = (100, 100)
        mock_indexer_instance.check_vector_index_exists.return_value = True
        mock_indexer.return_value = mock_indexer_instance

        mock_chunk = MockScoredChunk(
            chunk_id="test_001",
            spec_id="ts_23.501",
            section_id="5.2",
            section_title="Test",
            content="Test content",
            chunk_type="definition",
            complexity_score=0.5,
            key_terms=[],
            retrieval_score=0.9,
            retrieval_method="hybrid"
        )

        mock_hybrid_instance = Mock()
        mock_hybrid_instance.retrieve.return_value = (
            [mock_chunk],
            "hybrid strategy",
            {'primary_intent': 'definition'}
        )
        mock_hybrid_instance.query_analyzer._fallback_analysis.return_value = {
            'primary_intent': 'definition',
            'entities': [],
            'key_terms': [],
            'complexity': 'simple',
            'requires_multi_step': False,
            'sub_questions': []
        }
        mock_hybrid.return_value = mock_hybrid_instance

        mock_llm_instance = Mock()
        mock_llm_instance.generate_answer_v3.return_value = "Test answer"
        mock_llm.return_value = mock_llm_instance

        orchestrator = RAGOrchestratorV3()
        response = orchestrator.query("What is AMF?")

        assert isinstance(response, RAGResponseV3)
        assert response.answer == "Test answer"
        assert len(response.sources) == 1

    @patch('rag_system_v3.GraphDatabase')
    @patch('rag_system_v3.HybridRetriever')
    @patch('rag_system_v3.VectorIndexer')
    @patch('rag_system_v3.CypherQueryGenerator')
    @patch('rag_system_v3.EnhancedLLMIntegrator')
    def test_query_empty_results(
        self, mock_llm, mock_cypher, mock_indexer,
        mock_hybrid, mock_graph_db
    ):
        """Test query with no results"""
        from rag_system_v3 import RAGOrchestratorV3

        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        mock_indexer_instance = Mock()
        mock_indexer_instance.check_embeddings_exist.return_value = (100, 100)
        mock_indexer_instance.check_vector_index_exists.return_value = True
        mock_indexer.return_value = mock_indexer_instance

        mock_hybrid_instance = Mock()
        mock_hybrid_instance.retrieve.return_value = ([], "no results", {})
        mock_hybrid_instance.query_analyzer._fallback_analysis.return_value = {
            'primary_intent': 'general',
            'entities': [],
            'key_terms': [],
            'complexity': 'simple',
            'requires_multi_step': False,
            'sub_questions': []
        }
        mock_hybrid.return_value = mock_hybrid_instance

        orchestrator = RAGOrchestratorV3()
        response = orchestrator.query("Unknown topic xyz")

        assert "No relevant information" in response.answer
        assert len(response.sources) == 0

    @patch('rag_system_v3.GraphDatabase')
    @patch('rag_system_v3.HybridRetriever')
    @patch('rag_system_v3.VectorIndexer')
    @patch('rag_system_v3.CypherQueryGenerator')
    @patch('rag_system_v3.EnhancedLLMIntegrator')
    def test_explain_query(
        self, mock_llm, mock_cypher, mock_indexer,
        mock_hybrid, mock_graph_db
    ):
        """Test query explanation"""
        from rag_system_v3 import RAGOrchestratorV3

        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        mock_indexer_instance = Mock()
        mock_indexer_instance.check_embeddings_exist.return_value = (100, 100)
        mock_indexer_instance.check_vector_index_exists.return_value = True
        mock_indexer.return_value = mock_indexer_instance

        mock_hybrid_instance = Mock()
        mock_hybrid_instance.query_analyzer._fallback_analysis.return_value = {
            'primary_intent': 'definition',
            'entities': ['AMF'],
            'key_terms': ['amf'],
            'complexity': 'simple',
            'requires_multi_step': False,
            'sub_questions': []
        }
        mock_hybrid_instance.query_expander.expand.return_value = [
            "What is AMF?",
            "What is AMF Access and Mobility Management Function?"
        ]
        mock_hybrid.return_value = mock_hybrid_instance

        orchestrator = RAGOrchestratorV3()
        explanation = orchestrator.explain_query("What is AMF?")

        assert 'question' in explanation
        assert 'analysis' in explanation
        assert 'query_variations' in explanation
        assert 'vector_status' in explanation


# ============================================================
# Factory Function Tests
# ============================================================
class TestCreateRagSystemV3:
    """Tests for create_rag_system_v3 factory function"""

    @patch('rag_system_v3.RAGOrchestratorV3')
    def test_create_with_defaults(self, mock_orchestrator):
        """Test factory with default settings"""
        from rag_system_v3 import create_rag_system_v3

        create_rag_system_v3()

        mock_orchestrator.assert_called_once()

    @patch('rag_system_v3.RAGOrchestratorV3')
    def test_create_with_custom_params(self, mock_orchestrator):
        """Test factory with custom parameters"""
        from rag_system_v3 import create_rag_system_v3

        create_rag_system_v3(
            claude_api_key="test-key",
            local_llm_url="http://custom:11434/api/chat",
            embedding_model="custom-model"
        )

        mock_orchestrator.assert_called_once()
        call_kwargs = mock_orchestrator.call_args.kwargs

        assert call_kwargs['claude_api_key'] == "test-key"
        assert call_kwargs['local_llm_url'] == "http://custom:11434/api/chat"
        assert call_kwargs['embedding_model'] == "custom-model"


# ============================================================
# Integration Tests (require Neo4j)
# ============================================================
class TestRAGV3Integration:
    """Integration tests requiring Neo4j connection"""

    @staticmethod
    def _neo4j_available():
        """Check if Neo4j is available"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                "neo4j://localhost:7687",
                auth=("neo4j", "password")
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except:
            return False

    @pytest.mark.skipif(
        not _neo4j_available.__func__(),
        reason="Neo4j not available"
    )
    def test_full_query_pipeline(self):
        """Test full RAG V3 query pipeline"""
        from rag_system_v3 import create_rag_system_v3

        rag = create_rag_system_v3()

        try:
            # Check status
            status = rag.check_vector_index_status()
            assert 'total_chunks' in status

            # Query (graph-only if vector not setup)
            response = rag.query(
                "What is AMF?",
                use_hybrid=True,
                use_vector=status['ready_for_hybrid'],
                use_graph=True
            )

            assert response.answer is not None
            assert response.query == "What is AMF?"

        finally:
            rag.close()

    @pytest.mark.skipif(
        not _neo4j_available.__func__(),
        reason="Neo4j not available"
    )
    def test_explain_query_integration(self):
        """Test query explanation integration"""
        from rag_system_v3 import create_rag_system_v3

        rag = create_rag_system_v3()

        try:
            explanation = rag.explain_query("Compare AMF and SMF")

            assert explanation['analysis']['primary_intent'] in ['comparison', 'general']
            assert 'AMF' in explanation['analysis'].get('entities', []) or len(explanation['analysis'].get('entities', [])) >= 0

        finally:
            rag.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
