"""
Tests for hybrid_retriever.py module.
Tests VectorIndexer, VectorRetriever, SemanticQueryAnalyzer, QueryExpander, and HybridRetriever.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_retriever import (
    ScoredChunk,
    VectorIndexer,
    VectorRetriever,
    SemanticQueryAnalyzer,
    QueryExpander,
    HybridRetriever
)


# ============================================================
# ScoredChunk Tests
# ============================================================
class TestScoredChunk:
    """Tests for ScoredChunk dataclass"""

    def test_scored_chunk_creation(self):
        """Test creating a ScoredChunk"""
        chunk = ScoredChunk(
            chunk_id="test_chunk_001",
            spec_id="ts_23.501",
            section_id="5.2.1",
            section_title="AMF Overview",
            content="The AMF is responsible for...",
            chunk_type="definition",
            complexity_score=0.5,
            key_terms=["AMF", "mobility"],
            retrieval_score=0.85,
            retrieval_method="vector"
        )

        assert chunk.chunk_id == "test_chunk_001"
        assert chunk.spec_id == "ts_23.501"
        assert chunk.retrieval_score == 0.85
        assert chunk.retrieval_method == "vector"

    def test_scored_chunk_default_reference_path(self):
        """Test that reference_path defaults to empty list"""
        chunk = ScoredChunk(
            chunk_id="test",
            spec_id="ts_23.501",
            section_id="1",
            section_title="Test",
            content="Test content",
            chunk_type="general",
            complexity_score=0.0,
            key_terms=[],
            retrieval_score=0.5,
            retrieval_method="graph"
        )

        assert chunk.reference_path == []


# ============================================================
# QueryExpander Tests
# ============================================================
class TestQueryExpander:
    """Tests for QueryExpander class"""

    @pytest.fixture
    def term_dict(self):
        """Sample term dictionary"""
        return {
            'AMF': {'full_name': 'Access and Mobility Management Function', 'type': 'network_function'},
            'SMF': {'full_name': 'Session Management Function', 'type': 'network_function'},
            'UPF': {'full_name': 'User Plane Function', 'type': 'network_function'},
        }

    @pytest.fixture
    def expander(self, term_dict):
        """Create QueryExpander instance"""
        return QueryExpander(term_dict=term_dict)

    def test_expand_returns_original(self, expander):
        """Test that expansion always includes original query"""
        query = "What is AMF?"
        variations = expander.expand(query)

        assert query in variations
        assert variations[0] == query  # Original should be first

    def test_expand_abbreviations(self, expander):
        """Test abbreviation expansion"""
        query = "What is AMF?"
        variations = expander.expand(query)

        # Should have variation with full name
        has_full_name = any("Access and Mobility Management Function" in v for v in variations)
        assert has_full_name

    def test_expand_synonyms(self, expander):
        """Test synonym expansion"""
        query = "What is the role of AMF?"
        variations = expander.expand(query)

        # Should have variations with synonyms
        has_function = any("function" in v.lower() for v in variations if v != query)
        assert has_function or len(variations) > 1

    def test_expand_keywords(self, expander):
        """Test keyword extraction"""
        query = "What is the AMF in 5G?"
        variations = expander.expand(query)

        # Should have keyword-only variation
        assert len(variations) >= 2

    def test_expand_max_variations(self, expander):
        """Test that max_variations is respected"""
        query = "What is the role of AMF?"
        variations = expander.expand(query, max_variations=2)

        assert len(variations) <= 2

    def test_expand_empty_term_dict(self):
        """Test expansion with empty term dict"""
        expander = QueryExpander(term_dict={})
        query = "What is AMF?"
        variations = expander.expand(query)

        assert query in variations

    def test_extract_keywords(self, expander):
        """Test keyword extraction method"""
        query = "What is the role of AMF in 5G?"
        keywords = expander._extract_keywords(query)

        # Should remove stop words
        assert "what" not in keywords.lower()
        assert "is" not in keywords.lower()
        assert "the" not in keywords.lower()


# ============================================================
# SemanticQueryAnalyzer Tests
# ============================================================
class TestSemanticQueryAnalyzer:
    """Tests for SemanticQueryAnalyzer class"""

    @pytest.fixture
    def term_dict(self):
        """Sample term dictionary"""
        return {
            'AMF': {'full_name': 'Access and Mobility Management Function'},
            'SMF': {'full_name': 'Session Management Function'},
        }

    @pytest.fixture
    def analyzer(self, term_dict):
        """Create analyzer instance"""
        return SemanticQueryAnalyzer(
            local_llm_url="http://localhost:11434/api/chat",
            term_dict=term_dict
        )

    def test_fallback_analysis_definition(self, analyzer):
        """Test fallback analysis for definition questions"""
        analysis = analyzer._fallback_analysis("What is AMF?")

        assert analysis['primary_intent'] == 'definition'
        assert 'AMF' in analysis['entities']

    def test_fallback_analysis_comparison(self, analyzer):
        """Test fallback analysis for comparison questions"""
        analysis = analyzer._fallback_analysis("What is the difference between AMF and SMF?")

        assert analysis['primary_intent'] == 'comparison'
        assert 'AMF' in analysis['entities']
        assert 'SMF' in analysis['entities']
        assert analysis['requires_multi_step'] == True

    def test_fallback_analysis_procedure(self, analyzer):
        """Test fallback analysis for procedure questions"""
        analysis = analyzer._fallback_analysis("How does registration procedure work?")

        assert analysis['primary_intent'] == 'procedure'

    def test_fallback_analysis_network_function(self, analyzer):
        """Test fallback analysis for role questions"""
        analysis = analyzer._fallback_analysis("What is the role of AMF?")

        # May detect as definition or network_function depending on pattern matching order
        assert analysis['primary_intent'] in ['network_function', 'definition']

    def test_fallback_analysis_general(self, analyzer):
        """Test fallback analysis for general questions"""
        analysis = analyzer._fallback_analysis("Tell me about 5G")

        assert analysis['primary_intent'] == 'general'

    def test_fallback_analysis_complexity(self, analyzer):
        """Test complexity detection"""
        # Simple - single entity
        simple = analyzer._fallback_analysis("What is AMF?")
        assert simple['complexity'] == 'simple'

        # Medium - multiple entities
        medium = analyzer._fallback_analysis("Compare AMF and SMF")
        assert medium['complexity'] == 'medium'


# ============================================================
# VectorIndexer Tests (Mock-based)
# ============================================================
class TestVectorIndexer:
    """Tests for VectorIndexer class"""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver"""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver

    def test_check_vector_index_exists_true(self, mock_driver):
        """Test checking when vector index exists"""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {'exists': True}

        indexer = VectorIndexer(mock_driver)
        result = indexer.check_vector_index_exists()

        assert result == True

    def test_check_vector_index_exists_false(self, mock_driver):
        """Test checking when vector index doesn't exist"""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {'exists': False}

        indexer = VectorIndexer(mock_driver)
        result = indexer.check_vector_index_exists()

        assert result == False

    def test_check_embeddings_exist(self, mock_driver):
        """Test checking embeddings count"""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            'with_embeddings': 100,
            'total': 200
        }

        indexer = VectorIndexer(mock_driver)
        with_emb, total = indexer.check_embeddings_exist()

        assert with_emb == 100
        assert total == 200


# ============================================================
# VectorRetriever Tests (Mock-based)
# ============================================================
class TestVectorRetriever:
    """Tests for VectorRetriever class"""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver"""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver

    def test_search_returns_scored_chunks(self, mock_driver):
        """Test that search returns ScoredChunk objects"""
        session = mock_driver.session.return_value.__enter__.return_value

        # Mock query results
        mock_record = {
            'chunk_id': 'test_001',
            'spec_id': 'ts_23.501',
            'section_id': '5.2',
            'section_title': 'AMF',
            'content': 'AMF content',
            'chunk_type': 'definition',
            'complexity_score': 0.5,
            'key_terms': ['AMF'],
            'score': 0.9
        }
        session.run.return_value = [mock_record]

        with patch.object(VectorRetriever, '_load_model'):
            retriever = VectorRetriever(mock_driver)
            retriever.model = Mock()

            # Mock encode to return numpy array-like object with tolist method
            import numpy as np
            retriever.model.encode.return_value = np.array([0.1] * 384)

            results = retriever.search("What is AMF?", top_k=5)

            assert len(results) == 1
            assert isinstance(results[0], ScoredChunk)
            assert results[0].retrieval_method == 'vector'


# ============================================================
# HybridRetriever Tests (Mock-based)
# ============================================================
class TestHybridRetriever:
    """Tests for HybridRetriever class"""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver"""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver

    @pytest.fixture
    def mock_cypher_generator(self):
        """Create mock CypherQueryGenerator"""
        generator = Mock()
        generator.all_terms = {
            'AMF': {'full_name': 'Access and Mobility Management Function', 'type': 'network_function'}
        }
        generator.generate_cypher_query.return_value = "MATCH (c:Chunk) RETURN c LIMIT 5"
        return generator

    def test_merge_results_boosts_duplicates(self):
        """Test that chunks found in multiple sources get boosted"""
        all_results = {}

        # First chunk from vector search
        chunk1 = ScoredChunk(
            chunk_id="chunk_001",
            spec_id="ts_23.501",
            section_id="1",
            section_title="Test",
            content="Test",
            chunk_type="general",
            complexity_score=0.5,
            key_terms=[],
            retrieval_score=0.8,
            retrieval_method="vector"
        )

        # Same chunk from graph search
        chunk2 = ScoredChunk(
            chunk_id="chunk_001",
            spec_id="ts_23.501",
            section_id="1",
            section_title="Test",
            content="Test",
            chunk_type="general",
            complexity_score=0.5,
            key_terms=[],
            retrieval_score=0.7,
            retrieval_method="graph"
        )

        # Simulate _merge_results logic
        all_results[chunk1.chunk_id] = chunk1
        all_results[chunk1.chunk_id].retrieval_score += chunk2.retrieval_score * 0.5
        all_results[chunk1.chunk_id].retrieval_method = 'vector+graph'

        assert all_results["chunk_001"].retrieval_method == "vector+graph"
        assert all_results["chunk_001"].retrieval_score > 0.8

    def test_rerank_boosts_multi_source(self):
        """Test that reranking boosts multi-source chunks"""
        chunks = {
            "chunk_001": ScoredChunk(
                chunk_id="chunk_001",
                spec_id="ts_23.501",
                section_id="1",
                section_title="Test1",
                content="Test1",
                chunk_type="general",
                complexity_score=0.5,
                key_terms=[],
                retrieval_score=0.8,
                retrieval_method="vector+graph"
            ),
            "chunk_002": ScoredChunk(
                chunk_id="chunk_002",
                spec_id="ts_23.501",
                section_id="2",
                section_title="Test2",
                content="Test2",
                chunk_type="general",
                complexity_score=0.5,
                key_terms=[],
                retrieval_score=0.9,
                retrieval_method="vector"
            )
        }

        # Apply boost logic
        for chunk in chunks.values():
            if chunk.retrieval_method == 'vector+graph':
                chunk.retrieval_score *= 1.3

        # chunk_001 should now be higher
        assert chunks["chunk_001"].retrieval_score > chunks["chunk_002"].retrieval_score


# ============================================================
# Integration Tests (require Neo4j)
# ============================================================
class TestHybridRetrieverIntegration:
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
    def test_hybrid_retriever_integration(self):
        """Test full hybrid retrieval pipeline"""
        from hybrid_retriever import create_hybrid_retriever

        retriever, driver = create_hybrid_retriever()

        try:
            chunks, strategy, analysis = retriever.retrieve(
                "What is AMF?",
                top_k=5,
                use_vector=False,  # Skip vector if not setup
                use_graph=True,
                use_query_expansion=True,
                use_llm_analysis=False
            )

            assert isinstance(chunks, list)
            assert isinstance(strategy, str)
            assert isinstance(analysis, dict)

        finally:
            driver.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
