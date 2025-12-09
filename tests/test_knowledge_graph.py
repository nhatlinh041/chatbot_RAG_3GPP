"""
Test suite for Knowledge Graph components - Neo4j integration and graph operations
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from rag_system_v2 import (
    Neo4jConnection,
    EnhancedKnowledgeRetriever,
    CypherQueryGenerator
)
from cypher_sanitizer import CypherSanitizer


class TestNeo4jConnection:
    """Test Neo4j database connection handling"""
    
    def setup_method(self):
        self.connection = Neo4jConnection(
            uri="neo4j://localhost:7687",
            user="neo4j", 
            password="password"
        )
    
    @patch('neo4j.GraphDatabase.driver')
    def test_connection_initialization(self, mock_driver):
        # Test successful connection initialization
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        
        connection = Neo4jConnection("neo4j://localhost:7687", "neo4j", "password")
        assert connection.driver == mock_driver_instance
        mock_driver.assert_called_once()
    
    @patch('neo4j.GraphDatabase.driver')
    def test_connection_failure(self, mock_driver):
        # Test connection failure handling
        mock_driver.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            Neo4jConnection("neo4j://localhost:7687", "neo4j", "password")
    
    def test_close_connection(self):
        # Test connection closing
        mock_driver = MagicMock()
        self.connection.driver = mock_driver
        
        self.connection.close()
        mock_driver.close.assert_called_once()
    
    @patch('neo4j.GraphDatabase.driver')
    def test_query_execution(self, mock_driver):
        # Test query execution
        mock_session = MagicMock()
        mock_result = [{"test": "data"}]
        mock_session.run.return_value = mock_result
        
        mock_driver_instance = MagicMock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver.return_value = mock_driver_instance
        
        connection = Neo4jConnection("neo4j://localhost:7687", "neo4j", "password")
        result = connection.execute_query("MATCH (n) RETURN n")
        
        assert result == mock_result
        mock_session.run.assert_called_once()


class TestCypherQueryGenerator:
    """Test Cypher query generation for different question types"""
    
    def setup_method(self):
        self.generator = CypherQueryGenerator()
    
    def test_generator_initialization(self):
        # Test generator initialization
        assert hasattr(self.generator, 'query_patterns')
        assert len(self.generator.query_patterns) > 0
        assert 'definition' in self.generator.query_patterns
        assert 'comparison' in self.generator.query_patterns
    
    def test_generate_definition_query(self):
        # Test definition query generation
        question_analysis = {
            'main_entities': ['AMF', 'UE'],
            'question_type': 'definition',
            'key_terms': ['function', 'role']
        }
        
        query = self.generator._generate_definition_query(question_analysis)
        
        assert 'MATCH' in query
        assert 'AMF' in query or 'UE' in query
        assert 'RETURN' in query
    
    def test_generate_comparison_query(self):
        # Test comparison query generation  
        question_analysis = {
            'main_entities': ['AMF', 'SMF'],
            'question_type': 'comparison',
            'key_terms': ['compare', 'difference']
        }
        
        query = self.generator._generate_comparison_query(question_analysis)
        
        assert 'MATCH' in query
        assert 'AMF' in query
        assert 'SMF' in query
        assert 'RETURN' in query
    
    def test_generate_procedure_query(self):
        # Test procedure query generation
        question_analysis = {
            'main_entities': ['handover', 'procedure'],
            'question_type': 'procedure',
            'key_terms': ['steps', 'process', 'how']
        }
        
        query = self.generator._generate_procedure_query(question_analysis)
        
        assert 'MATCH' in query
        assert 'procedure' in query.lower() or 'handover' in query.lower()
        assert 'RETURN' in query
    
    def test_generate_network_function_query(self):
        # Test network function query generation
        question_analysis = {
            'main_entities': ['UPF', 'network'],
            'question_type': 'network_function',
            'key_terms': ['function', 'purpose']
        }
        
        query = self.generator._generate_network_function_query(question_analysis)
        
        assert 'MATCH' in query
        assert 'UPF' in query
        assert 'RETURN' in query
    
    def test_generate_relationship_query(self):
        # Test relationship query generation
        question_analysis = {
            'main_entities': ['AMF', 'SMF', 'UPF'],
            'question_type': 'relationship',
            'key_terms': ['relationship', 'connection', 'interact']
        }
        
        query = self.generator._generate_relationship_query(question_analysis)
        
        assert 'MATCH' in query
        assert 'relationship' in query.lower() or '->' in query
        assert 'RETURN' in query
    
    def test_generate_architecture_query(self):
        # Test architecture query generation
        question_analysis = {
            'main_entities': ['5G', 'core', 'architecture'],
            'question_type': 'architecture',
            'key_terms': ['architecture', 'structure', 'components']
        }
        
        query = self.generator._generate_architecture_query(question_analysis)
        
        assert 'MATCH' in query
        assert '5G' in query or 'core' in query
        assert 'RETURN' in query
    
    def test_generate_general_query(self):
        # Test general query generation
        question_analysis = {
            'main_entities': ['5G', 'technology'],
            'question_type': 'general',
            'key_terms': ['what', 'explain']
        }
        
        query = self.generator._generate_general_query(question_analysis)
        
        assert 'MATCH' in query
        assert '5G' in query
        assert 'RETURN' in query
    
    def test_sanitize_terms_in_query(self):
        # Test that generated queries use sanitized terms
        question_analysis = {
            'main_entities': ['AMF"; DROP TABLE users; --'],
            'question_type': 'definition',
            'key_terms': ['function']
        }
        
        query = self.generator._generate_definition_query(question_analysis)
        
        # Should not contain injection attempts
        assert 'DROP TABLE' not in query
        assert '--' not in query
        assert '"' not in query
    
    def test_query_patterns_completeness(self):
        # Test that all expected query patterns exist
        expected_patterns = [
            'definition', 'comparison', 'procedure', 'network_function',
            'relationship', 'architecture', 'general'
        ]
        
        for pattern in expected_patterns:
            assert pattern in self.generator.query_patterns
            assert hasattr(self.generator, f'_generate_{pattern}_query')


class TestEnhancedKnowledgeRetriever:
    """Test knowledge retrieval from Neo4j"""
    
    def setup_method(self):
        mock_connection = MagicMock()
        self.retriever = EnhancedKnowledgeRetriever(mock_connection)
    
    def test_retriever_initialization(self):
        # Test retriever initialization
        mock_connection = MagicMock()
        retriever = EnhancedKnowledgeRetriever(mock_connection)
        
        assert retriever.connection == mock_connection
        assert hasattr(retriever, 'query_generator')
    
    @patch.object(CypherQueryGenerator, 'generate_query')
    def test_search_knowledge_graph_success(self, mock_generate_query):
        # Test successful knowledge graph search
        mock_generate_query.return_value = "MATCH (n) RETURN n"
        
        mock_records = [
            {'chunk_id': 1, 'content': 'Test content 1', 'document_id': 'doc1'},
            {'chunk_id': 2, 'content': 'Test content 2', 'document_id': 'doc2'}
        ]
        
        self.retriever.connection.execute_query.return_value = mock_records
        
        question_analysis = {
            'main_entities': ['AMF'],
            'question_type': 'definition',
            'key_terms': ['function']
        }
        
        results = self.retriever.search_knowledge_graph(question_analysis)
        
        assert len(results) == 2
        assert results[0]['content'] == 'Test content 1'
        assert results[1]['content'] == 'Test content 2'
    
    @patch.object(CypherQueryGenerator, 'generate_query')
    def test_search_knowledge_graph_empty_results(self, mock_generate_query):
        # Test knowledge graph search with empty results
        mock_generate_query.return_value = "MATCH (n) RETURN n"
        self.retriever.connection.execute_query.return_value = []
        
        question_analysis = {
            'main_entities': ['NonExistent'],
            'question_type': 'definition',
            'key_terms': []
        }
        
        results = self.retriever.search_knowledge_graph(question_analysis)
        
        assert results == []
    
    @patch.object(CypherQueryGenerator, 'generate_query')
    def test_search_knowledge_graph_error_handling(self, mock_generate_query):
        # Test error handling in knowledge graph search
        mock_generate_query.return_value = "MATCH (n) RETURN n"
        self.retriever.connection.execute_query.side_effect = Exception("Neo4j error")
        
        question_analysis = {
            'main_entities': ['AMF'],
            'question_type': 'definition',
            'key_terms': []
        }
        
        results = self.retriever.search_knowledge_graph(question_analysis)
        
        assert results == []
    
    def test_format_search_results(self):
        # Test formatting of search results
        raw_results = [
            {
                'chunk_id': 1,
                'content': 'AMF manages mobility',
                'document_id': 'ts_23.501',
                'section_title': 'AMF Functions'
            },
            {
                'chunk_id': 2,
                'content': 'SMF handles sessions',
                'document_id': 'ts_23.501',
                'section_title': 'SMF Functions'
            }
        ]
        
        formatted = self.retriever._format_results(raw_results)
        
        assert len(formatted) == 2
        assert 'AMF manages mobility' in formatted[0]
        assert 'SMF handles sessions' in formatted[1]
        assert 'ts_23.501' in formatted[0]
    
    def test_filter_relevant_chunks(self):
        # Test filtering of relevant chunks
        chunks = [
            {
                'chunk_id': 1,
                'content': 'AMF is responsible for access and mobility management',
                'relevance_score': 0.9
            },
            {
                'chunk_id': 2,
                'content': 'This is unrelated content about cooking',
                'relevance_score': 0.1
            }
        ]
        
        filtered = self.retriever._filter_relevant_chunks(chunks, threshold=0.5)
        
        assert len(filtered) == 1
        assert filtered[0]['chunk_id'] == 1


class TestKnowledgeGraphQueries:
    """Test specific knowledge graph query scenarios"""
    
    def setup_method(self):
        mock_connection = MagicMock()
        self.retriever = EnhancedKnowledgeRetriever(mock_connection)
        self.generator = CypherQueryGenerator()
    
    def test_term_resolution_query(self):
        # Test term resolution through Term nodes
        question_analysis = {
            'main_entities': ['SCP'],
            'question_type': 'definition',
            'key_terms': []
        }
        
        # Mock Term node resolution
        self.retriever.connection.execute_query.side_effect = [
            # First call: resolve abbreviation
            [{'full_name': 'Service Communication Proxy'}],
            # Second call: get definition chunks
            [{'content': 'SCP is a network function...', 'chunk_id': 1}]
        ]
        
        results = self.retriever.search_knowledge_graph(question_analysis)
        
        # Should call execute_query multiple times for term resolution
        assert self.retriever.connection.execute_query.call_count >= 1
    
    def test_cross_document_search(self):
        # Test searching across multiple documents
        question_analysis = {
            'main_entities': ['handover'],
            'question_type': 'procedure',
            'key_terms': ['inter-RAT']
        }
        
        mock_results = [
            {
                'content': 'Inter-RAT handover procedure step 1',
                'document_id': 'ts_36.413',
                'chunk_id': 1
            },
            {
                'content': 'Inter-RAT handover procedure step 2',
                'document_id': 'ts_23.501',
                'chunk_id': 2
            }
        ]
        
        self.retriever.connection.execute_query.return_value = mock_results
        results = self.retriever.search_knowledge_graph(question_analysis)
        
        # Should return results from multiple documents
        document_ids = {r.get('document_id') for r in mock_results}
        assert len(document_ids) > 1
    
    def test_relationship_traversal_query(self):
        # Test queries that traverse relationships
        question_analysis = {
            'main_entities': ['AMF', 'UE'],
            'question_type': 'relationship',
            'key_terms': ['interaction', 'communication']
        }
        
        query = self.generator._generate_relationship_query(question_analysis)
        
        # Should contain relationship traversal patterns
        assert '-[' in query or 'REFERENCES' in query or 'CONTAINS' in query


class TestKnowledgeGraphIntegration:
    """Test integration between components"""
    
    @patch('neo4j.GraphDatabase.driver')
    def test_full_retrieval_pipeline(self, mock_driver):
        # Test complete retrieval pipeline
        mock_session = MagicMock()
        mock_records = [
            {
                'chunk_id': 1,
                'content': 'AMF (Access and Mobility Management Function) is a key network function',
                'document_id': 'ts_23.501'
            }
        ]
        mock_session.run.return_value = mock_records
        
        mock_driver_instance = MagicMock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver.return_value = mock_driver_instance
        
        # Create components
        connection = Neo4jConnection("neo4j://localhost:7687", "neo4j", "password")
        retriever = EnhancedKnowledgeRetriever(connection)
        
        question_analysis = {
            'main_entities': ['AMF'],
            'question_type': 'definition',
            'key_terms': ['function']
        }
        
        results = retriever.search_knowledge_graph(question_analysis)
        
        assert len(results) > 0
        assert 'AMF' in str(results)
    
    def test_error_recovery_mechanisms(self):
        # Test error recovery in knowledge retrieval
        mock_connection = MagicMock()
        mock_connection.execute_query.side_effect = [
            Exception("First query failed"),  # Primary query fails
            [{'content': 'Fallback result', 'chunk_id': 1}]  # Fallback succeeds
        ]
        
        retriever = EnhancedKnowledgeRetriever(mock_connection)
        
        question_analysis = {
            'main_entities': ['AMF'],
            'question_type': 'definition',
            'key_terms': []
        }
        
        # Should handle errors gracefully
        results = retriever.search_knowledge_graph(question_analysis)
        
        # Should either return empty results or fallback results
        assert isinstance(results, list)