"""
RAG System V3 for 3GPP Knowledge Graph.
Enhanced version with Hybrid Retrieval (Vector + Graph),
LLM Query Understanding, and Enhanced Prompts.
"""
import json
import re
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
import anthropic
import requests
from dataclasses import dataclass
from datetime import datetime
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG
from cypher_sanitizer import CypherSanitizer


def extract_mcq_question_only(question: str) -> Tuple[str, bool]:
    """
    Extract the question part only from MCQ format, excluding choices.
    This prevents choices from polluting retrieval results.

    Args:
        question: Full question text (may include A. B. C. D. choices)

    Returns:
        Tuple of (question_only, is_mcq)
        - question_only: Just the question text without choices
        - is_mcq: True if MCQ format was detected
    """
    # Pattern 1: Traditional MCQ with newline + A. B. C. D.
    # Split at first occurrence of newline followed by A./A)
    mcq_split = re.split(r'\n\s*[Aa][\.\)]\s*', question, maxsplit=1)
    if len(mcq_split) > 1:
        return mcq_split[0].strip(), True

    # Pattern 2: Inline (A) (B) (C) format
    inline_split = re.split(r'\s*\([Aa]\)\s*', question, maxsplit=1)
    if len(inline_split) > 1:
        return inline_split[0].strip(), True

    # Pattern 3: A: B: C: format (space before A:)
    colon_split = re.split(r'\s+[Aa]\s*:\s*', question, maxsplit=1)
    if len(colon_split) > 1 and len(colon_split[0]) > 10:  # Ensure we have actual question
        return colon_split[0].strip(), True

    # Not MCQ format
    return question, False

# Import new components
from hybrid_retriever import (
    HybridRetriever, VectorIndexer, VectorRetriever,
    SemanticQueryAnalyzer, QueryExpander, ScoredChunk,
    create_hybrid_retriever
)
from prompt_templates import PromptTemplates, ContextBuilder

# Import core components (shared with legacy V2)
from rag_core import (
    RetrievedChunk, RAGResponse, CypherQueryGenerator, LLMIntegrator
)

# Initialize logging
setup_centralized_logging()
logger = get_logger('RAG_System_V3')


class EnhancedLLMIntegrator(LLMIntegrator):
    """
    Enhanced LLM Integrator with query-aware prompt selection.
    Extends base LLMIntegrator with enhanced prompt templates.
    """

    def __init__(self, claude_api_key: str = None,
                 local_llm_url: str = "http://192.168.1.14:11434/api/chat"):
        super().__init__(claude_api_key, local_llm_url)
        self.logger = get_logger('Enhanced_LLM')

    def generate_answer_v3(self, query: str, chunks: List[ScoredChunk],
                          analysis: Dict = None, model: str = "claude") -> str:
        """
        Generate answer with enhanced prompt selection based on analysis.

        Args:
            query: User's question
            chunks: Retrieved ScoredChunks
            analysis: Query analysis dict
            model: LLM model to use

        Returns:
            Generated answer
        """
        self.logger.log(MAJOR, f"Generating V3 answer - model: {model}, chunks: {len(chunks)}")

        # Build context from chunks
        context = ContextBuilder.build_context(chunks, max_chars=25000)

        # Get appropriate prompt based on analysis
        prompt = PromptTemplates.get_prompt(
            query=query,
            context=context,
            analysis=analysis,
            entities=analysis.get('entities', []) if analysis else []
        )

        self.logger.log(DEBUG, f"Using {analysis.get('primary_intent', 'general')} prompt template")

        # Route to appropriate LLM
        if model == "claude":
            return self._generate_with_claude(prompt)
        else:
            return self._generate_with_ollama(prompt, model)

    def _generate_with_claude(self, prompt: str) -> str:
        """Generate with Claude API"""
        if not self.claude_client:
            self.logger.log(ERROR, "Claude API client not initialized")
            return "Claude model not available. Please check API key."

        try:
            self.logger.log(MINOR, "Sending request to Claude API")
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text
            self.logger.log(MAJOR, f"Claude response: {len(answer)} chars")
            return answer
        except Exception as e:
            self.logger.log(ERROR, f"Claude API error: {e}")
            return f"Error generating answer: {e}"

    def _generate_with_ollama(self, prompt: str, model: str) -> str:
        """Generate with local Ollama LLM"""
        try:
            self.logger.log(MINOR, f"Sending request to Ollama: {model}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = requests.post(self.local_llm_url, json=payload, timeout=120)
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "")
            self.logger.log(MAJOR, f"Ollama response: {len(answer)} chars")
            return answer
        except Exception as e:
            self.logger.log(ERROR, f"Ollama error: {e}")
            return f"Error with local LLM: {e}"


@dataclass
class RAGResponseV3:
    """Enhanced RAG response with more metadata"""
    answer: str
    sources: List[ScoredChunk]
    query: str
    retrieval_strategy: str
    query_analysis: Dict
    timestamp: datetime
    model_used: str
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class RAGOrchestratorV3:
    """
    Enhanced RAG orchestrator with hybrid retrieval and LLM query understanding.
    Main entry point for V3 RAG queries.
    """

    def __init__(self, neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 claude_api_key: str = None,
                 local_llm_url: str = "http://192.168.1.14:11434/api/chat",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize V3 RAG system with all components.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            claude_api_key: Anthropic API key
            local_llm_url: Ollama API endpoint
            embedding_model: Sentence-transformers model for embeddings
        """
        self.logger = get_logger('RAG_V3')
        self.logger.log(MAJOR, "Initializing RAG Orchestrator V3")

        # Neo4j connection
        self.neo4j_uri = neo4j_uri
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Initialize Cypher generator (needed for hybrid retriever)
        self.cypher_generator = CypherQueryGenerator(
            neo4j_driver=self.driver,
            local_llm_url=local_llm_url
        )

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            neo4j_driver=self.driver,
            cypher_generator=self.cypher_generator,
            embedding_model=embedding_model,
            local_llm_url=local_llm_url
        )

        # Initialize vector indexer (for setup/maintenance)
        self.vector_indexer = VectorIndexer(
            neo4j_driver=self.driver,
            embedding_model=embedding_model
        )

        # Initialize enhanced LLM integrator
        self.llm_integrator = EnhancedLLMIntegrator(
            claude_api_key=claude_api_key,
            local_llm_url=local_llm_url
        )

        # Store config
        self.local_llm_url = local_llm_url
        self.embedding_model = embedding_model

        self.logger.log(MAJOR, "RAG Orchestrator V3 ready")

    def close(self):
        """Clean up resources"""
        self.driver.close()
        self.logger.log(MINOR, "RAG Orchestrator V3 closed")

    def check_vector_index_status(self) -> Dict:
        """
        Check status of vector embeddings and index.

        Returns:
            Dict with status information
        """
        with_embeddings, total = self.vector_indexer.check_embeddings_exist()
        index_exists = self.vector_indexer.check_vector_index_exists()

        return {
            'total_chunks': total,
            'chunks_with_embeddings': with_embeddings,
            'embeddings_complete': with_embeddings == total,
            'vector_index_exists': index_exists,
            'ready_for_hybrid': index_exists and with_embeddings == total
        }

    def setup_vector_search(self, batch_size: int = 50) -> bool:
        """
        Setup vector search by creating embeddings and index.
        Call this once before using hybrid search.

        Args:
            batch_size: Number of chunks to process at once

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            self.logger.log(MAJOR, "Setting up vector search...")

            # Create embeddings
            self.logger.log(MAJOR, "Creating embeddings for chunks...")
            self.vector_indexer.create_embeddings_for_all_chunks(batch_size=batch_size)

            # Create vector index
            self.logger.log(MAJOR, "Creating vector index...")
            self.vector_indexer.create_vector_index()

            self.logger.log(MAJOR, "Vector search setup complete")
            return True

        except Exception as e:
            self.logger.log(ERROR, f"Vector search setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, question: str,
              model: str = "deepseek-r1:14b",
              use_hybrid: bool = True,
              use_vector: bool = True,
              use_graph: bool = True,
              use_query_expansion: bool = True,
              use_llm_analysis: bool = False,
              analysis_model: str = "deepseek-r1:14b",
              top_k: int = 6) -> RAGResponseV3:
        """
        Process RAG query with V3 enhancements.

        Args:
            question: User's question
            model: LLM model for answer generation
            use_hybrid: Use hybrid retrieval (vs graph-only fallback)
            use_vector: Include vector search in hybrid
            use_graph: Include graph search in hybrid
            use_query_expansion: Expand query with variations
            use_llm_analysis: Use LLM for query understanding
            analysis_model: Model for LLM analysis
            top_k: Number of chunks to retrieve

        Returns:
            RAGResponseV3 with answer and metadata
        """
        import time
        start_time = time.time()

        self.logger.log(MAJOR, f"RAG V3 query - model: {model}, hybrid: {use_hybrid}")
        self.logger.log(MINOR, f"Question: {question[:80]}...")

        # For MCQ: extract question only for retrieval (exclude choices to avoid noise)
        # Full question with choices will still be used for LLM generation
        retrieval_query, is_mcq = extract_mcq_question_only(question)
        if is_mcq:
            self.logger.log(MINOR, f"MCQ detected - using question only for retrieval: {retrieval_query[:60]}...")

        # Retrieval
        retrieval_start = time.time()

        if use_hybrid:
            # Check if vector search is available
            status = self.check_vector_index_status()
            if not status['ready_for_hybrid']:
                self.logger.log(ERROR, "Vector search not ready, falling back to graph-only")
                use_vector = False

            chunks, strategy, analysis = self.hybrid_retriever.retrieve(
                query=retrieval_query,  # Use question only for MCQ
                top_k=top_k,
                use_vector=use_vector,
                use_graph=use_graph,
                use_query_expansion=use_query_expansion,
                use_llm_analysis=use_llm_analysis,
                analysis_model=analysis_model
            )
        else:
            # Fallback to graph-only (V2 style)
            chunks, strategy, analysis = self._fallback_graph_retrieval(retrieval_query, top_k)

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Ensure MCQ intent is set if MCQ format was detected
        if is_mcq and analysis.get('primary_intent') != 'multiple_choice':
            self.logger.log(MINOR, "Overriding intent to 'multiple_choice' based on MCQ format detection")
            analysis['primary_intent'] = 'multiple_choice'

        # Handle empty results
        if not chunks:
            self.logger.log(ERROR, "No relevant information found")
            return RAGResponseV3(
                answer="No relevant information found in the knowledge base. Try rephrasing your question.",
                sources=[],
                query=question,
                retrieval_strategy=strategy,
                query_analysis=analysis,
                timestamp=datetime.now(),
                model_used=model,
                retrieval_time_ms=retrieval_time
            )

        # Generation
        generation_start = time.time()

        answer = self.llm_integrator.generate_answer_v3(
            query=question,
            chunks=chunks,
            analysis=analysis,
            model=model
        )

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        self.logger.log(MAJOR, f"RAG V3 complete - retrieval: {retrieval_time:.0f}ms, "
                              f"generation: {generation_time:.0f}ms, total: {total_time:.0f}ms")

        return RAGResponseV3(
            answer=answer,
            sources=chunks,
            query=question,
            retrieval_strategy=strategy,
            query_analysis=analysis,
            timestamp=datetime.now(),
            model_used=model,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time
        )

    def _fallback_graph_retrieval(self, question: str, top_k: int) -> Tuple[List[ScoredChunk], str, Dict]:
        """Fallback to graph-only retrieval (V2 style)"""
        self.logger.log(MINOR, "Using graph-only fallback retrieval")

        # Use rule-based analysis
        analysis = self.hybrid_retriever.query_analyzer._fallback_analysis(question)

        # Convert to expected format
        cypher_analysis = {
            'question_type': analysis.get('primary_intent', 'general'),
            'entities': [{'value': e, 'type': 'unknown', 'full_name': ''}
                        for e in analysis.get('entities', [])],
            'key_terms': analysis.get('key_terms', []),
            'complexity': analysis.get('complexity', 'simple'),
            'focus': 'content'
        }

        # Generate and execute Cypher query
        cypher_query = self.cypher_generator.generate_cypher_query(question, cypher_analysis)

        chunks = []
        with self.driver.session() as session:
            try:
                if cypher_query.startswith("COMPARISON_MULTI_STEP:"):
                    # Handle comparison
                    parts = cypher_query.split(":")
                    chunks = self._execute_comparison(parts[1], parts[2], session)
                else:
                    result = session.run(cypher_query)
                    for i, record in enumerate(result):
                        chunks.append(ScoredChunk(
                            chunk_id=record.get("chunk_id", ""),
                            spec_id=record.get("spec_id", ""),
                            section_id=record.get("section_id", ""),
                            section_title=record.get("section_title", ""),
                            content=record.get("content", ""),
                            chunk_type=record.get("chunk_type", ""),
                            complexity_score=record.get("complexity_score", 0.0),
                            key_terms=record.get("key_terms", []),
                            retrieval_score=1.0 - (i * 0.1),
                            retrieval_method='graph'
                        ))
            except Exception as e:
                self.logger.log(ERROR, f"Graph query failed: {e}")

        strategy = f"graph-only - {len(chunks)} chunks"
        return chunks[:top_k], strategy, analysis

    def _execute_comparison(self, entity1: str, entity2: str, session) -> List[ScoredChunk]:
        """Execute comparison retrieval"""
        chunks = []
        for entity in [entity1, entity2]:
            result = session.run("""
                MATCH (c:Chunk)
                WHERE toLower(c.content) CONTAINS toLower($entity)
                AND (c.chunk_type IN ['definition', 'architecture']
                     OR toLower(c.section_title) CONTAINS 'function'
                     OR toLower(c.section_title) CONTAINS 'overview')
                RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id,
                       c.section_id AS section_id, c.section_title AS section_title,
                       c.content AS content, c.chunk_type AS chunk_type,
                       c.complexity_score AS complexity_score, c.key_terms AS key_terms
                ORDER BY c.complexity_score ASC
                LIMIT 3
            """, entity=entity)

            for i, record in enumerate(result):
                chunks.append(ScoredChunk(
                    chunk_id=record['chunk_id'] or "",
                    spec_id=record['spec_id'] or "",
                    section_id=record['section_id'] or "",
                    section_title=record['section_title'] or "",
                    content=record['content'] or "",
                    chunk_type=record['chunk_type'] or "",
                    complexity_score=record['complexity_score'] or 0.0,
                    key_terms=record['key_terms'] or [],
                    retrieval_score=0.9 - (i * 0.1),
                    retrieval_method='graph'
                ))
        return chunks

    def explain_query(self, question: str, use_llm_analysis: bool = False,
                     analysis_model: str = "deepseek-r1:14b") -> Dict:
        """
        Explain the query strategy without executing (for debugging).

        Args:
            question: User's question
            use_llm_analysis: Use LLM for analysis
            analysis_model: Model for analysis

        Returns:
            Dict with analysis details
        """
        if use_llm_analysis:
            analysis = self.hybrid_retriever.query_analyzer.analyze(question, model=analysis_model)
        else:
            analysis = self.hybrid_retriever.query_analyzer._fallback_analysis(question)

        # Get query variations
        variations = self.hybrid_retriever.query_expander.expand(question)

        # Check vector status
        vector_status = self.check_vector_index_status()

        return {
            'question': question,
            'analysis': analysis,
            'query_variations': variations,
            'vector_status': vector_status,
            'recommended_strategy': 'hybrid' if vector_status['ready_for_hybrid'] else 'graph-only'
        }


class RAGConfigV3:
    """Configuration for V3 RAG system"""

    def __init__(self):
        self.neo4j_uri = "neo4j://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.claude_api_key = None
        self.local_llm_url = "http://192.168.1.14:11434/api/chat"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"


def create_rag_system_v3(claude_api_key: str = None,
                        local_llm_url: str = None,
                        embedding_model: str = None) -> RAGOrchestratorV3:
    """
    Factory function to create V3 RAG system.

    Args:
        claude_api_key: Anthropic API key
        local_llm_url: Ollama API endpoint
        embedding_model: Sentence-transformers model

    Returns:
        Configured RAGOrchestratorV3 instance
    """
    logger.log(MAJOR, "Creating RAG system V3")
    config = RAGConfigV3()

    if claude_api_key:
        config.claude_api_key = claude_api_key
    if local_llm_url:
        config.local_llm_url = local_llm_url
    if embedding_model:
        config.embedding_model = embedding_model

    return RAGOrchestratorV3(
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
        claude_api_key=config.claude_api_key,
        local_llm_url=config.local_llm_url,
        embedding_model=config.embedding_model
    )


if __name__ == "__main__":
    print("RAG System V3")
    print("=" * 60)
    print()
    print("Enhancements over V2:")
    print("- Hybrid Retrieval: Combines vector + graph search")
    print("- LLM Query Understanding: Better intent detection")
    print("- Query Expansion: More variations for better recall")
    print("- Enhanced Prompts: Specialized templates per intent")
    print()
    print("Usage:")
    print("  from rag_system_v3 import create_rag_system_v3")
    print()
    print("  # Create system")
    print("  rag = create_rag_system_v3()")
    print()
    print("  # First time setup (creates embeddings)")
    print("  rag.setup_vector_search()")
    print()
    print("  # Query")
    print("  response = rag.query('What is AMF?')")
    print("  print(response.answer)")
    print()
    print("  # Check status")
    print("  status = rag.check_vector_index_status()")
    print("  print(status)")
