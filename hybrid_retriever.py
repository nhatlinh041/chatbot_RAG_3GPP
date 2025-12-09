"""
Hybrid Retriever for 3GPP RAG System.
Combines Neo4j Vector Search with Graph-based Cypher queries.
Includes Vector Indexer, Vector Retriever, and Hybrid Retriever components.
"""
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from neo4j import GraphDatabase
import requests
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG

# Initialize logging
setup_centralized_logging()
logger = get_logger('Hybrid_Retriever')


@dataclass
class ScoredChunk:
    """Chunk with retrieval score and method info"""
    chunk_id: str
    spec_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    complexity_score: float
    key_terms: List[str]
    retrieval_score: float
    retrieval_method: str  # 'vector', 'graph', or 'vector+graph'
    reference_path: List[str] = field(default_factory=list)


class VectorIndexer:
    """
    Creates and manages vector embeddings for Neo4j chunks.
    Uses sentence-transformers for embedding generation.
    Stores embeddings in Neo4j for vector search.
    """

    def __init__(self, neo4j_driver, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize indexer with Neo4j driver and embedding model.

        Args:
            neo4j_driver: Neo4j driver instance
            embedding_model: Model name for sentence-transformers
        """
        self.driver = neo4j_driver
        self.logger = get_logger('Vector_Indexer')
        self.embedding_model_name = embedding_model
        self.model = None
        self.embedding_dim = None

    def _load_model(self):
        """Lazy load embedding model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.logger.log(MAJOR, f"Loading embedding model: {self.embedding_model_name}")
                self.model = SentenceTransformer(self.embedding_model_name)
                # Get embedding dimension
                test_embedding = self.model.encode("test")
                self.embedding_dim = len(test_embedding)
                self.logger.log(MAJOR, f"Model loaded - embedding dimension: {self.embedding_dim}")
            except ImportError:
                self.logger.log(ERROR, "sentence-transformers not installed. Run: pip install sentence-transformers")
                raise

    def check_vector_index_exists(self) -> bool:
        """Check if vector index already exists in Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                SHOW INDEXES
                YIELD name, type
                WHERE type = 'VECTOR' AND name = 'chunk_embeddings'
                RETURN count(*) > 0 AS exists
            """)
            record = result.single()
            return record['exists'] if record else False

    def check_embeddings_exist(self) -> Tuple[int, int]:
        """
        Check how many chunks have embeddings vs total chunks.

        Returns:
            Tuple of (chunks_with_embeddings, total_chunks)
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                RETURN
                    count(CASE WHEN c.embedding IS NOT NULL THEN 1 END) AS with_embeddings,
                    count(*) AS total
            """)
            record = result.single()
            return (record['with_embeddings'], record['total']) if record else (0, 0)

    def create_embeddings_for_all_chunks(self, batch_size: int = 50):
        """
        Create embeddings for all chunks that don't have one.

        Args:
            batch_size: Number of chunks to process at once
        """
        self._load_model()

        with self.driver.session() as session:
            # Get chunks without embeddings
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.embedding IS NULL
                RETURN c.chunk_id AS chunk_id,
                       c.content AS content,
                       c.section_title AS section_title
                ORDER BY c.chunk_id
            """)
            chunks = list(result)

        total_chunks = len(chunks)
        if total_chunks == 0:
            self.logger.log(MAJOR, "All chunks already have embeddings")
            return

        self.logger.log(MAJOR, f"Creating embeddings for {total_chunks} chunks")

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]

            # Prepare texts for embedding (content + section_title for better context)
            texts = []
            for chunk in batch:
                text = f"{chunk['section_title']}: {chunk['content'][:1000]}"
                texts.append(text)

            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)

            # Store in Neo4j
            with self.driver.session() as session:
                for chunk, embedding in zip(batch, embeddings):
                    session.run("""
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        SET c.embedding = $embedding
                    """, chunk_id=chunk['chunk_id'], embedding=embedding.tolist())

            processed = min(i + batch_size, total_chunks)
            self.logger.log(MINOR, f"Processed {processed}/{total_chunks} chunks")

        self.logger.log(MAJOR, f"Completed embedding creation for {total_chunks} chunks")

    def create_vector_index(self):
        """
        Create Neo4j vector index for similarity search.
        Requires Neo4j 5.11+ with vector index support.
        """
        self._load_model()

        # Check if index already exists
        if self.check_vector_index_exists():
            self.logger.log(MINOR, "Vector index 'chunk_embeddings' already exists")
            return

        self.logger.log(MAJOR, f"Creating vector index with dimension {self.embedding_dim}")

        with self.driver.session() as session:
            # Create vector index
            session.run(f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk)
                ON c.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {self.embedding_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)

        self.logger.log(MAJOR, "Vector index created successfully")

    def drop_vector_index(self):
        """Drop existing vector index"""
        with self.driver.session() as session:
            session.run("DROP INDEX chunk_embeddings IF EXISTS")
        self.logger.log(MINOR, "Vector index dropped")

    def clear_all_embeddings(self):
        """Remove all embeddings from chunks"""
        with self.driver.session() as session:
            session.run("MATCH (c:Chunk) REMOVE c.embedding")
        self.logger.log(MINOR, "All embeddings cleared")


class VectorRetriever:
    """
    Performs vector similarity search using Neo4j vector index.
    """

    def __init__(self, neo4j_driver, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retriever with Neo4j driver.

        Args:
            neo4j_driver: Neo4j driver instance
            embedding_model: Model name - must match the one used for indexing
        """
        self.driver = neo4j_driver
        self.logger = get_logger('Vector_Retriever')
        self.embedding_model_name = embedding_model
        self.model = None

    def _load_model(self):
        """Lazy load embedding model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.logger.log(MINOR, f"Loading embedding model: {self.embedding_model_name}")
                self.model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                self.logger.log(ERROR, "sentence-transformers not installed")
                raise

    def search(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        """
        Perform vector similarity search.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of ScoredChunk with vector similarity scores
        """
        self._load_model()

        # Generate query embedding
        query_embedding = self.model.encode(query)

        self.logger.log(MINOR, f"Vector search for: {query[:50]}...")

        # Search using Neo4j vector index
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id,
                       node.spec_id AS spec_id,
                       node.section_id AS section_id,
                       node.section_title AS section_title,
                       node.content AS content,
                       node.chunk_type AS chunk_type,
                       node.complexity_score AS complexity_score,
                       node.key_terms AS key_terms,
                       score
                ORDER BY score DESC
                LIMIT $top_k
            """, query_vector=query_embedding.tolist(), top_k=top_k)

            chunks = []
            for record in result:
                chunks.append(ScoredChunk(
                    chunk_id=record['chunk_id'] or "",
                    spec_id=record['spec_id'] or "",
                    section_id=record['section_id'] or "",
                    section_title=record['section_title'] or "",
                    content=record['content'] or "",
                    chunk_type=record['chunk_type'] or "",
                    complexity_score=record['complexity_score'] or 0.0,
                    key_terms=record['key_terms'] or [],
                    retrieval_score=record['score'],
                    retrieval_method='vector'
                ))

            self.logger.log(MAJOR, f"Vector search returned {len(chunks)} chunks")
            return chunks


class SemanticQueryAnalyzer:
    """
    Analyzes queries using LLM for better understanding.
    Extracts intents, entities, and sub-questions.
    """

    def __init__(self, local_llm_url: str = "http://192.168.1.14:11434/api/chat", term_dict: Dict = None):
        """
        Initialize analyzer with LLM endpoint.

        Args:
            local_llm_url: Ollama API endpoint
            term_dict: Dictionary of known terms (abbreviation -> full_name)
        """
        self.local_llm_url = local_llm_url
        self.term_dict = term_dict or {}
        self.logger = get_logger('Query_Analyzer')

    def analyze(self, query: str, model: str = "deepseek-r1:14b") -> Dict:
        """
        Analyze query using LLM to extract structured information.

        Args:
            query: User's question
            model: LLM model to use

        Returns:
            Analysis dict with intents, entities, complexity, etc.
        """
        self.logger.log(MINOR, f"Analyzing query: {query[:60]}...")

        # Build prompt
        prompt = self._build_analysis_prompt(query)

        # Call LLM
        try:
            response = self._call_llm(prompt, model)
            analysis = self._parse_response(response)
            self.logger.log(MAJOR, f"Analysis: intent={analysis.get('primary_intent')}, "
                                  f"entities={analysis.get('entities', [])}")
            return analysis
        except Exception as e:
            self.logger.log(ERROR, f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(query)

    def _build_analysis_prompt(self, query: str) -> str:
        """Build analysis prompt for LLM"""
        # Sample terms for context
        sample_terms = list(self.term_dict.items())[:20]
        terms_str = "\n".join([f"- {abbr}: {info['full_name']}" for abbr, info in sample_terms])

        return f"""Analyze this 3GPP telecommunications question and return a JSON object.

Question: "{query}"

Known 3GPP Terms (sample):
{terms_str}

Extract:
1. primary_intent: One of [definition, comparison, procedure, reference, network_function, relationship, specification, multiple_choice, general]
2. entities: List of 3GPP entities mentioned (abbreviations or full names)
3. key_terms: Important technical terms
4. complexity: One of [simple, medium, complex]
5. requires_multi_step: true if needs multiple retrieval steps (e.g., comparison)
6. sub_questions: If complex, break into simpler sub-questions

Return ONLY valid JSON:
{{
    "primary_intent": "...",
    "entities": ["AMF", "SMF"],
    "key_terms": ["registration", "procedure"],
    "complexity": "medium",
    "requires_multi_step": false,
    "sub_questions": []
}}"""

    def _call_llm(self, prompt: str, model: str) -> str:
        """Call local LLM"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.1
        }

        response = requests.post(self.local_llm_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response"""
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # If parsing fails, return fallback
        raise ValueError("Could not parse LLM response as JSON")

    def _fallback_analysis(self, query: str) -> Dict:
        """Rule-based fallback analysis"""
        query_lower = query.lower()

        # Detect intent
        if any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs']):
            intent = 'comparison'
        elif any(w in query_lower for w in ['what is', 'define', 'definition']):
            intent = 'definition'
        elif any(w in query_lower for w in ['how', 'procedure', 'process', 'steps']):
            intent = 'procedure'
        elif any(w in query_lower for w in ['role', 'function', 'responsibility']):
            intent = 'network_function'
        else:
            intent = 'general'

        # Extract entities from known terms
        entities = []
        for abbr in self.term_dict.keys():
            if re.search(r'\b' + re.escape(abbr) + r'\b', query, re.IGNORECASE):
                entities.append(abbr)

        return {
            'primary_intent': intent,
            'entities': entities,
            'key_terms': re.findall(r'\b\w{3,}\b', query_lower),
            'complexity': 'medium' if len(entities) > 1 else 'simple',
            'requires_multi_step': intent == 'comparison' and len(entities) >= 2,
            'sub_questions': []
        }


class QueryExpander:
    """
    Expands queries with variations to improve retrieval.
    """

    def __init__(self, term_dict: Dict = None):
        """
        Initialize expander with term dictionary.

        Args:
            term_dict: Dictionary mapping abbreviations to full names
        """
        self.term_dict = term_dict or {}
        self.logger = get_logger('Query_Expander')

        # Common synonyms for 3GPP domain
        self.synonyms = {
            'role': ['function', 'responsibility', 'duty', 'purpose'],
            'function': ['role', 'responsibility', 'purpose'],
            'procedure': ['process', 'flow', 'steps', 'mechanism'],
            'difference': ['comparison', 'distinguish', 'contrast'],
            'interaction': ['communication', 'interface', 'relationship'],
        }

    def expand(self, query: str, max_variations: int = 4) -> List[str]:
        """
        Generate query variations.

        Args:
            query: Original query
            max_variations: Maximum number of variations to return

        Returns:
            List of query variations including original
        """
        variations = [query]  # Original first

        # 1. Expand abbreviations
        expanded = self._expand_abbreviations(query)
        if expanded != query:
            variations.append(expanded)

        # 2. Add synonym variations
        for word, syns in self.synonyms.items():
            if word in query.lower():
                for syn in syns[:2]:  # Limit synonyms
                    var = re.sub(r'\b' + word + r'\b', syn, query, flags=re.IGNORECASE)
                    if var != query and var not in variations:
                        variations.append(var)

        # 3. Extract keywords only
        keywords = self._extract_keywords(query)
        if keywords and keywords != query:
            variations.append(keywords)

        # Remove duplicates and limit
        seen = set()
        unique = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique.append(v)

        self.logger.log(MINOR, f"Generated {len(unique)} query variations")
        return unique[:max_variations]

    def _expand_abbreviations(self, query: str) -> str:
        """Expand abbreviations to full names"""
        expanded = query
        for abbr, info in self.term_dict.items():
            full_name = info.get('full_name', info) if isinstance(info, dict) else info
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                # Replace with "abbr (full_name)" for better matching
                expanded = re.sub(pattern, f"{abbr} {full_name}", expanded, flags=re.IGNORECASE)
        return expanded

    def _extract_keywords(self, query: str) -> str:
        """Extract main keywords, remove stop words"""
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'does', 'do',
                    'can', 'which', 'where', 'when', 'why', 'in', 'of', 'to', 'for'}
        words = query.split()
        keywords = [w for w in words if w.lower() not in stopwords]
        return ' '.join(keywords)


class HybridRetriever:
    """
    Combines vector search and graph-based Cypher queries.
    Merges and reranks results from both sources.
    """

    def __init__(self, neo4j_driver, cypher_generator,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 local_llm_url: str = "http://192.168.1.14:11434/api/chat"):
        """
        Initialize hybrid retriever.

        Args:
            neo4j_driver: Neo4j driver instance
            cypher_generator: CypherQueryGenerator instance for graph queries
            embedding_model: Model for vector embeddings
            local_llm_url: Ollama API endpoint for query analysis
        """
        self.driver = neo4j_driver
        self.cypher_generator = cypher_generator
        self.logger = get_logger('Hybrid_Retriever')

        # Initialize components
        self.vector_retriever = VectorRetriever(neo4j_driver, embedding_model)
        self.query_analyzer = SemanticQueryAnalyzer(
            local_llm_url=local_llm_url,
            term_dict=getattr(cypher_generator, 'all_terms', {})
        )
        self.query_expander = QueryExpander(
            term_dict=getattr(cypher_generator, 'all_terms', {})
        )

        self.logger.log(MAJOR, "Hybrid Retriever initialized")

    def retrieve(self, query: str, top_k: int = 6,
                 use_vector: bool = True,
                 use_graph: bool = True,
                 use_query_expansion: bool = True,
                 use_llm_analysis: bool = False,
                 analysis_model: str = "deepseek-r1:14b") -> Tuple[List[ScoredChunk], str, Dict]:
        """
        Perform hybrid retrieval combining vector and graph search.

        Args:
            query: User's question
            top_k: Number of final results
            use_vector: Enable vector search
            use_graph: Enable graph-based search
            use_query_expansion: Enable query expansion
            use_llm_analysis: Use LLM for query analysis
            analysis_model: Model for LLM analysis

        Returns:
            Tuple of (scored_chunks, strategy_description, analysis)
        """
        self.logger.log(MAJOR, f"Hybrid retrieval - vector:{use_vector}, graph:{use_graph}, "
                              f"expansion:{use_query_expansion}, llm_analysis:{use_llm_analysis}")

        # Analyze query
        if use_llm_analysis:
            analysis = self.query_analyzer.analyze(query, model=analysis_model)
        else:
            analysis = self.query_analyzer._fallback_analysis(query)

        # Expand query if enabled
        query_variations = [query]
        if use_query_expansion:
            query_variations = self.query_expander.expand(query)

        # Collect results from both sources
        all_results = {}  # chunk_id -> ScoredChunk with aggregated scores

        # Vector search
        if use_vector:
            for q_var in query_variations:
                vector_results = self._safe_vector_search(q_var, top_k=10)
                self._merge_results(all_results, vector_results, source='vector')

        # Graph search
        if use_graph:
            graph_results = self._execute_graph_search(query, analysis, top_k=10)
            self._merge_results(all_results, graph_results, source='graph')

        # Rerank and select top results
        final_results = self._rerank(all_results, top_k=top_k)

        # Build strategy description
        methods = []
        if use_vector:
            methods.append("vector")
        if use_graph:
            methods.append("graph")
        strategy = f"hybrid ({'+'.join(methods)}) - {len(final_results)} chunks"

        self.logger.log(MAJOR, f"Hybrid retrieval complete: {strategy}")
        return final_results, strategy, analysis

    def _safe_vector_search(self, query: str, top_k: int) -> List[ScoredChunk]:
        """Perform vector search with error handling"""
        try:
            return self.vector_retriever.search(query, top_k=top_k)
        except Exception as e:
            self.logger.log(ERROR, f"Vector search failed: {e}")
            return []

    def _execute_graph_search(self, query: str, analysis: Dict, top_k: int) -> List[ScoredChunk]:
        """Execute graph-based Cypher search"""
        try:
            # Convert analysis to format expected by cypher_generator
            cypher_analysis = {
                'question_type': analysis.get('primary_intent', 'general'),
                'entities': [{'value': e, 'type': 'unknown', 'full_name': ''}
                           for e in analysis.get('entities', [])],
                'key_terms': analysis.get('key_terms', []),
                'complexity': analysis.get('complexity', 'simple'),
                'focus': 'content'
            }

            # Generate Cypher query
            cypher_query = self.cypher_generator.generate_cypher_query(query, cypher_analysis)

            # Check for comparison multi-step marker
            if cypher_query.startswith("COMPARISON_MULTI_STEP:"):
                return self._execute_comparison_retrieval(cypher_query)

            # Execute query
            with self.driver.session() as session:
                result = session.run(cypher_query)

                chunks = []
                for i, record in enumerate(result):
                    # Score based on position (higher for earlier results)
                    score = 1.0 - (i / (top_k * 2))

                    chunks.append(ScoredChunk(
                        chunk_id=record.get("chunk_id", ""),
                        spec_id=record.get("spec_id", ""),
                        section_id=record.get("section_id", ""),
                        section_title=record.get("section_title", ""),
                        content=record.get("content", ""),
                        chunk_type=record.get("chunk_type", ""),
                        complexity_score=record.get("complexity_score", 0.0),
                        key_terms=record.get("key_terms", []),
                        retrieval_score=score,
                        retrieval_method='graph',
                        reference_path=record.get("referenced_specs", [])
                    ))

                self.logger.log(MINOR, f"Graph search returned {len(chunks)} chunks")
                return chunks[:top_k]

        except Exception as e:
            self.logger.log(ERROR, f"Graph search failed: {e}")
            return []

    def _execute_comparison_retrieval(self, marker: str) -> List[ScoredChunk]:
        """Handle multi-step comparison retrieval"""
        parts = marker.split(":")
        if len(parts) < 3:
            return []

        entity1, entity2 = parts[1], parts[2]
        self.logger.log(MINOR, f"Comparison retrieval: {entity1} vs {entity2}")

        chunks = []
        with self.driver.session() as session:
            for entity in [entity1, entity2]:
                # Get chunks about this entity
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE toLower(c.content) CONTAINS toLower($entity)
                    AND (
                        toLower(c.section_title) CONTAINS 'function'
                        OR toLower(c.section_title) CONTAINS 'overview'
                        OR c.chunk_type IN ['definition', 'architecture']
                    )
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

    def _merge_results(self, all_results: Dict[str, ScoredChunk],
                       new_results: List[ScoredChunk], source: str):
        """Merge new results into all_results, boosting duplicates"""
        for chunk in new_results:
            chunk_id = chunk.chunk_id

            if chunk_id in all_results:
                # Chunk found in multiple sources - boost score
                existing = all_results[chunk_id]

                # Combine scores
                if source == 'vector':
                    existing.retrieval_score += chunk.retrieval_score * 0.5
                else:
                    existing.retrieval_score += chunk.retrieval_score * 0.5

                # Update method
                if source not in existing.retrieval_method:
                    existing.retrieval_method = 'vector+graph'
            else:
                all_results[chunk_id] = chunk

    def _rerank(self, all_results: Dict[str, ScoredChunk], top_k: int) -> List[ScoredChunk]:
        """Rerank merged results and return top_k"""

        # Apply boost for chunks found in multiple sources
        for chunk in all_results.values():
            if chunk.retrieval_method == 'vector+graph':
                chunk.retrieval_score *= 1.3  # 30% boost for multi-source match

        # Sort by score
        ranked = sorted(all_results.values(),
                       key=lambda x: x.retrieval_score,
                       reverse=True)

        return ranked[:top_k]


# Factory function for easy initialization
def create_hybrid_retriever(neo4j_uri: str = "neo4j://localhost:7687",
                           neo4j_user: str = "neo4j",
                           neo4j_password: str = "password",
                           cypher_generator=None,
                           embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                           local_llm_url: str = "http://192.168.1.14:11434/api/chat"):
    """
    Factory function to create HybridRetriever with all components.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        cypher_generator: CypherQueryGenerator instance (optional, will create if None)
        embedding_model: Sentence-transformers model name
        local_llm_url: Ollama API endpoint

    Returns:
        Tuple of (HybridRetriever, Neo4j driver)
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Create cypher generator if not provided
    if cypher_generator is None:
        from rag_system_v2 import CypherQueryGenerator
        cypher_generator = CypherQueryGenerator(neo4j_driver=driver, local_llm_url=local_llm_url)

    retriever = HybridRetriever(
        neo4j_driver=driver,
        cypher_generator=cypher_generator,
        embedding_model=embedding_model,
        local_llm_url=local_llm_url
    )

    return retriever, driver


if __name__ == "__main__":
    print("Hybrid Retriever Module")
    print("=" * 50)
    print("Components:")
    print("- VectorIndexer: Creates embeddings and Neo4j vector index")
    print("- VectorRetriever: Semantic similarity search")
    print("- SemanticQueryAnalyzer: LLM-based query understanding")
    print("- QueryExpander: Query variations for better recall")
    print("- HybridRetriever: Combines vector + graph search")
    print()
    print("Usage:")
    print("  from hybrid_retriever import create_hybrid_retriever, VectorIndexer")
    print("  retriever, driver = create_hybrid_retriever()")
    print()
    print("  # First time: Create embeddings and index")
    print("  indexer = VectorIndexer(driver)")
    print("  indexer.create_embeddings_for_all_chunks()")
    print("  indexer.create_vector_index()")
    print()
    print("  # Then search")
    print("  chunks, strategy, analysis = retriever.retrieve('What is AMF?')")
