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
from subject_classifier import SubjectClassifier, Subject

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
    subject: str = ""  # Subject classification (Lexicon, Standards specifications, etc.)
    subject_confidence: float = 0.0


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

        self.logger.log(MINOR, f"Vector search for: {query}...")

        # Search using Neo4j vector index (including subject)
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
                       node.subject AS subject,
                       node.subject_confidence AS subject_confidence,
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
                    retrieval_method='vector',
                    subject=record['subject'] or "",
                    subject_confidence=record['subject_confidence'] or 0.0
                ))

            self.logger.log(MAJOR, f"Vector search returned {len(chunks)} chunks")
            return chunks


class TermDefinitionResolver:
    """
    Resolves abbreviations to authoritative definitions from Neo4j Term nodes.
    Provides ground truth definitions before retrieval to prevent hallucination.
    """

    def __init__(self, neo4j_driver):
        """
        Initialize resolver with Neo4j driver.

        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.driver = neo4j_driver
        self.logger = get_logger('Term_Resolver')

    def resolve_terms(self, entities: List[str]) -> Dict[str, Dict]:
        """
        Lookup term definitions from Neo4j Term nodes.

        Args:
            entities: List of abbreviations (e.g., ['SCP', 'SEPP'])

        Returns:
            Dict mapping abbreviation to definition info:
            {
                'SCP': {
                    'full_name': 'Service Communication Proxy',
                    'specs': ['TS 23.501', 'TS 29.500'],
                    'source': 'term_node'
                }
            }
        """
        if not entities:
            return {}

        definitions = {}

        with self.driver.session() as session:
            for entity in entities:
                # Clean entity (remove punctuation, uppercase)
                clean_entity = entity.strip().upper()

                try:
                    result = session.run("""
                        MATCH (t:Term {abbreviation: $abbrev})
                        OPTIONAL MATCH (t)-[:DEFINED_IN]->(d:Document)
                        RETURN t.abbreviation AS abbrev,
                               t.full_name AS full_name,
                               collect(DISTINCT d.spec_id) AS specs
                    """, abbrev=clean_entity)

                    record = result.single()
                    if record:
                        definitions[entity] = {
                            'full_name': record['full_name'],
                            'specs': [s for s in record['specs'] if s],
                            'source': 'term_node'
                        }
                        self.logger.log(MAJOR,
                            f"Resolved {entity} = {record['full_name']}")
                    else:
                        self.logger.log(MINOR,
                            f"No Term node found for {entity}")
                except Exception as e:
                    self.logger.log(ERROR, f"Failed to resolve {entity}: {e}")

        return definitions


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
   - Use "multiple_choice" if question has options like "A. xxx", "B. xxx", "C. xxx", "D. xxx"
2. entities: List of 3GPP entities mentioned (abbreviations or full names)
3. key_terms: Important technical terms
4. complexity: One of [simple, medium, complex]
5. requires_multi_step: true if needs multiple retrieval steps (e.g., comparison)
6. needs_term_resolution: true if question asks about definitions/meanings/stand-for OR compares entities (to ensure correct full names)
7. sub_questions: If complex, break into simpler sub-questions

Return ONLY valid JSON:
{{
    "primary_intent": "...",
    "entities": ["AMF", "SMF"],
    "key_terms": ["registration", "procedure"],
    "complexity": "medium",
    "requires_multi_step": false,
    "needs_term_resolution": true,
    "sub_questions": []
}}

Examples:
- "What is AMF?" → primary_intent: "definition", needs_term_resolution: true
- "Compare SCP and SEPP" → primary_intent: "comparison", needs_term_resolution: true
- "Explain registration procedure" → primary_intent: "procedure", needs_term_resolution: false
- "What is AUSF responsible for?\\nA. Access management\\nB. Authentication\\nC. Session\\nD. User plane" → primary_intent: "multiple_choice" (has A/B/C/D options)
- Question with "choices": ["option1", "option2", "option3", "option4"] → primary_intent: "multiple_choice" (JSON choices format)
- "Which of the following is correct? (A) option1 (B) option2" → primary_intent: "multiple_choice" (inline format)"""

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

    def _detect_mcq_format(self, query: str) -> bool:
        """
        Comprehensively detect if query is a Multiple Choice Question.

        Supports multiple formats:
        1. "A. option" / "A) option" format (traditional MCQ)
        2. JSON-style with "choices" array
        3. Inline options like "(A) xxx (B) xxx"
        4. Numbered with letters "1. A) option"

        Args:
            query: The question text

        Returns:
            True if query appears to be MCQ format
        """
        # Pattern 1: Traditional MCQ format with newlines
        # Matches: A. option, A) option, a. option, a) option
        traditional_pattern = re.search(
            r'\n\s*[A-Da-d][\.\)]\s*\S+',  # Line starting with A./A) etc
            query
        )
        if traditional_pattern:
            # Count how many options (need at least 2 for MCQ)
            option_count = len(re.findall(r'\n\s*[A-Da-d][\.\)]\s*\S+', query))
            if option_count >= 2:
                self.logger.log(DEBUG, f"MCQ detected: traditional format ({option_count} options)")
                return True

        # Pattern 2: JSON-style "choices" array detection
        # Matches: "choices": [...] or 'choices': [...]
        choices_pattern = re.search(
            r'["\']?choices["\']?\s*[:\=]\s*\[',
            query,
            re.IGNORECASE
        )
        if choices_pattern:
            self.logger.log(DEBUG, "MCQ detected: JSON choices format")
            return True

        # Pattern 3: Inline options (A) xxx (B) xxx (C) xxx
        inline_pattern = re.findall(
            r'\([A-Da-d]\)\s*[^()]+',
            query
        )
        if len(inline_pattern) >= 3:  # At least A, B, C
            self.logger.log(DEBUG, f"MCQ detected: inline format ({len(inline_pattern)} options)")
            return True

        # Pattern 4: Options at start of line without newline separator
        # Matches multiple A: xxx B: xxx patterns
        colon_pattern = re.findall(
            r'\b[A-Da-d]\s*[:\-]\s*\w+',
            query
        )
        if len(colon_pattern) >= 3:
            self.logger.log(DEBUG, f"MCQ detected: colon format ({len(colon_pattern)} options)")
            return True

        # Pattern 5: Detect by keywords suggesting options
        mcq_keywords = [
            r'\bwhich\s+(?:of\s+the\s+)?following\b',
            r'\bselect\s+(?:the\s+)?(?:correct|best)\b',
            r'\bchoose\s+(?:the\s+)?(?:correct|best)\b',
            r'\bpick\s+(?:the\s+)?(?:correct|right)\b',
        ]
        for pattern in mcq_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                # Also need at least some options present
                has_options = re.search(r'[A-Da-d][\.\)\:]\s*\w', query)
                if has_options:
                    self.logger.log(DEBUG, "MCQ detected: keyword + options pattern")
                    return True

        return False

    def _fallback_analysis(self, query: str) -> Dict:
        """Rule-based fallback analysis"""
        query_lower = query.lower()

        # Detect intent with enhanced patterns
        # FIRST: Check for multiple choice format using comprehensive detection
        is_mcq = self._detect_mcq_format(query)
        if is_mcq:
            intent = 'multiple_choice'
            self.logger.log(MINOR, "Detected multiple choice question format")
        else:
            # Comparison detection with comprehensive patterns
            comparison_patterns = [
                'compare', 'comparison', 'difference', 'differences',
                'versus', ' vs ', 'differ', 'differs', 'different',
                'distinguish', 'distinguishes', 'differentiate', 'differentiates'
            ]
            # Check for comparison patterns including "how does X differ" type
            is_comparison = any(w in query_lower for w in comparison_patterns)
            # Also catch "X vs Y" or "between X and Y" patterns
            if not is_comparison and (' and ' in query_lower or ' vs ' in query_lower):
                # Check for comparison context words
                if any(w in query_lower for w in ['main', 'key', 'primary', 'major']):
                    is_comparison = True

            if is_comparison:
                intent = 'comparison'
            elif any(w in query_lower for w in ['what is', 'what does', 'define', 'definition', 'stand for', 'stands for']):
                # Check if it's actually a procedure question (e.g., "what is the first step")
                if any(w in query_lower for w in ['step', 'procedure', 'process']):
                    intent = 'procedure'
                else:
                    intent = 'definition'
            elif any(w in query_lower for w in ['how', 'procedure', 'process', 'steps']):
                intent = 'procedure'
            elif any(w in query_lower for w in ['role', 'function', 'responsibility']):
                intent = 'network_function'
            else:
                intent = 'general'

        # Extract entities from known terms
        entities = []
        # Common words that should not be considered entities
        COMMON_WORDS = {'IN', 'IS', 'ON', 'AT', 'TO', 'OF', 'A', 'AN', 'OR', 'IT', 'AS', 'BY', 'NO', 'SO'}

        for abbr in self.term_dict.keys():
            if re.search(r'\b' + re.escape(abbr) + r'\b', query, re.IGNORECASE):
                # Filter out common words
                if abbr.upper() not in COMMON_WORDS:
                    entities.append(abbr)

        # Decide if term resolution needed
        needs_resolution = intent in ['definition', 'comparison', 'network_function']

        return {
            'primary_intent': intent,
            'entities': entities,
            'key_terms': re.findall(r'\b\w{3,}\b', query_lower),
            'complexity': 'medium' if len(entities) > 1 else 'simple',
            'requires_multi_step': intent == 'comparison' and len(entities) >= 2,
            'needs_term_resolution': needs_resolution,
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
        Generate DIVERSE query variations to reduce overlap.

        Strategy:
        - Original query (general definition)
        - Practical/use case variation
        - Comparative variation
        - Contextual variation

        Args:
            query: Original query
            max_variations: Maximum number of variations to return

        Returns:
            List of diverse query variations
        """
        query_lower = query.lower()
        variations = [query]  # Original first

        # Detect query intent for smarter expansion
        is_definition = any(word in query_lower for word in ['what is', 'what does', 'define', 'stand for'])
        is_comparison = any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus'])
        is_procedure = any(word in query_lower for word in ['how', 'procedure', 'process', 'steps'])

        # Strategy 1: DIVERSIFY instead of making similar variations
        if is_definition:
            # Instead of: "What is X?", "Define X", "X definition" (too similar!)
            # Use: General + Practical + Comparative + Contextual

            # Extract entity (e.g., "AMF" from "What is AMF?")
            entities = self._extract_entities(query)
            if entities:
                entity = entities[0]
                # Variation 1: Original (already in list)
                # Variation 2: Practical/Use case
                variations.append(f"{entity} use cases in 5G")
                # Variation 3: Comparative
                variations.append(f"{entity} vs similar network functions")
                # Variation 4: Contextual
                variations.append(f"{entity} role in network architecture")

        elif is_comparison:
            # Already comparing - add related aspects
            entities = self._extract_entities(query)
            if len(entities) >= 2:
                e1, e2 = entities[0], entities[1]
                # Variation 1: Original
                # Variation 2: Functional aspect
                variations.append(f"{e1} and {e2} functions")
                # Variation 3: Interaction
                variations.append(f"how {e1} and {e2} interact")
                # Variation 4: Architecture
                variations.append(f"{e1} {e2} in 5G architecture")

        elif is_procedure:
            # Procedure query - add steps and context
            # Variation 1: Original
            # Variation 2: Step-by-step
            variations.append(query.replace("how", "steps for").replace("?", ""))
            # Variation 3: Sequence
            variations.append(query.replace("how", "sequence of").replace("?", ""))
            # Variation 4: Message flow
            if "procedure" in query_lower:
                variations.append(query.replace("procedure", "message flow"))

        else:
            # General query - use old strategy with improvements
            # 1. Expand abbreviations
            expanded = self._expand_abbreviations(query)
            if expanded != query:
                variations.append(expanded)

            # 2. Add ONE synonym variation (not all)
            added_synonym = False
            for word, syns in self.synonyms.items():
                if word in query_lower and not added_synonym:
                    syn = syns[0]  # Take only first synonym
                    var = re.sub(r'\b' + word + r'\b', syn, query, flags=re.IGNORECASE)
                    if var != query:
                        variations.append(var)
                        added_synonym = True
                        break

            # 3. Keywords only (last resort)
            keywords = self._extract_keywords(query)
            if keywords and keywords != query:
                variations.append(keywords)

        # Remove duplicates and limit
        seen = set()
        unique = []
        for v in variations:
            v_clean = v.strip()
            if v_clean and v_clean not in seen:
                seen.add(v_clean)
                unique.append(v_clean)

        self.logger.log(MINOR, f"Generated {len(unique)} DIVERSE query variations")
        return unique[:max_variations]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities (abbreviations) from query"""
        entities = []
        for abbr in self.term_dict.keys():
            if re.search(r'\b' + re.escape(abbr) + r'\b', query, re.IGNORECASE):
                entities.append(abbr)
        return entities

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
        self.term_resolver = TermDefinitionResolver(neo4j_driver)
        self.query_analyzer = SemanticQueryAnalyzer(
            local_llm_url=local_llm_url,
            term_dict=getattr(cypher_generator, 'all_terms', {})
        )
        self.query_expander = QueryExpander(
            term_dict=getattr(cypher_generator, 'all_terms', {})
        )
        # Subject classifier for subject-aware retrieval
        self.subject_classifier = SubjectClassifier()

        self.logger.log(MAJOR, "Hybrid Retriever initialized with subject-aware retrieval")

    def retrieve(self, query: str, top_k: int = 6,
                 use_vector: bool = True,
                 use_graph: bool = True,
                 use_query_expansion: bool = True,
                 use_llm_analysis: bool = True,
                 use_subject_boost: bool = True,
                 use_semantic_dedup: bool = False,
                 dedup_threshold: float = 0.85,
                 analysis_model: str = "deepseek-r1:14b") -> Tuple[List[ScoredChunk], str, Dict]:
        """
        Perform hybrid retrieval combining vector and graph search.

        Args:
            query: User's question
            top_k: Number of final results
            use_vector: Enable vector search
            use_graph: Enable graph-based search
            use_query_expansion: Enable query expansion (with diversification)
            use_llm_analysis: Use LLM for query analysis
            use_subject_boost: Enable subject-aware score boosting
            use_semantic_dedup: Use semantic similarity for deduplication (slower, more accurate)
            dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
            analysis_model: Model for LLM analysis

        Returns:
            Tuple of (scored_chunks, strategy_description, analysis)
        """
        dedup_method = "semantic" if use_semantic_dedup else "Jaccard"
        self.logger.log(MAJOR, f"Hybrid retrieval - vector:{use_vector}, graph:{use_graph}, "
                              f"expansion:{use_query_expansion}, llm_analysis:{use_llm_analysis}, "
                              f"subject_boost:{use_subject_boost}, dedup:{dedup_method}")

        # Analyze query
        if use_llm_analysis:
            analysis = self.query_analyzer.analyze(query, model=analysis_model)
        else:
            analysis = self.query_analyzer._fallback_analysis(query)

        # ALWAYS check for MCQ format and override intent if detected
        # This ensures MCQ questions always use the correct prompt template
        mcq_pattern = re.search(r'\n\s*[A-Da-d][\.\)]\s*\w+', query)
        if mcq_pattern and analysis.get('primary_intent') != 'multiple_choice':
            self.logger.log(MINOR, f"Overriding intent to 'multiple_choice' (detected MCQ format)")
            analysis['primary_intent'] = 'multiple_choice'

        # Detect expected subjects for score boosting
        expected_subjects = []
        if use_subject_boost:
            expected_subjects = self.subject_classifier.detect_query_subjects(query)
            if expected_subjects:
                subject_names = [(s.value, w) for s, w in expected_subjects]
                self.logger.log(MINOR, f"Expected subjects: {subject_names}")
                analysis['expected_subjects'] = subject_names

        # NEW: Resolve term definitions ONLY if LLM/analysis says it's needed
        needs_resolution = analysis.get('needs_term_resolution', False)
        term_definitions = {}

        if needs_resolution:
            entities = analysis.get('entities', [])
            term_definitions = self.term_resolver.resolve_terms(entities)
            if term_definitions:
                self.logger.log(MAJOR, f"Resolved {len(term_definitions)} term definitions (LLM requested)")
        else:
            self.logger.log(MINOR, "Skipping term resolution (not needed for this query)")

        analysis['term_definitions'] = term_definitions

        # Expand query if enabled
        query_variations = [query]
        if use_query_expansion:
            query_variations = self.query_expander.expand(query)

        # Traditional merging approach
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

        # Rerank and select top results with subject boosting and deduplication
        final_results = self._rerank(
            all_results,
            top_k=top_k,
            expected_subjects=expected_subjects,
            use_semantic_dedup=use_semantic_dedup,
            dedup_threshold=dedup_threshold
        )

        # Build strategy description
        methods = []
        if use_vector:
            methods.append("vector")
        if use_graph:
            methods.append("graph")
        if use_subject_boost and expected_subjects:
            methods.append("subject-boost")
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
                        reference_path=record.get("referenced_specs", []),
                        subject=record.get("subject", ""),
                        subject_confidence=record.get("subject_confidence", 0.0)
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
                # Get chunks about this entity (including subject)
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
                           c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                           c.subject AS subject, c.subject_confidence AS subject_confidence
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
                        retrieval_method='graph',
                        subject=record['subject'] or "",
                        subject_confidence=record['subject_confidence'] or 0.0
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

    def _rerank(self, all_results: Dict[str, ScoredChunk], top_k: int,
                expected_subjects: List[Tuple[Subject, float]] = None,
                use_semantic_dedup: bool = False,
                dedup_threshold: float = 0.85) -> List[ScoredChunk]:
        """
        Rerank merged results with subject boosting, content deduplication, and return top_k

        Args:
            all_results: Dict of chunk_id -> ScoredChunk
            top_k: Number of results to return
            expected_subjects: List of (Subject, weight) tuples for boosting
            use_semantic_dedup: Use semantic similarity for deduplication (slower but more accurate)
            dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
        """
        # Apply boost for chunks found in multiple sources
        for chunk in all_results.values():
            if chunk.retrieval_method == 'vector+graph':
                chunk.retrieval_score *= 1.3  # 30% boost for multi-source match

        # Apply subject-based boosting if expected subjects provided
        if expected_subjects:
            for chunk in all_results.values():
                if chunk.subject:
                    boost = self.subject_classifier.get_subject_boost(
                        chunk.subject, expected_subjects
                    )
                    if boost > 1.0:
                        chunk.retrieval_score *= boost
                        self.logger.log(DEBUG, f"Subject boost {boost:.2f}x for {chunk.chunk_id} "
                                              f"(subject: {chunk.subject})")

        # Sort by score first
        ranked = sorted(all_results.values(),
                       key=lambda x: x.retrieval_score,
                       reverse=True)

        # Content-based deduplication: remove chunks with very similar content
        deduplicated = self._deduplicate_by_content(
            ranked,
            similarity_threshold=dedup_threshold,
            use_semantic=use_semantic_dedup
        )

        self.logger.log(MINOR, f"Deduplication: {len(ranked)} -> {len(deduplicated)} chunks "
                              f"(removed {len(ranked) - len(deduplicated)} duplicates)")

        return deduplicated[:top_k]

    def _deduplicate_by_content(self, chunks: List[ScoredChunk],
                                similarity_threshold: float = 0.85,
                                use_semantic: bool = False) -> List[ScoredChunk]:
        """
        Remove chunks with very similar content, keeping highest-scored ones.

        Supports two modes:
        1. Jaccard similarity (fast, default)
        2. Semantic similarity (accurate, slower)

        Args:
            chunks: List of ScoredChunk sorted by score (descending)
            similarity_threshold: Threshold for considering content as duplicate (0.0-1.0)
            use_semantic: Use semantic similarity (embeddings) instead of Jaccard

        Returns:
            Deduplicated list of chunks
        """
        if len(chunks) <= 1:
            return chunks

        method = "semantic" if use_semantic else "Jaccard"
        self.logger.log(MINOR, f"Deduplication using {method} similarity")

        unique_chunks = []
        seen_contents = []

        for chunk in chunks:
            content = chunk.content.lower().strip()

            # Check if this content is too similar to any already seen
            is_duplicate = False
            for seen_content in seen_contents:
                similarity = self._calculate_content_similarity(
                    content, seen_content, use_semantic=use_semantic
                )
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    self.logger.log(DEBUG,
                        f"Skipping duplicate chunk {chunk.chunk_id} "
                        f"({method} similarity: {similarity:.2f})")
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_contents.append(content)

        return unique_chunks

    def _calculate_content_similarity(self, content1: str, content2: str,
                                       use_semantic: bool = False) -> float:
        """
        Calculate similarity between two text contents.

        Supports two modes:
        1. Jaccard (fast, word-level) - default
        2. Semantic (accurate, embedding-based) - optional

        Args:
            content1: First text content
            content2: Second text content
            use_semantic: Use semantic similarity (embeddings) instead of Jaccard

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if use_semantic:
            return self._semantic_similarity(content1, content2)
        else:
            return self._jaccard_similarity(content1, content2)

    def _jaccard_similarity(self, content1: str, content2: str) -> float:
        """
        Jaccard similarity: Fast, word-level matching.

        Args:
            content1: First text content
            content2: Second text content

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use word-level tokenization
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard: intersection / union
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _semantic_similarity(self, content1: str, content2: str) -> float:
        """
        Semantic similarity using embeddings: More accurate semantic matching.

        Uses existing embedding model from vector retriever.

        Args:
            content1: First text content
            content2: Second text content

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Ensure model is loaded
            self.vector_retriever._load_model()
            model = self.vector_retriever.model

            # Generate embeddings
            embed1 = model.encode(content1)
            embed2 = model.encode(content2)

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(embed1, embed2))
            magnitude1 = sum(a * a for a in embed1) ** 0.5
            magnitude2 = sum(b * b for b in embed2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            cosine_sim = dot_product / (magnitude1 * magnitude2)

            # Normalize to [0, 1] range (cosine is [-1, 1])
            normalized = (cosine_sim + 1) / 2

            return normalized

        except Exception as e:
            self.logger.log(ERROR, f"Semantic similarity failed: {e}, falling back to Jaccard")
            return self._jaccard_similarity(content1, content2)


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
        from rag_core import CypherQueryGenerator
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
