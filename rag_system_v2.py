"""
RAG System V2 for 3GPP Knowledge Graph.
Provides dynamic Cypher query generation and LLM-powered Q&A.
"""
import json
import re
from typing import List, Dict, Optional, Set
from neo4j import GraphDatabase
from pathlib import Path
import anthropic
import requests
from dataclasses import dataclass
from datetime import datetime
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG
from cypher_sanitizer import CypherSanitizer, create_safe_cypher_query

# Initialize logging
setup_centralized_logging()
logger = get_logger('RAG_System')


@dataclass
class RetrievedChunk:
    chunk_id: str
    spec_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    complexity_score: float
    key_terms: List[str]
    reference_path: List[str]


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievedChunk]
    query: str
    cypher_query: str
    retrieval_strategy: str
    timestamp: datetime


class LLMIntegrator:
    """
    Unified LLM integrator supporting API-based (Claude) and local LLM (Ollama) models.
    Routes requests to appropriate backend based on model parameter.
    """

    def __init__(self, claude_api_key: str = None, local_llm_url: str = "http://192.168.1.14:11434/api/chat"):
        """Initialize LLM clients for both API and local models"""
        self.logger = get_logger('LLM_Integrator')
        self.logger.log(MAJOR, "Initializing unified LLM integrator")

        # API-based LLM (Claude)
        self.claude_client = None
        if claude_api_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
            self.logger.log(MINOR, "Claude API client initialized")

        # Local LLM endpoint
        self.local_llm_url = local_llm_url
        self.logger.log(MINOR, f"Local LLM URL: {local_llm_url}")
        self.logger.log(MAJOR, "LLM integrator ready")

    def generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk], cypher_query: str,
                       model: str = "claude") -> str:
        """
        Generate answer using specified LLM model.
        Routes to API or local LLM based on model parameter.
        """
        self.logger.log(MAJOR, f"Generating answer - model: {model}, chunks: {len(retrieved_chunks)}")

        # Prepare context from chunks
        context = self._prepare_context(retrieved_chunks)
        self.logger.log(DEBUG, f"Context length: {len(context)} chars")

        # Route to appropriate handler
        if model == "claude":
            return self._generate_with_api(query, context, cypher_query)
        else:
            return self._generate_with_local_llm(query, context, cypher_query, model)

    def _prepare_context(self, retrieved_chunks: List[RetrievedChunk], max_chars: int = 25000) -> str:
        """Build context string from retrieved chunks with size limit"""
        context_parts = []
        total_chars = 0

        for chunk in retrieved_chunks:
            # Truncate long content
            content = chunk.content
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"

            chunk_text = f"""
**Source: {chunk.spec_id} - {chunk.section_title} (Type: {chunk.chunk_type})**
**Section ID: {chunk.section_id} | Complexity: {chunk.complexity_score:.2f}**
**Key Terms: {', '.join(chunk.key_terms[:5])}**

{content}
---
"""
            # Check size limit
            if total_chars + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n".join(context_parts)

    def _is_multiple_choice(self, query: str) -> bool:
        """Detect multiple choice question patterns"""
        indicators = [
            'it authorizes', 'it selects', 'it stores', 'it routes', 'it notifies',
            'which of the following', 'select the correct', 'choose the'
        ]
        return any(indicator in query.lower() for indicator in indicators)

    def _generate_with_api(self, query: str, context: str, cypher_query: str) -> str:
        """Generate answer using Claude API"""
        if not self.claude_client:
            self.logger.log(ERROR, "Claude API client not initialized")
            return "Claude model not available. Please check API key configuration."

        is_multiple_choice = self._is_multiple_choice(query)
        self.logger.log(DEBUG, f"Question type: {'multiple_choice' if is_multiple_choice else 'general'}")

        # Build prompt based on question type
        if is_multiple_choice:
            prompt = f"""You are an expert on 3GPP specifications. This is a multiple choice question. Based on the provided context from 3GPP documents, select the correct answer and explain why.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Query Strategy Used:** {cypher_query[:200]}...

**Instructions:**
1. Identify which statement is correct based on the provided context
2. Provide a clear explanation referencing the relevant specification sections
3. If multiple statements seem correct, explain the primary/main role or function
4. Use the exact wording from the context when possible
5. Reference the specific specifications and sections that support your answer

**IMPORTANT: Format your response using Markdown:**
- Use **bold** for key terms and network function names
- Use bullet points for lists
- Use `code` formatting for technical identifiers
- Include a "Sources" section at the end with specification references

**Answer:**"""
        else:
            prompt = f"""You are an expert on 3GPP specifications. Based on the provided context from 3GPP documents, please answer the following question:

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Database Query Strategy:** The system used a {cypher_query[:100]}... query to retrieve this information.

**Instructions:**
1. Provide a comprehensive answer based ONLY on the provided context
2. Reference specific sections and specifications when relevant
3. Explain the complexity and key terms when helpful
4. Use technical terminology appropriately

**CRITICAL - Anti-Hallucination Rules:**
- ONLY use information explicitly stated in the provided context
- If the context does not contain relevant information about the question, respond with: "I could not find relevant information about [topic] in the retrieved 3GPP specifications. Please try rephrasing your question or ask about a different topic."
- DO NOT make up, infer, or hallucinate any information not present in the context
- DO NOT use your general knowledge about 3GPP - ONLY use the provided context
- If you're unsure whether something is in the context, err on the side of saying you don't have that information

**IMPORTANT: Format your response using Markdown:**
- Use headings (## or ###) to structure your answer
- Use **bold** for key terms, network function names (AMF, SMF, UPF, etc.)
- Use bullet points or numbered lists for procedures/steps
- Use `code` formatting for technical identifiers, parameters, or protocols
- For complex relationships, you may include a Mermaid diagram using ```mermaid code block
- Include a "## Sources" section at the end listing the specification references (e.g., TS 23.501, TS 29.500)

**Answer:**"""

        # Call Claude API
        try:
            self.logger.log(MINOR, "Sending request to Claude API")
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text
            self.logger.log(MAJOR, f"Claude response received - {len(answer)} chars")
            return answer

        except Exception as e:
            self.logger.log(ERROR, f"Claude API error: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _generate_with_local_llm(self, query: str, context: str, cypher_query: str, model: str) -> str:
        """Generate answer using local LLM via Ollama"""
        is_multiple_choice = self._is_multiple_choice(query)

        # Build prompt based on question type
        if is_multiple_choice:
            prompt = f"""You are an expert on 3GPP specifications. This is a multiple choice question. Based on the provided context from 3GPP documents, select the correct answer and explain why.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Identify which statement is correct based on the provided context
2. Provide a clear explanation referencing the relevant specification sections
3. Use the exact wording from the context when possible

**Format your response using Markdown:**
- Use **bold** for key terms
- Use bullet points for explanations
- Include specification references at the end

**Answer:**"""
        else:
            prompt = f"""You are an expert on 3GPP specifications. Based on the provided context from 3GPP documents, please answer the following question:

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Provide a comprehensive answer based ONLY on the provided context
2. Reference specific sections and specifications when relevant
3. Use technical terminology appropriately

**CRITICAL - Anti-Hallucination Rules:**
- ONLY use information explicitly stated in the provided context above
- If the context does not contain relevant information about the question, respond with: "I could not find relevant information about [topic] in the retrieved 3GPP specifications."
- DO NOT make up, infer, or hallucinate any information not present in the context
- DO NOT use your general knowledge about 3GPP - ONLY use the provided context
- If unsure whether something is in the context, say you don't have that information

**Format your response using Markdown:**
- Use headings (## or ###) to structure your answer
- Use **bold** for key terms and network function names
- Use bullet points or numbered lists for procedures/steps
- Use `code` formatting for technical identifiers
- For complex relationships, include a Mermaid diagram using ```mermaid code block
- Include a "## Sources" section at the end with specification references

**Answer:**"""

        # Call local LLM
        try:
            self.logger.log(MINOR, f"Sending request to local LLM: {model}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }

            response = requests.post(self.local_llm_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            answer = result.get("message", {}).get("content", "No response generated")
            self.logger.log(MAJOR, f"Local LLM response received - {len(answer)} chars")
            return answer

        except requests.exceptions.RequestException as e:
            self.logger.log(ERROR, f"Local LLM connection error: {str(e)}")
            return f"Error connecting to local LLM: {str(e)}"
        except Exception as e:
            self.logger.log(ERROR, f"Local LLM error: {str(e)}")
            return f"Error generating answer with local LLM: {str(e)}"


class CypherQueryGenerator:
    """
    Generates Cypher queries based on question analysis.
    Supports 8 question types: definition, comparison, procedure, reference,
    network_function, relationship, specification, multiple_choice.

    Supports two modes:
    - Rule-based (default): Fast, deterministic query generation
    - LLM-based: Uses LocalLM to analyze and generate queries dynamically
    """

    def __init__(self, neo4j_driver=None, local_llm_url: str = None):
        """
        Initialize query patterns and load entities from KG.

        Args:
            neo4j_driver: Optional Neo4j driver for dynamic entity loading from Term nodes
            local_llm_url: Optional URL for local LLM (Ollama) for LLM-based query generation
        """
        self.logger = get_logger('Cypher_Generator')
        self.driver = neo4j_driver
        self.local_llm_url = local_llm_url or "http://192.168.1.14:11434/api/chat"

        # Cache KG schema for LLM context
        self._kg_schema_cache = None

        # Query pattern handlers
        self.query_patterns = {
            'definition': self._generate_definition_query,
            'comparison': self._generate_comparison_query,
            'procedure': self._generate_procedure_query,
            'reference': self._generate_reference_query,
            'network_function': self._generate_network_function_query,
            'relationship': self._generate_relationship_query,
            'specification': self._generate_specification_query,
            'multiple_choice': self._generate_multiple_choice_query
        }

        # Load entities dynamically from KG or use fallback
        if self.driver:
            self._load_entities_from_kg()
        else:
            # Fallback for testing without Neo4j - common 5G Core NFs
            self.logger.log(MINOR, "No Neo4j driver provided, using fallback terms")
            self.all_terms = {
                'AMF': {'full_name': 'Access and Mobility Management Function', 'type': 'network_function'},
                'SMF': {'full_name': 'Session Management Function', 'type': 'network_function'},
                'UPF': {'full_name': 'User Plane Function', 'type': 'network_function'},
                'NRF': {'full_name': 'Network Repository Function', 'type': 'network_function'},
                'PCF': {'full_name': 'Policy Control Function', 'type': 'network_function'},
                'UDM': {'full_name': 'Unified Data Management', 'type': 'network_function'},
                'AUSF': {'full_name': 'Authentication Server Function', 'type': 'network_function'},
                'NSSF': {'full_name': 'Network Slice Selection Function', 'type': 'network_function'},
                'NEF': {'full_name': 'Network Exposure Function', 'type': 'network_function'},
                'SCP': {'full_name': 'Service Communication Proxy', 'type': 'network_function'},
                'SEPP': {'full_name': 'Security Edge Protection Proxy', 'type': 'network_function'},
            }

    def _load_entities_from_kg(self):
        """Load all terms from Neo4j Term nodes"""
        self.logger.log(MAJOR, "Loading entities from Knowledge Graph Term nodes")

        query = """
        MATCH (t:Term)
        RETURN t.abbreviation AS abbrev,
               t.full_name AS full_name
        ORDER BY t.abbreviation
        """

        self.all_terms = {}

        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    abbrev = record['abbrev']
                    full_name = record['full_name']

                    # Infer type from full_name keywords
                    entity_type = self._infer_entity_type(full_name)

                    self.all_terms[abbrev] = {
                        'full_name': full_name,
                        'type': entity_type
                    }

            self.logger.log(MAJOR, f"Loaded {len(self.all_terms)} terms from KG")

        except Exception as e:
            self.logger.log(ERROR, f"Failed to load entities from KG: {e}")
            self.logger.log(MINOR, "Falling back to minimal term list")
            # Use minimal fallback if KG query fails
            self.all_terms = {
                'AMF': {'full_name': 'Access and Mobility Management Function', 'type': 'network_function'},
                'SMF': {'full_name': 'Session Management Function', 'type': 'network_function'},
            }

    def _infer_entity_type(self, full_name: str) -> str:
        """Infer entity type from full name keywords"""
        full_name_lower = full_name.lower()

        # Network functions usually have "function" in name
        if 'function' in full_name_lower or 'management' in full_name_lower:
            return 'network_function'
        # Protocols
        elif 'protocol' in full_name_lower:
            return 'protocol'
        # Procedures/processes
        elif 'procedure' in full_name_lower or 'process' in full_name_lower:
            return 'procedure'
        # Proxy, Server, Gateway entities
        elif any(word in full_name_lower for word in ['proxy', 'server', 'gateway']):
            return 'network_function'
        # Default
        else:
            return 'concept'

    def _get_kg_schema(self) -> str:
        """
        Get KG schema description for LLM context.
        Caches the result to avoid repeated queries.
        """
        if self._kg_schema_cache:
            return self._kg_schema_cache

        schema = """
## Neo4j Knowledge Graph Schema

### Node Types:
1. **Document** - 3GPP specification documents
   - Properties: spec_id, title, version, release, series
   - Example: spec_id="TS 23.501", title="System architecture for 5G System"

2. **Chunk** - Content segments from documents
   - Properties: chunk_id, spec_id, section_id, section_title, content, chunk_type, complexity_score, key_terms
   - chunk_type values: definition, procedure, architecture, reference, general
   - Example: section_title="5.2.1 AMF Overview", chunk_type="definition"

3. **Term** - Abbreviations and technical terms
   - Properties: abbreviation, full_name, source_specs
   - Example: abbreviation="SCP", full_name="Service Communication Proxy"

### Relationships:
1. (Document)-[:CONTAINS]->(Chunk) - Document contains chunks
2. (Chunk)-[:REFERENCES_SPEC]->(Document) - Cross-spec references
3. (Chunk)-[:REFERENCES_CHUNK]->(Chunk) - Within-doc references
4. (Term)-[:DEFINED_IN]->(Document) - Term source specification

### Available Term Samples (from KG):
"""
        # Add sample terms from the loaded terms
        if self.all_terms:
            sample_terms = list(self.all_terms.items())[:15]
            for abbrev, info in sample_terms:
                schema += f"- {abbrev}: {info['full_name']} (type: {info['type']})\n"

        self._kg_schema_cache = schema
        return schema

    def _call_local_llm(self, prompt: str, model: str = "deepseek-r1:7b") -> str:
        """Call local LLM and return response"""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = requests.post(self.local_llm_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            self.logger.log(ERROR, f"Local LLM call failed: {e}")
            return ""

    def analyze_question_with_llm(self, question: str, model: str = "deepseek-r1:7b") -> Dict[str, any]:
        """
        Analyze question using LocalLM for better understanding.
        Falls back to rule-based analysis if LLM fails.
        """
        self.logger.log(MINOR, f"LLM-based analysis for: {question[:60]}...")

        kg_schema = self._get_kg_schema()

        prompt = f"""You are a 3GPP telecommunications expert analyzing a user question for a knowledge graph query system.

{kg_schema}

### Question to Analyze:
"{question}"

### Your Task:
Analyze the question and return a JSON object with:
1. question_type: One of [definition, comparison, procedure, reference, network_function, relationship, specification, general]
2. entities: List of detected 3GPP entities (abbreviations or full names) with their types
3. key_terms: Important technical terms from the question
4. complexity: One of [simple, medium, complex]
5. focus: What aspect to focus on [content, relationship, procedure, architecture]

### Response Format (JSON only, no explanation):
```json
{{
    "question_type": "comparison",
    "entities": [
        {{"value": "SCP", "type": "network_function", "full_name": "Service Communication Proxy"}},
        {{"value": "SEPP", "type": "network_function", "full_name": "Security Edge Protection Proxy"}}
    ],
    "key_terms": ["difference", "between", "proxy"],
    "complexity": "medium",
    "focus": "content"
}}
```

Return ONLY the JSON object, no other text."""

        llm_response = self._call_local_llm(prompt, model)

        # Parse LLM response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = llm_response.strip()

            analysis = json.loads(json_str)
            self.logger.log(MAJOR, f"LLM analysis: type={analysis.get('question_type')}, entities={len(analysis.get('entities', []))}")
            return analysis

        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.log(ERROR, f"Failed to parse LLM response, falling back to rule-based: {e}")
            return self.analyze_question(question)

    def generate_cypher_with_llm(self, question: str, analysis: Dict, model: str = "deepseek-r1:7b") -> str:
        """
        Generate Cypher query using LocalLM based on question analysis.
        Returns a valid Cypher query tailored to the KG structure.
        """
        self.logger.log(MINOR, f"LLM-based Cypher generation for: {question[:60]}...")

        kg_schema = self._get_kg_schema()

        # Cypher examples for different question types
        cypher_examples = """
### Cypher Query Examples:

1. **Definition Query** (What is X?):
```cypher
OPTIONAL MATCH (t:Term) WHERE t.abbreviation = 'AMF'
WITH t, CASE WHEN t IS NOT NULL THEN t.full_name ELSE 'AMF' END AS search_term
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS toLower(search_term)
AND (toLower(c.section_title) CONTAINS 'definition' OR c.chunk_type = 'definition')
RETURN c.chunk_id, c.spec_id, c.section_title, c.content, c.chunk_type
ORDER BY c.complexity_score ASC LIMIT 5
```

2. **Comparison Query** (What is the difference between X and Y?):
```cypher
// Get chunks for both entities
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS 'scp' OR toLower(c.content) CONTAINS 'sepp'
AND (c.chunk_type IN ['definition', 'architecture'])
RETURN c.chunk_id, c.spec_id, c.section_title, c.content, c.chunk_type
ORDER BY c.complexity_score ASC LIMIT 8
```

3. **Procedure Query** (How does X work?):
```cypher
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS 'registration'
AND (c.chunk_type = 'procedure' OR toLower(c.section_title) CONTAINS 'procedure')
RETURN c.chunk_id, c.spec_id, c.section_title, c.content
ORDER BY c.complexity_score DESC LIMIT 6
```

4. **Relationship Query** (How do X and Y interact?):
```cypher
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS 'amf' AND toLower(c.content) CONTAINS 'smf'
AND (toLower(c.section_title) CONTAINS 'interface' OR c.chunk_type = 'architecture')
RETURN c.chunk_id, c.spec_id, c.section_title, c.content
LIMIT 6
```
"""

        prompt = f"""You are a Neo4j Cypher expert for a 3GPP knowledge graph.

{kg_schema}

{cypher_examples}

### Question to Query:
"{question}"

### Question Analysis:
{json.dumps(analysis, indent=2)}

### Your Task:
Generate a Cypher query that:
1. Uses the correct node types (Document, Chunk, Term)
2. Searches for the identified entities in content and section_title
3. Filters by appropriate chunk_type based on question_type
4. Returns: chunk_id, spec_id, section_id, section_title, content, chunk_type, complexity_score, key_terms
5. Orders by complexity_score and limits to 6-8 results
6. Uses OPTIONAL MATCH for Term resolution when entities are abbreviations

### IMPORTANT:
- Use toLower() for case-insensitive matching
- Always include proper WHERE clauses
- Return only the Cypher query, no explanation
- Query must be syntactically correct

### Cypher Query:
```cypher
"""

        llm_response = self._call_local_llm(prompt, model)

        # Extract Cypher from response
        try:
            # Handle markdown code blocks
            cypher_match = re.search(r'```cypher\s*(.*?)\s*```', llm_response, re.DOTALL)
            if cypher_match:
                cypher_query = cypher_match.group(1).strip()
            else:
                # Try to get raw query (assume response is the query)
                cypher_query = llm_response.strip()
                # Remove any leading explanation text
                if 'MATCH' in cypher_query:
                    cypher_query = cypher_query[cypher_query.find('MATCH'):]
                elif 'OPTIONAL' in cypher_query:
                    cypher_query = cypher_query[cypher_query.find('OPTIONAL'):]

            # Validate query has basic structure
            if not any(keyword in cypher_query.upper() for keyword in ['MATCH', 'RETURN']):
                raise ValueError("Generated query missing MATCH or RETURN")

            self.logger.log(MAJOR, f"LLM generated query: {cypher_query[:100]}...")
            return cypher_query

        except Exception as e:
            self.logger.log(ERROR, f"Failed to generate Cypher with LLM, falling back to rule-based: {e}")
            return self.generate_cypher_query(question, analysis)

    def analyze_question(self, question: str, use_llm: bool = False, llm_model: str = "deepseek-r1:7b") -> Dict[str, any]:
        """
        Analyze question to determine type, extract entities and key terms.
        Uses dynamic entity extraction from KG Term nodes.

        Args:
            question: User's question
            use_llm: If True, uses LocalLM for analysis (more accurate but slower)
            llm_model: Model to use for LLM-based analysis

        Returns:
            Analysis dict with question_type, entities, key_terms, complexity.
        """
        # Use LLM-based analysis if requested
        if use_llm:
            return self.analyze_question_with_llm(question, llm_model)

        self.logger.log(MINOR, f"Analyzing question: {question[:80]}...")
        question_lower = question.lower()

        analysis = {
            'question_type': 'general',
            'entities': [],
            'key_terms': [],
            'complexity': 'simple',
            'focus': 'content'
        }

        # Detect question type by keywords
        # FIX: Check comparison BEFORE definition to handle "What is the difference..."
        if any(word in question_lower for word in ['compare', 'different', 'difference', 'versus', 'vs', 'vs.', 'between']):
            analysis['question_type'] = 'comparison'
        elif any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning']):
            analysis['question_type'] = 'definition'
        elif any(word in question_lower for word in ['how', 'procedure', 'process', 'steps']):
            analysis['question_type'] = 'procedure'
        elif any(word in question_lower for word in ['reference', 'refer', 'mentioned', 'cited']):
            analysis['question_type'] = 'reference'
        elif any(word in question_lower for word in ['role of', 'function of', 'responsibility']):
            analysis['question_type'] = 'network_function'
        elif any(word in question_lower for word in ['relationship', 'interact', 'communication']):
            analysis['question_type'] = 'relationship'
        elif any(word in question_lower for word in ['specification', 'document', 'standard']):
            analysis['question_type'] = 'specification'
        elif self._is_multiple_choice(question):
            analysis['question_type'] = 'multiple_choice'

        # Extract entities dynamically from KG terms
        analysis['entities'] = self._extract_entities_from_kg(question)

        # Extract key terms
        analysis['key_terms'] = self._extract_key_terms(question)

        # Determine complexity
        if len(analysis['entities']) > 2 or any(word in question_lower for word in ['complex', 'detailed', 'comprehensive']):
            analysis['complexity'] = 'complex'
        elif len(analysis['entities']) > 1 or any(word in question_lower for word in ['relationship', 'interaction']):
            analysis['complexity'] = 'medium'

        self.logger.log(MAJOR, f"Analysis: type={analysis['question_type']}, entities={len(analysis['entities'])}, complexity={analysis['complexity']}")
        return analysis

    def _extract_entities_from_kg(self, question: str) -> List[Dict]:
        """
        Extract entities by matching question against KG terms.
        Uses word boundaries for accurate matching.
        """
        question_upper = question.upper()
        question_lower = question.lower()
        entities = []
        matched_abbreviations = set()  # Avoid duplicates

        # Common English words that should NOT be treated as abbreviations
        # even if they exist in the KG (e.g., "IN" = Intelligent Network)
        common_word_blocklist = {
            'IN', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IS', 'IT',
            'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE', 'A', 'I',
            'FOR', 'THE', 'AND', 'BUT', 'NOT', 'ARE', 'WAS', 'HAS', 'HAD',
            'CAN', 'MAY', 'ALL', 'ANY', 'NEW', 'OLD', 'ONE', 'TWO', 'WAY'
        }

        for abbrev, info in self.all_terms.items():
            # Skip if already matched
            if abbrev in matched_abbreviations:
                continue

            # Skip common English words to avoid false positives
            # (e.g., "IN" from "in 5G Core" should not match "Intelligent Network")
            if abbrev in common_word_blocklist:
                continue

            # Match abbreviation (case-insensitive, whole word only)
            # Use word boundaries to avoid partial matches (e.g., "AM" in "AMF")
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, question_upper, re.IGNORECASE):
                entities.append({
                    'type': info['type'],
                    'value': abbrev,
                    'full_name': info['full_name']
                })
                matched_abbreviations.add(abbrev)

            # Also try matching full name (case-insensitive)
            elif info['full_name'].lower() in question_lower:
                entities.append({
                    'type': info['type'],
                    'value': abbrev,
                    'full_name': info['full_name']
                })
                matched_abbreviations.add(abbrev)

        return entities

    def _is_multiple_choice(self, question: str) -> bool:
        """Detect multiple choice question patterns"""
        indicators = [
            'it authorizes', 'it selects', 'it stores', 'it routes', 'it notifies',
            'which of the following', 'select the correct', 'choose the'
        ]
        return any(indicator in question.lower() for indicator in indicators)

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract meaningful technical terms from question"""
        common_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'when', 'where', 'why', 'in', 'of', 'to', 'for'}
        words = re.findall(r'\b\w+\b', question.lower())
        return [word for word in words if len(word) > 2 and word not in common_words][:10]

    def generate_cypher_query(self, question: str, analysis: Dict, use_llm: bool = False, llm_model: str = "deepseek-r1:7b") -> str:
        """
        Generate appropriate Cypher query based on question analysis.
        Selects query pattern handler based on question_type.

        Args:
            question: User's question
            analysis: Question analysis dict
            use_llm: If True, uses LocalLM for query generation (more flexible but slower)
            llm_model: Model to use for LLM-based generation
        """
        # Use LLM-based generation if requested
        if use_llm:
            return self.generate_cypher_with_llm(question, analysis, llm_model)

        question_type = analysis['question_type']
        self.logger.log(MINOR, f"Generating {question_type} query")

        if question_type in self.query_patterns:
            query = self.query_patterns[question_type](question, analysis)
        else:
            query = self._generate_general_query(question, analysis)

        self.logger.log(DEBUG, f"Generated query: {query[:100]}...")
        return query

    def _generate_definition_query(self, question: str, analysis: Dict) -> str:
        """
        Generate query for definition questions using Term nodes.
        Multi-stage retrieval:
        1. Resolve abbreviation via Term nodes to get full name and source specs
        2. Search for content in relevant specs with full name
        3. Fallback to general search if no Term found
        """
        analysis = CypherSanitizer.sanitize_question_analysis(analysis)
        entities = analysis['entities']
        key_terms = analysis['key_terms']

        if entities:
            entity = CypherSanitizer.sanitize_search_term(entities[0]['value'])
            # Multi-stage query using Term nodes for term resolution
            return f"""
            // Stage 1: Resolve term via Term nodes
            OPTIONAL MATCH (t:Term)
            WHERE t.abbreviation = '{entity}'
               OR toLower(t.abbreviation) = toLower('{entity}')
            WITH t,
                 CASE WHEN t IS NOT NULL THEN t.full_name ELSE '{entity}' END AS search_term,
                 CASE WHEN t IS NOT NULL THEN t.source_specs ELSE [] END AS target_specs,
                 CASE WHEN t IS NOT NULL THEN t.full_name ELSE null END AS resolved_full_name

            // Stage 2: Search for definition content
            MATCH (c:Chunk)
            WHERE (
                // Search using both abbreviation and full name
                toLower(c.content) CONTAINS toLower('{entity}')
                OR (resolved_full_name IS NOT NULL AND toLower(c.content) CONTAINS toLower(resolved_full_name))
            )
            AND (
                // Prioritize definition/overview sections or sections containing the full term name
                toLower(c.section_title) CONTAINS 'definition'
                OR toLower(c.section_title) CONTAINS 'overview'
                OR toLower(c.section_title) CONTAINS 'introduction'
                OR toLower(c.section_title) CONTAINS 'general'
                OR (resolved_full_name IS NOT NULL AND toLower(c.section_title) CONTAINS toLower(resolved_full_name))
                OR c.chunk_type = 'definition'
                OR c.chunk_type = 'architecture'
            )

            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   resolved_full_name AS resolved_term, target_specs AS term_sources
            ORDER BY
                // Prioritize chunks from specs where the term is defined
                CASE WHEN size(target_specs) > 0 AND c.spec_id IN target_specs THEN 0 ELSE 1 END,
                // Then by section type relevance
                CASE WHEN toLower(c.section_title) CONTAINS 'definition' THEN 0
                     WHEN resolved_full_name IS NOT NULL AND toLower(c.section_title) CONTAINS toLower(resolved_full_name) THEN 1
                     WHEN toLower(c.section_title) CONTAINS 'overview' THEN 2
                     ELSE 3 END,
                c.complexity_score ASC
            LIMIT 6
            """
        else:
            search_terms = ' AND '.join([f"toLower(c.content) CONTAINS toLower('{term}')" for term in key_terms[:3]])
            return f"""
            MATCH (c:Chunk)
            WHERE {search_terms}
            AND (toLower(c.section_title) CONTAINS 'definition'
                 OR c.chunk_type = 'definition')
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   null AS resolved_term, [] AS term_sources
            ORDER BY c.complexity_score ASC
            LIMIT 5
            """

    def _generate_comparison_query(self, question: str, analysis: Dict) -> str:
        """
        Generate SIMPLE query for comparison questions.
        Returns a marker that triggers multi-step retrieval in the retriever.

        Strategy:
        1. Get Term definitions for each entity
        2. Get function/role chunks for each entity
        3. Let LLM analyze differences
        """
        analysis = CypherSanitizer.sanitize_question_analysis(analysis)
        entities = [e['value'] for e in analysis['entities']]

        if len(entities) >= 2:
            # Return special marker - actual retrieval handled by _execute_comparison_retrieval
            return f"COMPARISON_MULTI_STEP:{entities[0]}:{entities[1]}"
        else:
            return self._generate_general_query(question, analysis)

    def _generate_procedure_query(self, question: str, analysis: Dict) -> str:
        """Generate query for procedure questions - targets procedure sections"""
        entities = analysis['entities']
        key_terms = analysis['key_terms']

        if entities:
            entity = entities[0]['value']
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{entity}')
            AND (c.chunk_type = 'procedure'
                 OR toLower(c.section_title) CONTAINS 'procedure'
                 OR toLower(c.section_title) CONTAINS 'process'
                 OR toLower(c.section_title) CONTAINS 'flow')
            OPTIONAL MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   collect(ref_doc.spec_id) AS referenced_specs
            ORDER BY c.complexity_score DESC
            LIMIT 6
            """
        else:
            search_terms = ' AND '.join([f"toLower(c.content) CONTAINS toLower('{term}')" for term in key_terms[:3]])
            return f"""
            MATCH (c:Chunk)
            WHERE {search_terms}
            AND c.chunk_type = 'procedure'
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms
            ORDER BY c.complexity_score DESC
            LIMIT 6
            """

    def _generate_reference_query(self, question: str, analysis: Dict) -> str:
        """Generate query for reference questions - includes cross-references"""
        entities = analysis['entities']

        if entities:
            entity = entities[0]['value']
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{entity}')
            MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   ref_doc.spec_id AS referenced_spec, r.confidence AS ref_confidence
            ORDER BY r.confidence DESC, c.complexity_score DESC
            LIMIT 8
            """
        else:
            return self._generate_general_query(question, analysis)

    def _generate_network_function_query(self, question: str, analysis: Dict) -> str:
        """Generate query for network function role questions"""
        entities = analysis['entities']

        if entities:
            nf = entities[0]['value']
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{nf}')
            AND (toLower(c.content) CONTAINS 'role'
                 OR toLower(c.content) CONTAINS 'function'
                 OR toLower(c.content) CONTAINS 'responsibility'
                 OR toLower(c.section_title) CONTAINS 'role'
                 OR toLower(c.section_title) CONTAINS 'function')
            OPTIONAL MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   collect(DISTINCT ref_doc.spec_id) AS referenced_specs
            ORDER BY c.complexity_score DESC
            LIMIT 6
            """
        else:
            return self._generate_general_query(question, analysis)

    def _generate_relationship_query(self, question: str, analysis: Dict) -> str:
        """Generate query for relationship questions between entities"""
        entities = [e['value'] for e in analysis['entities']]

        if len(entities) >= 2:
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{entities[0]}')
            AND toLower(c.content) CONTAINS toLower('{entities[1]}')
            OPTIONAL MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   collect(DISTINCT ref_doc.spec_id) AS referenced_specs
            ORDER BY c.complexity_score DESC
            LIMIT 6
            """
        else:
            return self._generate_general_query(question, analysis)

    def _generate_specification_query(self, question: str, analysis: Dict) -> str:
        """Generate query for specification-related questions"""
        key_terms = analysis['key_terms']

        return f"""
        MATCH (d:Document)
        WHERE toLower(d.title) CONTAINS toLower('{key_terms[0] if key_terms else ""}')
        MATCH (d)-[:CONTAINS]->(c:Chunk)
        WHERE toLower(c.content) CONTAINS toLower('{key_terms[1] if len(key_terms) > 1 else key_terms[0] if key_terms else ""}')
        RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
               c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
               c.complexity_score AS complexity_score, c.key_terms AS key_terms,
               d.title AS document_title
        ORDER BY c.complexity_score DESC
        LIMIT 5
        """

    def _generate_multiple_choice_query(self, question: str, analysis: Dict) -> str:
        """Generate query for multiple choice questions - broader role/function search"""
        analysis = CypherSanitizer.sanitize_question_analysis(analysis)
        entities = analysis['entities']

        if entities:
            entity = CypherSanitizer.sanitize_search_term(entities[0]['value'])
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{entity}')
            AND (toLower(c.content) CONTAINS 'role'
                 OR toLower(c.content) CONTAINS 'function'
                 OR toLower(c.content) CONTAINS 'responsibility'
                 OR toLower(c.content) CONTAINS 'authoriz'
                 OR toLower(c.content) CONTAINS 'select'
                 OR toLower(c.content) CONTAINS 'route'
                 OR toLower(c.content) CONTAINS 'notif')
            OPTIONAL MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   collect(DISTINCT ref_doc.spec_id) AS referenced_specs
            ORDER BY c.complexity_score DESC
            LIMIT 8
            """
        else:
            return self._generate_general_query(question, analysis)

    def _generate_general_query(self, question: str, analysis: Dict) -> str:
        """Generate fallback query for unclassified questions"""
        analysis = CypherSanitizer.sanitize_question_analysis(analysis)
        key_terms = analysis['key_terms'][:3]

        if len(key_terms) >= 2:
            sanitized_terms = [CypherSanitizer.sanitize_search_term(term) for term in key_terms]
            search_conditions = ' AND '.join([f"toLower(c.content) CONTAINS toLower('{term}')" for term in sanitized_terms if term])
            return f"""
            MATCH (c:Chunk)
            WHERE {search_conditions}
            OPTIONAL MATCH (c)-[r:REFERENCES_SPEC]->(ref_doc:Document)
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms,
                   collect(DISTINCT ref_doc.spec_id) AS referenced_specs
            ORDER BY c.complexity_score DESC
            LIMIT 6
            """
        else:
            term = key_terms[0] if key_terms else 'system'
            return f"""
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS toLower('{term}')
            RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
                   c.section_title AS section_title, c.content AS content, c.chunk_type AS chunk_type,
                   c.complexity_score AS complexity_score, c.key_terms AS key_terms
            ORDER BY c.complexity_score DESC
            LIMIT 5
            """


class EnhancedKnowledgeRetriever:
    """
    Retrieves knowledge from Neo4j using dynamically generated Cypher queries.
    Includes fallback search on query failures.
    Supports both rule-based and LLM-based query generation.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, local_llm_url: str = None):
        """Initialize Neo4j connection and query generator"""
        self.logger = get_logger('Knowledge_Retriever')
        self.logger.log(MAJOR, f"Initializing Knowledge Retriever - URI: {neo4j_uri}")

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Pass driver and LLM URL to generator for dynamic entity loading and LLM-based generation
        self.cypher_generator = CypherQueryGenerator(neo4j_driver=self.driver, local_llm_url=local_llm_url)

        self.logger.log(MAJOR, "Knowledge Retriever ready")

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        self.logger.log(MINOR, "Neo4j connection closed")

    def _execute_comparison_retrieval(self, entity1: str, entity2: str) -> List[RetrievedChunk]:
        """
        Execute multi-step retrieval for comparison questions.

        Steps:
        1. Get Term definition for entity1
        2. Get Term definition for entity2
        3. Get function/role chunks for entity1
        4. Get function/role chunks for entity2
        """
        self.logger.log(MINOR, f"Multi-step comparison retrieval: {entity1} vs {entity2}")
        chunks = []

        with self.driver.session() as session:
            # Step 1 & 2: Get Term definitions
            for entity in [entity1, entity2]:
                result = session.run("""
                    MATCH (t:Term {abbreviation: $abbr})
                    RETURN t.abbreviation AS abbr, t.full_name AS full_name,
                           t.source_specs AS specs, t.primary_spec AS primary_spec
                """, abbr=entity)

                record = result.single()
                if record:
                    # Create a synthetic chunk with Term definition
                    term_content = f"{record['abbr']}: {record['full_name']}\n"
                    term_content += f"Defined in: {record['primary_spec']}\n"
                    term_content += f"Also referenced in: {', '.join(record['specs'][:5])}"

                    chunks.append(RetrievedChunk(
                        chunk_id=f"term_{entity}",
                        spec_id=record['primary_spec'] or "",
                        section_id="term_definition",
                        section_title=f"Definition of {entity}",
                        content=term_content,
                        chunk_type="term_definition",
                        complexity_score=0.1,
                        key_terms=[entity, record['full_name']],
                        reference_path=[]
                    ))

            # Step 3 & 4: Get function/role chunks for each entity
            for entity in [entity1, entity2]:
                # Simple query - find chunks about this entity's function/role
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE toLower(c.content) CONTAINS toLower($entity)
                    AND (
                        toLower(c.section_title) CONTAINS 'function'
                        OR toLower(c.section_title) CONTAINS 'overview'
                        OR toLower(c.section_title) CONTAINS 'architecture'
                        OR toLower(c.section_title) CONTAINS 'role'
                        OR toLower(c.section_title) CONTAINS 'description'
                        OR c.chunk_type IN ['definition', 'architecture', 'overview']
                    )
                    RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
                           c.content, c.chunk_type, c.complexity_score, c.key_terms
                    ORDER BY c.complexity_score ASC
                    LIMIT 3
                """, entity=entity)

                records = list(result)
                self.logger.log(MINOR, f"Found {len(records)} function chunks for {entity}")

                for record in records:
                    # Neo4j returns keys with 'c.' prefix when using RETURN c.field
                    chunks.append(RetrievedChunk(
                        chunk_id=record.get("c.chunk_id", ""),
                        spec_id=record.get("c.spec_id", ""),
                        section_id=record.get("c.section_id", ""),
                        section_title=record.get("c.section_title", ""),
                        content=record.get("c.content", ""),
                        chunk_type=record.get("c.chunk_type", ""),
                        complexity_score=record.get("c.complexity_score", 0.0),
                        key_terms=record.get("c.key_terms", []),
                        reference_path=[]
                    ))

        self.logger.log(MAJOR, f"Comparison retrieval: {len(chunks)} chunks for {entity1} vs {entity2}")
        return chunks

    def retrieve_with_cypher(self, question: str, use_llm: bool = False, llm_model: str = "deepseek-r1:7b") -> tuple[List[RetrievedChunk], str, str]:
        """
        Retrieve relevant chunks using dynamic Cypher query.
        Falls back to simple search on query failure.

        Args:
            question: User's question
            use_llm: If True, uses LocalLM for analysis and query generation
            llm_model: Model to use for LLM-based operations
        """
        self.logger.log(MAJOR, f"Retrieving for: {question[:60]}... (LLM mode: {use_llm})")

        # Analyze and generate query (optionally using LLM)
        analysis = self.cypher_generator.analyze_question(question, use_llm=use_llm, llm_model=llm_model)
        cypher_query = self.cypher_generator.generate_cypher_query(question, analysis, use_llm=use_llm, llm_model=llm_model)

        # Check for multi-step comparison marker
        if cypher_query.startswith("COMPARISON_MULTI_STEP:"):
            parts = cypher_query.split(":")
            if len(parts) >= 3:
                entity1, entity2 = parts[1], parts[2]
                chunks = self._execute_comparison_retrieval(entity1, entity2)
                strategy = f"comparison multi-step - {len(chunks)} chunks for {entity1} vs {entity2}"
                return chunks, f"Multi-step: {entity1} vs {entity2}", strategy

        # Execute regular query
        retrieved_chunks = []
        with self.driver.session() as session:
            try:
                # Validate query safety
                if not CypherSanitizer.validate_query_safety(cypher_query):
                    raise ValueError("Query contains potentially dangerous patterns")

                self.logger.log(MINOR, "Executing Cypher query")
                result = session.run(cypher_query)

                # Process results
                for record in result:
                    chunk = RetrievedChunk(
                        chunk_id=record.get("chunk_id", ""),
                        spec_id=record.get("spec_id", ""),
                        section_id=record.get("section_id", ""),
                        section_title=record.get("section_title", ""),
                        content=record.get("content", ""),
                        chunk_type=record.get("chunk_type", ""),
                        complexity_score=record.get("complexity_score", 0.0),
                        key_terms=record.get("key_terms", []),
                        reference_path=record.get("referenced_specs", [])
                    )
                    retrieved_chunks.append(chunk)

                self.logger.log(MAJOR, f"Retrieved {len(retrieved_chunks)} chunks")

            except Exception as e:
                self.logger.log(ERROR, f"Query execution failed: {e}")
                self.logger.log(MINOR, "Falling back to simple search")

                # Fallback query
                fallback_query = f"""
                MATCH (c:Chunk)
                WHERE toLower(c.content) CONTAINS toLower($search_term)
                RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
                       c.content, c.chunk_type, c.complexity_score, c.key_terms
                ORDER BY c.complexity_score DESC
                LIMIT 5
                """

                search_term = analysis.get('key_terms', ['system'])[0] if analysis.get('key_terms') else 'system'
                safe_search_term = CypherSanitizer.sanitize_search_term(search_term)
                self.logger.log(DEBUG, f"Fallback search term: {safe_search_term}")

                result = session.run(fallback_query, search_term=safe_search_term)

                for record in result:
                    chunk = RetrievedChunk(
                        chunk_id=record.get("chunk_id", ""),
                        spec_id=record.get("spec_id", ""),
                        section_id=record.get("section_id", ""),
                        section_title=record.get("section_title", ""),
                        content=record.get("content", ""),
                        chunk_type=record.get("chunk_type", ""),
                        complexity_score=record.get("complexity_score", 0.0),
                        key_terms=record.get("key_terms", []),
                        reference_path=[]
                    )
                    retrieved_chunks.append(chunk)

                self.logger.log(MAJOR, f"Fallback retrieved {len(retrieved_chunks)} chunks")
                cypher_query = fallback_query

        strategy = f"{analysis['question_type']} query - {len(retrieved_chunks)} chunks"
        return retrieved_chunks, cypher_query, strategy


class RAGOrchestratorV2:
    """
    Main RAG orchestrator coordinating retrieval and LLM generation.
    Entry point for RAG queries.
    Supports both rule-based and LLM-based query generation.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 claude_api_key: str = None, deepseek_api_url: str = None):
        """Initialize retriever and LLM integrator"""
        self.logger = get_logger('RAG_System')
        self.logger.log(MAJOR, "Initializing RAG Orchestrator V2")

        local_llm_url = deepseek_api_url if deepseek_api_url else "http://192.168.1.14:11434/api/chat"

        # Pass LLM URL to retriever for LLM-based query generation
        self.retriever = EnhancedKnowledgeRetriever(neo4j_uri, neo4j_user, neo4j_password, local_llm_url=local_llm_url)
        self.llm_integrator = LLMIntegrator(
            claude_api_key=claude_api_key,
            local_llm_url=local_llm_url
        )

        self.logger.log(MAJOR, "RAG Orchestrator V2 ready")

    def close(self):
        """Clean up resources"""
        self.retriever.close()
        self.logger.log(MINOR, "RAG Orchestrator closed")

    def query(self, question: str, model: str = "claude", use_llm_query: bool = False, query_llm_model: str = "deepseek-r1:7b") -> RAGResponse:
        """
        Process RAG query: retrieve knowledge and generate answer.

        Args:
            question: User's question
            model: LLM model for answer generation ("claude" or local model name like "deepseek-r1:7b")
            use_llm_query: If True, uses LocalLM for question analysis and Cypher generation
            query_llm_model: Model to use for LLM-based query generation
        """
        self.logger.log(MAJOR, f"RAG query - model: {model}, use_llm_query: {use_llm_query}")
        self.logger.log(MINOR, f"Question: {question[:80]}...")

        # Retrieve relevant knowledge (optionally using LLM for query generation)
        retrieved_chunks, cypher_query, strategy = self.retriever.retrieve_with_cypher(
            question, use_llm=use_llm_query, llm_model=query_llm_model
        )

        # Handle empty results
        if not retrieved_chunks:
            self.logger.log(ERROR, "No relevant information found")
            return RAGResponse(
                answer="No relevant information found in the knowledge base.",
                sources=[],
                query=question,
                cypher_query=cypher_query,
                retrieval_strategy=strategy,
                timestamp=datetime.now()
            )

        # Generate answer
        answer = self.llm_integrator.generate_answer(question, retrieved_chunks, cypher_query, model)

        self.logger.log(MAJOR, "RAG query completed")
        return RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
            query=question,
            cypher_query=cypher_query,
            retrieval_strategy=strategy,
            timestamp=datetime.now()
        )

    def explain_query_strategy(self, question: str) -> Dict:
        """Explain the query strategy for a given question (for debugging)"""
        analysis = self.retriever.cypher_generator.analyze_question(question)
        cypher_query = self.retriever.cypher_generator.generate_cypher_query(question, analysis)

        return {
            "question": question,
            "analysis": analysis,
            "cypher_query": cypher_query,
            "strategy_explanation": f"This is a {analysis['question_type']} question about {', '.join([e['value'] for e in analysis['entities']])}"
        }


class RAGConfigV2:
    """Default configuration for RAG system"""

    def __init__(self):
        self.neo4j_uri = "neo4j://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.claude_api_key = None


def create_rag_system_v2(claude_api_key: str = None, deepseek_api_url: str = None) -> RAGOrchestratorV2:
    """
    Factory function to create configured RAG system.

    Args:
        claude_api_key: Anthropic API key for Claude
        deepseek_api_url: URL for local Ollama LLM
    """
    logger.log(MAJOR, "Creating RAG system V2")
    config = RAGConfigV2()
    if claude_api_key:
        config.claude_api_key = claude_api_key

    return RAGOrchestratorV2(
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
        claude_api_key=config.claude_api_key,
        deepseek_api_url=deepseek_api_url
    )


if __name__ == "__main__":
    print("RAG System V2 loaded successfully!")
    print("Features:")
    print("- Dynamic Cypher query generation based on question context")
    print("- Enhanced knowledge retrieval with complexity scoring")
    print("- Multiple choice question handling")
    print("- Cross-reference analysis")
    print("- Support for Claude and local LLM models")
    print("\nTo use: rag_system = create_rag_system_v2('your-claude-api-key')")
