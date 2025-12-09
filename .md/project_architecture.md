3GPP Knowledge Graph & RAG System - Architecture
================================================

System Overview
---------------
AI-powered Q&A system for 3GPP 5G/telecom specifications.
Combines Knowledge Graph (Neo4j) with RAG for accurate technical answers.


Data Flow
---------
3GPP Docs (.docx) -> Document Processing -> JSON -> Neo4j KG -> RAG System -> LLM -> Chat UI


Core Components
---------------

1. Document Processing (document_processing/)
   - Converts 3GPP specs (.docx) to structured JSON
   - Extracts sections, chunks, cross-references
   - Output: 3GPP_JSON_DOC/processed_json_v2/

2. Knowledge Graph (Neo4j)
   - Nodes: Document, Chunk, Term
   - Relationships: CONTAINS, REFERENCES_SPEC, REFERENCES_CHUNK, DEFINED_IN
   - Term nodes store abbreviation→full_name mappings
   - Initialized via orchestrator.py or KG_builder.ipynb

3. RAG System V3 (rag_system_v3.py) - ACTIVE
   - HybridRetriever: Combines Vector Search + Graph Search
   - SemanticQueryAnalyzer: LLM-based query understanding
   - QueryExpander: Generate query variations for better recall
   - PromptTemplates: Intent-specific prompts (7 types)
   - EnhancedLLMIntegrator: Unified LLM backend (Claude + Ollama)

4. RAG System V2 (rag_system_v2.py) - Legacy
   - CypherQueryGenerator: 8 question types
   - EnhancedKnowledgeRetriever: Neo4j query execution
   - LLMIntegrator: Unified API/local LLM support

5. Web Interface (chatbot_project/)
   - Django 4.2.11
   - RAGManager singleton
   - Routes: / (chat UI), /api/ (JSON API)


Neo4j Schema
------------
(:Document {spec_id, title, version, total_chunks})
(:Chunk {chunk_id, spec_id, section_id, section_title, content, chunk_type, complexity_score, key_terms})
(:Term {abbreviation, full_name, term_type, source_specs, primary_spec})

(:Document)-[:CONTAINS]->(:Chunk)
(:Chunk)-[:REFERENCES_SPEC]->(:Document)
(:Chunk)-[:REFERENCES_CHUNK]->(:Chunk)
(:Term)-[:DEFINED_IN]->(:Document)


Question Types
--------------
- Definition: "What is AMF?"
- Comparison: "Compare AMF and SMF"
- Procedure: "How does registration work?"
- Reference: "What specs reference UPF?"
- Network Function: "Role of PCF"
- Relationship: "How do AMF and SMF interact?"
- Specification: "What does TS 23.501 cover?"
- Multiple Choice: TeleQnA format


LLM Support
-----------
API-based: Claude (Anthropic)
Local (Ollama): deepseek-r1:7b, deepseek-r1:14b, gemma3:12b, mistral:7b, llama3.1:8b
Default: deepseek-r1:14b


Logging System
--------------
5 levels: CRITICAL > ERROR > MAJOR > MINOR > DEBUG
Log file: logs/app.log (rotates on start, max 1GB)
Config: log_config.json


Key Files
---------
orchestrator.py           - System management
rag_system_v3.py          - RAG V3 (Hybrid) - Active
rag_system_v2.py          - RAG V2 (Graph-only) - Legacy
hybrid_retriever.py       - Vector + Graph retrieval
prompt_templates.py       - Intent-specific prompts
enhanced_query_processor.py - Advanced query understanding
term_extractor.py         - Abbreviation extraction
logging_config.py         - Logging (5 levels)
cypher_sanitizer.py       - Query security
log_config.json           - Log config
.env                      - Environment variables


Directory Structure
-------------------
3GPP/
├── .env, .env.example       # Environment config
├── orchestrator.py          # System management
├── rag_system_v3.py         # RAG V3 (Hybrid) - Active
├── rag_system_v2.py         # RAG V2 (Graph-only) - Legacy
├── hybrid_retriever.py      # Vector + Graph retrieval
├── prompt_templates.py      # Intent-specific prompts
├── enhanced_query_processor.py # Query understanding
├── term_extractor.py        # Abbreviation extraction
├── logging_config.py        # Logging
├── cypher_sanitizer.py      # Security
├── log_config.json          # Log config
├── chatbot_project/         # Django app
│   └── chatbot/
│       ├── views.py
│       └── templates/
├── document_processing/     # Doc processing
├── 3GPP_JSON_DOC/          # Processed data
├── tests/                   # Test suite (196 tests)
├── logs/                    # Log files
└── .md/                     # Documentation


Request Flow (V3 - Hybrid)
--------------------------
1. User sends question via chat UI
2. Django receives POST /api/
3. RAGManager.query() called
4. SemanticQueryAnalyzer analyzes query intent
5. QueryExpander generates query variations
6. HybridRetriever executes:
   - VectorRetriever: semantic similarity search
   - CypherQueryGenerator: graph traversal search
7. Results merged and reranked
8. PromptTemplates selects intent-specific prompt
9. EnhancedLLMIntegrator generates answer
10. Response returned to user

Request Flow (V2 - Legacy)
--------------------------
1. User sends question via chat UI
2. Django receives POST /api/
3. RAGManager.query() called
4. CypherQueryGenerator analyzes question type
5. Generates appropriate Cypher query
6. EnhancedKnowledgeRetriever executes on Neo4j
7. Retrieved chunks passed to LLMIntegrator
8. LLM generates answer with context
9. Response returned to user


RAG V3 Architecture
-------------------
```
User Query
    │
    ▼
SemanticQueryAnalyzer (intent detection)
    │
    ▼
QueryExpander (generate variations)
    │
    ▼
HybridRetriever
    ├── VectorRetriever (semantic search)
    │       │
    │       ▼
    │   Neo4j Vector Index
    │
    └── CypherQueryGenerator (graph search)
            │
            ▼
        Neo4j Knowledge Graph
    │
    ▼
Merger & Reranker (combine results)
    │
    ▼
PromptTemplates (intent-specific prompt)
    │
    ▼
EnhancedLLMIntegrator (Claude/Ollama)
    │
    ▼
RAGResponseV3
```
