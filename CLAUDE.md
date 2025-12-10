# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3GPP Knowledge Graph & RAG System - Processes 3GPP technical specifications into a Neo4j knowledge graph with AI-powered Q&A capabilities for 5G/telecom standards.

## Build & Run Commands

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r document_processing/requirements.txt

# Use orchestrator (recommended)
python orchestrator.py check           # Check system status
python orchestrator.py install         # Install dependencies in venv
python orchestrator.py start-neo4j     # Start Neo4j Docker container
python orchestrator.py init-kg         # Initialize KG + setup vector search (10-30 min)
python orchestrator.py init-kg --skip-vector  # Init KG only, skip vector search
python orchestrator.py init-kg --vector-only  # Setup vector search only (KG exists)
python orchestrator.py setup-v3        # Setup vector search only (alternative)
python orchestrator.py run             # Start Django server
python orchestrator.py ngrok           # Start ngrok tunnels
python orchestrator.py all             # Start everything: Neo4j, KG (if empty), ngrok, Django
python orchestrator.py all --init-kg   # Force KG re-initialization + vector search
python orchestrator.py stop            # Stop all services

# Manual commands
cd chatbot_project && python manage.py runserver

# Run regression tests (requires .venv activated)
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing
```

## Required Environment Variables

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export manually:
```bash
export CLAUDE_API_KEY="your-anthropic-api-key"
# Neo4j defaults to neo4j://localhost:7687 with neo4j/password
```

Note: Orchestrator automatically loads `.env` file on startup.

## Architecture

**Data Flow:**
```
3GPP Documents (.docx) → Document Processing → JSON → Neo4j KG → RAG System → LLM → Django Chat
```

**Core Components:**

1. **Document Processing** (`document_processing/`)
   - Converts 350+ 3GPP specs to structured JSON
   - Output: `3GPP_JSON_DOC/processed_json_v2/`

2. **Knowledge Graph** (`KG_builder.ipynb`)
   - Neo4j with Document, Chunk, and Term nodes
   - Relationships: CONTAINS, REFERENCES_SPEC, REFERENCES_CHUNK, DEFINED_IN
   - Term nodes store abbreviation→full_name mappings (e.g., SCP→"Service Communication Proxy")
   - Use orchestrator.py for automated initialization

3. **RAG System V3** (`rag_system_v3.py`) - **Active version**
   - Entry: `create_rag_system_v3(claude_api_key, local_llm_url)`
   - `HybridRetriever`: Combines Vector Search + Graph Search
   - `SemanticQueryAnalyzer`: LLM-based query understanding
   - `QueryExpander`: Generate query variations for better recall
   - `PromptTemplates`: Intent-specific prompt templates (7 types)
   - `EnhancedLLMIntegrator`: Unified LLM backend (Claude API + Ollama)
   - `CypherQueryGenerator`: 8 question types (imported from legacy v2)
   - Features: Hybrid retrieval, reranking, query expansion

4. **Web Interface** (`chatbot_project/`)
   - Django 4.2.11, SQLite for chat history
   - `RAGManager` singleton manages RAG system lifecycle
   - Routes: `/` (chat UI), `/api/` (JSON API)

## Key Patterns

**Logging:** Use centralized logging system with 5 custom levels
```python
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG
setup_centralized_logging()
logger = get_logger('ComponentName')  # RAG_System, Chatbot, Knowledge_Retriever, LLM_Integrator, Cypher_Generator

# Log levels (high to low): CRITICAL > ERROR > MAJOR > MINOR > DEBUG
logger.log(CRITICAL, "System failure")
logger.log(ERROR, "Operation failed")
logger.log(MAJOR, "Important event")
logger.log(MINOR, "Detailed info")
logger.log(DEBUG, "Debug info")
```

Log files: `logs/app.log` (rotates on startup and when >1GB)
Config: `log_config.json` (customize levels, format, max size)

**Cypher Sanitization:** Always sanitize user input for Neo4j queries
```python
from cypher_sanitizer import CypherSanitizer
safe_term = CypherSanitizer.sanitize_search_term(user_input)
```

**Using RAG System V3 (Recommended):**
```python
from rag_system_v3 import create_rag_system_v3

rag = create_rag_system_v3(
    claude_api_key="sk-...",
    local_llm_url="http://192.168.1.14:11434/api/chat"
)

# First time: setup vector search (10-30 min)
status = rag.check_vector_index_status()
if not status['ready_for_hybrid']:
    rag.setup_vector_search()

# Query with hybrid retrieval
response = rag.query(
    "Compare AMF and SMF",
    model="deepseek-r1:14b",
    use_hybrid=True,
    use_vector=True,
    use_graph=True
)
print(response.answer)
print(f"Strategy: {response.retrieval_strategy}")
```

**Note:** RAG System V2 has been removed. All functionality is now in V3.

## Code Style

- Comment before code block it describes
- Create markdown documentation in `.md/` folder for major changes
- Prefer plain text over heavy formatting in docs
- Provide mermaid diagrams for coding workflows where applicable in md files

## Testing

Test suite located in `tests/` folder:
- `test_cypher_sanitizer.py` - Security and input sanitization tests
- `test_logging_config.py` - Centralized logging tests
- `test_rag_system_v3.py` - RAG V3 component tests
- `test_hybrid_retriever.py` - Hybrid retrieval tests (21 tests)
- `test_prompt_templates.py` - Prompt template tests (25 tests)
- `test_hybrid_term_extraction.py` - Term extraction tests
- `test_subject_classifier.py` - Subject classification tests (14 tests)

Run `pytest tests/ -v` to execute all tests.

**Note:** V2-related tests have been moved to `trash/` folder (2025-12-09).

# IMPORTANT NOTES
- Always provide md file into .md folder for any change that affects architecture or workflows.
- Always test with local Neo4j instance before pushing changes.
- Always use local LLM instances for testing to avoid API costs.
- Always run test suite after major changes.
- Always update tests file to compatible with new changes but Always need my confirmation before changing test files.
- Do not use rm files or something similar, move all of them to trash folder

## Recent Changes (2025-12-09)

**Subject-Based KG Enhancement:**
- `subject_classifier.py`: NEW - Classifies chunks by subject type
  - 5 subject types: Standards specifications, Standards overview, Lexicon, Research publications, Research overview
  - Query subject detection for score boosting
- `KG_builder.ipynb`: Added SubjectNodeBuilder (cells 13-15)
  - Creates Subject nodes in Neo4j
  - Classifies and updates all chunks with subject property
  - Creates HAS_SUBJECT relationships
- `hybrid_retriever.py`: Subject-aware retrieval
  - Added `subject` and `subject_confidence` to ScoredChunk
  - Added `use_subject_boost` parameter to `retrieve()`
  - Boosts chunks matching expected subject types
- See `.md/enhance/subject_based_kg_enhancement.md` for details

## Changes (2025-12-08)

**RAG System V3 (NEW):**
- `rag_system_v3.py`: Main orchestrator with hybrid retrieval
- `hybrid_retriever.py`: Vector + Graph search (~750 LOC)
  - `VectorIndexer`: Create embeddings & Neo4j vector index
  - `VectorRetriever`: Semantic similarity search
  - `SemanticQueryAnalyzer`: LLM-based query understanding
  - `QueryExpander`: Generate query variations
  - `HybridRetriever`: Combine & rerank results
- `prompt_templates.py`: 7 intent-specific templates (~400 LOC)
  - Definition, Comparison, Procedure, Network Function
  - Relationship, Multiple Choice, General
- 64 new tests for V3 components
- See `.md/enhance/rag_system_v3_implementation.md` for details

**Orchestrator Updates:**
- Added `setup-v3` command for vector search initialization
- `check` command now shows V3 readiness status
- `all` command automatically starts Neo4j Docker container
- Waits for Neo4j to be ready (up to 60 seconds)
- Starts ngrok tunnels automatically

## Previous Changes (2025-12-07)

**Centralized Logging System:**
- 5 custom log levels: CRITICAL > ERROR > MAJOR > MINOR > DEBUG
- Auto-rotation: new file on startup, rotate when >1GB
- Config via `log_config.json`
- All logs to `logs/app.log`
- See `.md/logging_system.md` for details

**LLM Integrator Refactoring:**
- Unified `LLMIntegrator` class replaces separate `ClaudeIntegrator` and `DeepSeekIntegrator`
- Single class handles both API-based (Claude) and local LLM (Ollama) models
- Internal routing based on `model` parameter
- Removed ~150 lines of duplicate code
- See `.md/llm_integrator_unified.md` for details

**Term Node System:**
- `term_extractor.py`: Extracts abbreviations from 3GPP abbreviation sections
- `TermNodeBuilder` in `KG_builder.ipynb`: Creates Term nodes in Neo4j
- Definition/comparison queries now resolve abbreviations via Term nodes first
- Term nodes link to source specs via DEFINED_IN relationships
- See `.md/rag_query_redesign.md` for details

## Known Issues

- REFERENCES_CHUNK relationships may be missing (run ReferencesFixer in `KG_builder.ipynb` to fix)
- DeepSeek endpoint defaults to local Ollama instance (192.168.1.14:11434)
- Vector search requires Neo4j 5.11+ for vector index support

## File Structure

```
3GPP/
├── orchestrator.py          # System orchestrator
├── rag_system_v3.py         # RAG V3 (Hybrid) - Active
├── rag_core.py              # Shared RAG components
├── hybrid_retriever.py      # Vector + Graph retrieval (with subject boost)
├── subject_classifier.py    # Subject classification for chunks/queries
├── prompt_templates.py      # Intent-specific prompts
├── cypher_sanitizer.py      # Query security
├── term_extractor.py        # Abbreviation extraction
├── logging_config.py        # Centralized logging
├── log_config.json          # Log configuration
├── .env                     # Environment variables
├── chatbot_project/         # Django web app
├── document_processing/     # Document processing
├── 3GPP_JSON_DOC/          # Processed JSON files
├── tele_qna/               # Benchmark Q&A data (10k questions)
├── tests/                   # Test suite (V3 tests)
├── logs/                    # Log files
├── trash/                   # Removed/deprecated files
└── .md/                     # Documentation
```
