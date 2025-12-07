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
python orchestrator.py init-kg         # Initialize knowledge graph
python orchestrator.py run             # Start Django server
python orchestrator.py ngrok           # Start ngrok tunnels
python orchestrator.py all             # Start everything: Neo4j, KG (if empty), ngrok, Django
python orchestrator.py all --init-kg   # Force KG re-initialization
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

3. **RAG System V2** (`rag_system_v2.py`) - Active version
   - Entry: `create_rag_system_v2(claude_api_key, deepseek_api_url)`
   - `CypherQueryGenerator`: Generates queries based on 8 question types (definition, comparison, procedure, reference, network function, relationship, specification, multiple-choice)
   - `EnhancedKnowledgeRetriever`: Executes queries against Neo4j
   - `LLMIntegrator`: Unified LLM backend supporting:
     - API models: Claude (Anthropic API)
     - Local models: DeepSeek, Llama, etc. (via Ollama)
   - Uses internal routing to handle different model types

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

**Adding New Question Types:** Extend `CypherQueryGenerator.query_patterns` dict and add corresponding `_generate_X_query` method

**Using LLM Models:**
```python
# Create RAG system with both API and local LLM
rag = create_rag_system_v2(
    claude_api_key="sk-...",
    deepseek_api_url="http://192.168.1.14:11434/api/chat"
)

# Use Claude API
response = rag.query("What is AMF?", model="claude")

# Use local LLM models (Ollama)
response = rag.query("What is SMF?", model="deepseek-r1:7b")
response = rag.query("Explain UPF", model="llama3.2")
```

## Code Style

- Comment before code block it describes
- Create markdown documentation in `.md/` folder for major changes
- Prefer plain text over heavy formatting in docs
- should provide mermaid diagrams for coding workflows where applicable in md files

## Testing

Test suite located in `tests/` folder:
- `test_cypher_sanitizer.py` - Security and input sanitization tests
- `test_logging_config.py` - Centralized logging tests
- `test_rag_system_v2.py` - RAG system component tests
- `test_hybrid_term_extraction.py` - Term extraction tests

Run `pytest tests/ -v` to execute all tests.

# IMPORTANT NOTES
- Always provide md file into .md folder for any change that affects architecture or workflows.
- Always test with local Neo4j instance before pushing changes.
- Always use local LLM instances for testing to avoid API costs.
- Always run test suite after major changes.
- Always update tests file to compatible with new changes but Always need my confirmation before changing test files.
## Recent Changes (2025-12-07)

**Centralized Logging System:**
- 5 custom log levels: CRITICAL > ERROR > MAJOR > MINOR > DEBUG
- Auto-rotation: new file on startup, rotate when >1GB
- Config via `log_config.json`
- All logs to `logs/app.log`
- See `.md/logging_system.md` for details

**Orchestrator Updates:**
- `all` command now automatically starts Neo4j Docker container if not running
- Waits for Neo4j to be ready (up to 60 seconds)
- Waits for KG initialization to complete before starting Django
- Starts ngrok tunnels automatically
- Use `--init-kg` flag to force KG re-initialization

**LLM Integrator Refactoring:**
- Unified `LLMIntegrator` class replaces separate `ClaudeIntegrator` and `DeepSeekIntegrator`
- Single class handles both API-based (Claude) and local LLM (Ollama) models
- Internal routing based on `model` parameter
- Removed ~150 lines of duplicate code
- See `.md/llm_integrator_unified.md` for details

**Term Node System (NEW):**
- `term_extractor.py`: Extracts abbreviations from 3GPP abbreviation sections
- `TermNodeBuilder` in `KG_builder.ipynb`: Creates Term nodes in Neo4j
- Definition/comparison queries now resolve abbreviations via Term nodes first
- Term nodes link to source specs via DEFINED_IN relationships
- See `.md/rag_query_redesign.md` for details

## Known Issues

- REFERENCES_CHUNK relationships may be missing (run ReferencesFixer in `KG_builder.ipynb` to fix)
- DeepSeek endpoint defaults to local Ollama instance (192.168.1.14:11434)
- Term nodes need to be created after KG initialization (run TermNodeBuilder cells in KG_builder.ipynb)
