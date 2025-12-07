3GPP Knowledge Graph & RAG System
==================================

Project Overview
----------------
AI-powered Q&A chatbot for 3GPP 5G/telecom technical specifications.
Uses Knowledge Graph (Neo4j) + RAG (Retrieval Augmented Generation) + LLM.


Features
--------
- 350+ 3GPP specifications indexed
- 8 question types supported (definition, comparison, procedure, etc.)
- Multiple LLM backends (Claude API, local Ollama models)
- Web-based chat interface
- Centralized logging with 5 levels
- TeleQnA benchmark integration


Quick Start
-----------
1. Setup environment:
   source .venv/bin/activate
   cp .env.example .env
   # Edit .env with your API keys

2. Start system:
   python orchestrator.py all

3. Access chat UI:
   http://localhost:9999


Commands
--------
python orchestrator.py check        # Check system status
python orchestrator.py start-neo4j  # Start Neo4j Docker
python orchestrator.py init-kg      # Initialize knowledge graph
python orchestrator.py run          # Start Django server
python orchestrator.py all          # Start everything
python orchestrator.py stop         # Stop all services


Testing
-------
pytest tests/ -v                    # Run all tests
pytest tests/ -m "not slow"         # Skip slow tests


Documentation
-------------
.md/project_architecture.md   - System architecture
.md/logging_system.md         - Logging configuration
.md/llm_integrator_unified.md - LLM integration
.md/orchestrator.md           - Orchestrator usage
.md/tele_qna_*.md            - Benchmark documentation


Tech Stack
----------
- Python 3.12
- Django 4.2.11
- Neo4j (Docker)
- Anthropic Claude API
- Ollama (local LLMs)


Default Settings
----------------
- Default LLM: deepseek-r1:14b (local)
- Neo4j: localhost:7687
- Django: localhost:9999
- Log file: logs/app.log


Contact
-------
Project for 3GPP technical specification research and Q&A.
