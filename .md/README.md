3GPP Knowledge Graph & RAG System
==================================

Project Overview
----------------
AI-powered Q&A chatbot for 3GPP 5G/telecom technical specifications.
Uses Knowledge Graph (Neo4j) + RAG (Retrieval Augmented Generation) + LLM.


Features
--------
- 350+ 3GPP specifications indexed in Neo4j Knowledge Graph
- RAG System V3 with Hybrid Retrieval (Vector + Graph Search)
- 9 question types supported (definition, comparison, procedure, network_function, relationship, multiple_choice, multi_intent, troubleshooting, general)
- Advanced features:
  * Smarter query expansion with diversification
  * Semantic similarity deduplication
  * Subject-based retrieval boosting
  * Intent-aware prompt templates
- Multiple LLM backends (Claude API, local Ollama models)
- Web-based chat interface (Django)
- Centralized logging with 5 custom levels (CRITICAL > ERROR > MAJOR > MINOR > DEBUG)
- TeleQnA benchmark integration (10K questions)


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
Core Documentation:
.md/project_architecture.md            - System architecture overview
.md/benchmark_guide.md                 - Benchmark testing guide
.md/logging_system.md                  - Logging configuration

Recent Enhancements (2025-12-10):
.md/advanced_features_20251210.md      - Smarter query expansion + semantic deduplication
.md/fixes_summary_20251210.md          - MCQ instruction fix + content deduplication
.md/mcq_instruction_fix_20251210.md    - MCQ formatting fix details
.md/content_deduplication_fix_20251210.md - Deduplication implementation

System Components:
.md/llm_integrator_unified.md          - LLM integration (Claude + Ollama)
.md/orchestrator.md                    - Orchestrator usage
.md/enhance/                           - Enhancement documentation folder


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
