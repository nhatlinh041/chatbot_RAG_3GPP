Orchestrator Script
Date: 2025-12-07

Overview
The orchestrator.py script manages all components of the 3GPP Knowledge Graph & RAG System. It provides a unified interface for checking system status, initializing the knowledge graph, and running the Django chatbot server.


Usage

Check system status:
    python orchestrator.py check

Install dependencies in virtual environment:
    python orchestrator.py install

Start Neo4j Docker container:
    python orchestrator.py start-neo4j

Initialize knowledge graph from JSON files:
    python orchestrator.py init-kg

Start Django chatbot server:
    python orchestrator.py run

Initialize and run (combined):
    python orchestrator.py all

Options:
    --host HOST     Server host (default: 127.0.0.1)
    --port PORT     Server port (default: 8000)
    --no-clear      Don't clear existing KG data when initializing


Quick Start

0. Setup Docker permissions (first time only):
   See .md/docker_setup.md for detailed instructions
   Quick: sudo usermod -aG docker $USER && newgrp docker

1. python orchestrator.py install     # Install dependencies
2. python orchestrator.py check       # Verify setup
3. python orchestrator.py start-neo4j # Start Neo4j database
4. python orchestrator.py init-kg     # Build knowledge graph
5. python orchestrator.py run         # Start chatbot


Component Dependencies

```
Neo4j Database (must be running on localhost:7687)
    ↓
Knowledge Graph Population (from JSON files)
    ↓
RAG System Initialization
    ├─ CypherQueryGenerator
    ├─ EnhancedKnowledgeRetriever
    └─ LLMIntegrator (unified: Claude API + Ollama local models)
    ↓
Django Chatbot Server (http://localhost:8000)
```


System Check Output

The `check` command displays:
- Neo4j connection status and database statistics
- JSON data files availability
- Environment variables (CLAUDE_API_KEY)
- Django project status


Configuration

Default Neo4j settings (hardcoded):
    URI: neo4j://localhost:7687
    User: neo4j
    Password: password

JSON data location:
    3GPP_JSON_DOC/processed_json_v2/

Django project:
    chatbot_project/


Workflow Diagram

```
                    ┌─────────────────┐
                    │  orchestrator   │
                    │     check       │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Neo4j       │ │  JSON Files   │ │  Environment  │
    │   Status      │ │   Count       │ │   Variables   │
    └───────────────┘ └───────────────┘ └───────────────┘

                    ┌─────────────────┐
                    │  orchestrator   │
                    │    init-kg      │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Clear DB      │ │ Load JSON     │ │ Create Nodes  │
    │ (optional)    │ │ Files         │ │ & Relations   │
    └───────────────┘ └───────────────┘ └───────────────┘

                    ┌─────────────────┐
                    │  orchestrator   │
                    │      run        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │   Django Server   │
                    │  localhost:8000   │
                    └───────────────────┘
```


Error Handling

Neo4j not running:
    Error message displayed, suggests starting Neo4j first

No JSON files:
    Error message, init-kg will fail

CLAUDE_API_KEY not set:
    Warning displayed, RAG will use DeepSeek only

KG empty when running server:
    Warning displayed, suggests running init-kg first
