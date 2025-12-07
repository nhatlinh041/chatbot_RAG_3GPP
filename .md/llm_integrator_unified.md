# Unified LLM Integrator Refactoring

Date: 2025-12-07
Author: Claude Code

## Overview

Refactored RAG system V2 to use a single unified LLMIntegrator class instead of separate classes for each LLM provider. The new design uses internal routing to handle both API-based models (Claude) and local LLM models (DeepSeek, Llama, etc via Ollama).

## Changes Made

### 1. Created Unified LLMIntegrator Class

Single class that handles all LLM interactions with internal routing based on model parameter.

```python
class LLMIntegrator:
    def __init__(self, claude_api_key=None, local_llm_url="http://..."):
        # Initialize both API and local LLM clients
        self.claude_client = anthropic.Anthropic(api_key) if claude_api_key else None
        self.local_llm_url = local_llm_url

    def generate_answer(self, query, retrieved_chunks, cypher_query, model="claude"):
        # Route to appropriate handler based on model parameter
        if model == "claude":
            return self._generate_with_api(...)
        else:
            return self._generate_with_local_llm(...)
```

**Internal Methods:**
- `_generate_with_api()` - Handles Claude API calls
- `_generate_with_local_llm()` - Handles local LLM (Ollama) calls
- `_prepare_context()` - Shared context preparation
- `_is_multiple_choice()` - Shared question type detection

### 2. Simplified RAGOrchestratorV2

Before (2 separate integrators):
```python
self.integrators = {}
self.integrators['claude'] = ClaudeIntegrator(api_key)
self.integrators['deepseek'] = DeepSeekIntegrator(url)
```

After (1 unified integrator):
```python
self.llm_integrator = LLMIntegrator(
    claude_api_key=api_key,
    local_llm_url=url
)
```

### 3. Removed Backwards Compatibility

- No wrapper classes
- Clean, simple API
- Updated all tests to use new unified interface

## Architecture

```
LLMIntegrator (single class)
    |
    +-- _generate_with_api() -> Claude Anthropic API
    |       - Claude 3 Haiku
    |       - Claude 3 Sonnet (future)
    |       - Other Anthropic models (future)
    |
    +-- _generate_with_local_llm() -> Ollama API
            - deepseek-r1:7b
            - deepseek-r1:14b
            - llama3.2
            - Any Ollama model

RAGOrchestratorV2
    - self.llm_integrator (single instance)
    - query(question, model="claude")
```

## Usage Examples

### Creating RAG System
```python
# Initialize with both API and local LLM
rag = create_rag_system_v2(
    claude_api_key="sk-...",
    deepseek_api_url="http://192.168.1.14:11434/api/chat"
)

# Or just API
rag = create_rag_system_v2(claude_api_key="sk-...")

# Or just local LLM
rag = create_rag_system_v2(deepseek_api_url="http://...")
```

### Using Claude Model
```python
response = rag.query("What is AMF?", model="claude")
```

### Using Local LLM Models
```python
# DeepSeek R1 7B
response = rag.query("What is SMF?", model="deepseek-r1:7b")

# DeepSeek R1 14B
response = rag.query("Explain UPF", model="deepseek-r1:14b")

# Llama 3.2
response = rag.query("How does 5G work?", model="llama3.2")
```

## Workflow Diagram

```
User Query
    |
    v
RAGOrchestratorV2.query(question, model)
    |
    v
EnhancedKnowledgeRetriever.retrieve_with_cypher()
    |
    v
LLMIntegrator.generate_answer(query, chunks, cypher, model)
    |
    +-- if model == "claude"
    |       |
    |       v
    |   _generate_with_api()
    |       |
    |       v
    |   Anthropic API
    |
    +-- else (any other model name)
            |
            v
        _generate_with_local_llm()
            |
            v
        Ollama API (local)
```

## Benefits

1. **Simplicity**: Single class instead of multiple classes
2. **Maintainability**: Shared logic for context prep and question detection
3. **Extensibility**: Easy to add new model types by adding new routing logic
4. **Clean API**: Simple model parameter instead of complex integrator selection
5. **No Code Duplication**: Removed ~150 lines of duplicate code

## Testing

All 35 tests pass:
- 4 new tests for LLMIntegrator (Claude API and Local LLM)
- 31 existing tests for other components

```bash
pytest tests/test_rag_system_v2.py -v
# 35 passed in 1.58s
```

## Files Modified

### rag_system_v2.py
- Added LLMIntegrator class with unified interface
- Removed ClaudeIntegrator and DeepSeekIntegrator classes
- Updated RAGOrchestratorV2 to use single llm_integrator
- Removed backwards compatibility wrappers

### tests/test_rag_system_v2.py
- Replaced TestClaudeIntegratorV2 with TestLLMIntegrator
- Replaced TestDeepSeekIntegratorV2 tests
- Added tests for both API and local LLM modes
- All tests updated to use new unified API

## Migration Notes

Old code (no longer works):
```python
from rag_system_v2 import ClaudeIntegrator, DeepSeekIntegrator

claude = ClaudeIntegrator(api_key)
deepseek = DeepSeekIntegrator(url)
```

New code:
```python
from rag_system_v2 import LLMIntegrator

# Single integrator handles both
llm = LLMIntegrator(claude_api_key=api_key, local_llm_url=url)

# Use model parameter to route
answer = llm.generate_answer(query, chunks, cypher, model="claude")
answer = llm.generate_answer(query, chunks, cypher, model="deepseek-r1:7b")
```

## Future Enhancements

Easy to add new LLM providers:

```python
class LLMIntegrator:
    def __init__(self, claude_api_key=None, local_llm_url=None, openai_api_key=None):
        self.claude_client = ...
        self.local_llm_url = ...
        self.openai_client = OpenAI(api_key) if openai_api_key else None

    def generate_answer(self, query, chunks, cypher, model="claude"):
        if model == "claude":
            return self._generate_with_api(...)
        elif model.startswith("gpt-"):
            return self._generate_with_openai(...)
        else:
            return self._generate_with_local_llm(...)

    def _generate_with_openai(self, ...):
        # OpenAI-specific implementation
```
