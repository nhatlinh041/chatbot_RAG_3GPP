# Anti-Hallucination Rules for LLM Responses

## Problem

LLMs can hallucinate information when the retrieved context doesn't contain relevant information for the user's question. For example, when asking about SCP vs SEPP, if the retrieved chunks don't contain proper definitions, the LLM may fabricate information based on general knowledge rather than admitting lack of information.

## Solution

Added explicit anti-hallucination instructions to both Claude API and local LLM (Ollama) prompts.

**File**: [`rag_system_v2.py:171-176`](../rag_system_v2.py#L171-L176) (Claude API)
**File**: [`rag_system_v2.py:242-247`](../rag_system_v2.py#L242-L247) (Local LLM)

## Anti-Hallucination Rules Added

```text
**CRITICAL - Anti-Hallucination Rules:**
- ONLY use information explicitly stated in the provided context
- If the context does not contain relevant information about the question, respond with:
  "I could not find relevant information about [topic] in the retrieved 3GPP specifications."
- DO NOT make up, infer, or hallucinate any information not present in the context
- DO NOT use your general knowledge about 3GPP - ONLY use the provided context
- If unsure whether something is in the context, say you don't have that information
```

## Expected Behavior

### Before (Hallucination)
```
Q: What is the difference between SCP and SEPP?
A: [LLM fabricates detailed comparison using general knowledge,
    even when context chunks don't contain relevant information]
```

### After (Honest Response)
```
Q: What is the difference between SCP and SEPP?
A: I could not find relevant information about SCP and SEPP comparison
   in the retrieved 3GPP specifications. Please try rephrasing your
   question or ask about a different topic.
```

## Trade-offs

1. **Pros**: Prevents false information, builds user trust, makes it clear when context is insufficient
2. **Cons**: May refuse to answer some questions that could be partially answered

## Related Changes

- Also fixed CypherSanitizer to allow Cypher comments (was incorrectly blocking queries)
- This ensures better context retrieval, reducing the likelihood of needing to refuse

## Testing

Run with local LLM to test:
```bash
# Start chatbot
python orchestrator.py run

# Ask question that may not have good context
# LLM should refuse rather than hallucinate
```
