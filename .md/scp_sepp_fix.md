# SCP vs SEPP Query Fix

## Problem

Query "What is the difference between SCP and SEPP in 5G Core?" was returning incorrect results because:

1. **Entity extraction bug**: "IN" from "in 5G Core" was matched as "Intelligent Network" entity
2. **Wrong SCP definition**: SCP Term node had `full_name = "Service Control Part"` (legacy IN term) instead of `"Service Communication Proxy"` (5G term)
3. **Complex UNION query**: The original comparison query was overly complex and prone to failures
4. **LLM hallucination**: Even with anti-hallucination rules, the wrong context led to fabricated answers

## Root Cause Analysis

### Issue 1: Common Word Matching
**File**: [`rag_system_v2.py:693-742`](../rag_system_v2.py#L693-L742)

The entity extraction matched "IN" (Intelligent Network) from the phrase "in 5G Core" because:
- Word boundary pattern `\bIN\b` matched "in"
- Term "IN" = "Intelligent Network" existed in KG

### Issue 2: Term Priority
**File**: [`orchestrator.py:545-580`](../orchestrator.py#L545-L580)

When merging terms from multiple specs, the first definition found was kept:
- `ts_23.094`: SCP = Service Control Part (legacy IN spec)
- `ts_29.500`: SCP = Service Communication Proxy (5G Core spec)

Since legacy spec was processed first, wrong definition was stored.

## Fixes Applied

### Fix 1: Common Word Blocklist
Added blocklist of common English words to prevent false entity matches:

```python
common_word_blocklist = {
    'IN', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IS', 'IT',
    'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE', 'A', 'I',
    'FOR', 'THE', 'AND', 'BUT', 'NOT', 'ARE', 'WAS', 'HAS', 'HAD',
    'CAN', 'MAY', 'ALL', 'ANY', 'NEW', 'OLD', 'ONE', 'TWO', 'WAY'
}
```

### Fix 2: 5G Spec Priority
Modified `_merge_terms()` to prioritize 5G Core specs over legacy:

```python
# 5G Core spec prefixes - these should take priority
_5g_spec_prefixes = ('ts_23.5', 'ts_29.5', 'ts_23.4', 'ts_29.2')

# Update full_name if new source is 5G spec and existing is not
if new_is_5g and not existing_is_5g:
    term_dict[abbr]['full_name'] = term.full_name
    term_dict[abbr]['primary_spec'] = term.source_spec
```

### Fix 3: Updated SCP in Neo4j
```cypher
MATCH (t:Term {abbreviation: "SCP"})
SET t.full_name = "Service Communication Proxy",
    t.primary_spec = "ts_29.500_3.2"
```

## Verification

### Before Fix
```
entities: [
  {type: 'concept', value: 'IN', full_name: 'Intelligent Network'},  // WRONG
  {type: 'concept', value: 'SCP', full_name: 'Service Control Part'}, // WRONG
  {type: 'network_function', value: 'SEPP', ...}
]
```

### After Fix
```
entities: [
  {type: 'network_function', value: 'SCP', full_name: 'Service Communication Proxy'},  // CORRECT
  {type: 'network_function', value: 'SEPP', full_name: 'Security and Edge Protection Proxy'}
]
```

## Debug Script

Created [`debug_scp_sepp.py`](../debug_scp_sepp.py) to test SCP/SEPP query:

```bash
# Quick test (skip RAG response)
python debug_scp_sepp.py --skip-rag

# Full test with LLM
python debug_scp_sepp.py --model gemma3:12b
```

## Fix 3: Multi-Step Comparison Strategy

**File**: [`rag_system_v2.py:857-874`](../rag_system_v2.py#L857-L874) and [`rag_system_v2.py:1077-1156`](../rag_system_v2.py#L1077-L1156)

Replaced complex UNION query with simple multi-step retrieval:

```python
# _generate_comparison_query now returns a marker
return f"COMPARISON_MULTI_STEP:{entity1}:{entity2}"

# _execute_comparison_retrieval handles the actual retrieval
def _execute_comparison_retrieval(self, entity1: str, entity2: str):
    # Step 1: Get Term definition for entity1
    # Step 2: Get Term definition for entity2
    # Step 3: Get function/role chunks for entity1
    # Step 4: Get function/role chunks for entity2
```

### Benefits
- Simpler individual queries (less prone to errors)
- Clear separation of concerns
- Better chunk labeling (know which chunks belong to which entity)
- Easier to debug and extend

### Retrieved Chunks Example
```
Chunk 1: Term definition - SCP: Service Communication Proxy
Chunk 2: Term definition - SEPP: Security and Edge Protection Proxy
Chunk 3-5: SCP function chunks from ts_29.510
Chunk 6-8: SEPP function chunks from ts_29.500, ts_29.510
```

## Note for Future KG Init

After running `python orchestrator.py init-kg`, Term nodes will automatically use 5G definitions due to the `_merge_terms()` fix.
