# Definition Query Fix - Entity Section Title Matching

**Date:** 2025-12-09
**Issue:** SCP query returning wrong definition ("Service Configuration Profile" instead of "Service Communication Proxy")

## Problem Analysis

When querying "What is SCP?", the system returned irrelevant chunks because:

1. The definition query only matched sections with titles containing:
   - 'definition', 'overview', 'introduction', 'general'
   - Or chunk_type = 'definition' or 'architecture'

2. The actual SCP definition is in section `ts_23.501_6.2.19` with title **"SCP"** - which didn't match any of the above patterns.

3. Result: System returned Abbreviations sections and Resource Definition sections instead of the actual SCP definition.

## Root Cause

```cypher
// Old WHERE clause - missing entity abbreviation match
AND (
    toLower(c.section_title) CONTAINS 'definition'
    OR toLower(c.section_title) CONTAINS 'overview'
    ...
)
```

The 3GPP specifications often have dedicated sections named after the entity (e.g., "SCP", "AMF", "SMF") that contain the actual definition, but these weren't being matched.

## Solution

Added section_title matching with entity abbreviation:

```cypher
// New WHERE clause - includes entity abbreviation match
AND (
    // Match section_title exactly with entity abbreviation (e.g., section "SCP")
    toLower(c.section_title) = toLower('{entity}')
    // Or section_title contains the full name
    OR (resolved_full_name IS NOT NULL AND toLower(c.section_title) CONTAINS toLower(resolved_full_name))
    // Or standard definition/overview sections
    OR toLower(c.section_title) CONTAINS 'definition'
    ...
)
```

Updated ORDER BY to prioritize entity-named sections:

```cypher
ORDER BY
    // Prioritize section_title matching entity abbreviation exactly
    CASE WHEN toLower(c.section_title) = toLower('{entity}') THEN 0
         WHEN resolved_full_name IS NOT NULL AND toLower(c.section_title) CONTAINS toLower(resolved_full_name) THEN 1
         WHEN size(target_specs) > 0 AND c.spec_id IN target_specs THEN 2
         WHEN toLower(c.section_title) CONTAINS 'definition' THEN 3
         WHEN toLower(c.section_title) CONTAINS 'overview' THEN 4
         ELSE 5 END,
    c.complexity_score ASC
```

## Results

### Before Fix
```
Chunk 1: ts_23.501_3.2 - Abbreviations (not useful)
Chunk 2: ts_23.094_3.1 - Definitions (wrong spec)
Chunk 3: ts_29.510_6.2.3.6.2 - Resource Definition (not relevant)
```

### After Fix
```
Chunk 1: ts_23.501_6.2.19 - SCP (correct definition!)
Chunk 2: ts_23.501_3.2 - Abbreviations
Chunk 3: ts_23.094_3.1 - Definitions
```

## Affected File

- `rag_system_v2.py`: `_generate_definition_query()` method (lines 816-843)

## Test Verification

All 35 tests in `tests/test_rag_system_v2.py` passed after the fix.
