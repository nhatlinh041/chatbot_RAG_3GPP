# RAG Query Redesign - Term Dictionary & Query Strategy

## New Approach: Extract Abbreviations to Neo4j

Instead of hardcoding NF_FULL_NAMES, we will:
1. **Extract** abbreviation/definition from each TS document's Section 3 (Definitions & Abbreviations)
2. **Store** as `Term` nodes in Neo4j with links to source specs
3. **Query** Term nodes first → get full name + relevant specs → search content in those specs

## Neo4j Schema Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Term Node                                │
├─────────────────────────────────────────────────────────────────┤
│ abbreviation: "SCP"                                              │
│ full_name: "Service Communication Proxy"                        │
│ term_type: "abbreviation" | "definition"                        │
│ source_specs: ["ts_29.500", "ts_23.501"]                        │
│ primary_spec: "ts_29.500"  (first occurrence)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ DEFINED_IN
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Document Node                              │
│                     (existing schema)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Retrieval Flow

```
User Question: "What is SCP?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Stage 1: Term Resolution (NEW)                         │
│  Query: MATCH (t:Term) WHERE t.abbreviation = 'SCP'             │
│  Result: full_name="Service Communication Proxy"                 │
│          source_specs=["ts_29.500", "ts_23.501"]                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Stage 2: Targeted Content Search                       │
│  Query: Search for "Service Communication Proxy" OR "SCP"       │
│         ONLY in specs: ts_29.500, ts_23.501                     │
│  Filter: section_title contains term OR definition/overview     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Stage 3: Function/Role Search                          │
│  Query: Search for role/function of SCP in same specs           │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Term Extractor Module (NEW)

Create `term_extractor.py`:
```python
class TermExtractor:
    """Extract abbreviations and definitions from 3GPP JSON chunks"""

    def extract_abbreviations(self, chunk_content: str) -> List[Dict]:
        """
        Parse abbreviation section content like:
        "SCP	Service Communication Proxy"

        Returns: [{"abbr": "SCP", "full_name": "Service Communication Proxy"}]
        """

    def extract_definitions(self, chunk_content: str) -> List[Dict]:
        """
        Parse definition section content like:
        "NF service: A functionality exposed by an NF..."

        Returns: [{"term": "NF service", "definition": "A functionality..."}]
        """
```

### Phase 2: Update KG_builder

Add to `SimpleKGProcessorV2`:
```python
def create_term_nodes(self, session):
    """Create Term nodes from abbreviation/definition chunks"""

    # Find all abbreviation chunks (section 3.2)
    result = session.run("""
        MATCH (c:Chunk)
        WHERE toLower(c.section_title) CONTAINS 'abbreviation'
        RETURN c.chunk_id, c.spec_id, c.content
    """)

    for record in result:
        terms = self.term_extractor.extract_abbreviations(record['content'])
        for term in terms:
            session.run("""
                MERGE (t:Term {abbreviation: $abbr})
                ON CREATE SET t.full_name = $full_name,
                              t.term_type = 'abbreviation',
                              t.source_specs = [$spec_id]
                ON MATCH SET t.source_specs =
                    CASE WHEN NOT $spec_id IN t.source_specs
                         THEN t.source_specs + $spec_id
                         ELSE t.source_specs END

                WITH t
                MATCH (d:Document {spec_id: $spec_id})
                MERGE (t)-[:DEFINED_IN]->(d)
            """, abbr=term['abbr'], full_name=term['full_name'], spec_id=record['spec_id'])
```

### Phase 3: Update RAG Query Generation

```python
def _generate_definition_query(self, question: str, analysis: Dict) -> str:
    """Multi-stage definition query using Term nodes"""
    entity = analysis['entities'][0]['value']

    return f"""
    // Stage 1: Resolve term
    OPTIONAL MATCH (t:Term)
    WHERE t.abbreviation = '{entity}' OR toLower(t.full_name) CONTAINS toLower('{entity}')
    WITH t,
         CASE WHEN t IS NOT NULL THEN t.full_name ELSE '{entity}' END as search_term,
         CASE WHEN t IS NOT NULL THEN t.source_specs ELSE [] END as target_specs

    // Stage 2: Search in relevant specs first, then fallback to all
    MATCH (c:Chunk)
    WHERE (
        // Primary: search in source specs
        (size(target_specs) > 0 AND c.spec_id IN target_specs)
        OR
        // Fallback: search anywhere if no term found
        size(target_specs) = 0
    )
    AND (
        toLower(c.content) CONTAINS toLower('{entity}')
        OR toLower(c.content) CONTAINS toLower(search_term)
    )
    AND (
        toLower(c.section_title) CONTAINS toLower(search_term)
        OR toLower(c.section_title) CONTAINS 'definition'
        OR toLower(c.section_title) CONTAINS 'overview'
        OR c.chunk_type = 'definition'
    )

    RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
           c.content, c.chunk_type, c.complexity_score, c.key_terms,
           t.full_name as resolved_term, t.source_specs as term_sources
    ORDER BY
        CASE WHEN c.spec_id IN target_specs THEN 0 ELSE 1 END,
        c.complexity_score ASC
    LIMIT 6
    """
```

---

## Current Problems

### Problem 1: Definition Query
**Question:** "What is SCP in 5G Core network?"

**Current behavior:**
- Query tìm chunks có `section_title` chứa "definition/overview" VÀ content chứa "SCP"
- Không tìm được definition chính thức của SCP vì:
  - SCP = "Service Communication Proxy" - tên đầy đủ nằm trong glossary/abbreviation sections
  - Definition thực sự có thể nằm trong section khác (VD: "6.2.x SCP" trong TS 23.501)

**Expected behavior:**
1. Tìm full term của SCP (Service Communication Proxy) trong abbreviation/glossary sections
2. Tìm definition section dành riêng cho SCP (VD: "6.2.x Service Communication Proxy")
3. Tìm functional description của SCP

### Problem 2: Comparison Query
**Question:** "Compare UPF and SMF"

**Current behavior:**
- Chỉ tìm chunks chứa CẢ HAI "UPF" VÀ "SMF"
- Kết quả: chunks nói về cả 2 nhưng không có definition/function riêng

**Expected behavior:**
1. Tìm definition của UPF (full name + description)
2. Tìm definition của SMF (full name + description)
3. Tìm functions/roles của UPF
4. Tìm functions/roles của SMF
5. (Optional) Tìm chunks nói về relationship giữa UPF và SMF

## Proposed Solution

### Strategy 1: Multi-Stage Retrieval

```
┌─────────────────────────────────────────────────────────────────┐
│                    Question Analysis                             │
│  "What is SCP?" → type=definition, entity=SCP                   │
│  "Compare UPF and SMF" → type=comparison, entities=[UPF, SMF]   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 1: Term Resolution                         │
│  • Search abbreviation/glossary sections                        │
│  • Find full name: SCP → "Service Communication Proxy"          │
│  • Find spec reference: TS 23.501, TS 29.xxx                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 2: Definition Retrieval                    │
│  • Search section titles containing full term                   │
│  • Search "Network Functions" sections                          │
│  • Search definition/overview sections with entity              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 3: Function/Role Retrieval                 │
│  • Search for "functions of {entity}"                           │
│  • Search for "role of {entity}"                                │
│  • Search for "{entity} is responsible for"                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           (For Comparison) Stage 4: Combine Results              │
│  • Merge definition chunks for both entities                    │
│  • Add relationship chunks (optional)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Strategy 2: Enhanced Cypher Queries

#### New Definition Query (3-part UNION)

```cypher
// Part 1: Find abbreviation/full name
MATCH (c:Chunk)
WHERE (toLower(c.section_title) CONTAINS 'abbreviation'
       OR toLower(c.section_title) CONTAINS 'glossary'
       OR toLower(c.section_title) CONTAINS 'definition')
AND toLower(c.content) CONTAINS toLower('SCP')
RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
       c.content, c.chunk_type, c.complexity_score, c.key_terms, 1 as priority

UNION

// Part 2: Find dedicated section for the entity
MATCH (c:Chunk)
WHERE toLower(c.section_title) CONTAINS toLower('service communication proxy')
   OR (toLower(c.section_title) =~ '.*\\bscp\\b.*'
       AND toLower(c.section_title) CONTAINS 'function')
RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
       c.content, c.chunk_type, c.complexity_score, c.key_terms, 2 as priority

UNION

// Part 3: Find general description
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS toLower('SCP')
AND (toLower(c.content) CONTAINS 'service communication proxy'
     OR toLower(c.content) CONTAINS 'is responsible'
     OR toLower(c.content) CONTAINS 'provides')
RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
       c.content, c.chunk_type, c.complexity_score, c.key_terms, 3 as priority

ORDER BY priority ASC, complexity_score DESC
LIMIT 8
```

#### New Comparison Query (Parallel retrieval for each entity)

```cypher
// Get definitions for BOTH entities separately
// Entity 1: UPF
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS toLower('UPF')
AND (toLower(c.section_title) CONTAINS 'user plane function'
     OR toLower(c.section_title) CONTAINS 'upf'
     OR toLower(c.content) CONTAINS 'user plane function')
WITH c, 'UPF' as entity

UNION

// Entity 2: SMF
MATCH (c:Chunk)
WHERE toLower(c.content) CONTAINS toLower('SMF')
AND (toLower(c.section_title) CONTAINS 'session management function'
     OR toLower(c.section_title) CONTAINS 'smf'
     OR toLower(c.content) CONTAINS 'session management function')
WITH c, 'SMF' as entity

RETURN c.chunk_id, c.spec_id, c.section_id, c.section_title,
       c.content, c.chunk_type, c.complexity_score, c.key_terms, entity
ORDER BY entity, c.complexity_score DESC
LIMIT 10
```

### Implementation Plan

1. **Add NF Term Mapping Dictionary**
   ```python
   NF_FULL_NAMES = {
       'SCP': 'Service Communication Proxy',
       'AMF': 'Access and Mobility Management Function',
       'SMF': 'Session Management Function',
       'UPF': 'User Plane Function',
       'PCF': 'Policy Control Function',
       'UDM': 'Unified Data Management',
       'AUSF': 'Authentication Server Function',
       'NSSF': 'Network Slice Selection Function',
       'NEF': 'Network Exposure Function',
       'NRF': 'NF Repository Function',
       # Add more...
   }
   ```

2. **Modify `_generate_definition_query()`**
   - Add term resolution step
   - Use UNION query for multi-part search
   - Include abbreviation/glossary search
   - Search for dedicated NF sections

3. **Modify `_generate_comparison_query()`**
   - Retrieve definitions for EACH entity separately
   - Include functions/roles for each entity
   - Label chunks by entity for LLM context

4. **Add result post-processing**
   - Group chunks by entity for comparison questions
   - Ensure balanced representation of both entities

## Expected Improvements

| Question Type | Before | After |
|--------------|--------|-------|
| "What is SCP?" | Generic chunks mentioning SCP | Full name + dedicated section + role description |
| "Compare UPF and SMF" | Chunks containing both | Definition + functions for each, labeled by entity |

## Testing

After implementation, verify with:
1. "What is SCP in 5G Core?" → Should return Service Communication Proxy definition
2. "What is TSCTSF?" → Should return Time Sensitive Communication and Time Synchronization Function
3. "Compare UPF and SMF" → Should return definitions and functions for BOTH entities
4. "Difference between AMF and SMF" → Should return separate descriptions for each

## Implementation Status (2025-12-07)

### Completed:
1. **term_extractor.py** - Module to extract abbreviations from JSON chunks
2. **KG_builder.ipynb** - Added TermNodeBuilder class to create Term nodes
3. **rag_system_v2.py** - Updated query strategies:
   - `_generate_definition_query()` - Uses Term nodes for full name resolution
   - `_generate_comparison_query()` - Returns multi-step marker
   - `_execute_comparison_retrieval()` - NEW: Multi-step retrieval for comparisons
4. **orchestrator.py** - 5G spec priority for term merging

### Multi-Step Comparison Strategy (Current)

Complex UNION queries were replaced with simpler multi-step retrieval:

```python
# _generate_comparison_query returns marker
return f"COMPARISON_MULTI_STEP:{entity1}:{entity2}"

# _execute_comparison_retrieval handles actual retrieval
def _execute_comparison_retrieval(self, entity1: str, entity2: str):
    # Step 1: Get Term definition for entity1
    # Step 2: Get Term definition for entity2
    # Step 3: Get function/role chunks for entity1
    # Step 4: Get function/role chunks for entity2
```

Benefits:
- Simpler individual queries (less prone to errors)
- Clear separation of concerns
- Better chunk labeling (know which chunks belong to which entity)
- Easier to debug and extend

### Files Modified:
- `term_extractor.py` (extraction module)
- `KG_builder.ipynb` (Term node builder)
- `rag_system_v2.py` (query generation and retrieval)
- `orchestrator.py` (5G spec priority in `_merge_terms()`)

### To Apply Changes:
1. Run `python orchestrator.py init-kg` to create Term nodes with 5G priority
2. Test with sample queries

## Notes

- Term nodes store abbreviation → full_name mapping with source_specs list
- 5G specs (ts_23.5xx, ts_29.5xx) take priority over legacy specs
- Comparison queries use multi-step retrieval for better accuracy
- Common word blocklist prevents false entity matches (e.g., "IN" from "in 5G Core")
