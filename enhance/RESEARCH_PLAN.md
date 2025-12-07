# KẾ HOẠCH NGHIÊN CỨU CHO BÀI BÁO CÁO
Ngày: 2025-12-07

## MỤC TIÊU

Cải thiện độ chính xác (accuracy) của hệ thống RAG trên tele_qna dataset thông qua:
1. Hybrid retrieval (Vector + Knowledge Graph)
2. Query understanding & expansion
3. Advanced reasoning với KG structure

**Không quan tâm:**
- Version control của 3GPP specs
- Production optimization
- Cost optimization
- Web UI features

**Chỉ tập trung:**
- Knowledge Graph quality
- Retrieval accuracy
- Scientific contributions

---

## PHASE 0: DOCUMENT COLLECTION & PROCESSING (Week 0 - Prerequisites)

### Mục tiêu
Thu thập đủ 3GPP documents và process thành JSON format

### 0.1 Document Collection

**Current status:**
- Kiểm tra folder `3GPP_JSON_DOC/processed_json_v2/`
- Đếm số lượng specs đã có

**Required documents for tele_qna:**

Current status: 269 specs (mostly TS 23.xxx and TS 29.xxx)
Missing: TS 24.xxx, TS 36.xxx, TS 37.xxx, TS 38.xxx series

Danh sách đầy đủ các series cần bổ sung:

```bash
# ===== SERIES 21: LTE to 5G Migration =====
TS 21.905 - Vocabulary for 3GPP Specifications

# ===== SERIES 22: Service Requirements =====
TS 22.261 - Service requirements for the 5G system
TS 22.104 - Service requirements for cyber-physical control applications in vertical domains
TS 22.186 - Enhancement of 3GPP support for V2X scenarios

# ===== SERIES 23: Technical Realization (đã có nhiều) =====
# Core Architecture
TS 23.501 - System architecture for the 5G System (5GS) ✓
TS 23.502 - Procedures for the 5G System (5GS) ✓
TS 23.503 - Policy and charging control framework for the 5G System ✓

# Additional needed:
TS 23.214 - Architecture enhancements for control and user plane separation of EPC nodes
TS 23.247 - Architectural enhancements for 5G multicast-broadcast services
TS 23.273 - 5G System (5GS) Location Services (LCS)
TS 23.288 - Architecture enhancements for 5G System (5GS) to support network data analytics services
TS 23.316 - Wireless and wireline convergence access support for the 5G System (5GS)
TS 23.434 - Service Enabler Architecture Layer for Verticals (SEAL) ✓
TS 23.548 - 5G System Enhancements for Edge Computing

# ===== SERIES 24: Signaling Protocols (THIẾU HOÀN TOÀN) =====
# NAS (Non-Access Stratum)
TS 24.501 - Non-Access-Stratum (NAS) protocol for 5G System (5GS); Stage 3
TS 24.502 - Access to the 3GPP 5G Core Network (5GCN) via non-3GPP access networks
TS 24.526 - User Equipment (UE) policies for 5G System (5GS); Stage 3

# Session Management
TS 24.008 - Mobile radio interface Layer 3 specification; Core network protocols; Stage 3
TS 24.301 - Non-Access-Stratum (NAS) protocol for Evolved Packet System (EPS)

# Application Layer
TS 24.229 - IP multimedia call control protocol based on Session Initiation Protocol (SIP) and Session Description Protocol (SDP); Stage 3

# ===== SERIES 25: Radio Access (UMTS/WCDMA) =====
TS 25.401 - UTRAN overall description

# ===== SERIES 29: Core Network Signaling (đã có nhiều) =====
# Service-Based Architecture (SBA)
TS 29.500 - 5G System; Technical Realization of Service Based Architecture; Stage 3 ✓
TS 29.501 - 5G System; Principles and Guidelines for Services Definition; Stage 3
TS 29.571 - 5G System; Common Data Types for Service Based Interfaces; Stage 3 ✓

# Network Functions Services
TS 29.502 - 5G System; Session Management Services; Stage 3 ✓
TS 29.503 - 5G System; Unified Data Management Services; Stage 3 ✓
TS 29.504 - 5G System; Unified Data Repository Services; Stage 3 ✓
TS 29.505 - 5G System; Usage of the Unified Data Repository services for Subscription Data; Stage 3
TS 29.507 - 5G System; Access and Mobility Policy Control Service; Stage 3
TS 29.508 - 5G System; Session Management Event Exposure Service; Stage 3
TS 29.509 - 5G System; Authentication Server Services; Stage 3
TS 29.510 - 5G System; Network function repository services; Stage 3
TS 29.512 - 5G System; Session Management Policy Control Service; Stage 3
TS 29.513 - 5G System; Policy and Charging Control signalling flows and QoS parameter mapping; Stage 3
TS 29.514 - 5G System; Policy Authorization Service; Stage 3
TS 29.517 - 5G System; Application Function Event Exposure Service; Stage 3
TS 29.518 - 5G System; Access and Mobility Management Services; Stage 3 ✓
TS 29.519 - 5G System; Usage of the Unified Data Repository Service for Policy Data, Application Data and Structured Data for Exposure; Stage 3
TS 29.520 - 5G System; Network Data Analytics Services; Stage 3
TS 29.521 - 5G System; Binding Support Management Service; Stage 3
TS 29.522 - 5G System; Network Exposure Function Northbound APIs; Stage 3 ✓

# ===== SERIES 33: Security =====
TS 33.501 - Security architecture and procedures for 5G System
TS 33.210 - Network Domain Security (NDS); IP network layer security
TS 33.401 - 3GPP System Architecture Evolution (SAE); Security architecture

# ===== SERIES 36: LTE (E-UTRAN) Radio =====
TS 36.300 - Evolved Universal Terrestrial Radio Access (E-UTRA) and Evolved Universal Terrestrial Radio Access Network (E-UTRAN); Overall description; Stage 2
TS 36.331 - Evolved Universal Terrestrial Radio Access (E-UTRA); Radio Resource Control (RRC); Protocol specification
TS 36.401 - Evolved Universal Terrestrial Radio Access Network (E-UTRAN); Architecture description
TS 36.413 - Evolved Universal Terrestrial Radio Access Network (E-UTRAN); S1 Application Protocol (S1AP)
TS 36.423 - Evolved Universal Terrestrial Radio Access Network (E-UTRAN); X2 Application Protocol (X2AP)

# ===== SERIES 37: Multiple Radio Access Technologies =====
TS 37.340 - Evolved Universal Terrestrial Radio Access (E-UTRA) and NR; Multi-connectivity; Stage 2

# ===== SERIES 38: 5G NR (New Radio) - THIẾU HOÀN TOÀN =====
# Overall Description
TS 38.300 - NR; NR and NG-RAN Overall description; Stage 2

# Physical Layer
TS 38.201 - NR; Physical layer; General description
TS 38.202 - NR; Services provided by the physical layer
TS 38.211 - NR; Physical channels and modulation
TS 38.212 - NR; Multiplexing and channel coding
TS 38.213 - NR; Physical layer procedures for control
TS 38.214 - NR; Physical layer procedures for data
TS 38.215 - NR; Physical layer measurements

# MAC, RLC, PDCP
TS 38.321 - NR; Medium Access Control (MAC) protocol specification
TS 38.322 - NR; Radio Link Control (RLC) protocol specification
TS 38.323 - NR; Packet Data Convergence Protocol (PDCP) specification
TS 38.331 - NR; Radio Resource Control (RRC); Protocol specification

# NG-RAN Architecture
TS 38.401 - NG-RAN; Architecture description
TS 38.410 - NG-RAN; NG general aspects and principles
TS 38.413 - NG-RAN; NG Application Protocol (NGAP)
TS 38.420 - NG-RAN; Xn general aspects and principles
TS 38.423 - NG-RAN; Xn Application Protocol (XnAP)
TS 38.470 - NG-RAN; F1 general aspects and principles
TS 38.473 - NG-RAN; F1 Application Protocol (F1AP)

# Radio Requirements
TS 38.101-1 - NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone
TS 38.101-2 - NR; User Equipment (UE) radio transmission and reception; Part 2: Range 2 Standalone
TS 38.104 - NR; Base Station (BS) radio transmission and reception

# RRM (Radio Resource Management)
TS 38.133 - NR; Requirements for support of radio resource management
TS 38.300 - NR; Overall description; Stage 2
```

**Download priority (theo thứ tự quan trọng):**

1. **Priority 1 - Critical for tele_qna (TS 38.xxx series):**
   - TS 38.300, 38.401, 38.413, 38.423, 38.473
   - TS 38.211-215 (Physical layer)
   - TS 38.321-323, 38.331 (Protocol stack)

2. **Priority 2 - NAS & Signaling (TS 24.xxx series):**
   - TS 24.501, 24.502, 24.008, 24.301

3. **Priority 3 - Additional SBA services (TS 29.xxx):**
   - TS 29.501, 29.505-514, 29.517-522

4. **Priority 4 - Security & LTE (TS 33.xxx, 36.xxx):**
   - TS 33.501, 33.401
   - TS 36.300, 36.401, 36.413

5. **Priority 5 - Requirements & Multi-RAT (TS 22.xxx, 37.xxx):**
   - TS 22.261, 22.104
   - TS 37.340
```

**Tasks:**
```bash
# 1. Check current documents
cd 3GPP_JSON_DOC/processed_json_v2/
ls -l *.json | wc -l  # Count current docs

# 2. Download missing specs from 3GPP portal
# Visit: https://www.3gpp.org/ftp/Specs/archive/
# Download .docx versions of missing specs

# 3. Place in document_processing/input/
mkdir -p document_processing/input
# Copy .docx files here
```

### 0.2 Document Processing

**Process documents to JSON:**
```bash
# Run document processing pipeline
cd document_processing

# Process all .docx files
python process_documents.py --input input/ --output ../3GPP_JSON_DOC/processed_json_v2/

# Verify output
ls -l ../3GPP_JSON_DOC/processed_json_v2/*.json
```

**Expected output structure:**
```json
{
  "metadata": {
    "specification_id": "TS_23.501",
    "version": "17.5.0",
    "title": "System architecture for the 5G System (5GS)"
  },
  "chunks": [
    {
      "chunk_id": "TS_23.501_chunk_001",
      "section_id": "5.2.3",
      "section_title": "Network Functions",
      "content": "...",
      "chunk_type": "text",
      "content_metadata": {
        "word_count": 250,
        "complexity_score": 0.7,
        "key_terms": ["AMF", "SMF", "network function"]
      },
      "cross_references": {
        "external": [
          {"target_spec": "TS_23.502", "ref_id": "4.2.2", ...}
        ]
      }
    }
  ]
}
```

### 0.3 Quality Check

**Verify processed documents:**
```python
# enhance/verify_documents.py
import json
from pathlib import Path

def verify_json_quality():
    json_dir = Path("3GPP_JSON_DOC/processed_json_v2")

    stats = {
        'total_docs': 0,
        'total_chunks': 0,
        'docs_with_refs': 0,
        'avg_chunks_per_doc': 0
    }

    for json_file in json_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        stats['total_docs'] += 1
        stats['total_chunks'] += len(data['chunks'])

        # Check cross-references
        has_refs = any(
            chunk.get('cross_references', {}).get('external', [])
            for chunk in data['chunks']
        )
        if has_refs:
            stats['docs_with_refs'] += 1

    stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['total_docs']

    print("Document Processing Quality:")
    print(f"  Total specs: {stats['total_docs']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg chunks/doc: {stats['avg_chunks_per_doc']:.1f}")
    print(f"  Docs with cross-refs: {stats['docs_with_refs']}")

    # Target: At least 50 specs, 10K+ chunks
    assert stats['total_docs'] >= 50, "Need at least 50 specs"
    assert stats['total_chunks'] >= 10000, "Need at least 10K chunks"

    return stats
```

**Run verification:**
```bash
python enhance/verify_documents.py
```

### 0.4 Initialize Knowledge Graph

**Build KG from processed JSONs:**
```bash
# Start Neo4j
python orchestrator.py start-neo4j

# Wait for Neo4j ready (up to 60s)
# Then initialize KG
python orchestrator.py init-kg

# Or use all command
python orchestrator.py all --init-kg
```

**Verify KG:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    # Count nodes
    result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
    for record in result:
        print(f"{record['label']}: {record['count']}")

    # Count relationships
    result = session.run("MATCH ()-[r]->() RETURN type(r) as rel, count(r) as count")
    for record in result:
        print(f"{record['rel']}: {record['count']}")

# Expected:
# Document: ~50-100
# Chunk: ~10,000-50,000
# CONTAINS: ~10,000-50,000
# REFERENCES_SPEC: ~1,000-5,000
```

### 0.5 Success Criteria (Phase 0)

✅ **Must have:**
- Ít nhất 50 3GPP specs processed
- Ít nhất 10,000 chunks trong KG
- Cross-references working (REFERENCES_SPEC relationships)
- Neo4j KG initialized successfully

⚠️ **If not meeting criteria:**
1. Download more specs from 3GPP portal
2. Re-run document processing
3. Debug processing errors
4. Check JSON format correctness

**Time estimate:** 2-3 days (assuming manual document download)

---

## PHASE 1: BASELINE & EVALUATION FRAMEWORK (Week 1)

### Mục tiêu
Setup cơ sở để đo lường improvement

### 1.1 Prepare tele_qna Dataset
```python
# Cấu trúc tele_qna
{
    "question": "What is the role of AMF in 5G?",
    "answer": "The AMF (Access and Mobility...",
    "relevant_specs": ["TS 23.501"],
    "relevant_sections": ["5.2.3", "6.1"],
    "question_type": "definition"
}
```

**Tasks:**
- Load tele_qna dataset
- Phân loại questions theo type (definition, procedure, comparison, ...)
- Create train/validation/test splits (70/15/15)

**File:**
- `enhance/tele_qna_loader.py`
- `enhance/dataset_stats.py`

### 1.2 Baseline Measurements
Đo hiện tại system performance:

**Metrics:**
- **Retrieval Accuracy:**
  - Recall@k (k=5,10,20,50)
  - Precision@k
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)

- **Answer Quality:**
  - BLEU score vs ground truth
  - ROUGE score
  - BERTScore
  - Human evaluation (sample 100 questions)

**File:**
- `enhance/evaluate_baseline.py`
- `enhance/metrics.py`

### 1.3 Error Analysis
Phân tích khi nào system fail:

```python
def analyze_failures(results):
    failures = [r for r in results if r['correct'] == False]

    # Categorize failures
    categories = {
        'retrieval_miss': [],  # Không retrieve được đúng chunks
        'context_insufficient': [],  # Retrieve đúng nhưng thiếu context
        'reasoning_error': [],  # Context đủ nhưng LLM reasoning sai
        'ambiguous_question': []  # Question không rõ ràng
    }

    return categories
```

**Expected baseline (current system):**
- Recall@10: ~60-70%
- Precision@10: ~50-60%
- BLEU: ~0.3-0.4
- BERTScore: ~0.7-0.75

---

## PHASE 2: KNOWLEDGE GRAPH ENHANCEMENT (Week 2-3)

### Mục tiêu
Cải thiện KG structure để support better reasoning

### 2.1 Enhanced Entity Extraction

**Entities cần extract:**
```python
entities = {
    'network_functions': ['AMF', 'SMF', 'UPF', 'PCF', ...],
    'interfaces': ['N1', 'N2', 'N3', 'N4', ...],
    'procedures': ['Registration', 'Session Establishment', ...],
    'protocols': ['NAS', 'NGAP', 'HTTP/2', 'PFCP', ...],
    'parameters': ['QoS Flow', 'PDU Session', '5QI', ...]
}
```

**Implementation:**
```python
class EnhancedEntityExtractor:
    def extract_entities(self, chunk_text):
        entities = []

        # NER cho network functions
        nf_entities = self.extract_network_functions(chunk_text)

        # Pattern matching cho interfaces (N1, N2, ...)
        interface_entities = self.extract_interfaces(chunk_text)

        # Extract procedures (looking for procedure definitions)
        procedure_entities = self.extract_procedures(chunk_text)

        return {
            'network_functions': nf_entities,
            'interfaces': interface_entities,
            'procedures': procedure_entities
        }
```

**File:**
- `enhance/entity_extractor_v2.py`
- `tests/test_entity_extraction.py`

### 2.2 New Relationship Types

**Relationships cần thêm:**

```cypher
// Functional relationships
(AMF)-[:MANAGES]->(Mobility)
(SMF)-[:MANAGES]->(PDUSession)

// Interface relationships
(AMF)-[:USES_INTERFACE{name: 'N1'}]->(UE)
(AMF)-[:USES_INTERFACE{name: 'N2'}]->(gNB)

// Procedural relationships
(Registration)-[:INVOLVES]->(AMF)
(Registration)-[:INVOLVES]->(UDM)
(Registration)-[:STEP_1]->(NASMessage)

// Dependency relationships
(SessionEstablishment)-[:REQUIRES]->(Registration)
(Handover)-[:REQUIRES]->(ActiveSession)

// Specification relationships
(AMF)-[:DEFINED_IN]->(TS_23501)
(Procedure)-[:DESCRIBED_IN]->(Section_5_2_3)
```

**Implementation:**
```python
class RelationshipExtractor:
    def extract_relationships(self, chunk, all_entities):
        relationships = []

        # Extract "X manages Y" patterns
        manages_rels = self.extract_manages_pattern(chunk)

        # Extract "X uses interface Y" patterns
        interface_rels = self.extract_interface_pattern(chunk)

        # Extract procedural dependencies
        # "Before X, Y must complete"
        dependency_rels = self.extract_dependency_pattern(chunk)

        return relationships
```

**File:**
- `enhance/relationship_extractor.py`
- `enhance/kg_schema_v2.cypher`

### 2.3 KG Quality Metrics

```python
class KGQualityAnalyzer:
    def analyze_completeness(self):
        # % of entities có đầy đủ relationships
        # % of chunks được link với entities
        pass

    def analyze_consistency(self):
        # Check contradictions
        # Check missing mandatory relationships
        pass

    def analyze_coverage(self):
        # % of tele_qna questions có relevant entities trong KG
        pass
```

**Target metrics:**
- Entity coverage: >95% (tele_qna entities có trong KG)
- Relationship density: >3 rels/entity average
- Connectivity: >90% nodes reachable from any node

---

## PHASE 3: HYBRID RETRIEVAL (Week 4-5)

### Mục tiêu
Kết hợp Vector Search + Graph Traversal

### 3.1 Vector Embeddings

**Model selection:**
```python
models_to_test = [
    'all-MiniLM-L6-v2',  # Fast, baseline
    'all-mpnet-base-v2',  # Better quality
    'msmarco-distilbert-base-v4',  # Trained on QA
    'instructor-base'  # Can use instructions
]

# Test từng model trên tele_qna
# Pick model có best Recall@50
```

**Embedding strategy:**
```python
# Option 1: Chunk-level embeddings
embedding = model.encode(chunk.content)

# Option 2: Enriched embeddings (better for KG)
enriched_text = f"""
{chunk.section_title}
{chunk.content}
Entities: {', '.join(chunk.entities)}
Related to: {', '.join(chunk.related_specs)}
"""
embedding = model.encode(enriched_text)

# Compare both options
```

### 3.2 Hybrid Retrieval Strategy

**Architecture:**
```
Query
  |
  +-- Vector Search (fast, semantic)
  |     |
  |     +-> Top 30 chunks
  |
  +-- Graph Traversal (precise, structured)
        |
        +-> Related entities & chunks
        |
        v
  [Combine & Rerank]
        |
        v
  Top 10 final chunks
```

**Implementation:**
```python
class HybridRetriever:
    def retrieve(self, question, question_type):
        # 1. Analyze question
        entities = self.extract_question_entities(question)
        focus = self.detect_question_focus(question, question_type)

        # 2. Vector search
        vector_results = self.vector_search(question, top_k=30)

        # 3. Graph traversal strategy dựa vào question type
        if question_type == 'definition':
            # Start từ entity, get direct properties
            graph_results = self.traverse_definition(entities)

        elif question_type == 'procedure':
            # Follow procedural relationships
            graph_results = self.traverse_procedure(entities)

        elif question_type == 'comparison':
            # Get properties của 2 entities để so sánh
            graph_results = self.traverse_comparison(entities)

        # 4. Combine
        combined = self.combine_rrf(vector_results, graph_results)

        # 5. Rerank
        final = self.rerank(question, combined, top_k=10)

        return final
```

### 3.3 Graph Traversal Patterns

**For Definition Questions:**
```cypher
// "What is AMF?"
MATCH (e:Entity {name: 'AMF'})
MATCH (e)-[r]->(related)
RETURN e, r, related
LIMIT 20
```

**For Procedure Questions:**
```cypher
// "How does registration work?"
MATCH path = (p:Procedure {name: 'Registration'})-[:STEP*]->(steps)
RETURN path
ORDER BY steps.sequence
```

**For Comparison Questions:**
```cypher
// "Compare AMF and SMF"
MATCH (e1:Entity {name: 'AMF'})-[r1]->(props1)
MATCH (e2:Entity {name: 'SMF'})-[r2]->(props2)
RETURN e1, r1, props1, e2, r2, props2
```

**File:**
- `enhance/hybrid_retriever.py`
- `enhance/graph_traversal_patterns.py`
- `tests/test_hybrid_retrieval.py`

---

## PHASE 4: QUERY UNDERSTANDING (Week 6)

### Mục tiêu
Better understanding of questions

### 4.1 Question Type Classification

```python
class QuestionClassifier:
    types = [
        'definition',      # What is X?
        'function',        # What does X do?
        'procedure',       # How does X work?
        'comparison',      # Compare X and Y
        'relationship',    # How X relates to Y?
        'location',        # Where is X?
        'causation',       # Why X happens?
        'multiple_choice'  # Which of the following?
    ]

    def classify(self, question):
        # Use patterns + classifier model
        # Return type + confidence
        pass
```

### 4.2 Entity Recognition in Questions

```python
class QuestionEntityExtractor:
    def extract(self, question):
        # "What is the role of AMF in 5G registration?"
        # -> entities: ['AMF', 'registration']
        # -> focus: 'AMF'
        # -> context: 'registration'

        return {
            'entities': [...],
            'focus_entity': '...',
            'context': [...]
        }
```

### 4.3 Query Expansion for Telecom

**Telecom-specific expansions:**
```python
expansions = {
    'UE': ['User Equipment', 'mobile device', 'terminal'],
    'gNB': ['gNodeB', 'base station', 'RAN'],
    'registration': [
        'initial registration',
        'mobility registration update',
        'periodic registration update'
    ]
}

# Expand based on context
if 'registration' in question and 'types' in question:
    # Expand to all registration types
    expanded = add_registration_types(question)
```

**File:**
- `enhance/query_understanding.py`
- `enhance/telecom_glossary.json`

---

## PHASE 5: ADVANCED REASONING (Week 7)

### Mục tiêu
Improve reasoning using KG structure

### 5.1 Multi-hop Reasoning

**Example:**
```
Q: "Which NF is responsible for policy in PDU session?"

Answer requires multi-hop:
1. PDU Session -[MANAGED_BY]-> SMF
2. SMF -[USES_INTERFACE:N7]-> PCF
3. PCF -[MANAGES]-> Policy
→ Answer: PCF
```

**Implementation:**
```python
def multi_hop_reasoning(question, max_hops=3):
    # Extract start entities
    start_entities = extract_entities(question)

    # Extract goal (what we're looking for)
    goal = extract_goal(question)  # e.g., "responsible for policy"

    # Graph traversal
    paths = []
    for entity in start_entities:
        for hop in range(1, max_hops+1):
            cypher = f"""
            MATCH path = (start:Entity {{name: '{entity}'}})-[*{hop}]->(goal)
            WHERE goal.type = '{goal}'
            RETURN path
            ORDER BY length(path)
            LIMIT 10
            """
            paths.extend(execute_cypher(cypher))

    # Select best path based on question
    best_path = select_relevant_path(paths, question)

    return best_path
```

### 5.2 Contradiction Detection

```python
def detect_contradictions(retrieved_chunks):
    # Find contradicting information
    # E.g., "AMF uses N1" vs "AMF uses N2 only"

    contradictions = []
    for i, chunk1 in enumerate(retrieved_chunks):
        for chunk2 in retrieved_chunks[i+1:]:
            if are_contradicting(chunk1, chunk2):
                contradictions.append((chunk1, chunk2))

    return contradictions
```

### 5.3 Confidence Scoring

```python
def calculate_confidence(answer, retrieved_chunks, graph_path):
    confidence_factors = {
        'retrieval_score': 0.3,  # Vector similarity
        'graph_support': 0.3,    # Có graph path support không
        'source_quality': 0.2,   # Quality của chunks
        'consistency': 0.2       # Consistent across chunks
    }

    # Calculate each factor
    # Combine to final confidence

    return confidence_score
```

---

## PHASE 6: EVALUATION & PAPER WRITING (Week 8)

### 6.1 Comprehensive Evaluation

**Experiments:**

1. **Ablation Study:**
   - Baseline (Cypher only)
   - + Vector Search
   - + Enhanced KG
   - + Hybrid Retrieval
   - + Query Understanding
   - + Multi-hop Reasoning

2. **Component Analysis:**
   - Effect của different embedding models
   - Effect của graph traversal depth
   - Effect của reranking strategies
   - Effect của query expansion

3. **Question Type Analysis:**
   - Performance per question type
   - Which types benefit most từ KG?

**Metrics to report:**
- Recall@5, @10, @20
- Precision@5, @10
- MRR, MAP
- BLEU, ROUGE, BERTScore
- Latency
- Human evaluation results

### 6.2 Visualizations

**Figures for paper:**
1. KG statistics (nodes, edges, density)
2. Performance comparison (bar charts)
3. Ablation study results
4. Per-question-type performance
5. Example retrieval paths
6. Error analysis breakdown

**File:**
- `enhance/generate_paper_figures.py`
- `enhance/results_analysis.ipynb`

### 6.3 Paper Structure

```
1. Introduction
   - Problem: QA on technical documents
   - Contribution: Hybrid retrieval with KG

2. Related Work
   - RAG systems
   - Knowledge Graphs for QA
   - Telecom-specific QA

3. Methodology
   3.1 Knowledge Graph Construction
   3.2 Hybrid Retrieval
   3.3 Query Understanding

4. Experiments
   4.1 Dataset (tele_qna)
   4.2 Baselines
   4.3 Results
   4.4 Ablation Study
   4.5 Error Analysis

5. Discussion
   - When KG helps
   - Limitations
   - Future work

6. Conclusion
```

---

## EXPECTED RESULTS

### Baseline (Current)
- Recall@10: ~65%
- MRR: ~0.55
- BLEU: ~0.35

### After Phase 3 (Hybrid)
- Recall@10: ~80% (+15%)
- MRR: ~0.70 (+0.15)
- BLEU: ~0.45 (+0.10)

### After Phase 5 (Full System)
- Recall@10: ~85% (+20%)
- MRR: ~0.75 (+0.20)
- BLEU: ~0.50 (+0.15)

## SCIENTIFIC CONTRIBUTIONS

1. **Novel KG Schema for Telecom:**
   - Entity types
   - Relationship types
   - Extraction methods

2. **Hybrid Retrieval Strategy:**
   - Combination của vector + graph
   - Question-type-specific traversal
   - Reranking approach

3. **Query Understanding for Technical Domains:**
   - Entity recognition
   - Query expansion
   - Type classification

4. **Benchmark:**
   - Evaluation framework cho telecom QA
   - Analysis of what makes questions hard
   - Error categorization

## FILES TO CREATE

```
enhance/
├── RESEARCH_PLAN.md                 (this file)
├── tele_qna_loader.py
├── evaluate_baseline.py
├── metrics.py
├── entity_extractor_v2.py
├── relationship_extractor.py
├── kg_quality_analyzer.py
├── hybrid_retriever.py
├── graph_traversal_patterns.py
├── query_understanding.py
├── multi_hop_reasoning.py
├── telecom_glossary.json
├── generate_paper_figures.py
├── results_analysis.ipynb
└── tests/
    ├── test_hybrid_retrieval.py
    ├── test_query_understanding.py
    └── test_kg_quality.py
```

## TIMELINE

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Baseline | Evaluation framework + baseline numbers |
| 2-3 | KG Enhancement | Enhanced KG with new entities & relationships |
| 4-5 | Hybrid Retrieval | Working hybrid system |
| 6 | Query Understanding | Query expansion & classification |
| 7 | Advanced Reasoning | Multi-hop reasoning |
| 8 | Evaluation | Full results + paper draft |

**Total: 8 weeks**
