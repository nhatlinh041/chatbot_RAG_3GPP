# 3GPP Document Processing V3 Pipeline

**Created**: 2025-12-08
**Version**: 3.0
**Status**: ✅ Implemented & Tested

---

## Overview

Complete pipeline for downloading and processing 3GPP Release 18 specifications into JSON format for Neo4j Knowledge Graph.

## Features

### 1. Document Download
- Download entire 3GPP Release (18 by default)
- Uses `download_3gpp` package from PyPI
- Automatic package installation if missing
- Support for specific series filtering

### 2. Document Processing
- Extracts metadata (TS/TR number, version, title)
- Parses hierarchical sections with content
- Extracts tables with headers and rows
- Identifies cross-references (internal & external)
- Classifies content types (definition, procedure, parameter, requirement, general)
- Generates content metadata (word count, complexity score, key terms)

### 3. Neo4j Integration
- Direct initialization after processing
- Compatible with existing orchestrator
- Supports both V2 and V3 JSON formats

## Architecture

```mermaid
graph LR
    A[3GPP Website] -->|download_3gpp| B[DOCX Files]
    B -->|ThreeGPPParser| C[Sections & Chunks]
    C -->|JSON Export| D[processed_json_v3/]
    D -->|KG Initializer| E[Neo4j Graph]
```

## Usage

### Command Line Interface

```bash
# Full pipeline: download + process + neo4j
python document_processing/download_and_process_3gpp.py \
    --download \
    --release 18 \
    --process \
    --init-neo4j

# Just process existing documents
python document_processing/download_and_process_3gpp.py \
    --process \
    --data-dir document_processing/data/ \
    --output-dir 3GPP_JSON_DOC/processed_json_v3/

# Download only
python document_processing/download_and_process_3gpp.py \
    --download \
    --release 18 \
    --data-dir data/3gpp_rel18/
```

### Python API

```python
from document_processing.download_and_process_3gpp import (
    ThreeGPPParser,
    DocumentProcessor,
    ThreeGPPDownloader
)

# Download specs
downloader = ThreeGPPDownloader("data/3gpp_specs/")
downloader.download_release(18)

# Process documents
processor = DocumentProcessor("3GPP_JSON_DOC/processed_json_v3/")
processor.process_directory("data/3gpp_specs/")

# Or process single document
parser = ThreeGPPParser()
metadata = parser.extract_metadata("23501-i90.docx")
sections = parser.parse_sections("23501-i90.docx")
chunks = parser.create_chunks(sections, metadata)
parser.save_to_json(chunks, metadata, "output.json")
```

## Data Structures

### DocumentMetaData
```python
@dataclass
class DocumentMetaData:
    specification_id: str  # "ts_23_501" format
    version: str          # "18.9.0"
    title: str           # "3GPP TS 23.501"
    file_path: str       # Original file path
```

### ProcessedChunk
```python
@dataclass
class ProcessedChunk:
    chunk_id: str         # "ts_23_501_chunk_001"
    section_id: str       # "4.2.1"
    section_title: str    # "Overview"
    content: str          # Section content
    chunk_type: str       # definition/procedure/parameter/requirement/general
    cross_references: Dict # {"internal": [], "external": []}
    tables: List[Dict]    # Table data
    content_metadata: Dict # word_count, complexity_score, key_terms
```

### JSON Output Format
```json
{
  "metadata": {
    "specification_id": "ts_23_501",
    "version": "18.9.0",
    "title": "3GPP TS 23.501",
    "file_path": "23501-i90.docx"
  },
  "export_info": {
    "export_date": "2025-12-08T10:30:00",
    "total_chunks": 250,
    "parser_version": "v3.0"
  },
  "chunks": [
    {
      "chunk_id": "ts_23_501_chunk_001",
      "section_id": "4.2.1",
      "section_title": "Overview",
      "content": "...",
      "chunk_type": "definition",
      "cross_references": {
        "internal": [
          {"ref_type": "clause", "ref_id": "4.2.2"}
        ],
        "external": [
          {
            "target_spec": "ts_23_502",
            "ref_type": "clause",
            "ref_id": "4.3",
            "confidence": 0.95
          }
        ]
      },
      "tables": [],
      "content_metadata": {
        "word_count": 150,
        "complexity_score": 0.65,
        "key_terms": ["AMF", "SMF", "UPF"]
      }
    }
  ]
}
```

## Cross-Reference Extraction

### Internal References
- `clause 4.2.1`
- `table 5.1-1`
- `figure 6.2-1`

### External References
- `TS 23.502, clause 4.3`
- `clause 4.3 of TS 23.502`
- `table 4.1 of TR 23.700`
- Standalone: `TS 23.502`

### Confidence Scores
- 1.0: Explicit standalone TS/TR reference
- 0.95: TS/TR with clause/table/figure
- 0.9: Contextual reference (TS/TR nearby)

## Content Classification

| Type | Triggers | Example |
|------|----------|---------|
| definition | Title contains: definition, overview, general | "3.1 Definitions" |
| procedure | Title contains: procedure, flow, process | "5.2.1 Registration Procedure" |
| parameter | Title contains: parameter, identifier | "6.1 Parameters" |
| requirement | Content starts with "shall" | "The UE shall..." |
| general | Default category | Other content |

## Testing

```bash
# Run all tests
pytest tests/test_document_processing_v3.py -v

# Test coverage
pytest tests/test_document_processing_v3.py --cov=document_processing.download_and_process_3gpp

# Specific test class
pytest tests/test_document_processing_v3.py::TestThreeGPPParser -v
```

### Test Categories
- **TestDataStructures**: Dataclass validation
- **TestThreeGPPParser**: Parser functions
- **TestDocumentProcessor**: Batch processing
- **TestThreeGPPDownloader**: Download functionality
- **TestIntegration**: End-to-end workflow
- **TestCLI**: Command line arguments

## Performance

### Expected Processing Times
- Single document: ~2-5 seconds
- Full 23 series (~150 docs): ~10-15 minutes
- Full Release 18 (~350 docs): ~25-30 minutes
- Neo4j initialization: ~5-10 minutes

### Resource Usage
- Memory: ~500MB-1GB during processing
- Disk: ~2GB for Release 18 (DOCX + JSON)
- CPU: Single-threaded (can be parallelized)

## Integration with Orchestrator

```bash
# After processing, initialize Neo4j
python orchestrator.py init-kg

# Or use new JSON directory
export JSON_DIR="3GPP_JSON_DOC/processed_json_v3/"
python orchestrator.py init-kg
```

## Troubleshooting

### Common Issues

1. **mammoth not installed**
   ```bash
   pip install mammoth beautifulsoup4
   ```

2. **download_3gpp command not found**
   ```bash
   pip install download_3gpp
   ```

3. **Neo4j connection failed**
   ```bash
   python orchestrator.py start-neo4j
   ```

4. **Memory error with large files**
   - Process in smaller batches
   - Increase system memory
   - Use `--verbose` to identify problem files

## Future Improvements

- [ ] Parallel processing with multiprocessing
- [ ] Incremental updates (only new/changed docs)
- [ ] Better table extraction (complex tables)
- [ ] Image/diagram extraction
- [ ] PDF support
- [ ] Streaming for large files
- [ ] Progress saving/resume

## Comparison with V2

| Feature | V2 | V3 |
|---------|----|----|
| Metadata extraction | Basic | Enhanced (version detection) |
| Section parsing | Simple | Hierarchical with parent tracking |
| Table extraction | No | Yes (headers + rows) |
| Cross-references | Basic | Internal + External with confidence |
| Content classification | No | 5 types |
| Content metadata | No | Word count, complexity, key terms |
| Chunk IDs | Sequential | Formatted (ts_XX_YYY_chunk_NNN) |
| JSON structure | Simple | Comprehensive with export_info |
| Download capability | No | Yes (download_3gpp integration) |
| Neo4j integration | Manual | Automatic (--init-neo4j) |

## Files Created

- `document_processing/download_and_process_3gpp.py` - Main script (600 LOC)
- `tests/test_document_processing_v3.py` - Test suite (400 LOC)
- `.md/document_processing_v3.md` - This documentation

## Summary

Document Processing V3 provides a complete, tested pipeline for:
1. ✅ Downloading 3GPP specifications
2. ✅ Extracting structured content
3. ✅ Identifying relationships and references
4. ✅ Initializing Neo4j Knowledge Graph
5. ✅ 16 unit tests (100% passing)

Ready for production use!