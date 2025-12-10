TELE QNA BENCHMARK - QUICK START
=================================

Test the RAG system against 95 representative telecom questions across 9 categories.

SETUP
-----

1. Ensure Neo4j is running:
   python orchestrator.py start-neo4j

2. Set API key (if using Claude):
   export CLAUDE_API_KEY="your-api-key"

BASIC USAGE
-----------

# Run full benchmark (95 questions)
python tests/benchmark/run_tele_qna_benchmark.py

# Use specific model
python tests/benchmark/run_tele_qna_benchmark.py --model deepseek-r1:14b

# Run specific category
python tests/benchmark/run_tele_qna_benchmark.py --category Definition

# Custom output file
python tests/benchmark/run_tele_qna_benchmark.py --output my_results.json

FILES
-----

tests/benchmark/
├── run_tele_qna_benchmark.py              - Benchmark runner script
├── tele_qna_benchmark_comprehensive.json  - 95 benchmark questions
├── benchmark_results_latest.json          - Latest results
└── __init__.py                            - Module init

tests/
├── test_mcq_extraction.py                 - MCQ extraction tests
├── test_mcq_llm_integration.py            - MCQ LLM integration tests
└── tele_qna_representative_set.json       - Alternative test set

EXAMPLE OUTPUT
--------------

Overall Results:
  Total Questions:     95
  Correct:             80 (84.21%)
  Incorrect:           15
  Total Time:          ~20 min

Accuracy by Category:
  Use Case/Application       10/10 (100.0%)
  Network Function           11/11 (100.0%)
  Definition                 12/13 ( 92.3%)
  Policy & QoS                9/10 ( 90.0%)
  Procedure                   9/11 ( 81.8%)
  Technical Detail            8/10 ( 80.0%)
  Security                    8/10 ( 80.0%)
  Comparison                  7/10 ( 70.0%)
  Architecture                6/10 ( 60.0%)

TROUBLESHOOTING
---------------

Neo4j not connected:
  python orchestrator.py check
  python orchestrator.py start-neo4j

CLAUDE_API_KEY not set:
  export CLAUDE_API_KEY="sk-ant-..."
  Or use local model: --model deepseek-r1:14b

See .md/benchmark_guide.md for complete documentation
