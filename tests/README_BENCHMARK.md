TELE QNA BENCHMARK - QUICK START
=================================

Test the RAG system against 25 representative telecom questions

SETUP
-----

1. Ensure Neo4j is running:
   python orchestrator.py start-neo4j

2. Set API key (if using Claude):
   export CLAUDE_API_KEY="your-api-key"

BASIC USAGE
-----------

# Run full benchmark (25 questions)
python run_tele_qna_benchmark.py

# Quick test (first 5 questions)
python run_tele_qna_benchmark.py --limit 5

# Use local LLM instead of Claude
python run_tele_qna_benchmark.py --model deepseek-r1:7b

# Quiet mode (summary only)
python run_tele_qna_benchmark.py --quiet

FILES
-----

run_tele_qna_benchmark.py              - Standalone test runner
tests/test_tele_qna_benchmark.py       - Pytest-compatible version
tests/tele_qna_representative_set.json - 25 test questions
tests/results/benchmark_results.json   - Output results (created after run)

EXAMPLE OUTPUT
--------------

Overall Results:
  Total Questions:     25
  Correct:             20 (80.0%)
  Incorrect:           4
  Failed:              1
  Total Time:          45.67s
  Avg Time/Question:   1.83s

Accuracy by Subject:
  Standards specifications       11/13 ( 84.6%)
  Lexicon                         7/ 8 ( 87.5%)
  Standards overview              2/ 3 ( 66.7%)

TROUBLESHOOTING
---------------

Neo4j not connected:
  python orchestrator.py check
  python orchestrator.py start-neo4j

CLAUDE_API_KEY not set:
  export CLAUDE_API_KEY="sk-ant-..."
  Or use local model: --model deepseek-r1:7b

See .md/tele_qna_benchmark_usage.md for complete documentation
