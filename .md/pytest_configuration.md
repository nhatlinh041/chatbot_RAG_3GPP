# Pytest Configuration

## Overview
Configuration for running regression tests in the 3GPP project, excluding benchmark tests from regular test runs.

## Configuration File: pytest.ini

### Current Settings

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --ignore=tests/test_tele_qna_benchmark.py --ignore=tests/test_tele_qna_simple.py
filterwarnings =
    ignore::DeprecationWarning
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    benchmark: marks tests as benchmark tests (deselect with '-m "not benchmark"')
```

### Key Settings Explained

#### testpaths
- Specifies that tests are located in `tests/` directory
- Pytest will search for tests in this directory

#### python_files, python_classes, python_functions
- Defines naming conventions for test discovery
- Files must start with `test_`
- Classes must start with `Test`
- Functions must start with `test_`

#### addopts (Additional Options)
- `-v` - Verbose output
- `--tb=short` - Shorter traceback format
- `--ignore=tests/test_tele_qna_benchmark.py` - Skip benchmark test file
- `--ignore=tests/test_tele_qna_simple.py` - Skip simple benchmark file

#### filterwarnings
- `ignore::DeprecationWarning` - Suppress deprecation warnings

#### markers
Custom markers for categorizing tests:
- `slow` - For slow-running tests
- `integration` - For integration tests (require external services)
- `benchmark` - For benchmark/performance tests

## Test Files

### Regular Tests (Included in pytest tests/)
1. **test_cypher_sanitizer.py** - Cypher injection prevention tests
2. **test_hybrid_term_extraction.py** - Term extraction tests
3. **test_logging_config.py** - Centralized logging tests
4. **test_rag_system_v2.py** - RAG system component tests

### Benchmark Tests (Excluded from pytest tests/)
1. **test_tele_qna_benchmark.py** - Full benchmark suite with metrics
2. **test_tele_qna_simple.py** - Simple benchmark test

## Running Tests

### Run All Regular Tests
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests (excludes benchmarks)
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Run Specific Test Files
```bash
# Run only sanitizer tests
pytest tests/test_cypher_sanitizer.py

# Run only logging tests
pytest tests/test_logging_config.py -v
```

### Run Tests by Marker
```bash
# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Skip integration tests
pytest tests/ -m "not integration"
```

### Run Benchmark Tests (Manual)
Benchmark tests must be run explicitly:

```bash
# Run full benchmark suite
python tests/test_tele_qna_benchmark.py --model claude

# Run simple benchmark
python tests/test_tele_qna_simple.py
```

## Why Exclude Benchmark Tests?

### Reasons for Exclusion:
1. **Long running time** - Benchmark tests take 10-30 minutes
2. **API costs** - Benchmark uses Claude API calls
3. **External dependencies** - Requires Neo4j and full KG setup
4. **Not regression tests** - Benchmark is for performance evaluation, not correctness
5. **Manual execution** - Should be run intentionally, not automatically

### Regular Tests vs Benchmark Tests

| Aspect | Regular Tests | Benchmark Tests |
|--------|--------------|----------------|
| Runtime | < 5 seconds | 10-30 minutes |
| Purpose | Correctness | Performance |
| Dependencies | Minimal | Full system |
| API Calls | Mocked | Real Claude API |
| Run Frequency | Every commit | Weekly/monthly |
| Execution | Automatic | Manual |

## Test Statistics

Current test suite:
- **Total regular tests**: 99 tests
- **Test files**: 4 files
- **Benchmark files**: 2 files (excluded)
- **Average runtime**: ~3-5 seconds
- **Coverage**: 85%+ on core modules

## Adding New Tests

### For Regular Tests
1. Create `test_*.py` in `tests/` directory
2. Follow naming conventions (test_*, Test*, test_*)
3. Use fixtures from `conftest.py`
4. Add appropriate markers if needed

Example:
```python
import pytest

class TestNewFeature:
    """Tests for new feature"""

    def test_basic_functionality(self):
        """Test should do something"""
        assert True

    @pytest.mark.slow
    def test_slow_operation(self):
        """This test takes a while"""
        pass
```

### For Benchmark Tests
1. Create in `tests/` but add to `--ignore` in pytest.ini
2. Make it runnable as standalone script
3. Add to CLAUDE.md under "Testing" section

## Continuous Integration

### CI Pipeline Recommendations
```yaml
# .github/workflows/tests.yml example
- name: Run tests
  run: |
    source .venv/bin/activate
    pytest tests/ --cov=. --cov-report=xml

- name: Run weekly benchmark (optional)
  if: github.event.schedule == '0 0 * * 0'  # Sunday midnight
  run: |
    python tests/test_tele_qna_benchmark.py --model claude
```

## Troubleshooting

### Issue: All tests run including benchmarks
**Solution**: Check pytest.ini has correct `--ignore` options

### Issue: Tests not discovered
**Solution**: Ensure test files follow naming convention `test_*.py`

### Issue: Import errors
**Solution**: Activate virtual environment and install dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: Neo4j connection errors in tests
**Solution**: Most tests don't need Neo4j, check if you're running benchmark by mistake

## Best Practices

1. **Keep tests fast** - Regular tests should run in seconds
2. **Mock external dependencies** - Don't require Neo4j, Claude API for unit tests
3. **Use markers** - Mark slow/integration tests appropriately
4. **Separate concerns** - Benchmarks are not regression tests
5. **Update documentation** - Update this file when adding new test categories

## Related Files

- `pytest.ini` - Main pytest configuration
- `tests/conftest.py` - Shared fixtures and test setup
- `tests/README_BENCHMARK.md` - Benchmark test documentation
- `CLAUDE.md` - Project-level testing documentation
