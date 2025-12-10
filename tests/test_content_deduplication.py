#!/usr/bin/env python3
"""
Test content deduplication in HybridRetriever.
Ensures that chunks with very similar content are removed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_retriever import ScoredChunk, HybridRetriever
from unittest.mock import MagicMock


def create_mock_chunk(chunk_id: str, content: str, score: float = 1.0) -> ScoredChunk:
    """Helper to create a mock ScoredChunk"""
    return ScoredChunk(
        chunk_id=chunk_id,
        spec_id="TS 23.501",
        section_id="5.2.1",
        section_title="Test Section",
        content=content,
        chunk_type="definition",
        complexity_score=0.5,
        key_terms=["test"],
        retrieval_score=score,
        retrieval_method='vector',
        subject="standards_specifications",
        subject_confidence=0.9
    )


def test_similarity_calculation():
    """Test Jaccard similarity calculation"""
    print("=" * 70)
    print("Test 1: Similarity Calculation")
    print("=" * 70)

    # Create mock retriever (components are created internally)
    mock_driver = MagicMock()
    mock_generator = MagicMock()
    mock_generator.all_terms = {}

    retriever = HybridRetriever(
        neo4j_driver=mock_driver,
        cypher_generator=mock_generator
    )

    # Test cases
    test_cases = [
        {
            "content1": "The UPF is responsible for packet routing",
            "content2": "The UPF is responsible for packet routing",
            "expected_min": 0.99,  # Should be ~1.0 (identical)
            "description": "Identical content"
        },
        {
            "content1": "The UPF is responsible for packet routing and forwarding",
            "content2": "The UPF is responsible for packet routing",
            "expected_min": 0.70,  # High similarity (subset)
            "description": "Very similar content (subset)"
        },
        {
            "content1": "The UPF handles user plane data",
            "content2": "The AMF manages mobility",
            "expected_max": 0.30,  # Low similarity (different topics)
            "description": "Different content"
        },
        {
            "content1": "Service Communication Proxy (SCP) is a network function",
            "content2": "Service Communication Proxy (SCP) is a network function in 5G",
            "expected_min": 0.70,  # High similarity (minor addition)
            "description": "Similar with minor variation"
        }
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        similarity = retriever._calculate_content_similarity(
            test["content1"].lower(),
            test["content2"].lower()
        )

        print(f"\n{i}. {test['description']}")
        print(f"   Content 1: {test['content1'][:50]}...")
        print(f"   Content 2: {test['content2'][:50]}...")
        print(f"   Similarity: {similarity:.2f}")

        if "expected_min" in test:
            if similarity >= test["expected_min"]:
                print(f"   ✓ PASS: Similarity {similarity:.2f} >= {test['expected_min']}")
            else:
                print(f"   ✗ FAIL: Similarity {similarity:.2f} < {test['expected_min']}")
                all_passed = False

        if "expected_max" in test:
            if similarity <= test["expected_max"]:
                print(f"   ✓ PASS: Similarity {similarity:.2f} <= {test['expected_max']}")
            else:
                print(f"   ✗ FAIL: Similarity {similarity:.2f} > {test['expected_max']}")
                all_passed = False

    return all_passed


def test_deduplication():
    """Test content deduplication logic"""
    print("\n" + "=" * 70)
    print("Test 2: Content Deduplication")
    print("=" * 70)

    # Create mock retriever
    mock_driver = MagicMock()
    mock_generator = MagicMock()
    mock_generator.all_terms = {}

    retriever = HybridRetriever(
        neo4j_driver=mock_driver,
        cypher_generator=mock_generator
    )

    # Create test chunks with duplicate content
    # Using very similar content to ensure > 0.85 threshold
    chunks = [
        create_mock_chunk("chunk1", "The UPF is responsible for packet routing and forwarding", 1.0),
        create_mock_chunk("chunk2", "The UPF is responsible for packet routing and forwarding data", 0.95),  # Very similar - should be removed
        create_mock_chunk("chunk3", "The AMF manages access and mobility", 0.9),  # Different - should be kept
        create_mock_chunk("chunk4", "The UPF is responsible for packet routing and forwarding packets", 0.85),  # Very similar to chunk1 - should be removed
        create_mock_chunk("chunk5", "The SMF handles session management", 0.8),  # Different - should be kept
    ]

    print(f"\nInput: {len(chunks)} chunks")
    for chunk in chunks:
        print(f"  - {chunk.chunk_id} (score: {chunk.retrieval_score:.2f}): {chunk.content[:50]}...")

    # Run deduplication with 0.85 threshold
    deduplicated = retriever._deduplicate_by_content(chunks, similarity_threshold=0.85)

    print(f"\nOutput: {len(deduplicated)} chunks (removed {len(chunks) - len(deduplicated)} duplicates)")
    for chunk in deduplicated:
        print(f"  - {chunk.chunk_id} (score: {chunk.retrieval_score:.2f}): {chunk.content[:50]}...")

    # Verify results
    expected_kept = ["chunk1", "chunk3", "chunk5"]  # chunk2 and chunk4 should be removed
    actual_kept = [c.chunk_id for c in deduplicated]

    print(f"\nExpected chunks: {expected_kept}")
    print(f"Actual chunks: {actual_kept}")

    if set(actual_kept) == set(expected_kept):
        print("✓ PASS: Correct chunks kept after deduplication")
        return True
    else:
        print("✗ FAIL: Unexpected chunks after deduplication")
        return False


def test_deduplication_preserves_order():
    """Test that deduplication preserves score order"""
    print("\n" + "=" * 70)
    print("Test 3: Deduplication Preserves Score Order")
    print("=" * 70)

    mock_driver = MagicMock()
    mock_generator = MagicMock()
    mock_generator.all_terms = {}

    retriever = HybridRetriever(
        neo4j_driver=mock_driver,
        cypher_generator=mock_generator
    )

    # Create chunks sorted by score (descending)
    chunks = [
        create_mock_chunk("chunk1", "The UPF handles user plane", 1.0),
        create_mock_chunk("chunk2", "The AMF handles mobility", 0.9),
        create_mock_chunk("chunk3", "The SMF handles sessions", 0.8),
        create_mock_chunk("chunk4", "The PCF handles policy", 0.7),
    ]

    deduplicated = retriever._deduplicate_by_content(chunks, similarity_threshold=0.85)

    # Verify order is preserved (scores should be descending)
    scores = [c.retrieval_score for c in deduplicated]
    is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    print(f"Scores after deduplication: {scores}")

    if is_sorted:
        print("✓ PASS: Score order preserved")
        return True
    else:
        print("✗ FAIL: Score order not preserved")
        return False


def test_empty_and_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 70)
    print("Test 4: Edge Cases")
    print("=" * 70)

    mock_driver = MagicMock()
    mock_generator = MagicMock()
    mock_generator.all_terms = {}

    retriever = HybridRetriever(
        neo4j_driver=mock_driver,
        cypher_generator=mock_generator
    )

    all_passed = True

    # Test 1: Empty list
    result = retriever._deduplicate_by_content([], similarity_threshold=0.85)
    if len(result) == 0:
        print("✓ PASS: Empty list handled correctly")
    else:
        print("✗ FAIL: Empty list not handled correctly")
        all_passed = False

    # Test 2: Single chunk
    single = [create_mock_chunk("chunk1", "Test content", 1.0)]
    result = retriever._deduplicate_by_content(single, similarity_threshold=0.85)
    if len(result) == 1:
        print("✓ PASS: Single chunk handled correctly")
    else:
        print("✗ FAIL: Single chunk not handled correctly")
        all_passed = False

    # Test 3: All identical chunks
    identical = [
        create_mock_chunk("chunk1", "Identical content", 1.0),
        create_mock_chunk("chunk2", "Identical content", 0.9),
        create_mock_chunk("chunk3", "Identical content", 0.8),
    ]
    result = retriever._deduplicate_by_content(identical, similarity_threshold=0.85)
    if len(result) == 1:  # Should keep only the first (highest score)
        print("✓ PASS: All identical chunks reduced to 1")
    else:
        print(f"✗ FAIL: All identical chunks not reduced correctly (got {len(result)})")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    print("Testing Content Deduplication in HybridRetriever")
    print("=" * 70)

    test1 = test_similarity_calculation()
    test2 = test_deduplication()
    test3 = test_deduplication_preserves_order()
    test4 = test_empty_and_edge_cases()

    print("\n" + "=" * 70)
    if test1 and test2 and test3 and test4:
        print("🎉 ALL TESTS PASSED")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        exit(1)
