#!/usr/bin/env python3
"""
Standalone script to run tele_qna benchmark test
Usage: python run_tele_qna_benchmark.py [--model MODEL] [--limit N]
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rag_system_v2 import create_rag_system_v2
from logging_config import setup_centralized_logging, get_logger

# Setup logging
setup_centralized_logging()
logger = get_logger('TeleQNA_Benchmark')


def load_questions(questions_file: str, limit: int = None):
    """Load questions from JSON file"""
    with open(questions_file, 'r') as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    logger.info(f"Loaded {len(questions)} questions from {questions_file}")
    return questions


def evaluate_answer(question: dict, rag_answer: str) -> tuple:
    """
    Evaluate if RAG answer is correct

    Returns:
        (is_correct, details)
    """
    expected_idx = question['answer']
    expected_text = question['choices'][expected_idx].lower()
    rag_lower = rag_answer.lower()

    # Simple keyword matching - check if expected answer appears in response
    is_correct = expected_text in rag_lower

    # Alternative: check if any key terms from expected answer appear
    # This is a simple heuristic and could be improved

    return is_correct, {
        'expected_index': expected_idx,
        'expected_text': question['choices'][expected_idx],
        'rag_answer': rag_answer[:200]  # Truncate for display
    }


def run_benchmark(questions, rag_system, model='claude', verbose=True):
    """Run benchmark on questions"""
    results = []
    correct = 0
    failed = 0
    total_time = 0

    for idx, q in enumerate(questions, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Question {idx}/{len(questions)}")
            print(f"{'='*80}")
            print(f"Q: {q['question']}")
            print(f"Subject: {q['subject']} | Category: {q.get('category', 'N/A')}")
            print(f"Expected: {q['choices'][q['answer']]}")

        try:
            # Query RAG system
            start_time = time.time()
            response = rag_system.query(q['question'], model=model)
            duration = time.time() - start_time
            total_time += duration

            # Evaluate
            is_correct, details = evaluate_answer(q, response.answer)

            if is_correct:
                correct += 1

            if verbose:
                print(f"\nRAG Answer ({duration:.2f}s):")
                print(response.answer[:300])
                print(f"\nSources: {len(response.sources)}")
                print(f"Strategy: {response.retrieval_strategy}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

            results.append({
                'question_id': idx,
                'question': q['question'],
                'subject': q['subject'],
                'category': q.get('category', 'Unknown'),
                'is_correct': is_correct,
                'duration': duration,
                'sources_count': len(response.sources),
                'retrieval_strategy': response.retrieval_strategy,
                **details
            })

        except Exception as e:
            logger.error(f"Failed on question {idx}: {e}")
            failed += 1

            if verbose:
                print(f"\n✗ ERROR: {e}")

            results.append({
                'question_id': idx,
                'question': q['question'],
                'subject': q['subject'],
                'error': str(e),
                'is_correct': False,
                'duration': 0
            })

    return results, {
        'total': len(questions),
        'correct': correct,
        'incorrect': len(questions) - correct - failed,
        'failed': failed,
        'accuracy': (correct / len(questions) * 100) if questions else 0,
        'total_time': total_time,
        'avg_time': total_time / len(questions) if questions else 0
    }


def print_summary(stats, results):
    """Print benchmark summary"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    print(f"\nOverall Results:")
    print(f"  Total Questions:     {stats['total']}")
    print(f"  Correct:             {stats['correct']} ({stats['accuracy']:.1f}%)")
    print(f"  Incorrect:           {stats['incorrect']}")
    print(f"  Failed:              {stats['failed']}")
    print(f"  Total Time:          {stats['total_time']:.2f}s")
    print(f"  Avg Time/Question:   {stats['avg_time']:.2f}s")

    # Group by subject
    by_subject = {}
    for r in results:
        subj = r['subject']
        if subj not in by_subject:
            by_subject[subj] = {'total': 0, 'correct': 0}
        by_subject[subj]['total'] += 1
        if r['is_correct']:
            by_subject[subj]['correct'] += 1

    print(f"\nAccuracy by Subject:")
    for subj, data in sorted(by_subject.items()):
        acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"  {subj:30} {data['correct']:2}/{data['total']:2} ({acc:5.1f}%)")

    # Group by category
    by_category = {}
    for r in results:
        cat = r.get('category', 'Unknown')
        if cat not in by_category:
            by_category[cat] = {'total': 0, 'correct': 0}
        by_category[cat]['total'] += 1
        if r['is_correct']:
            by_category[cat]['correct'] += 1

    print(f"\nAccuracy by Category:")
    for cat, data in sorted(by_category.items()):
        acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"  {cat:30} {data['correct']:2}/{data['total']:2} ({acc:5.1f}%)")

    print("\n" + "="*80)


def save_results(results, stats, output_file):
    """Save results to JSON file"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'results': results
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run tele_qna benchmark')
    parser.add_argument('--model', default='claude',
                       help='Model to use (claude, deepseek-r1:7b, llama3.2, etc.)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions to test')
    parser.add_argument('--questions', default='tests/tele_qna_representative_set.json',
                       help='Path to questions JSON file')
    parser.add_argument('--output', default='tests/results/benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--neo4j-uri', default='neo4j://localhost:7687',
                       help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--neo4j-password', default='password',
                       help='Neo4j password')

    args = parser.parse_args()

    # Get API keys from environment
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    deepseek_api_url = os.getenv('DEEPSEEK_API_URL', 'http://192.168.1.14:11434/api/chat')

    if not claude_api_key and args.model == 'claude':
        print("ERROR: CLAUDE_API_KEY environment variable not set")
        print("Set it with: export CLAUDE_API_KEY='your-api-key'")
        return 1

    print("Initializing RAG system...")
    print(f"Model: {args.model}")
    print(f"Neo4j: {args.neo4j_uri}")

    # Initialize RAG system
    rag_system = create_rag_system_v2(
        claude_api_key=claude_api_key,
        deepseek_api_url=deepseek_api_url,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )

    try:
        # Load questions
        questions = load_questions(args.questions, args.limit)

        # Run benchmark
        print(f"\nStarting benchmark with {len(questions)} questions...")
        results, stats = run_benchmark(
            questions,
            rag_system,
            model=args.model,
            verbose=not args.quiet
        )

        # Print summary
        print_summary(stats, results)

        # Save results
        save_results(results, stats, args.output)

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        rag_system.close()
        print("\nRAG system closed")


if __name__ == '__main__':
    sys.exit(main())
