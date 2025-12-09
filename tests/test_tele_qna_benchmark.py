"""
Test script for tele_qna representative question set
Runs RAG system against benchmark questions and evaluates performance
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system_v2 import create_rag_system_v2, RAGResponse
from logging_config import setup_centralized_logging, get_logger

# Setup logging
setup_centralized_logging()
logger = get_logger('Tele_QNA_Benchmark')


class TeleQNABenchmark:
    """Benchmark test runner for tele_qna representative set"""

    def __init__(self, questions_file: str):
        """
        Initialize benchmark with questions file

        Args:
            questions_file: Path to JSON file with questions
        """
        self.questions_file = Path(questions_file)
        self.questions = self._load_questions()
        self.results = []

    def _load_questions(self) -> List[Dict]:
        """Load questions from JSON file"""
        with open(self.questions_file, 'r') as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {self.questions_file}")
        return questions

    def run_benchmark(self, rag_system, model: str = "claude") -> Dict:
        """
        Run benchmark test suite

        Args:
            rag_system: Initialized RAG system
            model: Model to use (claude, deepseek-r1:7b, llama3.2, etc.)

        Returns:
            Dictionary with benchmark results and statistics
        """
        logger.info(f"Starting benchmark with model: {model}")
        start_time = time.time()

        total_questions = len(self.questions)
        correct_answers = 0
        failed_queries = 0

        for idx, q in enumerate(self.questions, 1):
            logger.info(f"\n[{idx}/{total_questions}] Processing question: {q['question'][:80]}...")

            try:
                # Query RAG system
                q_start = time.time()
                response = rag_system.query(q['question'], model=model)
                q_duration = time.time() - q_start

                # Evaluate answer
                is_correct = self._evaluate_answer(q, response)

                result = {
                    'question_id': idx,
                    'question': q['question'],
                    'expected_answer': q['choices'][q['answer']],
                    'expected_index': q['answer'],
                    'subject': q['subject'],
                    'category': q.get('category', 'Unknown'),
                    'rag_answer': response.answer,
                    'is_correct': is_correct,
                    'sources_count': len(response.sources),
                    'cypher_query': response.cypher_query,
                    'retrieval_strategy': response.retrieval_strategy,
                    'query_duration': q_duration
                }

                self.results.append(result)

                if is_correct:
                    correct_answers += 1
                    logger.info(f"✓ CORRECT")
                else:
                    logger.info(f"✗ INCORRECT")

                logger.info(f"Duration: {q_duration:.2f}s, Sources: {len(response.sources)}")

            except Exception as e:
                logger.error(f"✗ FAILED: {str(e)}")
                failed_queries += 1

                result = {
                    'question_id': idx,
                    'question': q['question'],
                    'expected_answer': q['choices'][q['answer']],
                    'subject': q['subject'],
                    'error': str(e),
                    'is_correct': False,
                    'query_duration': 0
                }
                self.results.append(result)

        total_time = time.time() - start_time

        # Calculate statistics
        stats = self._calculate_statistics(total_time, correct_answers, failed_queries)

        return stats

    def _evaluate_answer(self, question: Dict, response: RAGResponse) -> bool:
        """
        Evaluate if RAG answer is correct

        Simple evaluation: check if expected answer text appears in RAG response
        """
        expected_answer = question['choices'][question['answer']].lower()
        rag_answer = response.answer.lower()

        # Check if expected answer is mentioned in the response
        # More sophisticated evaluation could be added here
        return expected_answer in rag_answer

    def _calculate_statistics(self, total_time: float, correct: int, failed: int) -> Dict:
        """Calculate benchmark statistics"""
        total = len(self.questions)

        stats = {
            'total_questions': total,
            'correct_answers': correct,
            'incorrect_answers': total - correct - failed,
            'failed_queries': failed,
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'total_time': total_time,
            'avg_time_per_question': total_time / total if total > 0 else 0,
            'questions_per_minute': (total / total_time * 60) if total_time > 0 else 0
        }

        # Statistics by subject
        stats['by_subject'] = self._stats_by_category('subject')

        # Statistics by question category
        stats['by_category'] = self._stats_by_category('category')

        return stats

    def _stats_by_category(self, category_key: str) -> Dict:
        """Calculate statistics grouped by a category"""
        categories = {}

        for result in self.results:
            cat = result.get(category_key, 'Unknown')

            if cat not in categories:
                categories[cat] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'failed': 0
                }

            categories[cat]['total'] += 1

            if 'error' in result:
                categories[cat]['failed'] += 1
            elif result.get('is_correct', False):
                categories[cat]['correct'] += 1
            else:
                categories[cat]['incorrect'] += 1

        # Calculate accuracy for each category
        for cat in categories:
            total = categories[cat]['total']
            correct = categories[cat]['correct']
            categories[cat]['accuracy'] = (correct / total * 100) if total > 0 else 0

        return categories

    def save_results(self, output_file: str):
        """Save detailed results to JSON file"""
        output_path = Path(output_file)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'questions_file': str(self.questions_file),
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self, stats: Dict):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("TELE QNA BENCHMARK RESULTS")
        print("="*80)

        print(f"\nOverall Statistics:")
        print(f"  Total Questions:     {stats['total_questions']}")
        print(f"  Correct Answers:     {stats['correct_answers']}")
        print(f"  Incorrect Answers:   {stats['incorrect_answers']}")
        print(f"  Failed Queries:      {stats['failed_queries']}")
        print(f"  Accuracy:            {stats['accuracy']:.2f}%")
        print(f"  Total Time:          {stats['total_time']:.2f}s")
        print(f"  Avg Time/Question:   {stats['avg_time_per_question']:.2f}s")
        print(f"  Questions/Minute:    {stats['questions_per_minute']:.2f}")

        print(f"\nAccuracy by Subject:")
        for subject, data in sorted(stats['by_subject'].items()):
            print(f"  {subject:30} {data['correct']:2}/{data['total']:2} ({data['accuracy']:5.1f}%)")

        print(f"\nAccuracy by Category:")
        for category, data in sorted(stats['by_category'].items()):
            print(f"  {category:30} {data['correct']:2}/{data['total']:2} ({data['accuracy']:5.1f}%)")

        print("\n" + "="*80)


def main():
    """Main entry point for benchmark test"""
    import argparse

    parser = argparse.ArgumentParser(description='Run tele_qna benchmark test')
    parser.add_argument('--questions',
                       default='tests/tele_qna_representative_set.json',
                       help='Path to questions JSON file')
    parser.add_argument('--model',
                       default='deepseek-r1:14b',
                       help='Model to use (deepseek-r1:14b, claude, deepseek-r1:7b, llama3.2, etc.)')
    parser.add_argument('--output',
                       default='tests/results/tele_qna_benchmark_results.json',
                       help='Output file for detailed results')

    args = parser.parse_args()

    # Get API keys from environment
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    deepseek_api_url = os.getenv('DEEPSEEK_API_URL', 'http://192.168.1.14:11434/api/chat')

    if not claude_api_key and args.model == 'claude':
        print("ERROR: CLAUDE_API_KEY environment variable not set")
        print("TIP: Use local models like 'deepseek-r1:14b' to avoid API requirements")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = create_rag_system_v2(
        claude_api_key=claude_api_key,
        deepseek_api_url=deepseek_api_url
    )

    try:
        # Initialize benchmark
        benchmark = TeleQNABenchmark(args.questions)

        # Run benchmark
        stats = benchmark.run_benchmark(rag_system, model=args.model)

        # Save results
        benchmark.save_results(args.output)

        # Print summary
        benchmark.print_summary(stats)

        return 0

    finally:
        # Clean up
        rag_system.close()
        logger.info("RAG system closed")


if __name__ == '__main__':
    sys.exit(main())
