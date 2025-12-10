"""
Tele QnA Benchmark Runner for 3GPP RAG System V3.

Runs comprehensive benchmark tests across multiple question categories
and generates detailed performance reports.

Usage:
    python run_tele_qna_benchmark.py [--model MODEL] [--category CATEGORY] [--output OUTPUT]

Examples:
    # Run full benchmark with default model
    python run_tele_qna_benchmark.py

    # Run with specific model
    python run_tele_qna_benchmark.py --model deepseek-r1:14b

    # Run specific category only
    python run_tele_qna_benchmark.py --category Definition

    # Save results to custom file
    python run_tele_qna_benchmark.py --output my_results.json
"""

import json
import argparse
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag_system_v3 import create_rag_system_v3
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG

# Initialize logging
setup_centralized_logging()
logger = get_logger('Tele_QnA_Benchmark')


class TeeLogger:
    """Write output to both console and file"""
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark question"""
    category: str
    question: str
    expected_answer_index: int
    expected_answer_text: str
    model_response: str
    extracted_choice: Optional[int]
    correct: bool
    retrieval_strategy: str
    execution_time_ms: float
    error: Optional[str] = None


@dataclass
class CategoryStats:
    """Statistics for a question category"""
    category: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_execution_time_ms: float
    errors: int


class TeleQnABenchmark:
    """Benchmark runner for Tele QnA questions"""

    # Default paths relative to this script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "tele_qna_benchmark_comprehensive.json")
    DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "benchmark_results_latest.json")

    def __init__(self, benchmark_file: str = None,
                 model: str = "deepseek-r1:14b",
                 output_file: str = None):
        """
        Initialize benchmark runner.

        Args:
            benchmark_file: Path to JSON file with benchmark questions
            model: LLM model to use for answering
            output_file: Output JSON file path
        """
        self.benchmark_file = benchmark_file or self.DEFAULT_BENCHMARK_FILE
        self.model = model
        self.output_file = output_file or self.DEFAULT_OUTPUT_FILE
        self.rag_system = None
        self.results: List[BenchmarkResult] = []

        # Setup output log file (same name as JSON but .log extension)
        self.log_file = self.output_file.replace('.json', '.log')
        self.tee_logger = None

        logger.log(MAJOR, f"Initializing Tele QnA Benchmark - Model: {model}")
        logger.log(MAJOR, f"Output JSON: {self.output_file}")
        logger.log(MAJOR, f"Output Log: {self.log_file}")

    def setup_rag_system(self):
        """Initialize RAG system V3"""
        logger.log(MAJOR, "Setting up RAG System V3")

        # Get API keys from environment
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        local_llm_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')

        # Create RAG system
        self.rag_system = create_rag_system_v3(
            claude_api_key=claude_api_key,
            local_llm_url=local_llm_url
        )

        # Check vector index status
        status = self.rag_system.check_vector_index_status()
        logger.log(MAJOR, f"Vector index status: {status}")

        if not status.get('ready_for_hybrid', False):
            logger.log(ERROR, "Vector index not ready! Run orchestrator.py setup-v3 first")
            raise RuntimeError("Vector index not ready for hybrid retrieval")

        logger.log(MAJOR, "RAG System V3 ready")

    def load_benchmark_data(self) -> List[Dict]:
        """Load benchmark questions from JSON file"""
        logger.log(MAJOR, f"Loading benchmark data from {self.benchmark_file}")

        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_questions = sum(len(cat['questions']) for cat in data)
        logger.log(MAJOR, f"Loaded {len(data)} categories with {total_questions} total questions")

        return data

    def extract_choice_from_response(self, response: str, choices: List[str]) -> Optional[int]:
        """
        Extract the chosen answer index from model response using robust regex patterns.

        Priority order:
        1. "Answer: C" at start of response (MCQ prompt format)
        2. "answer is C" / "answer: C" anywhere
        3. Letter at start of line ("C. explanation")
        4. "option C" / "choice C" (avoiding false positives)
        5. Fallback: Match choice text in response

        Args:
            response: Model's answer text
            choices: List of choice texts

        Returns:
            Index of chosen answer (0-based) or None if not found
        """
        logger.log(DEBUG, f"Extracting choice from response: {response[:100]}...")

        # Method 0: HIGHEST PRIORITY - Check for "Answer: X. [text]" at start of response
        # This is the format we explicitly request in MCQ prompt: "Answer: B. Authentication services"
        start_match = re.match(r'^Answer:\s*([A-Da-d])[\.\s]', response.strip(), re.IGNORECASE)
        if start_match:
            letter = start_match.group(1).upper()
            index = ord(letter) - ord('A')
            if 0 <= index < len(choices):
                logger.log(MINOR, f"Extracted choice {index} ({letter}) via start-of-response pattern")
                return index

        # Method 1: Priority patterns - check "answer" patterns first (most reliable)
        for i in range(len(choices)):
            choice_letter = chr(65 + i)  # A=65, B=66, C=67, D=68

            # Pattern 1: "answer is C" / "answer: C" variations
            # Matches: "answer: C", "answer is C.", "The answer: C", etc.
            pattern1 = rf'\b(?:the\s+)?answer(?:\s+is)?[\s:]+{choice_letter}(?:\.|,|\s|$)'
            if re.search(pattern1, response, re.IGNORECASE):
                logger.log(MINOR, f"Extracted choice {i} ({choice_letter}) via answer pattern")
                return i

        # Method 2: Letter at start of line (second priority)
        for i in range(len(choices)):
            choice_letter = chr(65 + i)

            # Matches: "C. 14 symbols" or "C: Based on..."
            pattern_start = rf'^{choice_letter}[\.:]\s+'
            if re.search(pattern_start, response, re.IGNORECASE | re.MULTILINE):
                logger.log(MINOR, f"Extracted choice {i} ({choice_letter}) via start-of-line pattern")
                return i

        # Method 3: "option C" / "choice C" (lower priority - can be in explanations)
        for i in range(len(choices)):
            choice_letter = chr(65 + i)

            # Only match if it's NOT preceded by "Note:" or "However," (avoid false positives)
            pattern_option = rf'(?<!Note:\s)(?<!However,\s)\b(?:option|choice)[\s:]+{choice_letter}(?:\.|,|\s|$)'
            if re.search(pattern_option, response, re.IGNORECASE):
                logger.log(MINOR, f"Extracted choice {i} ({choice_letter}) via option/choice pattern")
                return i

        # Method 2: Look for exact choice text
        # Find first occurrence of any choice text
        first_match_index = len(response)
        matched_choice = None

        for i, choice in enumerate(choices):
            pos = response.lower().find(choice.lower())
            if pos != -1 and pos < first_match_index:
                first_match_index = pos
                matched_choice = i

        if matched_choice is not None:
            logger.log(MINOR, f"Extracted choice {matched_choice} via text matching")
            return matched_choice

        # No match found
        logger.log(ERROR, f"Failed to extract choice from response: {response[:200]}")
        return None

    def run_question(self, question_data: Dict, category: str) -> BenchmarkResult:
        """
        Run a single benchmark question.

        Args:
            question_data: Question dict with 'question', 'choices', 'answer', etc.
            category: Category name

        Returns:
            BenchmarkResult with test outcome
        """
        question_text = question_data['question']
        choices = question_data['choices']
        expected_answer_index = question_data['answer']
        expected_answer_text = choices[expected_answer_index]

        logger.log(MINOR, f"Running question: {question_text[:60]}...")

        # Format question with choices
        formatted_question = f"{question_text}\n\n"
        for i, choice in enumerate(choices):
            choice_letter = chr(65 + i)  # A, B, C, D...
            formatted_question += f"{choice_letter}. {choice}\n"

        try:
            # Query RAG system
            start_time = datetime.now()

            response = self.rag_system.query(
                formatted_question,
                model=self.model,
                use_hybrid=True,
                use_vector=True,
                use_graph=True
            )

            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # Extract chosen answer
            extracted_choice = self.extract_choice_from_response(response.answer, choices)

            # Check correctness
            correct = (extracted_choice == expected_answer_index) if extracted_choice is not None else False

            logger.log(
                MAJOR if correct else ERROR,
                f"Question result: {'CORRECT' if correct else 'INCORRECT'} "
                f"(Expected: {expected_answer_index}, Got: {extracted_choice})"
            )

            return BenchmarkResult(
                category=category,
                question=question_text,
                expected_answer_index=expected_answer_index,
                expected_answer_text=expected_answer_text,
                model_response=response.answer,
                extracted_choice=extracted_choice,
                correct=correct,
                retrieval_strategy=response.retrieval_strategy,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            logger.log(ERROR, f"Error running question: {e}")

            return BenchmarkResult(
                category=category,
                question=question_text,
                expected_answer_index=expected_answer_index,
                expected_answer_text=expected_answer_text,
                model_response="",
                extracted_choice=None,
                correct=False,
                retrieval_strategy="error",
                execution_time_ms=0,
                error=str(e)
            )

    def save_incremental_results(self):
        """Save current results to JSON (incremental save after each question)"""
        try:
            report = self.generate_report()
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.log(ERROR, f"Failed to save incremental results: {e}")

    def run_category(self, category_data: Dict) -> List[BenchmarkResult]:
        """
        Run all questions in a category.

        Args:
            category_data: Category dict with 'category', 'description', 'questions'

        Returns:
            List of BenchmarkResult
        """
        category_name = category_data['category']
        questions = category_data['questions']

        logger.log(MAJOR, f"\n{'='*60}")
        logger.log(MAJOR, f"Running category: {category_name}")
        logger.log(MAJOR, f"Questions: {len(questions)}")
        logger.log(MAJOR, f"{'='*60}\n")

        results = []
        for i, question_data in enumerate(questions, 1):
            logger.log(MINOR, f"Question {i}/{len(questions)}")
            result = self.run_question(question_data, category_name)
            results.append(result)

            # Incremental save after each question
            self.results.append(result)
            self.save_incremental_results()

        return results

    def calculate_category_stats(self, category: str) -> CategoryStats:
        """Calculate statistics for a category"""
        category_results = [r for r in self.results if r.category == category]

        total = len(category_results)
        correct = sum(1 for r in category_results if r.correct)
        errors = sum(1 for r in category_results if r.error is not None)
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_time = sum(r.execution_time_ms for r in category_results) / total if total > 0 else 0

        return CategoryStats(
            category=category,
            total_questions=total,
            correct_answers=correct,
            accuracy=accuracy,
            avg_execution_time_ms=avg_time,
            errors=errors
        )

    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        logger.log(MAJOR, "Generating benchmark report")

        # Overall stats
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.correct)
        overall_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        # Category stats
        categories = list(set(r.category for r in self.results))
        category_stats = {cat: self.calculate_category_stats(cat) for cat in categories}

        # Prepare report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'benchmark_file': self.benchmark_file,
                'total_questions': total_questions
            },
            'overall': {
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'accuracy': round(overall_accuracy, 2),
                'total_errors': sum(1 for r in self.results if r.error is not None)
            },
            'categories': {
                cat: {
                    'total_questions': stats.total_questions,
                    'correct_answers': stats.correct_answers,
                    'accuracy': round(stats.accuracy, 2),
                    'avg_execution_time_ms': round(stats.avg_execution_time_ms, 2),
                    'errors': stats.errors
                }
                for cat, stats in category_stats.items()
            },
            'detailed_results': [
                {
                    'category': r.category,
                    'question': r.question,
                    'expected_answer': r.expected_answer_text,
                    'model_response': r.model_response[:500],  # Truncate long responses
                    'correct': r.correct,
                    'retrieval_strategy': r.retrieval_strategy,
                    'execution_time_ms': round(r.execution_time_ms, 2),
                    'error': r.error
                }
                for r in self.results
            ]
        }

        return report

    def print_summary(self, report: Dict):
        """Print benchmark summary to console"""
        print("\n" + "="*80)
        print("TELE QnA BENCHMARK RESULTS")
        print("="*80)
        print(f"Model: {report['metadata']['model']}")
        print(f"Total Questions: {report['overall']['total_questions']}")
        print(f"Correct Answers: {report['overall']['correct_answers']}")
        print(f"Overall Accuracy: {report['overall']['accuracy']}%")
        print(f"Total Errors: {report['overall']['total_errors']}")
        print("="*80)

        print("\nCATEGORY BREAKDOWN:")
        print("-"*80)
        print(f"{'Category':<25} {'Questions':<12} {'Correct':<10} {'Accuracy':<12} {'Avg Time (ms)'}")
        print("-"*80)

        for category, stats in sorted(report['categories'].items()):
            print(f"{category:<25} {stats['total_questions']:<12} {stats['correct_answers']:<10} "
                  f"{stats['accuracy']:>6.2f}%     {stats['avg_execution_time_ms']:>8.2f}")

        print("="*80 + "\n")

    def analyze_results(self, report: Dict) -> str:
        """
        Analyze benchmark results and generate insights.

        Returns:
            Analysis markdown text
        """
        logger.log(MAJOR, "Analyzing benchmark results")

        # Extract key metrics
        overall = report['overall']
        categories = report['categories']

        # Find best and worst categories
        sorted_cats = sorted(categories.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_category = sorted_cats[0] if sorted_cats else None
        worst_category = sorted_cats[-1] if sorted_cats else None

        # Find error patterns
        error_results = [r for r in self.results if r.error is not None]
        extraction_failures = [r for r in self.results if r.extracted_choice is None and r.error is None]

        # Performance analysis
        avg_time_all = sum(r.execution_time_ms for r in self.results) / len(self.results) if self.results else 0
        slow_questions = [r for r in self.results if r.execution_time_ms > avg_time_all * 2]

        # Generate analysis markdown
        analysis = f"""# Benchmark Analysis Report

## Overview
- **Model**: {report['metadata']['model']}
- **Timestamp**: {report['metadata']['timestamp']}
- **Total Questions**: {overall['total_questions']}
- **Overall Accuracy**: {overall['accuracy']}%

## Performance Summary

### Accuracy Distribution
"""

        for cat_name, cat_stats in sorted_cats:
            bar = '█' * int(cat_stats['accuracy'] / 5)
            analysis += f"- **{cat_name}**: {cat_stats['accuracy']:.1f}% {bar}\n"

        analysis += f"""
### Top Performing Category
- **Category**: {best_category[0] if best_category else 'N/A'}
- **Accuracy**: {best_category[1]['accuracy']:.1f}% if best_category else 'N/A'
- **Avg Time**: {best_category[1]['avg_execution_time_ms']:.1f}ms if best_category else 'N/A'

### Lowest Performing Category
- **Category**: {worst_category[0] if worst_category else 'N/A'}
- **Accuracy**: {worst_category[1]['accuracy']:.1f}% if worst_category else 'N/A'
- **Avg Time**: {worst_category[1]['avg_execution_time_ms']:.1f}ms if worst_category else 'N/A'

## Error Analysis

### Errors by Type
- **System Errors**: {len(error_results)} ({len(error_results)/len(self.results)*100:.1f}%)
- **Answer Extraction Failures**: {len(extraction_failures)} ({len(extraction_failures)/len(self.results)*100:.1f}%)

### Performance Issues
- **Average Execution Time**: {avg_time_all:.1f}ms
- **Slow Questions (>2x avg)**: {len(slow_questions)} ({len(slow_questions)/len(self.results)*100:.1f}%)

"""

        # Recommendations
        analysis += """## Recommendations

"""

        if overall['accuracy'] < 70:
            analysis += "⚠️ **Critical**: Overall accuracy below 70%. Consider:\n"
            analysis += "  - Reviewing prompt templates\n"
            analysis += "  - Improving retrieval strategy\n"
            analysis += "  - Fine-tuning model parameters\n\n"

        if len(extraction_failures) > len(self.results) * 0.1:
            analysis += "⚠️ **Warning**: High answer extraction failure rate (>10%). Consider:\n"
            analysis += "  - Updating regex patterns in `extract_choice_from_response()`\n"
            analysis += "  - Adding prompt instruction for clear answer format\n\n"

        if len(slow_questions) > len(self.results) * 0.2:
            analysis += "⚠️ **Performance**: Many slow queries (>20%). Consider:\n"
            analysis += "  - Optimizing vector search parameters\n"
            analysis += "  - Reducing context size\n"
            analysis += "  - Using faster LLM model\n\n"

        if worst_category and worst_category[1]['accuracy'] < 50:
            analysis += f"⚠️ **Attention**: Category '{worst_category[0]}' has very low accuracy. Consider:\n"
            analysis += "  - Reviewing category-specific prompt template\n"
            analysis += "  - Adding more training examples for this category\n\n"

        analysis += """
## Next Steps

1. Review detailed results in JSON file
2. Investigate failed questions manually
3. Update prompt templates for low-performing categories
4. Re-run benchmark after improvements

---
*Generated by Tele QnA Benchmark Runner*
"""

        return analysis

    def save_report(self, report: Dict, output_file: str):
        """Save report to JSON file"""
        logger.log(MAJOR, f"Saving report to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.log(MAJOR, f"Report saved successfully")

    def run(self, category_filter: Optional[str] = None):
        """
        Run full benchmark.

        Args:
            category_filter: If specified, only run this category
        """
        # Setup output logging to file
        self.tee_logger = TeeLogger(self.log_file)
        original_stdout = sys.stdout
        sys.stdout = self.tee_logger

        try:
            logger.log(CRITICAL, "Starting Tele QnA Benchmark")
            print(f"\n{'='*80}")
            print(f"Tele QnA Benchmark Runner")
            print(f"{'='*80}")
            print(f"Model: {self.model}")
            print(f"Output JSON: {self.output_file}")
            print(f"Output Log: {self.log_file}")
            print(f"{'='*80}\n")

            # Setup
            self.setup_rag_system()
            benchmark_data = self.load_benchmark_data()

            # Filter categories if needed
            if category_filter:
                benchmark_data = [cat for cat in benchmark_data if cat['category'] == category_filter]
                if not benchmark_data:
                    logger.log(ERROR, f"Category '{category_filter}' not found")
                    return

            # Run benchmark
            for category_data in benchmark_data:
                self.run_category(category_data)
                # Results already added incrementally in run_category

            # Generate final report
            report = self.generate_report()
            self.print_summary(report)
            self.save_report(report, self.output_file)

            # Auto-analyze results
            print("\n" + "="*80)
            print("GENERATING ANALYSIS...")
            print("="*80 + "\n")
            analysis = self.analyze_results(report)

            # Save analysis to markdown file
            analysis_file = self.output_file.replace('.json', '_analysis.md')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(analysis)

            print(analysis)
            print(f"\n✅ Analysis saved to: {analysis_file}")

            logger.log(CRITICAL, "Benchmark completed successfully")

        finally:
            # Restore stdout
            sys.stdout = original_stdout
            if self.tee_logger:
                self.tee_logger.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Tele QnA Benchmark for 3GPP RAG System")
    parser.add_argument('--model', default='deepseek-r1:14b', help='LLM model to use')
    parser.add_argument('--category', default=None, help='Specific category to test')
    parser.add_argument('--output', default=None, help='Output JSON file (default: tests/benchmark/benchmark_results_latest.json)')
    parser.add_argument('--benchmark-file', default=None,
                        help='Benchmark questions JSON file (default: tests/benchmark/tele_qna_benchmark_comprehensive.json)')

    args = parser.parse_args()

    # Run benchmark
    benchmark = TeleQnABenchmark(
        benchmark_file=args.benchmark_file,
        model=args.model,
        output_file=args.output
    )

    benchmark.run(
        category_filter=args.category
    )


if __name__ == '__main__':
    main()
