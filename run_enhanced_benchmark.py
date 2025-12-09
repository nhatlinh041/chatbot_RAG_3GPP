#!/usr/bin/env python3
"""
Enhanced benchmark runner for RAG System V2 and V3 comparison
Tests performance, accuracy, and different query strategies
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import logging and RAG systems
from logging_config import setup_centralized_logging, get_logger, MAJOR, MINOR, ERROR
from rag_system_v2 import create_rag_system_v2
from rag_system_v3 import create_rag_system_v3


class EnhancedBenchmark:
    """Enhanced benchmark for comparing RAG systems"""
    
    def __init__(self):
        # Initialize logging
        setup_centralized_logging()
        self.logger = get_logger('Enhanced_Benchmark')
        
        # Load test questions
        self.questions = self._load_test_questions()
        
        # Configuration
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.local_llm_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')
        
        # Results storage
        self.results = {
            'v2_results': [],
            'v3_results': [],
            'comparison': {},
            'timestamp': datetime.now().isoformat(),
            'config': {
                'local_llm_url': self.local_llm_url,
                'has_claude_key': bool(self.claude_api_key),
                'total_questions': len(self.questions)
            }
        }
    
    def _load_test_questions(self) -> List[Dict]:
        """Load test questions from JSON file"""
        questions_file = PROJECT_ROOT / "tests" / "tele_qna_representative_set.json"
        
        if not questions_file.exists():
            self.logger.log(ERROR, f"Questions file not found: {questions_file}")
            return []
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        self.logger.log(MINOR, f"Loaded {len(questions)} test questions")
        return questions
    
    def run_v2_benchmark(self, model: str = "deepseek-r1:14b", max_questions: int = None) -> List[Dict]:
        """Run benchmark on RAG V2 system"""
        self.logger.log(MAJOR, f"Starting RAG V2 benchmark with model: {model}")
        
        try:
            # Create RAG V2 system
            rag_v2 = create_rag_system_v2(
                claude_api_key=self.claude_api_key,
                deepseek_api_url=self.local_llm_url
            )
            
            # Run questions
            results = self._run_questions(rag_v2, "V2", model, max_questions)
            self.results['v2_results'] = results
            
            rag_v2.close()
            return results
            
        except Exception as e:
            self.logger.log(ERROR, f"RAG V2 benchmark failed: {str(e)}")
            return []
    
    def run_v3_benchmark(self, model: str = "deepseek-r1:14b", max_questions: int = None) -> List[Dict]:
        """Run benchmark on RAG V3 system"""
        self.logger.log(MAJOR, f"Starting RAG V3 benchmark with model: {model}")
        
        try:
            # Create RAG V3 system
            rag_v3 = create_rag_system_v3(
                claude_api_key=self.claude_api_key,
                local_llm_url=self.local_llm_url
            )
            
            # Check V3 readiness
            status = rag_v3.check_vector_index_status()
            if not status['ready_for_hybrid']:
                self.logger.log(ERROR, "V3 system not ready for hybrid search")
                return []
            
            # Run questions with different strategies
            results = []
            
            # Test with hybrid search
            hybrid_results = self._run_questions(
                rag_v3, "V3_Hybrid", model, max_questions,
                extra_params={'use_hybrid': True, 'use_vector': True, 'use_graph': True}
            )
            results.extend(hybrid_results)
            
            # Test with vector only
            vector_results = self._run_questions(
                rag_v3, "V3_Vector", model, max_questions,
                extra_params={'use_hybrid': True, 'use_vector': True, 'use_graph': False}
            )
            results.extend(vector_results)
            
            # Test with graph only  
            graph_results = self._run_questions(
                rag_v3, "V3_Graph", model, max_questions,
                extra_params={'use_hybrid': True, 'use_vector': False, 'use_graph': True}
            )
            results.extend(graph_results)
            
            self.results['v3_results'] = results
            rag_v3.close()
            return results
            
        except Exception as e:
            self.logger.log(ERROR, f"RAG V3 benchmark failed: {str(e)}")
            return []
    
    def _run_questions(
        self, 
        rag_system, 
        system_name: str, 
        model: str, 
        max_questions: int = None,
        extra_params: Dict = None
    ) -> List[Dict]:
        """Run questions on a RAG system"""
        
        questions_to_run = self.questions[:max_questions] if max_questions else self.questions
        results = []
        
        self.logger.log(MINOR, f"Running {len(questions_to_run)} questions on {system_name}")
        
        for i, question_data in enumerate(questions_to_run, 1):
            question = question_data['question']
            expected_answer = question_data.get('expected_answer', '')
            
            self.logger.log(MINOR, f"[{i}/{len(questions_to_run)}] {system_name}: {question[:50]}...")
            
            try:
                start_time = time.time()
                
                # Query the system
                query_params = {'model': model}
                if extra_params:
                    query_params.update(extra_params)
                
                response = rag_system.query(question, **query_params)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Extract response data
                if hasattr(response, 'answer'):
                    # V3 response
                    answer = response.answer
                    strategy = getattr(response, 'retrieval_strategy', 'unknown')
                    source_count = len(getattr(response, 'sources', []))
                    vector_time = getattr(response, 'vector_search_time', 0)
                    graph_time = getattr(response, 'graph_search_time', 0)
                else:
                    # V2 response
                    answer = response if isinstance(response, str) else str(response)
                    strategy = 'graph_only'
                    source_count = 0  # V2 doesn't expose source count directly
                    vector_time = 0
                    graph_time = duration
                
                # Simple accuracy check (contains key terms from expected answer)
                accuracy = self._check_accuracy(answer, expected_answer)
                
                result = {
                    'system': system_name,
                    'question_index': i,
                    'question': question,
                    'answer': answer,
                    'expected_answer': expected_answer,
                    'accuracy': accuracy,
                    'duration': round(duration, 2),
                    'source_count': source_count,
                    'strategy': strategy,
                    'vector_time': round(vector_time, 2),
                    'graph_time': round(graph_time, 2),
                    'model': model,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Log result
                status = "✓ CORRECT" if accuracy > 0.5 else "✗ INCORRECT"
                self.logger.log(MINOR, f"{status} - Duration: {duration:.2f}s, Sources: {source_count}")
                
            except Exception as e:
                self.logger.log(ERROR, f"Question {i} failed: {str(e)}")
                results.append({
                    'system': system_name,
                    'question_index': i,
                    'question': question,
                    'error': str(e),
                    'accuracy': 0,
                    'duration': -1,
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _check_accuracy(self, answer: str, expected: str) -> float:
        """Simple accuracy check based on keyword overlap"""
        if not expected:
            return 0.5  # Unknown expected answer
        
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        answer_words -= common_words
        expected_words -= common_words
        
        if not expected_words:
            return 0.5
        
        intersection = answer_words & expected_words
        return len(intersection) / len(expected_words)
    
    def run_comparison(self, model: str = "deepseek-r1:14b", max_questions: int = 10):
        """Run complete comparison between V2 and V3"""
        self.logger.log(MAJOR, f"Starting enhanced benchmark comparison - {max_questions} questions")
        
        # Run V2 benchmark
        self.logger.log(MAJOR, "=" * 50)
        self.logger.log(MAJOR, "RAG V2 Benchmark")
        self.logger.log(MAJOR, "=" * 50)
        v2_results = self.run_v2_benchmark(model, max_questions)
        
        # Run V3 benchmark
        self.logger.log(MAJOR, "=" * 50)  
        self.logger.log(MAJOR, "RAG V3 Benchmark")
        self.logger.log(MAJOR, "=" * 50)
        v3_results = self.run_v3_benchmark(model, max_questions)
        
        # Generate comparison
        self._generate_comparison()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _generate_comparison(self):
        """Generate comparison statistics"""
        v2_results = [r for r in self.results['v2_results'] if 'error' not in r]
        v3_results = [r for r in self.results['v3_results'] if 'error' not in r]
        
        if not v2_results or not v3_results:
            self.logger.log(ERROR, "Insufficient results for comparison")
            return
        
        # Group V3 results by strategy
        v3_hybrid = [r for r in v3_results if r['system'] == 'V3_Hybrid']
        v3_vector = [r for r in v3_results if r['system'] == 'V3_Vector']
        v3_graph = [r for r in v3_results if r['system'] == 'V3_Graph']
        
        def calc_stats(results):
            if not results:
                return {}
            return {
                'avg_duration': sum(r['duration'] for r in results) / len(results),
                'avg_accuracy': sum(r['accuracy'] for r in results) / len(results),
                'avg_sources': sum(r['source_count'] for r in results) / len(results),
                'success_rate': len([r for r in results if 'error' not in r]) / len(results),
                'total_queries': len(results)
            }
        
        self.results['comparison'] = {
            'v2_stats': calc_stats(v2_results),
            'v3_hybrid_stats': calc_stats(v3_hybrid),
            'v3_vector_stats': calc_stats(v3_vector),
            'v3_graph_stats': calc_stats(v3_graph)
        }
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = PROJECT_ROOT / "test_output" / f"enhanced_benchmark_{timestamp}.json"
        
        # Ensure directory exists
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.logger.log(MAJOR, f"Results saved to: {results_file}")
    
    def _print_summary(self):
        """Print benchmark summary"""
        comparison = self.results.get('comparison', {})
        
        if not comparison:
            self.logger.log(ERROR, "No comparison data available")
            return
        
        self.logger.log(MAJOR, "=" * 60)
        self.logger.log(MAJOR, "ENHANCED BENCHMARK SUMMARY")
        self.logger.log(MAJOR, "=" * 60)
        
        # Print stats for each system
        systems = [
            ('RAG V2', 'v2_stats'),
            ('RAG V3 (Hybrid)', 'v3_hybrid_stats'),
            ('RAG V3 (Vector Only)', 'v3_vector_stats'),
            ('RAG V3 (Graph Only)', 'v3_graph_stats')
        ]
        
        for system_name, stats_key in systems:
            stats = comparison.get(stats_key, {})
            if not stats:
                continue
                
            self.logger.log(MAJOR, f"\n{system_name}:")
            self.logger.log(MINOR, f"  Average Duration: {stats.get('avg_duration', 0):.2f}s")
            self.logger.log(MINOR, f"  Average Accuracy: {stats.get('avg_accuracy', 0):.2f}")
            self.logger.log(MINOR, f"  Average Sources: {stats.get('avg_sources', 0):.1f}")
            self.logger.log(MINOR, f"  Success Rate: {stats.get('success_rate', 0):.1%}")
            self.logger.log(MINOR, f"  Total Queries: {stats.get('total_queries', 0)}")
        
        # Best performing system
        best_accuracy_system = max(systems, key=lambda x: comparison.get(x[1], {}).get('avg_accuracy', 0))
        best_speed_system = min(systems, key=lambda x: comparison.get(x[1], {}).get('avg_duration', float('inf')))
        
        self.logger.log(MAJOR, f"\nBest Accuracy: {best_accuracy_system[0]}")
        self.logger.log(MAJOR, f"Best Speed: {best_speed_system[0]}")


def main():
    """Main benchmark runner"""
    parser = argparse.ArgumentParser(description='Enhanced RAG System Benchmark')
    parser.add_argument('--model', default='deepseek-r1:14b', help='LLM model to use')
    parser.add_argument('--questions', type=int, default=10, help='Number of questions to test')
    parser.add_argument('--system', choices=['v2', 'v3', 'both'], default='both', help='Which system(s) to test')
    
    args = parser.parse_args()
    
    benchmark = EnhancedBenchmark()
    
    if args.system == 'v2':
        benchmark.run_v2_benchmark(args.model, args.questions)
    elif args.system == 'v3':
        benchmark.run_v3_benchmark(args.model, args.questions)
    else:
        benchmark.run_comparison(args.model, args.questions)


if __name__ == "__main__":
    main()