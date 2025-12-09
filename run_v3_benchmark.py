#!/usr/bin/env python3
"""
Quick benchmark for RAG System V3 with new processed documents
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system_v3 import create_rag_system_v3
from logging_config import setup_centralized_logging, get_logger, MAJOR, ERROR

# Initialize logging
setup_centralized_logging()
logger = get_logger('V3_Benchmark')


def load_sample_questions():
    """Load sample questions for quick test"""
    return [
        {
            "question": "What is AMF?",
            "expected_answer": "Access and Mobility Management Function"
        },
        {
            "question": "Compare AMF and SMF",
            "expected_answer": "AMF handles access and mobility, SMF handles sessions"
        },
        {
            "question": "What is the role of UPF?",
            "expected_answer": "User Plane Function for packet routing"
        },
        {
            "question": "Describe the 5G registration procedure",
            "expected_answer": "Initial registration with AMF via gNB"
        },
        {
            "question": "What specs define 5G architecture?",
            "expected_answer": "TS 23.501 defines system architecture"
        },
        {
            "question": "How do AMF and SMF interact?",
            "expected_answer": "Via N11 interface for session management"
        },
        {
            "question": "What is SEPP?",
            "expected_answer": "Security Edge Protection Proxy"
        },
        {
            "question": "Define network function",
            "expected_answer": "Processing function in 5G architecture"
        },
        {
            "question": "What is SCP?",
            "expected_answer": "Service Communication Proxy"
        },
        {
            "question": "What does TS 23.502 cover?",
            "expected_answer": "Procedures for 5G System"
        }
    ]


def run_benchmark_test(rag_system, questions, model="deepseek-r1:14b"):
    """Run benchmark test"""
    results = []
    total_retrieval_time = 0
    total_generation_time = 0
    
    print(f"\n🚀 Running benchmark with {len(questions)} questions...")
    
    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[{i}/{len(questions)}] {question}")
        
        start_time = time.time()
        
        try:
            # Query with V3 hybrid system
            response = rag_system.query(
                question=question,
                model=model,
                use_hybrid=True,
                use_vector=True,
                use_graph=True,
                use_query_expansion=True,
                use_llm_analysis=False,  # Faster
                top_k=6
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            # Extract timing if available
            retrieval_time = getattr(response, 'retrieval_time_ms', 0)
            generation_time = getattr(response, 'generation_time_ms', 0)
            
            total_retrieval_time += retrieval_time
            total_generation_time += generation_time
            
            result = {
                "question": question,
                "answer": response.answer,
                "retrieval_strategy": getattr(response, 'retrieval_strategy', 'unknown'),
                "sources_count": len(response.sources) if hasattr(response, 'sources') else 0,
                "total_time_ms": round(total_time, 1),
                "retrieval_time_ms": round(retrieval_time, 1),
                "generation_time_ms": round(generation_time, 1),
                "model_used": getattr(response, 'model_used', model)
            }
            
            # Show quick preview
            print(f"   ✅ {result['retrieval_strategy']} | {result['sources_count']} sources | {result['total_time_ms']:.0f}ms")
            print(f"   📄 {response.answer[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result = {
                "question": question,
                "answer": f"Error: {e}",
                "error": str(e),
                "total_time_ms": 0
            }
        
        results.append(result)
    
    # Calculate stats
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    stats = {
        "total_questions": len(questions),
        "successful": len(successful),
        "failed": len(failed),
        "avg_total_time_ms": round(sum(r["total_time_ms"] for r in successful) / len(successful), 1) if successful else 0,
        "avg_retrieval_time_ms": round(total_retrieval_time / len(successful), 1) if successful else 0,
        "avg_generation_time_ms": round(total_generation_time / len(successful), 1) if successful else 0,
        "model": model
    }
    
    return results, stats


def print_summary(results, stats):
    """Print benchmark summary"""
    print("\n" + "="*60)
    print("📊 V3 BENCHMARK RESULTS")
    print("="*60)
    
    print(f"Total Questions: {stats['total_questions']}")
    print(f"✅ Successful: {stats['successful']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"🎯 Success Rate: {stats['successful']/stats['total_questions']*100:.1f}%")
    
    if stats['successful'] > 0:
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Average Total Time: {stats['avg_total_time_ms']:.0f}ms")
        print(f"   Average Retrieval: {stats['avg_retrieval_time_ms']:.0f}ms")
        print(f"   Average Generation: {stats['avg_generation_time_ms']:.0f}ms")
        
        # Retrieval strategies
        strategies = {}
        for r in results:
            if "error" not in r:
                strategy = r.get('retrieval_strategy', 'unknown')
                strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print(f"\n🔍 Retrieval Strategies:")
        for strategy, count in strategies.items():
            print(f"   {strategy}: {count} queries")
    
    print("="*60)


def save_results(results, stats, output_file):
    """Save benchmark results"""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "rag_version": "v3",
        "benchmark_type": "quick_test",
        "stats": stats,
        "results": results
    }
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="RAG System V3 Benchmark")
    parser.add_argument("--model", default="deepseek-r1:14b", help="LLM model to use")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to test")
    parser.add_argument("--output", default="tests/results/v3_benchmark.json", help="Output file")
    parser.add_argument("--setup-v3", action="store_true", help="Setup vector search first")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Check environment
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    local_llm_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')
    
    print("🔧 Initializing RAG System V3...")
    print(f"Model: {args.model}")
    print(f"Local LLM: {local_llm_url}")
    
    try:
        # Initialize RAG V3
        rag = create_rag_system_v3(
            claude_api_key=claude_api_key,
            local_llm_url=local_llm_url
        )
        
        # Check vector search status
        print("📊 Checking V3 system status...")
        status = rag.check_vector_index_status()
        print(f"   Vector Index Ready: {status.get('ready_for_hybrid', False)}")
        print(f"   Total Chunks: {status.get('total_chunks', 0)}")
        print(f"   Embeddings: {status.get('chunks_with_embeddings', 0)}")
        
        # Setup if requested
        if args.setup_v3 and not status.get('ready_for_hybrid', False):
            print("🔄 Setting up vector search...")
            rag.setup_vector_search(batch_size=50)
            print("✅ Vector search setup complete!")
        
        # Load questions
        questions = load_sample_questions()[:args.limit]
        
        # Run benchmark
        results, stats = run_benchmark_test(rag, questions, args.model)
        
        # Print results
        print_summary(results, stats)
        
        # Save results
        save_results(results, stats, args.output)
        
        return 0
        
    except Exception as e:
        logger.log(ERROR, f"Benchmark failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())