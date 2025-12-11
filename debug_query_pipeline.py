#!/usr/bin/env python3
"""
Debug Script for RAG Query Pipeline
Traces the entire query processing flow with detailed logging and reasoning.

Usage:
    python debug_query_pipeline.py "What is AMF?"
    python debug_query_pipeline.py "Compare SCP and SEPP"
    python debug_query_pipeline.py --file questions.txt
    python debug_query_pipeline.py --interactive
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.END}")


def print_step(step_num: int, text: str):
    print(f"{Colors.GREEN}[Step {step_num}]{Colors.END} {text}")


def print_info(label: str, value: Any):
    print(f"  {Colors.YELLOW}{label}:{Colors.END} {value}")


def print_reasoning(text: str):
    print(f"  {Colors.BLUE}💭 Reasoning:{Colors.END} {text}")


def print_warning(text: str):
    print(f"  {Colors.RED}⚠️  {text}{Colors.END}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✅ {text}{Colors.END}")


class QueryPipelineDebugger:
    """Debug the entire RAG query pipeline with detailed tracing"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.trace_log = []
        self.timings = {}

        # Initialize components
        print_section("Initializing Components")

        # Neo4j connection
        self.neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print_success(f"Neo4j connected: {self.neo4j_uri}")
        except Exception as e:
            print_warning(f"Neo4j connection failed: {e}")
            self.driver = None

        # Import RAG components
        try:
            from rag_system_v3 import extract_mcq_question_only
            from hybrid_retriever import (
                HybridRetriever, SemanticQueryAnalyzer,
                QueryExpander, TermDefinitionResolver, ScoredChunk
            )
            from prompt_templates import PromptTemplates, ContextBuilder
            from subject_classifier import SubjectClassifier
            from rag_core import CypherQueryGenerator

            self.extract_mcq = extract_mcq_question_only
            self.ScoredChunk = ScoredChunk
            self.PromptTemplates = PromptTemplates
            self.ContextBuilder = ContextBuilder

            # Initialize components
            local_llm_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')

            if self.driver:
                self.cypher_generator = CypherQueryGenerator(
                    neo4j_driver=self.driver,
                    local_llm_url=local_llm_url
                )

                # Get term dict from cypher generator
                term_dict = getattr(self.cypher_generator, 'all_terms', {})

                self.query_analyzer = SemanticQueryAnalyzer(
                    local_llm_url=local_llm_url,
                    term_dict=term_dict
                )
                self.query_expander = QueryExpander(term_dict=term_dict)
                self.term_resolver = TermDefinitionResolver(self.driver)
                self.subject_classifier = SubjectClassifier()

                self.hybrid_retriever = HybridRetriever(
                    neo4j_driver=self.driver,
                    cypher_generator=self.cypher_generator,
                    local_llm_url=local_llm_url
                )

            print_success("RAG components loaded")

        except ImportError as e:
            print_warning(f"Failed to import RAG components: {e}")
            raise

    def log_trace(self, stage: str, data: Dict):
        """Add entry to trace log"""
        self.trace_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'data': data
        })

    def time_stage(self, stage_name: str):
        """Context manager for timing stages"""
        class Timer:
            def __init__(timer_self, debugger, name):
                timer_self.debugger = debugger
                timer_self.name = name
                timer_self.start = None

            def __enter__(timer_self):
                timer_self.start = time.time()
                return timer_self

            def __exit__(timer_self, *args):
                elapsed = (time.time() - timer_self.start) * 1000
                timer_self.debugger.timings[timer_self.name] = elapsed

        return Timer(self, stage_name)

    def debug_query(self, query: str, model: str = "deepseek-r1:14b",
                    skip_llm: bool = False) -> Dict:
        """
        Debug entire query pipeline with detailed tracing.

        Args:
            query: User question
            model: LLM model to use
            skip_llm: Skip LLM generation (faster, for testing retrieval only)

        Returns:
            Dict with all debug information
        """
        print_header(f"DEBUG: Query Pipeline")
        print_info("Query", query)
        print_info("Model", model)
        print_info("Skip LLM", skip_llm)

        results = {
            'query': query,
            'model': model,
            'stages': {},
            'timings': {},
            'final_answer': None
        }

        # =========================================================
        # STAGE 1: MCQ Detection
        # =========================================================
        print_section("Stage 1: MCQ Detection")

        with self.time_stage('mcq_detection'):
            retrieval_query, is_mcq = self.extract_mcq(query)

        print_step(1, "extract_mcq_question_only()")
        print_info("Is MCQ", is_mcq)
        print_info("Retrieval Query", retrieval_query[:100] + "..." if len(retrieval_query) > 100 else retrieval_query)

        if is_mcq:
            print_reasoning("MCQ detected - using question-only for retrieval to avoid noise from choices")
        else:
            print_reasoning("Not MCQ format - using full query for retrieval")

        results['stages']['mcq_detection'] = {
            'is_mcq': is_mcq,
            'retrieval_query': retrieval_query,
            'original_query': query
        }

        self.log_trace('mcq_detection', results['stages']['mcq_detection'])

        # =========================================================
        # STAGE 2: Query Analysis
        # =========================================================
        print_section("Stage 2: Query Analysis (SemanticQueryAnalyzer)")

        with self.time_stage('query_analysis'):
            # Use fallback (rule-based) for faster debugging
            analysis = self.query_analyzer._fallback_analysis(retrieval_query)

        print_step(2, "SemanticQueryAnalyzer._fallback_analysis()")
        print_info("Primary Intent", analysis.get('primary_intent'))
        print_info("Entities", analysis.get('entities', []))
        print_info("Key Terms", analysis.get('key_terms', [])[:10])
        print_info("Complexity", analysis.get('complexity'))
        print_info("Requires Multi-step", analysis.get('requires_multi_step'))
        print_info("Needs Term Resolution", analysis.get('needs_term_resolution'))

        intent = analysis.get('primary_intent')
        print_reasoning(f"Intent '{intent}' detected based on query patterns")

        if analysis.get('needs_term_resolution'):
            print_reasoning("Term resolution needed for definition/comparison queries to get authoritative definitions")

        # Override MCQ intent if detected
        if is_mcq and intent != 'multiple_choice':
            analysis['primary_intent'] = 'multiple_choice'
            print_warning(f"Overriding intent to 'multiple_choice' (MCQ format detected)")

        results['stages']['query_analysis'] = analysis
        self.log_trace('query_analysis', analysis)

        # =========================================================
        # STAGE 3: Term Definition Resolution (Conditional)
        # =========================================================
        print_section("Stage 3: Term Definition Resolution (Conditional)")

        term_definitions = {}

        if analysis.get('needs_term_resolution'):
            print_step(3, "TermDefinitionResolver.resolve_terms()")

            with self.time_stage('term_resolution'):
                entities = analysis.get('entities', [])
                term_definitions = self.term_resolver.resolve_terms(entities)

            if term_definitions:
                print_success(f"Resolved {len(term_definitions)} terms:")
                for abbr, info in term_definitions.items():
                    print_info(f"  {abbr}", f"{info.get('full_name')} (from {info.get('specs', [])[:2]})")
                print_reasoning("Using authoritative definitions from Term nodes prevents hallucination")
            else:
                print_warning("No terms found in Term nodes")
                print_reasoning("Will rely on context from retrieved chunks for definitions")
        else:
            print_info("Skipped", "Term resolution not needed for this query type")
            print_reasoning(f"Intent '{intent}' doesn't require term lookup (e.g., procedure queries)")

        analysis['term_definitions'] = term_definitions
        results['stages']['term_resolution'] = {
            'resolved': bool(term_definitions),
            'terms': term_definitions
        }
        self.log_trace('term_resolution', results['stages']['term_resolution'])

        # =========================================================
        # STAGE 4: Query Expansion
        # =========================================================
        print_section("Stage 4: Query Expansion")

        with self.time_stage('query_expansion'):
            query_variations = self.query_expander.expand(retrieval_query)

        print_step(4, "QueryExpander.expand()")
        print_info("Variations Count", len(query_variations))
        for i, var in enumerate(query_variations):
            print_info(f"  Variation {i+1}", var[:80] + "..." if len(var) > 80 else var)

        print_reasoning("Diverse variations improve recall - different phrasings find different chunks")

        results['stages']['query_expansion'] = {
            'variations': query_variations
        }
        self.log_trace('query_expansion', results['stages']['query_expansion'])

        # =========================================================
        # STAGE 5: Subject Detection
        # =========================================================
        print_section("Stage 5: Subject Detection")

        with self.time_stage('subject_detection'):
            expected_subjects = self.subject_classifier.detect_query_subjects(retrieval_query)

        print_step(5, "SubjectClassifier.detect_query_subjects()")
        print_info("Expected Subjects", [(s.value, w) for s, w in expected_subjects])

        print_reasoning("Subject detection helps boost chunks from relevant source types")

        results['stages']['subject_detection'] = {
            'expected_subjects': [(s.value, w) for s, w in expected_subjects]
        }
        self.log_trace('subject_detection', results['stages']['subject_detection'])

        # =========================================================
        # STAGE 6: Hybrid Retrieval
        # =========================================================
        print_section("Stage 6: Hybrid Retrieval (Vector + Graph)")

        if not self.driver:
            print_warning("Neo4j not connected - skipping retrieval")
            results['stages']['retrieval'] = {'error': 'Neo4j not connected'}
        else:
            # 6a: Vector Search
            print_step(6, "6a. Vector Search")

            vector_chunks = []
            with self.time_stage('vector_search'):
                try:
                    for q_var in query_variations[:2]:  # Limit for speed
                        chunks = self.hybrid_retriever._safe_vector_search(q_var, top_k=5)
                        vector_chunks.extend(chunks)
                    print_info("Vector Results", len(vector_chunks))
                except Exception as e:
                    print_warning(f"Vector search failed: {e}")

            # 6b: Graph Search
            print_step(6, "6b. Graph Search")

            graph_chunks = []
            with self.time_stage('graph_search'):
                try:
                    graph_chunks = self.hybrid_retriever._execute_graph_search(
                        retrieval_query, analysis, top_k=5
                    )
                    print_info("Graph Results", len(graph_chunks))
                except Exception as e:
                    print_warning(f"Graph search failed: {e}")

            print_reasoning("Vector finds semantically similar content; Graph finds structurally related content")

            # 6c: Merge Results
            print_step(6, "6c. Merge Results")

            all_results = {}
            with self.time_stage('merge_results'):
                self.hybrid_retriever._merge_results(all_results, vector_chunks, 'vector')
                self.hybrid_retriever._merge_results(all_results, graph_chunks, 'graph')

            print_info("Merged Results", len(all_results))

            # Count multi-source chunks
            multi_source = sum(1 for c in all_results.values() if c.retrieval_method == 'vector+graph')
            print_info("Multi-source Chunks", f"{multi_source} (boosted 1.3x)")

            print_reasoning("Chunks found by both vector AND graph search are likely more relevant")

            # 6d: Rerank with Subject Boost
            print_step(6, "6d. Rerank with Subject Boost & Deduplication")

            with self.time_stage('rerank'):
                final_chunks = self.hybrid_retriever._rerank(
                    all_results,
                    top_k=6,
                    expected_subjects=expected_subjects
                )

            print_info("Final Chunks", len(final_chunks))

            # Show final chunks
            print("\n  📄 Retrieved Chunks:")
            for i, chunk in enumerate(final_chunks):
                print(f"    {i+1}. [{chunk.retrieval_method}] {chunk.spec_id} - {chunk.section_title[:40]}")
                print(f"       Score: {chunk.retrieval_score:.3f} | Subject: {chunk.subject or 'N/A'}")
                print(f"       Content: {chunk.content[:100]}...")

            print_reasoning("Deduplication removes similar content; Subject boost prioritizes relevant sources")

            results['stages']['retrieval'] = {
                'vector_count': len(vector_chunks),
                'graph_count': len(graph_chunks),
                'merged_count': len(all_results),
                'multi_source_count': multi_source,
                'final_count': len(final_chunks),
                'final_chunks': [
                    {
                        'chunk_id': c.chunk_id,
                        'spec_id': c.spec_id,
                        'section_title': c.section_title,
                        'retrieval_method': c.retrieval_method,
                        'score': c.retrieval_score,
                        'subject': c.subject,
                        'content_preview': c.content[:200]
                    }
                    for c in final_chunks
                ]
            }
            self.log_trace('retrieval', results['stages']['retrieval'])

        # =========================================================
        # STAGE 7: Prompt Template Selection
        # =========================================================
        print_section("Stage 7: Prompt Template Selection")

        intent = analysis.get('primary_intent', 'general')
        print_step(7, f"PromptTemplates.get_prompt(intent='{intent}')")

        # Build context
        if 'retrieval' in results['stages'] and 'final_chunks' in results['stages']['retrieval']:
            context = self.ContextBuilder.build_context(final_chunks, max_chars=15000)
        else:
            context = "[No context available - retrieval failed]"

        # Get prompt
        with self.time_stage('prompt_building'):
            prompt = self.PromptTemplates.get_prompt(
                query=query,  # Use original query (with choices for MCQ)
                context=context,
                analysis=analysis,
                entities=analysis.get('entities', [])
            )

        print_info("Template Used", intent)
        print_info("Context Length", f"{len(context)} chars")
        print_info("Prompt Length", f"{len(prompt)} chars")

        # Show prompt preview
        print("\n  📝 Prompt Preview (first 500 chars):")
        print(f"    {prompt[:500]}...")

        template_descriptions = {
            'definition': 'Structured definition with characteristics and references',
            'comparison': 'Side-by-side comparison table format',
            'procedure': 'Step-by-step flow with sequence diagram',
            'network_function': 'Role, functions, interfaces format',
            'relationship': 'Interaction details with diagram',
            'multiple_choice': 'Strict "Answer: X. [option]" format',
            'general': 'Comprehensive answer with anti-hallucination rules'
        }
        print_reasoning(f"Using {intent} template: {template_descriptions.get(intent, 'General format')}")

        results['stages']['prompt_building'] = {
            'intent': intent,
            'template_used': intent,
            'context_length': len(context),
            'prompt_length': len(prompt),
            'prompt_preview': prompt[:1000]
        }
        self.log_trace('prompt_building', results['stages']['prompt_building'])

        # =========================================================
        # STAGE 8: LLM Generation (Optional)
        # =========================================================
        print_section("Stage 8: LLM Generation")

        if skip_llm:
            print_info("Skipped", "LLM generation skipped (--skip-llm flag)")
            results['final_answer'] = "[LLM generation skipped]"
        else:
            print_step(8, f"EnhancedLLMIntegrator.generate() with {model}")

            try:
                from rag_system_v3 import EnhancedLLMIntegrator

                claude_api_key = os.getenv('CLAUDE_API_KEY')
                local_llm_url = os.getenv('LOCAL_LLM_URL', 'http://192.168.1.14:11434/api/chat')

                llm = EnhancedLLMIntegrator(
                    claude_api_key=claude_api_key,
                    local_llm_url=local_llm_url
                )

                with self.time_stage('llm_generation'):
                    if model == "claude":
                        answer = llm._generate_with_claude(prompt)
                    else:
                        answer = llm._generate_with_ollama(prompt, model)

                print_info("Answer Length", f"{len(answer)} chars")
                print_success("LLM generation complete")

                results['final_answer'] = answer
                results['stages']['llm_generation'] = {
                    'model': model,
                    'answer_length': len(answer)
                }

            except Exception as e:
                print_warning(f"LLM generation failed: {e}")
                results['final_answer'] = f"[Error: {e}]"
                results['stages']['llm_generation'] = {'error': str(e)}

        # =========================================================
        # SUMMARY
        # =========================================================
        print_header("Pipeline Summary")

        # Timings
        results['timings'] = self.timings
        print_section("Timing Breakdown")
        total_time = sum(self.timings.values())
        for stage, ms in self.timings.items():
            pct = (ms / total_time * 100) if total_time > 0 else 0
            print(f"  {stage}: {ms:.1f}ms ({pct:.1f}%)")
        print(f"\n  {Colors.BOLD}Total: {total_time:.1f}ms{Colors.END}")

        # Final Answer
        if results['final_answer'] and not skip_llm:
            print_section("Final Answer")
            print(results['final_answer'])

        return results

    def export_trace(self, filename: str = None):
        """Export trace log to JSON file"""
        if filename is None:
            filename = f"debug_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.trace_log, f, indent=2, ensure_ascii=False, default=str)

        print_success(f"Trace exported to {filename}")
        return filename

    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()


def main():
    parser = argparse.ArgumentParser(
        description='Debug RAG Query Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python debug_query_pipeline.py "What is AMF?"
    python debug_query_pipeline.py "Compare SCP and SEPP" --skip-llm
    python debug_query_pipeline.py --interactive
    python debug_query_pipeline.py "What is UPF?" --export-trace
        """
    )

    parser.add_argument('query', nargs='?', help='Query to debug')
    parser.add_argument('--model', default='deepseek-r1:14b', help='LLM model to use')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM generation')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--export-trace', action='store_true', help='Export trace to JSON')
    parser.add_argument('--file', '-f', help='File with queries (one per line)')

    args = parser.parse_args()

    # Initialize debugger
    debugger = QueryPipelineDebugger()

    try:
        if args.interactive:
            # Interactive mode
            print_header("Interactive Debug Mode")
            print("Type 'quit' or 'q' to exit\n")

            while True:
                try:
                    query = input(f"{Colors.CYAN}Query>{Colors.END} ").strip()
                    if query.lower() in ['quit', 'q', 'exit']:
                        break
                    if not query:
                        continue

                    results = debugger.debug_query(query, model=args.model, skip_llm=args.skip_llm)

                    if args.export_trace:
                        debugger.export_trace()

                except KeyboardInterrupt:
                    print("\n")
                    break

        elif args.file:
            # File mode
            with open(args.file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]

            for i, query in enumerate(queries):
                print(f"\n{'='*60}")
                print(f"Query {i+1}/{len(queries)}")
                print(f"{'='*60}")

                results = debugger.debug_query(query, model=args.model, skip_llm=args.skip_llm)

        elif args.query:
            # Single query mode
            results = debugger.debug_query(args.query, model=args.model, skip_llm=args.skip_llm)

            if args.export_trace:
                debugger.export_trace()

        else:
            # Demo with sample queries
            print_header("Demo Mode - Sample Queries")

            sample_queries = [
                "What is AMF?",
                "Compare SCP and SEPP",
                "What is the first step in UE registration procedure?\nA. Authentication\nB. Initial NAS message\nC. PDU session\nD. Handover"
            ]

            for query in sample_queries:
                results = debugger.debug_query(query, model=args.model, skip_llm=True)
                print("\n" + "="*80 + "\n")

    finally:
        debugger.close()


if __name__ == "__main__":
    main()
