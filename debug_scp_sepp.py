#!/usr/bin/env python3
"""
Debug script for SCP vs SEPP query issue.
Tests the full RAG pipeline to identify where hallucination occurs.

Usage:
    python debug_scp_sepp.py
    python debug_scp_sepp.py --model gemma3:12b
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase
from rag_system_v2 import CypherQueryGenerator, create_rag_system_v2
from logging_config import setup_centralized_logging, get_logger, MAJOR, MINOR, DEBUG

setup_centralized_logging()
logger = get_logger('SCP_SEPP_Debug')


def check_term_nodes(driver):
    """Check if SCP and SEPP exist in Term nodes"""
    print("\n" + "=" * 80)
    print("STEP 1: Checking Term Nodes in Neo4j")
    print("=" * 80)

    with driver.session() as session:
        # Check SCP
        result = session.run("""
            MATCH (t:Term)
            WHERE t.abbreviation = 'SCP' OR toLower(t.abbreviation) = 'scp'
            RETURN t.abbreviation, t.full_name, t.term_type, t.source_specs
        """)
        scp_records = list(result)

        if scp_records:
            for r in scp_records:
                print(f"\nSCP Found:")
                print(f"  Abbreviation: {r['t.abbreviation']}")
                print(f"  Full Name: {r['t.full_name']}")
                print(f"  Type: {r['t.term_type']}")
                print(f"  Source Specs: {r['t.source_specs']}")
        else:
            print("\n❌ SCP NOT FOUND in Term nodes!")

        # Check SEPP
        result = session.run("""
            MATCH (t:Term)
            WHERE t.abbreviation = 'SEPP' OR toLower(t.abbreviation) = 'sepp'
            RETURN t.abbreviation, t.full_name, t.term_type, t.source_specs
        """)
        sepp_records = list(result)

        if sepp_records:
            for r in sepp_records:
                print(f"\nSEPP Found:")
                print(f"  Abbreviation: {r['t.abbreviation']}")
                print(f"  Full Name: {r['t.full_name']}")
                print(f"  Type: {r['t.term_type']}")
                print(f"  Source Specs: {r['t.source_specs']}")
        else:
            print("\n❌ SEPP NOT FOUND in Term nodes!")

        return len(scp_records) > 0, len(sepp_records) > 0


def check_query_analysis(generator, question):
    """Check how question is analyzed"""
    print("\n" + "=" * 80)
    print("STEP 2: Question Analysis")
    print("=" * 80)

    analysis = generator.analyze_question(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnalysis Result:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    return analysis


def check_generated_query(generator, question, analysis):
    """Check the generated Cypher query"""
    print("\n" + "=" * 80)
    print("STEP 3: Generated Cypher Query")
    print("=" * 80)

    query = generator.generate_cypher_query(question, analysis)

    print(f"\nGenerated Query (first 1000 chars):")
    print(query[:1000])
    if len(query) > 1000:
        print("... [truncated]")

    return query


def check_retrieved_chunks(retriever, question):
    """Execute retrieval and check retrieved chunks"""
    print("\n" + "=" * 80)
    print("STEP 4: Retrieved Chunks (via Retriever)")
    print("=" * 80)

    try:
        chunks, cypher_query, strategy = retriever.retrieve_with_cypher(question)

        print(f"\nStrategy: {strategy}")
        print(f"Total chunks retrieved: {len(chunks)}")

        for i, chunk in enumerate(chunks[:8], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Spec ID: {chunk.spec_id}")
            print(f"Section: {chunk.section_title}")
            print(f"Type: {chunk.chunk_type}")

            content = chunk.content
            if content:
                # Check if content mentions SCP or SEPP
                has_scp = 'SCP' in content or 'Service Communication Proxy' in content
                has_sepp = 'SEPP' in content or 'Security Edge Protection Proxy' in content
                print(f"Contains SCP mention: {has_scp}")
                print(f"Contains SEPP mention: {has_sepp}")
                print(f"Content preview: {content[:300]}...")

        return chunks

    except Exception as e:
        print(f"\n❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_full_rag(question, model):
    """Test full RAG pipeline"""
    print("\n" + "=" * 80)
    print("STEP 5: Full RAG Response")
    print("=" * 80)

    claude_key = os.getenv('CLAUDE_API_KEY')
    deepseek_url = os.getenv('DEEPSEEK_API_URL', 'http://192.168.1.14:11434/api/chat')

    rag = create_rag_system_v2(
        claude_api_key=claude_key,
        deepseek_api_url=deepseek_url
    )

    print(f"\nUsing model: {model}")
    print(f"Question: {question}")
    print("\nGenerating response...")

    response = rag.query(question, model=model)

    print(f"\n--- RAG Response ---")
    print(f"Answer:\n{response.answer}")
    print(f"\nChunks used: {len(response.retrieved_chunks)}")
    print(f"Query type: {response.query_analysis.get('question_type', 'N/A')}")

    # Check if response mentions correct full names
    answer_lower = response.answer.lower()

    print("\n--- Fact Check ---")

    # SCP should be "Service Communication Proxy" NOT "Service Capability Platform"
    if 'service capability platform' in answer_lower:
        print("❌ HALLUCINATION: 'Service Capability Platform' is WRONG!")
        print("   SCP = Service Communication Proxy")
    elif 'service communication proxy' in answer_lower:
        print("✓ Correct: SCP = Service Communication Proxy")
    else:
        print("? SCP full name not mentioned in response")

    # SEPP should be "Security Edge Protection Proxy"
    if 'security edge protection proxy' in answer_lower:
        print("✓ Correct: SEPP = Security Edge Protection Proxy")
    elif 'session management' in answer_lower and 'sepp' in answer_lower:
        print("❌ HALLUCINATION: SEPP is NOT about Session Management!")
    else:
        print("? SEPP full name not mentioned in response")

    rag.close()
    return response


def main():
    parser = argparse.ArgumentParser(description='Debug SCP vs SEPP query')
    parser.add_argument('--model', type=str, default='gemma3:12b',
                       help='LLM model to use (default: gemma3:12b)')
    parser.add_argument('--skip-rag', action='store_true',
                       help='Skip full RAG test (only check Term nodes and query)')
    args = parser.parse_args()

    question = "What is the difference between SCP and SEPP in 5G Core?"

    print("=" * 80)
    print("DEBUG: SCP vs SEPP Query")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Model: {args.model}")

    # Connect to Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Import retriever
    from rag_system_v2 import EnhancedKnowledgeRetriever

    try:
        # Step 1: Check Term nodes
        scp_exists, sepp_exists = check_term_nodes(driver)

        # Step 2-3: Check query generation
        generator = CypherQueryGenerator(neo4j_driver=driver)
        analysis = check_query_analysis(generator, question)
        query = check_generated_query(generator, question, analysis)

        # Step 4: Check retrieved chunks using retriever
        retriever = EnhancedKnowledgeRetriever(neo4j_uri, neo4j_user, neo4j_password)
        chunks = check_retrieved_chunks(retriever, question)
        retriever.close()

        # Step 5: Full RAG test
        if not args.skip_rag:
            test_full_rag(question, args.model)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"SCP in Term nodes: {'✓' if scp_exists else '❌'}")
        print(f"SEPP in Term nodes: {'✓' if sepp_exists else '❌'}")
        print(f"Question type detected: {analysis.get('question_type', 'N/A')}")
        print(f"Entities found: {len(analysis.get('entities', []))}")
        print(f"Chunks retrieved: {len(chunks)}")

        if not scp_exists or not sepp_exists:
            print("\n⚠️  Missing Term nodes may cause hallucination!")
            print("   Run: python orchestrator.py init-kg")

    finally:
        driver.close()


if __name__ == '__main__':
    main()
