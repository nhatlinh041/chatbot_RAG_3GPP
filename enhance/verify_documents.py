"""
Verify document processing quality and KG readiness
Run this before starting Phase 1 research
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
import sys


def check_json_documents():
    """Check processed JSON documents"""
    print("=" * 60)
    print("CHECKING PROCESSED JSON DOCUMENTS")
    print("=" * 60)

    json_dir = Path("3GPP_JSON_DOC/processed_json_v2")

    if not json_dir.exists():
        print(f"❌ Directory not found: {json_dir}")
        print("   Please create directory and process documents first")
        return None

    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {json_dir}")
        print("   Please run document processing pipeline")
        return None

    stats = {
        'total_docs': 0,
        'total_chunks': 0,
        'docs_with_refs': 0,
        'avg_chunks_per_doc': 0,
        'specs': []
    }

    print(f"\n📂 Processing {len(json_files)} JSON files...\n")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            spec_id = data.get('metadata', {}).get('specification_id', json_file.stem)
            num_chunks = len(data.get('chunks', []))

            stats['total_docs'] += 1
            stats['total_chunks'] += num_chunks
            stats['specs'].append({'id': spec_id, 'chunks': num_chunks})

            # Check cross-references
            has_refs = any(
                chunk.get('cross_references', {}).get('external', [])
                for chunk in data.get('chunks', [])
            )
            if has_refs:
                stats['docs_with_refs'] += 1

        except Exception as e:
            print(f"⚠️  Error processing {json_file.name}: {e}")

    if stats['total_docs'] > 0:
        stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['total_docs']

    # Print results
    print("📊 DOCUMENT STATISTICS")
    print("-" * 60)
    print(f"  Total specifications: {stats['total_docs']}")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Avg chunks per doc: {stats['avg_chunks_per_doc']:.1f}")
    print(f"  Docs with cross-refs: {stats['docs_with_refs']}")

    # Show top specs
    print("\n📄 TOP SPECIFICATIONS (by chunk count):")
    top_specs = sorted(stats['specs'], key=lambda x: x['chunks'], reverse=True)[:10]
    for i, spec in enumerate(top_specs, 1):
        print(f"  {i:2d}. {spec['id']:30s} - {spec['chunks']:5,} chunks")

    # Check requirements
    print("\n✅ REQUIREMENTS CHECK")
    print("-" * 60)

    requirements = [
        (stats['total_docs'] >= 50, f"At least 50 specs (current: {stats['total_docs']})"),
        (stats['total_chunks'] >= 10000, f"At least 10K chunks (current: {stats['total_chunks']:,})"),
        (stats['docs_with_refs'] >= 10, f"At least 10 docs with refs (current: {stats['docs_with_refs']})")
    ]

    all_passed = True
    for passed, msg in requirements:
        status = "✅" if passed else "❌"
        print(f"  {status} {msg}")
        if not passed:
            all_passed = False

    return stats if all_passed else None


def check_knowledge_graph():
    """Check Neo4j Knowledge Graph"""
    print("\n" + "=" * 60)
    print("CHECKING KNOWLEDGE GRAPH")
    print("=" * 60)

    try:
        driver = GraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "password")
        )

        with driver.session() as session:
            # Test connection
            session.run("RETURN 1")
            print("\n✅ Neo4j connection successful")

            # Count nodes
            print("\n📦 NODE COUNTS:")
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)

            node_stats = {}
            for record in result:
                label = record['label']
                count = record['count']
                node_stats[label] = count
                print(f"  {label:20s}: {count:8,}")

            # Count relationships
            print("\n🔗 RELATIONSHIP COUNTS:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)

            rel_stats = {}
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                rel_stats[rel_type] = count
                print(f"  {rel_type:20s}: {count:8,}")

            # Check requirements
            print("\n✅ KG REQUIREMENTS CHECK")
            print("-" * 60)

            kg_requirements = [
                (node_stats.get('Document', 0) >= 50, f"At least 50 Documents (current: {node_stats.get('Document', 0)})"),
                (node_stats.get('Chunk', 0) >= 10000, f"At least 10K Chunks (current: {node_stats.get('Chunk', 0):,})"),
                (rel_stats.get('CONTAINS', 0) >= 10000, f"At least 10K CONTAINS (current: {rel_stats.get('CONTAINS', 0):,})"),
                (rel_stats.get('REFERENCES_SPEC', 0) >= 100, f"At least 100 REFERENCES_SPEC (current: {rel_stats.get('REFERENCES_SPEC', 0)})")
            ]

            kg_passed = True
            for passed, msg in kg_requirements:
                status = "✅" if passed else "❌"
                print(f"  {status} {msg}")
                if not passed:
                    kg_passed = False

            driver.close()
            return kg_passed

    except Exception as e:
        print(f"\n❌ Failed to connect to Neo4j: {e}")
        print("\nPlease ensure:")
        print("  1. Neo4j is running (python orchestrator.py start-neo4j)")
        print("  2. KG is initialized (python orchestrator.py init-kg)")
        return False


def main():
    print("\n" + "=" * 60)
    print("3GPP RAG SYSTEM - DOCUMENT & KG VERIFICATION")
    print("=" * 60)

    # Check documents
    doc_stats = check_json_documents()

    # Check KG
    kg_ok = check_knowledge_graph()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if doc_stats and kg_ok:
        print("\n✅ ✅ ✅ ALL CHECKS PASSED ✅ ✅ ✅")
        print("\nYou are ready to proceed with:")
        print("  → Phase 1: Baseline & Evaluation Framework")
        print("  → Phase 2: Knowledge Graph Enhancement")
        print("  → Phase 3: Hybrid Retrieval")
        print("\nNext step:")
        print("  cd enhance")
        print("  # Start implementing Phase 1")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("\nAction required:")

        if not doc_stats:
            print("\n📥 DOCUMENT PROCESSING:")
            print("  1. Download 3GPP specs from https://www.3gpp.org/ftp/Specs/archive/")
            print("  2. Place .docx files in: document_processing/input/")
            print("  3. Run: cd document_processing && python process_documents.py")

        if not kg_ok:
            print("\n🗄️  KNOWLEDGE GRAPH:")
            print("  1. Start Neo4j: python orchestrator.py start-neo4j")
            print("  2. Initialize KG: python orchestrator.py init-kg")
            print("  3. Or run both: python orchestrator.py all --init-kg")

        print("\nThen re-run this script:")
        print("  python enhance/verify_documents.py")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
