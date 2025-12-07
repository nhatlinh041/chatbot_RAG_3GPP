"""
Fix missing CONTAINS relationships in Knowledge Graph
The init-kg process created nodes but relationships were not created properly
"""

from neo4j import GraphDatabase
from tqdm import tqdm

def fix_contains_relationships():
    """Create CONTAINS relationships between Document and Chunk nodes"""

    driver = GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "password")
    )

    print("\n" + "=" * 70)
    print("FIXING CONTAINS RELATIONSHIPS")
    print("=" * 70)

    with driver.session() as session:
        # Check current state
        print("\nChecking current state...")
        result = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
        current_count = result.single()['count']
        print(f"  Current CONTAINS relationships: {current_count:,}")

        result = session.run("MATCH (d:Document) RETURN count(d) as count")
        doc_count = result.single()['count']
        print(f"  Total Documents: {doc_count}")

        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        chunk_count = result.single()['count']
        print(f"  Total Chunks: {chunk_count:,}")

        # Get all chunk-document pairs
        print("\nQuerying chunk-document pairs...")
        result = session.run("""
            MATCH (c:Chunk)
            RETURN c.spec_id as spec_id, c.chunk_id as chunk_id
        """)

        pairs = [(record['spec_id'], record['chunk_id']) for record in result]
        print(f"  Found {len(pairs):,} chunk-document pairs")

        # Create relationships in batches
        print("\nCreating CONTAINS relationships...")
        batch_size = 500
        total_created = 0

        for i in tqdm(range(0, len(pairs), batch_size)):
            batch = pairs[i:i+batch_size]

            # Create relationships for this batch
            result = session.run("""
                UNWIND $pairs as pair
                MATCH (d:Document {spec_id: pair.spec_id})
                MATCH (c:Chunk {chunk_id: pair.chunk_id})
                MERGE (d)-[:CONTAINS]->(c)
                RETURN count(*) as created
            """, pairs=[{"spec_id": spec_id, "chunk_id": chunk_id} for spec_id, chunk_id in batch])

            created = result.single()['created']
            total_created += created

        print(f"\n✅ Created {total_created:,} CONTAINS relationships")

        # Verify final state
        print("\nVerifying final state...")
        result = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
        final_count = result.single()['count']
        print(f"  Final CONTAINS relationships: {final_count:,}")

        if final_count == chunk_count:
            print(f"\n✅ ✅ ✅ SUCCESS! All {chunk_count:,} chunks connected to documents")
        else:
            print(f"\n⚠️  Warning: Expected {chunk_count:,} relationships, got {final_count:,}")

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    fix_contains_relationships()
