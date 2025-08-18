import json
import re
from typing import List, Dict, Optional, Set
from neo4j import GraphDatabase
from pathlib import Path
import anthropic
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievedChunk:
    chunk_id: str
    spec_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    depth_level: int
    reference_path: List[str]


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievedChunk]
    query: str
    retrieval_depth: int
    timestamp: datetime


class KnowledgeRetriever:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def close(self):
        self.driver.close()
    
    def _extract_base_spec_id(self, full_spec_id: str) -> str:
        """Extract base spec ID from full spec ID with section numbers"""
        # Convert "ts_29.503.6.10.9" to "ts_29.503"
        parts = full_spec_id.split('.')
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}"
        return full_spec_id
    
    def retrieve_with_depth(self, query: str, max_depth: int = 2, max_chunks_per_level: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve knowledge with customizable reference depth
        
        Args:
            query: Search query
            max_depth: Maximum reference depth to explore (0 = direct matches only)
            max_chunks_per_level: Maximum chunks to retrieve per depth level
        """
        retrieved_chunks = []
        visited_chunks = set()
        
        with self.driver.session() as session:
            # Level 0: Direct matches
            primary_chunks = self._get_direct_matches(session, query, max_chunks_per_level, visited_chunks)
            retrieved_chunks.extend(primary_chunks)
            
            # Follow references up to max_depth
            current_chunks = primary_chunks
            for depth in range(1, max_depth + 1):
                if not current_chunks:
                    break
                
                next_level_chunks = []
                for chunk in current_chunks:
                    referenced_chunks = self._get_referenced_chunks(
                        session, chunk.chunk_id, query, max_chunks_per_level, visited_chunks, depth, chunk.reference_path
                    )
                    next_level_chunks.extend(referenced_chunks)
                
                retrieved_chunks.extend(next_level_chunks)
                current_chunks = next_level_chunks
        
        return retrieved_chunks
    
    def _get_direct_matches(self, session, query: str, limit: int, visited_chunks: Set[str]) -> List[RetrievedChunk]:
        """Get chunks that directly match the query"""
        cypher_query = """
        MATCH (c:Chunk)
        WHERE toLower(c.content) CONTAINS toLower($query_text)
        AND NOT c.chunk_id IN $visited
        RETURN c.chunk_id as chunk_id, c.spec_id as spec_id, c.section_id as section_id,
               c.section_title as section_title, c.content as content, c.chunk_type as chunk_type
        ORDER BY size(c.content) DESC
        LIMIT $limit
        """
        
        result = session.run(cypher_query, query_text=query, limit=limit, visited=list(visited_chunks))
        chunks = []
        
        for record in result:
            chunk_id = record["chunk_id"]
            if chunk_id not in visited_chunks:
                visited_chunks.add(chunk_id)
                chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    spec_id=self._extract_base_spec_id(record["spec_id"]),
                    section_id=record["section_id"],
                    section_title=record["section_title"],
                    content=record["content"],
                    chunk_type=record["chunk_type"],
                    depth_level=0,
                    reference_path=[chunk_id]
                ))
        
        return chunks
    
    def _get_referenced_chunks(self, session, source_chunk_id: str, query: str, limit: int, 
                             visited_chunks: Set[str], depth: int, reference_path: List[str]) -> List[RetrievedChunk]:
        """Get chunks referenced by the source chunk that also relate to the query"""
        cypher_query = """
        MATCH (source:Chunk {chunk_id: $source_chunk_id})
        MATCH (source)-[:REFERENCES_CHUNK]->(ref_chunk:Chunk)
        WHERE toLower(ref_chunk.content) CONTAINS toLower($query_text)
        AND NOT ref_chunk.chunk_id IN $visited
        RETURN ref_chunk.chunk_id as chunk_id, ref_chunk.spec_id as spec_id,
               ref_chunk.section_id as section_id, ref_chunk.section_title as section_title,
               ref_chunk.content as content, ref_chunk.chunk_type as chunk_type
        ORDER BY size(ref_chunk.content) DESC
        LIMIT $limit
        """
        
        result = session.run(cypher_query, source_chunk_id=source_chunk_id, query_text=query, 
                           limit=limit, visited=list(visited_chunks))
        chunks = []
        
        for record in result:
            chunk_id = record["chunk_id"]
            if chunk_id not in visited_chunks:
                visited_chunks.add(chunk_id)
                chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    spec_id=self._extract_base_spec_id(record["spec_id"]),
                    section_id=record["section_id"],
                    section_title=record["section_title"],
                    content=record["content"],
                    chunk_type=record["chunk_type"],
                    depth_level=depth,
                    reference_path=reference_path + [chunk_id]
                ))
        
        return chunks
    
    def get_chunk_references(self, chunk_id: str) -> List[str]:
        """Get all chunks referenced by a specific chunk"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (source:Chunk {chunk_id: $chunk_id})-[:REFERENCES_CHUNK]->(ref:Chunk)
                RETURN ref.chunk_id as ref_chunk_id
            """, chunk_id=chunk_id)
            
            return [record["ref_chunk_id"] for record in result]


class ClaudeIntegrator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Generate answer using Claude with retrieved knowledge"""
        
        # Prepare context from retrieved chunks with limiting
        context_parts = []
        total_chars = 0
        max_chars = 30000  # Conservative limit
        
        for chunk in retrieved_chunks:
            # Truncate content if too long
            content = chunk.content
            if len(content) > 2000:  # Limit individual chunks
                content = content[:2000] + "... [truncated]"
            
            chunk_text = f"""
**Source: {chunk.spec_id} - {chunk.section_title} (Depth: {chunk.depth_level})**
**Section ID: {chunk.section_id}**

{content}
---
"""
            
            # Check if adding this chunk would exceed limit
            if total_chars + len(chunk_text) > max_chars:
                break
                
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert on 3GPP specifications. Based on the provided context from 3GPP documents, please answer the following question:

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Provide a comprehensive answer based on the provided context
2. Reference specific sections and specifications when relevant
3. If the context doesn't fully answer the question, clearly state what information is missing
4. Use technical terminology appropriately
5. Structure your answer clearly with headings if helpful

**Answer:**"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"


class RAGOrchestrator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, claude_api_key: str):
        self.retriever = KnowledgeRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.claude_integrator = ClaudeIntegrator(claude_api_key)
    
    def close(self):
        self.retriever.close()
    
    def _extract_search_terms(self, question: str) -> str:
        """Extract key search terms from a question"""
        # Common 3GPP/5G terms that should be prioritized
        important_terms = [
            'AMF', 'SMF', 'UPF', 'PCF', 'UDM', 'AUSF', 'NSSF', 'NEF', 'NRF',
            '5GC', '5GS', 'gNB', 'eNB', 'UE', 'QoS', 'PDU', 'DNN', 'SUPI',
            'authentication', 'registration', 'mobility', 'session', 'security',
            'policy', 'charging', 'slicing', 'handover', 'roaming'
        ]
        
        question_upper = question.upper()
        found_terms = []
        
        # Find important terms in the question
        for term in important_terms:
            if term.upper() in question_upper:
                found_terms.append(term)
        
        # If we found important terms, use them, otherwise use the full question
        if found_terms:
            return ' '.join(found_terms)
        else:
            return question
    
    def query(self, question: str, max_depth: int = 2, max_chunks_per_level: int = 5) -> RAGResponse:
        """
        Main RAG query interface
        
        Args:
            question: User's question
            max_depth: Maximum reference depth (0-5 recommended)
            max_chunks_per_level: Maximum chunks per depth level (3-10 recommended)
        """
        
        # Extract key terms from question for better retrieval
        search_query = self._extract_search_terms(question)
        
        # Retrieve relevant knowledge
        retrieved_chunks = self.retriever.retrieve_with_depth(
            query=search_query,
            max_depth=max_depth,
            max_chunks_per_level=max_chunks_per_level
        )
        
        if not retrieved_chunks:
            return RAGResponse(
                answer="No relevant information found in the knowledge base.",
                sources=[],
                query=question,
                retrieval_depth=max_depth,
                timestamp=datetime.now()
            )
        
        # Generate answer using Claude
        answer = self.claude_integrator.generate_answer(question, retrieved_chunks)
        
        return RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
            query=question,
            retrieval_depth=max_depth,
            timestamp=datetime.now()
        )
    
    def explain_retrieval(self, question: str, max_depth: int = 2) -> Dict:
        """Explain how the retrieval process works for a given question"""
        
        retrieved_chunks = self.retriever.retrieve_with_depth(
            query=question,
            max_depth=max_depth,
            max_chunks_per_level=3
        )
        
        explanation = {
            "query": question,
            "max_depth": max_depth,
            "total_chunks_found": len(retrieved_chunks),
            "chunks_by_depth": {},
            "reference_chains": []
        }
        
        # Group by depth
        for chunk in retrieved_chunks:
            depth = chunk.depth_level
            if depth not in explanation["chunks_by_depth"]:
                explanation["chunks_by_depth"][depth] = []
            
            explanation["chunks_by_depth"][depth].append({
                "chunk_id": chunk.chunk_id,
                "spec_id": chunk.spec_id,
                "section_title": chunk.section_title,
                "reference_path": chunk.reference_path
            })
        
        # Track reference chains
        for chunk in retrieved_chunks:
            if len(chunk.reference_path) > 1:
                explanation["reference_chains"].append({
                    "chain": chunk.reference_path,
                    "specs": [c.spec_id for c in retrieved_chunks if c.chunk_id in chunk.reference_path]
                })
        
        return explanation


# Example usage and configuration
class RAGConfig:
    def __init__(self):
        self.neo4j_uri = "neo4j://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.claude_api_key = None  # Set this with your Claude API key
        
        # Default retrieval settings
        self.default_max_depth = 2
        self.default_max_chunks_per_level = 5
        
        # Depth recommendations by use case
        self.depth_recommendations = {
            "quick_lookup": 0,      # Direct matches only
            "standard_query": 1,    # Direct + 1 level of references
            "comprehensive": 2,     # Direct + 2 levels of references
            "deep_research": 3,     # For complex interconnected topics
            "exhaustive": 4         # Maximum depth (use carefully)
        }


def create_rag_system(claude_api_key: str = None) -> RAGOrchestrator:
    """Factory function to create a configured RAG system"""
    config = RAGConfig()
    if claude_api_key:
        config.claude_api_key = claude_api_key
    
    if not config.claude_api_key:
        raise ValueError("Claude API key is required. Get one from https://console.anthropic.com/")
    
    return RAGOrchestrator(
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
        claude_api_key=config.claude_api_key
    )


if __name__ == "__main__":
    # Example usage (requires Claude API key)
    # rag_system = create_rag_system("your-claude-api-key-here")
    
    # try:
    #     # Standard query with depth 2
    #     response = rag_system.query(
    #         question="What is the role of AMF in 5G architecture?",
    #         max_depth=2,
    #         max_chunks_per_level=5
    #     )
    #     
    #     print(f"Answer: {response.answer}")
    #     print(f"Sources found: {len(response.sources)}")
    #     
    #     # Show retrieval explanation
    #     explanation = rag_system.explain_retrieval("AMF architecture", max_depth=2)
    #     print(f"Retrieval explanation: {json.dumps(explanation, indent=2)}")
    #     
    # finally:
    #     rag_system.close()
    
    print("RAG system module loaded successfully!")
    print("To use: rag_system = create_rag_system('your-claude-api-key')")