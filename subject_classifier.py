"""
Subject Classifier for Knowledge Graph Enhancement

Classifies chunks into subjects based on tele_qna benchmark categories:
- Standards specifications: Specific 3GPP procedures, IEs, messages
- Standards overview: Architecture, overview, introduction
- Lexicon: Abbreviations, definitions, terminology
- Research publications: Algorithms, techniques, methods
- Research overview: General concepts, surveys
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Subject(Enum):
    """Subject categories matching tele_qna benchmark"""
    STANDARDS_SPECIFICATIONS = "Standards specifications"
    STANDARDS_OVERVIEW = "Standards overview"
    LEXICON = "Lexicon"
    RESEARCH_PUBLICATIONS = "Research publications"
    RESEARCH_OVERVIEW = "Research overview"


@dataclass
class SubjectClassification:
    """Result of subject classification"""
    subject: Subject
    confidence: float
    reason: str


class SubjectClassifier:
    """Classify chunks and queries into subjects"""

    # Keywords indicating Standards specifications
    STANDARDS_SPEC_KEYWORDS = [
        'procedure', 'ie ', 'information element', 'message', 'timer',
        'state machine', 'nas ', 'rrc ', 'ngap ', 'xnap ', 'f1ap ',
        'pdcp', 'rlc', 'mac ', 'phy ', 'harq', 'drb', 'srb',
        'service operation', 'qos flow', 'pdu session'
    ]

    # Keywords indicating Standards overview
    STANDARDS_OVERVIEW_KEYWORDS = [
        'overview', 'architecture', 'introduction', 'general',
        'reference model', 'functional', 'deployment', 'use case',
        'service', 'feature', 'capability', 'scenario'
    ]

    # Keywords indicating Lexicon
    LEXICON_KEYWORDS = [
        'abbreviation', 'definition', 'terminology', 'acronym',
        'vocabulary', 'glossary'
    ]

    # Keywords indicating Research publications
    RESEARCH_PUB_KEYWORDS = [
        'algorithm', 'optimization', 'machine learning', 'deep learning',
        'neural network', 'theorem', 'proof', 'simulation', 'experimental',
        'performance analysis', 'complexity', 'convergence'
    ]

    # Keywords indicating Research overview
    RESEARCH_OVERVIEW_KEYWORDS = [
        'survey', 'review', 'state of the art', 'trend', 'challenge',
        'future', 'evolution', 'comparison', 'taxonomy'
    ]

    # 3GPP spec pattern (matches TS_23.501, TS 23.501, TR_38.401, etc.)
    SPEC_PATTERN = re.compile(r'TS[_\s]*\d+[\._]\d+|TR[_\s]*\d+[\._]\d+|3GPP\s+Release\s+\d+', re.IGNORECASE)

    def __init__(self):
        self.subject_priorities = {
            Subject.STANDARDS_SPECIFICATIONS: 1,
            Subject.STANDARDS_OVERVIEW: 2,
            Subject.LEXICON: 3,
            Subject.RESEARCH_PUBLICATIONS: 4,
            Subject.RESEARCH_OVERVIEW: 5
        }

    def classify_chunk(self, chunk: Dict) -> SubjectClassification:
        """
        Classify a chunk into a subject category

        Args:
            chunk: Chunk dict with section_title, content, chunk_type

        Returns:
            SubjectClassification with subject, confidence, reason
        """
        section_title = chunk.get('section_title', '').lower()
        content = chunk.get('content', '').lower()
        chunk_type = chunk.get('chunk_type', '').lower()
        spec_id = chunk.get('spec_id', '')

        # Check Lexicon first (highest confidence for abbreviations/definitions)
        if chunk_type in ['abbreviation', 'definition']:
            return SubjectClassification(
                subject=Subject.LEXICON,
                confidence=0.95,
                reason=f"chunk_type is {chunk_type}"
            )

        if any(kw in section_title for kw in self.LEXICON_KEYWORDS):
            return SubjectClassification(
                subject=Subject.LEXICON,
                confidence=0.9,
                reason=f"section_title contains lexicon keyword"
            )

        # Check if it's a 3GPP spec
        is_3gpp = bool(self.SPEC_PATTERN.search(spec_id))

        # Check Standards specifications keywords
        spec_score = sum(1 for kw in self.STANDARDS_SPEC_KEYWORDS
                        if kw in section_title or kw in content[:500])

        if spec_score >= 2 or any(kw in section_title for kw in ['procedure', 'ie ', 'message']):
            return SubjectClassification(
                subject=Subject.STANDARDS_SPECIFICATIONS,
                confidence=min(0.7 + spec_score * 0.05, 0.95),
                reason=f"matched {spec_score} standards spec keywords"
            )

        # Check Standards overview keywords
        overview_score = sum(1 for kw in self.STANDARDS_OVERVIEW_KEYWORDS
                            if kw in section_title)

        if overview_score >= 1:
            return SubjectClassification(
                subject=Subject.STANDARDS_OVERVIEW,
                confidence=0.7 + overview_score * 0.1,
                reason=f"section_title contains overview keyword"
            )

        # Check Research publications keywords
        research_score = sum(1 for kw in self.RESEARCH_PUB_KEYWORDS
                            if kw in content[:1000])

        if research_score >= 2:
            return SubjectClassification(
                subject=Subject.RESEARCH_PUBLICATIONS,
                confidence=min(0.6 + research_score * 0.1, 0.9),
                reason=f"matched {research_score} research pub keywords"
            )

        # Check Research overview keywords
        if any(kw in section_title or kw in content[:500] for kw in self.RESEARCH_OVERVIEW_KEYWORDS):
            return SubjectClassification(
                subject=Subject.RESEARCH_OVERVIEW,
                confidence=0.7,
                reason="matched research overview keyword"
            )

        # Default based on spec_id
        if is_3gpp:
            return SubjectClassification(
                subject=Subject.STANDARDS_SPECIFICATIONS,
                confidence=0.6,
                reason="default for 3GPP spec"
            )

        return SubjectClassification(
            subject=Subject.RESEARCH_OVERVIEW,
            confidence=0.5,
            reason="default classification"
        )

    def detect_query_subjects(self, query: str) -> List[Tuple[Subject, float]]:
        """
        Detect expected subject(s) from a query

        Args:
            query: User query string

        Returns:
            List of (Subject, weight) tuples, sorted by relevance
        """
        query_lower = query.lower()
        results = []

        # Standards specifications indicators
        spec_indicators = [
            '3gpp release', 'ts ', 'tr ', 'procedure', 'ie ',
            'information element', 'message format', 'state machine',
            'service operation', 'nas', 'rrc'
        ]
        if any(ind in query_lower for ind in spec_indicators):
            results.append((Subject.STANDARDS_SPECIFICATIONS, 1.5))

        # Lexicon indicators
        lexicon_indicators = [
            'what does', 'stand for', 'stands for', 'definition of',
            'what is the meaning', 'abbreviation', 'acronym'
        ]
        if any(ind in query_lower for ind in lexicon_indicators):
            results.append((Subject.LEXICON, 1.5))
            # Also add standards specs as backup for term definitions
            if (Subject.STANDARDS_SPECIFICATIONS, 1.5) not in results:
                results.append((Subject.STANDARDS_SPECIFICATIONS, 1.2))

        # Standards overview indicators
        overview_indicators = [
            'architecture', 'overview', 'how does', 'what is',
            'role of', 'function of', 'purpose of'
        ]
        if any(ind in query_lower for ind in overview_indicators):
            if not results:  # Only if no stronger match
                results.append((Subject.STANDARDS_OVERVIEW, 1.3))
                results.append((Subject.STANDARDS_SPECIFICATIONS, 1.1))

        # Research indicators
        research_indicators = [
            'algorithm', 'technique', 'method', 'optimization',
            'machine learning', 'neural network', 'performance'
        ]
        if any(ind in query_lower for ind in research_indicators):
            results.append((Subject.RESEARCH_PUBLICATIONS, 1.4))
            results.append((Subject.RESEARCH_OVERVIEW, 1.2))

        # Sort by weight descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results if results else [(Subject.STANDARDS_SPECIFICATIONS, 1.0)]

    def get_subject_boost(self, chunk_subject: str, expected_subjects: List[Tuple[Subject, float]]) -> float:
        """
        Calculate score boost based on subject match

        Args:
            chunk_subject: Subject string of the chunk
            expected_subjects: List of (Subject, weight) from query analysis

        Returns:
            Boost multiplier (1.0 = no boost)
        """
        if not expected_subjects:
            return 1.0

        for subject, weight in expected_subjects:
            if subject.value == chunk_subject:
                return weight

        return 1.0

    def classify_batch(self, chunks: List[Dict]) -> Dict[str, SubjectClassification]:
        """
        Classify multiple chunks

        Args:
            chunks: List of chunk dicts

        Returns:
            Dict mapping chunk_id to SubjectClassification
        """
        results = {}
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            results[chunk_id] = self.classify_chunk(chunk)
        return results


# Cypher queries for subject enhancement
SUBJECT_CYPHER_QUERIES = {
    'create_subjects': """
        // Create Subject nodes
        MERGE (s1:Subject {name: 'Standards specifications'})
        SET s1.priority = 1, s1.description = 'Specific 3GPP procedures, IEs, messages'

        MERGE (s2:Subject {name: 'Standards overview'})
        SET s2.priority = 2, s2.description = 'Architecture, overview, introduction'

        MERGE (s3:Subject {name: 'Lexicon'})
        SET s3.priority = 3, s3.description = 'Abbreviations, definitions, terminology'

        MERGE (s4:Subject {name: 'Research publications'})
        SET s4.priority = 4, s4.description = 'Algorithms, techniques, methods'

        MERGE (s5:Subject {name: 'Research overview'})
        SET s5.priority = 5, s5.description = 'General concepts, surveys'
    """,

    'add_chunk_subject': """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        SET c.subject = $subject, c.subject_confidence = $confidence
    """,

    'create_has_subject': """
        MATCH (c:Chunk), (s:Subject)
        WHERE c.subject = s.name
        MERGE (c)-[:HAS_SUBJECT]->(s)
    """,

    'get_chunks_by_subject': """
        MATCH (c:Chunk)-[:HAS_SUBJECT]->(s:Subject)
        WHERE s.name IN $subjects
        RETURN c, s.priority as subject_priority
        ORDER BY subject_priority ASC
    """,

    'subject_boosted_search': """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k * 2, $query_embedding)
        YIELD node AS c, score
        OPTIONAL MATCH (c)-[:HAS_SUBJECT]->(s:Subject)
        WITH c, score, s,
             CASE
                 WHEN s.name IN $expected_subjects THEN $boost_factor
                 ELSE 1.0
             END AS subject_boost
        RETURN c, score * subject_boost AS boosted_score
        ORDER BY boosted_score DESC
        LIMIT $top_k
    """
}


if __name__ == "__main__":
    # Test the classifier
    classifier = SubjectClassifier()

    # Test chunk classification
    test_chunks = [
        {
            'chunk_id': 'test_1',
            'section_title': 'Abbreviations',
            'content': 'AMF: Access and Mobility Management Function',
            'chunk_type': 'abbreviation',
            'spec_id': 'TS_23.501'
        },
        {
            'chunk_id': 'test_2',
            'section_title': '4.2.2 Registration procedure',
            'content': 'The registration procedure is used when...',
            'chunk_type': 'procedure',
            'spec_id': 'TS_23.502'
        },
        {
            'chunk_id': 'test_3',
            'section_title': 'System architecture overview',
            'content': 'The 5G system architecture consists of...',
            'chunk_type': 'overview',
            'spec_id': 'TS_23.501'
        }
    ]

    print("Chunk Classification Test:")
    print("=" * 50)
    for chunk in test_chunks:
        result = classifier.classify_chunk(chunk)
        print(f"Chunk: {chunk['section_title']}")
        print(f"  Subject: {result.subject.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reason: {result.reason}")
        print()

    # Test query subject detection
    test_queries = [
        "What does SCP stand for?",
        "What is the registration procedure in 3GPP Release 17?",
        "How does AMF work?",
        "What algorithm is used for beamforming optimization?"
    ]

    print("\nQuery Subject Detection Test:")
    print("=" * 50)
    for query in test_queries:
        subjects = classifier.detect_query_subjects(query)
        print(f"Query: {query}")
        print(f"  Expected subjects: {[(s.value, w) for s, w in subjects]}")
        print()
