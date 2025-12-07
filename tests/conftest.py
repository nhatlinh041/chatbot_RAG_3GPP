"""
Pytest configuration and shared fixtures for 3GPP project tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_question():
    """Sample 3GPP question for testing"""
    return "What is the main purpose of the AMF in 5G networks?"


@pytest.fixture
def sample_entities():
    """Sample extracted entities"""
    return [
        {'value': 'AMF', 'type': 'network_function'},
        {'value': '5G', 'type': 'technology'}
    ]


@pytest.fixture
def sample_analysis():
    """Sample question analysis result"""
    return {
        'question_type': 'definition',
        'entities': [{'value': 'AMF', 'type': 'network_function'}],
        'key_terms': ['AMF', 'purpose', '5G', 'networks'],
        'complexity': 'medium'
    }


@pytest.fixture
def malicious_inputs():
    """Collection of potentially malicious inputs for security testing"""
    return [
        "'; DELETE (n) --",
        "MATCH (n) DETACH DELETE n",
        "AMF' OR 1=1 --",
        "test\x00null",
        "DROP INDEX test",
        "CALL db.labels()",
        "LOAD CSV FROM 'file:///etc/passwd'",
        "'; CREATE (n:Malicious) --",
    ]


@pytest.fixture
def mock_neo4j_config():
    """Mock Neo4j configuration for testing"""
    return {
        'uri': 'neo4j://localhost:7687',
        'user': 'neo4j',
        'password': 'password'
    }
