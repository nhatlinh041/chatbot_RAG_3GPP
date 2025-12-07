#!/usr/bin/env python3
"""
Cypher Query Sanitization Module
Provides input sanitization to prevent Cypher injection attacks
"""

import re
from typing import List, Dict, Any


class CypherSanitizer:
    """Sanitizes user inputs for Cypher queries to prevent injection attacks"""
    
    # Dangerous patterns that should be blocked
    # NOTE: Cypher comments (//) are allowed since they're valid Neo4j syntax
    # and don't pose a security risk in read-only queries
    DANGEROUS_PATTERNS = [
        r'MATCH\s*\(\s*\)\s*DETACH\s*DELETE',  # DELETE operations
        r'DELETE',                              # Any DELETE
        r'CREATE\s+CONSTRAINT',                 # Schema modifications
        r'DROP\s+CONSTRAINT',
        r'CREATE\s+INDEX',
        r'DROP\s+INDEX',
        r'MERGE.*DELETE',                       # MERGE with DELETE
        r'SET.*=.*\$',                         # Parameter assignment
        r'REMOVE\s+',                          # REMOVE operations
        r'CALL\s+DB\.',                        # Database procedures
        r'CALL\s+DBMS\.',                      # DBMS procedures
        r'LOAD\s+CSV',                         # File operations
        r'USING\s+PERIODIC\s+COMMIT',          # Batch operations
        # Removed: r'//.*' - Cypher line comments are valid syntax
        r'/\*.*\*/',                           # Multi-line comments (may hide malicious code)
    ]
    
    # Characters that should be escaped or removed
    DANGEROUS_CHARS = [';', '\x00', '\x1a', '\x08', '\x09', '\x0a', '\x0d']
    
    @staticmethod
    def sanitize_string(input_string: str) -> str:
        """
        Sanitize a string for safe use in Cypher queries
        """
        if not input_string:
            return ""
        
        # Remove null bytes and control characters
        sanitized = input_string.replace('\x00', '')
        
        # Remove or escape dangerous characters
        for char in CypherSanitizer.DANGEROUS_CHARS:
            sanitized = sanitized.replace(char, '')
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit length to prevent buffer overflow attempts
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_search_term(term: str) -> str:
        """
        Sanitize search terms for CONTAINS operations
        """
        if not term:
            return ""
        
        # Basic sanitization
        sanitized = CypherSanitizer.sanitize_string(term)
        
        # Remove quotes and escape characters
        sanitized = sanitized.replace("'", "").replace('"', "").replace('\\', '')
        
        # Remove Cypher keywords that shouldn't be in search terms
        cypher_keywords = ['MATCH', 'WHERE', 'RETURN', 'DELETE', 'CREATE', 'SET', 'REMOVE']
        for keyword in cypher_keywords:
            sanitized = re.sub(rf'\b{keyword}\b', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_query_safety(query: str) -> bool:
        """
        Validate that a Cypher query doesn't contain dangerous patterns
        Returns True if safe, False if potentially dangerous
        """
        if not query:
            return True
        
        query_upper = query.upper()
        
        # Check for dangerous patterns
        for pattern in CypherSanitizer.DANGEROUS_PATTERNS:
            if re.search(pattern, query_upper):
                return False
        
        # Additional checks
        if query_upper.count('DELETE') > 0:
            return False
        
        if query_upper.count('CREATE') > 1:  # Allow one CREATE for relationships
            return False
        
        return True
    
    @staticmethod
    def sanitize_question_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize question analysis data
        """
        sanitized = {}
        
        # Sanitize entities
        if 'entities' in analysis:
            sanitized['entities'] = []
            for entity in analysis['entities']:
                if isinstance(entity, dict) and 'value' in entity:
                    sanitized_value = CypherSanitizer.sanitize_search_term(entity['value'])
                    if sanitized_value:  # Only add if not empty after sanitization
                        sanitized['entities'].append({
                            'value': sanitized_value,
                            'type': entity.get('type', 'unknown')
                        })
        
        # Sanitize key terms
        if 'key_terms' in analysis:
            sanitized['key_terms'] = []
            for term in analysis['key_terms']:
                sanitized_term = CypherSanitizer.sanitize_search_term(str(term))
                if sanitized_term:
                    sanitized['key_terms'].append(sanitized_term)
        
        # Copy safe fields
        safe_fields = ['question_type', 'complexity']
        for field in safe_fields:
            if field in analysis:
                sanitized[field] = analysis[field]
        
        return sanitized


def create_safe_cypher_query(template: str, **params) -> str:
    """
    Create a safe Cypher query by sanitizing all parameters
    """
    sanitized_params = {}
    
    for key, value in params.items():
        if isinstance(value, str):
            sanitized_params[key] = CypherSanitizer.sanitize_search_term(value)
        elif isinstance(value, (int, float)):
            sanitized_params[key] = value
        elif isinstance(value, list):
            sanitized_params[key] = [
                CypherSanitizer.sanitize_search_term(str(item)) if isinstance(item, str) else item
                for item in value
            ]
        else:
            sanitized_params[key] = str(value)
    
    try:
        query = template.format(**sanitized_params)
        
        # Final safety check
        if not CypherSanitizer.validate_query_safety(query):
            raise ValueError("Query contains potentially dangerous patterns")
        
        return query
    except Exception as e:
        raise ValueError(f"Failed to create safe query: {e}")


# Example usage for RAG system integration
def safe_contains_query(search_terms: List[str], limit: int = 5) -> str:
    """
    Create a safe CONTAINS query for multiple search terms
    """
    if not search_terms:
        raise ValueError("No search terms provided")
    
    # Sanitize all search terms
    safe_terms = [CypherSanitizer.sanitize_search_term(term) for term in search_terms]
    safe_terms = [term for term in safe_terms if term]  # Remove empty terms
    
    if not safe_terms:
        raise ValueError("No valid search terms after sanitization")
    
    # Limit the number of terms to prevent query complexity issues
    safe_terms = safe_terms[:5]
    
    # Validate limit parameter
    if not isinstance(limit, int) or limit <= 0 or limit > 50:
        limit = 5
    
    # Build safe query
    conditions = []
    for term in safe_terms:
        conditions.append(f"toLower(c.content) CONTAINS toLower('{term}')")
    
    query = f"""
    MATCH (c:Chunk)
    WHERE {' OR '.join(conditions)}
    RETURN c.chunk_id, c.spec_id, c.section_title, c.content, c.complexity_score
    ORDER BY c.complexity_score DESC
    LIMIT {limit}
    """
    
    return query