#!/usr/bin/env python3
"""
Enhanced Query Processor - Advanced query understanding and processing
Improves upon the existing RAG system with better query analysis and routing
"""

import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import existing components
from logging_config import setup_centralized_logging, get_logger, MAJOR, MINOR, ERROR, DEBUG
from cypher_sanitizer import CypherSanitizer


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Single entity, basic question
    MEDIUM = "medium"           # Multiple entities or relationships  
    COMPLEX = "complex"         # Multi-step reasoning required
    EXPERT = "expert"           # Highly technical, domain-specific


class QueryIntent(Enum):
    """Enhanced query intent classification"""
    DEFINITION = "definition"                   # What is X?
    COMPARISON = "comparison"                   # Compare X and Y
    PROCEDURE = "procedure"                     # How does X work?
    RELATIONSHIP = "relationship"               # How do X and Y interact?
    ARCHITECTURE = "architecture"              # System structure questions
    TROUBLESHOOTING = "troubleshooting"        # Problem solving
    SPECIFICATION = "specification"            # Technical details
    PROTOCOL = "protocol"                      # Protocol-specific questions
    PERFORMANCE = "performance"                # Performance/metrics questions
    MULTIPLE_CHOICE = "multiple_choice"        # MCQ format
    GENERAL = "general"                        # Fallback


@dataclass
class EnhancedQueryAnalysis:
    """Enhanced query analysis result"""
    intent: QueryIntent
    complexity: QueryComplexity
    main_entities: List[str]
    secondary_entities: List[str]
    key_terms: List[str]
    technical_terms: List[str]
    confidence_score: float
    requires_cross_document: bool
    suggested_search_strategy: str
    query_variations: List[str]
    sanitized_entities: List[str]


class EnhancedQueryProcessor:
    """Enhanced query processor with advanced analysis capabilities"""
    
    def __init__(self):
        # Initialize logging
        setup_centralized_logging()
        self.logger = get_logger('Enhanced_Query_Processor')
        
        # Load configuration and patterns
        self._load_patterns()
        self._load_technical_terms()
        
        self.logger.log(MAJOR, "Enhanced Query Processor initialized")
    
    def _load_patterns(self):
        """Load query pattern definitions"""
        
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.DEFINITION: {
                'patterns': [
                    r'what\s+is\s+(\w+)',
                    r'define\s+(\w+)',
                    r'explain\s+(\w+)',
                    r'(\w+)\s+stands?\s+for',
                    r'meaning\s+of\s+(\w+)',
                    r'purpose\s+of\s+(\w+)'
                ],
                'keywords': ['define', 'explain', 'meaning', 'purpose', 'stands for', 'what is']
            },
            
            QueryIntent.COMPARISON: {
                'patterns': [
                    r'compare\s+(\w+)\s+and\s+(\w+)',
                    r'difference\s+between\s+(\w+)\s+and\s+(\w+)',
                    r'(\w+)\s+vs\s+(\w+)',
                    r'(\w+)\s+versus\s+(\w+)',
                    r'similarities\s+between\s+(\w+)\s+and\s+(\w+)'
                ],
                'keywords': ['compare', 'difference', 'vs', 'versus', 'similarities', 'contrast']
            },
            
            QueryIntent.PROCEDURE: {
                'patterns': [
                    r'how\s+does\s+(\w+)\s+work',
                    r'steps\s+(for|of)\s+(\w+)',
                    r'procedure\s+(for|of)\s+(\w+)',
                    r'process\s+(for|of)\s+(\w+)',
                    r'how\s+to\s+(\w+)',
                    r'(\w+)\s+flow'
                ],
                'keywords': ['how', 'steps', 'procedure', 'process', 'flow', 'sequence']
            },
            
            QueryIntent.RELATIONSHIP: {
                'patterns': [
                    r'relationship\s+between\s+(\w+)\s+and\s+(\w+)',
                    r'how\s+do\s+(\w+)\s+and\s+(\w+)\s+interact',
                    r'connection\s+between\s+(\w+)\s+and\s+(\w+)',
                    r'(\w+)\s+communicates?\s+with\s+(\w+)',
                    r'interface\s+between\s+(\w+)\s+and\s+(\w+)'
                ],
                'keywords': ['relationship', 'interact', 'connection', 'communicate', 'interface']
            },
            
            QueryIntent.ARCHITECTURE: {
                'patterns': [
                    r'architecture\s+of\s+(\w+)',
                    r'structure\s+of\s+(\w+)',
                    r'components\s+of\s+(\w+)',
                    r'(\w+)\s+architecture',
                    r'design\s+of\s+(\w+)'
                ],
                'keywords': ['architecture', 'structure', 'components', 'design', 'topology']
            },
            
            QueryIntent.TROUBLESHOOTING: {
                'patterns': [
                    r'problems?\s+with\s+(\w+)',
                    r'issues?\s+with\s+(\w+)',
                    r'troubleshoot\s+(\w+)',
                    r'errors?\s+in\s+(\w+)',
                    r'why\s+is\s+(\w+)\s+not\s+working'
                ],
                'keywords': ['problem', 'issue', 'error', 'troubleshoot', 'debug', 'fix']
            },
            
            QueryIntent.SPECIFICATION: {
                'patterns': [
                    r'specification\s+for\s+(\w+)',
                    r'requirements\s+for\s+(\w+)',
                    r'standards?\s+for\s+(\w+)',
                    r'(\w+)\s+specification',
                    r'technical\s+details\s+of\s+(\w+)'
                ],
                'keywords': ['specification', 'requirements', 'standard', 'technical details']
            },
            
            QueryIntent.PROTOCOL: {
                'patterns': [
                    r'(\w+)\s+protocol',
                    r'protocol\s+for\s+(\w+)',
                    r'(\w+)\s+messages?',
                    r'signaling\s+for\s+(\w+)'
                ],
                'keywords': ['protocol', 'signaling', 'messages', 'communication']
            },
            
            QueryIntent.PERFORMANCE: {
                'patterns': [
                    r'performance\s+of\s+(\w+)',
                    r'throughput\s+of\s+(\w+)',
                    r'latency\s+of\s+(\w+)',
                    r'metrics\s+for\s+(\w+)',
                    r'benchmarks?\s+for\s+(\w+)'
                ],
                'keywords': ['performance', 'throughput', 'latency', 'metrics', 'benchmark']
            },
            
            QueryIntent.MULTIPLE_CHOICE: {
                'patterns': [
                    r'which\s+of\s+the\s+following',
                    r'select\s+the\s+correct',
                    r'choose\s+the\s+best',
                    r'A\)\s+.*\s+B\)\s+.*\s+C\)',
                    r'options?:\s*[A-Z]\)'
                ],
                'keywords': ['which of the following', 'select', 'choose', 'option']
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: {
                'entity_count': (1, 2),
                'word_count': (3, 10),
                'keywords': ['what', 'is', 'does', 'basic']
            },
            QueryComplexity.MEDIUM: {
                'entity_count': (2, 4),
                'word_count': (10, 20),
                'keywords': ['compare', 'relationship', 'between', 'how']
            },
            QueryComplexity.COMPLEX: {
                'entity_count': (3, 6),
                'word_count': (15, 40),
                'keywords': ['architecture', 'procedure', 'implementation', 'detailed']
            },
            QueryComplexity.EXPERT: {
                'entity_count': (4, 10),
                'word_count': (25, 100),
                'keywords': ['specification', 'protocol', 'standard', 'technical']
            }
        }
    
    def _load_technical_terms(self):
        """Load 3GPP technical terms and abbreviations"""
        
        # Common 3GPP entities and terms
        self.technical_terms = {
            # Core network functions
            'AMF', 'SMF', 'UPF', 'PCF', 'UDM', 'UDR', 'AUSF', 'NSSF', 'NEF', 'NRF',
            'SCP', 'SEPP', 'BSF', 'CHF', 'NWDAF', 'AF', 'DN',
            
            # Radio access
            'gNB', 'ng-eNB', 'UE', 'RAN', 'CU', 'DU', 'RU',
            
            # Interfaces
            'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10',
            'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N22',
            'Uu', 'F1', 'E1', 'Xn', 'NG',
            
            # Protocols
            'NAS', 'RRC', 'PDCP', 'RLC', 'MAC', 'PHY', 'NGAP', 'HTTP/2',
            'PFCP', 'SBI', 'SCTP', 'GTP-U',
            
            # Services
            'PDU', 'QoS', 'SLICE', 'DNN', 'S-NSSAI', 'PLMN', 'SUPI', 'GPSI',
            
            # Procedures
            'REGISTRATION', 'AUTHENTICATION', 'HANDOVER', 'SESSION',
            'MOBILITY', 'SECURITY', 'SUBSCRIPTION', 'POLICY'
        }
        
        # Service operation patterns
        self.service_patterns = [
            r'N[a-zA-Z]+_\w+_\w+',  # Service operations like Namf_Communication_UEContextTransfer
            r'\w+_\w+_\w+',         # General service patterns
        ]
        
        self.logger.log(MINOR, f"Loaded {len(self.technical_terms)} technical terms")
    
    def analyze_query(self, query: str) -> EnhancedQueryAnalysis:
        """Perform enhanced query analysis"""
        
        self.logger.log(MINOR, f"Analyzing query: {query[:50]}...")
        start_time = time.time()
        
        # Sanitize query first
        sanitized_query = CypherSanitizer.sanitize_search_term(query)
        
        # Extract entities and terms
        entities = self._extract_entities(sanitized_query)
        technical_terms = self._extract_technical_terms(sanitized_query)
        
        # Detect intent
        intent, confidence = self._detect_intent(sanitized_query)
        
        # Assess complexity
        complexity = self._assess_complexity(sanitized_query, entities, technical_terms)
        
        # Generate query variations
        variations = self._generate_query_variations(sanitized_query, entities)
        
        # Determine search strategy
        search_strategy = self._suggest_search_strategy(intent, complexity, len(entities))
        
        # Check if cross-document search is needed
        cross_document = self._requires_cross_document_search(sanitized_query, entities)
        
        # Sanitize entities for safe database queries
        sanitized_entities = [CypherSanitizer.sanitize_search_term(entity) for entity in entities]
        
        analysis_time = time.time() - start_time
        
        result = EnhancedQueryAnalysis(
            intent=intent,
            complexity=complexity,
            main_entities=entities[:3],  # Top 3 most important
            secondary_entities=entities[3:],
            key_terms=self._extract_key_terms(sanitized_query),
            technical_terms=list(technical_terms),
            confidence_score=confidence,
            requires_cross_document=cross_document,
            suggested_search_strategy=search_strategy,
            query_variations=variations,
            sanitized_entities=sanitized_entities
        )
        
        self.logger.log(MAJOR, 
            f"Query analysis complete - intent: {intent.value}, complexity: {complexity.value}, "
            f"entities: {len(entities)}, confidence: {confidence:.2f}, time: {analysis_time:.3f}s"
        )
        
        return result
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        
        entities = []
        query_upper = query.upper()
        
        # Extract technical terms
        for term in self.technical_terms:
            if term in query_upper:
                entities.append(term)
        
        # Extract service operations
        for pattern in self.service_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract capitalized words (likely entities)
        capitalized = re.findall(r'\b[A-Z][a-z]*\b', query)
        entities.extend([word.upper() for word in capitalized if len(word) > 2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities[:10]  # Limit to top 10
    
    def _extract_technical_terms(self, query: str) -> set:
        """Extract technical terms from query"""
        
        query_upper = query.upper()
        found_terms = set()
        
        for term in self.technical_terms:
            if term in query_upper:
                found_terms.add(term)
        
        return found_terms
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        return key_terms[:10]  # Limit to top 10
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect query intent with confidence score"""
        
        query_lower = query.lower()
        best_intent = QueryIntent.GENERAL
        best_score = 0.0
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            
            # Check patterns
            for pattern in config['patterns']:
                if re.search(pattern, query_lower):
                    score += 0.8
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += 0.3
            
            # Normalize score
            max_possible_score = len(config['patterns']) * 0.8 + len(config['keywords']) * 0.3
            if max_possible_score > 0:
                normalized_score = min(score / max_possible_score, 1.0)
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_intent = intent
        
        return best_intent, best_score
    
    def _assess_complexity(self, query: str, entities: List[str], technical_terms: set) -> QueryComplexity:
        """Assess query complexity"""
        
        word_count = len(query.split())
        entity_count = len(entities)
        tech_term_count = len(technical_terms)
        
        # Calculate complexity score
        complexity_score = 0
        
        # Word count contribution
        if word_count > 30:
            complexity_score += 3
        elif word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1
        
        # Entity count contribution
        if entity_count > 5:
            complexity_score += 3
        elif entity_count > 3:
            complexity_score += 2
        elif entity_count > 1:
            complexity_score += 1
        
        # Technical term contribution
        complexity_score += min(tech_term_count, 3)
        
        # Check for complexity keywords
        complexity_keywords = {
            'architecture': 2,
            'implementation': 2,
            'specification': 3,
            'protocol': 2,
            'procedure': 1,
            'detailed': 2,
            'technical': 1,
            'standard': 2
        }
        
        for keyword, weight in complexity_keywords.items():
            if keyword in query.lower():
                complexity_score += weight
        
        # Map score to complexity level
        if complexity_score >= 10:
            return QueryComplexity.EXPERT
        elif complexity_score >= 6:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 3:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE
    
    def _generate_query_variations(self, query: str, entities: List[str]) -> List[str]:
        """Generate query variations for improved search"""
        
        variations = [query]  # Original query
        
        # Entity-focused variations
        for entity in entities[:3]:  # Top 3 entities
            variations.append(f"What is {entity}?")
            variations.append(f"{entity} definition")
            variations.append(f"{entity} function purpose")
        
        # Keyword-based variations
        if 'compare' in query.lower() and len(entities) >= 2:
            variations.append(f"difference between {entities[0]} and {entities[1]}")
        
        if 'how' in query.lower():
            for entity in entities[:2]:
                variations.append(f"{entity} procedure process")
        
        # Remove duplicates and limit
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations[:5]  # Limit to 5 variations
    
    def _suggest_search_strategy(self, intent: QueryIntent, complexity: QueryComplexity, entity_count: int) -> str:
        """Suggest optimal search strategy"""
        
        # Strategy decision matrix
        if complexity in [QueryComplexity.EXPERT, QueryComplexity.COMPLEX]:
            if entity_count > 3:
                return "hybrid_expanded"  # Use both vector and graph with expansion
            else:
                return "hybrid_focused"   # Hybrid with focused search
        
        elif intent in [QueryIntent.COMPARISON, QueryIntent.RELATIONSHIP]:
            return "graph_preferred"      # Graph search better for relationships
        
        elif intent in [QueryIntent.DEFINITION, QueryIntent.GENERAL]:
            if entity_count == 1:
                return "vector_preferred" # Vector search good for single concepts
            else:
                return "hybrid_balanced"  # Balanced hybrid approach
        
        elif intent in [QueryIntent.PROCEDURE, QueryIntent.PROTOCOL]:
            return "graph_sequential"     # Graph search for sequential information
        
        else:
            return "auto"                 # Let system decide
    
    def _requires_cross_document_search(self, query: str, entities: List[str]) -> bool:
        """Determine if cross-document search is needed"""
        
        # Indicators that suggest cross-document search
        cross_doc_indicators = [
            'compare', 'relationship', 'interaction', 'interface',
            'architecture', 'system', 'overview', 'multiple'
        ]
        
        query_lower = query.lower()
        has_indicators = any(indicator in query_lower for indicator in cross_doc_indicators)
        
        # Multiple entities often require cross-document search
        multiple_entities = len(entities) > 2
        
        return has_indicators or multiple_entities
    
    def create_optimized_query_params(self, analysis: EnhancedQueryAnalysis) -> Dict[str, Any]:
        """Create optimized query parameters based on analysis"""
        
        params = {
            'use_hybrid': True,
            'use_vector': True,
            'use_graph': True,
            'expand_query': True,
            'use_llm_analysis': False,  # Enhanced processor replaces basic analysis
            'max_chunks': 30 if analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT] else 20,
            'vector_weight': 0.6,
            'graph_weight': 0.4
        }
        
        # Adjust based on search strategy
        strategy = analysis.suggested_search_strategy
        
        if strategy == "vector_preferred":
            params.update({
                'use_graph': False,
                'vector_weight': 1.0,
                'max_chunks': 25
            })
        elif strategy == "graph_preferred":
            params.update({
                'use_vector': False,
                'graph_weight': 1.0,
                'max_chunks': 25
            })
        elif strategy == "hybrid_expanded":
            params.update({
                'max_chunks': 40,
                'expand_query': True,
                'vector_weight': 0.5,
                'graph_weight': 0.5
            })
        elif strategy == "hybrid_focused":
            params.update({
                'max_chunks': 20,
                'expand_query': False,
                'vector_weight': 0.7,
                'graph_weight': 0.3
            })
        
        # Adjust based on intent
        if analysis.intent == QueryIntent.DEFINITION:
            params['vector_weight'] = 0.8
            params['graph_weight'] = 0.2
        elif analysis.intent == QueryIntent.RELATIONSHIP:
            params['vector_weight'] = 0.3
            params['graph_weight'] = 0.7
        elif analysis.intent == QueryIntent.COMPARISON:
            params['max_chunks'] = 35
            params['expand_query'] = True
        
        return params
    
    def explain_analysis(self, analysis: EnhancedQueryAnalysis) -> str:
        """Generate human-readable explanation of analysis"""
        
        explanation = f"""Enhanced Query Analysis:
        
Intent: {analysis.intent.value.title().replace('_', ' ')}
Complexity: {analysis.complexity.value.title()}
Confidence: {analysis.confidence_score:.1%}

Entities Found: {', '.join(analysis.main_entities)}
Technical Terms: {', '.join(analysis.technical_terms)}
Search Strategy: {analysis.suggested_search_strategy}

The query appears to be a {analysis.complexity.value} {analysis.intent.value.replace('_', ' ')} question.
"""
        
        if analysis.requires_cross_document:
            explanation += "Cross-document search recommended for comprehensive results.\n"
        
        if analysis.confidence_score < 0.5:
            explanation += "Note: Low confidence in intent detection - using general search approach.\n"
        
        return explanation


# Integration function for existing RAG system
def enhance_rag_with_processor(rag_system, processor: EnhancedQueryProcessor):
    """Enhance existing RAG system with the enhanced processor"""
    
    original_query = rag_system.query
    
    def enhanced_query(question: str, **kwargs):
        """Enhanced query method that uses the advanced processor"""
        
        # Analyze query with enhanced processor
        analysis = processor.analyze_query(question)
        
        # Create optimized parameters
        optimized_params = processor.create_optimized_query_params(analysis)
        
        # Merge with user-provided parameters (user params take precedence)
        final_params = {**optimized_params, **kwargs}
        
        # Add analysis to the response
        response = original_query(question, **final_params)
        
        # Enhance response with analysis information
        if hasattr(response, '__dict__'):
            response.enhanced_analysis = analysis
            response.optimization_applied = True
            response.search_strategy_used = analysis.suggested_search_strategy
        
        return response
    
    # Replace the query method
    rag_system.query = enhanced_query
    rag_system.enhanced_processor = processor
    
    return rag_system