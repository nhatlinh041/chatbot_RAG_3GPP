"""
Hybrid term extraction module for 3GPP/5G question analysis
Combines 3GPP term dictionary with LLM question analysis
"""

import json
import re
import os
import glob
from typing import List, Dict, Set
from dataclasses import dataclass

try:
    from fuzzywuzzy import fuzz
except ImportError:
    class SimpleFuzz:
        @staticmethod
        def ratio(a, b):
            return 100 if a.lower() == b.lower() else 0
    fuzz = SimpleFuzz()


@dataclass
class TechnicalTerm:
    name: str
    spec_id: str
    section: str
    synonyms: List[str]
    category: str


class TermDictionary:
    def __init__(self):
        self.terms: Dict[str, TechnicalTerm] = {}
        self.synonym_map: Dict[str, str] = {}

    def add_term(self, term: TechnicalTerm):
        """Add a technical term to the dictionary"""
        self.terms[term.name.lower()] = term
        for synonym in term.synonyms:
            self.synonym_map[synonym.lower()] = term.name.lower()

    def find_matches(self, query_terms: List[str], threshold: int = 80) -> List[str]:
        """Find matching terms using fuzzy matching"""
        matches = set()

        for query_term in query_terms:
            query_lower = query_term.lower()

            # Exact match
            if query_lower in self.terms:
                matches.add(query_lower)
                continue

            # Synonym match
            if query_lower in self.synonym_map:
                matches.add(self.synonym_map[query_lower])
                continue

            # Fuzzy match
            for term_name in self.terms.keys():
                if fuzz.ratio(query_lower, term_name) >= threshold:
                    matches.add(term_name)

        return list(matches)


class LLMQuestionAnalyzer:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def extract_terms(self, question: str) -> List[str]:
        """Extract technical terms from question using local LLM"""
        # Simulate LLM call (replace with actual local LLM)
        response = self._simulate_llm_call(question)

        # Parse response
        terms = [term.strip() for term in response.split(',')]
        return [term for term in terms if term]

    def _simulate_llm_call(self, question: str) -> str:
        """Simulate LLM response for testing"""
        # Simple pattern matching for demo
        patterns = {
            r'chf|charging function': 'CHF, Charging Function, Service Based Interface',
            r'amf|access management': 'AMF, Access and Mobility Management Function, 5G Core',
            r'smf|session management': 'SMF, Session Management Function, PDU Session',
            r'upf|user plane': 'UPF, User Plane Function, N4 Interface'
        }

        for pattern, response in patterns.items():
            if re.search(pattern, question.lower()):
                return response

        return "5G, Network Function, Service Based Interface"


class HybridTermExtractor:
    def __init__(self):
        self.term_dict = TermDictionary()
        self.llm_analyzer = LLMQuestionAnalyzer()
        self._build_3gpp_dictionary()

    def _build_3gpp_dictionary(self):
        """Build 3GPP term dictionary from processed JSON files"""

        json_dir = "/home/linguyen/3GPP/3GPP_JSON_DOC/processed_json_v2"

        if os.path.exists(json_dir):
            json_files = glob.glob(os.path.join(json_dir, "*.json"))

            for json_file in json_files[:10]:  # Limit for testing
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    spec_id = os.path.basename(json_file).replace('.json', '')

                    # Extract terms from Definitions sections
                    for chunk in data.get('chunks', []):
                        if chunk.get('section_title') == 'Definitions':
                            self._extract_definitions_from_chunk(chunk, spec_id)

                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
        else:
            # Fallback to sample terms if directory doesn't exist
            self._build_sample_dictionary()

    def _extract_definitions_from_chunk(self, chunk, spec_id):
        """Extract individual term definitions from a Definitions chunk"""
        content = chunk.get('content', '')
        section_id = chunk.get('section_id', '')

        # Pattern to match term definitions like "Binding indication (consumer):"
        pattern = r'([A-Z][^:]+?):\s*([^.]+(?:\.[^A-Z][^.]*)*\.)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        for term_name, definition in matches:
            term_name = term_name.strip()
            definition = definition.strip()

            # Extract synonyms and acronyms from the definition
            synonyms = self._extract_synonyms(term_name, definition)

            # Determine category based on content
            category = self._categorize_term(term_name, definition)

            technical_term = TechnicalTerm(
                name=term_name.lower(),
                spec_id=spec_id,
                section=section_id,
                synonyms=synonyms,
                category=category
            )

            self.term_dict.add_term(technical_term)

    def _extract_synonyms(self, term_name, definition):
        """Extract synonyms and acronyms from definition text"""
        synonyms = []

        # Look for acronyms in parentheses
        acronym_pattern = r'\(([A-Z]{2,})\)'
        acronyms = re.findall(acronym_pattern, definition)
        synonyms.extend([acr.lower() for acr in acronyms])

        # Look for "also known as" or similar phrases
        aka_pattern = r'(?:also known as|referred to as|called)\s+([^.,;]+)'
        aka_matches = re.findall(aka_pattern, definition, re.IGNORECASE)
        synonyms.extend([match.strip().lower() for match in aka_matches])

        # Add variations of the term name
        if '(' in term_name:
            base_term = term_name.split('(')[0].strip().lower()
            synonyms.append(base_term)

        return list(set(synonyms))

    def _categorize_term(self, term_name, definition):
        """Categorize term based on content"""
        term_lower = term_name.lower()
        def_lower = definition.lower()

        if any(word in term_lower for word in ['function', 'nf']):
            return 'network_function'
        elif any(word in term_lower for word in ['interface', 'endpoint']):
            return 'interface'
        elif any(word in def_lower for word in ['protocol', 'message']):
            return 'protocol'
        elif any(word in def_lower for word in ['charging', 'billing']):
            return 'charging'
        else:
            return 'general'

    def _build_sample_dictionary(self):
        """Fallback to sample terms if JSON files not available"""
        sample_terms = [
            TechnicalTerm("chf", "TS 32.240", "5.2.1",
                         ["charging function", "charging system"], "charging"),
            TechnicalTerm("amf", "TS 23.501", "4.2.2",
                         ["access and mobility management function"], "core"),
            TechnicalTerm("smf", "TS 23.501", "4.2.3",
                         ["session management function"], "core"),
            TechnicalTerm("service based interface", "TS 23.501", "4.1",
                         ["sbi", "service based architecture"], "interface"),
            TechnicalTerm("diameter", "TS 29.212", "4.1",
                         ["diameter protocol", "ro interface", "rf interface"], "protocol")
        ]

        for term in sample_terms:
            self.term_dict.add_term(term)

    def enhance_query(self, question: str) -> Dict:
        """Main method to enhance query with extracted terms"""
        # Step 1: Extract terms from question using LLM
        llm_terms = self.llm_analyzer.extract_terms(question)

        # Step 2: Match to 3GPP dictionary
        matched_terms = self.term_dict.find_matches(llm_terms)

        # Step 3: Generate enhanced search terms
        all_terms = set(matched_terms)

        # Add synonyms for matched terms
        for term in matched_terms:
            if term in self.term_dict.terms:
                all_terms.update(self.term_dict.terms[term].synonyms)

        return {
            'original_question': question,
            'llm_terms': llm_terms,
            'matched_terms': matched_terms,
            'enhanced_terms': list(all_terms),
            'cypher_query': self._generate_cypher_query(list(all_terms))
        }

    def _generate_cypher_query(self, terms: List[str]) -> str:
        """Generate enhanced Cypher query"""
        terms_str = str(terms).replace("'", '"')

        query = f"""
        WITH {terms_str} as search_terms

        MATCH (c:Chunk)
        WHERE any(term IN search_terms WHERE toLower(c.content) CONTAINS toLower(term))

        WITH c,
             size([term IN search_terms WHERE toLower(c.content) CONTAINS toLower(term)]) as relevance_score

        WHERE relevance_score > 0

        RETURN c.chunk_id, c.spec_id, c.section_title, c.content,
               relevance_score, c.complexity_score
        ORDER BY relevance_score DESC, c.complexity_score ASC
        LIMIT 10
        """

        return query.strip()
