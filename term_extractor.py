"""
Term Extractor for 3GPP Documents.
Extracts abbreviations and definitions from 3GPP specification chunks.
"""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from logging_config import get_logger, MAJOR, MINOR, DEBUG

logger = get_logger('Term_Extractor')


@dataclass
class ExtractedTerm:
    """Represents an extracted term (abbreviation or definition)"""
    abbreviation: str
    full_name: str
    term_type: str  # 'abbreviation' or 'definition'
    source_spec: str


class TermExtractor:
    """
    Extract abbreviations and definitions from 3GPP JSON chunks.

    Handles formats like:
    - Abbreviations: "SCP\tService Communication Proxy"
    - Definitions: "NF service: A functionality exposed by an NF..."
    """

    def __init__(self):
        self.logger = get_logger('Term_Extractor')

        # Pattern for abbreviations: ABBR<tab or spaces>Full Name
        # Matches: "SCP\tService Communication Proxy" or "5GC    5G Core Network"
        self.abbr_pattern = re.compile(
            r'^([A-Z0-9][-A-Za-z0-9/]*(?:\s+[A-Z0-9][-A-Za-z0-9/]*)*)\s*\t+\s*(.+)$',
            re.MULTILINE
        )

        # Alternative pattern for space-separated (when tabs aren't used)
        self.abbr_space_pattern = re.compile(
            r'^([A-Z][A-Z0-9/-]{1,15})\s{2,}([A-Z][A-Za-z0-9\s\-/()]+)$',
            re.MULTILINE
        )

        # Pattern for definitions: "Term: definition text" or "Term - definition text"
        self.def_pattern = re.compile(
            r'^([A-Za-z][A-Za-z0-9\s\-]+?):\s+(.+?)(?=\n[A-Z]|\n\n|\Z)',
            re.MULTILINE | re.DOTALL
        )

    def extract_abbreviations(self, content: str, spec_id: str = "") -> List[ExtractedTerm]:
        """
        Extract abbreviations from chunk content.

        Args:
            content: The text content of an abbreviation section
            spec_id: The specification ID (e.g., "ts_23.501")

        Returns:
            List of ExtractedTerm objects
        """
        terms = []

        # Skip the introductory text
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip introductory sentences
            if line.lower().startswith('for the purposes') or 'apply' in line.lower()[:50]:
                continue
            if 'tr 21.905' in line.lower() or 'precedence' in line.lower():
                continue

            # Try tab-separated format first
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    abbr = parts[0].strip()
                    full_name = parts[1].strip()

                    # Validate abbreviation
                    if self._is_valid_abbreviation(abbr) and full_name:
                        terms.append(ExtractedTerm(
                            abbreviation=abbr,
                            full_name=full_name,
                            term_type='abbreviation',
                            source_spec=spec_id
                        ))
                        continue

            # Try space-separated format
            match = self.abbr_space_pattern.match(line)
            if match:
                abbr = match.group(1).strip()
                full_name = match.group(2).strip()

                if self._is_valid_abbreviation(abbr) and full_name:
                    terms.append(ExtractedTerm(
                        abbreviation=abbr,
                        full_name=full_name,
                        term_type='abbreviation',
                        source_spec=spec_id
                    ))

        self.logger.log(MINOR, f"Extracted {len(terms)} abbreviations from {spec_id}")
        return terms

    def _is_valid_abbreviation(self, abbr: str) -> bool:
        """Check if string is a valid abbreviation"""
        if not abbr or len(abbr) < 2:
            return False

        # Must start with uppercase letter or digit
        if not (abbr[0].isupper() or abbr[0].isdigit()):
            return False

        # Should contain mostly uppercase letters, digits, or hyphens
        valid_chars = sum(1 for c in abbr if c.isupper() or c.isdigit() or c in '-/')
        return valid_chars >= len(abbr) * 0.6

    def extract_definitions(self, content: str, spec_id: str = "") -> List[ExtractedTerm]:
        """
        Extract definitions from chunk content.

        Args:
            content: The text content of a definitions section
            spec_id: The specification ID (e.g., "ts_23.501")

        Returns:
            List of ExtractedTerm objects with term and definition
        """
        terms = []

        # Skip introductory text and find definition entries
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip introductory sentences
            if line.lower().startswith('for the purposes') or 'apply' in line.lower()[:50]:
                continue
            if 'tr 21.905' in line.lower() or 'ts 23.501' in line.lower():
                continue

            # Look for "term: definition" pattern
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    term = parts[0].strip()
                    definition = parts[1].strip()

                    # Validate term (should be a readable term, not just an abbreviation)
                    if self._is_valid_definition_term(term) and len(definition) > 10:
                        terms.append(ExtractedTerm(
                            abbreviation=term,  # Using abbreviation field for the term
                            full_name=definition,  # Using full_name for the definition
                            term_type='definition',
                            source_spec=spec_id
                        ))

        self.logger.log(MINOR, f"Extracted {len(terms)} definitions from {spec_id}")
        return terms

    def _is_valid_definition_term(self, term: str) -> bool:
        """Check if string is a valid definition term"""
        if not term or len(term) < 2 or len(term) > 100:
            return False

        # Should contain letters
        has_letters = any(c.isalpha() for c in term)

        # Should not be just a number reference like "[1]"
        is_reference = term.startswith('[') and term.endswith(']')

        return has_letters and not is_reference

    def extract_all_from_chunk(self, chunk_content: str, section_title: str,
                               spec_id: str = "") -> List[ExtractedTerm]:
        """
        Extract all terms from a chunk based on its section title.

        Args:
            chunk_content: The text content of the chunk
            section_title: The section title (e.g., "Abbreviations", "Definitions")
            spec_id: The specification ID

        Returns:
            List of ExtractedTerm objects
        """
        section_lower = section_title.lower()

        if 'abbreviation' in section_lower:
            return self.extract_abbreviations(chunk_content, spec_id)
        elif 'definition' in section_lower:
            return self.extract_definitions(chunk_content, spec_id)

        return []


def extract_terms_from_json_files(json_dir: str) -> Dict[str, List[ExtractedTerm]]:
    """
    Extract all terms from processed JSON files.

    Args:
        json_dir: Path to directory containing processed JSON files

    Returns:
        Dictionary mapping spec_id to list of ExtractedTerm
    """
    import json
    from pathlib import Path

    extractor = TermExtractor()
    all_terms = {}

    json_path = Path(json_dir)
    for json_file in json_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            spec_id = data["metadata"]["specification_id"]
            terms = []

            for chunk in data.get("chunks", []):
                section_title = chunk.get("section_title", "")
                content = chunk.get("content", "")

                chunk_terms = extractor.extract_all_from_chunk(
                    content, section_title, spec_id
                )
                terms.extend(chunk_terms)

            if terms:
                all_terms[spec_id] = terms

        except Exception as e:
            logger.log(MAJOR, f"Error processing {json_file}: {e}")

    return all_terms


def build_term_dictionary(all_terms: Dict[str, List[ExtractedTerm]]) -> Dict[str, Dict]:
    """
    Build a consolidated term dictionary from extracted terms.
    Merges terms that appear in multiple specs.

    Args:
        all_terms: Dictionary from extract_terms_from_json_files()

    Returns:
        Dictionary mapping abbreviation to {full_name, source_specs, term_type}
    """
    term_dict = {}

    for spec_id, terms in all_terms.items():
        for term in terms:
            abbr = term.abbreviation

            if abbr not in term_dict:
                term_dict[abbr] = {
                    'abbreviation': abbr,
                    'full_name': term.full_name,
                    'term_type': term.term_type,
                    'source_specs': [spec_id],
                    'primary_spec': spec_id
                }
            else:
                # Add this spec as an additional source
                if spec_id not in term_dict[abbr]['source_specs']:
                    term_dict[abbr]['source_specs'].append(spec_id)

    return term_dict


if __name__ == "__main__":
    # Test the extractor
    print("Term Extractor for 3GPP Documents")
    print("=" * 50)

    # Test abbreviation extraction
    test_content = """For the purposes of the present document, the abbreviations given in TR 21.905 [1] and the following apply.
SCP	Service Communication Proxy
SEPP	Security Edge Protection Proxy
SMF	Session Management Function
UPF	User Plane Function
5GC	5G Core Network
AMF	Access and Mobility Management Function
"""

    extractor = TermExtractor()
    terms = extractor.extract_abbreviations(test_content, "ts_test")

    print(f"\nExtracted {len(terms)} abbreviations:")
    for term in terms:
        print(f"  {term.abbreviation}: {term.full_name}")

    print("\nTerm Extractor loaded successfully!")
