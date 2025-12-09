#!/usr/bin/env python3
"""
3GPP Document Download and Processing Pipeline

Downloads 3GPP Release 18 specifications, processes them to JSON,
and initializes the Neo4j Knowledge Graph.

Usage:
    python download_and_process_3gpp.py download     # Download documents only
    python download_and_process_3gpp.py extract      # Extract ZIP files
    python download_and_process_3gpp.py process      # Process downloaded docs to JSON
    python download_and_process_3gpp.py init-kg      # Initialize Neo4j from JSON
    python download_and_process_3gpp.py all          # Do everything
    python download_and_process_3gpp.py status       # Check current status
"""

import os
import sys
import re
import json
import zipfile
import subprocess
import argparse
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    colors = {
        "ok": Colors.GREEN,
        "error": Colors.RED,
        "warn": Colors.YELLOW,
        "info": Colors.BLUE,
        "progress": Colors.CYAN
    }
    symbols = {
        "ok": "✓",
        "error": "✗",
        "warn": "!",
        "info": "→",
        "progress": "◐"
    }
    color = colors.get(status, Colors.RESET)
    symbol = symbols.get(status, "→")
    print(f"{color}{symbol}{Colors.RESET} {message}")


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    # Directories
    download_dir: Path = None
    data_dir: Path = None
    output_dir: Path = None

    # Download settings
    release: int = 18
    series: List[str] = None  # None = all series, or ['23', '29', '38'] for specific

    # Neo4j settings
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    def __post_init__(self):
        # Set default directories
        if self.download_dir is None:
            self.download_dir = SCRIPT_DIR / "data" / f"rel{self.release}_download"
        if self.data_dir is None:
            self.data_dir = SCRIPT_DIR / "data" / f"rel{self.release}_extracted"
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "3GPP_JSON_DOC" / "processed_json_v3"

        if self.series is None:
            # 5G Core series: 23.xxx (System Architecture), 29.xxx (APIs),
            # 32.xxx (Charging), 38.xxx (Radio)
            self.series = ['23', '24', '26', '27', '28', '29', '32', '33', '36', '37', '38']


# =============================================================================
# Downloader: Download 3GPP documents
# =============================================================================

class ThreeGPPDownloader:
    """Download 3GPP specifications using download_3gpp package"""

    def __init__(self, config: Config):
        self.config = config
        self.download_dir = config.download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def check_download_3gpp_installed(self) -> bool:
        """Check if download_3gpp package is installed"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import download_3gpp"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def install_download_3gpp(self) -> bool:
        """Install download_3gpp package"""
        print_status("Installing download_3gpp package...", "info")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "download_3gpp"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print_status("download_3gpp installed successfully", "ok")
                return True
            else:
                print_status(f"Failed to install: {result.stderr}", "error")
                return False
        except Exception as e:
            print_status(f"Error installing package: {e}", "error")
            return False

    def download_release(self, release: int = None, series: List[str] = None) -> bool:
        """
        Download 3GPP release using download_3gpp package.

        Args:
            release: Release number (e.g., 18 for Rel-18)
            series: List of series to download (e.g., ['23', '29'])
        """
        release = release or self.config.release
        series = series or self.config.series

        print_header(f"Downloading 3GPP Release {release}")

        # Check/install download_3gpp
        if not self.check_download_3gpp_installed():
            if not self.install_download_3gpp():
                print_status("Failed to install download_3gpp, trying alternative method...", "warn")
                return self.download_via_requests(release, series)

        # Create release directory
        release_dir = self.download_dir / f"Rel-{release}"
        release_dir.mkdir(parents=True, exist_ok=True)

        try:
            if series:
                # Download specific series
                for s in series:
                    print_status(f"Downloading series {s}...", "progress")

                    # Use download_3gpp with series filter
                    cmd = ["download_3gpp", f"--rel={release}", f"--series={s}"]

                    result = subprocess.run(
                        cmd,
                        cwd=str(release_dir),
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout per series
                    )

                    if result.returncode != 0 and result.stderr:
                        print_status(f"Warning: Series {s}: {result.stderr[:200]}", "warn")
                    else:
                        print_status(f"Series {s} downloaded", "ok")
            else:
                # Download all
                print_status(f"Downloading all series for Release {release}...", "progress")
                print_status("This may take a long time (several GB of data)...", "info")

                cmd = ["download_3gpp", f"--rel={release}"]

                result = subprocess.run(
                    cmd,
                    cwd=str(release_dir),
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )

                if result.returncode != 0 and result.stderr:
                    print_status(f"Warning: {result.stderr[:500]}", "warn")

            # Count downloaded files
            zip_files = list(release_dir.rglob("*.zip"))
            docx_files = list(release_dir.rglob("*.docx"))

            print_status(f"Downloaded {len(zip_files)} ZIP files, {len(docx_files)} DOCX files", "ok")
            return len(zip_files) > 0 or len(docx_files) > 0

        except subprocess.TimeoutExpired:
            print_status("Download timed out. Try downloading specific series.", "error")
            return False
        except Exception as e:
            print_status(f"Download error: {e}", "error")
            return False

    def download_via_requests(self, release: int = 18, series: List[str] = None) -> bool:
        """
        Alternative download method using direct HTTP requests.
        Fallback if download_3gpp doesn't work.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print_status("requests/beautifulsoup4 not installed", "error")
            return False

        print_header(f"Downloading 3GPP Release {release} (Direct HTTP)")

        # 3GPP FTP archive URL
        FTP_BASE = "https://www.3gpp.org/ftp/Specs/archive"

        release_dir = self.download_dir / f"Rel-{release}"
        release_dir.mkdir(parents=True, exist_ok=True)

        series_to_download = series or self.config.series

        downloaded = 0
        failed = 0

        # Release letter mapping: 18 -> 'i', 17 -> 'h', etc.
        release_letter = chr(ord('a') + release - 10)  # Rel-10='a', Rel-18='i'

        for s in series_to_download:
            series_url = f"{FTP_BASE}/{s}_series/"
            series_dir = release_dir / f"{s}_series"
            series_dir.mkdir(exist_ok=True)

            print_status(f"Fetching series {s} from {series_url}...", "progress")

            try:
                response = requests.get(series_url, timeout=30)
                if response.status_code != 200:
                    print_status(f"Cannot access series {s} (HTTP {response.status_code})", "warn")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find ZIP files for this release
                for link in soup.find_all('a'):
                    href = link.get('href', '')

                    # Match pattern: 23501-i80.zip (release letter + version)
                    if href.endswith('.zip') and f'-{release_letter}' in href.lower():
                        file_url = f"{series_url}{href}"
                        file_path = series_dir / href

                        if file_path.exists():
                            print_status(f"  Skipping {href} (exists)", "info")
                            continue

                        try:
                            print_status(f"  Downloading {href}...", "info")
                            file_response = requests.get(file_url, timeout=120, stream=True)

                            if file_response.status_code == 200:
                                with open(file_path, 'wb') as f:
                                    for chunk in file_response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                downloaded += 1
                            else:
                                print_status(f"  Failed: HTTP {file_response.status_code}", "warn")
                                failed += 1

                        except requests.exceptions.Timeout:
                            print_status(f"  Timeout: {href}", "warn")
                            failed += 1
                        except Exception as e:
                            print_status(f"  Failed: {href} - {e}", "warn")
                            failed += 1

            except Exception as e:
                print_status(f"Error fetching series {s}: {e}", "warn")

        print_status(f"Downloaded {downloaded} files, {failed} failed", "ok" if failed == 0 else "warn")
        return downloaded > 0


# =============================================================================
# Extractor: Extract ZIP files
# =============================================================================

class ThreeGPPExtractor:
    """Extract and organize downloaded 3GPP ZIP files"""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Pattern to identify main body files
        self.main_body_pattern = re.compile(r'main.*body|_1_|s\d+_s\d+', re.IGNORECASE)
        self.exclude_pattern = re.compile(r'cover|annex|history|amendment', re.IGNORECASE)

    def extract_all(self, source_dir: Path = None) -> int:
        """
        Extract all ZIP files to organized directory structure.

        Returns:
            Number of DOCX files extracted
        """
        print_header("Extracting ZIP Files")

        source = source_dir or self.config.download_dir

        if not source.exists():
            print_status(f"Source directory not found: {source}", "error")
            return 0

        # Find all ZIP files
        zip_files = list(source.rglob("*.zip"))
        print_status(f"Found {len(zip_files)} ZIP files", "info")

        extracted_count = 0

        for zip_path in zip_files:
            try:
                # Determine series from path or filename
                series = self._get_series_from_path(zip_path)
                if not series:
                    continue

                # Create series directory
                series_dir = self.data_dir / f"{series}_series"
                series_dir.mkdir(exist_ok=True)

                # Extract to spec folder
                spec_name = zip_path.stem
                extract_dir = series_dir / spec_name
                extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for file_info in zf.filelist:
                        filename = file_info.filename

                        # Skip directories and temp files
                        if filename.endswith('/') or filename.startswith('~') or '/' in filename:
                            # Extract nested file
                            base_name = Path(filename).name
                            if base_name.startswith('~'):
                                continue
                        else:
                            base_name = filename

                        # Extract DOCX files (main body only)
                        if base_name.endswith('.docx'):
                            if self._is_main_body(base_name):
                                # Extract to flat structure
                                target_path = extract_dir / base_name
                                with zf.open(file_info) as src:
                                    with open(target_path, 'wb') as dst:
                                        dst.write(src.read())
                                extracted_count += 1

            except zipfile.BadZipFile:
                print_status(f"Bad ZIP file: {zip_path.name}", "warn")
            except Exception as e:
                print_status(f"Error extracting {zip_path.name}: {e}", "warn")

        # Also copy direct DOCX files
        for docx_path in source.rglob("*.docx"):
            if docx_path.name.startswith('~'):
                continue

            if self._is_main_body(docx_path.name):
                series = self._get_series_from_path(docx_path)
                if series:
                    series_dir = self.data_dir / f"{series}_series"
                    series_dir.mkdir(exist_ok=True)

                    spec_name = docx_path.stem
                    target_dir = series_dir / spec_name
                    target_dir.mkdir(exist_ok=True)

                    target_path = target_dir / docx_path.name
                    if not target_path.exists():
                        shutil.copy(docx_path, target_path)
                        extracted_count += 1

        print_status(f"Extracted {extracted_count} main body DOCX files", "ok")
        return extracted_count

    def _get_series_from_path(self, path: Path) -> Optional[str]:
        """Extract series number from path"""
        # Try folder name first
        for part in path.parts:
            match = re.search(r'(\d{2})_series', part)
            if match:
                return match.group(1)

        # Try filename
        match = re.search(r'^(\d{2})\d{3}', path.name)
        if match:
            return match.group(1)

        return None

    def _is_main_body(self, filename: str) -> bool:
        """Check if file is a main body document"""
        if self.exclude_pattern.search(filename):
            return False

        # Main body file or matches pattern
        if self.main_body_pattern.search(filename):
            return True

        # Simple spec file (e.g., 23501-i80.docx)
        if re.match(r'^\d{5}-[a-z]\d{2}\.docx$', filename, re.IGNORECASE):
            return True

        return False


# =============================================================================
# Processor: Process DOCX to JSON
# =============================================================================

@dataclass
class DocumentMetaData:
    specification_id: str
    version: str
    title: str
    file_path: str
    release: str = ""


@dataclass
class SectionStructure:
    section_id: str
    title: str
    level: int
    parent_section: Optional[str] = None
    content: str = ""
    tables: List[Dict] = None


@dataclass
class ProcessedChunk:
    chunk_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    cross_references: Dict
    content_metadata: Dict = None
    tables: List[Dict] = None


class ThreeGPPParser:
    """Parse 3GPP DOCX documents into structured JSON"""

    def __init__(self):
        self.section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')

        # Reference patterns
        self.clause_pattern = re.compile(r'(?:clause|section)\s+(\d+(?:\.\d+)*)', re.IGNORECASE)
        self.table_pattern = re.compile(r'table\s+(\d+(?:\.\d+)*(?:-\d+)?)', re.IGNORECASE)
        self.figure_pattern = re.compile(r'figure\s+(\d+(?:\.\d+)*(?:-\d+)?)', re.IGNORECASE)

        # External reference patterns
        self.external_patterns = [
            re.compile(r'(?:3GPP\s+)?(?:TS|TR)\s+(\d+\.\d+)(?:,?\s+(?:clause|section)\s+(\d+(?:\.\d+)*))', re.IGNORECASE),
            re.compile(r'(?:clause|section)\s+(\d+(?:\.\d+)*)\s+of\s+(?:3GPP\s+)?(?:TS|TR)\s+(\d+\.\d+)', re.IGNORECASE),
            re.compile(r'(?:3GPP\s+)?(?:TS|TR)\s+(\d+\.\d+)(?!\s*(?:,?\s*(?:clause|section|table|figure)))', re.IGNORECASE)
        ]

    def extract_metadata(self, document_path: str) -> DocumentMetaData:
        """Extract metadata from document"""
        import mammoth
        from bs4 import BeautifulSoup

        with open(document_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)

        soup = BeautifulSoup(result.value, 'html.parser')
        text = soup.get_text()[:5000]  # First 5000 chars

        # Find TS number and version
        ts_version_match = re.search(r'3GPP\s+TS\s+(\d+\.\d+)\s+V(\d+\.\d+\.\d+)', text)

        if ts_version_match:
            ts_number = ts_version_match.group(1)
            version = ts_version_match.group(2)
            specification_id = f"ts_{ts_number}"
            title = f"3GPP TS {ts_number}"

            # Extract release from version (e.g., 18.9.0 -> Rel-18)
            release = f"Rel-{version.split('.')[0]}"
        else:
            # Fallback: extract from filename
            filename = Path(document_path).stem
            filename_match = re.search(r'(\d{2})(\d{3})', filename)
            if filename_match:
                ts_number = f"{filename_match.group(1)}.{filename_match.group(2)}"
                specification_id = f"ts_{ts_number}"
                title = f"3GPP TS {ts_number}"
            else:
                specification_id = f"ts_{filename}"
                title = f"3GPP {filename}"
            version = "Unknown"
            release = "Unknown"

        return DocumentMetaData(
            specification_id=specification_id,
            version=version,
            title=title,
            file_path=str(document_path),
            release=release
        )

    def parse_sections(self, document_path: str) -> List[SectionStructure]:
        """Parse document sections with content"""
        import mammoth
        from bs4 import BeautifulSoup

        with open(document_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)

        soup = BeautifulSoup(result.value, 'html.parser')
        sections = []
        current_section = None
        in_content = False

        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table']):
            text = element.get_text().strip()

            # Handle tables
            if element.name == 'table':
                if current_section:
                    table_data = self._extract_table(element)
                    if current_section.tables is None:
                        current_section.tables = []
                    current_section.tables.append(table_data)
                continue

            if not text:
                continue

            # Start content after first h1
            if element.name == 'h1' and not in_content:
                in_content = True
            if not in_content:
                continue

            # Check if it's a section heading
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                match = self.section_pattern.match(text)
                if match:
                    section_number = match.group(1)
                    section_title = match.group(2)
                    level = len(section_number.split('.'))
                    parent = '.'.join(section_number.split('.')[:-1]) if level > 1 else None

                    current_section = SectionStructure(
                        section_id=section_number,
                        title=section_title,
                        level=level,
                        parent_section=parent,
                        content="",
                        tables=[]
                    )
                    sections.append(current_section)
                    continue

            # Add content to current section
            if current_section:
                if current_section.content:
                    current_section.content += '\n' + text
                else:
                    current_section.content = text

        return sections

    def _extract_table(self, table_element) -> Dict:
        """Extract table data"""
        table_data = {"headers": [], "rows": []}
        rows = table_element.find_all('tr')

        if rows:
            headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
            table_data["headers"] = headers

            for row in rows[1:]:
                row_data = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if row_data:
                    table_data["rows"].append(row_data)

        return table_data

    def extract_cross_references(self, content: str, spec_id: str) -> Dict:
        """Extract cross-references from content"""
        internal_refs = []
        external_refs = []

        # Normalize spec_id for comparison
        spec_num = spec_id.replace('ts_', '')

        # Internal references
        for pattern, ref_type in [(self.clause_pattern, 'clause'),
                                  (self.table_pattern, 'table'),
                                  (self.figure_pattern, 'figure')]:
            for match in pattern.finditer(content):
                ref_num = match.group(1)

                # Check if it's an external reference
                context_start = max(0, match.start() - 50)
                context_end = min(len(content), match.end() + 50)
                context = content[context_start:context_end]

                ts_match = re.search(r'(?:3GPP\s+)?(?:TS|TR)\s+(\d+\.\d+)', context, re.IGNORECASE)

                if ts_match and ts_match.group(1) != spec_num:
                    external_refs.append({
                        "target_spec": f"ts_{ts_match.group(1)}",
                        "ref_type": ref_type,
                        "ref_id": ref_num,
                        "confidence": 0.9
                    })
                else:
                    internal_refs.append({
                        "ref_type": ref_type,
                        "ref_id": ref_num
                    })

        # Standalone external spec references
        for pattern in self.external_patterns:
            for match in pattern.finditer(content):
                groups = match.groups()
                if len(groups) >= 1:
                    target_num = groups[0]
                    target_spec = f"ts_{target_num}"
                    if target_num != spec_num:
                        external_refs.append({
                            "target_spec": target_spec,
                            "ref_type": "spec",
                            "ref_id": "",
                            "confidence": 0.8
                        })

        # Deduplicate external refs
        seen = set()
        unique_external = []
        for ref in external_refs:
            key = f"{ref['target_spec']}_{ref['ref_type']}_{ref['ref_id']}"
            if key not in seen:
                seen.add(key)
                unique_external.append(ref)

        return {
            "internal": internal_refs,
            "external": unique_external
        }

    def classify_content_type(self, title: str, content: str) -> str:
        """Classify content type"""
        title_lower = title.lower()
        content_lower = content.lower()[:500]

        if any(term in title_lower for term in ['definition', 'overview', 'general', 'scope']):
            return 'definition'
        elif any(term in title_lower for term in ['abbreviation']):
            return 'abbreviation'
        elif any(term in title_lower for term in ['procedure', 'flow', 'process', 'registration', 'handover']):
            return 'procedure'
        elif any(term in title_lower for term in ['parameter', 'identifier', 'ie', 'element']):
            return 'parameter'
        elif any(term in title_lower for term in ['reference', 'normative']):
            return 'reference'
        elif 'shall' in content_lower:
            return 'requirement'
        elif any(term in title_lower for term in ['interface', 'api', 'service']):
            return 'interface'
        else:
            return 'general'

    def compute_complexity(self, content: str) -> float:
        """Compute content complexity score"""
        word_count = len(content.split())

        # Count technical patterns
        tech_patterns = [
            r'\b[A-Z]{2,5}\b',  # Abbreviations
            r'clause\s+\d+',
            r'table\s+\d+',
            r'TS\s+\d+\.\d+',
        ]

        tech_count = sum(len(re.findall(p, content)) for p in tech_patterns)

        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count / 1000) * 0.5 + (tech_count / 50) * 0.5)
        return round(complexity, 3)

    def extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content"""
        # Find abbreviations
        abbrs = re.findall(r'\b([A-Z]{2,6})\b', content)

        # Count and get most common
        from collections import Counter
        counter = Counter(abbrs)

        return [term for term, count in counter.most_common(10)]

    def create_chunks(self, sections: List[SectionStructure],
                      metadata: DocumentMetaData) -> List[ProcessedChunk]:
        """Create content chunks"""
        chunks = []

        for section in sections:
            if not section.content.strip():
                continue

            content = section.content
            word_count = len(content.split())

            content_metadata = {
                "word_count": word_count,
                "complexity_score": self.compute_complexity(content),
                "key_terms": self.extract_key_terms(content)
            }

            chunk = ProcessedChunk(
                chunk_id=f"{metadata.specification_id}_{section.section_id}",
                section_id=section.section_id,
                section_title=section.title,
                content=content,
                chunk_type=self.classify_content_type(section.title, content),
                cross_references=self.extract_cross_references(content, metadata.specification_id),
                content_metadata=content_metadata,
                tables=section.tables
            )
            chunks.append(chunk)

        return chunks

    def process_document(self, document_path: str) -> Tuple[List[ProcessedChunk], DocumentMetaData]:
        """Process a single document"""
        metadata = self.extract_metadata(document_path)
        sections = self.parse_sections(document_path)
        chunks = self.create_chunks(sections, metadata)
        return chunks, metadata

    def save_to_json(self, chunks: List[ProcessedChunk],
                     metadata: DocumentMetaData,
                     output_dir: Path) -> str:
        """Save chunks to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{metadata.specification_id}.json"

        data = {
            "metadata": asdict(metadata),
            "export_info": {
                "export_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "pipeline_version": "v3"
            },
            "chunks": [asdict(chunk) for chunk in chunks]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(output_path)


class ThreeGPPProcessor:
    """Process all downloaded 3GPP documents"""

    def __init__(self, config: Config):
        self.config = config
        self.parser = ThreeGPPParser()
        self.output_dir = config.output_dir

        self.main_body_pattern = re.compile(r'main.*body|_1_|s\d+_s\d+', re.IGNORECASE)
        self.exclude_pattern = re.compile(r'cover|annex|history|amendment', re.IGNORECASE)

    def find_docx_files(self, data_dir: Path = None) -> List[Path]:
        """Find all main body DOCX files"""
        source = data_dir or self.config.data_dir
        docx_files = []

        for docx_path in source.rglob("*.docx"):
            filename = docx_path.name

            if filename.startswith('~'):
                continue

            if self._is_main_body(filename):
                docx_files.append(docx_path)

        return sorted(docx_files)

    def _is_main_body(self, filename: str) -> bool:
        """Check if file is main body document"""
        if self.exclude_pattern.search(filename):
            return False

        if self.main_body_pattern.search(filename):
            return True

        if re.match(r'^\d{5}-[a-z]\d{2}\.docx$', filename, re.IGNORECASE):
            return True

        return False

    def process_all(self, data_dir: Path = None, parallel: bool = True,
                    max_workers: int = 4) -> Tuple[int, int]:
        """Process all documents"""
        print_header("Processing Documents to JSON")

        docx_files = self.find_docx_files(data_dir)
        print_status(f"Found {len(docx_files)} main body DOCX files", "info")

        if not docx_files:
            print_status("No documents found to process", "warn")
            return 0, 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        fail_count = 0

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        if parallel and len(docx_files) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single, docx_path): docx_path
                    for docx_path in docx_files
                }

                iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing") if use_tqdm else as_completed(futures)

                for future in iterator:
                    docx_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        print_status(f"Error: {docx_path.name}: {e}", "warn")
                        fail_count += 1
        else:
            iterator = tqdm(docx_files, desc="Processing") if use_tqdm else docx_files

            for docx_path in iterator:
                try:
                    if self._process_single(docx_path):
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print_status(f"Error: {docx_path.name}: {e}", "warn")
                    fail_count += 1

        print_status(f"Processed {success_count} documents, {fail_count} failed",
                     "ok" if fail_count == 0 else "warn")
        return success_count, fail_count

    def _process_single(self, docx_path: Path) -> bool:
        """Process a single document"""
        try:
            chunks, metadata = self.parser.process_document(str(docx_path))

            if chunks:
                self.parser.save_to_json(chunks, metadata, self.output_dir)
                return True
            return False

        except Exception:
            return False


# =============================================================================
# Knowledge Graph Initializer
# =============================================================================

class KnowledgeGraphInitializer:
    """Initialize Neo4j Knowledge Graph from JSON files"""

    def __init__(self, config: Config):
        self.config = config
        self.json_dir = config.output_dir

    def initialize(self, clear_first: bool = True) -> bool:
        """Initialize knowledge graph from JSON files"""
        try:
            from neo4j import GraphDatabase
            from tqdm import tqdm
        except ImportError as e:
            print_status(f"Missing dependency: {e}", "error")
            return False

        print_header("Initializing Knowledge Graph")

        if not self.json_dir.exists():
            print_status(f"JSON directory not found: {self.json_dir}", "error")
            return False

        json_files = list(self.json_dir.glob("*.json"))
        if not json_files:
            print_status("No JSON files found", "error")
            return False

        print_status(f"Found {len(json_files)} JSON files", "info")

        print_status("Connecting to Neo4j...", "info")
        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print_status("Neo4j connection established", "ok")

            if clear_first:
                print_status("Clearing existing database...", "info")
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                print_status("Database cleared", "ok")

            # Load JSON files
            print_status("Loading JSON files...", "info")
            documents = {}
            chunks = []

            for json_file in tqdm(json_files, desc="Loading JSON"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    spec_id = data["metadata"]["specification_id"]
                    documents[spec_id] = data
                    chunks.extend(data["chunks"])

            print_status(f"Loaded {len(documents)} documents with {len(chunks)} chunks", "ok")

            # Create constraints
            print_status("Creating database constraints...", "info")
            with driver.session() as session:
                try:
                    session.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.spec_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT term_abbr IF NOT EXISTS FOR (t:Term) REQUIRE t.abbreviation IS UNIQUE")
                except:
                    pass

            # Create Document nodes
            print_status("Creating Document nodes...", "info")
            with driver.session() as session:
                for spec_id, data in tqdm(documents.items(), desc="Documents"):
                    session.run("""
                        MERGE (d:Document {spec_id: $spec_id})
                        SET d.version = $version,
                            d.title = $title,
                            d.release = $release,
                            d.total_chunks = $total_chunks
                    """,
                        spec_id=spec_id,
                        version=data["metadata"]["version"],
                        title=data["metadata"]["title"],
                        release=data["metadata"].get("release", ""),
                        total_chunks=data["export_info"]["total_chunks"]
                    )
            print_status(f"Created {len(documents)} Document nodes", "ok")

            # Create Chunk nodes
            print_status("Creating Chunk nodes...", "info")
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="Chunks"):
                    chunk_id = chunk["chunk_id"]

                    if "_" in chunk_id:
                        parts = chunk_id.rsplit("_", 1)
                        spec_id = parts[0]
                    else:
                        spec_id = chunk_id

                    content_meta = chunk.get("content_metadata", {})

                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.section_id = $section_id,
                            c.section_title = $section_title,
                            c.content = $content,
                            c.chunk_type = $chunk_type,
                            c.spec_id = $spec_id,
                            c.word_count = $word_count,
                            c.complexity_score = $complexity_score,
                            c.key_terms = $key_terms
                    """,
                        chunk_id=chunk["chunk_id"],
                        section_id=chunk["section_id"],
                        section_title=chunk["section_title"],
                        content=chunk["content"],
                        chunk_type=chunk["chunk_type"],
                        spec_id=spec_id,
                        word_count=content_meta.get("word_count", 0),
                        complexity_score=content_meta.get("complexity_score", 0.0),
                        key_terms=content_meta.get("key_terms", [])
                    )
            print_status(f"Created {len(chunks)} Chunk nodes", "ok")

            # Create CONTAINS relationships
            print_status("Creating CONTAINS relationships...", "info")
            with driver.session() as session:
                session.run("""
                    MATCH (d:Document), (c:Chunk)
                    WHERE d.spec_id = c.spec_id
                    MERGE (d)-[:CONTAINS]->(c)
                """)

            # Create REFERENCES_SPEC relationships
            print_status("Creating REFERENCES_SPEC relationships...", "info")
            ref_count = 0
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="References"):
                    cross_refs = chunk.get("cross_references", {})

                    for ref in cross_refs.get("external", []):
                        ref_uid = hashlib.md5(
                            f"{chunk['chunk_id']}_{ref['target_spec']}_{ref.get('ref_id', '')}".encode()
                        ).hexdigest()[:10]

                        result = session.run("""
                            MATCH (source:Chunk {chunk_id: $source_id})
                            MATCH (target_doc:Document {spec_id: $target_spec})
                            MERGE (source)-[r:REFERENCES_SPEC {ref_uid: $ref_uid}]->(target_doc)
                            SET r.ref_type = $ref_type,
                                r.ref_id = $ref_id,
                                r.confidence = $confidence
                            RETURN count(*) as created
                        """,
                            source_id=chunk["chunk_id"],
                            target_spec=ref["target_spec"],
                            ref_uid=ref_uid,
                            ref_type=ref.get("ref_type", ""),
                            ref_id=ref.get("ref_id", ""),
                            confidence=ref.get("confidence", 0.0)
                        )
                        rec = result.single()
                        if rec:
                            ref_count += rec["created"]

            print_status(f"Created {ref_count} REFERENCES_SPEC relationships", "ok")

            # Create Term nodes
            print_status("Creating Term nodes from abbreviations...", "info")
            term_count = self._create_term_nodes(driver, chunks)
            print_status(f"Created {term_count} Term nodes", "ok")

            print_status("Knowledge Graph initialized successfully!", "ok")
            return True

        except Exception as e:
            print_status(f"Error initializing graph: {e}", "error")
            import traceback
            traceback.print_exc()
            return False
        finally:
            driver.close()

    def _create_term_nodes(self, driver, chunks: list) -> int:
        """Create Term nodes from abbreviation sections"""
        try:
            from term_extractor import TermExtractor
        except ImportError:
            print_status("term_extractor not found, skipping Term nodes", "warn")
            return 0

        extractor = TermExtractor()
        term_dict = {}

        for chunk in chunks:
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("content", "")
            chunk_id = chunk["chunk_id"]

            if "_" in chunk_id:
                parts = chunk_id.rsplit("_", 1)
                spec_id = parts[0]
            else:
                spec_id = chunk_id

            if 'abbreviation' in section_title:
                terms = extractor.extract_abbreviations(content, spec_id)
                self._merge_terms(term_dict, terms)
            elif 'definition' in section_title:
                terms = extractor.extract_definitions(content, spec_id)
                self._merge_terms(term_dict, terms)

        term_count = 0
        with driver.session() as session:
            for abbr, term_info in term_dict.items():
                try:
                    session.run("""
                        MERGE (t:Term {abbreviation: $abbreviation})
                        SET t.full_name = $full_name,
                            t.term_type = $term_type,
                            t.source_specs = $source_specs,
                            t.primary_spec = $primary_spec
                    """,
                        abbreviation=abbr,
                        full_name=term_info['full_name'],
                        term_type=term_info['term_type'],
                        source_specs=term_info['source_specs'],
                        primary_spec=term_info['primary_spec']
                    )

                    for source_spec in term_info['source_specs']:
                        session.run("""
                            MATCH (t:Term {abbreviation: $abbreviation})
                            MATCH (d:Document {spec_id: $spec_id})
                            MERGE (t)-[:DEFINED_IN]->(d)
                        """,
                            abbreviation=abbr,
                            spec_id=source_spec
                        )

                    term_count += 1
                except:
                    pass

        return term_count

    def _merge_terms(self, term_dict: Dict, terms: list):
        """Merge extracted terms into dictionary"""
        for term in terms:
            abbr = term.abbreviation
            if abbr not in term_dict:
                term_dict[abbr] = {
                    'abbreviation': abbr,
                    'full_name': term.full_name,
                    'term_type': term.term_type,
                    'source_specs': [term.source_spec],
                    'primary_spec': term.source_spec
                }
            else:
                if term.source_spec not in term_dict[abbr]['source_specs']:
                    term_dict[abbr]['source_specs'].append(term.source_spec)


# =============================================================================
# Main Pipeline
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.downloader = ThreeGPPDownloader(self.config)
        self.extractor = ThreeGPPExtractor(self.config)
        self.processor = ThreeGPPProcessor(self.config)
        self.kg_initializer = KnowledgeGraphInitializer(self.config)

    def status(self):
        """Show current status"""
        print_header("Pipeline Status")

        # Download status
        download_dir = self.config.download_dir
        if download_dir.exists():
            zip_count = len(list(download_dir.rglob("*.zip")))
            print_status(f"Downloaded: {zip_count} ZIP files in {download_dir}", "ok" if zip_count > 0 else "warn")
        else:
            print_status("Downloaded: No files yet", "warn")

        # Extracted status
        data_dir = self.config.data_dir
        if data_dir.exists():
            docx_count = len(list(data_dir.rglob("*.docx")))
            print_status(f"Extracted: {docx_count} DOCX files in {data_dir}", "ok" if docx_count > 0 else "warn")
        else:
            print_status("Extracted: No files yet", "warn")

        # Processed status
        output_dir = self.config.output_dir
        if output_dir.exists():
            json_count = len(list(output_dir.glob("*.json")))
            print_status(f"Processed: {json_count} JSON files in {output_dir}", "ok" if json_count > 0 else "warn")
        else:
            print_status("Processed: No JSON files yet", "warn")

        # Neo4j status
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            with driver.session() as session:
                result = session.run("""
                    MATCH (d:Document) WITH count(d) as docs
                    MATCH (c:Chunk) WITH docs, count(c) as chunks
                    MATCH (t:Term) WITH docs, chunks, count(t) as terms
                    RETURN docs, chunks, terms
                """)
                record = result.single()
                print_status(f"Neo4j: {record['docs']} Documents, {record['chunks']} Chunks, {record['terms']} Terms", "ok")
            driver.close()
        except Exception as e:
            print_status(f"Neo4j: Not connected ({e})", "warn")

    def download(self):
        """Download 3GPP documents"""
        return self.downloader.download_release()

    def extract(self):
        """Extract downloaded ZIP files"""
        return self.extractor.extract_all()

    def process(self):
        """Process documents to JSON"""
        return self.processor.process_all()

    def init_kg(self):
        """Initialize Knowledge Graph"""
        return self.kg_initializer.initialize()

    def run_all(self):
        """Run complete pipeline"""
        print_header("3GPP Document Processing Pipeline")
        print_status(f"Release: {self.config.release}", "info")
        print_status(f"Series: {self.config.series}", "info")
        print_status(f"Output: {self.config.output_dir}", "info")
        print()

        # Step 1: Download
        if not self.download():
            print_status("Download failed or no new files", "warn")

        # Step 2: Extract
        extracted = self.extract()
        if extracted == 0:
            # Try processing existing data_dir
            existing_docx = len(list(self.config.data_dir.rglob("*.docx"))) if self.config.data_dir.exists() else 0
            if existing_docx == 0:
                print_status("No files to process. Check download.", "error")
                return False

        # Step 3: Process
        success, fail = self.process()
        if success == 0:
            print_status("No documents processed.", "error")
            return False

        # Step 4: Initialize KG
        if not self.init_kg():
            print_status("Knowledge Graph initialization failed.", "error")
            return False

        print_header("Pipeline Complete!")
        self.status()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="3GPP Document Download and Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_and_process_3gpp.py status
  python download_and_process_3gpp.py download --release 18 --series 23,29
  python download_and_process_3gpp.py process
  python download_and_process_3gpp.py all
        """
    )
    parser.add_argument(
        "command",
        choices=["download", "extract", "process", "init-kg", "all", "status"],
        help="Command to run"
    )
    parser.add_argument(
        "--release", "-r",
        type=int,
        default=18,
        help="3GPP Release number (default: 18)"
    )
    parser.add_argument(
        "--series", "-s",
        type=str,
        default=None,
        help="Comma-separated series numbers (e.g., '23,29,38')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for JSON files"
    )

    args = parser.parse_args()

    # Create config
    config = Config(release=args.release)

    if args.series:
        config.series = [s.strip() for s in args.series.split(',')]

    if args.output:
        config.output_dir = Path(args.output)

    # Create pipeline
    pipeline = Pipeline(config)

    # Run command
    commands = {
        "download": pipeline.download,
        "extract": pipeline.extract,
        "process": pipeline.process,
        "init-kg": pipeline.init_kg,
        "all": pipeline.run_all,
        "status": pipeline.status
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
