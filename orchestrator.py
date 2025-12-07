#!/usr/bin/env python3
"""
3GPP Knowledge Graph & RAG System Orchestrator

Manages all system components:
- Neo4j database connection verification
- Knowledge Graph initialization from JSON files
- Django chatbot server

Usage:
    python orchestrator.py check      # Check all component status
    python orchestrator.py init-kg    # Initialize/rebuild knowledge graph
    python orchestrator.py run        # Start Django chatbot server
    python orchestrator.py all        # Init KG + Run server
"""

import os
import sys
import argparse
import subprocess
import time
import signal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file():
    """Load environment variables from .env file"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value


# Load .env file on import
load_env_file()


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    colors = {
        "ok": Colors.GREEN,
        "error": Colors.RED,
        "warn": Colors.YELLOW,
        "info": Colors.BLUE
    }
    symbols = {
        "ok": "✓",
        "error": "✗",
        "warn": "!",
        "info": "→"
    }
    color = colors.get(status, Colors.RESET)
    symbol = symbols.get(status, "→")
    print(f"{color}{symbol}{Colors.RESET} {message}")


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*50}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*50}{Colors.RESET}\n")


class Neo4jManager:
    """Manage Neo4j database connection and status"""

    def __init__(self):
        self.uri = "neo4j://localhost:7687"
        self.user = "neo4j"
        self.password = "password"
        self.driver = None
        self.docker_container = "neo4j-server"

    def check_connection(self) -> bool:
        """Check if Neo4j is running and accessible"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            return False

    def _get_docker_cmd(self) -> list:
        """Get docker command with sudo if needed"""
        # Check if we can run docker without sudo
        check = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True
        )

        if check.returncode == 0:
            return ["docker"]
        else:
            # Need sudo
            return ["sudo", "docker"]

    def start_docker(self) -> bool:
        """Start Neo4j Docker container"""
        print_status(f"Starting Neo4j Docker container: {self.docker_container}", "info")

        try:
            docker_cmd = self._get_docker_cmd()

            # Check if container exists
            check_result = subprocess.run(
                docker_cmd + ["ps", "-a", "--filter", f"name={self.docker_container}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )

            container_exists = self.docker_container in check_result.stdout

            if not container_exists:
                # Create new container
                print_status("Container doesn't exist, creating new one...", "info")
                create_result = subprocess.run(
                    docker_cmd + [
                        "run", "-d",
                        "--name", self.docker_container,
                        "-p", "7474:7474",
                        "-p", "7687:7687",
                        "-e", f"NEO4J_AUTH={self.user}/{self.password}",
                        "neo4j:latest"
                    ],
                    capture_output=True,
                    text=True
                )

                if create_result.returncode != 0:
                    print_status(f"Failed to create container: {create_result.stderr}", "error")
                    print_status("Make sure Docker is installed and you have permissions", "info")
                    print_status("Option 1: Add yourself to docker group and re-login:", "info")
                    print_status("  sudo usermod -aG docker $USER", "info")
                    print_status("  newgrp docker", "info")
                    print_status("Option 2: Run orchestrator with sudo:", "info")
                    print_status("  sudo python3 orchestrator.py start-neo4j", "info")
                    return False

                print_status(f"Container '{self.docker_container}' created", "ok")

            else:
                # Check if container is already running
                running_check = subprocess.run(
                    docker_cmd + ["ps", "--filter", f"name={self.docker_container}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True
                )

                if self.docker_container in running_check.stdout:
                    print_status(f"Container '{self.docker_container}' is already running", "ok")
                    return True

                # Start existing container
                print_status("Starting existing container...", "info")
                start_result = subprocess.run(
                    docker_cmd + ["start", self.docker_container],
                    capture_output=True,
                    text=True
                )

                if start_result.returncode != 0:
                    print_status(f"Failed to start container: {start_result.stderr}", "error")
                    return False

                print_status(f"Container '{self.docker_container}' started", "ok")

            # Wait for Neo4j to be ready
            print_status("Waiting for Neo4j to be ready...", "info")
            for i in range(30):
                time.sleep(2)
                if self.check_connection():
                    print_status("Neo4j is ready!", "ok")
                    return True
                if (i + 1) % 5 == 0:  # Print every 5 attempts
                    print_status(f"Still waiting... ({i+1}/30)", "info")

            print_status("Neo4j started but not responding yet", "warn")
            print_status("It may take a few more moments. Check: docker logs neo4j-server", "info")
            return True

        except FileNotFoundError:
            print_status("Docker not found. Please install Docker.", "error")
            print_status("Visit: https://docs.docker.com/get-docker/", "info")
            return False
        except Exception as e:
            print_status(f"Error starting Docker: {e}", "error")
            return False

    def stop_docker(self) -> bool:
        """Stop Neo4j Docker container"""
        print_status(f"Stopping Neo4j Docker container: {self.docker_container}", "info")

        try:
            docker_cmd = self._get_docker_cmd()
            result = subprocess.run(
                docker_cmd + ["stop", self.docker_container],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print_status(f"Container '{self.docker_container}' stopped", "ok")
                return True
            else:
                print_status(f"Container was not running or doesn't exist", "info")
                return True

        except Exception as e:
            print_status(f"Error stopping Docker: {e}", "error")
            return False

    def get_statistics(self) -> dict:
        """Get database statistics"""
        if not self.driver:
            return {}

        try:
            with self.driver.session() as session:
                # Count nodes
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                """)
                stats = {"nodes": {}}
                for record in result:
                    if record['label']:
                        stats["nodes"][record['label']] = record['count']

                # Count relationships
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                """)
                stats["relationships"] = {}
                for record in result:
                    stats["relationships"][record['rel_type']] = record['count']

                return stats
        except Exception:
            return {}

    def close(self):
        """Close driver connection"""
        if self.driver:
            self.driver.close()


class KnowledgeGraphInitializer:
    """Initialize Knowledge Graph from JSON files"""

    def __init__(self):
        self.json_dir = PROJECT_ROOT / "3GPP_JSON_DOC" / "processed_json_v2"
        self.neo4j_uri = "neo4j://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"

    def check_json_files(self) -> tuple:
        """Check if JSON files exist"""
        if not self.json_dir.exists():
            return False, 0

        json_files = list(self.json_dir.glob("*.json"))
        return len(json_files) > 0, len(json_files)

    def initialize_graph(self, clear_first: bool = True) -> bool:
        """Initialize knowledge graph from JSON files"""
        import json
        import hashlib
        from neo4j import GraphDatabase
        from tqdm import tqdm

        print_status("Connecting to Neo4j...", "info")
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )

        try:
            # Test connection
            with driver.session() as session:
                session.run("RETURN 1")
            print_status("Neo4j connection established", "ok")

            # Clear database if requested
            if clear_first:
                print_status("Clearing existing database...", "info")
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                print_status("Database cleared", "ok")

            # Load JSON files
            print_status(f"Loading JSON files from {self.json_dir}...", "info")
            documents = {}
            chunks = []

            json_files = list(self.json_dir.glob("*.json"))
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
                except:
                    pass

            # Create documents
            print_status("Creating Document nodes...", "info")
            with driver.session() as session:
                for spec_id, data in tqdm(documents.items(), desc="Documents"):
                    session.run("""
                        MERGE (d:Document {spec_id: $spec_id})
                        SET d.version = $version,
                            d.title = $title,
                            d.total_chunks = $total_chunks
                    """,
                        spec_id=spec_id,
                        version=data["metadata"]["version"],
                        title=data["metadata"]["title"],
                        total_chunks=data["export_info"]["total_chunks"]
                    )
            print_status(f"Created {len(documents)} Document nodes", "ok")

            # Create chunks
            print_status("Creating Chunk nodes...", "info")
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="Chunks"):
                    # Extract spec_id from chunk_id
                    # Format: "ts_XX_YYY_chunk_NNN" -> "ts_XX.YYY"
                    chunk_id = chunk["chunk_id"]
                    if "_chunk_" in chunk_id:
                        # Split by "_chunk_" and take the first part
                        spec_part = chunk_id.split("_chunk_")[0]
                        # Replace second underscore with dot
                        parts = spec_part.split("_")
                        if len(parts) >= 3:
                            spec_id = f"{parts[0]}_{parts[1]}.{parts[2]}"
                        else:
                            spec_id = spec_part
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
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="References"):
                    source_chunk_id = chunk["chunk_id"]
                    cross_refs = chunk.get("cross_references", {})

                    for ref in cross_refs.get("external", []):
                        ref_uid = hashlib.md5(
                            f"{source_chunk_id}_{ref['target_spec']}_{ref['ref_id']}".encode()
                        ).hexdigest()[:10]

                        result = session.run("""
                            MATCH (source:Chunk {chunk_id: $source_id})
                            MATCH (target_doc:Document {spec_id: $target_spec})
                            RETURN count(*) as exists
                        """, source_id=source_chunk_id, target_spec=ref["target_spec"])

                        if result.single()["exists"] > 0:
                            session.run("""
                                MATCH (source:Chunk {chunk_id: $source_id})
                                MATCH (target_doc:Document {spec_id: $target_spec})
                                CREATE (source)-[r:REFERENCES_SPEC]->(target_doc)
                                SET r.ref_id = $ref_id,
                                    r.ref_type = $ref_type,
                                    r.confidence = $confidence,
                                    r.is_external = true,
                                    r.ref_uid = $ref_uid
                            """,
                                source_id=source_chunk_id,
                                target_spec=ref["target_spec"],
                                ref_id=ref["ref_id"],
                                ref_type=ref["ref_type"],
                                confidence=ref["confidence"],
                                ref_uid=ref_uid
                            )

            print_status("Knowledge Graph base structure initialized!", "ok")

            # Create Term nodes from abbreviation/definition chunks
            print_status("Creating Term nodes from abbreviations...", "info")
            term_count = self._create_term_nodes(driver, chunks)
            print_status(f"Created {term_count} Term nodes", "ok")

            print_status("Knowledge Graph initialized successfully!", "ok")
            return True

        except Exception as e:
            print_status(f"Error initializing graph: {e}", "error")
            return False
        finally:
            driver.close()

    def _create_term_nodes(self, driver, chunks: list) -> int:
        """Create Term nodes from abbreviation/definition sections in chunks"""
        from term_extractor import TermExtractor

        extractor = TermExtractor()
        term_dict = {}

        # Extract terms from abbreviation/definition chunks
        for chunk in chunks:
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("content", "")

            # Get spec_id from chunk_id
            chunk_id = chunk["chunk_id"]
            if "_chunk_" in chunk_id:
                spec_part = chunk_id.split("_chunk_")[0]
                parts = spec_part.split("_")
                if len(parts) >= 3:
                    spec_id = f"{parts[0]}_{parts[1]}.{parts[2]}"
                else:
                    spec_id = spec_part
            else:
                spec_id = chunk_id

            # Only process abbreviation/definition sections
            if 'abbreviation' in section_title:
                terms = extractor.extract_abbreviations(content, spec_id)
                self._merge_terms(term_dict, terms)
            elif 'definition' in section_title:
                terms = extractor.extract_definitions(content, spec_id)
                self._merge_terms(term_dict, terms)

        # Create Term nodes in Neo4j
        created_count = 0
        with driver.session() as session:
            # Create constraint
            try:
                session.run("CREATE CONSTRAINT term_abbr IF NOT EXISTS FOR (t:Term) REQUIRE t.abbreviation IS UNIQUE")
            except:
                pass

            # Create Term nodes
            for abbr, term_data in term_dict.items():
                try:
                    session.run("""
                        MERGE (t:Term {abbreviation: $abbr})
                        SET t.full_name = $full_name,
                            t.term_type = $term_type,
                            t.source_specs = $source_specs,
                            t.primary_spec = $primary_spec
                    """,
                        abbr=abbr,
                        full_name=term_data['full_name'],
                        term_type=term_data['term_type'],
                        source_specs=term_data['source_specs'],
                        primary_spec=term_data['primary_spec']
                    )

                    # Create DEFINED_IN relationships to Documents
                    for spec_id in term_data['source_specs']:
                        session.run("""
                            MATCH (t:Term {abbreviation: $abbr})
                            MATCH (d:Document {spec_id: $spec_id})
                            MERGE (t)-[:DEFINED_IN]->(d)
                        """,
                            abbr=abbr,
                            spec_id=spec_id
                        )

                    created_count += 1
                except Exception as e:
                    pass  # Skip problematic terms

        return created_count

    def _merge_terms(self, term_dict: dict, terms: list):
        """Merge terms into consolidated dictionary.
        Prioritizes 5G Core specs (ts_23.5xx, ts_29.5xx) over legacy specs.
        """
        # 5G Core spec prefixes - these should take priority for definitions
        _5g_spec_prefixes = ('ts_23.5', 'ts_29.5', 'ts_23.4', 'ts_29.2')

        def is_5g_spec(spec_id: str) -> bool:
            """Check if spec is a 5G Core related specification"""
            return any(spec_id.startswith(prefix) for prefix in _5g_spec_prefixes)

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
                # Add source spec if not already present
                if term.source_spec not in term_dict[abbr]['source_specs']:
                    term_dict[abbr]['source_specs'].append(term.source_spec)

                # Update full_name if new source is a 5G spec and existing is not
                # This ensures 5G definitions take priority over legacy definitions
                # e.g., SCP = "Service Communication Proxy" (5G) over "Service Control Part" (IN)
                existing_is_5g = is_5g_spec(term_dict[abbr]['primary_spec'])
                new_is_5g = is_5g_spec(term.source_spec)

                if new_is_5g and not existing_is_5g:
                    term_dict[abbr]['full_name'] = term.full_name
                    term_dict[abbr]['primary_spec'] = term.source_spec


class DjangoChatbotManager:
    """Manage Django chatbot server"""

    def __init__(self):
        self.chatbot_dir = PROJECT_ROOT / "chatbot_project"
        self.venv_dir = PROJECT_ROOT / ".venv"
        self.process = None

    def get_python_executable(self) -> str:
        """Get the Python executable, preferring venv if available"""
        # Check for .venv in project root
        venv_python = self.venv_dir / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)

        # Check for venv (alternative name)
        venv_python_alt = PROJECT_ROOT / "venv" / "bin" / "python"
        if venv_python_alt.exists():
            return str(venv_python_alt)

        # Fall back to current Python
        return sys.executable

    def check_environment(self) -> dict:
        """Check required environment variables"""
        return {
            "CLAUDE_API_KEY": bool(os.getenv("CLAUDE_API_KEY")),
        }

    def check_venv(self) -> bool:
        """Check if virtual environment exists"""
        return self.venv_dir.exists() or (PROJECT_ROOT / "venv").exists()

    def check_django_installed(self) -> bool:
        """Check if Django is installed in the venv"""
        python_exe = self.get_python_executable()
        try:
            result = subprocess.run(
                [python_exe, "-c", "import django"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def start_ngrok(self) -> bool:
        """Start ngrok tunnels"""
        print_status("Starting ngrok tunnels...", "info")

        try:
            process = subprocess.Popen(
                ["ngrok", "start", "--all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print_status("ngrok started (running in background)", "ok")
            print_status("View tunnels at: http://127.0.0.1:4040", "info")
            return True

        except FileNotFoundError:
            print_status("ngrok not found. Install from: https://ngrok.com/download", "error")
            return False
        except Exception as e:
            print_status(f"Error starting ngrok: {e}", "error")
            return False

    def stop_ngrok(self) -> bool:
        """Stop ngrok processes"""
        try:
            result = subprocess.run(
                ["pkill", "-f", "ngrok"],
                capture_output=True
            )
            if result.returncode == 0:
                print_status("ngrok stopped", "ok")
            else:
                print_status("ngrok was not running", "info")
            return True
        except Exception as e:
            print_status(f"Error stopping ngrok: {e}", "error")
            return False

    def stop_django(self) -> bool:
        """Stop Django server processes"""
        try:
            result = subprocess.run(
                ["pkill", "-f", "manage.py runserver"],
                capture_output=True
            )
            if result.returncode == 0:
                print_status("Django server stopped", "ok")
            else:
                print_status("Django server was not running", "info")
            return True
        except Exception as e:
            print_status(f"Error stopping Django: {e}", "error")
            return False

    def install_dependencies(self) -> bool:
        """Install dependencies in the virtual environment"""
        python_exe = self.get_python_executable()
        requirements_file = PROJECT_ROOT / "requirements.txt"

        if not requirements_file.exists():
            print_status("requirements.txt not found", "error")
            return False

        print_status(f"Installing dependencies from {requirements_file}", "info")
        print_status(f"Using: {python_exe}", "info")
        print()

        try:
            # Upgrade pip first
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                check=True
            )

            # Install requirements
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )

            print()
            print_status("Dependencies installed successfully", "ok")
            return True

        except subprocess.CalledProcessError as e:
            print_status(f"Installation failed: {e}", "error")
            return False

    def run_server(self, host: str = "127.0.0.1", port: int = 9999) -> bool:
        """Run Django development server"""
        env = os.environ.copy()
        python_exe = self.get_python_executable()

        # Check for API key
        if not env.get("CLAUDE_API_KEY"):
            print_status("CLAUDE_API_KEY not set - RAG will use DeepSeek only", "warn")

        # Check venv
        if not self.check_venv():
            print_status("Virtual environment not found (.venv/)", "error")
            print_status("Run: python -m venv .venv", "info")
            return False

        # Check Django installed
        if not self.check_django_installed():
            print_status("Django not installed in virtual environment", "error")
            print_status(f"Run: {python_exe} -m pip install -r requirements.txt", "info")
            print_status("Or:  source .venv/bin/activate && pip install django", "info")
            return False

        print_status(f"Using Python: {python_exe}", "info")
        print_status(f"Starting Django server at http://{host}:{port}/", "info")
        print_status("Press Ctrl+C to stop the server", "info")
        print()

        try:
            self.process = subprocess.Popen(
                [python_exe, "manage.py", "runserver", f"{host}:{port}"],
                cwd=str(self.chatbot_dir),
                env=env
            )
            self.process.wait()
            return True
        except KeyboardInterrupt:
            print()
            print_status("Shutting down server...", "info")
            if self.process:
                self.process.terminate()
                self.process.wait()
            print_status("Server stopped", "ok")
            return True


class Orchestrator:
    """Main orchestrator for all components"""

    def __init__(self):
        self.neo4j = Neo4jManager()
        self.kg_init = KnowledgeGraphInitializer()
        self.django = DjangoChatbotManager()

    def check_all(self):
        """Check status of all components"""
        print_header("System Status Check")

        # Check Neo4j
        print(f"{Colors.BOLD}Neo4j Database:{Colors.RESET}")
        if self.neo4j.check_connection():
            print_status(f"Connected to {self.neo4j.uri}", "ok")
            stats = self.neo4j.get_statistics()
            if stats:
                print_status(f"Documents: {stats.get('nodes', {}).get('Document', 0)}", "info")
                print_status(f"Chunks: {stats.get('nodes', {}).get('Chunk', 0)}", "info")
                print_status(f"Terms: {stats.get('nodes', {}).get('Term', 0)}", "info")
                rels = stats.get('relationships', {})
                print_status(f"CONTAINS: {rels.get('CONTAINS', 0)}", "info")
                print_status(f"REFERENCES_SPEC: {rels.get('REFERENCES_SPEC', 0)}", "info")
                print_status(f"DEFINED_IN: {rels.get('DEFINED_IN', 0)}", "info")
            else:
                print_status("Database is empty - run 'init-kg' to populate", "warn")
        else:
            print_status("Not connected - ensure Neo4j is running", "error")

        self.neo4j.close()

        # Check JSON files
        print(f"\n{Colors.BOLD}JSON Data Files:{Colors.RESET}")
        has_files, count = self.kg_init.check_json_files()
        if has_files:
            print_status(f"Found {count} JSON files in processed_json_v2/", "ok")
        else:
            print_status("No JSON files found - run document processing first", "error")

        # Check environment
        print(f"\n{Colors.BOLD}Environment Variables:{Colors.RESET}")
        env_status = self.django.check_environment()
        for var, is_set in env_status.items():
            if is_set:
                print_status(f"{var} is set", "ok")
            else:
                print_status(f"{var} not set", "warn")

        # Check Django
        print(f"\n{Colors.BOLD}Django Chatbot:{Colors.RESET}")
        if self.django.chatbot_dir.exists():
            print_status(f"Project directory exists: {self.django.chatbot_dir}", "ok")
        else:
            print_status("Django project not found", "error")

        # Check virtual environment
        print(f"\n{Colors.BOLD}Virtual Environment:{Colors.RESET}")
        if self.django.check_venv():
            python_exe = self.django.get_python_executable()
            print_status(f"Found: {python_exe}", "ok")

            # Check Django installed
            if self.django.check_django_installed():
                print_status("Django is installed", "ok")
            else:
                print_status("Django NOT installed", "error")
                print_status(f"Run: {python_exe} -m pip install -r requirements.txt", "info")
        else:
            print_status("Not found - create with: python -m venv .venv", "error")

        print()

    def init_knowledge_graph(self, clear_first: bool = True):
        """Initialize knowledge graph"""
        print_header("Knowledge Graph Initialization")

        # Check Neo4j first
        if not self.neo4j.check_connection():
            print_status("Neo4j is not running. Please start Neo4j first.", "error")
            print_status("Run: neo4j start (or start Neo4j Desktop)", "info")
            return False

        self.neo4j.close()

        # Check JSON files
        has_files, count = self.kg_init.check_json_files()
        if not has_files:
            print_status("No JSON files found to process", "error")
            return False

        print_status(f"Found {count} JSON files to process", "info")

        # Initialize
        success = self.kg_init.initialize_graph(clear_first=clear_first)

        if success:
            print_header("Initialization Complete")
            # Show final stats
            if self.neo4j.check_connection():
                stats = self.neo4j.get_statistics()
                print(f"Final Statistics:")
                print(f"  Documents: {stats.get('nodes', {}).get('Document', 0)}")
                print(f"  Chunks: {stats.get('nodes', {}).get('Chunk', 0)}")
                print(f"  Terms: {stats.get('nodes', {}).get('Term', 0)}")
                print(f"  Relationships: {sum(stats.get('relationships', {}).values())}")
                self.neo4j.close()

        return success

    def run_chatbot(self, host: str = "127.0.0.1", port: int = 9999) -> bool:
        """Run Django chatbot server"""
        print_header("Starting Django Chatbot")

        # Check Neo4j
        if not self.neo4j.check_connection():
            print_status("Neo4j is not running - RAG queries will fail", "warn")
        else:
            stats = self.neo4j.get_statistics()
            if not stats.get('nodes', {}).get('Document'):
                print_status("Knowledge Graph is empty - run 'init-kg' first", "warn")

        self.neo4j.close()

        # Run server
        return self.django.run_server(host=host, port=port)

    def run_all(self, host: str = "127.0.0.1", port: int = 9999, init_kg: bool = True):
        """Start all services: Neo4j, initialize KG, start ngrok, and run chatbot"""
        print_header("Full System Startup")

        # Start Neo4j if not running
        if not self.neo4j.check_connection():
            print_status("Neo4j not running, starting Docker container...", "info")
            if not self.neo4j.start_docker():
                print_status("Failed to start Neo4j, continuing anyway...", "warn")

        # Initialize KG
        kg_ready = False
        if self.neo4j.check_connection():
            if init_kg:
                print_status("Initializing Knowledge Graph...", "info")
                self.neo4j.close()
                # Wait for KG initialization to complete
                kg_ready = self.init_knowledge_graph(clear_first=True)
                if not kg_ready:
                    print_status("Failed to initialize KG, continuing anyway...", "warn")
                else:
                    print_status("Knowledge Graph initialization complete", "ok")
            else:
                stats = self.neo4j.get_statistics()
                if stats.get('nodes', {}).get('Document', 0) > 0:
                    print_status("Knowledge Graph already populated", "ok")
                    print_status("Skipping initialization (use --init-kg to rebuild)", "info")
                    kg_ready = True
                else:
                    print_status("Knowledge Graph is empty, initializing...", "info")
                    self.neo4j.close()
                    kg_ready = self.init_knowledge_graph()
                    if not kg_ready:
                        print_status("Failed to initialize KG, continuing anyway...", "warn")
                    else:
                        print_status("Knowledge Graph initialization complete", "ok")
        else:
            print_status("Neo4j not available, skipping KG check", "warn")

        self.neo4j.close()

        # Start ngrok
        print_status("Starting ngrok tunnels...", "info")
        self.django.start_ngrok()

        # Run chatbot
        self.run_chatbot(host=host, port=port)

    def start_neo4j(self):
        """Start Neo4j Docker container"""
        print_header("Starting Neo4j")

        # Check if already running
        if self.neo4j.check_connection():
            print_status("Neo4j is already running", "ok")
            return True

        # Start Docker container
        return self.neo4j.start_docker()

    def start_ngrok(self):
        """Start ngrok tunnels"""
        print_header("Starting ngrok")
        return self.django.start_ngrok()

    def stop_all(self):
        """Stop all running services"""
        print_header("Stopping All Services")

        # Stop Django
        print_status("Stopping Django server...", "info")
        self.django.stop_django()

        # Stop ngrok
        print_status("Stopping ngrok...", "info")
        self.django.stop_ngrok()

        # Stop Neo4j
        print_status("Stopping Neo4j...", "info")
        self.neo4j.stop_docker()

        print()
        print_status("All services stopped", "ok")
        return True

    def install_deps(self):
        """Install dependencies in virtual environment"""
        print_header("Installing Dependencies")

        # Check venv exists
        if not self.django.check_venv():
            print_status("Virtual environment not found", "error")
            print_status("Creating virtual environment...", "info")

            # Create venv
            venv_path = PROJECT_ROOT / ".venv"
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True
                )
                print_status(f"Created virtual environment at {venv_path}", "ok")
            except subprocess.CalledProcessError as e:
                print_status(f"Failed to create venv: {e}", "error")
                return False

        # Install dependencies
        return self.django.install_dependencies()


def main():
    parser = argparse.ArgumentParser(
        description="3GPP Knowledge Graph & RAG System Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  check       Check status of all components
  install     Install dependencies in virtual environment
  start-neo4j Start Neo4j Docker container
  init-kg     Initialize/rebuild knowledge graph from JSON files
  run         Start Django chatbot server
  ngrok       Start ngrok tunnels (ngrok start --all)
  all         Start everything: Neo4j, KG (if empty), ngrok, and Django server
  stop        Stop all services (Django, ngrok, Neo4j)

Examples:
  python orchestrator.py check
  python orchestrator.py install
  python orchestrator.py start-neo4j
  python orchestrator.py init-kg
  python orchestrator.py run --port 8080
  python orchestrator.py ngrok
  python orchestrator.py all              # Start all (skip KG init if already populated)
  python orchestrator.py all --init-kg    # Start all + force KG re-initialization
  python orchestrator.py stop
        """
    )

    parser.add_argument(
        "command",
        choices=["check", "install", "start-neo4j", "init-kg", "run", "ngrok", "all", "stop"],
        help="Command to execute"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for Django server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port for Django server (default: 9999)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data when initializing KG"
    )
    parser.add_argument(
        "--init-kg",
        action="store_true",
        help="Force KG initialization when using 'all' command"
    )

    args = parser.parse_args()

    orchestrator = Orchestrator()

    if args.command == "check":
        orchestrator.check_all()

    elif args.command == "install":
        success = orchestrator.install_deps()
        sys.exit(0 if success else 1)

    elif args.command == "start-neo4j":
        success = orchestrator.start_neo4j()
        sys.exit(0 if success else 1)

    elif args.command == "init-kg":
        success = orchestrator.init_knowledge_graph(clear_first=not args.no_clear)
        sys.exit(0 if success else 1)

    elif args.command == "run":
        orchestrator.run_chatbot(host=args.host, port=args.port)

    elif args.command == "ngrok":
        success = orchestrator.start_ngrok()
        sys.exit(0 if success else 1)

    elif args.command == "all":
        orchestrator.run_all(host=args.host, port=args.port, init_kg=args.init_kg)

    elif args.command == "stop":
        success = orchestrator.stop_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
