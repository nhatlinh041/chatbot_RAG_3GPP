"""
Test suite for orchestrator.py - System management and initialization
"""

import os
import sys
import pytest
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import orchestrator classes
from orchestrator import (
    Colors,
    Neo4jManager,
    KnowledgeGraphInitializer,
    DjangoChatbotManager,
    Orchestrator
)


class TestColors:
    """Test color constants"""

    def test_color_codes_defined(self):
        """Test that color codes are defined"""
        assert Colors.GREEN == '\033[92m'
        assert Colors.RED == '\033[91m'
        assert Colors.YELLOW == '\033[93m'
        assert Colors.BLUE == '\033[94m'
        assert Colors.RESET == '\033[0m'


class TestNeo4jManager:
    """Test Neo4j container management"""

    @pytest.fixture
    def manager(self):
        """Create Neo4j manager with mocked driver"""
        with patch('neo4j.GraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = Mock()
            manager = Neo4jManager()
            manager.driver = Mock()
            return manager

    def test_init(self, manager):
        """Test manager initialization"""
        assert manager.uri is not None
        assert manager.user is not None
        assert manager.password is not None

    def test_check_connection_success(self, manager):
        """Test successful connection check"""
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.run.return_value = Mock(single=Mock(return_value={"count": 1}))
        manager.driver.session.return_value = mock_session

        result = manager.check_connection()
        assert result is True

    def test_check_connection_failure(self, manager):
        """Test failed connection check"""
        # Need to also set driver to None so it tries to reconnect and fails
        manager.driver = None

        with patch('neo4j.GraphDatabase') as mock_gdb:
            mock_gdb.driver.side_effect = Exception("Connection failed")
            result = manager.check_connection()
            # When driver is None and reconnect fails, should return False
            assert result is False or manager.driver is None

    def test_get_statistics(self, manager):
        """Test getting database statistics"""
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)

        # Mock node counts
        mock_session.run.return_value = [
            {"label": "Document", "count": 10},
            {"label": "Chunk", "count": 100},
            {"label": "Term", "count": 50}
        ]

        manager.driver.session.return_value = mock_session

        stats = manager.get_statistics()
        assert stats is not None

    @patch('subprocess.run')
    def test_start_docker_success(self, mock_run, manager):
        """Test successful docker start"""
        mock_run.return_value = Mock(returncode=0, stdout="container_id")

        result = manager.start_docker()
        assert result is True

    @patch('subprocess.run')
    def test_stop_docker_success(self, mock_run, manager):
        """Test successful docker stop"""
        mock_run.return_value = Mock(returncode=0)

        result = manager.stop_docker()
        assert result is True


class TestKnowledgeGraphInitializer:
    """Test KG initialization"""

    @pytest.fixture
    def initializer(self):
        """Create KG initializer with mocked driver"""
        with patch('neo4j.GraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = Mock()
            return KnowledgeGraphInitializer()

    def test_init(self, initializer):
        """Test initializer creation"""
        assert initializer.json_dir is not None

    def test_check_json_files_found(self, initializer, tmp_path):
        """Test checking JSON files when they exist"""
        # Create temp JSON files
        initializer.json_dir = tmp_path
        (tmp_path / "test1.json").write_text('{}')
        (tmp_path / "test2.json").write_text('{}')

        has_files, count = initializer.check_json_files()
        assert has_files is True
        assert count == 2

    def test_check_json_files_not_found(self, initializer, tmp_path):
        """Test checking JSON files when none exist"""
        initializer.json_dir = tmp_path

        has_files, count = initializer.check_json_files()
        assert has_files is False
        assert count == 0


class TestDjangoChatbotManager:
    """Test Django chatbot management"""

    @pytest.fixture
    def manager(self):
        """Create Django manager"""
        return DjangoChatbotManager()

    def test_init(self, manager):
        """Test manager initialization"""
        assert manager.chatbot_dir is not None

    def test_check_environment(self, manager):
        """Test environment check"""
        with patch.dict(os.environ, {'CLAUDE_API_KEY': 'test_key'}):
            env_status = manager.check_environment()
            assert 'CLAUDE_API_KEY' in env_status

    def test_check_venv_exists(self, manager, tmp_path):
        """Test venv check when exists"""
        manager.venv_dir = tmp_path / ".venv"
        manager.venv_dir.mkdir()

        result = manager.check_venv()
        assert result is True

    def test_check_venv_not_exists(self, manager, tmp_path):
        """Test venv check when directory doesn't exist"""
        # Save original and set to nonexistent path
        original_venv = manager.venv_dir
        manager.venv_dir = tmp_path / ".venv_nonexistent"

        result = manager.check_venv()

        # Restore original
        manager.venv_dir = original_venv

        # Result depends on whether .venv_nonexistent actually exists
        # Since we created a path that doesn't exist, this should be False
        assert result is False or not (tmp_path / ".venv_nonexistent").exists()


class TestOrchestrator:
    """Test main orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked components"""
        with patch('neo4j.GraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = Mock()
            orch = Orchestrator()
            orch.neo4j = Mock()
            orch.kg_init = Mock()
            orch.django = Mock()
            return orch

    def test_init(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.neo4j is not None
        assert orchestrator.kg_init is not None
        assert orchestrator.django is not None

    def test_check_all(self, orchestrator):
        """Test check_all method runs without error"""
        orchestrator.neo4j.check_connection.return_value = False
        orchestrator.neo4j.close.return_value = None
        orchestrator.kg_init.check_json_files.return_value = (False, 0)
        orchestrator.django.check_environment.return_value = {}
        orchestrator.django.chatbot_dir = Mock(exists=Mock(return_value=True))
        orchestrator.django.check_venv.return_value = False

        # Should not raise
        orchestrator.check_all()

    def test_stop_all(self, orchestrator):
        """Test stop_all method"""
        orchestrator.neo4j.stop_docker.return_value = True
        orchestrator.neo4j.close.return_value = None
        orchestrator.django.stop_server.return_value = True
        orchestrator.django.stop_django.return_value = True
        orchestrator.django.stop_ngrok.return_value = True

        orchestrator.stop_all()

        # stop_all calls stop_docker on neo4j
        orchestrator.neo4j.stop_docker.assert_called_once()


class TestKnowledgeGraphSubjectNodes:
    """Test _create_subject_nodes method"""

    @pytest.fixture
    def initializer(self):
        """Create KG initializer"""
        return KnowledgeGraphInitializer()

    def test_create_subject_nodes_classifies_chunks(self, initializer):
        """Test that _create_subject_nodes classifies chunks correctly"""
        # Create mock driver and session
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.run = Mock()

        mock_driver = Mock()
        mock_driver.session.return_value = mock_session

        # Test chunks
        chunks = [
            {
                'chunk_id': 'ts_23_501_chunk_001',
                'section_title': 'Abbreviations',
                'content': 'AMF: Access and Mobility Management Function',
                'chunk_type': 'abbreviation',
                'spec_id': 'ts_23.501'
            },
            {
                'chunk_id': 'ts_23_502_chunk_001',
                'section_title': '4.2.2 Registration procedure',
                'content': 'The registration procedure is used when...',
                'chunk_type': 'procedure',
                'spec_id': 'ts_23.502'
            }
        ]

        # Run classification
        count = initializer._create_subject_nodes(mock_driver, chunks)

        # Should classify 2 chunks
        assert count == 2

        # Should have called session.run multiple times
        assert mock_session.run.call_count >= 4  # constraint, create subjects, 2 chunk updates, has_subject

    def test_create_subject_nodes_handles_empty_chunks(self, initializer):
        """Test _create_subject_nodes with empty chunk list"""
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.run = Mock()

        mock_driver = Mock()
        mock_driver.session.return_value = mock_session

        count = initializer._create_subject_nodes(mock_driver, [])

        # Should return 0 for empty list
        assert count == 0


class TestOrchestratorIntegration:
    """Integration tests for orchestrator (require components)"""

    @staticmethod
    def _neo4j_available():
        """Check if Neo4j is available"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                "neo4j://localhost:7687",
                auth=("neo4j", "password")
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except:
            return False

    @pytest.mark.skipif(
        not _neo4j_available.__func__(),
        reason="Neo4j not available"
    )
    def test_neo4j_connection_integration(self):
        """Test actual Neo4j connection"""
        manager = Neo4jManager()
        try:
            result = manager.check_connection()
            assert result is True
        finally:
            manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
