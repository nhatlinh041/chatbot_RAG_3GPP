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

# Import orchestrator functions
import orchestrator


class TestOrchestratorConfig:
    """Test orchestrator configuration"""
    
    def test_config_creation(self):
        # Test basic config creation
        config = OrchestratorConfig()
        assert config.neo4j_container_name == "neo4j-server"
        assert config.django_port == 8000
        assert config.neo4j_wait_timeout == 60
    
    def test_config_with_custom_values(self):
        # Test config with custom values
        config = OrchestratorConfig(
            neo4j_container_name="custom-neo4j",
            django_port=9000,
            neo4j_wait_timeout=120
        )
        assert config.neo4j_container_name == "custom-neo4j"
        assert config.django_port == 9000  
        assert config.neo4j_wait_timeout == 120


class TestSystemChecker:
    """Test system status checking"""
    
    def setup_method(self):
        self.checker = SystemChecker()
    
    @patch('orchestrator.subprocess.run')
    def test_check_docker_running_success(self, mock_run):
        # Test successful docker check
        mock_run.return_value = MagicMock(returncode=0)
        result = self.checker.check_docker_running()
        assert result is True
    
    @patch('orchestrator.subprocess.run')
    def test_check_docker_running_failure(self, mock_run):
        # Test failed docker check
        mock_run.return_value = MagicMock(returncode=1)
        result = self.checker.check_docker_running()
        assert result is False
    
    @patch('orchestrator.os.path.exists')
    def test_check_json_files_exist(self, mock_exists):
        # Test JSON files existence check
        mock_exists.return_value = True
        result = self.checker.check_json_files()
        assert "Found" in result
    
    @patch('orchestrator.os.path.exists')
    def test_check_json_files_missing(self, mock_exists):
        # Test missing JSON files
        mock_exists.return_value = False
        result = self.checker.check_json_files()
        assert "Not found" in result
    
    @patch('orchestrator.os.getenv')
    def test_check_env_variables(self, mock_getenv):
        # Test environment variable check
        mock_getenv.side_effect = lambda x: "test_key" if x == "CLAUDE_API_KEY" else None
        result = self.checker.check_env_variables()
        assert "CLAUDE_API_KEY is set" in result


class TestNeo4jManager:
    """Test Neo4j container management"""
    
    def setup_method(self):
        self.manager = Neo4jManager()
    
    @patch('orchestrator.subprocess.run')
    def test_start_neo4j_success(self, mock_run):
        # Test successful Neo4j start
        mock_run.return_value = MagicMock(returncode=0, stdout="Started")
        result = self.manager.start_neo4j()
        assert result is True
    
    @patch('orchestrator.subprocess.run')
    def test_start_neo4j_failure(self, mock_run):
        # Test failed Neo4j start
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        result = self.manager.start_neo4j()
        assert result is False
    
    @patch('orchestrator.subprocess.run') 
    def test_stop_neo4j_success(self, mock_run):
        # Test successful Neo4j stop
        mock_run.return_value = MagicMock(returncode=0)
        result = self.manager.stop_neo4j()
        assert result is True
    
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.Neo4jManager.is_neo4j_ready')
    def test_wait_for_neo4j_ready(self, mock_is_ready, mock_sleep):
        # Test waiting for Neo4j to be ready
        mock_is_ready.side_effect = [False, False, True]
        result = self.manager.wait_for_neo4j(timeout=30)
        assert result is True
    
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.Neo4jManager.is_neo4j_ready')
    def test_wait_for_neo4j_timeout(self, mock_is_ready, mock_sleep):
        # Test timeout when waiting for Neo4j
        mock_is_ready.return_value = False
        result = self.manager.wait_for_neo4j(timeout=5)
        assert result is False


class TestDependencyManager:
    """Test dependency management"""
    
    def setup_method(self):
        self.manager = DependencyManager()
    
    @patch('orchestrator.subprocess.run')
    def test_install_dependencies_success(self, mock_run):
        # Test successful dependency installation
        mock_run.return_value = MagicMock(returncode=0)
        result = self.manager.install_dependencies()
        assert result is True
    
    @patch('orchestrator.subprocess.run')
    def test_install_dependencies_failure(self, mock_run):
        # Test failed dependency installation  
        mock_run.return_value = MagicMock(returncode=1)
        result = self.manager.install_dependencies()
        assert result is False
    
    @patch('orchestrator.os.path.exists')
    def test_check_venv_exists(self, mock_exists):
        # Test virtual environment check
        mock_exists.return_value = True
        result = self.manager.check_venv()
        assert result is True
    
    @patch('orchestrator.os.path.exists')
    def test_check_venv_missing(self, mock_exists):
        # Test missing virtual environment
        mock_exists.return_value = False
        result = self.manager.check_venv()
        assert result is False


class TestServiceManager:
    """Test service management (Django, ngrok)"""
    
    def setup_method(self):
        self.manager = ServiceManager()
    
    @patch('orchestrator.subprocess.Popen')
    def test_start_django_success(self, mock_popen):
        # Test successful Django start
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        result = self.manager.start_django()
        assert result == mock_process
    
    @patch('orchestrator.subprocess.Popen')
    def test_start_ngrok_success(self, mock_popen):
        # Test successful ngrok start
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        result = self.manager.start_ngrok()
        assert result == mock_process
    
    def test_stop_service(self):
        # Test service stopping
        mock_process = MagicMock()
        result = self.manager.stop_service(mock_process)
        mock_process.terminate.assert_called_once()
        assert result is True
    
    def test_stop_service_none(self):
        # Test stopping None service
        result = self.manager.stop_service(None)
        assert result is True


class TestOrchestratorIntegration:
    """Test full orchestrator integration"""
    
    def setup_method(self):
        self.orchestrator = Orchestrator()
    
    @patch('orchestrator.SystemChecker.check_all')
    def test_orchestrator_check_command(self, mock_check):
        # Test check command
        mock_check.return_value = "System OK"
        result = self.orchestrator.run_command("check")
        assert result is True
    
    @patch('orchestrator.Neo4jManager.start_neo4j')
    def test_orchestrator_start_neo4j_command(self, mock_start):
        # Test start-neo4j command
        mock_start.return_value = True
        result = self.orchestrator.run_command("start-neo4j")
        assert result is True
    
    @patch('orchestrator.DependencyManager.install_dependencies')
    def test_orchestrator_install_command(self, mock_install):
        # Test install command
        mock_install.return_value = True
        result = self.orchestrator.run_command("install")
        assert result is True
    
    def test_orchestrator_invalid_command(self):
        # Test invalid command
        result = self.orchestrator.run_command("invalid-command")
        assert result is False
    
    @patch('orchestrator.Neo4jManager.stop_neo4j')
    @patch('orchestrator.ServiceManager.stop_all')
    def test_orchestrator_stop_command(self, mock_stop_services, mock_stop_neo4j):
        # Test stop command
        mock_stop_neo4j.return_value = True
        mock_stop_services.return_value = True
        result = self.orchestrator.run_command("stop")
        assert result is True


class TestOrchestratorCommandLineInterface:
    """Test command line interface functionality"""
    
    @patch('sys.argv', ['orchestrator.py', 'check'])
    @patch('orchestrator.Orchestrator.run_command')
    def test_cli_check_command(self, mock_run):
        # Test CLI check command
        mock_run.return_value = True
        # Would normally test main() function
        # This tests the command parsing logic
        assert True  # Placeholder for actual CLI testing
    
    @patch('sys.argv', ['orchestrator.py', 'all', '--init-kg'])
    def test_cli_all_command_with_flag(self):
        # Test CLI all command with flag
        # This would test flag parsing
        assert True  # Placeholder for actual CLI testing


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator"""
    
    def setup_method(self):
        self.orchestrator = Orchestrator()
    
    @patch('orchestrator.Neo4jManager.start_neo4j')
    def test_neo4j_start_error_handling(self, mock_start):
        # Test error handling when Neo4j fails to start
        mock_start.side_effect = Exception("Docker error")
        result = self.orchestrator.run_command("start-neo4j")
        assert result is False
    
    @patch('orchestrator.SystemChecker.check_all')
    def test_system_check_error_handling(self, mock_check):
        # Test error handling during system check
        mock_check.side_effect = Exception("System error")
        result = self.orchestrator.run_command("check")
        assert result is False


class TestOrchestratorEnvironmentLoading:
    """Test environment variable loading"""
    
    @patch('orchestrator.load_dotenv')
    def test_load_env_file(self, mock_load_dotenv):
        # Test loading .env file
        orchestrator = Orchestrator()
        mock_load_dotenv.assert_called_once()
    
    @patch('orchestrator.os.getenv')
    def test_env_variable_access(self, mock_getenv):
        # Test accessing environment variables
        mock_getenv.return_value = "test_value"
        orchestrator = Orchestrator()
        # Test that environment variables are accessible
        assert True  # Placeholder for actual environment testing