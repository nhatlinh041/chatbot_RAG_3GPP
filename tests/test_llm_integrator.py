"""
Test suite for LLM Integrator - Unified Claude & Local LLM integration
"""

import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from rag_system_v2 import LLMIntegrator, LLMConfig


class TestLLMConfig:
    """Test LLM configuration class"""
    
    def test_config_creation_defaults(self):
        # Test default configuration
        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.timeout == 30
        assert config.retries == 3
    
    def test_config_creation_custom(self):
        # Test custom configuration
        config = LLMConfig(
            temperature=0.5,
            max_tokens=2000,
            timeout=60,
            retries=5
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 60
        assert config.retries == 5


class TestLLMIntegratorInitialization:
    """Test LLM integrator initialization"""
    
    def test_init_with_claude_only(self):
        # Test initialization with Claude API key only
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url=None
        )
        assert integrator.claude_api_key == "test_key"
        assert integrator.local_llm_url is None
        assert integrator.claude_client is not None
    
    def test_init_with_local_llm_only(self):
        # Test initialization with local LLM only
        integrator = LLMIntegrator(
            claude_api_key=None,
            local_llm_url="http://localhost:11434/api/chat"
        )
        assert integrator.claude_api_key is None
        assert integrator.local_llm_url == "http://localhost:11434/api/chat"
    
    def test_init_with_both(self):
        # Test initialization with both Claude and local LLM
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
        assert integrator.claude_api_key == "test_key"
        assert integrator.local_llm_url == "http://localhost:11434/api/chat"
        assert integrator.claude_client is not None
    
    def test_init_with_neither_raises_error(self):
        # Test initialization with neither API key nor local URL raises error
        with pytest.raises(ValueError, match="Either claude_api_key or local_llm_url must be provided"):
            LLMIntegrator(claude_api_key=None, local_llm_url=None)


class TestLLMIntegratorModelRouting:
    """Test model routing logic"""
    
    def setup_method(self):
        self.integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
    
    def test_is_claude_model_true(self):
        # Test Claude model detection
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307", 
            "claude-3-opus-20240229",
            "claude-2.1",
            "claude-2.0"
        ]
        for model in claude_models:
            assert self.integrator._is_claude_model(model) is True
    
    def test_is_claude_model_false(self):
        # Test non-Claude model detection
        local_models = [
            "deepseek-r1:7b",
            "deepseek-r1:14b",
            "llama3:8b",
            "mixtral:8x7b",
            "custom-model"
        ]
        for model in local_models:
            assert self.integrator._is_claude_model(model) is False
    
    def test_is_local_model_available_true(self):
        # Test local model availability when URL exists
        assert self.integrator._is_local_model_available() is True
    
    def test_is_local_model_available_false(self):
        # Test local model availability when URL is None
        integrator = LLMIntegrator(claude_api_key="test_key", local_llm_url=None)
        assert integrator._is_local_model_available() is False


class TestLLMIntegratorClaudeGeneration:
    """Test Claude API generation"""
    
    def setup_method(self):
        with patch('anthropic.Anthropic'):
            self.integrator = LLMIntegrator(
                claude_api_key="test_key", 
                local_llm_url="http://localhost:11434/api/chat"
            )
    
    @patch('anthropic.Anthropic')
    def test_generate_claude_success(self, mock_anthropic_class):
        # Test successful Claude generation
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock the response structure
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        # Create new integrator with mocked client
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
        integrator.claude_client = mock_client
        
        result = integrator._generate_claude_response(
            "test prompt",
            "claude-3-5-sonnet-20241022"
        )
        
        assert result == "Test response"
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_generate_claude_api_error(self, mock_anthropic_class):
        # Test Claude API error handling
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
        integrator.claude_client = mock_client
        
        result = integrator._generate_claude_response(
            "test prompt", 
            "claude-3-5-sonnet-20241022"
        )
        
        assert "Error generating response" in result
    
    def test_generate_claude_no_client(self):
        # Test Claude generation without client
        integrator = LLMIntegrator(
            claude_api_key=None,
            local_llm_url="http://localhost:11434/api/chat"
        )
        
        result = integrator._generate_claude_response(
            "test prompt",
            "claude-3-5-sonnet-20241022"
        )
        
        assert "Claude client not available" in result


class TestLLMIntegratorLocalGeneration:
    """Test local LLM generation"""
    
    def setup_method(self):
        self.integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
    
    @patch('requests.post')
    def test_generate_local_success(self, mock_post):
        # Test successful local LLM generation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Local response"}
        }
        mock_post.return_value = mock_response
        
        result = self.integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b"
        )
        
        assert result == "Local response"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_local_http_error(self, mock_post):
        # Test local LLM HTTP error
        mock_post.side_effect = Exception("Connection error")
        
        result = self.integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b"
        )
        
        assert "Error generating response" in result
    
    @patch('requests.post')
    def test_generate_local_invalid_response(self, mock_post):
        # Test invalid response format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}
        mock_post.return_value = mock_response
        
        result = self.integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b"
        )
        
        assert "Error extracting response" in result
    
    def test_generate_local_no_url(self):
        # Test local generation without URL
        integrator = LLMIntegrator(claude_api_key="test_key", local_llm_url=None)
        
        result = integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b"
        )
        
        assert "Local LLM not available" in result


class TestLLMIntegratorMainInterface:
    """Test main generate_response interface"""
    
    def setup_method(self):
        with patch('anthropic.Anthropic'):
            self.integrator = LLMIntegrator(
                claude_api_key="test_key",
                local_llm_url="http://localhost:11434/api/chat"
            )
    
    @patch.object(LLMIntegrator, '_generate_claude_response')
    def test_generate_response_claude_model(self, mock_claude):
        # Test routing to Claude for Claude models
        mock_claude.return_value = "Claude response"
        
        result = self.integrator.generate_response(
            "test prompt",
            model="claude-3-5-sonnet-20241022"
        )
        
        assert result == "Claude response"
        mock_claude.assert_called_once_with("test prompt", "claude-3-5-sonnet-20241022")
    
    @patch.object(LLMIntegrator, '_generate_local_response')
    def test_generate_response_local_model(self, mock_local):
        # Test routing to local LLM for local models
        mock_local.return_value = "Local response"
        
        result = self.integrator.generate_response(
            "test prompt",
            model="deepseek-r1:7b"
        )
        
        assert result == "Local response"
        mock_local.assert_called_once_with("test prompt", "deepseek-r1:7b")
    
    @patch.object(LLMIntegrator, '_generate_claude_response')
    def test_generate_response_fallback_to_claude(self, mock_claude):
        # Test fallback to Claude when local not available
        mock_claude.return_value = "Claude fallback response"
        
        integrator = LLMIntegrator(claude_api_key="test_key", local_llm_url=None)
        result = integrator.generate_response(
            "test prompt",
            model="deepseek-r1:7b"
        )
        
        assert result == "Claude fallback response"
        mock_claude.assert_called_once()
    
    def test_generate_response_no_models_available(self):
        # Test error when no models are available
        integrator = LLMIntegrator(claude_api_key=None, local_llm_url=None)
        
        with pytest.raises(ValueError):
            integrator.generate_response("test prompt", model="any-model")


class TestLLMIntegratorRetryLogic:
    """Test retry logic and error handling"""
    
    def setup_method(self):
        self.integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
    
    @patch('requests.post')
    def test_local_generation_retry_success(self, mock_post):
        # Test retry logic with eventual success
        # First call fails, second succeeds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Success on retry"}}
        
        mock_post.side_effect = [
            Exception("First failure"),
            mock_response
        ]
        
        result = self.integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b",
            retries=2
        )
        
        assert result == "Success on retry"
        assert mock_post.call_count == 2
    
    @patch('requests.post')
    def test_local_generation_retry_exhausted(self, mock_post):
        # Test retry logic with all attempts failing
        mock_post.side_effect = Exception("Persistent failure")
        
        result = self.integrator._generate_local_response(
            "test prompt",
            "deepseek-r1:7b",
            retries=2
        )
        
        assert "Error generating response" in result
        assert mock_post.call_count == 3  # Initial + 2 retries


class TestLLMIntegratorConfiguration:
    """Test configuration handling"""
    
    def test_default_config_applied(self):
        # Test default configuration is applied
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
        
        assert integrator.config.temperature == 0.7
        assert integrator.config.max_tokens == 4000
        assert integrator.config.timeout == 30
    
    def test_custom_config_applied(self):
        # Test custom configuration is applied
        custom_config = LLMConfig(temperature=0.5, max_tokens=2000)
        
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat",
            config=custom_config
        )
        
        assert integrator.config.temperature == 0.5
        assert integrator.config.max_tokens == 2000


class TestLLMIntegratorIntegration:
    """Test integration with actual components"""
    
    @patch('anthropic.Anthropic')
    @patch('requests.post')
    def test_full_integration_scenario(self, mock_post, mock_anthropic):
        # Test full integration scenario
        # Setup Claude mock
        mock_claude_client = MagicMock()
        mock_anthropic.return_value = mock_claude_client
        mock_claude_response = MagicMock()
        mock_claude_response.content = [MagicMock(text="Claude integration response")]
        mock_claude_client.messages.create.return_value = mock_claude_response
        
        # Setup local LLM mock
        mock_local_response = MagicMock()
        mock_local_response.status_code = 200
        mock_local_response.json.return_value = {
            "message": {"content": "Local integration response"}
        }
        mock_post.return_value = mock_local_response
        
        integrator = LLMIntegrator(
            claude_api_key="test_key",
            local_llm_url="http://localhost:11434/api/chat"
        )
        
        # Test Claude model
        claude_result = integrator.generate_response(
            "What is 5G?",
            model="claude-3-5-sonnet-20241022"
        )
        assert claude_result == "Claude integration response"
        
        # Test local model
        local_result = integrator.generate_response(
            "What is 5G?",
            model="deepseek-r1:7b"
        )
        assert local_result == "Local integration response"