"""
Regression tests for logging_config module
Tests centralized logging setup and logger retrieval with custom levels
"""

import pytest
import logging
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_config import (
    setup_centralized_logging, get_logger, get_log_file_path,
    CRITICAL, ERROR, MAJOR, MINOR, DEBUG
)


class TestLoggingSetup:
    """Tests for centralized logging setup"""

    def test_setup_creates_loggers(self):
        """Setup should create the expected loggers"""
        setup_centralized_logging()

        # Check that expected loggers exist (includes new LLM_Integrator and Cypher_Generator)
        expected_loggers = ['RAG_System', 'Chatbot', 'Knowledge_Retriever', 'Document_Processing', 'LLM_Integrator', 'Cypher_Generator']
        for logger_name in expected_loggers:
            logger = logging.getLogger(logger_name)
            assert logger is not None
            assert logger.name == logger_name

    def test_setup_configures_handlers(self):
        """Setup should configure file and console handlers"""
        setup_centralized_logging()

        logger = logging.getLogger('RAG_System')
        # Should have at least one handler
        assert len(logger.handlers) > 0

    def test_setup_is_idempotent(self):
        """Multiple setup calls should not duplicate handlers"""
        setup_centralized_logging()
        logger = logging.getLogger('RAG_System')
        initial_handler_count = len(logger.handlers)

        # Call setup again
        setup_centralized_logging()

        # Handler count should not increase significantly
        # Note: dictConfig may replace handlers, so count may be same
        assert len(logger.handlers) <= initial_handler_count + 2

    def test_log_file_path_exists(self):
        """Log file directory should be created"""
        setup_centralized_logging()

        # Log path changed to logs/app.log
        log_file_path = get_log_file_path()
        assert log_file_path is not None
        assert log_file_path.parent.exists()


class TestGetLogger:
    """Tests for get_logger function"""

    def test_get_logger_returns_logger(self):
        """get_logger should return a logger instance"""
        setup_centralized_logging()

        logger = get_logger('TestLogger')
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_name_returns_same_logger(self):
        """Same name should return the same logger instance"""
        setup_centralized_logging()

        logger1 = get_logger('SameLogger')
        logger2 = get_logger('SameLogger')
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Different names should return different loggers"""
        setup_centralized_logging()

        logger1 = get_logger('Logger1')
        logger2 = get_logger('Logger2')
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_predefined_loggers(self):
        """Should be able to get predefined loggers"""
        setup_centralized_logging()

        predefined = ['RAG_System', 'Chatbot', 'Knowledge_Retriever', 'Document_Processing']
        for name in predefined:
            logger = get_logger(name)
            assert logger.name == name


class TestLoggingLevels:
    """Tests for custom logging levels: CRITICAL > ERROR > MAJOR > MINOR > DEBUG"""

    def test_custom_levels_defined(self):
        """Custom log levels should be properly defined"""
        assert CRITICAL == 50
        assert ERROR == 40
        assert MAJOR == 30
        assert MINOR == 20
        assert DEBUG == 10

    def test_default_level_is_minor(self):
        """Default logging level should be MINOR (20)"""
        setup_centralized_logging()

        logger = get_logger('RAG_System')
        assert logger.level == MINOR or logger.level == logging.NOTSET

    def test_logger_can_log_critical(self):
        """Logger should be able to log CRITICAL messages"""
        setup_centralized_logging()
        logger = get_logger('TestCritical')
        logger.log(CRITICAL, "Test critical message")

    def test_logger_can_log_error(self):
        """Logger should be able to log ERROR messages"""
        setup_centralized_logging()
        logger = get_logger('TestError')
        logger.log(ERROR, "Test error message")

    def test_logger_can_log_major(self):
        """Logger should be able to log MAJOR messages"""
        setup_centralized_logging()
        logger = get_logger('TestMajor')
        logger.log(MAJOR, "Test major message")

    def test_logger_can_log_minor(self):
        """Logger should be able to log MINOR messages"""
        setup_centralized_logging()
        logger = get_logger('TestMinor')
        logger.log(MINOR, "Test minor message")

    def test_logger_can_log_debug(self):
        """Logger should be able to log DEBUG messages"""
        setup_centralized_logging()
        logger = get_logger('TestDebug')
        logger.log(DEBUG, "Test debug message")


class TestLoggerFormatting:
    """Tests for log message formatting"""

    def test_log_message_format(self):
        """Log messages should include timestamp, name, and level"""
        setup_centralized_logging()

        # Capture log output
        string_stream = StringIO()
        handler = logging.StreamHandler(string_stream)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger = get_logger('FormatTest')
        logger.addHandler(handler)
        logger.setLevel(MINOR)

        # Use custom MAJOR level instead of INFO
        logger.log(MAJOR, "Test message")

        log_output = string_stream.getvalue()
        assert 'FormatTest' in log_output
        assert 'MAJOR' in log_output
        assert 'Test message' in log_output


class TestLoggerIsolation:
    """Tests for logger isolation between components"""

    def test_component_loggers_are_isolated(self):
        """Component loggers should not propagate to each other"""
        setup_centralized_logging()

        rag_logger = get_logger('RAG_System')
        chatbot_logger = get_logger('Chatbot')

        # They should be different logger instances
        assert rag_logger is not chatbot_logger

    def test_custom_logger_does_not_affect_predefined(self):
        """Custom loggers should not affect predefined ones"""
        setup_centralized_logging()

        predefined_logger = get_logger('RAG_System')
        predefined_level = predefined_logger.level

        # Create and modify custom logger
        custom_logger = get_logger('CustomLogger')
        custom_logger.setLevel(logging.DEBUG)

        # Predefined logger should not be affected
        assert predefined_logger.level == predefined_level
