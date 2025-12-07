"""
Centralized logging configuration for 3GPP Knowledge Graph project.
Provides 5 custom log levels and automatic log rotation.
"""
import logging
import logging.handlers
import os
import json
from pathlib import Path
from datetime import datetime


# Custom log levels: CRITICAL > ERROR > MAJOR > MINOR > DEBUG
CRITICAL = 50
ERROR = 40
MAJOR = 30
MINOR = 20
DEBUG = 10

# Register custom level names
logging.addLevelName(CRITICAL, 'CRITICAL')
logging.addLevelName(ERROR, 'ERROR')
logging.addLevelName(MAJOR, 'MAJOR')
logging.addLevelName(MINOR, 'MINOR')
logging.addLevelName(DEBUG, 'DEBUG')


class CustomLogger(logging.Logger):
    """Logger with custom log level methods"""

    def major(self, msg, *args, **kwargs):
        if self.isEnabledFor(MAJOR):
            self._log(MAJOR, msg, args, **kwargs)

    def minor(self, msg, *args, **kwargs):
        if self.isEnabledFor(MINOR):
            self._log(MINOR, msg, args, **kwargs)


# Set custom logger class
logging.setLoggerClass(CustomLogger)


def load_log_config() -> dict:
    """
    Load logging configuration from log_config.json.
    Returns default config if file not found.
    """
    config_path = Path(__file__).parent / "log_config.json"

    # Default configuration
    default_config = {
        "log_level": "MINOR",
        "console_level": "MAJOR",
        "file_level": "DEBUG",
        "max_size_mb": 1024,
        "log_dir": "logs",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception:
            pass

    return default_config


def get_level_number(level_name: str) -> int:
    """Convert level name string to level number"""
    levels = {
        'CRITICAL': CRITICAL,
        'ERROR': ERROR,
        'MAJOR': MAJOR,
        'MINOR': MINOR,
        'DEBUG': DEBUG
    }
    return levels.get(level_name.upper(), MINOR)


class StartupRotatingHandler(logging.handlers.RotatingFileHandler):
    """
    Custom handler that creates new log file on startup and rotates when size exceeds limit.
    """

    def __init__(self, filename, max_bytes, **kwargs):
        # Create logs directory if needed
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate existing log on startup
        if log_path.exists() and log_path.stat().st_size > 0:
            self._rotate_on_startup(log_path)

        super().__init__(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=10,
            **kwargs
        )

    def _rotate_on_startup(self, log_path: Path):
        """Archive existing log file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}"

        try:
            log_path.rename(archive_name)
        except Exception:
            pass


# Global state to track initialization
_initialized = False
_log_file_path = None


def setup_centralized_logging():
    """
    Initialize centralized logging with rotation and custom levels.
    Creates new log file on each startup.
    """
    global _initialized, _log_file_path

    if _initialized:
        return

    # Load configuration
    config = load_log_config()

    # Setup log file path
    log_dir = Path(__file__).parent / config['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_file_path = log_dir / "app.log"

    # Calculate max size in bytes (default 1GB)
    max_bytes = config['max_size_mb'] * 1024 * 1024

    # Create formatters
    file_formatter = logging.Formatter(config['format'])
    console_formatter = logging.Formatter(config['console_format'])

    # Create handlers
    file_handler = StartupRotatingHandler(
        filename=str(_log_file_path),
        max_bytes=max_bytes,
        encoding='utf-8'
    )
    file_handler.setLevel(get_level_number(config['file_level']))
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(get_level_number(config['console_level']))
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(get_level_number(config['log_level']))

    # Clear existing handlers and add new ones
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure component loggers
    components = ['RAG_System', 'Chatbot', 'Knowledge_Retriever', 'Document_Processing', 'LLM_Integrator', 'Cypher_Generator']
    for component in components:
        logger = logging.getLogger(component)
        logger.setLevel(get_level_number(config['log_level']))
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False

    _initialized = True

    # Log startup message
    logger = get_logger('System')
    logger.log(MAJOR, f"Centralized logging initialized - Log file: {_log_file_path}")


def get_logger(name: str) -> CustomLogger:
    """
    Get a configured logger instance with custom level methods.

    Args:
        name: Logger name (e.g., 'RAG_System', 'Chatbot')

    Returns:
        CustomLogger instance with major() and minor() methods
    """
    return logging.getLogger(name)


def get_log_file_path() -> Path:
    """Return the current log file path"""
    return _log_file_path


# Create default config file if not exists
def create_default_config():
    """Create default log_config.json if not exists"""
    config_path = Path(__file__).parent / "log_config.json"

    if not config_path.exists():
        default_config = {
            "log_level": "MINOR",
            "console_level": "MAJOR",
            "file_level": "DEBUG",
            "max_size_mb": 1024,
            "log_dir": "logs",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }

        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)


# Auto-create config on module import
create_default_config()
