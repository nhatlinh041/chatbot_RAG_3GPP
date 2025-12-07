Centralized Logging System
==========================

Overview
--------
Custom logging system with 5 levels, automatic log rotation, and centralized configuration.
All components log to a single file (logs/app.log).


Log Levels (high to low priority)
---------------------------------
CRITICAL (50) - System failures, unrecoverable errors
ERROR (40)    - Operation failures, exceptions
MAJOR (30)    - Important events, successful operations (replaces INFO)
MINOR (20)    - Detailed operation info, request handling (replaces INFO)
DEBUG (10)    - Verbose debugging info


Configuration
-------------
Edit log_config.json to customize:

{
  "log_level": "MINOR",        // Minimum level for loggers
  "console_level": "MAJOR",    // Minimum level for console output
  "file_level": "DEBUG",       // Minimum level for file output
  "max_size_mb": 1024,         // Max log file size (1GB), then rotate
  "log_dir": "logs",           // Log directory
  "format": "...",             // File log format
  "console_format": "..."      // Console log format
}


Log Rotation
------------
- New log file created on each application start
- Old logs renamed with timestamp: app_20251207_123456.log
- Rotates when file exceeds max_size_mb (default 1GB)
- Keeps up to 10 backup files


Usage
-----
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG

setup_centralized_logging()
logger = get_logger('ComponentName')

logger.log(CRITICAL, "System failure")
logger.log(ERROR, "Operation failed")
logger.log(MAJOR, "Request completed successfully")
logger.log(MINOR, "Processing request...")
logger.log(DEBUG, "Variable value: x=123")


Component Loggers
-----------------
- RAG_System: Main RAG orchestrator
- Chatbot: Django chat views
- Knowledge_Retriever: Neo4j query execution
- LLM_Integrator: Claude/Ollama API calls
- Cypher_Generator: Query generation
- Document_Processing: Document parsing


Log Output Location
-------------------
logs/app.log


Workflow
--------

```
Application Start
      |
      v
setup_centralized_logging()
      |
      v
Check _initialized flag
      |
      +-- Already init --> Return
      |
      v
Load log_config.json
      |
      v
Create logs/ directory
      |
      v
Rotate existing app.log (add timestamp)
      |
      v
Create handlers (file + console)
      |
      v
Configure component loggers
      |
      v
Set _initialized = True
```


Files Changed
-------------
- logging_config.py: Complete rewrite with custom levels
- log_config.json: New config file (auto-created)
- rag_system_v2.py: Updated to use custom levels
- chatbot/views.py: Updated to use custom levels
- tests/test_logging_config.py: Updated for new levels
