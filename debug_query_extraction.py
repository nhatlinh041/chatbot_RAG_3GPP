"""
Debug script to extract all Cypher queries generated for a question.
Allows testing queries directly in Neo4j browser.

Usage:
    python debug_query_extraction.py "What is the difference between SCP and SEPP?"
    python debug_query_extraction.py "What is AMF?" --save-to-file
"""

import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from rag_system_v2 import create_rag_system_v2
from logging_config import setup_centralized_logging, get_logger, MAJOR, MINOR, DEBUG

# Setup logging
setup_centralized_logging()
logger = get_logger('Query_Debugger')


class QueryDebugger:
    """Captures and displays all Cypher queries for debugging."""

    def __init__(self, claude_api_key: str = None, deepseek_api_url: str = None):
        """
        Initialize the query debugger.

        Args:
            claude_api_key: Anthropic API key for Claude
            deepseek_api_url: URL for DeepSeek/Ollama endpoint
        """
        logger.log(MAJOR, "Initializing Query Debugger")

        # Create RAG system
        self.rag = create_rag_system_v2(
            claude_api_key=claude_api_key,
            deepseek_api_url=deepseek_api_url
        )

        # Store captured queries
        self.queries = []

    def extract_queries(self, question: str) -> List[Dict[str, Any]]:
        """
        Extract all Cypher queries generated for a question.

        Args:
            question: User question

        Returns:
            List of query information dictionaries
        """
        logger.log(MAJOR, f"Extracting queries for: {question}")

        self.queries = []

        # Intercept session.run() to capture queries
        original_driver = self.rag.retriever.driver
        original_session_class = original_driver.session

        def patched_session(*args, **kwargs):
            """Create a session with patched run() method"""
            session = original_session_class(*args, **kwargs)
            original_run = session.run

            def patched_run(cypher_query, parameters=None, **run_kwargs):
                """Intercept and log query execution"""
                query_info = {
                    'query': cypher_query,
                    'parameters': parameters or {},
                    'query_type': self._identify_query_type(cypher_query)
                }
                self.queries.append(query_info)
                logger.log(DEBUG, f"Captured query: {query_info['query_type']}")

                # Call original run method
                return original_run(cypher_query, parameters, **run_kwargs)

            session.run = patched_run
            return session

        # Temporarily replace session method
        self.rag.retriever.driver.session = patched_session

        try:
            # Trigger query generation by calling retrieve_with_cypher
            logger.log(MINOR, "Triggering query generation...")
            self.rag.retriever.retrieve_with_cypher(question)

        except Exception as e:
            logger.log(MAJOR, f"Error during query extraction: {str(e)}")

        finally:
            # Restore original session method
            self.rag.retriever.driver.session = original_session_class

        logger.log(MAJOR, f"Extracted {len(self.queries)} queries")
        return self.queries

    def _identify_query_type(self, cypher_query: str) -> str:
        """Identify the type of Cypher query based on content."""
        query_lower = cypher_query.lower()

        if 'term' in query_lower and 'abbreviation' in query_lower:
            return 'Term Lookup'
        elif 'chunk' in query_lower and 'content' in query_lower:
            return 'Content Search'
        elif 'references_chunk' in query_lower:
            return 'Reference Traversal'
        elif 'document' in query_lower:
            return 'Document Metadata'
        else:
            return 'Custom Query'

    def format_query_for_neo4j(self, query_info: Dict[str, Any]) -> str:
        """
        Format a query with parameters for direct execution in Neo4j browser.

        Args:
            query_info: Dictionary containing query and parameters

        Returns:
            Formatted Cypher query string
        """
        query = query_info['query']
        params = query_info['parameters']

        if not params:
            return query

        # Replace parameter placeholders with actual values
        formatted_query = query
        for key, value in params.items():
            placeholder = f"${key}"

            # Format value based on type
            if isinstance(value, str):
                formatted_value = f"'{value}'"
            elif isinstance(value, (int, float)):
                formatted_value = str(value)
            elif isinstance(value, list):
                formatted_value = str(value).replace("'", '"')
            else:
                formatted_value = str(value)

            formatted_query = formatted_query.replace(placeholder, formatted_value)

        return formatted_query

    def display_queries(self, queries: List[Dict[str, Any]], save_to_file: str = None):
        """
        Display all queries in a readable format.

        Args:
            queries: List of query information dictionaries
            save_to_file: Optional filename to save queries
        """
        output_lines = []

        output_lines.append("=" * 80)
        output_lines.append(f"EXTRACTED {len(queries)} CYPHER QUERIES")
        output_lines.append("=" * 80)
        output_lines.append("")

        for idx, query_info in enumerate(queries, 1):
            output_lines.append(f"Query #{idx}: {query_info['query_type']}")
            output_lines.append("-" * 80)

            # Original query with parameters
            output_lines.append("\n// Original Query (with parameters):")
            output_lines.append(query_info['query'])

            if query_info['parameters']:
                output_lines.append("\n// Parameters:")
                for key, value in query_info['parameters'].items():
                    output_lines.append(f"//   ${key} = {value}")

            # Formatted query for Neo4j browser
            output_lines.append("\n// Ready for Neo4j Browser (parameters substituted):")
            formatted = self.format_query_for_neo4j(query_info)
            output_lines.append(formatted)

            output_lines.append("\n" + "=" * 80)
            output_lines.append("")

        output_text = "\n".join(output_lines)

        # Print to console
        print(output_text)

        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            logger.log(MAJOR, f"Queries saved to: {save_to_file}")
            print(f"\n✓ Queries saved to: {save_to_file}")


def main():
    """Main entry point for the debug script."""
    parser = argparse.ArgumentParser(
        description='Extract Cypher queries for debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_query_extraction.py "What is AMF?"
  python debug_query_extraction.py "What is the difference between SCP and SEPP?" --save-to-file
  python debug_query_extraction.py "What is AMF?" --output queries.txt
        """
    )

    parser.add_argument(
        'question',
        type=str,
        help='Question to analyze'
    )

    parser.add_argument(
        '--save-to-file',
        action='store_true',
        help='Save queries to a file (auto-generated filename)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Specify output filename'
    )

    parser.add_argument(
        '--claude-key',
        type=str,
        default=None,
        help='Claude API key (defaults to CLAUDE_API_KEY env var)'
    )

    parser.add_argument(
        '--deepseek-url',
        type=str,
        default=None,
        help='DeepSeek API URL (defaults to DEEPSEEK_API_URL env var)'
    )

    args = parser.parse_args()

    # Get API credentials
    claude_key = args.claude_key or os.getenv('CLAUDE_API_KEY')
    deepseek_url = args.deepseek_url or os.getenv('DEEPSEEK_API_URL', 'http://192.168.1.14:11434/api/chat')

    if not claude_key:
        print("Warning: CLAUDE_API_KEY not found. Some features may not work.")

    # Initialize debugger
    debugger = QueryDebugger(
        claude_api_key=claude_key,
        deepseek_api_url=deepseek_url
    )

    # Extract queries
    queries = debugger.extract_queries(args.question)

    if not queries:
        print("No queries were generated for this question.")
        return

    # Determine output filename
    output_file = None
    if args.output:
        output_file = args.output
    elif args.save_to_file:
        # Auto-generate filename
        safe_question = "".join(c if c.isalnum() else "_" for c in args.question[:50])
        output_file = f"debug_queries_{safe_question}.txt"

    # Display queries
    debugger.display_queries(queries, save_to_file=output_file)


if __name__ == '__main__':
    main()
