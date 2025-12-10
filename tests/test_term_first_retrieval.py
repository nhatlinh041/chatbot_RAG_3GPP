"""
Tests for Term-First Retrieval Strategy.
Verifies that term definitions are resolved correctly based on query intent.
"""
import pytest
from neo4j import GraphDatabase
from hybrid_retriever import (
    TermDefinitionResolver,
    SemanticQueryAnalyzer,
    HybridRetriever
)
from rag_core import CypherQueryGenerator


class TestTermDefinitionResolver:
    """Test TermDefinitionResolver component"""

    @pytest.fixture
    def neo4j_driver(self):
        """Create Neo4j driver for testing"""
        driver = GraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "password")
        )
        yield driver
        driver.close()

    @pytest.fixture
    def resolver(self, neo4j_driver):
        """Create TermDefinitionResolver instance"""
        return TermDefinitionResolver(neo4j_driver)

    def test_resolve_single_term(self, resolver):
        """Test resolving a single abbreviation"""
        result = resolver.resolve_terms(['AMF'])

        assert 'AMF' in result
        assert result['AMF']['full_name'] == 'Access and Mobility Management Function'
        assert result['AMF']['source'] == 'term_node'
        assert isinstance(result['AMF']['specs'], list)

    def test_resolve_multiple_terms(self, resolver):
        """Test resolving multiple abbreviations"""
        result = resolver.resolve_terms(['SCP', 'SEPP'])

        assert 'SCP' in result
        assert 'SEPP' in result
        assert result['SCP']['full_name'] == 'Service Communication Proxy'
        assert result['SEPP']['full_name'] == 'Security Edge Protection Proxy'

    def test_resolve_sepp_correct_definition(self, resolver):
        """Test that SEPP has correct definition (regression test)"""
        result = resolver.resolve_terms(['SEPP'])

        assert 'SEPP' in result
        # Must NOT have "and" or be missing "Edge"
        full_name = result['SEPP']['full_name']
        assert full_name == 'Security Edge Protection Proxy'
        assert 'and Edge' not in full_name  # No "Security and Edge"
        assert 'Protection Edge' not in full_name  # No "Security Protection Edge"

    def test_resolve_unknown_term(self, resolver):
        """Test resolving non-existent abbreviation"""
        result = resolver.resolve_terms(['UNKNOWN_TERM'])

        assert 'UNKNOWN_TERM' not in result

    def test_resolve_empty_list(self, resolver):
        """Test resolving empty entity list"""
        result = resolver.resolve_terms([])

        assert result == {}

    def test_resolve_mixed_case(self, resolver):
        """Test that resolver handles mixed case correctly"""
        result = resolver.resolve_terms(['amf', 'Smf', 'UPF'])

        # Should resolve regardless of case
        assert 'amf' in result or 'AMF' in result
        assert 'Smf' in result or 'SMF' in result
        assert 'UPF' in result


class TestSemanticQueryAnalyzer:
    """Test SemanticQueryAnalyzer with term resolution decision"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with minimal term dict"""
        term_dict = {
            'AMF': {'full_name': 'Access and Mobility Management Function', 'type': 'network_function'},
            'SMF': {'full_name': 'Session Management Function', 'type': 'network_function'},
            'SCP': {'full_name': 'Service Communication Proxy', 'type': 'network_function'},
            'SEPP': {'full_name': 'Security Edge Protection Proxy', 'type': 'network_function'},
        }
        return SemanticQueryAnalyzer(term_dict=term_dict)

    def test_definition_query_needs_resolution(self, analyzer):
        """Definition questions should need term resolution"""
        analysis = analyzer._fallback_analysis("What is AMF?")

        assert analysis['primary_intent'] == 'definition'
        assert 'AMF' in analysis['entities']
        assert analysis['needs_term_resolution'] is True

    def test_comparison_query_needs_resolution(self, analyzer):
        """Comparison questions should need term resolution"""
        analysis = analyzer._fallback_analysis("Compare SCP and SEPP")

        assert analysis['primary_intent'] == 'comparison'
        assert 'SCP' in analysis['entities']
        assert 'SEPP' in analysis['entities']
        assert analysis['needs_term_resolution'] is True

    def test_network_function_query_needs_resolution(self, analyzer):
        """Network function questions should need term resolution"""
        analysis = analyzer._fallback_analysis("What is the role of AMF?")

        # Can be classified as 'definition' or 'network_function' - both need resolution
        assert analysis['primary_intent'] in ['definition', 'network_function']
        assert 'AMF' in analysis['entities']
        assert analysis['needs_term_resolution'] is True

    def test_procedure_query_no_resolution(self, analyzer):
        """Procedure questions should NOT need term resolution"""
        analysis = analyzer._fallback_analysis("Explain registration procedure")

        assert analysis['primary_intent'] == 'procedure'
        assert analysis['needs_term_resolution'] is False

    def test_general_query_no_resolution(self, analyzer):
        """General questions should NOT need term resolution"""
        analysis = analyzer._fallback_analysis("How does network slicing work?")

        # Should not need term resolution for general/procedure questions
        assert analysis['needs_term_resolution'] is False

    def test_stand_for_pattern_recognized(self, analyzer):
        """'Stand for' pattern should be recognized as definition query"""
        analysis = analyzer._fallback_analysis("What does AMF stand for?")

        assert analysis['primary_intent'] == 'definition'
        assert 'AMF' in analysis['entities']
        assert analysis['needs_term_resolution'] is True

    def test_what_does_pattern_recognized(self, analyzer):
        """'What does' pattern should be recognized as definition query"""
        analysis = analyzer._fallback_analysis("What does SEPP stand for in 5G security?")

        assert analysis['primary_intent'] == 'definition'
        assert 'SEPP' in analysis['entities']
        assert analysis['needs_term_resolution'] is True

    def test_common_word_filtering(self, analyzer):
        """Common words like IN, IS should be filtered from entities"""
        analysis = analyzer._fallback_analysis("Compare SCP and SEPP in terms of their roles")

        # Should have SCP and SEPP, but NOT "IN" from "in terms of"
        assert 'SCP' in analysis['entities']
        assert 'SEPP' in analysis['entities']
        assert 'IN' not in analysis['entities']

    def test_procedure_vs_definition_disambiguation(self, analyzer):
        """'What is the first step' should be procedure, not definition"""
        analysis = analyzer._fallback_analysis("What is the first step in UE registration?")

        # Should be classified as procedure since it contains "step"
        assert analysis['primary_intent'] == 'procedure'
        # UE might still be extracted as entity
        # But no term resolution needed for procedures
        assert analysis['needs_term_resolution'] is False


class TestHybridRetrieverIntegration:
    """Integration tests for HybridRetriever with term resolution"""

    @pytest.fixture
    def neo4j_driver(self):
        """Create Neo4j driver"""
        driver = GraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "password")
        )
        yield driver
        driver.close()

    @pytest.fixture
    def retriever(self, neo4j_driver):
        """Create HybridRetriever with all components"""
        cypher_gen = CypherQueryGenerator(
            neo4j_driver=neo4j_driver,
            local_llm_url="http://192.168.1.14:11434/api/chat"
        )
        return HybridRetriever(
            neo4j_driver=neo4j_driver,
            cypher_generator=cypher_gen,
            local_llm_url="http://192.168.1.14:11434/api/chat"
        )

    def test_comparison_resolves_terms(self, retriever):
        """Test that comparison queries resolve term definitions"""
        chunks, strategy, analysis = retriever.retrieve(
            "Compare SCP and SEPP",
            use_vector=False,
            use_graph=True,
            use_query_expansion=False
        )

        # Check analysis
        assert analysis['primary_intent'] == 'comparison'
        assert analysis['needs_term_resolution'] is True

        # Check term definitions were resolved
        term_defs = analysis.get('term_definitions', {})
        assert 'SCP' in term_defs
        assert 'SEPP' in term_defs
        assert term_defs['SCP']['full_name'] == 'Service Communication Proxy'
        assert term_defs['SEPP']['full_name'] == 'Security Edge Protection Proxy'

    def test_definition_resolves_terms(self, retriever):
        """Test that definition queries resolve term definitions"""
        chunks, strategy, analysis = retriever.retrieve(
            "What is AMF?",
            use_vector=False,
            use_graph=True,
            use_query_expansion=False
        )

        # Check analysis
        assert analysis['primary_intent'] == 'definition'
        assert analysis['needs_term_resolution'] is True

        # Check term definitions
        term_defs = analysis.get('term_definitions', {})
        assert 'AMF' in term_defs
        assert term_defs['AMF']['full_name'] == 'Access and Mobility Management Function'

    def test_procedure_skips_resolution(self, retriever):
        """Test that procedure queries skip term resolution"""
        chunks, strategy, analysis = retriever.retrieve(
            "Explain registration procedure",
            use_vector=False,
            use_graph=True,
            use_query_expansion=False
        )

        # Check analysis
        assert analysis['primary_intent'] == 'procedure'
        assert analysis['needs_term_resolution'] is False

        # Check no term definitions resolved
        term_defs = analysis.get('term_definitions', {})
        assert term_defs == {}

    def test_sepp_definition_is_correct(self, retriever):
        """Regression test: Ensure SEPP definition is correct"""
        chunks, strategy, analysis = retriever.retrieve(
            "What is SEPP?",
            use_vector=False,
            use_graph=True
        )

        term_defs = analysis.get('term_definitions', {})
        assert 'SEPP' in term_defs

        full_name = term_defs['SEPP']['full_name']
        # Must be exactly "Security Edge Protection Proxy"
        assert full_name == 'Security Edge Protection Proxy'
        # Must NOT contain errors
        assert 'and Edge' not in full_name  # No "Security and Edge Protection Proxy"
        assert 'Protection Edge' not in full_name  # No "Security Protection Edge Proxy"


class TestPromptTemplateIntegration:
    """Test that term definitions are properly injected into prompts"""

    def test_comparison_prompt_includes_definitions(self):
        """Test comparison prompt includes authoritative definitions"""
        from prompt_templates import PromptTemplates

        term_defs = {
            'SCP': {
                'full_name': 'Service Communication Proxy',
                'specs': ['TS 23.501'],
                'source': 'term_node'
            },
            'SEPP': {
                'full_name': 'Security Edge Protection Proxy',
                'specs': ['TS 33.501'],
                'source': 'term_node'
            }
        }

        analysis = {
            'primary_intent': 'comparison',
            'entities': ['SCP', 'SEPP'],
            'term_definitions': term_defs
        }

        prompt = PromptTemplates.get_prompt(
            query="Compare SCP and SEPP",
            context="[mock context]",
            analysis=analysis
        )

        # Check that authoritative definitions section exists
        assert 'AUTHORITATIVE DEFINITIONS' in prompt
        assert 'Service Communication Proxy' in prompt
        assert 'Security Edge Protection Proxy' in prompt
        assert 'TS 23.501' in prompt
        assert 'TS 33.501' in prompt

        # Check critical rules mention authoritative definitions
        assert 'TRUST the authoritative definitions' in prompt or \
               'PRIORITIZE authoritative definitions' in prompt

    def test_definition_prompt_includes_definitions(self):
        """Test definition prompt includes authoritative definitions"""
        from prompt_templates import PromptTemplates

        term_defs = {
            'AMF': {
                'full_name': 'Access and Mobility Management Function',
                'specs': ['TS 23.501', 'TS 29.518'],
                'source': 'term_node'
            }
        }

        analysis = {
            'primary_intent': 'definition',
            'entities': ['AMF'],
            'term_definitions': term_defs
        }

        prompt = PromptTemplates.get_prompt(
            query="What is AMF?",
            context="[mock context]",
            analysis=analysis
        )

        # Check authoritative definitions
        assert 'AUTHORITATIVE DEFINITIONS' in prompt
        assert 'Access and Mobility Management Function' in prompt
        assert 'TS 23.501' in prompt

    def test_procedure_prompt_no_definitions(self):
        """Test procedure prompt when no term definitions needed"""
        from prompt_templates import PromptTemplates

        analysis = {
            'primary_intent': 'procedure',
            'entities': [],
            'term_definitions': {}
        }

        prompt = PromptTemplates.get_prompt(
            query="Explain registration procedure",
            context="[mock context]",
            analysis=analysis
        )

        # Should not have authoritative definitions section
        # (or it should be empty)
        # The prompt should still work without it
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestEndToEndScenarios:
    """End-to-end tests for common query scenarios"""

    @pytest.fixture
    def neo4j_driver(self):
        """Create Neo4j driver"""
        driver = GraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "password")
        )
        yield driver
        driver.close()

    @pytest.fixture
    def retriever(self, neo4j_driver):
        """Create HybridRetriever"""
        cypher_gen = CypherQueryGenerator(
            neo4j_driver=neo4j_driver,
            local_llm_url="http://192.168.1.14:11434/api/chat"
        )
        return HybridRetriever(
            neo4j_driver=neo4j_driver,
            cypher_generator=cypher_gen
        )

    def test_scenario_compare_scp_sepp(self, retriever):
        """Full scenario: Compare SCP and SEPP"""
        chunks, strategy, analysis = retriever.retrieve(
            "Compare SCP and SEPP",
            use_vector=False,
            use_graph=True
        )

        # Verify term resolution happened
        assert analysis['needs_term_resolution'] is True
        term_defs = analysis['term_definitions']
        assert len(term_defs) == 2

        # Verify correct definitions
        assert term_defs['SCP']['full_name'] == 'Service Communication Proxy'
        assert term_defs['SEPP']['full_name'] == 'Security Edge Protection Proxy'

        # Verify chunks retrieved
        assert len(chunks) > 0

    def test_scenario_what_is_amf(self, retriever):
        """Full scenario: What is AMF?"""
        chunks, strategy, analysis = retriever.retrieve(
            "What is AMF?",
            use_vector=False,
            use_graph=True
        )

        # Verify term resolution
        assert analysis['needs_term_resolution'] is True
        assert 'AMF' in analysis['term_definitions']
        assert analysis['term_definitions']['AMF']['full_name'] == \
            'Access and Mobility Management Function'

    def test_scenario_registration_procedure(self, retriever):
        """Full scenario: Explain registration procedure"""
        chunks, strategy, analysis = retriever.retrieve(
            "Explain registration procedure",
            use_vector=False,
            use_graph=True
        )

        # Verify NO term resolution
        assert analysis['needs_term_resolution'] is False
        assert analysis['term_definitions'] == {}

        # Should still retrieve chunks
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
