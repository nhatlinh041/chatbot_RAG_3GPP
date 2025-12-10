"""
Tests for prompt_templates.py module.
Tests PromptTemplates and ContextBuilder classes.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
from dataclasses import dataclass
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_templates import PromptTemplates, ContextBuilder


# ============================================================
# Mock Chunk Classes
# ============================================================
@dataclass
class MockChunk:
    """Mock chunk for testing"""
    chunk_id: str
    spec_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    complexity_score: float = 0.5
    key_terms: List[str] = None

    def __post_init__(self):
        if self.key_terms is None:
            self.key_terms = []


# ============================================================
# ContextBuilder Tests
# ============================================================
class TestContextBuilder:
    """Tests for ContextBuilder class"""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        return [
            MockChunk(
                chunk_id="chunk_001",
                spec_id="ts_23.501",
                section_id="5.2.1",
                section_title="AMF Overview",
                content="The AMF handles mobility management and registration.",
                chunk_type="definition"
            ),
            MockChunk(
                chunk_id="chunk_002",
                spec_id="ts_23.501",
                section_id="5.2.2",
                section_title="SMF Overview",
                content="The SMF handles session management.",
                chunk_type="definition"
            )
        ]

    def test_build_context_basic(self, sample_chunks):
        """Test basic context building"""
        context = ContextBuilder.build_context(sample_chunks)

        assert "ts_23.501" in context
        assert "AMF Overview" in context
        assert "SMF Overview" in context
        assert "mobility management" in context

    def test_build_context_with_metadata(self, sample_chunks):
        """Test context building with metadata"""
        context = ContextBuilder.build_context(sample_chunks, include_metadata=True)

        assert "Source:" in context
        assert "Section:" in context
        assert "definition" in context

    def test_build_context_without_metadata(self, sample_chunks):
        """Test context building without metadata"""
        context = ContextBuilder.build_context(sample_chunks, include_metadata=False)

        assert "Source:" not in context
        assert "mobility management" in context

    def test_build_context_max_chars(self):
        """Test context size limit"""
        # Create chunks with long content
        long_chunks = [
            MockChunk(
                chunk_id="chunk_001",
                spec_id="ts_23.501",
                section_id="1",
                section_title="Test",
                content="A" * 5000,  # Long content
                chunk_type="general"
            )
            for _ in range(10)
        ]

        context = ContextBuilder.build_context(long_chunks, max_chars=1000)

        # Should be truncated
        assert len(context) <= 2000  # Some overhead for metadata

    def test_build_context_truncates_long_content(self):
        """Test that long content is truncated"""
        chunks = [
            MockChunk(
                chunk_id="chunk_001",
                spec_id="ts_23.501",
                section_id="1",
                section_title="Test",
                content="A" * 3000,  # > 1500 chars
                chunk_type="general"
            )
        ]

        context = ContextBuilder.build_context(chunks)

        assert "[truncated]" in context

    def test_build_context_with_labels(self, sample_chunks):
        """Test context building with entity labels"""
        labels = {
            "chunk_001": "AMF",
            "chunk_002": "SMF"
        }

        context = ContextBuilder.build_context_with_labels(sample_chunks, labels)

        assert "[AMF]" in context
        assert "[SMF]" in context

    def test_build_context_empty_chunks(self):
        """Test context building with empty chunks list"""
        context = ContextBuilder.build_context([])

        assert context == ""


# ============================================================
# PromptTemplates Tests
# ============================================================
class TestPromptTemplates:
    """Tests for PromptTemplates class"""

    @pytest.fixture
    def sample_context(self):
        """Sample context string"""
        return """
**Source: ts_23.501 - AMF Overview** (Type: definition)
**Section: 5.2.1**

The AMF handles mobility management and registration procedures.

---
"""

    def test_get_prompt_definition(self, sample_context):
        """Test definition prompt generation"""
        analysis = {
            'primary_intent': 'definition',
            'entities': ['AMF']
        }

        prompt = PromptTemplates.get_prompt(
            query="What is AMF?",
            context=sample_context,
            analysis=analysis
        )

        assert "What is AMF?" in prompt
        assert "Definition" in prompt
        assert "Anti-Hallucination" in prompt

    def test_get_prompt_comparison(self, sample_context):
        """Test comparison prompt generation"""
        analysis = {
            'primary_intent': 'comparison',
            'entities': ['AMF', 'SMF']
        }

        prompt = PromptTemplates.get_prompt(
            query="Compare AMF and SMF",
            context=sample_context,
            analysis=analysis
        )

        assert "Compare" in prompt
        assert "AMF" in prompt
        assert "SMF" in prompt
        assert "Differences" in prompt or "difference" in prompt.lower()

    def test_get_prompt_procedure(self, sample_context):
        """Test procedure prompt generation"""
        analysis = {
            'primary_intent': 'procedure',
            'entities': []
        }

        prompt = PromptTemplates.get_prompt(
            query="How does registration work?",
            context=sample_context,
            analysis=analysis
        )

        assert "Step" in prompt or "step" in prompt.lower()
        assert "procedure" in prompt.lower()

    def test_get_prompt_network_function(self, sample_context):
        """Test network function prompt generation"""
        analysis = {
            'primary_intent': 'network_function',
            'entities': ['AMF']
        }

        prompt = PromptTemplates.get_prompt(
            query="What is the role of AMF?",
            context=sample_context,
            analysis=analysis
        )

        assert "role" in prompt.lower() or "function" in prompt.lower()
        assert "AMF" in prompt

    def test_get_prompt_relationship(self, sample_context):
        """Test relationship prompt generation"""
        analysis = {
            'primary_intent': 'relationship',
            'entities': ['AMF', 'SMF']
        }

        prompt = PromptTemplates.get_prompt(
            query="How do AMF and SMF interact?",
            context=sample_context,
            analysis=analysis
        )

        assert "Relationship" in prompt or "relationship" in prompt.lower()
        assert "Interaction" in prompt or "interaction" in prompt.lower()

    def test_get_prompt_multiple_choice(self, sample_context):
        """Test multiple choice prompt generation"""
        analysis = {
            'primary_intent': 'multiple_choice',
            'entities': ['AMF']
        }

        prompt = PromptTemplates.get_prompt(
            query="Which statement about AMF is correct?",
            context=sample_context,
            analysis=analysis
        )

        assert "multiple choice" in prompt.lower()
        assert "Answer" in prompt

    def test_get_prompt_multi_intent(self, sample_context):
        """Test multi-intent prompt generation"""
        analysis = {
            'primary_intent': 'comparison',
            'entities': ['AMF', 'SMF'],
            'requires_multi_step': True,
            'sub_questions': ['What is AMF?', 'What is SMF?']
        }

        prompt = PromptTemplates.get_prompt(
            query="Compare AMF and SMF, explain their roles",
            context=sample_context,
            analysis=analysis
        )

        # Multi-intent with comparison uses comparison template
        assert "AMF" in prompt and "SMF" in prompt

    def test_get_prompt_general(self, sample_context):
        """Test general prompt generation"""
        analysis = {
            'primary_intent': 'general',
            'entities': []
        }

        prompt = PromptTemplates.get_prompt(
            query="Tell me about 5G",
            context=sample_context,
            analysis=analysis
        )

        assert "comprehensive" in prompt.lower()
        assert "Anti-Hallucination" in prompt

    def test_get_prompt_without_analysis(self, sample_context):
        """Test prompt generation without analysis"""
        prompt = PromptTemplates.get_prompt(
            query="What is AMF?",
            context=sample_context,
            analysis=None
        )

        # Should use general prompt
        assert "comprehensive" in prompt.lower()

    def test_get_definition_prompt_direct(self, sample_context):
        """Test direct call to definition prompt"""
        prompt = PromptTemplates.get_definition_prompt(
            query="What is AMF?",
            context=sample_context,
            entities=['AMF']
        )

        assert "Definition" in prompt
        assert "Key Characteristics" in prompt
        assert "Specification Reference" in prompt

    def test_get_comparison_prompt_direct(self, sample_context):
        """Test direct call to comparison prompt"""
        prompt = PromptTemplates.get_comparison_prompt(
            query="Compare AMF and SMF",
            context=sample_context,
            entities=['AMF', 'SMF']
        )

        assert "Comparison" in prompt
        assert "Key Differences" in prompt
        assert "Similarities" in prompt
        assert "AMF" in prompt
        assert "SMF" in prompt

    def test_get_procedure_prompt_direct(self, sample_context):
        """Test direct call to procedure prompt"""
        prompt = PromptTemplates.get_procedure_prompt(
            query="How does registration work?",
            context=sample_context
        )

        assert "Step-by-Step" in prompt or "step" in prompt.lower()
        assert "Procedure" in prompt
        assert "mermaid" in prompt.lower()

    def test_get_multiple_choice_prompt_direct(self, sample_context):
        """Test direct call to multiple choice prompt"""
        prompt = PromptTemplates.get_multiple_choice_prompt(
            query="Which is correct about AMF?",
            context=sample_context
        )

        assert "multiple choice" in prompt.lower()
        # Check for new MCQ prompt format requirements
        assert "Answer:" in prompt  # New format instruction
        assert "FIRST LINE MUST BE" in prompt  # Format enforcement
        assert "brief explanation" in prompt.lower()  # Explanation instruction

    def test_prompts_include_context(self, sample_context):
        """Test that all prompts include the context"""
        intents = ['definition', 'comparison', 'procedure', 'general']

        for intent in intents:
            analysis = {'primary_intent': intent, 'entities': ['AMF', 'SMF']}
            prompt = PromptTemplates.get_prompt(
                query="Test query",
                context=sample_context,
                analysis=analysis
            )

            assert sample_context in prompt or "ts_23.501" in prompt

    def test_prompts_include_anti_hallucination(self, sample_context):
        """Test that key prompts include anti-hallucination rules"""
        intents_with_anti_hallucination = ['definition', 'general', 'multiple_choice']

        for intent in intents_with_anti_hallucination:
            analysis = {'primary_intent': intent, 'entities': ['AMF']}
            prompt = PromptTemplates.get_prompt(
                query="Test query",
                context=sample_context,
                analysis=analysis
            )

            # Check for anti-hallucination markers
            has_rule = (
                "hallucination" in prompt.lower() or
                "only" in prompt.lower() or
                "context" in prompt.lower()
            )
            assert has_rule


# ============================================================
# Edge Cases
# ============================================================
class TestPromptTemplatesEdgeCases:
    """Test edge cases for prompt templates"""

    def test_empty_entities(self):
        """Test prompts with empty entities"""
        prompt = PromptTemplates.get_definition_prompt(
            query="What is X?",
            context="Some context",
            entities=[]
        )

        # Should not crash
        assert "What is X?" in prompt

    def test_single_entity_comparison(self):
        """Test comparison with single entity"""
        prompt = PromptTemplates.get_comparison_prompt(
            query="Compare AMF",
            context="Some context",
            entities=['AMF']
        )

        # Should handle gracefully
        assert "AMF" in prompt

    def test_none_entities(self):
        """Test with None entities"""
        prompt = PromptTemplates.get_definition_prompt(
            query="What is AMF?",
            context="Some context",
            entities=None
        )

        assert prompt is not None

    def test_empty_context(self):
        """Test with empty context"""
        prompt = PromptTemplates.get_general_prompt(
            query="What is AMF?",
            context=""
        )

        assert "What is AMF?" in prompt

    def test_very_long_query(self):
        """Test with very long query"""
        long_query = "What is " + "AMF " * 100 + "?"
        prompt = PromptTemplates.get_definition_prompt(
            query=long_query,
            context="Some context",
            entities=['AMF']
        )

        assert prompt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
