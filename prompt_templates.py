"""
Enhanced Prompt Templates for 3GPP RAG System.
Provides specialized prompts for different question types.
"""
from typing import List, Dict, Optional
from logging_config import get_logger, MINOR

logger = get_logger('Prompt_Templates')


class PromptTemplates:
    """
    Enhanced prompt templates for different question intents.
    Each template is optimized for specific question types.
    """

    @staticmethod
    def get_prompt(query: str, context: str, analysis: Dict = None,
                   entities: List[str] = None) -> str:
        """
        Get appropriate prompt based on query analysis.

        Args:
            query: User's question
            context: Retrieved context from chunks
            analysis: Query analysis dict with intent, entities, etc.
            entities: List of entities (fallback if not in analysis)

        Returns:
            Formatted prompt for LLM
        """
        if analysis is None:
            analysis = {}

        intent = analysis.get('primary_intent', 'general')
        detected_entities = analysis.get('entities', entities or [])

        # Select appropriate template
        if intent == 'definition':
            return PromptTemplates.get_definition_prompt(query, context, detected_entities)
        elif intent == 'comparison':
            return PromptTemplates.get_comparison_prompt(query, context, detected_entities)
        elif intent == 'procedure':
            return PromptTemplates.get_procedure_prompt(query, context)
        elif intent == 'network_function':
            return PromptTemplates.get_network_function_prompt(query, context, detected_entities)
        elif intent == 'relationship':
            return PromptTemplates.get_relationship_prompt(query, context, detected_entities)
        elif intent == 'multiple_choice':
            return PromptTemplates.get_multiple_choice_prompt(query, context)
        elif analysis.get('requires_multi_step', False):
            return PromptTemplates.get_multi_intent_prompt(query, context, analysis)
        else:
            return PromptTemplates.get_general_prompt(query, context)

    @staticmethod
    def get_definition_prompt(query: str, context: str, entities: List[str] = None) -> str:
        """Prompt for definition questions"""
        entities = entities or []
        entity_str = ', '.join(entities) if entities else "the term"

        return f"""You are a 3GPP telecommunications expert. Answer this definition question precisely.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Provide clear, technical definition based ONLY on the context
2. Structure your answer:
   - **Definition**: Clear 1-2 sentence definition
   - **Key Characteristics**: Bullet points of main features/functions
   - **Related Components**: How it relates to other 3GPP entities (if mentioned in context)
   - **Specification Reference**: Which specs define it

3. Use proper 3GPP terminology
4. Bold important terms like **{entities[0] if entities else 'key terms'}**, **5G Core**, etc.
5. If abbreviation, show format: **ABBR** (Full Name)

**CRITICAL Anti-Hallucination Rules:**
- Use ONLY information from the provided context
- If context lacks sufficient info, state: "Based on the provided specifications, I can confirm [what you found]. Additional details about [missing info] are not available in these sections."
- DO NOT use general knowledge - stick to the context
- Cite spec sections (e.g., "According to TS 23.501...")

**Format:** Use Markdown with clear sections and bullet points."""

    @staticmethod
    def get_comparison_prompt(query: str, context: str, entities: List[str] = None) -> str:
        """Prompt for comparison questions"""
        entities = entities or ['Entity1', 'Entity2']
        e1 = entities[0] if len(entities) > 0 else 'Entity1'
        e2 = entities[1] if len(entities) > 1 else 'Entity2'

        return f"""You are a 3GPP telecommunications expert. Compare these entities based on specifications.

**Question:** {query}

**Entities to compare:** {e1} vs {e2}

**Context from 3GPP specifications:**
{context}

**Instructions:**
Structure your answer as follows:

## Comparison: {e1} vs {e2}

### Overview
Brief description of each entity (1-2 sentences each)

### Key Differences

| Aspect | {e1} | {e2} |
|--------|------|------|
| Primary Role | ... | ... |
| Key Functions | ... | ... |
| Interfaces | ... | ... |
| Position in Architecture | ... | ... |

### Similarities
- Common aspects between them (if any)

### Interaction
- How they work together (if applicable)
- Include Mermaid diagram if helpful:

```mermaid
graph LR
    A[{e1}] -->|interaction| B[{e2}]
```

### Specification References
- List relevant spec sections

**CRITICAL Rules:**
- Base comparison ONLY on provided context
- If information missing for one entity, explicitly state it
- Use exact terminology from 3GPP specs
- Cite specific sections for claims"""

    @staticmethod
    def get_procedure_prompt(query: str, context: str) -> str:
        """Prompt for procedure/process questions"""
        return f"""You are a 3GPP telecommunications expert. Explain this procedure step-by-step.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
Structure your answer as follows:

## Procedure Overview
Brief description of what this procedure accomplishes (1-2 sentences)

## Prerequisites
- What must be in place before this procedure (if mentioned)

## Step-by-Step Flow

1. **[Step Name]**
   - Description of what happens
   - Entities involved: **Entity1**, **Entity2**
   - Messages/signals exchanged (if mentioned)

2. **[Step Name]**
   - Continue with each step...

## Message Flow Diagram
If procedure involves multiple entities, include:

```mermaid
sequenceDiagram
    participant A as Entity1
    participant B as Entity2
    A->>B: Message 1
    B->>A: Response
```

## Success Criteria
- How to know procedure completed successfully (if mentioned)

## Specification References
- Cite relevant sections

**CRITICAL Rules:**
- Follow EXACT sequence from specifications
- Don't skip steps mentioned in context
- Don't add steps not in context
- Use proper message/signal names from specs"""

    @staticmethod
    def get_network_function_prompt(query: str, context: str, entities: List[str] = None) -> str:
        """Prompt for network function role/responsibility questions"""
        entities = entities or []
        nf = entities[0] if entities else "the network function"

        return f"""You are a 3GPP telecommunications expert. Explain the role and functions of this network component.

**Question:** {query}

**Network Function:** {nf}

**Context from 3GPP specifications:**
{context}

**Instructions:**
Structure your answer as follows:

## {nf} Overview
- Full name and brief description
- Position in 5G architecture

## Key Functions and Responsibilities
List each major function:
- **Function 1**: Description
- **Function 2**: Description
- (continue for all functions mentioned in context)

## Interfaces
- Which other NFs does it interact with
- Key reference points (e.g., N11, N12)

## Architecture Diagram
If helpful, include:

```mermaid
graph TB
    NF[{nf}] --> A[Connected NF 1]
    NF --> B[Connected NF 2]
```

## Specification References
- Primary specs that define this NF

**CRITICAL Rules:**
- List ONLY functions mentioned in context
- Use exact terminology from specs
- Don't add functions from general knowledge"""

    @staticmethod
    def get_relationship_prompt(query: str, context: str, entities: List[str] = None) -> str:
        """Prompt for relationship/interaction questions"""
        entities = entities or []
        entities_str = ' and '.join(entities) if entities else "the entities"

        return f"""You are a 3GPP telecommunications expert. Explain the relationship between these components.

**Question:** {query}

**Entities involved:** {entities_str}

**Context from 3GPP specifications:**
{context}

**Instructions:**
Structure your answer as follows:

## Relationship Overview
Brief description of how these entities relate

## Interaction Details
- Type of relationship (e.g., service-based, reference point)
- Communication protocol/interface used
- Message flows between them

## Diagram

```mermaid
graph LR
    A[Entity1] -->|interface/protocol| B[Entity2]
```

## Use Cases
- When/why this interaction occurs
- Example scenarios from specs

## Specification References
- Cite relevant sections

**CRITICAL Rules:**
- Describe ONLY relationships mentioned in context
- Use exact interface/protocol names from specs
- If relationship not clear from context, state it"""

    @staticmethod
    def get_multiple_choice_prompt(query: str, context: str) -> str:
        """Prompt for multiple choice questions"""
        return f"""You are a 3GPP telecommunications expert. This is a multiple choice question.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Carefully read each option in the question
2. Find evidence in the context for/against each option
3. Select the CORRECT answer based on specification content
4. Explain your reasoning with specific references

**Response Format:**

## Answer
**[Letter/Option]** - [Brief statement of correct answer]

## Explanation
- Why this answer is correct (cite specific context)
- Why other options are incorrect (briefly)

## Specification Reference
- Section/spec that supports this answer

**CRITICAL Rules:**
- Base answer ONLY on provided context
- If context doesn't clearly support any option, state which option is MOST likely based on available information
- Use exact wording from context when explaining"""

    @staticmethod
    def get_multi_intent_prompt(query: str, context: str, analysis: Dict) -> str:
        """Prompt for complex multi-part questions"""
        sub_questions = analysis.get('sub_questions', [])
        entities = analysis.get('entities', [])

        sub_q_text = ""
        if sub_questions:
            sub_q_text = "\n**Sub-questions identified:**\n"
            for i, sq in enumerate(sub_questions, 1):
                sub_q_text += f"{i}. {sq}\n"

        return f"""You are a 3GPP telecommunications expert. Answer this complex multi-part question.

**Question:** {query}
{sub_q_text}
**Entities involved:** {', '.join(entities) if entities else 'various 3GPP components'}

**Context from 3GPP specifications:**
{context}

**Instructions:**
This is a complex question. Address each part systematically:

## Part 1: [First aspect of the question]
[Answer with context references]

## Part 2: [Second aspect of the question]
[Answer with context references]

## Synthesis
[How the parts relate - the big picture]

## Diagram (if helpful)
```mermaid
[Appropriate diagram type]
```

## Specification References
- List all relevant specs and sections

**CRITICAL Rules:**
- Address ALL parts of the question
- If one part cannot be fully answered, state it clearly
- Maintain technical accuracy
- Base everything on provided context"""

    @staticmethod
    def get_general_prompt(query: str, context: str) -> str:
        """General purpose prompt for unclassified questions"""
        return f"""You are a 3GPP telecommunications expert. Answer this question based on the specifications.

**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. Provide a comprehensive answer based ONLY on the provided context
2. Reference specific sections and specifications when relevant
3. Explain technical terms when helpful
4. Use proper 3GPP terminology

**Response Format:**
- Use headings (## or ###) to structure your answer
- Bold **key terms** and network function names
- Use bullet points for lists
- Use `code` formatting for technical identifiers
- For complex relationships, include a Mermaid diagram
- Include a "## Sources" section with specification references

**CRITICAL Anti-Hallucination Rules:**
- Use ONLY information from the provided context
- If context doesn't contain relevant information, say: "Based on the retrieved specifications, I could not find detailed information about [topic]. The available context covers [what is covered]."
- DO NOT use general knowledge about 3GPP - only the provided context
- If unsure, say so rather than guessing

**Answer:**"""


class ContextBuilder:
    """
    Builds optimized context from retrieved chunks.
    """

    @staticmethod
    def build_context(chunks: List, max_chars: int = 25000,
                     include_metadata: bool = True) -> str:
        """
        Build context string from chunks.

        Args:
            chunks: List of RetrievedChunk or ScoredChunk
            max_chars: Maximum context length
            include_metadata: Include chunk metadata

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0

        for chunk in chunks:
            # Get content (handle both dataclass types)
            content = getattr(chunk, 'content', '')
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"

            if include_metadata:
                spec_id = getattr(chunk, 'spec_id', '')
                section_title = getattr(chunk, 'section_title', '')
                chunk_type = getattr(chunk, 'chunk_type', '')
                section_id = getattr(chunk, 'section_id', '')

                # Build chunk text with metadata
                chunk_text = f"""
**Source: {spec_id} - {section_title}** (Type: {chunk_type})
**Section: {section_id}**

{content}

---
"""
            else:
                chunk_text = f"{content}\n\n---\n"

            # Check size limit
            if total_chars + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n".join(context_parts)

    @staticmethod
    def build_context_with_labels(chunks: List, labels: Dict[str, str] = None) -> str:
        """
        Build context with entity labels for comparison questions.

        Args:
            chunks: List of chunks
            labels: Dict mapping chunk_id to entity label

        Returns:
            Labeled context string
        """
        labels = labels or {}
        context_parts = []

        for chunk in chunks:
            chunk_id = getattr(chunk, 'chunk_id', '')
            label = labels.get(chunk_id, '')

            content = getattr(chunk, 'content', '')
            if len(content) > 1500:
                content = content[:1500] + "..."

            spec_id = getattr(chunk, 'spec_id', '')
            section_title = getattr(chunk, 'section_title', '')

            if label:
                chunk_text = f"""
**[{label}] {spec_id} - {section_title}**

{content}

---
"""
            else:
                chunk_text = f"""
**{spec_id} - {section_title}**

{content}

---
"""
            context_parts.append(chunk_text)

        return "\n".join(context_parts)


if __name__ == "__main__":
    print("Prompt Templates Module")
    print("=" * 50)
    print("Available prompt types:")
    print("- definition: For 'What is X?' questions")
    print("- comparison: For 'Compare X and Y' questions")
    print("- procedure: For 'How does X work?' questions")
    print("- network_function: For 'Role of X' questions")
    print("- relationship: For 'How do X and Y interact?' questions")
    print("- multiple_choice: For MCQ questions")
    print("- multi_intent: For complex multi-part questions")
    print("- general: Fallback for other questions")
    print()
    print("Usage:")
    print("  from prompt_templates import PromptTemplates, ContextBuilder")
    print("  context = ContextBuilder.build_context(chunks)")
    print("  prompt = PromptTemplates.get_prompt(query, context, analysis)")
