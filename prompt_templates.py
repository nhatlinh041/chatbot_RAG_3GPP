"""
Enhanced Prompt Templates for 3GPP RAG System.
Provides specialized prompts for different question types.
"""
import re
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
        term_definitions = analysis.get('term_definitions', {})

        # Select appropriate template
        if intent == 'definition':
            return PromptTemplates.get_definition_prompt(query, context, detected_entities, term_definitions)
        elif intent == 'comparison':
            return PromptTemplates.get_comparison_prompt(query, context, detected_entities, term_definitions)
        elif intent == 'procedure':
            return PromptTemplates.get_procedure_prompt(query, context, term_definitions)
        elif intent == 'network_function':
            return PromptTemplates.get_network_function_prompt(query, context, detected_entities, term_definitions)
        elif intent == 'relationship':
            return PromptTemplates.get_relationship_prompt(query, context, detected_entities, term_definitions)
        elif intent == 'multiple_choice':
            return PromptTemplates.get_multiple_choice_prompt(query, context, term_definitions)
        elif analysis.get('requires_multi_step', False):
            return PromptTemplates.get_multi_intent_prompt(query, context, analysis)
        else:
            return PromptTemplates.get_general_prompt(query, context, term_definitions)

    @staticmethod
    def get_definition_prompt(query: str, context: str, entities: List[str] = None,
                             term_definitions: Dict = None) -> str:
        """Prompt for definition questions"""
        entities = entities or []
        term_definitions = term_definitions or {}
        entity_str = ', '.join(entities) if entities else "the term"

        # Build authoritative definitions section
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        return f"""You are a 3GPP telecommunications expert. Answer this definition question precisely.

{defs_section}
**Question:** {query}

**Context from 3GPP specifications:**
{context}

**Instructions:**
1. START with the AUTHORITATIVE DEFINITION from above (if available)
2. Use the exact full name provided - do not modify or paraphrase it
3. Structure your answer:
   - **Definition**: Use the authoritative full name, then elaborate with context
   - **Key Characteristics**: Bullet points of main features/functions
   - **Related Components**: How it relates to other 3GPP entities (if mentioned in context)
   - **Specification Reference**: Which specs define it

4. Use proper 3GPP terminology
5. Bold important terms like **{entities[0] if entities else 'key terms'}**, **5G Core**, etc.
6. If abbreviation, show format: **ABBR** (Full Name from authoritative definition)

**CRITICAL Anti-Hallucination Rules:**
- PRIORITIZE authoritative definitions above over any text in context
- If context contradicts the authoritative definition, use the authoritative one
- Use ONLY information from the provided context for elaboration
- If context lacks sufficient info, state: "Based on the provided specifications, I can confirm [what you found]. Additional details about [missing info] are not available in these sections."
- DO NOT use general knowledge - stick to authoritative definitions and context
- Cite spec sections (e.g., "According to TS 23.501...")

**Format:** Use Markdown with clear sections and bullet points.

{PromptTemplates._build_choice_instruction('definition', query)}"""

    @staticmethod
    def _build_definitions_section(term_definitions: Dict) -> str:
        """Helper to build authoritative definitions section"""
        if not term_definitions:
            return ""

        defs_lines = ["**AUTHORITATIVE DEFINITIONS (from 3GPP Term Database):**"]
        for abbrev, info in term_definitions.items():
            full_name = info.get('full_name', '')
            specs = info.get('specs', [])
            specs_str = f" (Defined in: {', '.join(specs[:3])})" if specs else ""
            defs_lines.append(f"- **{abbrev}**: {full_name}{specs_str}")

        defs_lines.append("")  # Empty line after section
        return "\n".join(defs_lines)

    @staticmethod
    def _build_choice_instruction(intent: str = None, query: str = "") -> str:
        """
        Helper to build explicit choice format instruction.
        Only returns instruction if question is actually multiple choice.

        Args:
            intent: Primary intent from query analysis
            query: Original question text

        Returns:
            Choice instruction string or empty string
        """
        # Only add choice instruction for multiple_choice intent
        if intent != 'multiple_choice':
            return ""

        # Extract option labels from the question
        options = []
        for line in query.split('\n'):
            # Match patterns like "A)", "A.", "(A)", "Option A:", etc.
            match = re.match(r'^\s*[\(\[]?([A-Da-d])[\)\.][:)\s]', line.strip())
            if match:
                options.append(match.group(1).upper())

        if not options:
            # Fallback: look for inline options like "a) text b) text"
            inline_matches = re.findall(r'[\(\[]?([A-Da-d])[\)\.]', query)
            if len(inline_matches) >= 2:  # At least 2 options
                options = [m.upper() for m in inline_matches[:4]]  # Max 4 options

        # Build option list string
        if options:
            options_str = ", ".join(options[:-1]) + f", or {options[-1]}" if len(options) > 1 else options[0]
        else:
            options_str = "A, B, C, or D"

        return f"""
**IMPORTANT - Multiple Choice Question:**
This is a multiple choice question with options: {options_str}

You MUST:
1. Quote the exact question and all options from above
2. Analyze each option based on the provided context
3. State your answer at the end using this EXACT format:

**The correct answer is option ({options[0] if options else 'X'})**

where the letter matches one of the options in the question.
"""

    @staticmethod
    def get_comparison_prompt(query: str, context: str, entities: List[str] = None,
                             term_definitions: Dict = None) -> str:
        """Prompt for comparison questions"""
        entities = entities or ['Entity1', 'Entity2']
        term_definitions = term_definitions or {}
        e1 = entities[0] if len(entities) > 0 else 'Entity1'
        e2 = entities[1] if len(entities) > 1 else 'Entity2'

        # Build authoritative definitions section
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        return f"""You are a 3GPP telecommunications expert. Compare these entities based on specifications.

{defs_section}
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
- YOU MUST compare ONLY these two entities: {e1} and {e2}
- DO NOT compare any other entities, even if they appear in the context
- USE the AUTHORITATIVE DEFINITIONS for full names - they are the ground truth
- If authoritative definitions contradict context, TRUST the authoritative definitions
- Start Overview section with proper full names from authoritative definitions
- Base comparison ONLY on provided context
- If information missing for one entity, explicitly state it
- Use exact terminology from 3GPP specs
- Cite specific sections for claims

{PromptTemplates._build_choice_instruction('comparison', query)}"""

    @staticmethod
    def get_procedure_prompt(query: str, context: str, term_definitions: Dict = None) -> str:
        """Prompt for procedure/process questions"""
        term_definitions = term_definitions or {}
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        return f"""You are a 3GPP telecommunications expert. Explain this procedure step-by-step.

{defs_section}

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
- Use proper message/signal names from specs

{PromptTemplates._build_choice_instruction('procedure', query)}"""

    @staticmethod
    def get_network_function_prompt(query: str, context: str, entities: List[str] = None,
                                   term_definitions: Dict = None) -> str:
        """Prompt for network function role/responsibility questions"""
        entities = entities or []
        term_definitions = term_definitions or {}
        nf = entities[0] if entities else "the network function"
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        return f"""You are a 3GPP telecommunications expert. Explain the role and functions of this network component.

{defs_section}
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
- Don't add functions from general knowledge

{PromptTemplates._build_choice_instruction('network_function', query)}"""

    @staticmethod
    def get_relationship_prompt(query: str, context: str, entities: List[str] = None,
                              term_definitions: Dict = None) -> str:
        """Prompt for relationship/interaction questions"""
        entities = entities or []
        term_definitions = term_definitions or {}
        defs_section = PromptTemplates._build_definitions_section(term_definitions)
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
- If relationship not clear from context, state it

{PromptTemplates._build_choice_instruction('relationship', query)}"""

    @staticmethod
    def get_multiple_choice_prompt(query: str, context: str, term_definitions: Dict = None) -> str:
        """
        Prompt for multiple choice questions.
        CRITICAL: Output format must start with "Answer: X. [option text]" for extraction.
        """
        term_definitions = term_definitions or {}
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        # Extract options from query
        options = []
        for line in query.split('\n'):
            match = re.match(r'^\s*([A-Da-d])[\.\)]\s*(.+)', line.strip())
            if match:
                options.append((match.group(1).upper(), match.group(2).strip()))

        # Build options reminder
        options_text = ""
        if options:
            options_text = "\n**Available Options:**\n"
            for letter, text in options:
                options_text += f"- {letter}. {text}\n"

        # Build example based on actual options
        example_letter = options[2][0] if len(options) > 2 else "C"
        example_text = options[2][1] if len(options) > 2 else "14 symbols"

        return f"""You are a 3GPP telecommunications expert answering a multiple choice question.

{defs_section}
**Question:** {query}
{options_text}
**Context from 3GPP specifications:**
{context}

**CRITICAL OUTPUT FORMAT - YOU MUST FOLLOW THIS EXACTLY:**

Answer: [LETTER]. [EXACT TEXT OF THE CHOSEN OPTION]

[Then provide brief explanation]

**Example correct output:**
Answer: {example_letter}. {example_text}

According to the 3GPP specifications, this is correct because...

**RULES:**
1. FIRST LINE MUST BE: "Answer: X. [exact option text]"
   - X is the letter (A, B, C, or D)
   - Follow with a period and the EXACT text of that option (copy it verbatim)
2. Then provide brief explanation (2-3 sentences max)
3. Base answer ONLY on provided context
4. Do NOT add markdown headers or formatting before the answer

**WRONG formats (DO NOT USE):**
- "Answer: C" (missing option text) ❌
- "The answer is C. 14 symbols" ❌
- "**Answer:** C. 14 symbols" ❌
- "C. 14 symbols" (missing "Answer:") ❌

**CORRECT format:**
Answer: {example_letter}. {example_text}

[Brief explanation based on context]"""

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
- Base everything on provided context

{PromptTemplates._build_choice_instruction('multi_intent', query)}"""

    @staticmethod
    def get_general_prompt(query: str, context: str, term_definitions: Dict = None) -> str:
        """General purpose prompt for unclassified questions"""
        term_definitions = term_definitions or {}
        defs_section = PromptTemplates._build_definitions_section(term_definitions)

        return f"""You are a 3GPP telecommunications expert. Answer this question based on the specifications.

{defs_section}
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

{PromptTemplates._build_choice_instruction('general', query)}

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
