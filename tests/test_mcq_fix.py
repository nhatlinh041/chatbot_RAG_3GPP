#!/usr/bin/env python3
"""
Test script to verify the MCQ instruction fix.
Ensures that non-MCQ questions don't get MCQ formatting instructions.
"""

from prompt_templates import PromptTemplates

# Test cases
test_cases = [
    {
        "name": "Non-MCQ Definition Question",
        "query": "What is UPF responsible for in 5G architecture?",
        "intent": "definition",
        "should_have_mcq_instruction": False
    },
    {
        "name": "Non-MCQ Comparison Question",
        "query": "Compare AMF and SMF",
        "intent": "comparison",
        "should_have_mcq_instruction": False
    },
    {
        "name": "MCQ with Options on Separate Lines",
        "query": """What is the primary function of AMF?
A) User plane data routing
B) Session management
C) Access and mobility management
D) Policy control""",
        "intent": "multiple_choice",
        "should_have_mcq_instruction": True,
        "expected_options": ["A", "B", "C", "D"]
    },
    {
        "name": "MCQ with Inline Options",
        "query": "The UPF is responsible for: a) routing b) authentication c) both d) neither",
        "intent": "multiple_choice",
        "should_have_mcq_instruction": True,
        "expected_options": ["A", "B", "C", "D"]
    },
    {
        "name": "Non-MCQ Network Function Question",
        "query": "What are the key responsibilities of the SMF?",
        "intent": "network_function",
        "should_have_mcq_instruction": False
    }
]

def test_mcq_instruction_fix():
    """Test that MCQ instructions only appear for actual MCQ questions"""

    print("=" * 70)
    print("Testing MCQ Instruction Fix")
    print("=" * 70)

    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Query: {test_case['query'][:60]}...")
        print(f"Intent: {test_case['intent']}")

        # Get the choice instruction
        instruction = PromptTemplates._build_choice_instruction(
            intent=test_case['intent'],
            query=test_case['query']
        )

        has_instruction = len(instruction.strip()) > 0
        should_have = test_case['should_have_mcq_instruction']

        # Check if instruction presence matches expectation
        if has_instruction == should_have:
            print(f"✓ PASS: MCQ instruction {'present' if has_instruction else 'absent'} as expected")

            # For MCQ, verify options are extracted correctly
            if should_have and 'expected_options' in test_case:
                expected_opts = test_case['expected_options']
                has_all_options = all(opt in instruction for opt in expected_opts)
                if has_all_options:
                    print(f"✓ PASS: All expected options {expected_opts} found in instruction")
                else:
                    print(f"✗ FAIL: Not all options found. Expected: {expected_opts}")
                    all_passed = False
        else:
            print(f"✗ FAIL: MCQ instruction {'present' if has_instruction else 'absent'}, "
                  f"but should be {'present' if should_have else 'absent'}")
            all_passed = False

        # Show instruction snippet if present
        if has_instruction:
            snippet = instruction.strip()[:150]
            print(f"  Instruction snippet: {snippet}...")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_full_prompts():
    """Test that full prompts work correctly"""

    print("\n" + "=" * 70)
    print("Testing Full Prompt Generation")
    print("=" * 70)

    # Test non-MCQ question
    query = "What is UPF responsible for in 5G architecture?"
    context = "Sample context about UPF..."
    analysis = {
        'primary_intent': 'definition',
        'entities': ['UPF'],
        'term_definitions': {}
    }

    prompt = PromptTemplates.get_prompt(query, context, analysis)

    print(f"\nNon-MCQ Definition Question:")
    print(f"Query: {query}")

    # Check that MCQ instruction is NOT in the prompt
    mcq_markers = [
        "The correct answer is option",
        "multiple choice question with options",
        "option (X)"
    ]

    has_mcq_instruction = any(marker in prompt for marker in mcq_markers)

    if not has_mcq_instruction:
        print("✓ PASS: No MCQ instruction in non-MCQ prompt")
    else:
        print("✗ FAIL: MCQ instruction found in non-MCQ prompt!")
        # Show where it appears
        for marker in mcq_markers:
            if marker in prompt:
                idx = prompt.find(marker)
                print(f"  Found '{marker}' at position {idx}")
                print(f"  Context: ...{prompt[max(0,idx-50):idx+100]}...")
        return False

    # Test MCQ question
    mcq_query = """What is the primary function of AMF?
A) User plane data routing
B) Session management
C) Access and mobility management
D) Policy control"""

    mcq_analysis = {
        'primary_intent': 'multiple_choice',
        'entities': ['AMF'],
        'term_definitions': {}
    }

    mcq_prompt = PromptTemplates.get_multiple_choice_prompt(mcq_query, context)

    print(f"\nMCQ Question:")
    print(f"Query: {mcq_query[:50]}...")

    # MCQ prompt should have proper instructions
    if "multiple choice" in mcq_prompt.lower():
        print("✓ PASS: MCQ prompt has multiple choice context")
    else:
        print("✗ FAIL: MCQ prompt missing multiple choice context")
        return False

    print("\n✓ ALL FULL PROMPT TESTS PASSED")
    return True


if __name__ == "__main__":
    # Run tests
    test1_passed = test_mcq_instruction_fix()
    test2_passed = test_full_prompts()

    # Final result
    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - Fix is working correctly!")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED - Please review the implementation")
        exit(1)
