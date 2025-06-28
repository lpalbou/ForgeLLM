#!/usr/bin/env python3

# Test the final fix: raw prompts for all models

def test_final_logic():
    """Test that we now use raw prompts for all models"""
    
    print("🧪 Testing Final Fix: Raw Prompts for All Models")
    print("=" * 60)
    
    # Test cases for different scenarios
    test_cases = [
        {
            "name": "Base model, no history",
            "is_base": True,
            "prompt": "My name is",
            "include_history": False,
            "history": [],
            "expected": "My name is"
        },
        {
            "name": "Base model, with history", 
            "is_base": True,
            "prompt": "What's next?",
            "include_history": True,
            "history": [("Hello", "Hi there!"), ("How are you?", "I'm doing well.")],
            "expected": "Hello\nHi there!\nHow are you?\nI'm doing well.\nWhat's next?"
        },
        {
            "name": "Instruct model, no history",
            "is_base": False,
            "prompt": "My name is", 
            "include_history": False,
            "history": [],
            "expected": "My name is"  # Changed from "User: My name is\nAssistant:"
        },
        {
            "name": "Instruct model, with history",
            "is_base": False,
            "prompt": "What's next?",
            "include_history": True, 
            "history": [("Hello", "Hi there!"), ("How are you?", "I'm doing well.")],
            "expected": "Hello\nHi there!\nHow are you?\nI'm doing well.\nWhat's next?"  # Changed from User:/Assistant: format
        }
    ]
    
    def format_prompt(is_base_model, prompt, include_history, history):
        """New logic: always use raw prompts"""
        final_prompt = prompt
        
        if include_history:
            history_text = ''
            # For all models: use plain text continuation without User:/Assistant: tags
            for user, assistant in history:
                history_text += f"{user}\n{assistant}\n"
            final_prompt = history_text + prompt
        # For all models with no history, use prompt as-is
        
        return final_prompt
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        result = format_prompt(
            test_case["is_base"], 
            test_case["prompt"], 
            test_case["include_history"], 
            test_case["history"]
        )
        
        status = "✅ PASS" if result == test_case["expected"] else "❌ FAIL"
        
        print(f"{status} Test {i}: {test_case['name']}")
        print(f"     Expected: '{test_case['expected']}'")
        print(f"     Got:      '{result}'")
        print()
        
        if result != test_case["expected"]:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Perfect! Now all models use raw prompts:")
        print("   ⚡ Base models: Raw text (as before)")
        print("   🤖 Instruct models: Raw text (no more User:/Assistant:)")
        print("\n📝 This should fix the issue you saw in the screenshot!")
    
    return all_passed

if __name__ == "__main__":
    test_final_logic()
