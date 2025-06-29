#!/usr/bin/env python3

# Test script to verify the base model detection and prompt formatting logic

def test_base_model_detection():
    """Test the base model detection logic"""
    
    # Test cases
    test_cases = [
        ("mlx-community/gemma-3-4b-pt-8bit", True, "Has 'pt' pattern"),
        ("mlx-community/gemma-3-4b-it-8bit", False, "Has 'it' pattern but not at word boundary"),
        ("microsoft/Phi-3-mini-4k-instruct", False, "Has 'instruct' pattern"),
        ("google/gemma-3-4b-base", True, "Has 'base' pattern"),
        ("Qwen/Qwen3-8B-Base", True, "Has 'Base' pattern"),
        ("mlx-community/Meta-Llama-3.1-8B-bf16", True, "No instruct patterns"),
        ("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16", False, "Has 'Instruct' pattern"),
    ]
    
    def is_base_model(model_name):
        """Replicate the JavaScript logic in Python"""
        model_name = model_name.lower()
        
        # Instruction-tuned model patterns
        instruct_patterns = [
            'instruct', 'chat', 'sft', 'dpo', 'rlhf', 
            'assistant', 'alpaca', 'vicuna', 'wizard', 'orca',
            'dolphin', 'openhermes', 'airoboros', 'nous',
            'claude', 'gpt', 'turbo', 'dialogue', 'conversation'
        ]
        
        # Special patterns that need word boundary checking
        special_patterns = ['it']
        
        # Base model patterns
        base_patterns = ['base', 'pt', 'pretrain', 'foundation']
        
        # Check for explicit base model indicators first
        has_base_pattern = any(
            f'-{pattern}' in model_name or 
            f'_{pattern}' in model_name or
            f'-{pattern}-' in model_name or
            f'_{pattern}_' in model_name or
            model_name.endswith(f'-{pattern}') or
            model_name.endswith(f'_{pattern}') or
            model_name.endswith(pattern)
            for pattern in base_patterns
        )
        
        if has_base_pattern:
            return True
        
        # Check for regular instruct patterns
        has_instruct_pattern = any(pattern in model_name for pattern in instruct_patterns)
        
        # Check special patterns with word boundaries
        if not has_instruct_pattern:
            import re
            has_instruct_pattern = any(
                bool(re.search(r'\b' + pattern + r'\b', model_name, re.IGNORECASE))
                for pattern in special_patterns
            )
        
        return not has_instruct_pattern
    
    print("🧪 Testing Base Model Detection Logic")
    print("=" * 60)
    
    all_passed = True
    for model_name, expected, reason in test_cases:
        result = is_base_model(model_name)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} {model_name}")
        print(f"     Expected: {'Base' if expected else 'Instruct'}, Got: {'Base' if result else 'Instruct'}")
        print(f"     Reason: {reason}")
        print()
        
        if result != expected:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

def test_prompt_formatting():
    """Test the prompt formatting logic"""
    
    print("\n🧪 Testing Prompt Formatting Logic")
    print("=" * 60)
    
    # Test cases: (is_base_model, original_prompt, include_history, history, expected_final_prompt)
    test_cases = [
        # Base model, no history
        (True, "My name is", False, [], "My name is"),
        
        # Base model, with history  
        (True, "What's next?", True, [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well.")
        ], "Hello\nHi there!\nHow are you?\nI'm doing well.\nWhat's next?"),
        
        # Instruct model, no history
        (False, "My name is", False, [], "User: My name is\nAssistant:"),
        
        # Instruct model, with history
        (False, "What's next?", True, [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well.")
        ], "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm doing well.\nUser: What's next?\nAssistant:"),
    ]
    
    def format_prompt(is_base_model, prompt, include_history, history):
        """Replicate the JavaScript prompt formatting logic"""
        final_prompt = prompt
        
        if include_history:
            history_text = ''
            if is_base_model:
                # For base models: use plain text continuation
                for user, assistant in history:
                    history_text += f"{user}\n{assistant}\n"
                final_prompt = history_text + prompt
            else:
                # For instruct models: use User:/Assistant: formatting
                for user, assistant in history:
                    history_text += f"User: {user}\nAssistant: {assistant}\n"
                final_prompt = history_text + f"User: {prompt}\nAssistant:"
        elif not is_base_model:
            # Single turn for instruct models
            final_prompt = f"User: {prompt}\nAssistant:"
        
        return final_prompt
    
    all_passed = True
    for i, (is_base, prompt, include_history, history, expected) in enumerate(test_cases, 1):
        result = format_prompt(is_base, prompt, include_history, history)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        model_type = "Base" if is_base else "Instruct"
        hist_str = "with history" if include_history else "no history"
        
        print(f"{status} Test {i}: {model_type} model, {hist_str}")
        print(f"     Input prompt: '{prompt}'")
        print(f"     Expected: '{expected}'")
        print(f"     Got:      '{result}'")
        print()
        
        if result != expected:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    detection_passed = test_base_model_detection()
    formatting_passed = test_prompt_formatting()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    print(f"Base model detection: {'✅ PASSED' if detection_passed else '❌ FAILED'}")
    print(f"Prompt formatting:    {'✅ PASSED' if formatting_passed else '❌ FAILED'}")
    
    if detection_passed and formatting_passed:
        print("\n🎉 ALL TESTS PASSED! The logic should work correctly in the web interface.")
    else:
        print("\n⚠️  Some tests failed. The logic needs to be fixed.")
