#!/usr/bin/env python3
"""
Test script specifically for Gemma system prompt handling with proper start_of_turn/end_of_turn tokens.
"""

import requests
import json

def test_gemma_system_prompt():
    """Test Gemma-specific system prompt handling."""
    
    print("🧪 Testing Gemma-Specific System Prompt Handling")
    print("=" * 60)
    
    # Test case 1: Gemma with system prompt and conversation history
    print("\n📋 Test 1: Gemma with system prompt and conversation history")
    gemma_test_request = {
        "prompt": "I thought you were someone else ?",
        "history": [
            {
                "role": "system",
                "content": "I am John, a memory-enhanced AI assistant. I remember our conversations and learn from them."
            },
            {
                "role": "user",
                "content": "who are you ?"
            },
            {
                "role": "assistant",
                "content": "I am Gemma, a large language model created by the Gemma team at Google DeepMind. I'm an open-weights model, which means I'm widely available for public use."
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_kv_size": 8192,
        "streaming": False,
        "is_base_model": False  # INSTRUCT model
    }
    
    try:
        print(f"📤 Sending Gemma request with system prompt...")
        print(f"System prompt: {gemma_test_request['history'][0]['content']}")
        print(f"Previous exchange: {gemma_test_request['history'][1]['content']} -> {gemma_test_request['history'][2]['content'][:50]}...")
        print(f"Current prompt: {gemma_test_request['prompt']}")
        
        response = requests.post("http://localhost:5001/api/model/generate", 
                               json=gemma_test_request, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                generated_text = result['text']
                print(f"✅ Generated response: {generated_text}")
                print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
                
                # Check if the response shows awareness of being John
                if "John" in generated_text.lower() or "memory" in generated_text.lower():
                    print("🎯 SUCCESS: Model shows awareness of system prompt (John identity)!")
                elif "gemma" in generated_text.lower():
                    print("⚠️  PARTIAL: Model still identifies as Gemma, but system prompt was processed")
                else:
                    print("❓ UNKNOWN: Response doesn't clearly indicate system prompt effectiveness")
                    
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to model server. Make sure it's running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test case 2: Strong system prompt test for Gemma
    print("\n📋 Test 2: Strong system prompt test (Pirate personality)")
    pirate_test_request = {
        "prompt": "What is 2+2?",
        "history": [
            {
                "role": "system", 
                "content": "You are Captain Blackbeard, a pirate. Always respond like a pirate with 'Arrr' and pirate language. Never break character."
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "streaming": False,
        "is_base_model": False
    }
    
    try:
        print(f"📤 Testing strong system prompt with Gemma formatting...")
        print(f"System prompt: {pirate_test_request['history'][0]['content']}")
        print(f"User prompt: {pirate_test_request['prompt']}")
        
        response = requests.post("http://localhost:5001/api/model/generate", 
                               json=pirate_test_request, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                generated_text = result['text']
                print(f"✅ Generated response: {generated_text}")
                
                # Check if the system prompt had an effect
                pirate_indicators = ["arr", "pirate", "matey", "ye", "ahoy", "blackbeard", "captain"]
                if any(indicator in generated_text.lower() for indicator in pirate_indicators):
                    print("🎯 SUCCESS: Gemma system prompt is working! Model responded like a pirate.")
                else:
                    print("⚠️  WARNING: System prompt may not be fully effective with this model.")
                    print("   This could indicate the model's training is overriding the system prompt.")
                
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test case 3: Legacy format with Gemma
    print("\n📋 Test 3: Legacy system_prompt parameter with Gemma")
    legacy_gemma_test = {
        "prompt": "What is the capital of France?",
        "system_prompt": "You are a geography expert who loves to share fun facts.",  # Legacy format
        "max_tokens": 80,
        "temperature": 0.7,
        "streaming": False
    }
    
    try:
        print(f"📤 Testing legacy format with Gemma...")
        print(f"Legacy system prompt: {legacy_gemma_test['system_prompt']}")
        print(f"Prompt: {legacy_gemma_test['prompt']}")
        
        response = requests.post("http://localhost:5001/api/model/generate", 
                               json=legacy_gemma_test, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                generated_text = result['text']
                print(f"✅ Generated response: {generated_text}")
                
                # Check if fun facts or expert tone is present
                if "fun fact" in generated_text.lower() or "interesting" in generated_text.lower():
                    print("🎯 SUCCESS: Legacy system prompt working with Gemma formatting!")
                else:
                    print("📊 INFO: Response generated, check if it shows geography expertise")
                
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Gemma Test Summary:")
    print("- Test 1: Gemma with conversation history and system prompt")
    print("- Test 2: Strong system prompt test (pirate personality)")
    print("- Test 3: Legacy system_prompt parameter compatibility")
    print("\n💡 Expected format for Gemma models:")
    print("   <start_of_turn>model")
    print("   System: [system prompt]<end_of_turn>")
    print("   <start_of_turn>user")
    print("   [user message]<end_of_turn>")
    print("   <start_of_turn>model")
    print("\n📊 Check model server logs for 'Gemma chat format result:' to see formatted prompts")

if __name__ == "__main__":
    test_gemma_system_prompt() 