#!/usr/bin/env python3
"""
Test script to verify the new system prompt handling implementation.
"""

import requests
import json
import time

def test_system_prompt_handling():
    """Test the new system prompt handling with different scenarios."""
    
    # Base URLs
    model_server_url = "http://localhost:5001"
    web_server_url = "http://localhost:5000"
    
    print("🧪 Testing System Prompt Handling Implementation")
    print("=" * 60)
    
    # Test case 1: INSTRUCT model with history (new format)
    print("\n📋 Test 1: INSTRUCT model with conversation history")
    test_request = {
        "prompt": "I thought you were someone else ?",
        "history": [
            {
                "role": "system",
                "content": "I am John, a memory-enhanced AI"
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
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_kv_size": 8192,
        "streaming": False,
        "is_base_model": False  # INSTRUCT model
    }
    
    try:
        print(f"📤 Sending request to model server...")
        print(f"System prompt: {test_request['history'][0]['content']}")
        print(f"Last exchange: {test_request['history'][-2]['content']} -> {test_request['history'][-1]['content'][:50]}...")
        print(f"Current prompt: {test_request['prompt']}")
        
        response = requests.post(f"{model_server_url}/api/model/generate", 
                               json=test_request, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Success! Generated text: {result['text'][:200]}...")
                print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to model server. Make sure it's running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test case 2: BASE model with system prompt (new format)
    print("\n📋 Test 2: BASE model with system prompt")
    base_test_request = {
        "prompt": "You are a helpful assistant.\n\nHello, how are you?",  # Pre-formatted by frontend
        "history": [],
        "max_tokens": 50,
        "temperature": 0.7,
        "streaming": False,
        "is_base_model": True  # BASE model
    }
    
    try:
        print(f"📤 Sending BASE model request...")
        print(f"Pre-formatted prompt: {base_test_request['prompt']}")
        
        response = requests.post(f"{model_server_url}/api/model/generate", 
                               json=base_test_request, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Success! Generated text: {result['text'][:200]}...")
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to model server. Make sure it's running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test case 3: Legacy format (backward compatibility)
    print("\n📋 Test 3: Legacy system_prompt parameter (backward compatibility)")
    legacy_test_request = {
        "prompt": "What is the capital of France?",
        "system_prompt": "You are a geography expert.",  # Legacy format
        "max_tokens": 50,
        "temperature": 0.7,
        "streaming": False
    }
    
    try:
        print(f"📤 Sending legacy format request...")
        print(f"Legacy system prompt: {legacy_test_request['system_prompt']}")
        print(f"Prompt: {legacy_test_request['prompt']}")
        
        response = requests.post(f"{model_server_url}/api/model/generate", 
                               json=legacy_test_request, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Success! Generated text: {result['text'][:200]}...")
            else:
                print(f"❌ Generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to model server. Make sure it's running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("- Test 1: New INSTRUCT format with conversation history")
    print("- Test 2: New BASE format with pre-formatted prompt")
    print("- Test 3: Legacy system_prompt parameter compatibility")
    print("\n💡 Check the model server logs for detailed processing information")

if __name__ == "__main__":
    test_system_prompt_handling() 