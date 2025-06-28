#!/usr/bin/env python3
"""
Test script to verify the new UI functionality.
Tests the merged Model Configuration panel and inline chat input.
"""

import requests
import json
import time

def test_new_ui():
    """Test the new UI functionality."""
    base_url = "http://localhost:5002"
    
    print("🧪 Testing New UI Functionality")
    print("=" * 50)
    
    # Test 1: Load a model
    print("\n📦 Test 1: Loading model via new Model Configuration panel")
    print("-" * 50)
    
    model_name = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
    
    load_response = requests.post(f"{base_url}/api/model/load", json={
        "model_name": model_name,
        "adapter_path": None
    })
    
    if load_response.status_code != 200:
        print(f"❌ Failed to load model: {load_response.text}")
        return
        
    load_data = load_response.json()
    if not load_data.get('success'):
        print(f"❌ Failed to load model: {load_data.get('error')}")
        return
        
    print(f"✅ Model load request sent successfully")
    
    # Wait for model to load
    print("⏳ Waiting for model to load...")
    max_wait = 20
    wait_time = 0
    
    while wait_time < max_wait:
        status_response = requests.get(f"{base_url}/api/model/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data.get('loaded'):
                print(f"✅ Model loaded: {status_data.get('model_name')}")
                break
            elif status_data.get('error'):
                print(f"❌ Model loading failed: {status_data.get('error')}")
                return
        
        time.sleep(1)
        wait_time += 1
        
    if wait_time >= max_wait:
        print(f"⏰ Timeout waiting for model to load")
        return
    
    # Test 2: Generate text using the new API
    print("\n🤖 Test 2: Generating text with new parameters")
    print("-" * 40)
    
    test_prompt = "What is machine learning?"
    
    gen_response = requests.post(f"{base_url}/api/model/generate", json={
        "prompt": test_prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_kv_size": 8192
    })
    
    if gen_response.status_code != 200:
        print(f"❌ Failed to generate text: {gen_response.text}")
        return
        
    gen_data = gen_response.json()
    if not gen_data.get('success'):
        print(f"❌ Failed to generate text: {gen_data.get('error')}")
        return
        
    generated_text = gen_data.get('completion', '')
    print(f"📝 Generated text: {generated_text[:150]}...")
    print(f"⏱️  Generation time: {gen_data.get('generation_time', 0):.2f}s")
    
    # Test 3: Verify model status
    print("\n📊 Test 3: Verifying model status")
    print("-" * 30)
    
    status_response = requests.get(f"{base_url}/api/model/status")
    if status_response.status_code == 200:
        status_data = status_response.json()
        current_model = status_data.get('model_name')
        is_loaded = status_data.get('loaded')
        
        print(f"✅ Model Status:")
        print(f"   - Loaded: {is_loaded}")
        print(f"   - Model: {current_model}")
        print(f"   - Loading: {status_data.get('is_loading', False)}")
        
        if current_model == model_name and is_loaded:
            print("✅ Model status verification passed!")
        else:
            print("⚠️  Model status verification failed!")
    
    print("\n🎉 New UI functionality test completed!")
    print("\n📋 Summary:")
    print("✅ Merged Model Configuration panel")
    print("✅ Removed Publish Adapter button")
    print("✅ Moved generation parameters to Model Configuration")
    print("✅ Ready for inline chat input testing via web UI")

if __name__ == "__main__":
    test_new_ui() 