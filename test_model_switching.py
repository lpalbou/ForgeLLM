#!/usr/bin/env python3
"""
Test script to verify model switching functionality.
This script loads different models and generates text to verify the model actually changes.
"""

import requests
import json
import time

def test_model_switching():
    """Test that model switching works correctly."""
    base_url = "http://localhost:5002"
    
    # Test models - using different models to see clear differences
    models = [
        "mlx-community/Mistral-7B-Instruct-v0.2",
        "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
    ]
    
    test_prompt = "who are you ?"
    
    print("🧪 Testing Model Switching Functionality")
    print("=" * 50)
    
    for i, model_name in enumerate(models, 1):
        print(f"\n📦 Test {i}: Loading {model_name}")
        print("-" * 40)
        
        # Load the model
        load_response = requests.post(f"{base_url}/api/model/load", json={
            "model_name": model_name,
            "adapter_path": None
        })
        
        if load_response.status_code != 200:
            print(f"❌ Failed to load model: {load_response.text}")
            continue
            
        load_data = load_response.json()
        if not load_data.get('success'):
            print(f"❌ Failed to load model: {load_data.get('error')}")
            continue
            
        print(f"✅ Model load request sent successfully")
        
        # Wait for model to load
        print("⏳ Waiting for model to load...")
        max_wait = 30  # 30 seconds max wait
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
                    break
            
            time.sleep(1)
            wait_time += 1
            
        if wait_time >= max_wait:
            print(f"⏰ Timeout waiting for model to load")
            continue
            
        # Generate text
        print(f"🤖 Generating text with prompt: '{test_prompt}'")
        gen_response = requests.post(f"{base_url}/api/model/generate", json={
            "prompt": test_prompt,
            "max_tokens": 50,
            "temperature": 0.7
        })
        
        if gen_response.status_code != 200:
            print(f"❌ Failed to generate text: {gen_response.text}")
            continue
            
        gen_data = gen_response.json()
        if not gen_data.get('success'):
            print(f"❌ Failed to generate text: {gen_data.get('error')}")
            continue
            
        generated_text = gen_data.get('completion', '')
        print(f"📝 Generated text: {generated_text[:200]}...")
        print(f"⏱️  Generation time: {gen_data.get('generation_time', 0):.2f}s")
        
        # Verify the model is actually the one we loaded
        status_response = requests.get(f"{base_url}/api/model/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            current_model = status_data.get('model_name')
            if current_model == model_name:
                print(f"✅ Confirmed current model: {current_model}")
            else:
                print(f"⚠️  Model mismatch! Expected: {model_name}, Got: {current_model}")
        
        print(f"✅ Test {i} completed successfully!")
        
        # Small delay between tests
        if i < len(models):
            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    print("\n🎉 Model switching test completed!")

if __name__ == "__main__":
    test_model_switching() 