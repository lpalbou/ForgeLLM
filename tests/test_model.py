#!/usr/bin/env python
"""
Simple script to test model loading and generation.
"""

import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_name> [prompt]")
        print("Example: python test_model.py mlx-community/gemma-3-1b-it-bf16 'Tell me about continued pre-training'")
        return 1
    
    model_name = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"
    
    # Test model loading
    logger.info(f"Testing model: {model_name}")
    
    try:
        # Import here to avoid loading mlx until needed
        import mlx.core as mx
        from mlx_lm import load
        
        start_time = time.time()
        logger.info("Starting model load...")
        
        # Load the model directly
        model, tokenizer = load(model_name)
        
        end_time = time.time()
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Print model info
        logger.info(f"Model device: {mx.default_device()}")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Test generation
        logger.info(f"Generating with prompt: '{prompt}'")
        
        from mlx_lm import generate
        
        start_time = time.time()
        
        # Try different parameter combinations
        try:
            logger.info("Trying with temp parameter...")
            response = generate(model, tokenizer, prompt=prompt, max_tokens=100, temp=0.7)
        except Exception as e:
            logger.info(f"Failed with temp parameter: {e}")
            try:
                logger.info("Trying with temperature parameter...")
                response = generate(model, tokenizer, prompt=prompt, max_tokens=100, temperature=0.7)
            except Exception as e:
                logger.info(f"Failed with temperature parameter: {e}")
                logger.info("Trying without temperature parameter...")
                response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        
        end_time = time.time()
        
        # Print the response
        logger.info(f"Generated in {end_time - start_time:.2f} seconds:")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        print(response)
        print("="*50)
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 