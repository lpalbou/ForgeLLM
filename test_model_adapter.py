#!/usr/bin/env python
"""
Script to test model loading with adapters.
"""

import sys
import time
import logging
import os
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test model loading and generation with adapters")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--adapter", help="Path to adapter weights")
    parser.add_argument("--prompt", default="Tell me about continued pre-training", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Test model loading
    logger.info(f"Testing model: {args.model}")
    if args.adapter:
        logger.info(f"With adapter: {args.adapter}")
    
    try:
        # Import here to avoid loading mlx until needed
        import mlx.core as mx
        from mlx_lm import load
        
        start_time = time.time()
        logger.info("Starting model load...")
        
        # Load the model
        model, tokenizer = load(args.model, adapter_path=args.adapter)
        
        end_time = time.time()
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Print model info
        logger.info(f"Model device: {mx.default_device()}")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Test generation
        logger.info(f"Generating with prompt: '{args.prompt}'")
        
        from mlx_lm import generate
        
        start_time = time.time()
        response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
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