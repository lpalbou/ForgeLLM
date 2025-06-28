#!/usr/bin/env python
"""
ForgeLLM CLI - Command Line Interface for ForgeLLM
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='ForgeLLM CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Model management commands')
    model_subparsers = model_parser.add_subparsers(dest='model_command', help='Model command to run')
    
    # Model load command
    load_parser = model_subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('--model', required=True, help='Model name or path')
    load_parser.add_argument('--adapter', help='Optional adapter path')
    
    # Model test command
    test_parser = model_subparsers.add_parser('test', help='Test a model with a prompt')
    test_parser.add_argument('--model', required=True, help='Model name or path')
    test_parser.add_argument('--adapter', help='Optional adapter path')
    test_parser.add_argument('--prompt', required=True, help='Prompt to test with')
    test_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    test_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    
    # Dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Dataset management commands')
    dataset_parser.add_argument('--input-dir', help='Input directory containing dataset files')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Training commands')
    train_parser.add_argument('--model-name', help='Model name or path')
    train_parser.add_argument('--input-dir', help='Input directory containing dataset files')
    train_parser.add_argument('--output-dir', help='Output directory for checkpoints')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, default=5e-6, help='Learning rate')
    train_parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum iterations')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text from a model')
    generate_parser.add_argument('--model', required=True, help='Model name or path')
    generate_parser.add_argument('--adapter-path', help='Optional adapter path')
    generate_parser.add_argument('--prompt', required=True, help='Prompt to generate from')
    generate_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    generate_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    
    args = parser.parse_args()
    
    if args.command == 'model':
        if args.model_command == 'load':
            load_model(args.model, args.adapter)
        elif args.model_command == 'test':
            test_model(args.model, args.adapter, args.prompt, args.max_tokens, args.temperature)
    elif args.command == 'dataset':
        if args.input_dir:
            analyze_dataset(args.input_dir)
        else:
            logger.error("Input directory required for dataset command")
    elif args.command == 'train':
        train_model(
            args.model_name,
            args.input_dir,
            args.output_dir,
            args.batch_size,
            args.learning_rate,
            args.max_iterations
        )
    elif args.command == 'generate':
        generate_text(
            args.model,
            args.adapter_path,
            args.prompt,
            args.max_tokens,
            args.temperature
        )
    else:
        parser.print_help()

def load_model(model_name, adapter_path=None):
    """Load a model and verify it works."""
    logger.info(f"Loading model {model_name} with adapter {adapter_path}")
    
    try:
        # Import here to avoid loading mlx until needed
        import mlx.core as mx
        from mlx_lm import load
        
        start_time = time.time()
        logger.info("Starting model load...")
        
        # Load the model directly
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        
        end_time = time.time()
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Print model info
        logger.info(f"Model device: {mx.default_device()}")
        logger.info(f"Model type: {type(model).__name__}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def test_model(model_name, adapter_path=None, prompt="Hello, how are you?", max_tokens=100, temperature=0.7):
    """Test a model with a prompt."""
    logger.info(f"Testing model {model_name} with adapter {adapter_path}")
    
    try:
        # Import here to avoid loading mlx until needed
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        
        # Load the model
        logger.info("Loading model...")
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        logger.info("Model loaded successfully")
        
        # Generate text
        logger.info(f"Generating with prompt: '{prompt}'")
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # Generate text using stream_generate and collect all chunks
        response_text = ""
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            response_text += chunk.text
        
        end_time = time.time()
        
        # Print the response
        logger.info(f"Generated in {end_time - start_time:.2f} seconds:")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        print(response_text)
        print("="*50)
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def analyze_dataset(input_dir):
    """Analyze a dataset."""
    logger.info(f"Analyzing dataset in {input_dir}")
    
    try:
        # Check if directory exists
        if not os.path.exists(input_dir):
            logger.error(f"Directory does not exist: {input_dir}")
            return False
        
        # Count files
        file_count = 0
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.jsonl') or file.endswith('.txt'):
                    file_count += 1
        
        logger.info(f"Found {file_count} files in {input_dir}")
        
        # Print some sample files
        logger.info("Sample files:")
        for root, _, files in os.walk(input_dir):
            for file in sorted(files)[:5]:
                if file.endswith('.jsonl') or file.endswith('.txt'):
                    logger.info(f"  {os.path.join(root, file)}")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return False

def train_model(model_name, input_dir, output_dir, batch_size, learning_rate, max_iterations):
    """Train a model."""
    logger.info(f"Training model {model_name} with data from {input_dir}")
    logger.info(f"Parameters: batch_size={batch_size}, learning_rate={learning_rate}, max_iterations={max_iterations}")
    
    try:
        # Import here to avoid loading modules until needed
        import subprocess
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            "python", "-m", "continued_pretraining",
            "--model", model_name,
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--max-iterations", str(max_iterations)
        ]
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def generate_text(model_name, adapter_path, prompt, max_tokens, temperature):
    """Generate text from a model."""
    logger.info(f"Generating text with model {model_name}")
    
    try:
        # Import here to avoid loading mlx until needed
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        
        # Load the model
        logger.info("Loading model...")
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        logger.info("Model loaded successfully")
        
        # Generate text
        logger.info(f"Generating with prompt: '{prompt}'")
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # Generate text using stream_generate and collect all chunks
        response_text = ""
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            response_text += chunk.text
        
        end_time = time.time()
        
        # Print the response
        logger.info(f"Generated in {end_time - start_time:.2f} seconds:")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        print(response_text)
        print("="*50)
        
        return True
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return False

if __name__ == "__main__":
    main() 