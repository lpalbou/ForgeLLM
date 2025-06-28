#!/usr/bin/env python
"""
ForgeLLM CLI - Command Line Interface for ForgeLLM
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime
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
    test_parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens to generate')
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
    generate_parser.add_argument('--prompt', help='Prompt to generate from (if not provided, starts REPL mode)')
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
        if args.prompt:
            generate_text(
                args.model,
                args.adapter_path,
                args.prompt,
                args.max_tokens,
                args.temperature
            )
        else:
            # Start REPL mode
            start_repl(
                args.model,
                args.adapter_path,
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
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # SOTA CLI streaming: Real-time character-by-character output
        logger.info(f"Generating with prompt: '{prompt}' (streaming to terminal)")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            print(chunk.text, end='', flush=True)  # Stream to terminal in real-time
        
        end_time = time.time()
        
        # Final formatting
        print("\n" + "="*50)
        logger.info(f"Streaming generation completed in {end_time - start_time:.2f} seconds")
        
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
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # SOTA CLI streaming: Real-time character-by-character output
        logger.info(f"Generating with prompt: '{prompt}' (streaming to terminal)")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        ):
            print(chunk.text, end='', flush=True)  # Stream to terminal in real-time
        
        end_time = time.time()
        
        # Final formatting
        print("\n" + "="*50)
        logger.info(f"Streaming generation completed in {end_time - start_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return False

def start_repl(model_name, adapter_path=None, max_tokens=100, temperature=0.7):
    """Start REPL mode for interactive conversation."""
    print(f"\n🤖 ForgeLLM REPL - Interactive Chat")
    print(f"Model: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print("\nCommands:")
    print("  /help - Show this help")
    print("  /q, /exit, /quit - Exit REPL")
    print("  /save <filename> - Save conversation history")
    print("  /load <filename> - Load conversation history")
    print("  /stats - Show session statistics")
    print("  /system [prompt] - Show/set system prompt")
    print("\nType your message and press Enter to chat!\n")
    
    try:
        # Import here to avoid loading mlx until needed
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        
        # Load the model
        print("Loading model...")
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        print("✅ Model loaded successfully!\n")
        
        # Initialize session state
        conversation_history = []
        system_prompt = "You are a helpful AI assistant."
        session_stats = {
            'turns': 0,
            'prompt_tokens': 0,
            'response_tokens': 0,
            'start_time': datetime.now()
        }
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # REPL loop
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input.split(' ', 1)
                    command = command_parts[0].lower()
                    args = command_parts[1] if len(command_parts) > 1 else ""
                    
                    if command in ['/q', '/exit', '/quit']:
                        print("\n👋 Goodbye!")
                        break
                    
                    elif command == '/help':
                        print("\n📖 Available Commands:")
                        print("  /help - Show this help")
                        print("  /q, /exit, /quit - Exit REPL")
                        print("  /save <filename> - Save conversation history")
                        print("  /load <filename> - Load conversation history")
                        print("  /stats - Show session statistics")
                        print("  /system [prompt] - Show/set system prompt")
                        print()
                        continue
                    
                    elif command == '/stats':
                        duration = datetime.now() - session_stats['start_time']
                        print(f"\n📊 Session Statistics:")
                        print(f"  Duration: {duration}")
                        print(f"  Conversation turns: {session_stats['turns']}")
                        print(f"  Prompt tokens (est): {session_stats['prompt_tokens']}")
                        print(f"  Response tokens (est): {session_stats['response_tokens']}")
                        print(f"  Total tokens (est): {session_stats['prompt_tokens'] + session_stats['response_tokens']}")
                        print()
                        continue
                    
                    elif command == '/system':
                        if args:
                            system_prompt = args
                            print(f"✅ System prompt updated: {system_prompt}\n")
                        else:
                            print(f"📝 Current system prompt: {system_prompt}\n")
                        continue
                    
                    elif command == '/save':
                        if not args:
                            print("❌ Please provide a filename: /save <filename>\n")
                            continue
                        
                        filename = args.strip()
                        if not filename.endswith('.json'):
                            filename += '.json'
                        
                        try:
                            # Use same format as web UI (simplified)
                            # Convert session_stats to JSON-serializable format
                            serializable_stats = {
                                'turns': session_stats['turns'],
                                'prompt_tokens': session_stats['prompt_tokens'],
                                'response_tokens': session_stats['response_tokens'],
                                'start_time': session_stats['start_time'].isoformat(),
                                'duration_seconds': (datetime.now() - session_stats['start_time']).total_seconds()
                            }
                            
                            save_data = {
                                'metadata': {
                                    'model_name': model_name,
                                    'adapter_path': adapter_path,
                                    'system_prompt': system_prompt,
                                    'max_tokens': max_tokens,
                                    'temperature': temperature,
                                    'saved_at': datetime.now().isoformat(),
                                    'session_stats': serializable_stats
                                },
                                'messages': conversation_history
                            }
                            
                            with open(filename, 'w') as f:
                                json.dump(save_data, f, indent=2)
                            
                            print(f"✅ Conversation saved to {filename}\n")
                        except Exception as e:
                            print(f"❌ Error saving conversation: {e}\n")
                        continue
                    
                    elif command == '/load':
                        if not args:
                            print("❌ Please provide a filename: /load <filename>\n")
                            continue
                        
                        filename = args.strip()
                        if not filename.endswith('.json'):
                            filename += '.json'
                        
                        try:
                            with open(filename, 'r') as f:
                                save_data = json.load(f)
                            
                            # Load conversation history
                            conversation_history = save_data.get('messages', [])
                            
                            # Load metadata if available
                            metadata = save_data.get('metadata', {})
                            if 'system_prompt' in metadata:
                                system_prompt = metadata['system_prompt']
                            if 'max_tokens' in metadata:
                                max_tokens = metadata['max_tokens']
                            if 'temperature' in metadata:
                                temperature = metadata['temperature']
                                sampler = make_sampler(temp=temperature)
                            
                            print(f"✅ Conversation loaded from {filename}")
                            print(f"   Loaded {len(conversation_history)} messages")
                            if metadata:
                                print(f"   Model: {metadata.get('model_name', 'unknown')}")
                                print(f"   System: {system_prompt}")
                            print()
                            
                            # Show conversation history
                            if conversation_history:
                                print("📜 Conversation History:")
                                for msg in conversation_history[-5:]:  # Show last 5 messages
                                    role = "👤" if msg['role'] == 'user' else "🤖"
                                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                                    print(f"   {role} {content}")
                                if len(conversation_history) > 5:
                                    print(f"   ... and {len(conversation_history) - 5} more messages")
                                print()
                        except Exception as e:
                            print(f"❌ Error loading conversation: {e}\n")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {command}")
                        print("   Type /help for available commands\n")
                        continue
                
                # Regular chat message
                print("🤖 Assistant: ", end='', flush=True)
                
                # Add user message to history
                conversation_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Build full prompt with system message and conversation history
                full_prompt = f"System: {system_prompt}\n\n"
                for msg in conversation_history:
                    role_name = "Human" if msg['role'] == 'user' else "Assistant"
                    full_prompt += f"{role_name}: {msg['content']}\n"
                full_prompt += "Assistant:"
                
                # Generate response
                start_time = time.time()
                response_text = ""
                
                for chunk in stream_generate(
                    model, 
                    tokenizer, 
                    prompt=full_prompt, 
                    max_tokens=max_tokens,
                    sampler=sampler
                ):
                    print(chunk.text, end='', flush=True)
                    response_text += chunk.text
                
                print("\n")  # New line after response
                
                # Add assistant response to history
                conversation_history.append({
                    'role': 'assistant',
                    'content': response_text.strip()
                })
                
                # Update stats (rough token estimates)
                session_stats['turns'] += 1
                session_stats['prompt_tokens'] += len(full_prompt.split())
                session_stats['response_tokens'] += len(response_text.split())
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue
    
    except Exception as e:
        logger.error(f"Error in REPL mode: {e}")
        return False

if __name__ == "__main__":
    main() 