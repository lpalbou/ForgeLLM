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

# Add the project root to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our model architecture manager
try:
    from forgellm.utils.model_architectures import get_model_architecture_manager
    ARCHITECTURE_MANAGER = get_model_architecture_manager()
    logger.info("Successfully loaded ModelArchitectureManager for CLI")
except ImportError as e:
    logger.warning(f"Could not import ModelArchitectureManager: {e}")
    ARCHITECTURE_MANAGER = None

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
    
    # Info command for model architecture information
    info_parser = subparsers.add_parser('info', help='Get information about model architecture and formatting')
    info_parser.add_argument('--model', required=True, help='Model name or path to analyze')
    info_parser.add_argument('--show-example', action='store_true', help='Show example formatted prompt')
    
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
    elif args.command == 'info':
        show_model_info(args.model, args.show_example)
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
        
        # Format prompt using WEB UI LOGIC (copy exact formatting from app.js)
        final_prompt = prompt  # Default fallback
        
        if ARCHITECTURE_MANAGER:
            # Detect if this is a BASE model (copy logic from web UI)
            is_base_model = not ARCHITECTURE_MANAGER.is_instruct_model(model_name)
            architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
            
            logger.info(f"Model {model_name} detected as: {architecture} ({'BASE' if is_base_model else 'INSTRUCT'})")
            
            if is_base_model:
                # BASE MODEL: Use prompt as-is (no chat template formatting)
                final_prompt = prompt
                logger.debug("Using BASE model formatting (prompt as-is)")
            else:
                # INSTRUCT MODEL: Use proper message formatting for chat template
                messages = [{"role": "user", "content": prompt}]
                final_prompt = ARCHITECTURE_MANAGER.format_messages(messages, model_name)
                logger.debug(f"Using {architecture} INSTRUCT formatting")
        else:
            logger.warning("Using fallback formatting (no ModelArchitectureManager)")
        
        # Generate text
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # SOTA CLI streaming: Real-time character-by-character output
        logger.info(f"Generating with formatted prompt (streaming to terminal)")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=final_prompt, 
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
        
        # Format prompt using WEB UI LOGIC (copy exact formatting from app.js)
        final_prompt = prompt  # Default fallback
        
        if ARCHITECTURE_MANAGER:
            # Detect if this is a BASE model (copy logic from web UI)
            is_base_model = not ARCHITECTURE_MANAGER.is_instruct_model(model_name)
            architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
            
            logger.info(f"Model {model_name} detected as: {architecture} ({'BASE' if is_base_model else 'INSTRUCT'})")
            
            if is_base_model:
                # BASE MODEL: Use prompt as-is (no chat template formatting)
                final_prompt = prompt
                logger.debug("Using BASE model formatting (prompt as-is)")
            else:
                # INSTRUCT MODEL: Use proper message formatting for chat template
                messages = [{"role": "user", "content": prompt}]
                final_prompt = ARCHITECTURE_MANAGER.format_messages(messages, model_name)
                logger.debug(f"Using {architecture} INSTRUCT formatting")
        else:
            logger.warning("Using fallback formatting (no ModelArchitectureManager)")
        
        # Generate text
        start_time = time.time()
        
        # Create sampler with proper parameters
        sampler = make_sampler(temp=temperature)
        
        # SOTA CLI streaming: Real-time character-by-character output
        logger.info(f"Generating with formatted prompt (streaming to terminal)")
        print("\n" + "="*50 + "\nGENERATED OUTPUT:\n" + "="*50)
        
        for chunk in stream_generate(
            model, 
            tokenizer, 
            prompt=final_prompt, 
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
    print("  /info - Show model architecture information")
    print("  /format - Show current formatting details")
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
                        print("  /info - Show model architecture information")
                        print("  /format - Show current formatting details")
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
                    
                    elif command == '/info':
                        print(f"\n🔍 Model Architecture Information:")
                        if ARCHITECTURE_MANAGER:
                            architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
                            is_instruct = ARCHITECTURE_MANAGER.is_instruct_model(model_name)
                            arch_info = ARCHITECTURE_MANAGER.get_architecture_info(architecture)
                            
                            print(f"  📋 Architecture: {architecture}")
                            print(f"  🤖 Model Type: {'INSTRUCT' if is_instruct else 'BASE'}")
                            print(f"  💬 System Support: {arch_info.get('system_support_type', 'N/A')}")
                            
                            if "qwen" in model_name.lower():
                                if "base" in model_name.lower():
                                    print(f"  🔥 Special: Qwen BASE model")
                                else:
                                    print(f"  🔥 Special: Qwen INSTRUCT (default)")
                            
                            if architecture == "gemma":
                                print(f"  🔥 Special: Uses assistant turns for system messages")
                        else:
                            print("  ❌ ModelArchitectureManager not available")
                        print()
                        continue
                    
                    elif command == '/format':
                        print(f"\n📄 Current Formatting Details:")
                        if ARCHITECTURE_MANAGER:
                            architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
                            is_instruct = ARCHITECTURE_MANAGER.is_instruct_model(model_name)
                            
                            print(f"  🎯 Detected Architecture: {architecture}")
                            print(f"  🤖 Model Type: {'INSTRUCT' if is_instruct else 'BASE'}")
                            print(f"  💬 System Prompt: '{system_prompt}'")
                            
                            # Show example formatting
                            if system_prompt and system_prompt.strip():
                                example_messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": "Example user message"}
                                ]
                            else:
                                example_messages = [
                                    {"role": "user", "content": "Example user message"}
                                ]
                            
                            if is_instruct:
                                formatted = ARCHITECTURE_MANAGER.format_messages(example_messages, model_name)
                                print(f"  📝 Example Format Preview:")
                                print(f"     {repr(formatted[:100])}...")
                            else:
                                print(f"  📝 BASE model - prompts used as-is")
                        else:
                            print("  ❌ ModelArchitectureManager not available")
                        print()
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
                
                # Build full prompt using WEB UI LOGIC (copy exact formatting from app.js)
                full_prompt = user_input  # Default fallback
                
                if ARCHITECTURE_MANAGER:
                    # Detect if this is a BASE model (copy logic from web UI isCurrentModelBase)
                    is_base_model = not ARCHITECTURE_MANAGER.is_instruct_model(model_name)
                    architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
                    
                    logger.info(f"Model type: {'BASE' if is_base_model else 'INSTRUCT'} ({architecture})")
                    
                    if is_base_model:
                        # BASE MODEL: Use plain text continuation (copy from web UI)
                        logger.debug("Using BASE model formatting (plain text continuation)")
                        history_text = ""
                        
                        # Add system prompt at the beginning if provided
                        if system_prompt and system_prompt.strip():
                            history_text += f"{system_prompt}\n\n"
                        
                        # Add conversation history as plain text
                        for msg in conversation_history[:-1]:  # Exclude current message
                            if msg['role'] == 'user':
                                history_text += f"{msg['content']}\n"
                            elif msg['role'] == 'assistant':
                                history_text += f"{msg['content']}\n"
                        
                        # Final prompt is: system + history + current input
                        full_prompt = history_text + user_input
                        
                    else:
                        # INSTRUCT MODEL: Use proper message format for chat template
                        logger.debug("Using INSTRUCT model formatting (message structure)")
                        messages = []
                        
                        # Add system message if provided
                        if system_prompt and system_prompt.strip():
                            messages.append({
                                'role': 'system',
                                'content': system_prompt
                            })
                        
                        # Add conversation history
                        for msg in conversation_history[:-1]:  # Exclude the current user message
                            messages.append(msg)
                        
                        # Add current user message
                        messages.append({
                            'role': 'user',
                            'content': user_input
                        })
                        
                        # Format using architecture manager
                        full_prompt = ARCHITECTURE_MANAGER.format_messages(messages, model_name)
                    
                else:
                    # Fallback: Basic formatting when architecture manager not available
                    logger.warning("Using fallback formatting (no ModelArchitectureManager)")
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
                
                # Update stats (accurate token counts using the tokenizer)
                session_stats['turns'] += 1
                # Use the actual tokenizer for accurate token counting
                session_stats['prompt_tokens'] += len(tokenizer.encode(full_prompt))
                session_stats['response_tokens'] += len(tokenizer.encode(response_text))
                
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

def show_model_info(model_name, show_example=False):
    """Show information about a model's architecture and formatting."""
    print(f"\n🔍 Model Information: {model_name}")
    print("=" * 60)
    
    if not ARCHITECTURE_MANAGER:
        print("❌ ModelArchitectureManager not available")
        return False
    
    try:
        # Detect architecture and model type
        architecture = ARCHITECTURE_MANAGER.detect_architecture(model_name)
        is_instruct = ARCHITECTURE_MANAGER.is_instruct_model(model_name)
        arch_info = ARCHITECTURE_MANAGER.get_architecture_info(architecture)
        
        # Basic info
        print(f"📋 Architecture: {architecture}")
        print(f"🤖 Model Type: {'INSTRUCT' if is_instruct else 'BASE'}")
        print(f"📝 Description: {arch_info.get('description', 'N/A')}")
        print(f"🔧 Message Format: {arch_info.get('message_format', 'N/A')}")
        print(f"💬 System Support: {arch_info.get('system_support_type', 'N/A')}")
        
        # Special handling notes
        if arch_info.get('system_note'):
            print(f"⚠️  Note: {arch_info['system_note']}")
        
        # Patterns that triggered detection
        patterns = arch_info.get('patterns', [])
        if patterns:
            print(f"🎯 Detection Patterns: {', '.join(patterns)}")
        
        # Special cases
        special_cases = []
        if "qwen" in model_name.lower():
            if "base" in model_name.lower():
                special_cases.append("Qwen BASE model (detected by 'base' in name)")
            else:
                special_cases.append("Qwen INSTRUCT model (default for Qwen)")
        
        if architecture == "gemma":
            special_cases.append("Uses assistant turns for system messages")
        
        if special_cases:
            print(f"🔥 Special Handling: {'; '.join(special_cases)}")
        
        # Show example formatting if requested
        if show_example:
            print("\n📄 Example Formatting:")
            print("-" * 40)
            
            # Example with system prompt
            if is_instruct:
                example_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ]
                formatted = ARCHITECTURE_MANAGER.format_messages(example_messages, model_name)
                print("With system prompt:")
                print(repr(formatted))
                print()
                
                # Example without system prompt
                example_messages_no_sys = [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
                formatted_no_sys = ARCHITECTURE_MANAGER.format_messages(example_messages_no_sys, model_name)
                print("Without system prompt:")
                print(repr(formatted_no_sys))
            else:
                print("BASE model - prompt used as-is:")
                print(repr("Hello, how are you?"))
        
        print("\n✅ Analysis complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return False

if __name__ == "__main__":
    main() 