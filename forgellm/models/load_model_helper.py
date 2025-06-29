import os
import sys
import json
import argparse
import signal
from pathlib import Path
import time

def handle_generate_signal(signum, frame):
    """Handle signal to generate text."""
    try:
        # Check if prompt file exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(script_dir, "prompt.txt")
        params_file = os.path.join(script_dir, "params.json")
        output_file = os.path.join(script_dir, "output.txt")
        
        if not os.path.exists(prompt_file) or not os.path.exists(params_file):
            print("Prompt file or params file not found")
            return
        
        # Read prompt
        with open(prompt_file, 'r') as f:
            prompt = f.read()
        
        # Read params
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Generate text
        from mlx_lm import generate
        
        # Set up generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "max_tokens": params.get("max_tokens", 100),
        }
        
        # Add optional parameters
        if params.get("temperature") is not None and params.get("temperature") > 0:
            generation_kwargs["temp"] = params.get("temperature")
        if params.get("top_p") is not None and 0 < params.get("top_p") < 1.0:
            generation_kwargs["top_p"] = params.get("top_p")
        
        # Handle repetition penalty
        try:
            # Import sampling utilities for newer versions
            from mlx_lm.sample_utils import make_repetition_penalty
            
            if params.get("repetition_penalty") is not None and params.get("repetition_penalty") > 1.0:
                # Create a repetition penalty processor with default context size
                context_size = 20
                rep_penalty_fn = make_repetition_penalty(params.get("repetition_penalty"), context_size=context_size)
                generation_kwargs["logits_processors"] = [rep_penalty_fn]
        except ImportError:
            # Fall back to older method
            if params.get("repetition_penalty") is not None and params.get("repetition_penalty") > 1.0:
                generation_kwargs["repetition_penalty"] = params.get("repetition_penalty")
        
        # Add max_kv_size if provided
        if params.get("max_kv_size") is not None and params.get("max_kv_size") > 0:
            generation_kwargs["max_kv_size"] = params.get("max_kv_size")
        
        # Generate text
        print(f"Generating with parameters: {generation_kwargs}")
        response = generate(model, tokenizer, **generation_kwargs)
        
        # Write output
        with open(output_file, 'w') as f:
            f.write(response)
        
        # Clean up
        os.remove(prompt_file)
        os.remove(params_file)
        
    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        # Write error to output file
        with open(output_file, 'w') as f:
            f.write(f"Error: {e}")

def handle_stop_signal(signum, frame):
    """Handle signal to stop generation."""
    print("Stopping generation")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Load a model in a separate process')
    parser.add_argument('--model', required=True, help='Model name or path')
    parser.add_argument('--adapter', help='Adapter path')
    args = parser.parse_args()
    
    try:
        # Import here to avoid loading mlx until needed
        from mlx_lm import load
        
        print(f"Loading model {args.model} with adapter {args.adapter}")
        global model, tokenizer
        model, tokenizer = load(args.model, adapter_path=args.adapter)
        print("Model loaded successfully")
        
        # Determine model type
        model_type = "full"
        if args.adapter:
            model_type = "lora"  # Default to lora if adapter is used
            
            # Check if adapter config exists
            adapter_dir = Path(args.adapter).parent
            config_path = adapter_dir / "adapter_config.json"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        
                    if "fine_tune_type" in config:
                        model_type = config["fine_tune_type"].lower()
                        
                    # Check for LoRA-specific keys
                    if "lora_rank" in config or "r" in config:
                        model_type = "lora"
                        
                    # Check for DoRA-specific keys
                    if "dora_rank" in config:
                        model_type = "dora"
                except Exception:
                    pass
        
        # Determine chat format
        model_lower = args.model.lower()
        chat_format = "plain"
        
        if "gemma" in model_lower:
            chat_format = "gemma"
        elif "llama" in model_lower or "meta-llama" in model_lower:
            chat_format = "llama"
        elif "mistral" in model_lower:
            chat_format = "mistral"
        elif "qwen" in model_lower:
            chat_format = "qwen"
        elif "phi" in model_lower:
            chat_format = "phi"
        elif "gpt" in model_lower or "openai" in model_lower:
            chat_format = "openai"
        
        # Write ready file
        ready_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_ready.json")
        with open(ready_file, 'w') as f:
            json.dump({
                "model_type": model_type,
                "chat_format": chat_format
            }, f)
        
        # Register signal handlers
        signal.signal(signal.SIGUSR1, handle_generate_signal)
        signal.signal(signal.SIGINT, handle_stop_signal)
        
        # Keep the process running
        while True:
            time.sleep(1)
    
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 