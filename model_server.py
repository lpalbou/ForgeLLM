#!/usr/bin/env python
"""
Simple HTTP server for model inference.
"""

import os
import sys
import time
import json
import logging
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import threading
import traceback

# Add the forgellm package to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # model_server.py is in the root, so current_dir IS the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the ModelArchitectureManager
try:
    from forgellm.utils.model_architectures import get_model_architecture_manager
    ARCHITECTURE_MANAGER = get_model_architecture_manager()
    logger.info("Successfully loaded ModelArchitectureManager")
except ImportError as e:
    logger.warning(f"Could not import ModelArchitectureManager: {e}")
    logger.warning(f"Current working directory: {os.getcwd()}")
    logger.warning(f"Python path: {sys.path[:3]}")
    ARCHITECTURE_MANAGER = None

# Global variables
MODEL = None
TOKENIZER = None
MODEL_NAME = None
ADAPTER_PATH = None
IS_LOADING = False
LOADING_ERROR = None

class ModelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for model inference."""
    
    def _set_headers(self, status_code=200, content_type='application/json'):
        """Set response headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self._set_headers()
        self.wfile.write(b'')
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith('/api/model/status'):
            self._handle_status()
        else:
            self._set_headers(404)
            response = {'success': False, 'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(post_data)
        except json.JSONDecodeError:
            self._set_headers(400)
            response = {'success': False, 'error': 'Invalid JSON'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        if self.path.startswith('/api/model/load'):
            self._handle_load(data)
        elif self.path.startswith('/api/model/generate'):
            self._handle_generate(data)
        else:
            self._set_headers(404)
            response = {'success': False, 'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())
    
    def _handle_status(self):
        """Handle model status requests."""
        global MODEL, MODEL_NAME, ADAPTER_PATH, IS_LOADING, LOADING_ERROR
        
        # Basic response
        response = {
            'success': True,
            'loaded': MODEL is not None,
            'is_loading': IS_LOADING,
            'model_name': MODEL_NAME,
            'adapter_path': ADAPTER_PATH
        }
        
        # Add error if there is one
        if LOADING_ERROR:
            response['error'] = str(LOADING_ERROR)
        
        self._set_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_load(self, data):
        """Handle model loading requests."""
        global MODEL, TOKENIZER, MODEL_NAME, ADAPTER_PATH, IS_LOADING, LOADING_ERROR
        
        model_name = data.get('model_name')
        adapter_path = data.get('adapter_path')
        
        if not model_name:
            self._set_headers(400)
            response = {'success': False, 'error': 'Missing model_name'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Start loading in a separate thread
        IS_LOADING = True
        LOADING_ERROR = None
        MODEL_NAME = model_name
        ADAPTER_PATH = adapter_path
        
        threading.Thread(target=load_model, args=(model_name, adapter_path)).start()
        
        self._set_headers()
        response = {
            'success': True,
            'message': f'Model {model_name} loading started',
            'model_name': model_name,
            'adapter_path': adapter_path
        }
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_generate(self, data):
        """Handle text generation requests."""
        global MODEL, TOKENIZER, MODEL_NAME
        
        if not MODEL or not TOKENIZER:
            self._set_headers(400)
            response = {'success': False, 'error': 'No model loaded'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        prompt = data.get('prompt')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        repetition_penalty = data.get('repetition_penalty', 1.1)
        max_kv_size = data.get('max_kv_size')
        streaming = data.get('streaming', False)
        
        # NEW: Handle history array and model type hint from frontend
        history = data.get('history', [])
        is_base_model_hint = data.get('is_base_model', None)
        
        # LEGACY: Still support old system_prompt parameter for backward compatibility
        legacy_system_prompt = data.get('system_prompt', '')
        
        if not prompt:
            self._set_headers(400)
            response = {'success': False, 'error': 'Missing prompt'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        try:
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
            
            # Detect if this is an instruct model (use hint if available)
            if is_base_model_hint is not None:
                is_instruct = not is_base_model_hint
                logger.info(f"Using frontend hint: Model {MODEL_NAME} is {'BASE' if is_base_model_hint else 'INSTRUCT'}")
            elif ARCHITECTURE_MANAGER:
                is_instruct = ARCHITECTURE_MANAGER.is_instruct_model(MODEL_NAME)
                logger.info(f"Model {MODEL_NAME} detected as instruct model: {is_instruct} (via ArchitectureManager)")
            else:
                is_instruct = is_instruct_model(MODEL_NAME)
                logger.info(f"Model {MODEL_NAME} detected as instruct model: {is_instruct} (fallback detection)")
            
            # NEW: Intelligent prompt formatting using ModelArchitectureManager
            final_prompt = prompt
            
            if history and is_instruct:
                # INSTRUCT MODEL with history: Use architecture-specific formatting
                try:
                    # CRITICAL FIX: Transform system messages for models that don't support them
                    transformed_history = history
                    logger.info(f"🔍 DEBUG: Starting transformation logic for {len(history)} messages")
                    logger.info(f"🔍 DEBUG: ARCHITECTURE_MANAGER available: {ARCHITECTURE_MANAGER is not None}")
                    
                    if ARCHITECTURE_MANAGER:
                        architecture = ARCHITECTURE_MANAGER.detect_architecture(MODEL_NAME)
                        arch_config = ARCHITECTURE_MANAGER.get_architecture_config(architecture)
                        system_as_assistant = arch_config.get("system_as_assistant", False)
                        
                        logger.info(f"🔍 DEBUG: Architecture: {architecture}, system_as_assistant: {system_as_assistant}")
                        
                        # If this architecture treats system messages as assistant turns (e.g., Gemma)
                        if system_as_assistant:
                            logger.info(f"🔄 TRANSFORMING system messages to assistant messages for {architecture}")
                            transformed_history = []
                            for msg in history:
                                if msg.get("role") == "system":
                                    # Convert system message to assistant message (Gemma speaks as itself)
                                    transformed_msg = {
                                        "role": "assistant", 
                                        "content": msg.get('content', '')
                                    }
                                    transformed_history.append(transformed_msg)
                                    logger.info(f"✅ Transformed: {msg} -> {transformed_msg}")
                                else:
                                    transformed_history.append(msg)
                                    logger.info(f"➡️  Kept as-is: {msg}")
                        else:
                            logger.info(f"❌ No transformation applied (system_as_assistant = {system_as_assistant})")
                    else:
                        logger.warning("❌ ARCHITECTURE_MANAGER not available for transformation")
                    
                    # Add current user message to transformed history
                    messages = transformed_history + [{"role": "user", "content": prompt}]
                    
                    if ARCHITECTURE_MANAGER:
                        # Use the architecture manager for proper formatting
                        logger.info(f"Using ModelArchitectureManager for formatting {len(messages)} messages")
                        final_prompt = ARCHITECTURE_MANAGER.format_messages(messages, MODEL_NAME)
                        logger.info(f"Architecture-based format result: {final_prompt[:200]}...")
                    elif hasattr(TOKENIZER, 'apply_chat_template') and TOKENIZER.chat_template:
                        # Fallback: Try to use tokenizer chat template
                        logger.info("Using tokenizer chat template for INSTRUCT model")
                        final_prompt = TOKENIZER.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        logger.info(f"Chat template result: {final_prompt[:200]}...")
                    else:
                        # Last resort: Manual formatting for INSTRUCT models
                        logger.info("No architecture manager or chat template available, using manual INSTRUCT formatting")
                        formatted_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                formatted_messages.append(f"System: {msg['content']}")
                            elif msg["role"] == "user":
                                formatted_messages.append(f"Human: {msg['content']}")
                            elif msg["role"] == "assistant":
                                formatted_messages.append(f"Assistant: {msg['content']}")
                        final_prompt = "\n".join(formatted_messages) + "\nAssistant:"
                        
                except Exception as e:
                    logger.warning(f"Error applying formatting: {e}, falling back to raw prompt")
                    final_prompt = prompt
                    
            elif history and not is_instruct:
                # BASE MODEL with history: Prompt is already formatted by frontend
                logger.info("Using pre-formatted prompt for BASE model (system prompt already included)")
                final_prompt = prompt
                
            elif legacy_system_prompt and legacy_system_prompt.strip():
                # LEGACY: Handle old system_prompt parameter for backward compatibility
                logger.info(f"Using legacy system prompt: {legacy_system_prompt[:50]}...")
                
                if is_instruct and ARCHITECTURE_MANAGER:
                    # Use architecture manager for legacy system prompts
                    logger.info("Using ModelArchitectureManager for legacy system prompt formatting")
                    final_prompt = ARCHITECTURE_MANAGER.format_single_turn(prompt, legacy_system_prompt, MODEL_NAME)
                elif is_instruct:
                    # Fallback formatting for instruct models
                    if "User:" in prompt and "Assistant:" in prompt:
                        final_prompt = f"System: {legacy_system_prompt}\n\n{prompt}"
                    else:
                        final_prompt = f"System: {legacy_system_prompt}\n\nHuman: {prompt}\nAssistant:"
                else:
                    # For base models, prepend directly
                    final_prompt = f"{legacy_system_prompt}\n\n{prompt}"
                    
            elif is_instruct and not history:
                # INSTRUCT MODEL without history: Add minimal formatting if needed
                if ARCHITECTURE_MANAGER:
                    # Use architecture manager for single-turn formatting
                    logger.info("Using ModelArchitectureManager for single-turn INSTRUCT formatting")
                    final_prompt = ARCHITECTURE_MANAGER.format_single_turn(prompt, "", MODEL_NAME)
                elif "Human:" not in prompt and "User:" not in prompt and "Assistant:" not in prompt:
                    # Fallback: Add basic instruct formatting
                    final_prompt = f"Human: {prompt}\nAssistant:"
                else:
                    final_prompt = prompt
            else:
                # BASE MODEL without history or system prompt: Use as-is
                final_prompt = prompt
            
            logger.info(f"Final prompt being used: {final_prompt[:200]}...")
            
            # Create sampler with proper parameters
            sampler = make_sampler(temp=temperature, top_p=top_p)
            
            # Create repetition penalty processor if specified
            logits_processors = []
            if repetition_penalty and repetition_penalty != 1.0:
                repetition_processor = make_repetition_penalty(penalty=repetition_penalty)
                logits_processors.append(repetition_processor)
            
            start_time = time.time()
            
            # Prepare generation kwargs
            generation_kwargs = {
                'max_tokens': max_tokens,
                'sampler': sampler
            }
            
            if logits_processors:
                generation_kwargs['logits_processors'] = logits_processors
                
            if max_kv_size:
                generation_kwargs['max_kv_size'] = max_kv_size
            
            if streaming:
                # Streaming response
                self._set_headers(content_type='text/plain')
                
                # Count prompt tokens
                prompt_tokens = len(TOKENIZER.encode(final_prompt))
                completion_text = ""
                
                # Generate and stream text chunks
                for chunk in stream_generate(MODEL, TOKENIZER, prompt=final_prompt, **generation_kwargs):
                    completion_text += chunk.text
                    chunk_data = json.dumps({
                        'type': 'chunk',
                        'text': chunk.text,
                        'timestamp': time.time()
                    }) + '\n'
                    self.wfile.write(chunk_data.encode())
                    self.wfile.flush()
                
                # Count completion tokens
                completion_tokens = len(TOKENIZER.encode(completion_text))
                
                # Send completion signal with token counts
                end_time = time.time()
                completion_data = json.dumps({
                    'type': 'complete',
                    'generation_time': end_time - start_time,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }) + '\n'
                self.wfile.write(completion_data.encode())
                self.wfile.flush()
            else:
                # Non-streaming response (original behavior)
                response_text = ""
                chunk_count = 0
                for chunk in stream_generate(MODEL, TOKENIZER, prompt=final_prompt, **generation_kwargs):
                    chunk_count += 1
                    logger.info(f"Chunk {chunk_count}: '{chunk.text}'")
                    response_text += chunk.text
                logger.info(f"Total chunks received: {chunk_count}, total length: {len(response_text)}")
                
                end_time = time.time()
                
                # Clean up the response for instruct models
                if is_instruct:
                    # Remove any repeated patterns or strange artifacts
                    logger.info(f"Response before cleaning: {response_text[:100]}...")
                    response_text = clean_instruct_response(response_text)
                    logger.info(f"Response after cleaning: {response_text[:100]}...")
                
                # Count tokens
                prompt_tokens = len(TOKENIZER.encode(final_prompt))
                completion_tokens = len(TOKENIZER.encode(response_text))
                
                self._set_headers()
                response = {
                    'success': True,
                    'text': response_text,
                    'generation_time': end_time - start_time,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }
                self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            import traceback
            traceback.print_exc()
            self._set_headers(500)
            response = {'success': False, 'error': str(e)}
            self.wfile.write(json.dumps(response).encode())

def load_model(model_name, adapter_path=None):
    """Load a model in a separate thread."""
    global MODEL, TOKENIZER, IS_LOADING, LOADING_ERROR
    
    try:
        logger.info(f"Loading model {model_name} with adapter {adapter_path}")
        
        # Convert published model names to actual local paths
        actual_model_path = model_name
        if model_name.startswith('published/'):
            # Convert published/model_name/timestamp to actual cache path
            from pathlib import Path
            cache_path = Path.home() / '.cache' / 'huggingface' / 'hub'
            
            # Extract the model parts
            published_parts = model_name.replace('published/', '').split('/')
            if len(published_parts) >= 2:
                model_part = published_parts[0]
                timestamp_part = published_parts[1]
                
                # Look for the actual directory
                expected_dir = f"models--published--{model_part}--{timestamp_part}"
                actual_path = cache_path / expected_dir
                
                if actual_path.exists():
                    actual_model_path = str(actual_path)
                    logger.info(f"Converted published model path: {model_name} -> {actual_model_path}")
                else:
                    logger.warning(f"Published model directory not found: {actual_path}")
        
        # Unload previous model first to free memory
        if MODEL is not None:
            logger.info("Unloading previous model to free memory")
            MODEL = None
            TOKENIZER = None
            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
                logger.info("Cleared MLX metal cache")
            except:
                pass
        
        # Import here to avoid loading mlx until needed
        from mlx_lm import load
        
        start_time = time.time()
        
        try:
            # Try to load with adapter first
            model, tokenizer = load(actual_model_path, adapter_path=adapter_path)
        except FileNotFoundError as e:
            if "adapter_config.json" in str(e):
                # Handle missing adapter_config.json by loading without adapter
                logger.warning(f"adapter_config.json not found, loading model without adapter: {e}")
                model, tokenizer = load(actual_model_path, adapter_path=None)
            else:
                # Re-raise if it's a different file not found error
                raise
                
        end_time = time.time()
        
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Update global variables
        MODEL = model
        TOKENIZER = tokenizer
        IS_LOADING = False
        LOADING_ERROR = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        IS_LOADING = False
        LOADING_ERROR = str(e)

def is_instruct_model(model_name):
    """Detect if a model is an instruct model based on its name."""
    if not model_name:
        return False
    
    model_name_lower = model_name.lower()
    
    # Special handling for Qwen models: they are instruct by default EXCEPT if "base" is in the name
    if "qwen" in model_name_lower:
        # For Qwen models, check if it's explicitly marked as base
        if "base" in model_name_lower:
            logger.info(f"Qwen model '{model_name}' detected as BASE (contains 'base')")
            return False
        else:
            logger.info(f"Qwen model '{model_name}' detected as INSTRUCT (default for Qwen)")
            return True
    
    # First check for explicit BASE model patterns
    base_patterns = [
        "base", "pt", "pretrained", "pre-trained", "foundation", 
        "raw", "vanilla", "untuned", "completion"
    ]
    
    if any(pattern in model_name_lower for pattern in base_patterns):
        logger.info(f"Model '{model_name}' detected as BASE (contains base pattern)")
        return False
    
    # Then check for INSTRUCT model patterns
    instruct_patterns = [
        "-it-", "_it_", "instruct", "-i-", "chat", "-c-", "assistant", "-sft", 
        "it", "dpo", "rlhf", "alpaca", "vicuna", "wizard", "conversation"
    ]
    
    if any(pattern in model_name_lower for pattern in instruct_patterns):
        logger.info(f"Model '{model_name}' detected as INSTRUCT (contains instruct pattern)")
        return True
    
    # Default: if no clear pattern, assume BASE (safer for unknown models)
    logger.info(f"Model '{model_name}' detected as BASE (no clear pattern, defaulting to BASE)")
    return False

def is_gemma_model(model_name):
    """Detect if a model is a Gemma model based on its name."""
    if not model_name:
        return False
    
    model_name_lower = model_name.lower()
    gemma_patterns = ["gemma", "recurrentgemma"]
    return any(pattern in model_name_lower for pattern in gemma_patterns)

def format_gemma_chat(messages):
    """Format messages for Gemma models using proper start_of_turn/end_of_turn tokens."""
    formatted_parts = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            # For Gemma, system messages are formatted as model turns with special content
            formatted_parts.append(f"<start_of_turn>model\nSystem: {content}<end_of_turn>")
        elif role == "user":
            formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    
    # Add the generation prompt for the next model turn
    formatted_prompt = "\n".join(formatted_parts) + "\n<start_of_turn>model\n"
    return formatted_prompt

def clean_instruct_response(text):
    """Clean up the response from an instruct model."""
    # Remove repetitive patterns
    if "I am glad that you are happy. I am not able to do that." in text:
        # Find the first occurrence and cut off after that
        idx = text.find("I am glad that you are happy. I am not able to do that.")
        if idx > 0:
            text = text[:idx]
    
    # Remove Korean characters and other artifacts that sometimes appear
    import re
    text = re.sub(r'데이트\?+', '', text)
    text = re.sub(r'ylene\)\?+', '', text)
    
    # Remove repetitive question marks
    text = re.sub(r'\?{2,}', '?', text)
    
    # Remove any trailing non-English text
    text = re.sub(r'[^\x00-\x7F]+$', '', text)
    
    return text.strip()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple HTTP server for model inference")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--model", help="Model to preload")
    parser.add_argument("--adapter", help="Adapter to preload")
    
    args = parser.parse_args()
    
    # Preload model if specified
    if args.model:
        logger.info(f"Preloading model {args.model}")
        load_model(args.model, args.adapter)
    
    # Start server
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, ModelHandler)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        httpd.server_close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 