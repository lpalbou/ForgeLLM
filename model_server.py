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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            else:
                is_instruct = is_instruct_model(MODEL_NAME)
                logger.info(f"Model {MODEL_NAME} detected as instruct model: {is_instruct}")
            
            # NEW: Intelligent prompt formatting based on model type and history
            final_prompt = prompt
            is_gemma = is_gemma_model(MODEL_NAME)
            
            if history and is_instruct:
                # INSTRUCT MODEL with history: Use appropriate formatting
                try:
                    # Add current user message to history
                    messages = history + [{"role": "user", "content": prompt}]
                    
                    if is_gemma:
                        # GEMMA-SPECIFIC: Use proper Gemma chat format
                        logger.info("Using Gemma-specific chat formatting")
                        final_prompt = format_gemma_chat(messages)
                        logger.info(f"Gemma chat format result: {final_prompt[:200]}...")
                    elif hasattr(TOKENIZER, 'apply_chat_template') and TOKENIZER.chat_template:
                        # Try to use chat template for other models
                        logger.info("Using tokenizer chat template for INSTRUCT model")
                        final_prompt = TOKENIZER.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        logger.info(f"Chat template result: {final_prompt[:200]}...")
                    else:
                        # Fallback: Manual formatting for INSTRUCT models
                        logger.info("No chat template available, using manual INSTRUCT formatting")
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
                    logger.warning(f"Error applying chat template: {e}, falling back to raw prompt")
                    final_prompt = prompt
                    
            elif history and not is_instruct:
                # BASE MODEL with history: Prompt is already formatted by frontend
                logger.info("Using pre-formatted prompt for BASE model (system prompt already included)")
                final_prompt = prompt
                
            elif legacy_system_prompt and legacy_system_prompt.strip():
                # LEGACY: Handle old system_prompt parameter for backward compatibility
                logger.info(f"Using legacy system prompt: {legacy_system_prompt[:50]}...")
                
                if is_instruct:
                    if is_gemma:
                        # GEMMA-SPECIFIC: Use proper Gemma format for legacy system prompts
                        logger.info("Using Gemma-specific formatting for legacy system prompt")
                        final_prompt = f"<start_of_turn>model\nSystem: {legacy_system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    elif "User:" in prompt and "Assistant:" in prompt:
                        final_prompt = f"System: {legacy_system_prompt}\n\n{prompt}"
                    else:
                        final_prompt = f"System: {legacy_system_prompt}\n\nHuman: {prompt}\nAssistant:"
                else:
                    # For base models, prepend directly
                    final_prompt = f"{legacy_system_prompt}\n\n{prompt}"
                    
            elif is_instruct and not history:
                # INSTRUCT MODEL without history: Add minimal formatting if needed
                if is_gemma:
                    # GEMMA-SPECIFIC: Use proper Gemma format
                    if "<start_of_turn>" not in prompt:
                        final_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    else:
                        final_prompt = prompt
                elif "Human:" not in prompt and "User:" not in prompt and "Assistant:" not in prompt:
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
                
                # Generate and stream text chunks
                for chunk in stream_generate(MODEL, TOKENIZER, prompt=final_prompt, **generation_kwargs):
                    chunk_data = json.dumps({
                        'type': 'chunk',
                        'text': chunk.text,
                        'timestamp': time.time()
                    }) + '\n'
                    self.wfile.write(chunk_data.encode())
                    self.wfile.flush()
                
                # Send completion signal
                end_time = time.time()
                completion_data = json.dumps({
                    'type': 'complete',
                    'generation_time': end_time - start_time
                }) + '\n'
                self.wfile.write(completion_data.encode())
                self.wfile.flush()
            else:
                # Non-streaming response (original behavior)
                response_text = ""
                for chunk in stream_generate(MODEL, TOKENIZER, prompt=final_prompt, **generation_kwargs):
                    response_text += chunk.text
                
                end_time = time.time()
                
                # Clean up the response for instruct models
                if is_instruct:
                    # Remove any repeated patterns or strange artifacts
                    response_text = clean_instruct_response(response_text)
                
                self._set_headers()
                response = {
                    'success': True,
                    'text': response_text,
                    'generation_time': end_time - start_time
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
            model, tokenizer = load(model_name, adapter_path=adapter_path)
        except FileNotFoundError as e:
            if "adapter_config.json" in str(e):
                # Handle missing adapter_config.json by loading without adapter
                logger.warning(f"adapter_config.json not found, loading model without adapter: {e}")
                model, tokenizer = load(model_name, adapter_path=None)
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
    
    # Common patterns for instruct model names
    instruct_patterns = [
        "-it-", "instruct", "-i-", "chat", "-c-", "assistant", "-sft"
    ]
    
    model_name_lower = model_name.lower()
    return any(pattern in model_name_lower for pattern in instruct_patterns)

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