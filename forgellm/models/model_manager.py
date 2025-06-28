"""
Model Manager - Handles loading, saving, and inference for models
"""

import os
import logging
import threading
import gc
import concurrent.futures
import subprocess
import json
import time
import signal
from typing import Dict, List, Optional, Any, Tuple, Union
import glob
from pathlib import Path
import shutil
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate
from .model_publisher import ModelPublisher
import psutil
import requests

# Configure logging
logger = logging.getLogger(__name__)


class ModelManager:
    """
    ModelManager class for managing models.
    This implementation uses a separate process for model loading and inference.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one ModelManager instance."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the ModelManager."""
        if self._initialized:
            return
        
        # Set up models directory
        self.models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models'))
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
        
        # Create model directories if they don't exist
        self.base_models_dir = os.path.join(self.models_dir, 'base')
        self.cpt_models_dir = os.path.join(self.models_dir, 'cpt')
        self.ift_models_dir = os.path.join(self.models_dir, 'ift')
        
        os.makedirs(self.base_models_dir, exist_ok=True)
        logger.info(f"Created model directory: {self.base_models_dir}")
        
        os.makedirs(self.cpt_models_dir, exist_ok=True)
        logger.info(f"Created model directory: {self.cpt_models_dir}")
        
        os.makedirs(self.ift_models_dir, exist_ok=True)
        logger.info(f"Created model directory: {self.ift_models_dir}")
        
        # Model server settings
        self.server_host = os.environ.get('MODEL_SERVER_HOST', 'localhost')
        self.server_port = int(os.environ.get('MODEL_SERVER_PORT', 5001))
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_path = None
        self.loaded = False
        self.loading = False
        self.error = None
        
        # Server process
        self.server_process = None
        
        # Start the model server if not already running
        self._ensure_server_running()
        
        self._initialized = True
        print("Initialized ModelManager")
    
    def _ensure_server_running(self):
        """Ensure the model server is running."""
        try:
            response = requests.get(f"{self.server_url}/api/model/status", timeout=1)
            if response.status_code == 200:
                logger.info("Model server is already running")
                return
        except:
            logger.info("Model server is not running, starting it")
        
        # Start the model server
        model_server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model_server.py')
        
        if not os.path.exists(model_server_path):
            logger.error(f"Model server script not found at {model_server_path}")
            raise FileNotFoundError(f"Model server script not found at {model_server_path}")
        
        cmd = [
            "python", 
            model_server_path, 
            "--host", self.server_host, 
            "--port", str(self.server_port)
        ]
        
        logger.info(f"Starting model server with command: {' '.join(cmd)}")
        
        # Start the server as a subprocess
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        for _ in range(10):
            try:
                response = requests.get(f"{self.server_url}/api/model/status", timeout=1)
                if response.status_code == 200:
                    logger.info("Model server started successfully")
                    return
            except:
                time.sleep(0.5)
        
        logger.error("Failed to start model server")
        raise RuntimeError("Failed to start model server")
    
    def load(self, model_name, adapter_path=None):
        """
        Load a model.
        
        Args:
            model_name (str): Name of the model to load.
            adapter_path (str, optional): Path to the adapter weights.
        """
        # Check if the model server is running
        self._ensure_server_running()
        
        # Send request to load the model
        data = {
            'model_name': model_name,
            'adapter_path': adapter_path
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/model/load",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.info(f"Model {model_name} loading started")
                    
                    # Update state
                    self.model_name = model_name
                    self.adapter_path = adapter_path
                    self.loading = True
                    self.loaded = False
                    self.error = None
                    
                    # Start a thread to check loading status
                    threading.Thread(target=self._check_loading_status).start()
                    
                    return True
                else:
                    logger.error(f"Failed to load model: {result.get('error')}")
                    self.error = result.get('error')
                    return False
            else:
                logger.error(f"Failed to load model: {response.status_code} {response.text}")
                self.error = f"HTTP error: {response.status_code}"
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.error = str(e)
            return False
    
    def _check_loading_status(self):
        """Check the loading status of the model."""
        while True:
            try:
                response = requests.get(f"{self.server_url}/api/model/status", timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('loaded'):
                        logger.info("Model loaded successfully")
                        self.loading = False
                        self.loaded = True
                        self.error = None
                        return
                    elif result.get('error'):
                        logger.error(f"Error loading model: {result.get('error')}")
                        self.loading = False
                        self.loaded = False
                        self.error = result.get('error')
                        return
                    elif not result.get('is_loading'):
                        # Not loading anymore but not loaded either, must be an error
                        logger.error("Model loading failed")
                        self.loading = False
                        self.loaded = False
                        self.error = "Model loading failed"
                        return
                
                # Still loading, wait and check again
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error checking loading status: {e}")
                self.loading = False
                self.loaded = False
                self.error = str(e)
                return
    
    def unload(self):
        """Unload the model."""
        if not self.loaded and not self.loading:
            logger.info("No model loaded")
            return True
        
        # Reset state
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_path = None
        self.loaded = False
        self.loading = False
        self.error = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Model unloaded")
        return True
    
    def generate(self, prompt, max_tokens=100, temperature=0.7, history=None, top_p=None, repetition_penalty=None, system_prompt=None, max_kv_size=None):
        """
        Generate text from the model.
        
        Args:
            prompt (str): Prompt to generate from.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Temperature for sampling.
            history (list, optional): Chat history for conversation models.
            top_p (float, optional): Top-p sampling parameter.
            repetition_penalty (float, optional): Penalty for repeating tokens.
            system_prompt (str, optional): System prompt for chat models.
            max_kv_size (int, optional): Maximum KV cache size.
        
        Returns:
            str: Generated text.
        """
        if not self.loaded:
            if self.loading:
                logger.error("Model is still loading")
                return "Error: Model is still loading"
            else:
                logger.error("No model loaded")
                return "Error: No model loaded"
        
        # Send request to generate text
        data = {
            'prompt': prompt,
            'max_tokens': max_tokens
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            data['temperature'] = temperature
        if history is not None:
            data['history'] = history
        if top_p is not None:
            data['top_p'] = top_p
        if repetition_penalty is not None:
            data['repetition_penalty'] = repetition_penalty
        if system_prompt is not None:
            data['system_prompt'] = system_prompt
        if max_kv_size is not None:
            data['max_kv_size'] = max_kv_size
        
        try:
            response = requests.post(
                f"{self.server_url}/api/model/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('text')
                else:
                    logger.error(f"Failed to generate text: {result.get('error')}")
                    return f"Error: {result.get('error')}"
            else:
                logger.error(f"Failed to generate text: {response.status_code} {response.text}")
                return f"Error: HTTP error {response.status_code}"
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def get_status(self):
        """
        Get the status of the model.
        
        Returns:
            dict: Status information.
        """
        try:
            response = requests.get(f"{self.server_url}/api/model/status", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get status: {response.status_code} {response.text}")
                return {
                    'success': False,
                    'error': f"HTTP error: {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_models(self):
        """
        List available models.
        
        Returns:
            dict: Dictionary with lists of models.
        """
        base_models = []
        cpt_models = []
        ift_models = []
        
        # List base models
        if os.path.exists(self.base_models_dir):
            base_models = [d for d in os.listdir(self.base_models_dir) if os.path.isdir(os.path.join(self.base_models_dir, d))]
        
        # List CPT models
        if os.path.exists(self.cpt_models_dir):
            cpt_models = [d for d in os.listdir(self.cpt_models_dir) if os.path.isdir(os.path.join(self.cpt_models_dir, d))]
        
        # List IFT models
        if os.path.exists(self.ift_models_dir):
            ift_models = [d for d in os.listdir(self.ift_models_dir) if os.path.isdir(os.path.join(self.ift_models_dir, d))]
        
        return {
            'base': base_models,
            'cpt': cpt_models,
            'ift': ift_models
        }

    def _create_model_dirs(self):
        """Create model directories if they don't exist."""
        for model_type in ['base', 'cpt', 'ift']:
            model_dir = os.path.join(self.models_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")

    def load_model(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        """Load a model into memory (alias for load).
        
        Args:
            model_name: The name or path of the model to load
            adapter_path: Optional path to adapter weights
        """
        return self.load(model_name, adapter_path)

    def unload_model(self) -> None:
        """Unload the model from memory (alias for unload)."""
        return self.unload()

    def generate_text(self, params: Dict[str, Any]) -> str:
        """Generate text from the model using a parameters dictionary.
        
        Args:
            params: Dictionary of generation parameters
                - prompt: The prompt to generate from
                - history: Optional chat history
                - max_tokens: Maximum number of tokens to generate
                - temperature: Sampling temperature (higher = more random)
                - top_p: Nucleus sampling parameter (lower = more focused)
                - repetition_penalty: Penalty for repeating tokens
                - system_prompt: Optional system prompt for chat models
            
        Returns:
            The generated text
        """
        prompt = params.get('prompt', '')
        history = params.get('history')
        max_tokens = params.get('max_tokens', 100)
        temperature = params.get('temperature')
        top_p = params.get('top_p')
        repetition_penalty = params.get('repetition_penalty')
        system_prompt = params.get('system_prompt')
        max_kv_size = params.get('max_kv_size')
        
        return self.generate(
            prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            system_prompt=system_prompt,
            max_kv_size=max_kv_size
        )

    def stop_generation(self) -> None:
        """Stop the current generation."""
        # Send SIGINT to the model process
        if self.model_process and self.model_process.poll() is None:
            os.kill(self.model_process.pid, signal.SIGINT)

    def memory_usage_gb(self) -> float:
        """Get the current memory usage in GB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'loaded': self.loaded,
            'model_name': self.model_name,
            'adapter_path': self.adapter_path,
            'model_type': getattr(self, 'model_type', None),
            'chat_format': getattr(self, 'chat_format', 'plain'),
            'memory_usage_gb': self.memory_usage_gb()
        }

    def _init(self):
        """Initialize instance variables."""
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_type = None
        self.model_config = None
        self.adapter_path = None
        self.loaded = False
        self.model_name = None
        self.chat_format = "plain"
        logger.info("ModelManager initialized")

    def __call__(self):  # noqa: D401  (simple method)
        """Allow using the singleton as a callable."""
        return self
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model path, handling published models and HF cache."""
        logger.info(f"Resolving model path for: {model_name}")
        
        # If it's a local path that exists, return it directly
        if Path(model_name).exists():
            logger.info(f"Found local path: {model_name}")
            return model_name
            
        # Check if it's a published model
        published_path = Path("published") / model_name
        if published_path.exists():
            logger.info(f"Found published model: {published_path}")
            return str(published_path)
            
        # Check HuggingFace cache
        try:
            cache_root = Path.home() / '.cache' / 'huggingface' / 'hub'
            candidate = cache_root / ('models--' + model_name.replace('/', '--'))
            if candidate.exists():
                logger.info(f"Found model in HF cache: {candidate}")
                return str(candidate)
            else:
                logger.info(f"Model not found in HF cache, will download from HF Hub: {model_name}")
        except Exception as e:
            logger.warning(f"Error checking HF cache: {e}")
            
        # Return original name (might be a HF model ID)
        logger.info(f"Using original model name: {model_name}")
        return model_name

    def _is_base_model(self, model_name: str) -> bool:
        """Detect if a model is a base model (not instruction-tuned) using SOTA practices.
        
        This method checks:
        1. HuggingFace model tags and metadata
        2. Model card content and descriptions  
        3. Standard naming conventions
        """
        try:
            # Try to get model info from HuggingFace Hub
            from huggingface_hub import model_info, hf_hub_download
            
            try:
                info = model_info(model_name)
                
                # Check model tags - most reliable method
                if hasattr(info, 'tags') and info.tags:
                    # Instruction-tuned models typically have these tags
                    instruct_tags = {
                        'conversational', 'chat', 'instruct', 'instruction-tuned',
                        'instruction-following', 'assistant', 'dialogue'
                    }
                    
                    # Convert tags to lowercase for comparison
                    model_tags = {tag.lower() for tag in info.tags}
                    
                    # If any instruct tags are found, it's NOT a base model
                    if model_tags.intersection(instruct_tags):
                        return False
                
                # Check model card description
                if hasattr(info, 'cardData') and info.cardData:
                    card_text = str(info.cardData).lower()
                    instruct_keywords = [
                        'instruction', 'chat', 'conversational', 'assistant',
                        'dialogue', 'instruct', 'fine-tuned', 'aligned'
                    ]
                    
                    if any(keyword in card_text for keyword in instruct_keywords):
                        return False
                        
            except Exception:
                # If HuggingFace Hub access fails, fall back to name-based detection
                pass
                
        except ImportError:
            # If huggingface_hub is not available, use name-based detection
            pass
        
        # Fallback: Name-based detection (less reliable but works offline)
        name_lower = model_name.lower()
        
        # Common patterns that indicate instruction-tuned models
        instruct_patterns = [
            'instruct', 'chat', 'it', 'sft', 'dpo', 'rlhf', 
            'assistant', 'alpaca', 'vicuna', 'wizard', 'orca',
            'dolphin', 'openhermes', 'airoboros', 'nous',
            'claude', 'gpt', 'turbo', 'dialogue', 'conversation'
        ]
        
        # If any instruct pattern is found, it's NOT a base model
        for pattern in instruct_patterns:
            if pattern in name_lower:
                return False
        
        # Additional check: if name contains 'base' explicitly, it's likely a base model
        if 'base' in name_lower:
            return True
            
        # Default assumption: if no clear instruct indicators, treat as base model
        # This is safer for text completion use cases
        return True

    def _format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format the prompt for the model based on model type."""
        # Check if this is a base model
        if hasattr(self, 'model_name') and self.model_name and self._is_base_model(self.model_name):
            # For base models, use raw text completion
            if system_prompt:
                return f"{system_prompt}\n\n{user_prompt}"
            else:
                return user_prompt
        
        # For instruct models, use chat formatting
        # Handle different chat formats
        if self.chat_format == "gemma":
            # Gemma format
            if system_prompt:
                return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                return f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif self.chat_format == "llama":
            # Llama format
            if system_prompt:
                return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            else:
                return f"<s>[INST] {user_prompt} [/INST]"
        elif self.chat_format == "mistral":
            # Mistral format
            if system_prompt:
                return f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
            else:
                return f"<s>[INST] {user_prompt} [/INST]"
        elif self.chat_format == "qwen":
            # Qwen format
            system_content = system_prompt or "You are a helpful assistant."
            return (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif self.chat_format == "phi":
            # Phi format
            if system_prompt:
                return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            else:
                return f"<|user|>\n{user_prompt}\n<|assistant|>\n"
        elif self.chat_format == "openai":
            # OpenAI format
            if system_prompt:
                return f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"
            else:
                return f"user: {user_prompt}\nassistant:"
        else:
            # Default format for instruct models (simple User/Assistant format)
            if system_prompt:
                return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                return f"User: {user_prompt}\nAssistant:"

    def _build_conversation(self, history: list[dict], user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Build a conversation prompt from history."""
        # Check if this is a base model
        if hasattr(self, 'model_name') and self.model_name and self._is_base_model(self.model_name):
            # For base models, build a simple text continuation
            conversation = ""
            if system_prompt:
                conversation += f"{system_prompt}\n\n"
                
            # Add history as natural text flow
            for message in history:
                content = message.get("content", "")
                conversation += f"{content}\n"
                
            # Add the new user prompt
            conversation += user_prompt
            return conversation
        
        # For instruct models, use chat formatting
        # Handle different chat formats
        if self.chat_format == "gemma":
            # Gemma format
            conversation = ""
            if system_prompt:
                conversation += f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
                
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
                
            conversation += f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            return conversation
            
        elif self.chat_format == "llama":
            # Llama format
            conversation = "<s>"
            if system_prompt:
                conversation += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            else:
                conversation += "[INST] "
                
            # Add alternating messages
            for i, message in enumerate(history):
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    if i > 0:
                        conversation += "[INST] "
                    conversation += f"{content} [/INST]"
                else:
                    conversation += f" {content} "
                    
            # Add final user prompt
            conversation += f"[INST] {user_prompt} [/INST]"
            return conversation
            
        elif self.chat_format == "mistral":
            # Mistral format
            conversation = "<s>"
            
            # Process history in pairs
            i = 0
            while i < len(history) - 1:
                user_msg = history[i].get("content", "") if history[i].get("role") == "user" else ""
                assistant_msg = history[i+1].get("content", "") if history[i+1].get("role") == "assistant" else ""
                
                if user_msg and assistant_msg:
                    conversation += f"[INST] {user_msg} [/INST] {assistant_msg} "
                
                i += 2
            
            # Add final user prompt with system instruction if provided
            if system_prompt:
                conversation += f"[INST] {system_prompt}\n{user_prompt} [/INST]"
            else:
                conversation += f"[INST] {user_prompt} [/INST]"
            
            return conversation
            
        elif self.chat_format == "qwen":
            # Qwen format
            system_content = system_prompt or "You are a helpful assistant."
            conversation = f"<|im_start|>system\n{system_content}<|im_end|>\n"
            
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
                
            conversation += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            return conversation
            
        elif self.chat_format == "phi":
            # Phi format
            conversation = ""
            if system_prompt:
                conversation += f"<|system|>\n{system_prompt}\n"
                
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"<|{role}|>\n{content}\n"
                
            conversation += f"<|user|>\n{user_prompt}\n<|assistant|>\n"
            return conversation
            
        elif self.chat_format == "openai":
            # OpenAI format
            conversation = ""
            if system_prompt:
                conversation += f"system: {system_prompt}\n"
                
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"{role}: {content}\n"
                
            conversation += f"user: {user_prompt}\nassistant:"
            return conversation
            
        else:
            # Default format for instruct models (simple User/Assistant format)
            conversation = ""
            if system_prompt:
                conversation += f"{system_prompt}\n\n"
                
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"{role}: {content}\n"
                
            conversation += f"user: {user_prompt}\nassistant:"
            return conversation 