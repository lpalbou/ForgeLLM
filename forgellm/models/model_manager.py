"""
Model Manager - Handles loading, saving, and inference for models
"""

import os
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
import glob
from pathlib import Path
import shutil
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from forgellm.models.model_publisher import ModelPublisher
import psutil

# Configure logging
logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton helper that owns a single model instance in memory.

    The previous testing logic instantiated *CPTRepl* inside the Flask process.  That
    design made every request spin-up heavy CLI-oriented logic and duplicated
    memory across threads.  *ModelManager* offers a light-weight, thread-safe API
    that the web interface (or any other backend) can use:

        >>> manager = ModelManager()
        >>> manager.load("models/cpt/my_run", adapter_path="0005000_adapters.safetensors")
        >>> text = manager.generate("Hello")
        >>> manager.unload()

    Key advantages:
    1. Ensures **exactly one** model lives in RAM at any moment – avoids OOM.
    2. Makes generation an *idempotent* call that returns the raw completion
       portion (prompt stripped).
    3. Provides simple telemetry (RAM usage) that the frontend can surface.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def __init__(self):
        """Initialize model manager."""
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_type = None
        self.model_config = None
        self.model_publisher = ModelPublisher()
        
        # Set models directory to the root models directory
        self.models_dir = os.environ.get('MODELS_DIR', os.path.join(os.getcwd(), 'models'))
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
        
        # Create model directories if they don't exist
        self._create_model_dirs()

    def _create_model_dirs(self):
        """Create model directories if they don't exist."""
        for model_type in ['base', 'cpt', 'ift']:
            model_dir = os.path.join(self.models_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")

    def load(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        """Load a model into memory.
        
        Args:
            model_name: The name or path of the model to load
            adapter_path: Optional path to adapter weights
        """
        with self._lock:
            # Check if the same model is already loaded
            if (self.model_name == model_name and 
                self.adapter_path == adapter_path and 
                self.model is not None):
                logger.info(f"Model {model_name} already loaded")
                return
                
            # Unload any existing model first
            self.unload()
            
            try:
                logger.info(f"Loading model {model_name} with adapter {adapter_path}")
                
                # Resolve potential 'published/...' repo ids to absolute cache paths
                resolved_name = self._resolve_model_path(model_name)
                
                # Determine model type (full, LoRA, DoRA)
                self.model_type = self._determine_model_type(resolved_name, adapter_path)
                
                # Load the model
                self.model, self.tokenizer = load(resolved_name, adapter_path=adapter_path)
                self.model_name = model_name
                self.adapter_path = adapter_path
                self.loaded = True
                
                # Determine chat format based on model name
                self.chat_format = self._determine_chat_format(model_name)
                
                # Force garbage collection to clean up any memory
                gc.collect()
                
                # Log memory usage
                mem_usage = self.memory_usage_gb()
                logger.info(f"Model loaded successfully. Memory usage: {mem_usage:.1f} GB")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.loaded = False
                raise

    def load_model(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        """Load a model into memory (alias for load).
        
        Args:
            model_name: The name or path of the model to load
            adapter_path: Optional path to adapter weights
        """
        return self.load(model_name, adapter_path)

    def unload(self) -> None:
        """Unload the model from memory."""
        with self._lock:
            if hasattr(self, "model") and self.model is not None:
                logger.info("Unloading model")
                # Set to None to allow garbage collection
                self.model = None
                self.tokenizer = None
                self.loaded = False
                
                # Force garbage collection
                gc.collect()
                
                # Log memory usage after unloading
                mem_usage = self.memory_usage_gb()
                logger.info(f"Model unloaded. Memory usage: {mem_usage:.1f} GB")

    def unload_model(self) -> None:
        """Unload the model from memory (alias for unload)."""
        return self.unload()

    def generate(
        self,
        prompt: str,
        *,
        history: Optional[list[dict]] = None,
        max_tokens: int = 100,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        system_prompt: Optional[str] = None,
        max_kv_size: Optional[int] = None,
    ) -> str:
        """Generate text from the model.
        
        Args:
            prompt: The prompt to generate from
            history: Optional chat history
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter (lower = more focused)
            repetition_penalty: Penalty for repeating tokens
            system_prompt: Optional system prompt for chat models
            max_kv_size: Maximum KV cache size
            
        Returns:
            The generated text
        """
        with self._lock:
            if not self.loaded:
                raise RuntimeError("No model loaded. Call load() first.")
            
            try:
                # Format the prompt based on history and system prompt
                if history:
                    formatted_prompt = self._build_conversation(history, prompt, system_prompt)
                else:
                    formatted_prompt = self._format_prompt(prompt, system_prompt)
                
                # Set up generation parameters
                generation_kwargs = {
                    "prompt": formatted_prompt,
                    "max_tokens": max_tokens,
                }
                
                # Add optional parameters
                if temperature is not None and temperature > 0:
                    generation_kwargs["temp"] = temperature
                if top_p is not None and 0 < top_p < 1.0:
                    generation_kwargs["top_p"] = top_p
                
                # Handle repetition penalty with newer mlx-lm versions
                try:
                    # Import sampling utilities for newer versions
                    from mlx_lm.sample_utils import make_repetition_penalty
                    
                    if repetition_penalty is not None and repetition_penalty > 1.0:
                        # Create a repetition penalty processor with default context size
                        context_size = 20
                        rep_penalty_fn = make_repetition_penalty(repetition_penalty, context_size=context_size)
                        generation_kwargs["logits_processors"] = [rep_penalty_fn]
                except ImportError:
                    # Fall back to older method
                    if repetition_penalty is not None and repetition_penalty > 1.0:
                        generation_kwargs["repetition_penalty"] = repetition_penalty
                
                # Add max_kv_size if provided
                if max_kv_size is not None and max_kv_size > 0:
                    generation_kwargs["max_kv_size"] = max_kv_size
                
                # Generate text
                logger.info(f"Generating with parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
                response = generate(self.model, self.tokenizer, **generation_kwargs)
                
                # Return only the generated portion (not the prompt)
                if response.startswith(formatted_prompt):
                    return response[len(formatted_prompt):].lstrip()
                return response.strip()
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

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
        if not prompt:
            raise ValueError("Prompt is required")
            
        # Extract parameters with defaults
        history = params.get('history')
        max_tokens = params.get('max_tokens', 100)
        temperature = params.get('temperature')
        top_p = params.get('top_p')
        repetition_penalty = params.get('repetition_penalty')
        system_prompt = params.get('system_prompt')
        
        return self.generate(
            prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            system_prompt=system_prompt
        )

    def stop_generation(self) -> None:
        """Stop the current generation process."""
        # This is a placeholder for now
        # In the future, we could implement a way to stop generation
        logger.info("Stop generation requested (not implemented)")

    def memory_usage_gb(self) -> float:
        """Get the current memory usage in GB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.loaded:
            return {
                "loaded": False,
                "memory_usage_gb": self.memory_usage_gb()
            }
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "adapter_path": self.adapter_path,
            "model_type": self.model_type,
            "chat_format": self.chat_format,
            "memory_usage_gb": self.memory_usage_gb()
        }

    def _init(self):
        """Initialize the singleton instance."""
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_path = None
        self.loaded = False
        self.model_type = "unknown"  # full, lora, dora
        self.chat_format = "plain"   # plain, gemma, llama, mistral, etc.
        logger.info("ModelManager initialized")

    def __call__(self):  # noqa: D401  (simple method)
        """Allow using the singleton as a callable."""
        return self
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model path, handling published models and HF cache."""
        # If it's a local path that exists, return it directly
        if Path(model_name).exists():
            return model_name
            
        # Check if it's a published model
        published_path = Path("published") / model_name
        if published_path.exists():
            return str(published_path)
            
        # Check HuggingFace cache
        try:
            cache_root = Path.home() / '.cache' / 'huggingface' / 'hub'
            candidate = cache_root / ('models--' + model_name.replace('/', '--'))
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
            
        # Return original name (might be a HF model ID)
        return model_name
    
    def _determine_model_type(self, model_path: str, adapter_path: Optional[str]) -> str:
        """Determine the model type based on path and adapter."""
        if adapter_path is None:
            return "full"
            
        # Check if adapter config exists
        adapter_dir = Path(adapter_path).parent
        config_path = adapter_dir / "adapter_config.json"
        
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                if "fine_tune_type" in config:
                    return config["fine_tune_type"].lower()
                    
                # Check for LoRA-specific keys
                if "lora_rank" in config or "r" in config:
                    return "lora"
                    
                # Check for DoRA-specific keys
                if "dora_rank" in config:
                    return "dora"
            except Exception:
                pass
                
        # Default to LoRA if adapter is used but type can't be determined
        return "lora"
    
    def _determine_chat_format(self, model_name: str) -> str:
        """Determine the chat format based on model name."""
        model_lower = model_name.lower()
        
        if "gemma" in model_lower:
            return "gemma"
        elif "llama" in model_lower or "meta-llama" in model_lower:
            return "llama"
        elif "mistral" in model_lower:
            return "mistral"
        elif "qwen" in model_lower:
            return "qwen"
        elif "phi" in model_lower:
            return "phi"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        else:
            return "plain"

    def _format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format the prompt for the model based on model type."""
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
            # Default format (plain text)
            if system_prompt:
                return f"{system_prompt}\n\n{user_prompt}"
            else:
                return user_prompt

    def _build_conversation(self, history: list[dict], user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Build a conversation prompt from history."""
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
                conversation += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
            conversation += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
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
            # Default format - simple alternating messages
            conversation = ""
            if system_prompt:
                conversation += f"System: {system_prompt}\n\n"
                
            for message in history:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"{role.capitalize()}: {content}\n\n"
                
            conversation += f"User: {user_prompt}\n\nAssistant:"
            return conversation 