"""
Model Architecture Manager for handling different LLM chat templates and formatting.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ModelArchitectureManager:
    """Manages model architectures and provides formatting capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the manager with architecture configurations."""
        if config_path is None:
            # Default to the JSON file in the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "model_architectures.json")
        
        self.config_path = config_path
        self.architectures = {}
        self.message_formats = {}
        self.tool_formats = {}
        self._load_config()
    
    def _load_config(self):
        """Load the model architecture configuration from JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.architectures = config.get("architectures", {})
            self.message_formats = config.get("message_formats", {})
            self.tool_formats = config.get("tool_formats", {})
            
            logger.info(f"Loaded {len(self.architectures)} model architectures from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Model architecture config file not found: {self.config_path}")
            self._load_fallback_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing model architecture config: {e}")
            self._load_fallback_config()
        except Exception as e:
            logger.error(f"Error loading model architecture config: {e}")
            self._load_fallback_config()
    
    def _load_fallback_config(self):
        """Load a minimal fallback configuration."""
        logger.warning("Loading fallback model architecture configuration")
        self.architectures = {
            "generic": {
                "description": "Generic/unknown architecture fallback",
                "message_format": "basic",
                "system_prefix": "System: ",
                "system_suffix": "\n\n",
                "user_prefix": "Human: ",
                "user_suffix": "\n",
                "assistant_prefix": "Assistant: ",
                "assistant_suffix": "\n",
                "patterns": []
            }
        }
        self.message_formats = {
            "basic": "Simple role: content format (e.g., 'User:...')."
        }
        self.tool_formats = {}
    
    def detect_architecture(self, model_name: str) -> str:
        """
        Detect the architecture family for a given model name.
        
        Args:
            model_name: The name/path of the model
            
        Returns:
            The architecture key (e.g., 'gemma', 'qwen', 'llama')
        """
        if not model_name:
            return "generic"
        
        model_name_lower = model_name.lower()
        
        # Check each architecture's patterns
        for arch_name, arch_config in self.architectures.items():
            if arch_name == "generic":
                continue  # Skip generic, it's the fallback
                
            patterns = arch_config.get("patterns", [])
            for pattern in patterns:
                if pattern.lower() in model_name_lower:
                    logger.info(f"Detected architecture '{arch_name}' for model '{model_name}' (pattern: '{pattern}')")
                    return arch_name
        
        # No specific architecture detected, use generic
        logger.info(f"No specific architecture detected for model '{model_name}', using 'generic'")
        return "generic"
    
    def is_instruct_model(self, model_name: str) -> bool:
        """
        Determine if a model is an instruct/chat model based on its name.
        
        Args:
            model_name: The name/path of the model
            
        Returns:
            True if it's an instruct model, False if it's a base model
        """
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
            "-base", "_base", "-pt-", "_pt_", "pretrained", "pre-trained", "foundation", 
            "raw", "vanilla", "untuned", "completion"
        ]
        
        if any(pattern in model_name_lower for pattern in base_patterns):
            logger.info(f"Model '{model_name}' detected as BASE (contains base pattern)")
            return False
        
        # Then check for INSTRUCT model patterns
        instruct_patterns = [
            "instruct", "-chat", "_chat", "_it_", "-it-", "sft", "dpo", "rlhf", 
            "assistant", "alpaca", "vicuna", "wizard", "conversation"
        ]
        
        if any(pattern in model_name_lower for pattern in instruct_patterns):
            logger.info(f"Model '{model_name}' detected as INSTRUCT (contains instruct pattern)")
            return True
        
        # Default: if no clear pattern, assume BASE (safer for unknown models)
        logger.info(f"Model '{model_name}' detected as BASE (no clear pattern, defaulting to BASE)")
        return False
    
    def get_architecture_config(self, architecture: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific architecture.
        
        Args:
            architecture: The architecture name
            
        Returns:
            The architecture configuration dictionary
        """
        return self.architectures.get(architecture, self.architectures.get("generic", {}))
    
    def format_messages(self, messages: List[Dict[str, str]], model_name: str) -> str:
        """
        Format a list of messages according to the model's architecture.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model_name: The name/path of the model
            
        Returns:
            Formatted prompt string ready for the model
        """
        architecture = self.detect_architecture(model_name)
        arch_config = self.get_architecture_config(architecture)
        
        logger.info(f"Formatting {len(messages)} messages for architecture '{architecture}'")
        
        # Get the formatting tokens
        system_prefix = arch_config.get("system_prefix", "")
        system_suffix = arch_config.get("system_suffix", "")
        user_prefix = arch_config.get("user_prefix", "")
        user_suffix = arch_config.get("user_suffix", "")
        assistant_prefix = arch_config.get("assistant_prefix", "")
        assistant_suffix = arch_config.get("assistant_suffix", "")
        
        formatted_parts = []
        
        # Process each message
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                if system_prefix or system_suffix:
                    # Architecture has explicit system message support
                    formatted_parts.append(f"{system_prefix}{content}{system_suffix}")
                else:
                    # Special handling for architectures without explicit system support
                    if arch_config.get("system_as_assistant", False):
                        # Architecture uses assistant/model turns for system messages (e.g., Gemma)
                        # This is a "trick" to make models follow system instructions
                        formatted_parts.append(f"{assistant_prefix}System: {content}{assistant_suffix}")
                        logger.debug(f"Formatting system message as assistant turn for {architecture}")
                    elif architecture == "deepseek":
                        # DeepSeek: No explicit system support, prepend to conversation
                        formatted_parts.append(f"System: {content}\n\n")
                    else:
                        # Generic fallback: treat as user message with system prefix
                        formatted_parts.append(f"{user_prefix}System: {content}{user_suffix}")
                        
            elif role == "user":
                formatted_parts.append(f"{user_prefix}{content}{user_suffix}")
                
            elif role == "assistant":
                formatted_parts.append(f"{assistant_prefix}{content}{assistant_suffix}")
        
        # Join all parts
        formatted_prompt = "".join(formatted_parts)
        
        # Add generation prompt for the next assistant turn
        formatted_prompt += assistant_prefix
        
        logger.debug(f"Formatted prompt preview: {formatted_prompt[:200]}...")
        return formatted_prompt
    
    def format_single_turn(self, prompt: str, system_prompt: str, model_name: str) -> str:
        """
        Format a single turn conversation with optional system prompt.
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt
            model_name: The name/path of the model
            
        Returns:
            Formatted prompt string
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.format_messages(messages, model_name)
    
    def get_supported_architectures(self) -> List[str]:
        """Get a list of all supported architecture names."""
        return list(self.architectures.keys())
    
    def get_architecture_info(self, architecture: str) -> Dict[str, Any]:
        """
        Get detailed information about an architecture.
        
        Args:
            architecture: The architecture name
            
        Returns:
            Dictionary with architecture details including description, source, etc.
        """
        config = self.get_architecture_config(architecture)
        
        # Determine system support type
        has_explicit_system = bool(config.get("system_prefix") or config.get("system_suffix"))
        system_as_assistant = config.get("system_as_assistant", False)
        
        system_support_type = "explicit" if has_explicit_system else (
            "assistant_turn" if system_as_assistant else "basic"
        )
        
        return {
            "name": architecture,
            "description": config.get("description", ""),
            "source": config.get("source", ""),
            "message_format": config.get("message_format", ""),
            "patterns": config.get("patterns", []),
            "supports_system": has_explicit_system or system_as_assistant,
            "system_support_type": system_support_type,
            "system_note": config.get("system_note", "")
        }

# Global instance for easy access
_global_manager = None

def get_model_architecture_manager() -> ModelArchitectureManager:
    """Get the global ModelArchitectureManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelArchitectureManager()
    return _global_manager 