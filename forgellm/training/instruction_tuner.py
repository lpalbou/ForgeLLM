"""
SOTA Instruction Tuning with Hybrid Data Mixture
"""

import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np

from .training.config import InstructTuningConfig
from .models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ConversationExtractor:
    """Extract conversations from markdown documents with XML conversation tags"""
    
    def __init__(self):
        """Initialize the extractor"""
        self.conversation_pattern = re.compile(r'<conversation>(.*?)</conversation>', re.DOTALL)
        self.message_pattern = re.compile(r'<message from="(user|assistant)">(.*?)</message>', re.DOTALL)
    
    def extract_conversations(self, input_dir: str) -> List[Dict[str, Any]]:
        """Extract conversations from markdown documents with XML conversation tags"""
        conversations = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return []
        
        # Find all markdown files
        markdown_files = list(input_path.glob("**/*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files in {input_dir}")
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract conversations
                conversation_matches = self.conversation_pattern.findall(content)
                
                for conversation_text in conversation_matches:
                    # Extract messages
                    message_matches = self.message_pattern.findall(conversation_text)
                    
                    if not message_matches:
                        continue
                    
                    # Convert to conversation format
                    messages = []
                    for role, content in message_matches:
                        messages.append({
                            "role": role,
                            "content": content.strip()
                        })
                    
                    # Add to conversations
                    if len(messages) >= 2:  # At least one user and one assistant message
                        conversations.append({
                            "source": str(file_path),
                            "messages": messages
                        })
            
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Extracted {len(conversations)} conversations from markdown documents")
        return conversations


class InstructionDataProcessor:
    """Process instruction data for fine-tuning"""
    
    def __init__(self, config: InstructTuningConfig):
        """Initialize the processor"""
        self.config = config
        self.conversation_extractor = ConversationExtractor()
    
    def prepare_instruction_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Prepare instruction data for fine-tuning"""
        # Create data directory
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract conversations from custom dataset
        custom_conversations = self.conversation_extractor.extract_conversations("dataset")
        
        # Load general instruction datasets
        general_instructions = self.load_general_instruction_data()
        
        # Combine datasets with the specified ratio
        custom_count = int(self.config.max_train_examples * self.config.dataset_ratio)
        general_count = self.config.max_train_examples - custom_count
        
        # Sample from each dataset
        np.random.seed(self.config.seed)
        
        if len(custom_conversations) > custom_count:
            custom_sample = np.random.choice(
                custom_conversations, 
                size=custom_count, 
                replace=False
            ).tolist()
        else:
            custom_sample = custom_conversations
            
        if len(general_instructions) > general_count:
            general_sample = np.random.choice(
                general_instructions, 
                size=general_count, 
                replace=False
            ).tolist()
        else:
            general_sample = general_instructions
        
        # Combine samples
        combined_instructions = custom_sample + general_sample
        np.random.shuffle(combined_instructions)
        
        # Split into train and validation
        val_size = min(self.config.max_val_examples, int(len(combined_instructions) * 0.1))
        train_instructions = combined_instructions[:-val_size]
        val_instructions = combined_instructions[-val_size:]
        
        logger.info(f"Prepared {len(train_instructions)} training examples and {len(val_instructions)} validation examples")
        logger.info(f"Custom dataset ratio: {len(custom_sample) / len(combined_instructions):.1%}")
        
        return train_instructions, val_instructions
    
    def load_general_instruction_data(self) -> List[Dict]:
        """Load general instruction datasets"""
        # This is a placeholder - in a real implementation, we would load datasets from HuggingFace or local files
        # For now, we'll return a small sample of instructions
        instructions = []
        
        # Sample instructions (these would normally be loaded from files)
        sample_instructions = [
            {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed to perform a task, these systems are trained on large datasets and learn to recognize patterns, make decisions, and improve over time through experience."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain the concept of fine-tuning in language models."},
                    {"role": "assistant", "content": "Fine-tuning in language models refers to the process of taking a pre-trained model and further training it on a specific dataset for a particular task. This allows the model to adapt its general knowledge to more specialized domains or tasks while requiring much less data and computational resources than training from scratch."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What are the advantages of parameter-efficient fine-tuning methods like LoRA?"},
                    {"role": "assistant", "content": "Parameter-efficient fine-tuning methods like LoRA (Low-Rank Adaptation) offer several advantages: 1) They require much less memory and computational resources than full fine-tuning, 2) They reduce the risk of catastrophic forgetting by updating only a small subset of parameters, 3) They result in small adapter weights that are easy to distribute and switch between, and 4) They often achieve comparable performance to full fine-tuning despite updating far fewer parameters."}
                ]
            }
        ]
        
        # Add sample instructions
        instructions.extend(sample_instructions)
        
        # In a real implementation, we would load datasets from HuggingFace
        try:
            from datasets import load_dataset
            
            # Load OpenOrca dataset (small sample)
            try:
                openorca = load_dataset("Open-Orca/OpenOrca", split="train[:100]")
                for item in openorca:
                    instructions.append({
                        "messages": [
                            {"role": "user", "content": item["question"]},
                            {"role": "assistant", "content": item["response"]}
                        ]
                    })
            except Exception as e:
                logger.warning(f"Failed to load OpenOrca dataset: {e}")
            
            # Load Alpaca dataset (small sample)
            try:
                alpaca = load_dataset("tatsu-lab/alpaca", split="train[:100]")
                for item in alpaca:
                    instructions.append({
                        "messages": [
                            {"role": "user", "content": item["instruction"]},
                            {"role": "assistant", "content": item["output"]}
                        ]
                    })
            except Exception as e:
                logger.warning(f"Failed to load Alpaca dataset: {e}")
        
        except ImportError:
            logger.warning("datasets library not available, using only sample instructions")
        
        logger.info(f"Loaded {len(instructions)} general instruction examples")
        return instructions
    
    def save_instruction_data(self, train_data: List[Dict], val_data: List[Dict]) -> Tuple[str, str]:
        """Save instruction data to files"""
        train_file = Path(self.config.data_dir) / "train.jsonl"
        val_file = Path(self.config.data_dir) / "valid.jsonl"
        
        # Save training data
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps({"messages": item["messages"]}) + "\n")
        
        # Save validation data
        with open(val_file, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps({"messages": item["messages"]}) + "\n")
        
        logger.info(f"Saved instruction data to {train_file} and {val_file}")
        return str(train_file), str(val_file)


class SOTAInstructTrainer:
    """SOTA Instruction Tuning with Hybrid Approach"""
    
    def __init__(self, config: InstructTuningConfig):
        """Initialize the trainer"""
        self.config = config
        self.data_processor = InstructionDataProcessor(config)
    
    def run_complete_pipeline(self):
        """Run the complete instruction tuning pipeline"""
        logger.info("=== Starting SOTA Instruction Tuning Pipeline ===")
        
        # Prepare instruction data
        train_data, val_data = self.data_processor.prepare_instruction_data()
        train_file, val_file = self.data_processor.save_instruction_data(train_data, val_data)
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run instruction tuning with MLX-LM
        self._run_mlx_instruction_tuning(train_file, val_file)
        
        logger.info("=== Instruction Tuning Pipeline Completed ===")
    
    def _run_mlx_instruction_tuning(self, train_file: str, val_file: str):
        """Run instruction tuning with MLX-LM"""
        import subprocess
        import sys
        import yaml
        
        # Create MLX-LM YAML configuration
        mlx_config = {
            # Model configuration
            "model": self.config.base_model_name,
            "adapter_path": self.config.base_model_path,
            "train": True,
            
            # Training data
            "data": str(Path(train_file).parent),
            
            # Fine-tuning configuration
            "fine_tune_type": "lora",  # Always use LoRA for instruction tuning
            "num_layers": self.config.lora_layers,
            
            # Training parameters
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "iters": self.config.max_iterations,
            "save_every": self.config.save_every,
            "steps_per_eval": self.config.eval_every,
            "adapter_path_out": str(output_dir),
            "max_seq_length": self.config.max_seq_length,
            
            # Learning rate schedule
            "lr_schedule": {
                "name": self.config.lr_schedule,
                "arguments": [
                    self.config.learning_rate,
                    self.config.max_iterations,
                    self.config.learning_rate * self.config.min_lr_ratio
                ],
                "warmup": self.config.warmup_steps
            },
            
            # Optimizer configuration
            "optimizer": "adamw",
            "optimizer_config": {
                "adamw": {
                    "weight_decay": 0.01
                }
            },
            
            # Advanced settings
            "grad_checkpoint": True,
            "mask_prompt": True  # Enable prompt masking for instruction tuning
        }
        
        # Create MLX-LM YAML configuration file
        config_file = Path(self.config.output_dir) / f"mlx_config_{int(time.time())}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(mlx_config, f, default_flow_style=False)
        
        logger.info(f"MLX-LM config saved to {config_file}")
        
        # Build MLX-LM command
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--config", str(config_file)
        ]
        
        # Run MLX-LM command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Process output
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                logger.info(f"MLX-LM: {line.strip()}")
        
        # Check return code
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ MLX-LM instruction tuning completed successfully")
        else:
            logger.error(f"❌ MLX-LM instruction tuning failed with return code {return_code}")
            raise RuntimeError(f"Instruction tuning failed with return code {return_code}")
    
    def validate_model(self):
        """Validate the instruction-tuned model"""
        logger.info("=== Validating Instruction-Tuned Model ===")
        
        # Find the latest checkpoint
        output_dir = Path(self.config.output_dir)
        checkpoints = list(output_dir.glob("*_adapters.safetensors"))
        
        if not checkpoints:
            logger.warning("No checkpoints found")
            return
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using checkpoint: {latest_checkpoint}")
        
        # Load the model
        model_manager = ModelManager()
        model_manager.load(self.config.base_model_name, str(latest_checkpoint))
        
        # Test with sample prompts
        test_prompts = [
            "Explain the concept of continued pre-training in language models.",
            "What are the advantages of instruction fine-tuning?",
            "How does LoRA work for parameter-efficient fine-tuning?"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Prompt: {prompt}")
            
            response = model_manager.generate(
                prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            logger.info(f"Response: {response}")
        
        # Unload the model
        model_manager.unload()
        
        logger.info("=== Validation Complete ===") 