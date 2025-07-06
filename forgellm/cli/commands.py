"""
CLI commands for ForgeLLM
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_cli():
    """Set up CLI commands for ForgeLLM."""
    parser = argparse.ArgumentParser(description="ForgeLLM - MLX-LM training and inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    setup_train_command(train_parser)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text from a model")
    setup_generate_command(generate_parser)
    
    # Dataset info command
    dataset_parser = subparsers.add_parser("dataset", help="Get dataset information")
    setup_dataset_command(dataset_parser)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate training dashboard")
    setup_dashboard_command(dashboard_parser)
    
    # Publish command
    publish_parser = subparsers.add_parser("publish", help="Publish model to shareable format")
    setup_publish_command(publish_parser)
    
    # Instruction tuning command
    instruct_parser = subparsers.add_parser("instruct", help="Instruction-tune a model")
    setup_instruction_tuning_command(instruct_parser)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration files")
    setup_config_command(config_parser)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Call the appropriate function based on the command
    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return 1


def setup_train_command(parser):
    """Set up train command arguments."""
    # Add config file option
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Standard arguments (will be overridden by config file if provided)
    parser.add_argument("--model-name", type=str, help="Model name or path")
    parser.add_argument("--input-dir", type=str, help="Input directory containing documents", default="dataset")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints", default="models")
    parser.add_argument("--data-dir", type=str, help="Directory for processed training data", default="data/pretraining")
    parser.add_argument("--batch-size", type=int, help="Training batch size", default=4)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=5e-6)
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length", default=2048)
    parser.add_argument("--max-tokens-per-file", type=int, help="Maximum tokens per file", default=1000000)
    parser.add_argument("--save-every", type=int, help="Save checkpoint every N iterations", default=500)
    parser.add_argument("--max-checkpoints", type=int, help="Maximum checkpoints to keep", default=5)
    parser.add_argument("--max-iterations", type=int, help="Maximum training iterations", default=10000)
    parser.add_argument("--warmup-steps", type=int, help="Learning rate warmup steps", default=150)
    parser.add_argument("--fine-tune-type", type=str, choices=["full", "lora", "dora"], help="Fine-tuning type", default="full")
    parser.add_argument("--num-layers", type=int, help="Number of layers to fine-tune (-1 for all)", default=-1)
    parser.add_argument("--data-mixture-ratio", type=float, help="Domain data mixture ratio (0-1)", default=0.95)
    parser.add_argument("--overfitting-threshold", type=float, help="Overfitting detection threshold", default=0.30)
    parser.add_argument("--early-stopping-patience", type=int, help="Early stopping patience", default=3)
    parser.add_argument("--min-loss-improvement", type=float, help="Minimum loss improvement", default=0.001)
    parser.add_argument("--validation-split", type=float, help="Validation split ratio", default=0.1)
    parser.add_argument("--validation-fast-pct", type=float, help="Percentage of validation set used for quick validation checks (0.0-1.0)", default=1.0)
    parser.add_argument("--steps-per-eval", type=int, help="Steps between evaluations", default=25)
    parser.add_argument("--steps-per-report", type=int, help="Steps between reports", default=5)
    parser.add_argument("--enable-early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--use-lr-rewarming", action="store_true", help="Use learning rate rewarming")
    parser.add_argument("--lr-schedule", type=str, choices=["cosine_decay", "linear_decay", "constant"], 
                        help="Learning rate schedule", default="cosine_decay")
    parser.add_argument("--lr-decay-factor", type=float, help="Learning rate decay factor", default=0.1)
    parser.add_argument("--weight-decay", type=float, help="Weight decay for AdamW", default=0.01)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--val-batches", type=int, help="Override the number of validation batches", default=None)
    
    parser.set_defaults(func=run_train_command)


def setup_generate_command(parser):
    """Set up generate command arguments."""
    parser.add_argument("--model", type=str, help="Model name or path", required=True)
    parser.add_argument("--adapter-path", type=str, help="Path to adapter weights", default=None)
    parser.add_argument("--prompt", type=str, help="Prompt for generation", required=True)
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate", default=100)
    parser.add_argument("--temperature", type=float, help="Sampling temperature", default=0.7)
    parser.add_argument("--top-p", type=float, help="Nucleus sampling parameter", default=0.9)
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty", default=1.1)
    parser.add_argument("--system-prompt", type=str, help="System prompt", default=None)
    
    parser.set_defaults(func=run_generate_command)


def setup_dataset_command(parser):
    """Set up dataset command arguments."""
    parser.add_argument("--input-dir", type=str, help="Input directory containing documents", default="dataset")
    
    parser.set_defaults(func=run_dataset_command)


def setup_dashboard_command(parser):
    """Set up dashboard generation command."""
    parser.add_argument("json_file", help="Training metrics JSON file")
    parser.add_argument("--output-dir", help="Output directory for dashboard", default="training_dashboard")
    parser.add_argument("--output-name", help="Name of the output image file", default="training_dashboard.png")
    
    parser.set_defaults(func=run_dashboard_generation)


def setup_publish_command(parser):
    """Set up model publishing command."""
    parser.add_argument("checkpoint_path", help="Path to checkpoint file")
    parser.add_argument("--output-dir", help="Output directory for published model", default=None)
    
    parser.set_defaults(func=run_publish_model)


def setup_instruction_tuning_command(parser):
    """Set up instruction tuning command."""
    # Add config file option
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Standard arguments (will be overridden by config file if provided)
    parser.add_argument("--base-model-path", help="Path to continued pre-trained model")
    parser.add_argument("--base-model-name", help="Base model name")
    parser.add_argument("--output-dir", help="Output directory for instruction-tuned model", default="models/instruct_tuned")
    parser.add_argument("--data-dir", help="Directory for instruction tuning data", default="data/instruction_tuning")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=4)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=5e-6)
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length", default=2500)
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations", default=100)
    parser.add_argument("--save-every", type=int, help="Save checkpoint every N iterations", default=25)
    parser.add_argument("--eval-every", type=int, help="Evaluate every N iterations", default=25)
    parser.add_argument("--lora-layers", type=int, help="Number of LoRA layers", default=16)
    parser.add_argument("--warmup-steps", type=int, help="Learning rate warmup steps", default=50)
    parser.add_argument("--lr-schedule", type=str, choices=["cosine", "linear", "constant"], help="Learning rate schedule", default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, help="Minimum LR as ratio of max LR", default=0.1)
    parser.add_argument("--dataset-ratio", type=float, help="Ratio of dataset used", default=0.1)
    parser.add_argument("--max-train-examples", type=int, help="Maximum training examples", default=10000)
    parser.add_argument("--max-val-examples", type=int, help="Maximum validation examples", default=1000)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    
    parser.set_defaults(func=run_instruction_tuning)


def setup_config_command(parser):
    """Set up configuration management command."""
    subparsers = parser.add_subparsers(dest="config_command", help="Config command")
    
    # Create default configs
    create_parser = subparsers.add_parser("create-defaults", help="Create default configuration files")
    create_parser.set_defaults(func=run_create_default_configs)
    
    # Show config
    show_parser = subparsers.add_parser("show", help="Show configuration")
    show_parser.add_argument("config_file", help="Path to configuration file")
    show_parser.set_defaults(func=run_show_config)
    
    # Export config
    export_parser = subparsers.add_parser("export", help="Export configuration to file")
    export_parser.add_argument("--type", choices=["cpt", "ift"], help="Configuration type", required=True)
    export_parser.add_argument("--output", help="Output file path", required=True)
    export_parser.set_defaults(func=run_export_config)
    
    parser.set_defaults(func=lambda args: parser.print_help())


def run_train_command(args):
    """Run train command."""
    from .training.config import TrainingConfig
    from .training.trainer import ContinuedPretrainer
    
    try:
        # Create configuration
        if args.config:
            # Load from config file
            logger.info(f"Loading configuration from {args.config}")
            config = TrainingConfig.load(args.config)
            
            # Override with command line arguments if provided
            config_dict = vars(args)
            for key, value in config_dict.items():
                if key in TrainingConfig.__annotations__ and value is not None and key != "config":
                    logger.info(f"Overriding config parameter {key} with value {value}")
                    setattr(config, key, value)
        else:
            # Check if model_name is provided
            if args.model_name is None:
                logger.error("Either --config or --model-name must be provided")
                return 1
                
            # Create from command line arguments
            logger.info("Creating configuration from command line arguments")
            config_dict = vars(args)
            valid_params = {k: v for k, v in config_dict.items() 
                           if k in TrainingConfig.__annotations__ and v is not None}
            
            logger.info(f"Configuration parameters: {valid_params}")
            config = TrainingConfig(**valid_params)
        
        # Log the configuration
        logger.info(f"Training configuration: {config.to_dict()}")
        
        # Initialize trainer
        trainer = ContinuedPretrainer(config)
        
        # Run training
        logger.info("Starting training")
        trainer.run_training()
        
        logger.info("Training completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error running training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def run_generate_command(args):
    """Run generate command."""
    from .models.model_manager import ModelManager
    
    # Initialize model manager
    model_manager = ModelManager()
    
    try:
        # Load model
        model_manager.load(args.model, args.adapter_path)
        
        # Generate text
        response = model_manager.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            system_prompt=args.system_prompt
        )
        
        # Print response
        if isinstance(response, dict) and response.get('success'):
            print(response.get('text', response))
        else:
            print(response)
        
        # Unload model
        model_manager.unload()
        
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


def run_dataset_command(args):
    """Run dataset command."""
    from .training.data_processor import DocumentProcessor
    
    # Initialize document processor
    from .training.config import TrainingConfig
    config = TrainingConfig(model_name="dummy", input_dir=args.input_dir)
    doc_processor = DocumentProcessor(config)
    
    # Collect documents
    try:
        documents = doc_processor.collect_documents()
        
        # Count tokens using accurate method
        total_tokens = 0
        from ..utils.text_stats import count_tokens_accurate
        
        for doc_path in documents:
            text = doc_processor.extract_text_from_file(doc_path)
            if text:
                file_tokens = count_tokens_accurate(text)
                total_tokens += file_tokens
        
        # Print summary
        print(f"Found {len(documents)} documents with {total_tokens:,} tokens (accurate count)")
        print(f"Input directory: {args.input_dir}")
        
        return 0
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return 1


def run_dashboard_generation(args):
    """Run dashboard generation command."""
    try:
        # Import dashboard generator
        from .training.dashboard import create_comprehensive_dashboard
        
        # Call dashboard generation function
        create_comprehensive_dashboard(
            args.json_file,
            output_dir=args.output_dir,
            output_name=args.output_name
        )
        
        print(f"Dashboard generated at {os.path.join(args.output_dir, args.output_name)}")
        return 0
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        return 1


def run_publish_model(args):
    """Run model publishing command."""
    from ..models.model_publisher import publish_checkpoint
    
    try:
        output_dir = publish_checkpoint(args.checkpoint_path, args.output_dir)
        print(f"Model published to {output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Model publishing failed: {e}")
        return 1


def run_instruction_tuning(args):
    """Run instruction tuning command."""
    from .training.config import InstructTuningConfig
    from .training.instruction_tuner import InstructionTuner
    
    # Create configuration
    if args.config:
        # Load from config file
        config = InstructTuningConfig.load(args.config)
        
        # Override with command line arguments if provided
        config_dict = vars(args)
        for key, value in config_dict.items():
            if key in InstructTuningConfig.__annotations__ and value is not None and key != "config":
                setattr(config, key, value)
    else:
        # Check if required arguments are provided
        if args.base_model_path is None or args.base_model_name is None:
            logger.error("Either --config or both --base-model-path and --base-model-name must be provided")
            return 1
            
        # Create from command line arguments
        config_dict = vars(args)
        config = InstructTuningConfig(**{k: v for k, v in config_dict.items() 
                                      if k in InstructTuningConfig.__annotations__ and v is not None})
    
    # Initialize tuner
    tuner = InstructionTuner(config)
    
    # Run instruction tuning
    tuner.run_tuning()
    
    return 0


def run_create_default_configs(args):
    """Run create default configs command."""
    from .training.config import create_default_configs
    
    try:
        create_default_configs()
        print("Default configuration files created in configs/ directory")
        return 0
    except Exception as e:
        logger.error(f"Creating default configs failed: {e}")
        return 1


def run_show_config(args):
    """Run show config command."""
    import yaml
    
    try:
        # Load config file
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Print config
        print(yaml.dump(config, default_flow_style=False))
        return 0
    except Exception as e:
        logger.error(f"Showing config failed: {e}")
        return 1


def run_export_config(args):
    """Run export config command."""
    from .training.config import TrainingConfig, InstructTuningConfig
    
    try:
        # Create config based on type
        if args.type == "cpt":
            config = TrainingConfig(model_name="mlx-community/gemma-3-4b-it-bf16", input_dir="dataset")
        else:  # ift
            config = InstructTuningConfig(
                base_model_path="models/cpt/latest",
                base_model_name="mlx-community/gemma-3-4b-it-bf16"
            )
        
        # Save config
        config.save(args.output)
        print(f"Configuration exported to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Exporting config failed: {e}")
        return 1 