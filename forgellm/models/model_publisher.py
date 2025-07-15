"""
Model Publisher - Handles comprehensive publishing of models with documentation and dashboards
"""

import os
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Import our training dashboard generator
from ..training.dashboard import DashboardGenerator, identify_best_checkpoints, load_training_data

logger = logging.getLogger(__name__)

class ModelPublisher:
    """Handles comprehensive publishing of models to a central repository with full documentation"""
    
    def __init__(self):
        """Initialize model publisher"""
        # Use the HuggingFace cache directory for published models
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_generator = DashboardGenerator()
    
    def publish_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Publish a checkpoint to a central repository with comprehensive documentation
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dict with publication info
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Try to resolve the path if it doesn't exist directly
            if not checkpoint_path.exists():
                # Try different base paths
                for base_path in [".", "forgellm", Path.cwd()]:
                    full_path = Path(base_path) / checkpoint_path
                    if full_path.exists():
                        checkpoint_path = full_path
                        break
                else:
                    return {"success": False, "error": f"Checkpoint {checkpoint_path} not found"}
            
            # Get the training directory (parent of checkpoint file)
            training_dir = checkpoint_path.parent
            
            # Find the training JSON file
            training_json = self._find_training_json(training_dir)
            if not training_json:
                return {"success": False, "error": f"Training JSON file not found in {training_dir}"}
            
            # Load training data and configuration
            training_data = self._load_training_data(training_json)
            if not training_data:
                return {"success": False, "error": "Failed to load training data"}
            
            # Load adapter configuration
            adapter_config_path = training_dir / "adapter_config.json"
            if not adapter_config_path.exists():
                return {"success": False, "error": f"Adapter config not found at {adapter_config_path}"}
            
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            
            # Get base model name
            base_model = adapter_config.get("base_model_name") or adapter_config.get("model")
            if not base_model:
                return {"success": False, "error": "Base model name not found in adapter config"}
            
            # Create publication directory name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = checkpoint_path.stem.replace("_adapters", "")
            model_dir_name = f"{training_dir.name}_{checkpoint_name}_{timestamp}"
            
            # Create the published model directory
            published_dir = self.cache_dir / f"models--published--{model_dir_name}"
            published_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Publishing model to {published_dir}")
            
            # Step 1: Convert LoRA to full model using MLX-LM
            conversion_result = self._convert_lora_to_full_model(
                base_model, checkpoint_path, published_dir
            )
            if not conversion_result["success"]:
                return conversion_result
            
            # Step 2: Generate comprehensive training dashboard
            dashboard_path = self._generate_training_dashboard(
                training_json, published_dir, training_data
            )
            
            # Step 3: Create comprehensive README.md
            readme_path = self._create_comprehensive_readme(
                training_data, adapter_config, checkpoint_path, 
                published_dir, dashboard_path, base_model, model_dir_name
            )
            
            # Step 4: Create model card metadata
            self._create_model_card_metadata(
                training_data, adapter_config, published_dir, base_model
            )
            
            # Step 5: Create reproducibility script
            self._create_reproducibility_script(
                training_data, published_dir
            )
            
            logger.info(f"Model successfully published to {published_dir}")
            
            return {
                "success": True,
                "output_dir": str(published_dir),
                "model_name": f"published/{model_dir_name}",
                "readme_path": str(readme_path),
                "dashboard_path": str(dashboard_path) if dashboard_path else None,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Model publishing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_training_json(self, training_dir: Path) -> Optional[Path]:
        """Find the training JSON file for this training session"""
        # Look for CPT_*.json files in the training directory and parent directories
        search_dirs = [training_dir, training_dir.parent, training_dir.parent.parent]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                json_files = list(search_dir.glob("CPT_*.json"))
                if json_files:
                    # Find the one that matches our training directory
                    for json_file in json_files:
                        with open(json_file) as f:
                            data = json.load(f)
                            if training_dir.name in data.get("output_path", ""):
                                return json_file
                    # If no exact match, return the most recent one
                    return max(json_files, key=lambda x: x.stat().st_mtime)
        
        return None
    
    def _load_training_data(self, json_path: Path) -> Optional[Dict]:
        """Load and validate training data from JSON file"""
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ["config", "metrics", "base_model"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field '{field}' in training data")
                    return None
            
            return data
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None
    
    def _convert_lora_to_full_model(self, base_model: str, checkpoint_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Convert LoRA adapter to full model using MLX-LM"""
        try:
            # MLX-LM fuse expects the adapter directory, not the specific .safetensors file
            adapter_dir = checkpoint_path.parent
            
            # Use mlx_lm.fuse to convert LoRA to full model
            cmd = [
                "python", "-m", "mlx_lm.fuse",
                "--model", base_model,
                "--adapter-path", str(adapter_dir),
                "--save-path", str(output_dir)
            ]
            
            logger.info(f"Converting LoRA to full model: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info("Model conversion completed successfully")
            return {"success": True}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Model conversion failed: {e.stderr}")
            return {"success": False, "error": f"Model conversion failed: {e.stderr}"}
        except Exception as e:
            logger.error(f"Model conversion error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_training_dashboard(self, json_path: Path, output_dir: Path, training_data: Dict) -> Optional[Path]:
        """Generate comprehensive training dashboard"""
        try:
            # Create assets directory
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # Configure matplotlib to use non-GUI backend for thread safety
            import matplotlib
            matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
            
            # Generate dashboard using our DashboardGenerator
            dashboard_path = self.dashboard_generator.create_dashboard(
                str(json_path),
                str(assets_dir),
                "training_dashboard.png",
                dpi=300,
                figsize=(20, 16)
            )
            
            if dashboard_path and Path(dashboard_path).exists():
                logger.info(f"Training dashboard generated: {dashboard_path}")
                return Path(dashboard_path)
            else:
                logger.warning("Failed to generate training dashboard")
                return None
                
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            # Try to create a simple placeholder image instead
            try:
                self._create_placeholder_dashboard(output_dir / "assets")
                return output_dir / "assets" / "training_dashboard.png"
            except:
                return None
    
    def _create_comprehensive_readme(self, training_data: Dict, adapter_config: Dict, 
                                   checkpoint_path: Path, output_dir: Path, 
                                   dashboard_path: Optional[Path], base_model: str,
                                   model_name: str) -> Path:
        """Create comprehensive README.md with detailed training information"""
        
        config = training_data["config"]
        metrics = training_data["metrics"]
        
        # Get best checkpoints analysis
        try:
            best_checkpoints = identify_best_checkpoints(training_data, 3)
            if isinstance(best_checkpoints, str):
                # If function returns error string, create empty list
                best_checkpoints = []
        except Exception as e:
            logger.warning(f"Failed to identify best checkpoints: {e}")
            best_checkpoints = []
        
        # Find our specific checkpoint in the metrics
        checkpoint_iteration = self._extract_iteration_from_checkpoint(checkpoint_path)
        checkpoint_metrics = self._find_checkpoint_metrics(metrics, checkpoint_iteration)
        
        # Calculate training statistics
        training_stats = self._calculate_training_statistics(metrics, config)
        
        # Calculate checkpoint-specific stats
        checkpoint_stats = self._calculate_checkpoint_training_stats(checkpoint_iteration, config, training_data)
        
        # Get dataset information
        dataset_info = self._analyze_dataset_information(config, training_data)
        
        readme_content = f"""---
library_name: mlx
pipeline_tag: text-generation
base_model: {base_model}
tags:
- mlx
- forgellm
- continued-pretraining
- dataset
license: mit
---

# {model_name.replace('_', ' ').title()}

This model was created by **ForgeLLM** ([GitHub](https://github.com/lpalbou/ForgeLLM) | [PyPI](https://pypi.org/project/forgellm/)), a comprehensive toolkit for continued pre-training and fine-tuning of language models using MLX-LM on Apple Silicon.

## 🔍 Model Overview

| Parameter | Value |
|-----------|-------|
| **Base Model** | `{base_model}` |
| **Training Type** | {training_data.get('training_type', 'CPT')} |
| **Fine-tuning Type** | {config.get('fine_tune_type', 'full')} |
| **Checkpoint** | {checkpoint_path.name} (Iteration {checkpoint_iteration}) |
| **Publication Date** | {datetime.now().strftime('%Y-%m-%d')} |
| **Training Duration** | {self._format_duration(training_data.get('start_time'), training_data.get('end_time'))} |

## 📊 Training Dashboard

![Training Dashboard](assets/training_dashboard.png)

The comprehensive dashboard above shows detailed training metrics including:
- **Loss Curves**: Training and validation loss progression
- **Perplexity Analysis**: Model performance over time  
- **Learning Rate Schedule**: Adaptive learning rate changes
- **Memory Usage**: Peak memory consumption patterns
- **Training Speed**: Tokens/second throughput
- **Convergence Analysis**: Loss smoothness and stability
- **Multi-Criteria Checkpoint Selection**: SOTA methodology for optimal checkpoint identification

## 🎯 Checkpoint Selection Rationale

This published model uses **checkpoint {checkpoint_iteration}** selected based on comprehensive analysis:

{self._format_checkpoint_rationale(checkpoint_metrics, best_checkpoints, checkpoint_iteration)}

### Training Progress at This Checkpoint
| Metric | Value |
|--------|-------|
| **Tokens per Iteration** | {checkpoint_stats['tokens_per_iteration']:,} |
| **Total Tokens Trained** | {checkpoint_stats['total_tokens_trained']:,} |
| **Epochs Completed** | {checkpoint_stats['epochs_completed']:.2f} |
| **Training Progress** | {(checkpoint_iteration / config.get('max_iterations', checkpoint_iteration) * 100):.1f}% |

### Best Checkpoints Analysis

{self._format_best_checkpoints_table(best_checkpoints)}

## ⚙️ Training Configuration

### Core Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | {config.get('batch_size', 'N/A')} | Training batch size |
| **Learning Rate** | {config.get('learning_rate', 'N/A')} | Peak learning rate |
| **Max Sequence Length** | {config.get('max_seq_length', 'N/A')} | Maximum input sequence length |
| **Training Iterations** | {config.get('max_iterations', 'N/A')} | Total training iterations |
| **Warmup Steps** | {config.get('warmup_steps', 'N/A')} | Learning rate warmup period |

### Advanced Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | {config.get('optimizer', 'adamw')} | Optimization algorithm |
| **Weight Decay** | {config.get('weight_decay', 'N/A')} | L2 regularization strength |
| **LR Schedule** | {config.get('lr_schedule', 'cosine_decay')} | Learning rate scheduling |
| **LR Decay Factor** | {config.get('lr_decay_factor', 'N/A')} | Learning rate decay multiplier |
| **Gradient Checkpointing** | {config.get('grad_checkpoint', False)} | Memory optimization technique |
| **Early Stopping** | {config.get('enable_early_stopping', False)} | Automatic training termination |

## 📚 Dataset Information

{self._format_dataset_information(dataset_info, config)}

## 📈 Training Performance

{self._format_training_performance(training_stats)}

## 🔬 Technical Details

### Software Stack
- **ForgeLLM**: {self._get_forgellm_version()} ([GitHub](https://github.com/lpalbou/ForgeLLM) | [PyPI](https://pypi.org/project/forgellm/))
- **MLX-LM**: Latest compatible version ([GitHub](https://github.com/ml-explore/mlx-lm) | [PyPI](https://pypi.org/project/mlx-lm/))
- **MLX**: Apple's machine learning framework ([GitHub](https://github.com/ml-explore/mlx) | [Documentation](https://ml-explore.github.io/mlx/build/html/index.html))
- **Platform**: Apple Silicon (Metal Performance Shaders)

### Model Architecture
- **Base Architecture**: {self._extract_model_architecture(base_model)}
- **Parameter Count**: {self._estimate_parameter_count(base_model)}
- **Precision**: {self._extract_precision(base_model)}

## 🚀 Usage

### Quick Start
```python
from mlx_lm import load, generate

# Load the model
model, tokenizer = load("published/{model_name}")

# Generate text
response = generate(
    model, 
    tokenizer,
    prompt="Your prompt here",
    max_tokens=500,
    temperature=0.7
)
print(response)
```

### Advanced Usage
```python
import mlx.core as mx
from mlx_lm import load, generate

# Load with specific configuration
model, tokenizer = load(
    "published/{model_name}",
    tokenizer_config={{"trust_remote_code": True}}
)

# Generate with custom parameters
response = generate(
    model,
    tokenizer,
    prompt="Write a detailed explanation of quantum computing",
    max_tokens=1000,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)
```

## 🔄 Reproducibility

To reproduce this training:

1. **Install ForgeLLM**:
   ```bash
   pip install forgellm
   ```

2. **Use the provided configuration**:
   ```bash
   python -m forgellm.cli.commands train --config reproducibility_config.yaml
   ```

See `reproducibility_config.yaml` in this repository for the exact configuration used.

## 🙏 Acknowledgments

This model was created using:
- **[ForgeLLM](https://github.com/lpalbou/ForgeLLM)**: Comprehensive toolkit for continued pre-training and fine-tuning
- **[MLX-LM](https://github.com/ml-explore/mlx-lm)**: Apple's MLX-based LLM training and inference library
- **[MLX](https://github.com/ml-explore/mlx)**: Apple's machine learning framework for Apple Silicon ([Documentation](https://ml-explore.github.io/mlx/build/html/index.html))

## 📖 Citation

If you use this model in your research, please cite:

```bibtex
@software{{forgellm2025,
  title={{ForgeLLM: A Comprehensive Toolkit for Language Model Training}},
  author={{Laurent-Philippe Albou}},
  year={{2025}},
  url={{https://github.com/lpalbou/ForgeLLM}},
  note={{Version {self._get_forgellm_version()}}}
}}
```

## 📄 License

This model is released under the MIT License. The base model `{base_model}` follows its original license terms.

## 🔗 Links

- **ForgeLLM GitHub**: https://github.com/lpalbou/ForgeLLM
- **ForgeLLM PyPI**: https://pypi.org/project/forgellm/
- **Base Model**: https://huggingface.co/{base_model}
- **Training Dashboard**: [assets/training_dashboard.png](assets/training_dashboard.png)

---

*Generated by ForgeLLM v{self._get_forgellm_version()} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Comprehensive README created: {readme_path}")
        return readme_path
    
    def _create_model_card_metadata(self, training_data: Dict, adapter_config: Dict, 
                                  output_dir: Path, base_model: str):
        """Create model card metadata file"""
        metadata = {
            "library_name": "mlx",
            "pipeline_tag": "text-generation",
            "base_model": base_model,
            "tags": ["mlx", "forgellm", "continued-pretraining", "dataset"],
            "license": "mit",
            "training_details": {
                "training_type": training_data.get("training_type"),
                "session_id": training_data.get("session_id"),
                "start_time": training_data.get("start_time"),
                "end_time": training_data.get("end_time"),
                "config": training_data.get("config")
            },
            "generated_by": "ForgeLLM",
            "version": self._get_forgellm_version(),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_dir / "model_card_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model card metadata created: {metadata_path}")
    
    def _create_reproducibility_script(self, training_data: Dict, output_dir: Path):
        """Create reproducibility configuration file"""
        config = training_data["config"].copy()
        
        # Create a clean config for reproduction
        repro_config = {
            "# ForgeLLM Reproducibility Configuration": None,
            "# Generated automatically during model publication": None,
            "# Use: python -m forgellm.cli.commands train --config reproducibility_config.yaml": None,
            "training_type": config.get("training_type", "CPT"),
            "model": training_data.get("base_model"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "max_iterations": config.get("max_iterations"),
            "max_seq_length": config.get("max_seq_length"),
            "warmup_steps": config.get("warmup_steps"),
            "save_every": config.get("save_every"),
            "steps_per_report": config.get("steps_per_report"),
            "steps_per_eval": config.get("steps_per_eval"),
            "val_batches": config.get("val_batches"),
            "optimizer": config.get("optimizer", "adamw"),
            "weight_decay": config.get("weight_decay"),
            "lr_schedule": config.get("lr_schedule", "cosine_decay"),
            "seed": config.get("seed", 42),
            "input_dir": config.get("input_dir"),
            "data_dir": config.get("data_dir")
        }
        
        # Remove None values and comments
        clean_config = {k: v for k, v in repro_config.items() if v is not None and not k.startswith("#")}
        
        repro_path = output_dir / "reproducibility_config.yaml"
        
        # Write YAML manually for better formatting
        with open(repro_path, 'w') as f:
            f.write("# ForgeLLM Reproducibility Configuration\n")
            f.write("# Generated automatically during model publication\n")
            f.write("# Use: python -m forgellm.cli.commands train --config reproducibility_config.yaml\n\n")
            
            for key, value in clean_config.items():
                if isinstance(value, str):
                    f.write(f"{key}: \"{value}\"\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Reproducibility config created: {repro_path}")
    
    # Helper methods for formatting and analysis
    def _extract_iteration_from_checkpoint(self, checkpoint_path: Path) -> int:
        """Extract iteration number from checkpoint filename"""
        name = checkpoint_path.stem
        # Look for pattern like "0000300_adapters"
        parts = name.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    
    def _find_checkpoint_metrics(self, metrics: List[Dict], iteration: int) -> Optional[Dict]:
        """Find metrics for specific checkpoint iteration"""
        for metric in metrics:
            if metric.get("iteration") == iteration:
                return metric
        return None
    
    def _calculate_training_statistics(self, metrics: List[Dict], config: Dict) -> Dict:
        """Calculate comprehensive training statistics"""
        if not metrics:
            return {}
        
        # Extract numerical data
        train_losses = [m.get("train_loss") for m in metrics if m.get("train_loss") is not None]
        val_losses = [m.get("val_loss") for m in metrics if m.get("val_loss") is not None]
        tokens_per_sec = [m.get("tokens_per_sec") for m in metrics if m.get("tokens_per_sec") is not None]
        memory_usage = [m.get("peak_memory_gb") for m in metrics if m.get("peak_memory_gb") is not None]
        
        stats = {
            "total_iterations": len(metrics),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "avg_tokens_per_sec": np.mean(tokens_per_sec) if tokens_per_sec else None,
            "peak_memory_gb": max(memory_usage) if memory_usage else None,
            "total_trained_tokens": config.get("dataset_total_tokens", 0)
        }
        
        return stats
    
    def _format_duration(self, start_time: str, end_time: str) -> str:
        """Format training duration"""
        if not start_time or not end_time:
            return "N/A"
        
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = end - start
            
            hours = duration.total_seconds() // 3600
            minutes = (duration.total_seconds() % 3600) // 60
            
            return f"{int(hours)}h {int(minutes)}m"
        except:
            return "N/A"
    
    def _format_checkpoint_rationale(self, checkpoint_metrics: Optional[Dict], 
                                   best_checkpoints: List[Dict], checkpoint_iteration: int) -> str:
        """Format checkpoint selection rationale"""
        if not checkpoint_metrics:
            return f"- **Selected**: Iteration {checkpoint_iteration} (metrics not available)"
        
        val_loss = checkpoint_metrics.get("val_loss")
        train_loss = checkpoint_metrics.get("train_loss")
        
        rationale = f"- **Validation Loss**: {val_loss:.4f}\n"
        rationale += f"- **Training Loss**: {train_loss:.4f}\n"
        
        if checkpoint_metrics.get("val_perplexity"):
            rationale += f"- **Perplexity**: {checkpoint_metrics['val_perplexity']:.2f}\n"
        
        # Check if this is among the best checkpoints
        checkpoint_rank = None
        for i, cp in enumerate(best_checkpoints[:3], 1):
            if cp.get("iteration") == checkpoint_iteration:
                checkpoint_rank = i
                break
        
        if checkpoint_rank:
            rationale += f"- **Status**: ✅ Rank #{checkpoint_rank} among top 3 best checkpoints\n"
        else:
            rationale += "- **Status**: ⚠️ Custom selection (not in top 3 best checkpoints)\n"
        
        return rationale
    
    def _calculate_checkpoint_training_stats(self, checkpoint_iteration: int, config: Dict, training_data: Dict) -> Dict:
        """Calculate training statistics specific to this checkpoint"""
        batch_size = config.get('batch_size', 1)
        max_seq_length = config.get('max_seq_length', 1024)
        dataset_total_tokens = config.get('dataset_total_tokens', 0)
        
        # Calculate tokens trained up to this checkpoint
        tokens_per_iteration = batch_size * max_seq_length
        total_tokens_trained = checkpoint_iteration * tokens_per_iteration
        
        # Calculate epochs (if we have dataset size)
        epochs_completed = 0
        if dataset_total_tokens > 0:
            epochs_completed = total_tokens_trained / dataset_total_tokens
        
        return {
            'tokens_per_iteration': tokens_per_iteration,
            'total_tokens_trained': total_tokens_trained,
            'epochs_completed': epochs_completed,
            'checkpoint_iteration': checkpoint_iteration
        }
    
    def _format_best_checkpoints_table(self, best_checkpoints: List[Dict]) -> str:
        """Format best checkpoints table"""
        if not best_checkpoints:
            return """*Comprehensive multi-criteria checkpoint selection methodology using validation loss, generalization gap, loss stability, and convergence trends to identify optimal model states.*

No checkpoint analysis available - this may indicate insufficient validation data or analysis errors."""
        
        table = "| Rank | Iteration | Val Loss | Train Loss | Perplexity | Selection Reason |\n"
        table += "|------|-----------|----------|------------|------------|------------------|\n"
        
        for i, cp in enumerate(best_checkpoints[:3], 1):
            iteration = cp.get("iteration", "N/A")
            val_loss = f"{cp.get('val_loss', 0):.4f}" if cp.get('val_loss') else "N/A"
            train_loss = f"{cp.get('train_loss', 0):.4f}" if cp.get('train_loss') else "N/A"
            perplexity = f"{cp.get('val_perplexity', 0):.2f}" if cp.get('val_perplexity') else "N/A"
            reason = cp.get('selection_reason', 'Multi-criteria analysis')
            
            table += f"| {i} | {iteration} | {val_loss} | {train_loss} | {perplexity} | {reason} |\n"
        
        return table
    
    def _analyze_dataset_information(self, config: Dict, training_data: Dict) -> Dict:
        """Analyze dataset information"""
        return {
            "input_dir": config.get("input_dir", "N/A"),
            "data_dir": config.get("data_dir", "N/A"),
            "total_tokens": config.get("dataset_total_tokens", 0),
            "max_tokens_per_file": config.get("max_tokens_per_file", "N/A"),
            "validation_split": config.get("validation_split", 0.1)
        }
    
    def _format_dataset_information(self, dataset_info: Dict, config: Dict) -> str:
        """Format dataset information section"""
        val_split = dataset_info.get('validation_split', 0.1)
        total_tokens = dataset_info.get('total_tokens', 0)
        val_tokens = int(total_tokens * val_split) if total_tokens > 0 else 0
        train_tokens = total_tokens - val_tokens if total_tokens > 0 else 0
        val_batches = config.get('val_batches', 'N/A')
        
        info = f"""### Dataset Overview
| Property | Value |
|----------|-------|
| **Source Directory** | `{dataset_info.get('input_dir', 'N/A')}` |
| **Data Directory** | `{dataset_info.get('data_dir', 'N/A')}` |
| **Total Tokens** | {total_tokens:,} |
| **Max Tokens/File** | {dataset_info.get('max_tokens_per_file', 'N/A'):,} |

### Training Data Configuration
| Parameter | Value |
|-----------|-------|
| **Training Tokens** | {train_tokens:,} ({(1-val_split)*100:.1f}% of dataset) |
| **Sequence Length** | {config.get('max_seq_length', 'N/A')} tokens |
| **Data Mixture Ratio** | {config.get('data_mixture_ratio', 'N/A')} |

### Validation Data Configuration  
| Parameter | Value |
|-----------|-------|
| **Validation Tokens** | {val_tokens:,} ({val_split*100:.1f}% of dataset) |
| **Validation Batches** | {val_batches} |
| **Validation Frequency** | Every {config.get('steps_per_eval', 'N/A')} training steps |"""
        
        return info
    
    def _format_training_performance(self, stats: Dict) -> str:
        """Format training performance section"""
        # Helper function to format numerical values
        def format_value(value, format_str, default='N/A'):
            if value is not None and value != 'N/A':
                try:
                    return format_str.format(value)
                except:
                    return str(value)
            return default
        
        performance = f"""### Performance Metrics
| Metric | Value |
|--------|-------|
| **Total Iterations** | {stats.get('total_iterations', 'N/A'):,} |
| **Final Training Loss** | {format_value(stats.get('final_train_loss'), '{:.4f}')} |
| **Final Validation Loss** | {format_value(stats.get('final_val_loss'), '{:.4f}')} |
| **Best Validation Loss** | {format_value(stats.get('best_val_loss'), '{:.4f}')} |
| **Average Speed** | {format_value(stats.get('avg_tokens_per_sec'), '{:.1f} tokens/sec')} |
| **Peak Memory Usage** | {format_value(stats.get('peak_memory_gb'), '{:.1f} GB')} |
| **Total Tokens Processed** | {stats.get('total_trained_tokens', 0):,} |"""
        
        return performance
    
    def _get_forgellm_version(self) -> str:
        """Get ForgeLLM version"""
        # Try to get version from forgellm package
        try:
            import forgellm
            return getattr(forgellm, '__version__', '0.4.1')
        except ImportError:
            return '0.3.7'
    
    def _extract_model_architecture(self, base_model: str) -> str:
        """Extract model architecture from model name"""
        model_lower = base_model.lower()
        if 'qwen' in model_lower:
            return 'Qwen (Transformer)'
        elif 'gemma' in model_lower:
            return 'Gemma (Transformer)'
        elif 'llama' in model_lower:
            return 'LLaMA (Transformer)'
        elif 'mistral' in model_lower:
            return 'Mistral (Transformer)'
        elif 'phi' in model_lower:
            return 'Phi (Transformer)'
        else:
            return 'Transformer'
    
    def _estimate_parameter_count(self, base_model: str) -> str:
        """Estimate parameter count from model name"""
        model_lower = base_model.lower()
        if '1.7b' in model_lower or '1_7b' in model_lower:
            return '~1.7B parameters'
        elif '4b' in model_lower or '4_b' in model_lower:
            return '~4B parameters'
        elif '7b' in model_lower or '7_b' in model_lower:
            return '~7B parameters'
        elif '13b' in model_lower or '13_b' in model_lower:
            return '~13B parameters'
        else:
            return 'Unknown'
    
    def _extract_precision(self, base_model: str) -> str:
        """Extract precision from model name"""
        model_lower = base_model.lower()
        if 'bf16' in model_lower:
            return 'BFloat16'
        elif 'fp16' in model_lower:
            return 'Float16'
        elif 'fp32' in model_lower:
            return 'Float32'
        elif '8bit' in model_lower:
            return '8-bit'
        elif '4bit' in model_lower:
            return '4-bit'
        else:
            return 'Mixed Precision'
    
    def _create_placeholder_dashboard(self, assets_dir: Path):
        """Create a simple placeholder dashboard when generation fails"""
        try:
            # Configure matplotlib to use non-GUI backend for thread safety
            import matplotlib
            matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create a simple placeholder
            ax.text(0.5, 0.6, 'Training Dashboard', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=24, fontweight='bold', transform=ax.transAxes)
            
            ax.text(0.5, 0.4, 'Dashboard generation failed\nPlease check training logs for detailed metrics', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=14, transform=ax.transAxes)
            
            # Add a border
            rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, 
                                   edgecolor='gray', facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(assets_dir / "training_dashboard.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Created placeholder dashboard")
            
        except Exception as e:
            logger.warning(f"Failed to create placeholder dashboard: {e}")
            raise


def publish_checkpoint(checkpoint_path: str, output_dir: str = None) -> str:
    """
    Standalone function to publish a checkpoint (for CLI compatibility)
    
    Args:
        checkpoint_path: Path to checkpoint
        output_dir: Output directory (ignored, uses HF cache)
        
    Returns:
        Path to published model directory
    """
    publisher = ModelPublisher()
    result = publisher.publish_checkpoint(checkpoint_path)
    
    if result["success"]:
        return result["output_dir"]
    else:
        raise Exception(result["error"])