"""
Model Fuser - Handles fusion of foundation models with LoRA/DoRA adapters using MLX-LM
"""

import os
import logging
import threading
import time
import glob
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import subprocess
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ModelFuser:
    """
    ModelFuser class for fusing foundation models with LoRA/DoRA adapters using MLX-LM.
    """

    def __init__(self):
        """Initialize the ModelFuser."""
        # Set up models directory
        self.models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models'))
        logger.info(f"ModelFuser initialized with models directory: {self.models_dir}")
        
        # Set up HuggingFace cache directory for fused models
        self.hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Fusion state
        self.is_fusing = False
        self.current_job = None
        self.progress = 0
        self.status_message = ""
        self.error = None
        
        # Progress tracking
        self.start_time = None
        self.estimated_time = None
        
        logger.info("ModelFuser initialized")

    def get_available_base_models(self) -> List[Dict[str, Any]]:
        """Get list of available base models for fusion."""
        models = []
        
        # Scan local models directory
        for model_type in ['base', 'cpt', 'ift']:
            type_dir = os.path.join(self.models_dir, model_type)
            if os.path.exists(type_dir):
                for model_path in glob.glob(os.path.join(type_dir, '*')):
                    if os.path.isdir(model_path):
                        model_name = os.path.basename(model_path)
                        
                        # Calculate model size
                        size_gb = self._calculate_directory_size(model_path)
                        
                        models.append({
                            "name": model_name,
                            "path": os.path.join(model_type, model_name),
                            "full_path": model_path,
                            "type": model_type.upper(),
                            "size": round(size_gb, 2)
                        })
        
        # Scan HuggingFace cache
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(hf_cache_dir):
            for model_dir in glob.glob(os.path.join(hf_cache_dir, "models--*")):
                if os.path.isdir(model_dir):
                    # Extract model name from directory
                    dir_name = os.path.basename(model_dir)
                    if dir_name.startswith("models--"):
                        model_name = dir_name[8:].replace("--", "/")
                        
                        # Calculate model size
                        size_gb = self._calculate_directory_size(model_dir)
                        
                        # Only include models > 100MB to filter out small files
                        if size_gb > 0.1:
                            models.append({
                                "name": model_name,
                                "path": model_dir,
                                "full_path": model_dir,
                                "type": "HF_CACHE",
                                "size": round(size_gb, 2)
                            })
        
        # Sort by size (largest first)
        models.sort(key=lambda x: x["size"], reverse=True)
        
        return models

    def get_available_adapters(self) -> List[Dict[str, Any]]:
        """Get list of available LoRA/DoRA adapters for fusion."""
        adapters = []
        
        # Scan CPT models directory for adapters
        cpt_dir = os.path.join(self.models_dir, 'cpt')
        if os.path.exists(cpt_dir):
            for adapter_path in glob.glob(os.path.join(cpt_dir, '*')):
                if os.path.isdir(adapter_path):
                    # Check if this contains adapter files (both .npz and .safetensors formats)
                    adapter_files_npz = glob.glob(os.path.join(adapter_path, 'adapters.npz'))
                    adapter_files_safetensors = glob.glob(os.path.join(adapter_path, 'adapters.safetensors'))
                    
                    if adapter_files_npz or adapter_files_safetensors:
                        adapter_name = os.path.basename(adapter_path)
                        
                        # Calculate adapter size
                        size_gb = self._calculate_directory_size(adapter_path)
                        
                        # Try to determine adapter type from config
                        adapter_type = self._detect_adapter_type(adapter_path)
                        
                        adapters.append({
                            "name": adapter_name,
                            "path": adapter_path,
                            "full_path": adapter_path,
                            "type": adapter_type,
                            "size": round(size_gb, 2)
                        })
        
        # Sort by name
        adapters.sort(key=lambda x: x["name"])
        
        return adapters

    def _detect_adapter_type(self, adapter_path: str) -> str:
        """Detect the type of adapter (LoRA/DoRA) from config files."""
        try:
            # Check for adapter_config.json
            config_path = os.path.join(adapter_path, 'adapter_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'peft_type' in config:
                        return config['peft_type'].upper()
                    elif 'adapter_type' in config:
                        return config['adapter_type'].upper()
            
            # Check for config.json
            config_path = os.path.join(adapter_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'fine_tune_type' in config:
                        return config['fine_tune_type'].upper()
            
            # Default assumption
            return "LoRA"
        except Exception as e:
            logger.warning(f"Error detecting adapter type for {adapter_path}: {e}")
            return "LoRA"

    def _calculate_directory_size(self, directory: str) -> float:
        """Calculate directory size in GB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**3)
        except Exception as e:
            logger.warning(f"Error calculating size for {directory}: {e}")
            return 0
    
    def _is_valid_base_model(self, base_model_path: str) -> bool:
        """Check if base model path is valid (local path or HuggingFace model name)."""
        # If it's a local path, check if it exists
        if os.path.exists(base_model_path):
            return True
        
        # If it contains a slash, it might be a HuggingFace model name (e.g., "mlx-community/gemma-3-4b-it-bf16")
        if "/" in base_model_path and not base_model_path.startswith("/"):
            # This looks like a HuggingFace model name - MLX-LM can handle these
            return True
        
        # Otherwise, it's invalid
        return False
    
    def _create_readme(self, output_path: str, job_info: Dict[str, Any]) -> None:
        """Create a README.md file for the fused model."""
        try:
            readme_path = os.path.join(output_path, "README.md")
            
            # Extract model and adapter names for display
            base_model_name = os.path.basename(job_info["base_model_path"])
            if base_model_name.startswith("models--"):
                base_model_name = base_model_name[8:].replace("--", "/")
            
            adapter_name = os.path.basename(job_info["adapter_path"])
            
            # Create README content
            readme_content = f"""# {job_info['output_name']}

## Model Description

{job_info['description']}

## Model Details

- **Base Model**: {base_model_name}
- **Adapter**: {adapter_name}
- **Fusion Method**: MLX-LM Fuse
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

This model was created by fusing a base model with a LoRA/DoRA adapter using MLX-LM.

```python
# Load the model using MLX-LM
from mlx_lm import load, generate

model, tokenizer = load("{job_info['output_name']}")
response = generate(model, tokenizer, prompt="Your prompt here", max_tokens=100)
print(response)
```

## Technical Details

- **Fusion Process**: The base model and adapter were fused using the MLX-LM fuse command
- **Output Format**: HuggingFace compatible format
- **Storage Location**: HuggingFace cache directory

---

*This model was created using the forgeLLM training interface.*
"""
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"Created README.md for fused model: {readme_path}")
            
        except Exception as e:
            logger.warning(f"Error creating README.md: {e}")
            # Don't fail the fusion process if README creation fails

    def start_fusion(self, base_model_path: str, adapter_path: str, suffix: str = "", description: str = "") -> Dict[str, Any]:
        """Start fusion of a base model with an adapter."""
        if self.is_fusing:
            return {"success": False, "error": "Another fusion is already in progress"}
        
        try:
            # Check if base model path exists (allow HuggingFace model names)
            if not self._is_valid_base_model(base_model_path):
                return {"success": False, "error": f"Base model path does not exist: {base_model_path}"}
            
            if not os.path.exists(adapter_path):
                return {"success": False, "error": f"Adapter path does not exist: {adapter_path}"}
            
            # Check if adapter has the required files (both .npz and .safetensors formats)
            adapter_files_npz = glob.glob(os.path.join(adapter_path, 'adapters.npz'))
            adapter_files_safetensors = glob.glob(os.path.join(adapter_path, 'adapters.safetensors'))
            
            if not adapter_files_npz and not adapter_files_safetensors:
                return {"success": False, "error": f"No adapters.npz or adapters.safetensors found in: {adapter_path}"}
            
            # Generate output path
            adapter_name = os.path.basename(adapter_path)
            
            # Determine the base name for the fused model
            if "/" in base_model_path and not base_model_path.startswith("/"):
                # This is a HuggingFace model name (e.g., "mlx-community/gemma-3-4b-it-bf16")
                base_name = base_model_path
            else:
                # This is a local path
                base_model_name = os.path.basename(base_model_path)
                if base_model_name.startswith("models--"):
                    # Input is from HF cache, extract the actual model name
                    base_name = base_model_name[8:].replace("--", "/")
                else:
                    # Input is from local models directory
                    base_name = base_model_name
            
            # Create the fused model name
            fused_name = f"{base_name}_fused_{adapter_name}"
            
            # Add suffix if provided
            if suffix:
                fused_name += suffix
            
            # Convert to HF cache format
            hf_output_name = f"models--{fused_name.replace('/', '--')}"
            output_path = os.path.join(self.hf_cache_dir, hf_output_name)
            
            # Check if output already exists
            if os.path.exists(output_path):
                return {"success": False, "error": f"Fused model already exists: {fused_name}"}
            
            # Create job info
            job_id = f"fuse_{int(time.time())}"
            self.current_job = {
                "id": job_id,
                "base_model_path": base_model_path,
                "adapter_path": adapter_path,
                "output_path": output_path,
                "output_name": fused_name,
                "suffix": suffix,
                "description": description,
                "start_time": datetime.now().isoformat()
            }
            
            # Reset state
            self.is_fusing = True
            self.progress = 0
            self.status_message = "Starting fusion..."
            self.error = None
            self.start_time = time.time()
            
            # Start fusion in a separate thread
            fusion_thread = threading.Thread(target=self._fuse_model_thread)
            fusion_thread.daemon = True
            fusion_thread.start()
            
            logger.info(f"Started fusion job {job_id}: {base_model_path} + {adapter_path} -> {output_path}")
            
            return {
                "success": True,
                "job_id": job_id,
                "output_name": fused_name,
                "message": "Fusion started successfully"
            }
            
        except Exception as e:
            logger.error(f"Error starting fusion: {e}")
            return {"success": False, "error": str(e)}

    def _fuse_model_thread(self):
        """Run the fusion process in a separate thread."""
        try:
            if not self.current_job:
                raise ValueError("No current job")
            
            base_model_path = self.current_job["base_model_path"]
            adapter_path = self.current_job["adapter_path"]
            output_path = self.current_job["output_path"]
            
            logger.info(f"Starting fusion: {base_model_path} + {adapter_path} -> {output_path}")
            
            # Update status
            self.status_message = "Preparing fusion..."
            self.progress = 5
            
            # Create temporary output directory
            temp_output = output_path + "_temp"
            os.makedirs(temp_output, exist_ok=True)
            
            try:
                # Update status
                self.status_message = "Running MLX-LM fuse..."
                self.progress = 10
                
                # Prepare the MLX-LM fuse command
                cmd = [
                    "python", "-m", "mlx_lm.fuse",
                    "--model", base_model_path,
                    "--adapter-path", adapter_path,
                    "--save-path", temp_output
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
                
                # Run the fusion command
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Monitor the process output
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.info(f"MLX-LM fuse: {output.strip()}")
                        
                        # Update progress based on output patterns
                        if "Loading" in output:
                            self.progress = 20
                            self.status_message = "Loading models..."
                        elif "Fusing" in output or "Merging" in output:
                            self.progress = 50
                            self.status_message = "Fusing layers..."
                        elif "Saving" in output:
                            self.progress = 80
                            self.status_message = "Saving fused model..."
                        elif "Done" in output or "Complete" in output:
                            self.progress = 90
                            self.status_message = "Finalizing..."
                
                # Check if process completed successfully
                return_code = process.poll()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                
                # Update status
                self.status_message = "Moving to final location..."
                self.progress = 95
                
                # Move from temp to final location
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.move(temp_output, output_path)
                
                # Create README.md if description is provided
                if self.current_job.get("description"):
                    self._create_readme(output_path, self.current_job)
                
                # Success
                self.progress = 100
                self.status_message = "Fusion completed successfully!"
                self.is_fusing = False
                
                logger.info(f"Fusion completed successfully: {output_path}")
                
            except Exception as e:
                # Cleanup temp directory
                if os.path.exists(temp_output):
                    shutil.rmtree(temp_output)
                raise e
                
        except Exception as e:
            logger.error(f"Error during fusion: {e}")
            self.error = str(e)
            self.status_message = f"Error: {str(e)}"
            self.is_fusing = False

    def get_fusion_status(self) -> Dict[str, Any]:
        """Get current fusion status."""
        status = {
            "is_fusing": self.is_fusing,
            "progress": self.progress,
            "status_message": self.status_message,
            "error": self.error
        }
        
        if self.current_job:
            status.update({
                "job_id": self.current_job["id"],
                "base_model_path": self.current_job["base_model_path"],
                "adapter_path": self.current_job["adapter_path"],
                "output_name": self.current_job["output_name"],
                "start_time": self.current_job["start_time"]
            })
            
            # Calculate elapsed and estimated time
            if self.start_time:
                elapsed = time.time() - self.start_time
                status["elapsed_time"] = elapsed
                
                if self.progress > 0:
                    estimated_total = elapsed / (self.progress / 100)
                    remaining = max(0, estimated_total - elapsed)
                    status["estimated_remaining"] = remaining
        
        return status

    def stop_fusion(self) -> Dict[str, Any]:
        """Stop current fusion process."""
        if not self.is_fusing:
            return {"success": False, "error": "No fusion in progress"}
        
        try:
            # Set flag to stop
            self.is_fusing = False
            self.status_message = "Stopping fusion..."
            
            self.status_message = "Fusion stopped"
            self.current_job = None
            
            logger.info("Fusion stopped by user")
            
            return {"success": True, "message": "Fusion stopped"}
            
        except Exception as e:
            logger.error(f"Error stopping fusion: {e}")
            return {"success": False, "error": str(e)}

    def get_fused_models(self) -> List[Dict[str, Any]]:
        """Get list of fused models."""
        models = []
        
        # Scan HuggingFace cache for fused models
        if os.path.exists(self.hf_cache_dir):
            for model_dir in glob.glob(os.path.join(self.hf_cache_dir, "models--*_fused_*")):
                if os.path.isdir(model_dir):
                    # Extract model name from directory
                    dir_name = os.path.basename(model_dir)
                    if dir_name.startswith("models--"):
                        model_name = dir_name[8:].replace("--", "/")
                        
                        # Calculate model size
                        size_gb = self._calculate_directory_size(model_dir)
                        
                        # Get creation time
                        creation_time = os.path.getctime(model_dir)
                        
                        models.append({
                            "name": model_name,
                            "path": model_dir,
                            "size": round(size_gb, 2),
                            "created": datetime.fromtimestamp(creation_time).isoformat(),
                            "type": "FUSED"
                        })
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x["created"], reverse=True)
        
        return models
