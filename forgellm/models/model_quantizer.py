"""
Model Quantizer - Handles quantization of models using MLX-LM
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

class ModelQuantizer:
    """
    ModelQuantizer class for quantizing models using MLX-LM.
    """

    def __init__(self):
        """Initialize the ModelQuantizer."""
        # Set up models directory
        self.models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models'))
        logger.info(f"ModelQuantizer initialized with models directory: {self.models_dir}")
        
        # Set up HuggingFace cache directory for quantized models
        self.hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Also keep local quantized directory for backwards compatibility
        self.quantized_models_dir = os.path.join(self.models_dir, 'quantized')
        os.makedirs(self.quantized_models_dir, exist_ok=True)
        
        # Quantization state
        self.is_quantizing = False
        self.current_job = None
        self.progress = 0
        self.status_message = ""
        self.error = None
        
        # Progress tracking
        self.start_time = None
        self.estimated_time = None
        
        logger.info("ModelQuantizer initialized")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for quantization.
        
        Returns:
            List of model dictionaries with name, path, type, and size
        """
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
                    # Extract model name from directory (e.g., models--microsoft--DialoGPT-medium)
                    dir_name = os.path.basename(model_dir)
                    if dir_name.startswith("models--"):
                        model_name = dir_name[8:].replace("--", "/")  # Remove "models--" prefix and convert -- to /
                        
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

    def _calculate_directory_size(self, directory: str) -> float:
        """Calculate directory size in GB.
        
        Args:
            directory: Path to directory
            
        Returns:
            Size in GB
        """
        try:
            # Use subprocess to run du command for accurate directory size
            result = subprocess.run(
                ['du', '-sb', directory],  # -sb gives size in bytes
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0:
                # Parse the output to get size in bytes
                size_bytes = int(result.stdout.strip().split()[0])
                return size_bytes / (1024**3)  # Convert to GB
            else:
                # Fallback to Python calculation
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

    def start_quantization(self, model_path: str, bits: int = 4, group_size: int = 64) -> Dict[str, Any]:
        """Start quantization of a model.
        
        Args:
            model_path: Path to the model to quantize
            bits: Number of bits for quantization (4 or 8)
            group_size: Group size for quantization (32, 64, or 128)
            
        Returns:
            Dictionary with success status and job info
        """
        if self.is_quantizing:
            return {"success": False, "error": "Another quantization is already in progress"}
        
        try:
            # Validate parameters
            if bits not in [4, 8]:
                return {"success": False, "error": "Bits must be 4 or 8"}
            
            if group_size not in [32, 64, 128]:
                return {"success": False, "error": "Group size must be 32, 64, or 128"}
            
            # Check if model path exists
            if not os.path.exists(model_path):
                return {"success": False, "error": f"Model path does not exist: {model_path}"}
            
            # Generate output path with quantization suffix
            model_name = os.path.basename(model_path)
            
            # Determine the base name for the quantized model
            if model_name.startswith("models--"):
                # Input is from HF cache, extract the actual model name
                base_name = model_name[8:].replace("--", "/")  # Remove "models--" and convert back
            else:
                # Input is from local models directory
                base_name = model_name
            
            if group_size == 64:
                # Default group size, just add bits
                output_suffix = f"_Q{bits}"
            else:
                # Non-default group size, add both
                output_suffix = f"_Q{bits}_G{group_size}"
            
            # Create the quantized model name
            quantized_name = f"{base_name}{output_suffix}"
            
            # Convert to HF cache format
            hf_output_name = f"models--{quantized_name.replace('/', '--')}"
            output_path = os.path.join(self.hf_cache_dir, hf_output_name)
            
            # For display purposes, use the quantized name
            output_name = quantized_name
            
            # Check if output already exists
            if os.path.exists(output_path):
                return {"success": False, "error": f"Quantized model already exists: {output_name}"}
            
            # Create job info
            job_id = f"quant_{int(time.time())}"
            self.current_job = {
                "id": job_id,
                "input_path": model_path,
                "output_path": output_path,
                "output_name": output_name,
                "bits": bits,
                "group_size": group_size,
                "start_time": time.time(),
                "status": "starting"
            }
            
            # Reset state
            self.is_quantizing = True
            self.progress = 0
            self.status_message = "Initializing quantization..."
            self.error = None
            self.start_time = time.time()
            self.estimated_time = None
            
            # Start quantization in a separate thread
            threading.Thread(target=self._quantize_model_thread, daemon=True).start()
            
            logger.info(f"Started quantization job {job_id}: {model_path} -> {output_path}")
            
            return {
                "success": True, 
                "job_id": job_id,
                "output_name": output_name
            }
            
        except Exception as e:
            logger.error(f"Error starting quantization: {e}")
            self.is_quantizing = False
            self.error = str(e)
            return {"success": False, "error": str(e)}

    def _quantize_model_thread(self):
        """Thread function that performs the actual quantization."""
        try:
            job = self.current_job
            if not job:
                return
            
            # Update status
            self.status_message = "Preparing quantization..."
            self.progress = 10
            
            # Determine the model name/path for MLX-LM
            if job["input_path"].startswith(os.path.expanduser("~/.cache/huggingface")):
                # HuggingFace cached model - extract model name
                model_name = job["input_path"].split("models--")[1].replace("--", "/")
                
                # Check if this is a local model (like published/) or a real HuggingFace model
                if model_name.startswith("published/") or "2025-" in model_name:
                    # This is a local model that happens to be in HF cache - use full path
                    hf_path = job["input_path"]
                    logger.info(f"Using local model path for quantization: {hf_path}")
                else:
                    # This might be a real HuggingFace model - use model name
                    hf_path = model_name
                    logger.info(f"Using HuggingFace model name for quantization: {hf_path}")
            else:
                # Local model - use the full path
                hf_path = job["input_path"]
                logger.info(f"Using local model path for quantization: {hf_path}")
            
            self.progress = 20
            self.status_message = "Starting quantization process..."
            
            # MLX-LM expects the output directory to NOT exist, so don't create it beforehand
            # Just ensure the parent directory exists
            parent_dir = os.path.dirname(job["output_path"])
            os.makedirs(parent_dir, exist_ok=True)
            
            # Use MLX-LM convert command line tool
            import subprocess
            cmd = [
                "python", "-m", "mlx_lm.convert",
                "--hf-path", hf_path,
                "--mlx-path", job["output_path"],
                "--quantize",
                "--q-bits", str(job["bits"]),
                "--q-group-size", str(job["group_size"])
            ]
            
            logger.info(f"Running quantization command: {' '.join(cmd)}")
            self.progress = 30
            self.status_message = "Quantizing model weights..."
            
            # Run the quantization command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the process
            stdout_lines = []
            stderr_lines = []
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break
                
                # Read any available output
                try:
                    import select
                    ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                    
                    if process.stdout in ready:
                        line = process.stdout.readline()
                        if line:
                            stdout_lines.append(line.strip())
                            logger.debug(f"Quantization stdout: {line.strip()}")
                            
                            # Update progress based on output
                            if "Converting" in line:
                                self.progress = min(self.progress + 5, 80)
                            elif "Saving" in line:
                                self.progress = 85
                                self.status_message = "Saving quantized model..."
                    
                    if process.stderr in ready:
                        line = process.stderr.readline()
                        if line:
                            stderr_lines.append(line.strip())
                            logger.debug(f"Quantization stderr: {line.strip()}")
                
                except ImportError:
                    # Fallback for systems without select
                    time.sleep(0.5)
                    self.progress = min(self.progress + 2, 80)
            
            # Wait for process to complete and get final output
            stdout, stderr = process.communicate()
            if stdout:
                stdout_lines.extend(stdout.strip().split('\n'))
            if stderr:
                stderr_lines.extend(stderr.strip().split('\n'))
            
            # Check if the process succeeded
            if process.returncode == 0:
                self.progress = 90
                self.status_message = "Creating quantization metadata..."
                
                # Create quantization info file
                quant_info = {
                    "quantization": {
                        "bits": job["bits"],
                        "group_size": job["group_size"],
                        "method": "mlx_lm",
                        "original_model": job["input_path"],
                        "quantized_at": datetime.now().isoformat()
                    }
                }
                
                with open(os.path.join(job["output_path"], "quantization_info.json"), 'w') as f:
                    json.dump(quant_info, f, indent=2)
                
                # Complete
                self.progress = 100
                self.status_message = f"Quantization completed successfully! Saved to: {job['output_name']}"
                
                # Update job status
                job["status"] = "completed"
                job["end_time"] = time.time()
                job["duration"] = job["end_time"] - job["start_time"]
                
                logger.info(f"Quantization completed: {job['output_name']} in {job['duration']:.1f}s")
            else:
                # Process failed - check if it's a model card error
                error_msg = "\n".join(stderr_lines) if stderr_lines else f"Process failed with return code {process.returncode}"
                
                # Check if the error is related to model card creation but quantization succeeded
                if ("HFValidationError" in error_msg and "Repo id must be in the form" in error_msg and 
                    os.path.exists(job["output_path"]) and os.listdir(job["output_path"])):
                    
                    logger.warning(f"Model card creation failed but quantization appears successful: {error_msg}")
                    
                    # Treat as success since the model files exist
                    self.progress = 90
                    self.status_message = "Creating quantization metadata (model card creation failed)..."
                    
                    # Create quantization info file
                    quant_info = {
                        "quantization": {
                            "bits": job["bits"],
                            "group_size": job["group_size"],
                            "method": "mlx_lm",
                            "original_model": job["input_path"],
                            "quantized_at": datetime.now().isoformat(),
                            "note": "Model card creation failed but quantization succeeded"
                        }
                    }
                    
                    with open(os.path.join(job["output_path"], "quantization_info.json"), 'w') as f:
                        json.dump(quant_info, f, indent=2)
                    
                    # Complete as success
                    self.progress = 100
                    self.status_message = f"Quantization completed successfully! Saved to: {job['output_name']} (model card creation skipped)"
                    
                    # Update job status
                    job["status"] = "completed"
                    job["end_time"] = time.time()
                    job["duration"] = job["end_time"] - job["start_time"]
                    
                    logger.info(f"Quantization completed with model card warning: {job['output_name']} in {job['duration']:.1f}s")
                else:
                    # Real failure
                    raise Exception(f"Quantization process failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            self.error = str(e)
            self.status_message = f"Quantization failed: {e}"
            if self.current_job:
                self.current_job["status"] = "failed"
                self.current_job["error"] = str(e)
        finally:
            self.is_quantizing = False

    def get_quantization_status(self) -> Dict[str, Any]:
        """Get current quantization status.
        
        Returns:
            Dictionary with quantization status information
        """
        if not self.is_quantizing and not self.current_job:
            return {
                "is_quantizing": False,
                "progress": 0,
                "status_message": "Ready for quantization",
                "error": None
            }
        
        # Calculate estimated time remaining
        estimated_remaining = None
        if self.is_quantizing and self.start_time and self.progress > 0:
            elapsed = time.time() - self.start_time
            if self.progress > 10:  # Only estimate after some progress
                total_estimated = elapsed * (100 / self.progress)
                estimated_remaining = max(0, total_estimated - elapsed)
        
        return {
            "is_quantizing": self.is_quantizing,
            "progress": self.progress,
            "status_message": self.status_message,
            "error": self.error,
            "current_job": self.current_job,
            "estimated_remaining": estimated_remaining
        }

    def stop_quantization(self) -> Dict[str, Any]:
        """Stop current quantization (if possible).
        
        Returns:
            Dictionary with success status
        """
        if not self.is_quantizing:
            return {"success": False, "error": "No quantization in progress"}
        
        # Note: This is a simple implementation - in practice, stopping mid-quantization
        # might be difficult since MLX operations are not easily interruptible
        self.error = "Quantization stopped by user"
        self.status_message = "Quantization stopped"
        self.is_quantizing = False
        
        if self.current_job:
            self.current_job["status"] = "stopped"
        
        return {"success": True, "message": "Quantization stop requested"}

    def get_quantized_models(self) -> List[Dict[str, Any]]:
        """Get list of quantized models.
        
        Returns:
            List of quantized model dictionaries
        """
        models = []
        
        # Scan HuggingFace cache for quantized models
        if os.path.exists(self.hf_cache_dir):
            for model_dir in glob.glob(os.path.join(self.hf_cache_dir, "models--*")):
                if os.path.isdir(model_dir):
                    model_name = os.path.basename(model_dir)
                    
                    # Check if this is a quantized model (contains _Q4 or _Q8)
                    if "_Q4" in model_name or "_Q8" in model_name:
                        # Calculate model size
                        size_gb = self._calculate_directory_size(model_dir)
                        
                        # Load quantization info if available
                        quant_info_path = os.path.join(model_dir, "quantization_info.json")
                        quant_info = {}
                        if os.path.exists(quant_info_path):
                            try:
                                with open(quant_info_path, 'r') as f:
                                    quant_info = json.load(f).get("quantization", {})
                            except Exception as e:
                                logger.warning(f"Error loading quantization info for {model_name}: {e}")
                        
                        # Convert HF cache name back to readable name
                        if model_name.startswith("models--"):
                            # Remove "models--" prefix and convert "--" back to "/"
                            display_name = model_name[8:].replace("--", "/")
                        else:
                            display_name = model_name
                        
                        models.append({
                            "name": display_name,
                            "path": model_dir,
                            "full_path": model_dir,
                            "size": round(size_gb, 2),
                            "bits": quant_info.get("bits"),
                            "group_size": quant_info.get("group_size"),
                            "quantized_at": quant_info.get("quantized_at"),
                            "original_model": quant_info.get("original_model"),
                            "type": "QUANTIZED"
                        })
        
        # Also scan legacy quantized models directory for backwards compatibility
        if os.path.exists(self.quantized_models_dir):
            for model_path in glob.glob(os.path.join(self.quantized_models_dir, '*')):
                if os.path.isdir(model_path):
                    model_name = os.path.basename(model_path)
                    
                    # Calculate model size
                    size_gb = self._calculate_directory_size(model_path)
                    
                    # Load quantization info if available
                    quant_info_path = os.path.join(model_path, "quantization_info.json")
                    quant_info = {}
                    if os.path.exists(quant_info_path):
                        try:
                            with open(quant_info_path, 'r') as f:
                                quant_info = json.load(f).get("quantization", {})
                        except Exception as e:
                            logger.warning(f"Error loading quantization info for {model_name}: {e}")
                    
                    models.append({
                        "name": model_name,
                        "path": os.path.join('quantized', model_name),
                        "full_path": model_path,
                        "size": round(size_gb, 2),
                        "bits": quant_info.get("bits"),
                        "group_size": quant_info.get("group_size"),
                        "quantized_at": quant_info.get("quantized_at"),
                        "original_model": quant_info.get("original_model"),
                        "type": "QUANTIZED_LEGACY"
                    })
        
        # Sort by creation time (newest first), handle None values
        models.sort(key=lambda x: x.get("quantized_at") or "", reverse=True)
        
        return models 