"""
API Request Validators
=====================

This module contains validators for API requests to ensure proper parameter handling.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def validate_training_request(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate training request data
    
    Args:
        request_data: The request data to validate
        
    Returns:
        Tuple of (is_valid, error_message, validated_data)
    """
    # Check required parameters
    required_params = ['model_name', 'input_dir', 'output_dir']
    for param in required_params:
        if param not in request_data:
            return False, f"Missing required parameter: {param}", None
    
    # Create a copy of the request data
    validated_data = request_data.copy()
    
    # Handle paths - ensure input_dir is an absolute path
    if not os.path.isabs(validated_data['input_dir']):
        # Convert relative path to absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        validated_data['input_dir'] = os.path.abspath(os.path.join(base_dir, '..', '..', validated_data['input_dir']))
        logger.info(f"Converted input_dir to absolute path: {validated_data['input_dir']}")
    
    # Validate parameter types
    type_validations = {
        'model_name': str,
        'input_dir': str,
        'output_dir': str,
        'batch_size': int,
        'learning_rate': float,
        'max_iterations': int,
        'max_seq_length': int,
        'warmup_steps': int,
        'save_every': int,
        'data_mixture_ratio': float,
        'overfitting_threshold': float,
    }
    
    for param, expected_type in type_validations.items():
        if param in validated_data and not isinstance(validated_data[param], expected_type):
            try:
                # Try to convert to the expected type
                validated_data[param] = expected_type(validated_data[param])
            except (ValueError, TypeError):
                return False, f"Invalid type for parameter {param}: expected {expected_type.__name__}", None
    
    return True, None, validated_data

def validate_model_request(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate model request data
    
    Args:
        request_data: The request data to validate
        
    Returns:
        Tuple of (is_valid, error_message, validated_data)
    """
    # Check required parameters
    if 'model_name' not in request_data:
        return False, "Missing required parameter: model_name", None
    
    # Create a copy of the request data
    validated_data = request_data.copy()
    
    # Handle adapter_path if present
    if 'adapter_path' in validated_data and not os.path.isabs(validated_data['adapter_path']):
        # Convert relative path to absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        validated_data['adapter_path'] = os.path.abspath(os.path.join(base_dir, '..', '..', validated_data['adapter_path']))
        logger.info(f"Converted adapter_path to absolute path: {validated_data['adapter_path']}")
    
    return True, None, validated_data

def validate_generation_request(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate text generation request data
    
    Args:
        request_data: The request data to validate
        
    Returns:
        Tuple of (is_valid, error_message, validated_data)
    """
    # Check required parameters
    if 'prompt' not in request_data:
        return False, "Missing required parameter: prompt", None
    
    # Create a copy of the request data
    validated_data = request_data.copy()
    
    # Validate parameter types
    type_validations = {
        'prompt': str,
        'max_tokens': int,
        'temperature': float,
        'top_p': float,
        'top_k': int,
        'repetition_penalty': float,
    }
    
    for param, expected_type in type_validations.items():
        if param in validated_data and not isinstance(validated_data[param], expected_type):
            try:
                # Try to convert to the expected type
                validated_data[param] = expected_type(validated_data[param])
            except (ValueError, TypeError):
                return False, f"Invalid type for parameter {param}: expected {expected_type.__name__}", None
    
    return True, None, validated_data 