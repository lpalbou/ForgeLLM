"""Learning rate scheduling utilities for training."""

import math
from typing import Callable, Dict, Any, Optional


def cosine_decay(
    step: int, 
    initial_lr: float, 
    decay_steps: int, 
    final_lr: float
) -> float:
    """Cosine decay learning rate schedule.
    
    Args:
        step: Current step
        initial_lr: Initial learning rate
        decay_steps: Number of decay steps
        final_lr: Final learning rate
        
    Returns:
        Learning rate for the current step
    """
    if decay_steps <= 0:
        return initial_lr
        
    if step >= decay_steps:
        return final_lr
        
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - cosine_decay)
    return final_lr + (initial_lr - final_lr) * decayed


def linear_decay(
    step: int, 
    initial_lr: float, 
    decay_steps: int, 
    final_lr: float
) -> float:
    """Linear decay learning rate schedule.
    
    Args:
        step: Current step
        initial_lr: Initial learning rate
        decay_steps: Number of decay steps
        final_lr: Final learning rate
        
    Returns:
        Learning rate for the current step
    """
    if decay_steps <= 0:
        return initial_lr
        
    if step >= decay_steps:
        return final_lr
        
    decay_rate = (initial_lr - final_lr) / decay_steps
    return initial_lr - decay_rate * step


def constant_lr(
    step: int, 
    initial_lr: float, 
    decay_steps: int, 
    final_lr: float
) -> float:
    """Constant learning rate schedule.
    
    Args:
        step: Current step
        initial_lr: Initial learning rate
        decay_steps: Number of decay steps (unused)
        final_lr: Final learning rate (unused)
        
    Returns:
        Learning rate for the current step
    """
    return initial_lr


def apply_warmup(
    step: int,
    lr: float,
    warmup_steps: int,
    initial_lr: float
) -> float:
    """Apply warmup to learning rate.
    
    Args:
        step: Current step
        lr: Learning rate after decay
        warmup_steps: Number of warmup steps
        initial_lr: Initial learning rate
        
    Returns:
        Learning rate with warmup applied
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return lr
        
    # Linear warmup
    warmup_factor = step / warmup_steps
    return initial_lr * warmup_factor


def get_scheduler(name: str) -> Callable:
    """Get scheduler function by name.
    
    Args:
        name: Scheduler name
        
    Returns:
        Scheduler function
    """
    schedulers = {
        "cosine_decay": cosine_decay,
        "linear_decay": linear_decay,
        "constant": constant_lr
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available schedulers: {list(schedulers.keys())}")
        
    return schedulers[name]


def calculate_lr(
    step: int,
    initial_lr: float,
    decay_steps: int,
    warmup_steps: int,
    final_lr: Optional[float] = None,
    scheduler_name: str = "cosine_decay"
) -> float:
    """Calculate learning rate for the current step.
    
    Args:
        step: Current step
        initial_lr: Initial learning rate
        decay_steps: Number of decay steps
        warmup_steps: Number of warmup steps
        final_lr: Final learning rate (default: 10% of initial_lr)
        scheduler_name: Scheduler name
        
    Returns:
        Learning rate for the current step
    """
    if final_lr is None:
        final_lr = initial_lr * 0.1
        
    scheduler = get_scheduler(scheduler_name)
    
    if step < warmup_steps:
        # Linear warmup
        return apply_warmup(step, initial_lr, warmup_steps, initial_lr)
    else:
        # Apply decay after warmup
        decay_step = step - warmup_steps
        decay_steps_after_warmup = decay_steps - warmup_steps
        
        return scheduler(
            decay_step,
            initial_lr,
            max(1, decay_steps_after_warmup),
            final_lr
        ) 