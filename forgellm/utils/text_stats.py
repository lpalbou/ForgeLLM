#!/usr/bin/env python3
"""
Centralized Text Statistics Utility for ForgeLLM

This module provides consistent and accurate text analysis across the entire application,
including proper token counting using actual tokenizers rather than word splitting.

Key Features:
- Accurate token counting using model-specific tokenizers
- Word, line, and page estimates
- Consistent API across training, chat, and CLI components
- Support for different tokenizer types (MLX, HuggingFace, etc.)
"""

import logging
import re
from typing import Dict, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Avoid importing ProcessTracker in utility modules to prevent singleton conflicts
# ProcessTracker is a singleton that manages long-running training processes
# and should not be imported by utility scripts that might exit and trigger cleanup


class TextStatsCalculator:
    """
    Centralized text statistics calculator with accurate tokenization.
    
    This class provides consistent text analysis across the entire ForgeLLM application,
    ensuring that token counts are accurate and consistent whether used in training
    estimates, chat statistics, or CLI tools.
    """
    
    def __init__(self, tokenizer=None, model_name: Optional[str] = None):
        """
        Initialize the text stats calculator.
        
        Args:
            tokenizer: MLX or HuggingFace tokenizer instance (optional)
            model_name: Model name for fallback tokenizer loading (optional)
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._fallback_tokenizer = None
        
    def get_stats(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive text statistics for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
            - tokens: Accurate token count using proper tokenizer
            - words: Word count
            - lines: Line count
            - pages: Estimated pages (single-spaced, ~250 words per page)
            - chars: Character count
            - chars_no_spaces: Character count excluding spaces
            - avg_words_per_line: Average words per line
            - tokenizer_used: Which tokenizer was used for token counting
        """
        if not text or not text.strip():
            return self._empty_stats()
            
        # Basic counts
        chars = len(text)
        chars_no_spaces = len(text.replace(' ', ''))
        lines = len(text.splitlines())
        words = len(text.split())
        
        # Calculate average words per line (avoid division by zero)
        avg_words_per_line = words / lines if lines > 0 else 0
        
        # Estimate pages (250 words per page is a standard estimate for single-spaced text)
        pages = words / 250.0
        
        # Get accurate token count
        tokens, tokenizer_used = self._count_tokens_accurate(text)
        
        return {
            'tokens': tokens,
            'words': words,
            'lines': lines,
            'pages': round(pages, 2),
            'chars': chars,
            'chars_no_spaces': chars_no_spaces,
            'avg_words_per_line': round(avg_words_per_line, 1),
            'tokenizer_used': tokenizer_used
        }
    
    def _count_tokens_accurate(self, text: str) -> tuple[int, str]:
        """
        Count tokens using the most accurate method available.
        
        Priority order:
        1. Provided tokenizer (most accurate, if available)
        2. tiktoken with cl100k_base (downloaded once, works for all models)
        3. Word-based estimation (fallback)
        
        NEVER DOWNLOADS MODEL-SPECIFIC TOKENIZERS - use tiktoken for all models.
        """
        # Method 1: Use provided tokenizer if available (most accurate)
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, 'encode'):
                    tokens = len(self.tokenizer.encode(text))
                    return tokens, f"model_tokenizer_{type(self.tokenizer).__name__}"
                elif hasattr(self.tokenizer, '__call__'):
                    # Some tokenizers are callable
                    encoded = self.tokenizer(text)
                    if hasattr(encoded, 'input_ids'):
                        tokens = len(encoded.input_ids)
                        return tokens, f"hf_tokenizer_{type(self.tokenizer).__name__}"
            except Exception as e:
                logger.warning(f"Failed to use provided tokenizer: {e}")
        
        # Method 2: Use tiktoken cl100k_base (downloaded once, works for all models)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            return len(tokens), "tiktoken_cl100k_base"
        except Exception as e:
            logger.warning(f"Failed to use tiktoken: {e}")
        
        # Method 3: Word-based estimation (no downloads, always works)
        words = len(text.split())
        estimated_tokens = int(words * 1.4)  # Conservative estimate
        return estimated_tokens, "word_estimation"
    
    def _load_fallback_tokenizer(self):
        """
        Try to load a tokenizer for the current model as fallback.
        NEVER DOWNLOAD MODEL-SPECIFIC TOKENIZERS - only use tiktoken's cl100k_base.
        """
        # EXPLICITLY FORBIDDEN: Never download model-specific tokenizers
        # tiktoken's cl100k_base is downloaded once and works for all models
        return None
    
    def _empty_stats(self) -> Dict[str, Union[int, float]]:
        """Return empty statistics for empty/None text."""
        return {
            'tokens': 0,
            'words': 0,
            'lines': 0,
            'pages': 0.0,
            'chars': 0,
            'chars_no_spaces': 0,
            'avg_words_per_line': 0.0,
            'tokenizer_used': 'none'
        }
    
    def format_stats(self, stats: Dict[str, Union[int, float]], detailed: bool = False) -> str:
        """
        Format statistics into a human-readable string.
        
        Args:
            stats: Statistics dictionary from get_stats()
            detailed: Whether to include detailed breakdown
            
        Returns:
            Formatted string representation
        """
        if not detailed:
            return f"{stats['tokens']:,} tokens, {stats['words']:,} words, {stats['lines']:,} lines, {stats['pages']:.1f} pages"
        
        return (
            f"Text Statistics:\n"
            f"  Tokens: {stats['tokens']:,} (via {stats['tokenizer_used']})\n"
            f"  Words: {stats['words']:,}\n"
            f"  Lines: {stats['lines']:,}\n"
            f"  Pages: {stats['pages']:.2f} (est. 250 words/page)\n"
            f"  Characters: {stats['chars']:,} ({stats['chars_no_spaces']:,} without spaces)\n"
            f"  Avg words/line: {stats['avg_words_per_line']:.1f}"
        )


# Convenience functions for common use cases
def get_text_stats(text: str, tokenizer=None, model_name: Optional[str] = None) -> Dict[str, Union[int, float]]:
    """
    Convenience function to get text statistics.
    
    Args:
        text: Text to analyze
        tokenizer: Optional tokenizer instance
        model_name: Optional model name for fallback tokenizer
        
    Returns:
        Statistics dictionary
    """
    calculator = TextStatsCalculator(tokenizer=tokenizer, model_name=model_name)
    return calculator.get_stats(text)


def count_tokens_accurate(text: str, tokenizer=None, model_name: Optional[str] = None) -> int:
    """
    Convenience function to get just the token count.
    
    Args:
        text: Text to analyze
        tokenizer: Optional tokenizer instance  
        model_name: Optional model name for fallback tokenizer
        
    Returns:
        Accurate token count
    """
    stats = get_text_stats(text, tokenizer=tokenizer, model_name=model_name)
    return stats['tokens']


def format_text_stats(text: str, tokenizer=None, model_name: Optional[str] = None, detailed: bool = False) -> str:
    """
    Convenience function to get formatted text statistics.
    
    Args:
        text: Text to analyze
        tokenizer: Optional tokenizer instance
        model_name: Optional model name for fallback tokenizer
        detailed: Whether to include detailed breakdown
        
    Returns:
        Formatted statistics string
    """
    calculator = TextStatsCalculator(tokenizer=tokenizer, model_name=model_name)
    stats = calculator.get_stats(text)
    return calculator.format_stats(stats, detailed=detailed)


# Legacy compatibility functions (to be used during migration)
def estimate_tokens_from_words(word_count: int) -> int:
    """
    Legacy function for word-based token estimation.
    
    Note: This is less accurate than proper tokenization and should be
    replaced with count_tokens_accurate() when possible.
    """
    return int(word_count * 1.4)


def validate_token_count(text: str, reported_count: int, tokenizer=None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a reported token count against accurate calculation.
    
    Args:
        text: Original text
        reported_count: Previously calculated token count
        tokenizer: Optional tokenizer instance
        model_name: Optional model name
        
    Returns:
        Dictionary with validation results
    """
    accurate_stats = get_text_stats(text, tokenizer=tokenizer, model_name=model_name)
    accurate_count = accurate_stats['tokens']
    
    difference = abs(accurate_count - reported_count)
    percentage_error = (difference / accurate_count * 100) if accurate_count > 0 else 0
    
    return {
        'reported_count': reported_count,
        'accurate_count': accurate_count,
        'difference': difference,
        'percentage_error': round(percentage_error, 1),
        'tokenizer_used': accurate_stats['tokenizer_used'],
        'is_accurate': percentage_error < 5.0,  # Within 5% is considered accurate
        'recommendation': 'accurate' if percentage_error < 5.0 else 'needs_update'
    }


if __name__ == "__main__":
    # Example usage and testing
    sample_text = """
    This is a sample text for testing the text statistics calculator.
    It contains multiple lines and various punctuation marks.
    The calculator should provide accurate token counts using proper tokenization.
    """
    
    print("=== Text Statistics Calculator Test ===")
    stats = get_text_stats(sample_text)
    print(format_text_stats(sample_text, detailed=True))
    
    print(f"\nToken count only: {count_tokens_accurate(sample_text)}")
    
    # Test validation
    fake_count = len(sample_text.split())  # Word count (inaccurate)
    validation = validate_token_count(sample_text, fake_count)
    print(f"\nValidation test:")
    print(f"  Reported (word count): {validation['reported_count']}")
    print(f"  Accurate: {validation['accurate_count']}")
    print(f"  Error: {validation['percentage_error']}%")
    print(f"  Recommendation: {validation['recommendation']}") 