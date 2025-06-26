"""
Data processing utilities for continued pre-training and fine-tuning
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from forgellm.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Efficiently process documents for continued pre-training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.supported_extensions = {'.txt', '.md', '.rst', '.py', '.json'}
        
    def is_valid_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        return (
            file_path.is_file() and 
            file_path.suffix.lower() in self.supported_extensions and
            file_path.stat().st_size > 0  # Non-empty files
        )
    
    def extract_text_from_file(self, file_path: Path) -> Optional[str]:
        """Extract text content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                
            if not content:
                return None
                
            # Add metadata for better context
            metadata = f"# File: {file_path.relative_to(Path(self.config.input_dir))}\n\n"
            return metadata + content
            
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None
    
    def collect_documents(self) -> List[Path]:
        """Recursively collect all valid documents"""
        documents = []
        input_path = Path(self.config.input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_path} does not exist")
            
        for file_path in input_path.rglob('*'):
            if self.is_valid_file(file_path):
                documents.append(file_path)
                
        logger.info(f"Found {len(documents)} documents to process")
        return sorted(documents)
    
    def chunk_text(self, text: str, max_length: int = 2048) -> List[str]:
        """
        Chunk text into smaller pieces while preserving meaning.
        Each chunk will later have an EOS token appended when saved to JSONL.
        """
        # Be more conservative to avoid truncation warnings
        target_length = max_length // 3  # Use 1/3 of max length for safety (e.g., 682 words for 2048 tokens)
        
        # Split by paragraphs first, then sentences
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Split long paragraphs into sentences
            sentences = paragraph.split('. ')
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > target_length and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                    
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        # Filter chunks and ensure they're not too long
        valid_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Filter very short chunks
                # Double-check chunk length and split if needed
                words = chunk.split()
                if len(words) > target_length:
                    # Split oversized chunks more aggressively
                    for i in range(0, len(words), target_length // 2):  # Overlap for context
                        sub_chunk = ' '.join(words[i:i + target_length // 2])
                        if len(sub_chunk.strip()) > 50:
                            valid_chunks.append(sub_chunk)
                else:
                    valid_chunks.append(chunk)
                    
        logger.info(f"Created {len(valid_chunks)} chunks with max ~{target_length} words each")
        return valid_chunks


class DataMixtureProcessor:
    """Handle data mixture strategies for continued pre-training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def create_general_data_samples(self, num_samples: int) -> List[str]:
        """Create general data samples to mix with domain data"""
        
        # Option 1: Use training document as general data (more realistic)
        training_doc_path = Path("training and fine tuning.md")
        if training_doc_path.exists():
            try:
                with open(training_doc_path, 'r', encoding='utf-8') as f:
                    training_content = f.read().strip()
                
                # Chunk the training document
                doc_processor = DocumentProcessor(self.config)
                training_chunks = doc_processor.chunk_text(training_content, self.config.max_seq_length)
                
                # Repeat chunks to reach desired sample count
                samples = []
                for i in range(num_samples):
                    chunk_idx = i % len(training_chunks)
                    samples.append(training_chunks[chunk_idx])
                
                logger.info(f"Using training document as general data source ({len(training_chunks)} unique chunks)")
                return samples
                
            except Exception as e:
                logger.warning(f"Failed to load training document: {e}")
        
        # Option 2: Fallback to minimal general samples (better than before)
        general_samples = [
            "Artificial intelligence and machine learning are transforming how we process information and make decisions in complex systems.",
            "The development of neural networks has enabled computers to recognize patterns and learn from data in ways that mimic human cognition.",
            "Natural language processing allows machines to understand, interpret, and generate human language with increasing sophistication.",
            "Deep learning architectures like transformers have revolutionized the field of artificial intelligence and language modeling.",
            "The training of large language models requires massive computational resources and carefully curated datasets.",
            "Backpropagation and gradient descent form the mathematical foundation for how neural networks learn from examples.",
            "Parameter-efficient fine-tuning methods like LoRA enable adaptation of large models without full retraining costs.",
            "Continued pre-training allows models to acquire domain-specific knowledge while preserving general capabilities.",
        ]
        
        # Create varied samples
        samples = []
        for i in range(num_samples):
            base_sample = general_samples[i % len(general_samples)]
            # Add some variation without "Sample X:" prefix
            if i >= len(general_samples):
                variation_suffix = f" This represents fundamental concepts in the field."
                samples.append(base_sample + variation_suffix)
            else:
                samples.append(base_sample)
                
        logger.info(f"Using fallback general data samples ({len(general_samples)} unique samples)")
        return samples
    
    def mix_domain_and_general_data(self, domain_chunks: List[str]) -> List[str]:
        """Mix domain data with general data according to mixture ratio"""
        num_domain = len(domain_chunks)
        num_general = int(num_domain * (1 - self.config.data_mixture_ratio) / self.config.data_mixture_ratio)
        
        logger.info(f"Creating data mixture: {num_domain} domain samples + {num_general} general samples")
        logger.info(f"Mixture ratio: {self.config.data_mixture_ratio:.1%} domain, {1-self.config.data_mixture_ratio:.1%} general")
        
        general_chunks = self.create_general_data_samples(num_general)
        
        # Combine and shuffle
        all_chunks = domain_chunks + general_chunks
        np.random.seed(self.config.seed)
        np.random.shuffle(all_chunks)
        
        return all_chunks


class PretrainingDataProcessor:
    """Process documents into MLX-LM training format with data mixture"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.doc_processor = DocumentProcessor(config)
        self.mixture_processor = DataMixtureProcessor(config)
        
    def create_training_data(self) -> Tuple[int, int, int]:
        """
        Process all documents and create train.jsonl and valid.jsonl files
        Returns: (num_train_examples, num_valid_examples, total_dataset_tokens)
        """
        # Ensure data directory exists
        data_path = Path(self.config.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Collect and process documents
        documents = self.doc_processor.collect_documents()
        
        if not documents:
            raise ValueError("No documents found to process")
            
        domain_text_chunks = []
        total_tokens = 0
        
        logger.info("Processing domain documents...")
        for doc_path in documents:
            text = self.doc_processor.extract_text_from_file(doc_path)
            if text:
                # Preserve raw text with minimal processing
                chunks = self.doc_processor.chunk_text(text, self.config.max_seq_length)
                domain_text_chunks.extend(chunks)
                total_tokens += sum(len(chunk.split()) for chunk in chunks)
                
        logger.info(f"Processed {len(domain_text_chunks)} domain text chunks with ~{total_tokens:,} tokens")
        
        # Apply data mixture strategy
        all_text_chunks = self.mixture_processor.mix_domain_and_general_data(domain_text_chunks)
        
        # Shuffle and split
        np.random.seed(self.config.seed)
        np.random.shuffle(all_text_chunks)
        
        split_idx = int(len(all_text_chunks) * (1 - self.config.validation_split))
        train_chunks = all_text_chunks[:split_idx]
        valid_chunks = all_text_chunks[split_idx:]
        
        # Save training data
        train_file = data_path / "train.jsonl"
        valid_file = data_path / "valid.jsonl"
        
        self._save_jsonl(train_chunks, train_file)
        self._save_jsonl(valid_chunks, valid_file)
        
        logger.info(f"Created {len(train_chunks)} training and {len(valid_chunks)} validation examples")
        logger.info(f"Data mixture applied: {self.config.data_mixture_ratio:.1%} domain + {1-self.config.data_mixture_ratio:.1%} general")
        
        # Compute total token count for full dataset (train + valid)
        total_tokens_dataset = sum(len(chunk.split()) for chunk in all_text_chunks)

        return len(train_chunks), len(valid_chunks), total_tokens_dataset
    
    def _save_jsonl(self, text_chunks: List[str], output_file: Path):
        """Save text chunks as JSONL file for MLX-LM"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in text_chunks:
                # Add explicit EOS token at the end of each chunk for better document boundary learning
                chunk_with_eos = chunk + " </s>"
                # Use simple text format for continued pre-training
                json_obj = {"text": chunk_with_eos}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n') 