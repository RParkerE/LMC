import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from typing import Dict, List, Tuple
import glob

class ClinicalDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', window_size: int = 10):
        """
        Initialize the clinical dataset.
        
        Args:
            data_path: Path to the processed data directory
            split: 'train', 'val', or 'test'
            window_size: Size of context window
        """
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        
        # Load vocabulary and metadata mappings
        with open(os.path.join(data_path, 'vocab.json'), 'r') as f:
            self.vocab = json.load(f)
        with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Load data
        self.data = self._load_data()
        
        # Set sizes
        self.vocab_size = len(self.vocab)
        self.metadata_size = len(self.metadata)
    
    def _load_data(self) -> List[Dict]:
        """Load the data for the specified split."""
        # First try CASI data
        casi_file = os.path.join(self.data_path, 'casi', f'{self.split}.json')
        if os.path.exists(casi_file):
            with open(casi_file, 'r') as f:
                return json.load(f)
        
        # If CASI data not found, try MIMIC-III chunks
        mimic_dir = os.path.join(self.data_path, 'sections')
        if os.path.exists(mimic_dir):
            chunk_files = sorted(glob.glob(os.path.join(mimic_dir, f'processed_notes_chunk_*.json')))
            if chunk_files:
                # For validation and test, use a subset of chunks
                if self.split == 'val':
                    chunk_files = chunk_files[:len(chunk_files)//10]  # Use 10% of chunks for validation
                elif self.split == 'test':
                    chunk_files = chunk_files[len(chunk_files)//10:len(chunk_files)//5]  # Use next 10% for test
                else:  # train
                    chunk_files = chunk_files[len(chunk_files)//5:]  # Use remaining 80% for training
                
                # Load data from chunks
                data = []
                for chunk_file in chunk_files:
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        data.extend(chunk_data)
                return data
        
        raise FileNotFoundError(f"No data found for split '{self.split}' in {self.data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training example.
        
        Returns:
            Dictionary containing:
                - word_idx: Index of the center word
                - metadata_idx: Index of the metadata
                - context_words: Indices of context words
                - target_words: Indices of target words
        """
        example = self.data[idx]
        
        # Convert words to indices
        word_idx = self.vocab.get(example['acronym'], self.vocab['<UNK>'])
        metadata_idx = self.metadata.get(example['section_header'], self.metadata['<UNK>'])
        
        # Get context words
        context_words = []
        for word in example['context']:
            context_words.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        # Pad or truncate context
        if len(context_words) < self.window_size * 2:
            context_words = context_words + [self.vocab['<PAD>']] * (self.window_size * 2 - len(context_words))
        else:
            context_words = context_words[:self.window_size * 2]
        
        # Get target words (same as context for reconstruction)
        target_words = context_words.copy()
        
        return {
            'word_idx': torch.tensor(word_idx, dtype=torch.long),
            'metadata_idx': torch.tensor(metadata_idx, dtype=torch.long),
            'context_words': torch.tensor(context_words, dtype=torch.long),
            'target_words': torch.tensor(target_words, dtype=torch.long)
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of dictionaries from __getitem__
    
    Returns:
        Dictionary of batched tensors
    """
    return {
        'word_idx': torch.stack([item['word_idx'] for item in batch]),
        'metadata_idx': torch.stack([item['metadata_idx'] for item in batch]),
        'context_words': torch.stack([item['context_words'] for item in batch]),
        'target_words': torch.stack([item['target_words'] for item in batch])
    }

def create_vocab_and_metadata(data_path: str, min_freq: int = 5) -> Tuple[Dict, Dict]:
    """
    Create vocabulary and metadata mappings from raw data.
    
    Args:
        data_path: Path to raw data directory
        min_freq: Minimum frequency for a word to be included in vocabulary
    
    Returns:
        Tuple of (vocab_dict, metadata_dict)
    """
    # Initialize counters
    word_counts = {}
    metadata_set = set()
    
    # Process all files
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            with open(os.path.join(data_path, filename), 'r') as f:
                data = json.load(f)
                
                # Count words
                for example in data:
                    word = example['word']
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
                    # Collect metadata
                    metadata_set.add(example['metadata'])
    
    # Create vocabulary
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3
    }
    
    # Add words that meet frequency threshold
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    # Create metadata mapping
    metadata = {'<UNK>': 0}
    for meta in sorted(metadata_set):
        metadata[meta] = len(metadata)
    
    return vocab, metadata

def save_vocab_and_metadata(vocab: Dict, metadata: Dict, output_path: str):
    """Save vocabulary and metadata mappings to files."""
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)
    
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2) 