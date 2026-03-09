"""Dataset for transformer training on VQVAE token sequences."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union


class TokenSequenceDataset(Dataset):
    """
    Dataset for loading VQVAE token sequences.
    
    Each sample is a 64-token sequence representing an 8×8 grid
    of discrete VQVAE codebook indices.
    """
    
    def __init__(self, data_path: Union[str, Path], split: str = 'train'):
        """
        Args:
            data_path: Path to token_sequences.pt file
            split: 'train' or 'val'
        """
        self.data_path = Path(data_path)
        self.split = split
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Token dataset not found: {self.data_path}")
        
        print(f"Loading token dataset from {self.data_path}...")
        data = torch.load(self.data_path, weights_only=False)
        
        if split == 'train':
            self.tokens = data['train_tokens']
        elif split == 'val':
            self.tokens = data['val_tokens']
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        self.config = data['config']
        
        print(f"Loaded {split} split: {len(self.tokens):,} sequences")
        print(f"  Vocab size: {self.config['vocab_size']}")
        print(f"  Sequence length: {self.config['sequence_length']}")
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        """
        Returns:
            tokens: (64,) token sequence
        """
        return self.tokens[idx]
    
    def get_vocab_size(self):
        return self.config['vocab_size']
    
    def get_sequence_length(self):
        return self.config['sequence_length']
