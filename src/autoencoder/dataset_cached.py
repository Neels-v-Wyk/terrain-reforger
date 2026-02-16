import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import os
from collections import OrderedDict
import logging

# Configure logger
logger = logging.getLogger(__name__)

class CachedTileDataset(Dataset):
    """
    Dataset that loads chunks from multiple .pt files (one per world/batch).
    Supports:
    1. Pre-loading everything into RAM (if size permits).
    2. LRU Caching of file contents (if dataset > RAM).
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path], 
        preload: bool = True,
        max_cache_size: int = 5,
        verbose: bool = True
    ):
        """
        Args:
            data_dir: Directory containing .pt files.
            preload: If True, attempts to load all files into RAM at start.
                     If False, uses LRU cache.
            max_cache_size: Number of files to keep in memory (LRU) when preload=False.
            verbose: Print status messages.
        """
        self.data_dir = Path(data_dir)
        self.preload = preload
        self.max_cache_size = max_cache_size
        self.verbose = verbose
        
        # 1. Find all files
        self.file_paths = sorted(list(self.data_dir.glob("*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")
        
        if self.verbose:
            print(f"Found {len(self.file_paths)} data files in {self.data_dir}")

        # 2. Build Index (Global Index -> File Index, Local Index)
        # We need to know how many chunks are in each file.
        # This requires reading at least the metadata/header.
        self.global_index: List[Tuple[int, int]] = []  # [(file_idx, local_chunk_idx), ...]
        self.total_chunks = 0
        self.file_chunk_counts = []
        
        # Optimization: We can store the loaded data here if preloading
        self.ram_cache: Dict[int, torch.Tensor] = {} # file_idx -> Tensor(N, C, H, W)
        
        # LRU Cache state
        self.lru_cache: OrderedDict[int, torch.Tensor] = OrderedDict() 
        
        self._build_index()
        
    def _build_index(self):
        """
        Scan files to determine total length and build mapping.
        If preload=True, this also loads the data.
        """
        cumulative_count = 0
        
        for i, fpath in enumerate(self.file_paths):
            try:
                # We have to load to count, unless we trust a sidecar file. 
                # Assuming standard .pt loading for now.
                data = torch.load(fpath, map_location='cpu')
                
                # Handle different formats
                if isinstance(data, dict):
                    chunks = data.get('chunks')
                elif isinstance(data, torch.Tensor):
                    chunks = data
                else:
                    logger.warning(f"File {fpath} has unknown format. Skipping.")
                    continue
                
                if chunks is None:
                    continue

                if not isinstance(chunks, torch.Tensor):
                     # If it's a list, stack it (assuming it's small enough per file)
                     chunks = torch.stack(chunks)
                
                num_chunks = chunks.shape[0]
                self.file_chunk_counts.append(num_chunks)
                
                # Add to index
                # Optimization: usage of ranges instead of per-item list might be lighter for huge datasets
                # But for <1M items, a list of tuples is fine and O(1) access.
                for local_idx in range(num_chunks):
                    self.global_index.append((i, local_idx))
                
                if self.preload:
                    self.ram_cache[i] = chunks
                
                cumulative_count += num_chunks
                
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Index build progress: {i+1}/{len(self.file_paths)} files processed...")
                    
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
        
        self.total_chunks = cumulative_count
        if self.verbose:
             mode = "RAM (Preloaded)" if self.preload else f"Disk (LRU={self.max_cache_size})"
             print(f"Dataset ready. Total chunks: {self.total_chunks}. Mode: {mode}")

    def __len__(self):
        return self.total_chunks
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError("Index out of bounds")
        
        file_idx, local_idx = self.global_index[idx]
        
        chunk_tensor = self._get_file_chunks(file_idx)
        return chunk_tensor[local_idx]

    def _get_file_chunks(self, file_idx: int) -> torch.Tensor:
        """Retrieve the tensor containing all chunks for a file_idx."""
        
        # 1. RAM Mode
        if self.preload:
            return self.ram_cache[file_idx]
        
        # 2. LRU Check
        if file_idx in self.lru_cache:
            # Move to end (most recently used)
            self.lru_cache.move_to_end(file_idx)
            return self.lru_cache[file_idx]
        
        # 3. Load from Disk
        # Enforce LRU size
        if len(self.lru_cache) >= self.max_cache_size:
            self.lru_cache.popitem(last=False) # Remove FIFO (least recently used)
            
        fpath = self.file_paths[file_idx]
        data = torch.load(fpath, map_location='cpu')
        
        if isinstance(data, dict):
            chunks = data.get('chunks')
        else:
            chunks = data
            
        if not isinstance(chunks, torch.Tensor):
            chunks = torch.stack(chunks)
            
        self.lru_cache[file_idx] = chunks
        return chunks
