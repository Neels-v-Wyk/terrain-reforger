import torch
from torch.utils.data import Sampler
import random
from typing import Iterator, List, Dict

class InterleavedFileSampler(Sampler[int]):
    """
    A Sampler that shuffles data in a cache-aware manner to avoid thrashing.
    
    It groups files into 'active sets' of size <= max_cache_size.
    It serves all indices from the current active set (randomly shuffled) 
    before moving to the next set of files.
    
    This ensures that at any point, we only need 'max_cache_size' files open,
    maximizing cache hits while maintaining local randomness (mixing chunks from 
    different files within the active set).
    """
    def __init__(self, dataset, max_cache_size: int = 5, shuffle_files: bool = True):
        """
        Args:
            dataset: The dataset (must have .global_index attribute: List[Tuple[file_idx, local_idx]])
            max_cache_size: Number of files to keep 'hot' at once. 
                            Should match the dataset's cache size.
            shuffle_files: Whether to shuffle the order of file groups every epoch.
        """
        self.dataset = dataset
        self.max_cache_size = max_cache_size
        self.shuffle_files = shuffle_files
        
        # Build map: file_idx -> list of global indices
        # We need to know that global_idx 'i' corresponds to file 'X'
        if not hasattr(dataset, 'global_index'):
             raise ValueError("Dataset must have 'global_index' attribute to use InterleavedFileSampler")

        self.file_to_indices: Dict[int, List[int]] = {}
        self._build_index_map()

    def _build_index_map(self):
        # Scan the global_inidex to bucket indices by file
        for global_idx, (file_idx, _) in enumerate(self.dataset.global_index):
            if file_idx not in self.file_to_indices:
                self.file_to_indices[file_idx] = []
            self.file_to_indices[file_idx].append(global_idx)
            
        self.file_indices = list(self.file_to_indices.keys())

    def __iter__(self) -> Iterator[int]:
        # 1. Shuffle order of files (optional, but recommended for randomness across epochs)
        # We shuffle the *list of files*, not the contents yet
        files = self.file_indices.copy()
        if self.shuffle_files:
            random.shuffle(files)
            
        # 2. Chunk the files into groups of size max_cache_size
        # These are the files that will be 'hot' in the cache simultaneously.
        for i in range(0, len(files), self.max_cache_size):
            active_files = files[i : i + self.max_cache_size]
            
            # 3. Collect all global indices for these active files
            active_indices = []
            for f_idx in active_files:
                active_indices.extend(self.file_to_indices[f_idx])
                
            # 4. Shuffle these indices (Local Randomness)
            # This mixes chunks from different files within the active set.
            # This is key: we get batches mixed from File A, B, C, D, E.
            random.shuffle(active_indices)
            
            # 5. Yield them
            yield from active_indices

    def __len__(self):
        return len(self.dataset)
