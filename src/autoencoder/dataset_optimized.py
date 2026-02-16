"""
Optimized dataset for 9-channel natural world format.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, List, Tuple
import lihzahrd

from ..terraria.chunk_processor_optimized import extract_optimized_chunk
from ..terraria.world_handler import load_world
from ..terraria.sampling_strategies import (
    DiversitySampler,
    ChunkStats,
    analyze_chunk,
    deduplicate_chunks
)


class PreprocessedTileDataset(Dataset):
    """
    Dataset that loads pre-extracted chunks from a .pt file.
    Efficient for multi-world training.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        
        print(f"Loading preprocessed dataset from {self.data_path}...")
        data = torch.load(self.data_path)
        
        self.chunks = data['chunks']  # (N, 9, H, W)
        self.stats = data['stats']
        self.config = data['config']
        
        print(f"Loaded {len(self.chunks)} chunks from {len(data['source_files'])} worlds.")
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]


class OptimizedTerrariaTileDataset(Dataset):
    """
    PyTorch Dataset for Terraria tiles in optimized 9-channel format.
    """
    
    def __init__(
        self,
        world_path: Union[str, Path],
        region: Tuple[int, int, int, int],
        chunk_size: int = 32,
        overlap: int = 8,
        use_diversity_filter: bool = True,
        min_diversity: float = 0.20,
        deduplicate: bool = True
    ):
        """
        Args:
            world_path: Path to .wld file
            region: (x_start, y_start, x_end, y_end) region to extract from
            chunk_size: Size of square chunks
            overlap: Overlap between adjacent chunks
            use_diversity_filter: Whether to filter out boring chunks
            min_diversity: Minimum diversity score (0-1)
            deduplicate: Remove near-duplicate chunks
        """
        self.world_path = Path(world_path)
        self.region = region
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_diversity_filter = use_diversity_filter
        self.min_diversity = min_diversity
        self.deduplicate = deduplicate
        
        # Load world
        self.world = load_world(self.world_path)
        
        # Generate chunks
        self.chunks, self.chunk_stats = self._generate_chunks()
        
        print(f"\nOptimized dataset created:")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Chunk size: {self.chunk_size}x{self.chunk_size}")
        print(f"  Channels: 9 (optimized)")
        self._print_statistics()
    
    def _generate_chunks(self) -> Tuple[List[torch.Tensor], List[ChunkStats]]:
        """Generate all chunks from the world region."""
        x_start, y_start, x_end, y_end = self.region
        step = self.chunk_size - self.overlap
        
        chunks = []
        stats_list = []
        
        # Initialize sampler if using diversity filter
        sampler = DiversitySampler(
            min_diversity=self.min_diversity,
            max_rejection_rate=0.5,
            adaptive=True
        ) if self.use_diversity_filter else None
        
        total_generated = 0
        total_possible = ((y_end - y_start) // step) * ((x_end - x_start) // step)
        print(f"Processing {total_possible} potential chunks from region...")
        
        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                chunk = extract_optimized_chunk(self.world, x, y, self.chunk_size, self.chunk_size)
                # Convert from (H, W, C) to (C, H, W) for PyTorch
                tensor = torch.from_numpy(chunk).permute(2, 0, 1).float()
                
                total_generated += 1
                if total_generated % 100 == 0:
                    acceptance_rate = len(chunks) / total_generated if total_generated > 0 else 0
                    print(f"  Progress: {total_generated}/{total_possible} chunks processed, {len(chunks)} accepted ({acceptance_rate:.1%})")
                
                # Apply diversity filter
                if sampler:
                    should_accept, stats = sampler.should_accept(tensor)
                    if not should_accept:
                        continue
                else:
                    stats = analyze_chunk(tensor)
                
                chunks.append(tensor)
                stats_list.append(stats)
        
        # Deduplication
        if self.deduplicate and len(chunks) > 0:
            print(f"  Deduplicating {len(chunks)} chunks...")
            dedup_result = deduplicate_chunks(chunks, stats_list)
            if isinstance(dedup_result, tuple):
                chunks, stats_list = dedup_result
            else:
                chunks = dedup_result
            print(f"  After deduplication: {len(chunks)} unique chunks")
        
        return chunks, stats_list
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.chunk_stats:
            return
        
        # Collect statistics
        diversities = [s.diversity_score for s in self.chunk_stats]
        
        # Count block types across all chunks
        from collections import Counter
        block_counter = Counter()
        for chunk in self.chunks:
            block_types = chunk[0, :, :].flatten().long().numpy()  # Channel 0 = block_type
            block_counter.update(block_types)
        
        print(f"\n  Diversity scores:")
        print(f"    Min: {min(diversities):.3f}")
        print(f"    Max: {max(diversities):.3f}")
        print(f"    Mean: {sum(diversities) / len(diversities):.3f}")
        
        print(f"\n  Top 10 most common block indices:")
        for block_idx, count in block_counter.most_common(10):
            percentage = count / (len(self.chunks) * self.chunk_size * self.chunk_size) * 100
            print(f"    Block index {int(block_idx)}: {count} tiles ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            (9, H, W) tensor
        """
        return self.chunks[idx]
