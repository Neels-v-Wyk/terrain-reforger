import torch
from torch.utils.data import Dataset
from terraria.chunk_processor import extract_chunk
from terraria.chunk_processor import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from terraria.sampling_strategies import (
    DiversitySampler, 
    analyze_chunk, 
    deduplicate_chunks,
    ChunkStats
)
from typing import Optional, List, Tuple


class TerrariaTileDataset(Dataset):
    """
    Dataset for Terraria world chunks with optional diversity filtering.
    
    Supports:
    - Rejection sampling to avoid overrepresented chunks (e.g., pure stone)
    - Deduplication of similar chunks
    - Statistics tracking for data quality
    """
    
    def __init__(self, 
                 world, 
                 region, 
                 chunk_size=DEFAULT_CHUNK_SIZE, 
                 overlap=DEFAULT_OVERLAP,
                 use_diversity_filter: bool = True,
                 min_diversity: float = 0.15,
                 deduplicate: bool = False,
                 verbose: bool = True):
        """
        Args:
            world: Loaded lihzahrd World object
            region: (x_start, y_start, x_end, y_end) tuple
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            use_diversity_filter: Filter out boring/repetitive chunks
            min_diversity: Minimum diversity score (0-1) to accept
            deduplicate: Remove near-duplicate chunks
            verbose: Print filtering statistics
        """
        self.world = world
        self.region = region
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_diversity_filter = use_diversity_filter
        self.min_diversity = min_diversity
        self.deduplicate = deduplicate
        self.verbose = verbose
        
        # Generate and filter chunks
        self.chunks, self.chunk_stats = self._generate_chunks()
        
        if verbose:
            self._print_statistics()
    
    def _generate_chunks(self) -> Tuple[List[torch.Tensor], List[ChunkStats]]:
        """Generate chunks with optional filtering."""
        x_start, y_start, x_end, y_end = self.region
        chunks = []
        stats_list = []
        step = self.chunk_size - self.overlap
        
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
                chunk = extract_chunk(self.world, x, y, self.chunk_size, self.chunk_size)
                # Convert from (H, W, C) to (C, H, W) for PyTorch
                tensor = torch.from_numpy(chunk.to_tensor()).permute(2, 0, 1).float()
                
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
            original_count = len(chunks)
            chunks = deduplicate_chunks(chunks, similarity_threshold=0.95)
            # Update stats list to match
            stats_list = stats_list[:len(chunks)]
            
            if self.verbose:
                print(f"Deduplication: {original_count} → {len(chunks)} chunks "
                      f"({original_count - len(chunks)} duplicates removed)")
        
        return chunks, stats_list
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.chunks:
            print("Warning: No chunks in dataset!")
            return
        
        import numpy as np
        
        diversity_scores = [s.diversity_score for s in self.chunk_stats]
        unique_blocks = [s.unique_block_types for s in self.chunk_stats]
        block_coverages = [s.block_coverage for s in self.chunk_stats]
        
        print("\n" + "="*80)
        print("Dataset Statistics")
        print("="*80)
        print(f"Total chunks: {len(self.chunks)}")
        print(f"Region: {self.region}")
        print(f"Chunk size: {self.chunk_size}x{self.chunk_size}")
        print(f"\nDiversity Scores:")
        print(f"  Mean: {np.mean(diversity_scores):.3f}")
        print(f"  Std:  {np.std(diversity_scores):.3f}")
        print(f"  Min:  {np.min(diversity_scores):.3f}")
        print(f"  Max:  {np.max(diversity_scores):.3f}")
        print(f"\nBlock Type Diversity:")
        print(f"  Mean unique types: {np.mean(unique_blocks):.1f}")
        print(f"  Max unique types:  {np.max(unique_blocks)}")
        print(f"\nBlock Coverage:")
        print(f"  Mean: {np.mean(block_coverages):.1%}")
        
        # Distribution by diversity
        print(f"\nDiversity Distribution:")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(diversity_scores, bins=bins)
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} chunks ({hist[i]/len(self.chunks)*100:5.1f}%)")
        
        # Find most common block types
        from collections import Counter
        all_block_types = []
        for chunk in self.chunks:
            block_types = chunk[0].numpy().flatten()
            block_active = chunk[1].numpy().flatten()
            active_blocks = block_types[block_active > 0.5]
            all_block_types.extend(active_blocks.astype(int).tolist())
        
        if all_block_types:
            print(f"\nTop 10 Most Common Block Types:")
            from terraria.tensor_utils import BLOCK_NAMES
            counter = Counter(all_block_types)
            for block_id, count in counter.most_common(10):
                pct = count / len(all_block_types) * 100
                name = BLOCK_NAMES.get(block_id, f"Unknown {block_id}")
                print(f"  {name:25s}: {count:6d} ({pct:5.1f}%)")
        
        print("="*80 + "\n")
    
    def get_chunk_stats(self, idx: int) -> ChunkStats:
        """Get statistics for a specific chunk."""
        return self.chunk_stats[idx]
    
    def filter_by_diversity(self, min_score: float) -> 'TerrariaTileDataset':
        """
        Create a new dataset with only high-diversity chunks.
        
        Args:
            min_score: Minimum diversity score to keep
            
        Returns:
            New filtered dataset
        """
        filtered_chunks = []
        filtered_stats = []
        
        for chunk, stats in zip(self.chunks, self.chunk_stats):
            if stats.diversity_score >= min_score:
                filtered_chunks.append(chunk)
                filtered_stats.append(stats)
        
        # Create new dataset object (shallow copy with new chunks)
        new_dataset = TerrariaTileDataset.__new__(TerrariaTileDataset)
        new_dataset.world = self.world
        new_dataset.region = self.region
        new_dataset.chunk_size = self.chunk_size
        new_dataset.overlap = self.overlap
        new_dataset.use_diversity_filter = self.use_diversity_filter
        new_dataset.min_diversity = min_score
        new_dataset.deduplicate = False
        new_dataset.verbose = False
        new_dataset.chunks = filtered_chunks
        new_dataset.chunk_stats = filtered_stats
        
        return new_dataset

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if idx >= len(self.chunks):
            raise IndexError("Index out of range")
        return self.chunks[idx]
