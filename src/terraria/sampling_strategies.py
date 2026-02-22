"""
Intelligent sampling strategies for Terraria world chunks.

Handles data imbalance by rejecting overly-common chunks (e.g., pure stone)
and preferring diverse, interesting chunks for training.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from collections import Counter
from dataclasses import dataclass
import torch


@dataclass
class ChunkStats:
    """Statistics about a chunk's content diversity."""
    unique_block_types: int
    unique_wall_types: int
    block_entropy: float
    wall_entropy: float
    has_liquid: bool
    has_wiring: bool
    block_coverage: float  # % of tiles with active blocks
    diversity_score: float  # Overall score
    
    def __repr__(self):
        return (f"ChunkStats(blocks={self.unique_block_types}, "
                f"walls={self.unique_wall_types}, "
                f"entropy={self.block_entropy:.2f}, "
                f"diversity={self.diversity_score:.2f})")


def calculate_entropy(values: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a distribution.
    
    Higher entropy = more diverse/interesting
    Lower entropy = more uniform/boring
    
    Args:
        values: Array of categorical values
        
    Returns:
        Entropy value (0 to log2(num_unique_values))
    """
    # Count frequencies
    counts = np.bincount(values.astype(int).flatten())
    counts = counts[counts > 0]  # Remove zeros
    
    # Calculate probabilities
    total = np.sum(counts)
    probs = counts / total
    
    # Calculate entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return float(entropy)


def analyze_chunk(chunk_tensor: torch.Tensor) -> ChunkStats:
    """
    Analyze a chunk and calculate diversity statistics.

    Expects the 8-channel format (C, H, W) or (H, W, C) where C=8:
        0: block_type index  1: block_shape  2: wall_type index
        3: liquid_type       4: wire_red     5: wire_blue
        6: wire_green        7: actuator

    Args:
        chunk_tensor: Tensor of shape (8, H, W) or (H, W, 8)

    Returns:
        ChunkStats with diversity metrics
    """
    # Convert to numpy
    if isinstance(chunk_tensor, torch.Tensor):
        chunk = chunk_tensor.cpu().numpy()
    else:
        chunk = chunk_tensor

    # Ensure (H, W, C) format
    if chunk.shape[0] == 8:
        chunk = np.transpose(chunk, (1, 2, 0))

    # Extract channels (8-channel format)
    block_types = chunk[:, :, 0].astype(int)
    # Block is active when its index is non-zero (0 = Air in natural_ids)
    block_active = (block_types > 0).astype(np.float32)
    wall_types = chunk[:, :, 2].astype(int)
    liquid_type = chunk[:, :, 3].astype(int)
    wiring = chunk[:, :, 4:8]

    # Calculate statistics
    unique_blocks = len(np.unique(block_types[block_active > 0.5]))
    unique_walls = len(np.unique(wall_types[wall_types > 0]))
    
    block_entropy = calculate_entropy(block_types[block_active > 0.5]) if unique_blocks > 1 else 0.0
    wall_entropy = calculate_entropy(wall_types[wall_types > 0]) if unique_walls > 1 else 0.0

    has_liquid = bool(np.any(liquid_type > 0))
    has_wiring = bool(np.any(wiring > 0.5))
    
    block_coverage = float(np.mean(block_active > 0.5))
    
    # Calculate diversity score (0-1, higher = more interesting)
    # Factors:
    # - Block type diversity (entropy normalized by max possible)
    # - Wall type diversity
    # - Presence of special features (liquids, wiring)
    # - Variety (not 100% filled or 100% empty)
    
    # Avoid log2(0) or log2(1) which cause warnings
    max_block_entropy = np.log2(max(min(unique_blocks, 10), 2))  # Minimum 2 for log2
    max_wall_entropy = np.log2(max(min(unique_walls, 10), 2))
    
    diversity_score = 0.0
    
    # Block diversity (0-0.4 points)
    if max_block_entropy > 0:
        diversity_score += 0.4 * (block_entropy / max_block_entropy)
    
    # Wall diversity (0-0.2 points)
    if max_wall_entropy > 0:
        diversity_score += 0.2 * (wall_entropy / max_wall_entropy)
    
    # Unique types bonus (0-0.2 points)
    diversity_score += 0.1 * min(unique_blocks / 10.0, 1.0)
    diversity_score += 0.1 * min(unique_walls / 10.0, 1.0)
    
    # Special features (0-0.1 points)
    if has_liquid:
        diversity_score += 0.05
    if has_wiring:
        diversity_score += 0.05
    
    # Avoid pure filled/empty (0-0.1 points)
    # Prefer 30-90% coverage
    if 0.3 <= block_coverage <= 0.9:
        diversity_score += 0.1
    elif block_coverage < 0.1 or block_coverage > 0.95:
        diversity_score -= 0.1
    
    return ChunkStats(
        unique_block_types=unique_blocks,
        unique_wall_types=unique_walls,
        block_entropy=block_entropy,
        wall_entropy=wall_entropy,
        has_liquid=has_liquid,
        has_wiring=has_wiring,
        block_coverage=block_coverage,
        diversity_score=max(0.0, min(1.0, diversity_score))
    )


class DiversitySampler:
    """
    Samples chunks with preference for diverse/interesting content.
    
    Uses rejection sampling to avoid over-representation of common chunks
    (like pure stone underground areas).
    """
    
    def __init__(self, 
                 min_diversity: float = 0.15,
                 max_rejection_rate: float = 0.5,
                 adaptive: bool = True):
        """
        Args:
            min_diversity: Minimum diversity score to accept (0-1)
            max_rejection_rate: Maximum fraction of chunks to reject
            adaptive: Adjust thresholds based on available data
        """
        self.min_diversity = min_diversity
        self.max_rejection_rate = max_rejection_rate
        self.adaptive = adaptive
        
        # Track statistics
        self.total_seen = 0
        self.total_accepted = 0
        self.diversity_scores = []
    
    def should_accept(self, chunk: torch.Tensor) -> Tuple[bool, ChunkStats]:
        """
        Decide whether to accept a chunk for training.
        
        Args:
            chunk: Chunk tensor to evaluate
            
        Returns:
            (should_accept, chunk_stats)
        """
        stats = analyze_chunk(chunk)
        
        self.total_seen += 1
        self.diversity_scores.append(stats.diversity_score)
        
        # Adaptive threshold adjustment
        if self.adaptive and len(self.diversity_scores) > 100:
            # If we're rejecting too much, lower the threshold
            current_rejection_rate = 1 - (self.total_accepted / self.total_seen)
            
            if current_rejection_rate > self.max_rejection_rate:
                # Lower threshold (become less picky)
                percentile = (1 - self.max_rejection_rate) * 100
                self.min_diversity = np.percentile(self.diversity_scores[-100:], percentile)
            elif current_rejection_rate < 0.05:  # Rejecting less than 5%
                # Raise threshold (become more picky)
                # If we're accepting almost everything, try to reject at least the bottom 10%
                # This helps filter out empty chunks or very repetitive terrain
                self.min_diversity = max(self.min_diversity, np.percentile(self.diversity_scores[-100:], 10))
        
        # Accept if diversity is above threshold
        accept = stats.diversity_score >= self.min_diversity
        
        if accept:
            self.total_accepted += 1
        
        return accept, stats
    
    def get_statistics(self) -> dict:
        """Get sampling statistics."""
        if self.total_seen == 0:
            return {}
        
        return {
            'total_seen': self.total_seen,
            'total_accepted': self.total_accepted,
            'rejection_rate': 1 - (self.total_accepted / self.total_seen),
            'current_threshold': self.min_diversity,
            'avg_diversity': np.mean(self.diversity_scores) if self.diversity_scores else 0,
            'diversity_std': np.std(self.diversity_scores) if self.diversity_scores else 0,
        }


class BalancedBatchSampler:
    """
    Creates batches with balanced representation of different chunk types.
    
    Groups chunks by their primary block type and ensures each batch
    has diverse content rather than all being similar.
    """
    
    def __init__(self, batch_size: int = 32, diversity_bins: int = 5):
        """
        Args:
            batch_size: Target batch size
            diversity_bins: Number of diversity levels to balance across
        """
        self.batch_size = batch_size
        self.diversity_bins = diversity_bins
        
        # Store chunks by diversity bin
        self.bins = [[] for _ in range(diversity_bins)]
        self.bin_stats = [[] for _ in range(diversity_bins)]
    
    def add_chunk(self, chunk: torch.Tensor, stats: Optional[ChunkStats] = None):
        """Add a chunk to the appropriate bin."""
        if stats is None:
            stats = analyze_chunk(chunk)
        
        # Determine bin (0 = boring, 4 = very interesting)
        bin_idx = min(int(stats.diversity_score * self.diversity_bins), 
                     self.diversity_bins - 1)
        
        self.bins[bin_idx].append(chunk)
        self.bin_stats[bin_idx].append(stats)
    
    def get_balanced_batch(self) -> Optional[List[torch.Tensor]]:
        """
        Sample a balanced batch from all bins.
        
        Returns:
            List of chunks, or None if not enough chunks available
        """
        # Calculate how many from each bin
        non_empty_bins = [i for i, bin in enumerate(self.bins) if len(bin) > 0]
        
        if not non_empty_bins:
            return None
        
        chunks_per_bin = self.batch_size // len(non_empty_bins)
        remainder = self.batch_size % len(non_empty_bins)
        
        batch = []
        
        for bin_idx in non_empty_bins:
            n_samples = chunks_per_bin + (1 if remainder > 0 else 0)
            remainder -= 1
            
            # Sample from this bin
            bin_size = len(self.bins[bin_idx])
            if bin_size < n_samples:
                # Not enough, take all
                batch.extend(self.bins[bin_idx])
                self.bins[bin_idx] = []
            else:
                # Sample without replacement
                indices = np.random.choice(bin_size, n_samples, replace=False)
                sampled = [self.bins[bin_idx][i] for i in indices]
                batch.extend(sampled)
                
                # Remove sampled chunks
                remaining = [chunk for i, chunk in enumerate(self.bins[bin_idx]) 
                           if i not in indices]
                self.bins[bin_idx] = remaining
        
        return batch if len(batch) >= self.batch_size // 2 else None
    
    def get_statistics(self) -> dict:
        """Get bin statistics."""
        return {
            'bin_sizes': [len(bin) for bin in self.bins],
            'total_chunks': sum(len(bin) for bin in self.bins),
            'bins_stats': [
                {
                    'count': len(self.bin_stats[i]),
                    'avg_diversity': np.mean([s.diversity_score for s in self.bin_stats[i]]) 
                                    if self.bin_stats[i] else 0
                }
                for i in range(self.diversity_bins)
            ]
        }


def deduplicate_chunks(chunks: List[torch.Tensor], 
                       stats_list: Optional[List[ChunkStats]] = None,
                       similarity_threshold: float = 0.95) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[ChunkStats]]]:
    """
    Remove near-duplicate chunks using simple hash comparison.
    
    Args:
        chunks: List of chunk tensors
        stats_list: Optional list of chunk statistics to filter in parallel
        similarity_threshold: Similarity threshold for deduplication
        
    Returns:
        Deduplicated list of chunks, or (chunks, stats) if stats_list provided
    """
    # Handle the case where stats_list is passed as the second positional argument but the caller 
    # intended it to be similarity_threshold (backward compatibility or cross-file confusion)
    # However, in this case, the caller (dataset.py) passed stats_list as 2nd arg
    # intentionally expecting it to be handled, while dataset.py passes similarity_threshold as kwarg.
    
    # If the second argument is a float (and not a list), treat it as similarity_threshold
    if isinstance(stats_list, (float, int)) and not isinstance(stats_list, list):
        similarity_threshold = float(stats_list)
        stats_list = None

    if not chunks:
        if stats_list is not None:
             return [], []
        return []
    
    # Use block type histogram as fingerprint
    fingerprints = []
    unique_chunks = []
    unique_stats = []
    
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, torch.Tensor):
            chunk_np = chunk.cpu().numpy()
        else:
            chunk_np = chunk
        
        # Get block types
        if chunk_np.shape[0] == 17:
            block_types = chunk_np[0].flatten()
        else:
            block_types = chunk_np[:, :, 0].flatten()
        
        # Create histogram fingerprint
        hist, _ = np.histogram(block_types, bins=50, range=(0, 700))
        fingerprint = hist / (hist.sum() + 1e-10)
        
        # Check similarity to existing chunks
        is_duplicate = False
        for existing_fp in fingerprints:
            similarity = 1 - np.sum(np.abs(fingerprint - existing_fp)) / 2
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            fingerprints.append(fingerprint)
            unique_chunks.append(chunk)
            if stats_list is not None and i < len(stats_list):
                 unique_stats.append(stats_list[i])
    
    if stats_list is not None:
        return unique_chunks, unique_stats
    return unique_chunks
