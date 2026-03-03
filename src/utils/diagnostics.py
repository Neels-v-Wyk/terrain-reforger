"""Training diagnostics and analysis utilities."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, any]:
    """
    Compute per-class accuracy metrics.
    
    Args:
        logits: (B, num_classes, H, W) prediction logits
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes
        
    Returns:
        Dictionary with overall accuracy and per-class accuracies
    """
    predictions = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # Overall accuracy
    correct = (predictions == targets).float()
    overall_acc = correct.mean().item()
    
    # Per-class accuracy
    per_class_acc = {}
    per_class_support = {}
    
    for class_idx in range(num_classes):
        mask = (targets == class_idx)
        if mask.sum() > 0:
            class_correct = correct[mask].mean().item()
            per_class_acc[class_idx] = class_correct
            per_class_support[class_idx] = mask.sum().item()
        else:
            per_class_acc[class_idx] = 0.0
            per_class_support[class_idx] = 0
    
    return {
        "overall": overall_acc,
        "per_class": per_class_acc,
        "support": per_class_support,
    }


def compute_class_distribution(
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[int, any]:
    """
    Compute class frequency distribution.
    
    Args:
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class_idx -> count
    """
    counts = torch.bincount(targets.reshape(-1), minlength=num_classes)
    return {i: counts[i].item() for i in range(num_classes)}


def get_top_confused_pairs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    top_k: int = 10,
) -> List[Tuple[int, int, int]]:
    """
    Find the most frequently confused class pairs.
    
    Args:
        logits: (B, num_classes, H, W) prediction logits
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes
        top_k: Number of top confused pairs to return
        
    Returns:
        List of (true_class, predicted_class, count) tuples
    """
    predictions = torch.argmax(logits, dim=1).reshape(-1)
    targets = targets.reshape(-1)
    
    # Count misclassifications
    confusion = defaultdict(int)
    mask = predictions != targets
    
    for true_cls, pred_cls in zip(targets[mask].tolist(), predictions[mask].tolist()):
        if true_cls != pred_cls:
            confusion[(true_cls, pred_cls)] += 1
    
    # Sort by count
    sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    
    return [(true_cls, pred_cls, count) for (true_cls, pred_cls), count in sorted_confusion[:top_k]]


def analyze_codebook_health(
    codebook_indices: torch.Tensor,
    num_embeddings: int,
) -> Dict[str, float]:
    """
    Analyze codebook utilization and health.
    
    Args:
        codebook_indices: (B, H, W) or (B*H*W,) indices of used codes
        num_embeddings: Total number of codebook entries
        
    Returns:
        Dictionary with utilization statistics
    """
    indices_flat = codebook_indices.reshape(-1)
    counts = torch.bincount(indices_flat, minlength=num_embeddings).float()
    
    # Basic stats
    total_usage = indices_flat.numel()
    active_codes = (counts > 0).sum().item()
    dead_codes = num_embeddings - active_codes
    
    # Usage distribution
    usage_freq = counts / total_usage
    
    # Entropy (normalized)
    eps = 1e-10
    entropy = -(usage_freq * torch.log(usage_freq + eps)).sum().item()
    max_entropy = np.log(num_embeddings)
    entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Top code concentration
    sorted_counts, _ = torch.sort(counts, descending=True)
    top1_share = sorted_counts[0].item() / total_usage if total_usage > 0 else 0.0
    top5_share = sorted_counts[:5].sum().item() / total_usage if total_usage > 0 else 0.0
    top10_share = sorted_counts[:10].sum().item() / total_usage if total_usage > 0 else 0.0
    
    # Active codes above uniform distribution
    uniform_usage = 1.0 / num_embeddings
    above_uniform = (usage_freq > uniform_usage).sum().item()
    above_half_uniform = (usage_freq > 0.5 * uniform_usage).sum().item()
    
    return {
        "active_codes": active_codes,
        "dead_codes": dead_codes,
        "usage_percent": (active_codes / num_embeddings) * 100,
        "entropy_normalized": entropy_normalized,
        "top1_share": top1_share,
        "top5_share": top5_share,
        "top10_share": top10_share,
        "active_above_uniform": above_uniform,
        "active_above_half_uniform": above_half_uniform,
    }


def get_diagnostic_summary(
    results: Dict[str, List],
    model_config: Dict,
    recent_window: int = 100,
) -> str:
    """
    Generate a human-readable diagnostic summary.
    
    Args:
        results: Training results dictionary
        model_config: Model configuration
        recent_window: Number of recent updates to analyze
        
    Returns:
        Formatted diagnostic string
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("DIAGNOSTIC SUMMARY")
    lines.append("=" * 80)
    
    # Loss trends
    if "loss_vals" in results and len(results["loss_vals"]) > 0:
        recent_losses = results["loss_vals"][-recent_window:]
        if len(recent_losses) > 10:
            first_half = np.mean(recent_losses[:len(recent_losses)//2])
            second_half = np.mean(recent_losses[len(recent_losses)//2:])
            improvement = first_half - second_half
            
            lines.append(f"\nLoss Trend (last {len(recent_losses)} updates):")
            lines.append(f"  First half avg: {first_half:.4f}")
            lines.append(f"  Second half avg: {second_half:.4f}")
            lines.append(f"  Improvement: {improvement:.4f}")
            
            if improvement < 0.01:
                lines.append("  ⚠️  PLATEAU DETECTED - Loss not improving significantly")
    
    # Component loss analysis
    component_keys = ["block_loss", "wall_loss", "liquid_loss", "shape_loss", "continuous_loss"]
    lines.append("\nComponent Loss Breakdown:")
    for key in component_keys:
        if key in results and len(results[key]) > 0:
            recent = results[key][-recent_window:]
            avg = np.mean(recent)
            std = np.std(recent)
            lines.append(f"  {key:20s}: {avg:.4f} ± {std:.4f}")
    
    # Codebook health
    if "perplexities" in results and len(results["perplexities"]) > 0:
        recent_perp = results["perplexities"][-recent_window:]
        avg_perp = np.mean(recent_perp)
        utilization_pct = (avg_perp / model_config.get("n_embeddings", 512)) * 100
        
        lines.append(f"\nCodebook Health:")
        lines.append(f"  Average perplexity: {avg_perp:.2f}")
        lines.append(f"  Utilization: {utilization_pct:.1f}%")
        
        if utilization_pct < 50:
            lines.append("  ⚠️  LOW UTILIZATION - Consider reducing codebook size or adjusting EMA settings")
        elif utilization_pct > 90:
            lines.append("  ✓ Good codebook utilization")
    
    # Suggestions
    lines.append("\nSuggestions:")
    
    # Check for plateau
    if "block_loss" in results and len(results["block_loss"]) > 50:
        recent_block = results["block_loss"][-50:]
        if np.std(recent_block) < 0.01:
            lines.append("  🔧 Block loss plateauing → Try --block-loss-weighted if not already enabled")
    
    # Check perplexity
    if "perplexities" in results and len(results["perplexities"]) > 0:
        recent_perp = results["perplexities"][-recent_window:]
        avg_perp = np.mean(recent_perp)
        if avg_perp < model_config.get("n_embeddings", 512) * 0.3:
            lines.append("  🔧 Low codebook usage → Increase --ema-reset-threshold or reduce --n-embeddings")
    
    lines.append("=" * 80 + "\n")
    
    return "\n".join(lines)
