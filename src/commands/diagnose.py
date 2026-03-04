"""Diagnostic command for quick model validation."""

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Sequence
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.autoencoder.dataset_cached import CachedTileDataset
from src.autoencoder.dataset import PreprocessedTileDataset
from src.autoencoder.vqvae import VQVAE, compute_loss, NUM_NATURAL_BLOCKS
from src.utils.device import get_device
from src.utils.diagnostics import (
    compute_class_accuracy,
    compute_class_distribution,
    get_top_confused_pairs,
    analyze_codebook_health,
    get_diagnostic_summary,
)
from src.terraria.natural_ids import BLOCK_INDEX_TO_ID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnostics on VQ-VAE training")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed dataset")
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for diagnostics (default: 0.2)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint to load and evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/diagnostic.json",
        help="Output path for diagnostic report",
    )
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay rate")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA quantizer")
    parser.add_argument(
        "--block-loss-weighted",
        action="store_true",
        help="Use class-weighted block loss",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Run diagnostic evaluation."""
    device = get_device()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC MODE")
    print("=" * 80 + "\n")
    print(f"Device: {device}")
    
    # Load dataset
    data_path = Path(args.data)
    if data_path.is_dir():
        print(f"Loading dataset from directory: {data_path}")
        all_files = sorted(list(data_path.glob("*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {data_path}")
        
        # Sample subset
        subset_count = max(1, int(len(all_files) * args.subset_fraction))
        random.shuffle(all_files)
        subset_files = all_files[:subset_count]
        
        print(f"  Total files: {len(all_files)}")
        print(f"  Using subset: {len(subset_files)} ({args.subset_fraction * 100:.1f}%)")
        
        dataset = CachedTileDataset(
            file_paths=subset_files,
            preload=True,
            verbose=True,
        )
    else:
        print(f"Loading preprocessed dataset: {args.data}")
        full_dataset = PreprocessedTileDataset(args.data)
        
        # Sample subset
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        subset_count = max(1, int(len(indices) * args.subset_fraction))
        subset_indices = indices[:subset_count]
        
        print(f"  Total samples: {len(full_dataset)}")
        print(f"  Using subset: {len(subset_indices)} ({args.subset_fraction * 100:.1f}%)")
        
        dataset = Subset(full_dataset, subset_indices)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"  Batches per epoch: {len(loader)}\n")
    
    # Model configuration
    model_config = {
        "embedding_dim": 32,
        "h_dim": 128,
        "res_h_dim": 64,
        "n_embeddings": 512,
        "beta": args.beta,
        "use_ema": not args.no_ema,
        "ema_decay": args.ema_decay,
        "ema_reset_threshold": 0.5,
        "ema_reset_interval": 500,
        "enable_encoder_decoder_tracking": False,  # Disable for diagnostics (faster)
    }
    
    model = VQVAE(**model_config).to(device)
    
    # Optionally load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("  Checkpoint loaded\n")
    
    # Run diagnostic epoch(s)
    model.train()  # Use training mode to see training dynamics
    
    all_losses = []
    all_perplexities = []
    all_block_accuracies = []
    all_class_distributions = []
    all_encodings = []
    start_time = time.time()
    
    print("Running diagnostic epochs...")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_block_acc = []
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            
            with torch.no_grad():  # No gradient computation for diagnostics
                embedding_loss, x_hat, perplexity, logits = model(batch)
                
                total_loss, loss_dict = compute_loss(
                    batch,
                    x_hat,
                    logits,
                    embedding_loss,
                    block_loss_weighted=args.block_loss_weighted,
                )
                
                # Compute block accuracy
                block_logits = logits[0]
                block_targets = batch[:, 0, :, :].long()
                acc_dict = compute_class_accuracy(block_logits, block_targets, NUM_NATURAL_BLOCKS)
                
                # Collect metrics
                epoch_losses.append(loss_dict)
                all_perplexities.append(perplexity.item())
                epoch_block_acc.append(acc_dict["overall"])
                
                # Collect class distribution
                dist = compute_class_distribution(block_targets, NUM_NATURAL_BLOCKS)
                all_class_distributions.append(dist)
                
                # Collect encodings for codebook analysis
                if hasattr(model, 'last_encodings') and model.last_encodings is not None:
                    all_encodings.append(model.last_encodings.cpu())
                
                all_losses.append(loss_dict)
            
            if batch_idx % 10 == 0:
                recent_loss = sum(l["total"] for l in epoch_losses[-10:]) / max(1, len(epoch_losses[-10:]))
                recent_acc = sum(epoch_block_acc[-10:]) / max(1, len(epoch_block_acc[-10:]))
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(loader)}] "
                    f"Loss: {recent_loss:.4f} Block Acc: {recent_acc*100:.1f}%"
                )
        
        avg_loss = sum(l["total"] for l in epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_block_acc) / len(epoch_block_acc)
        print(f"\nEpoch {epoch+1} completed: Loss={avg_loss:.4f}, Block Acc={avg_acc*100:.1f}%\n")
    
    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Diagnostic run completed in {elapsed/60:.1f} minutes\n")
    
    # Aggregate statistics
    print("=" * 80)
    print("DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Loss breakdown
    print("\nLoss Metrics:")
    avg_total = sum(l["total"] for l in all_losses) / len(all_losses)
    avg_block = sum(l["block"] for l in all_losses) / len(all_losses)
    avg_shape = sum(l["shape"] for l in all_losses) / len(all_losses)
    avg_wall = sum(l["wall"] for l in all_losses) / len(all_losses)
    avg_liquid = sum(l["liquid"] for l in all_losses) / len(all_losses)
    avg_continuous = sum(l["continuous"] for l in all_losses) / len(all_losses)
    avg_embedding = sum(l["embedding"] for l in all_losses) / len(all_losses)
    
    print(f"  Total Loss:      {avg_total:.4f}")
    print(f"  Block Loss:      {avg_block:.4f}")
    print(f"  Shape Loss:      {avg_shape:.4f}")
    print(f"  Wall Loss:       {avg_wall:.4f}")
    print(f"  Liquid Loss:     {avg_liquid:.4f}")
    print(f"  Continuous Loss: {avg_continuous:.4f}")
    print(f"  Embedding Loss:  {avg_embedding:.4f}")
    
    # Perplexity
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    print(f"\nCodebook Health:")
    print(f"  Average Perplexity: {avg_perplexity:.2f}")
    print(f"  Utilization: {(avg_perplexity / model_config['n_embeddings']) * 100:.1f}%")
    
    # Codebook analysis
    codebook_stats = {}
    if all_encodings:
        all_encodings_tensor = torch.cat(all_encodings, dim=0)
        codebook_stats = analyze_codebook_health(all_encodings_tensor, model_config['n_embeddings'])
        print(f"  Active codes: {codebook_stats['active_codes']}/{model_config['n_embeddings']}")
        print(f"  Dead codes: {codebook_stats['dead_codes']}")
        print(f"  Top-1 code share: {codebook_stats['top1_share']*100:.2f}%")
        print(f"  Top-5 code share: {codebook_stats['top5_share']*100:.2f}%")
        print(f"  Entropy (normalized): {codebook_stats['entropy_normalized']:.3f}")
    
    # Class distribution analysis
    print("\nClass Distribution (Top 20 most frequent blocks):")
    # Aggregate all class distributions
    total_dist = {}
    for dist in all_class_distributions:
        for class_idx, count in dist.items():
            total_dist[class_idx] = total_dist.get(class_idx, 0) + count
    
    sorted_classes = sorted(total_dist.items(), key=lambda x: x[1], reverse=True)
    total_count = sum(total_dist.values())
    cumulative = 0
    
    for i, (class_idx, count) in enumerate(sorted_classes[:20]):
        percent = (count / total_count) * 100
        cumulative += percent
        block_id = BLOCK_INDEX_TO_ID.get(class_idx, class_idx)
        print(f"  {i+1:2d}. Class {class_idx:3d} (ID={block_id:3d}): {count:8d} ({percent:5.2f}%) | Cumulative: {cumulative:5.1f}%")
    
    # Suggestions
    print("\n" + "=" * 80)
    print("SUGGESTIONS")
    print("=" * 80)
    
    suggestions = []
    
    # Check block loss
    if avg_block > 0.4:
        suggestions.append("🔧 Block loss is high → Consider using --block-loss-weighted to handle class imbalance")
    
    # Check codebook utilization
    if avg_perplexity < model_config['n_embeddings'] * 0.4:
        suggestions.append(f"🔧 Low codebook utilization ({avg_perplexity:.0f}/{model_config['n_embeddings']}) → Try --ema-reset-threshold 0.75")
    
    # Check class imbalance
    if sorted_classes:
        top5_percent = sum(c for _, c in sorted_classes[:5]) / total_count * 100
        if top5_percent > 80:
            suggestions.append(f"🔧 Severe class imbalance (top 5 classes = {top5_percent:.1f}%) → Use --block-loss-weighted")
    
    # Check if loss is very high
    if avg_total > 3.0:
        suggestions.append("🔧 High total loss → Consider running LR finder: terrain model lr-find")
    
    if suggestions:
        for s in suggestions:
            print(s)
    else:
        print("✓ No obvious issues detected. Model appears healthy.")
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "config": model_config,
        "data_subset_fraction": args.subset_fraction,
        "epochs": args.epochs,
        "total_batches": len(all_losses),
        "metrics": {
            "avg_total_loss": avg_total,
            "avg_block_loss": avg_block,
            "avg_shape_loss": avg_shape,
            "avg_wall_loss": avg_wall,
            "avg_liquid_loss": avg_liquid,
            "avg_continuous_loss": avg_continuous,
            "avg_embedding_loss": avg_embedding,
            "avg_perplexity": avg_perplexity,
            "avg_block_accuracy": sum(all_block_accuracies) / len(all_block_accuracies) if all_block_accuracies else 0.0,
        },
        "codebook": codebook_stats if all_encodings else {},
        "class_distribution_top20": [
            {"class_idx": int(idx), "block_id": BLOCK_INDEX_TO_ID.get(idx, idx), "count": int(cnt)}
            for idx, cnt in sorted_classes[:20]
        ],
        "suggestions": suggestions,
        "elapsed_seconds": elapsed,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Diagnostic report saved to: {output_path}")
    print("=" * 80 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
