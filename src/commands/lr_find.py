"""Learning rate finder command (Leslie Smith LR range test)."""

import argparse
import random
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from src.autoencoder.dataset_cached import CachedTileDataset
from src.autoencoder.dataset import PreprocessedTileDataset
from src.autoencoder.vqvae import VQVAE, compute_loss
from src.utils.device import get_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find optimal learning rate using LR range test")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed dataset")
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate to test (default: 1e-6)",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate to test (default: 1e-2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of steps in the LR range test (default: 300)",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use (default: 0.1)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.05,
        help="Exponential smoothing factor for loss (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/lr_finder.png",
        help="Output path for LR finder plot",
    )
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--no-plot", action="store_true", help="Don't display plot, just save")
    return parser


def exponential_moving_average(values, smoothing=0.05):
    """Apply exponential moving average smoothing."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(smoothing * v + (1 - smoothing) * smoothed[-1])
    return smoothed


def find_optimal_lr(lrs, losses):
    """
    Find optimal LR as the point of steepest descent before loss diverges.
    Returns (optimal_lr, suggested_min, suggested_max).
    """
    if len(losses) < 10:
        return lrs[len(lrs)//2], lrs[0], lrs[-1]
    
    # Find the minimum loss
    min_loss_idx = np.argmin(losses)
    
    # Find steepest gradient (most negative) before the minimum
    gradients = np.gradient(losses)
    steepest_idx = np.argmin(gradients[:min_loss_idx]) if min_loss_idx > 10 else min_loss_idx // 2
    
    optimal_lr = lrs[steepest_idx]
    
    # Suggested range: 1/10 to 1x of optimal
    suggested_min = optimal_lr / 10
    suggested_max = optimal_lr
    
    return optimal_lr, suggested_min, suggested_max


def run(args: argparse.Namespace) -> None:
    """Run LR range test."""
    device = get_device()
    print("\n" + "=" * 80)
    print("LEARNING RATE FINDER (Leslie Smith method)")
    print("=" * 80 + "\n")
    print(f"Device: {device}")
    print(f"Testing range: {args.min_lr:.2e} → {args.max_lr:.2e} over {args.steps} steps\n")
    
    # Load dataset
    data_path = Path(args.data)
    if data_path.is_dir():
        print(f"Loading dataset from directory: {data_path}")
        all_files = sorted(list(data_path.glob("*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {data_path}")
        
        subset_count = max(1, int(len(all_files) * args.subset_fraction))
        random.shuffle(all_files)
        subset_files = all_files[:subset_count]
        
        print(f"  Using {len(subset_files)}/{len(all_files)} files ({args.subset_fraction * 100:.1f}%)")
        
        dataset = CachedTileDataset(
            file_paths=subset_files,
            preload=True,
            verbose=False,
        )
    else:
        print(f"Loading preprocessed dataset: {args.data}")
        full_dataset = PreprocessedTileDataset(args.data)
        
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        subset_count = max(1, int(len(indices) * args.subset_fraction))
        subset_indices = indices[:subset_count]
        
        print(f"  Using {len(subset_indices)}/{len(full_dataset)} samples ({args.subset_fraction * 100:.1f}%)")
        
        dataset = Subset(full_dataset, subset_indices)
    
    # Create data loader with enough samples
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model configuration
    model_config = {
        "embedding_dim": 32,
        "h_dim": 128,
        "res_h_dim": 64,
        "n_embeddings": 512,
        "beta": args.beta,
        "use_ema": True,
        "ema_decay": 0.99,
        "ema_reset_threshold": 0.5,
        "ema_reset_interval": 500,
        "enable_encoder_decoder_tracking": False,
    }
    
    model = VQVAE(**model_config).to(device)
    optimizer = Adam(model.parameters(), lr=args.min_lr)
    
    # LR schedule: exponential from min_lr to max_lr
    lr_mult = (args.max_lr / args.min_lr) ** (1.0 / args.steps)
    
    lrs = []
    losses = []
    
    print("Running LR range test...")
    print("-" * 80)
    
    model.train()
    step = 0
    data_iter = iter(loader)
    
    while step < args.steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        
        # Update learning rate
        current_lr = args.min_lr * (lr_mult ** step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Forward pass
        embedding_loss, x_hat, perplexity, logits = model(batch)
        total_loss, loss_dict = compute_loss(
            batch,
            x_hat,
            logits,
            embedding_loss,
            block_loss_weighted=False,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record
        lrs.append(current_lr)
        losses.append(loss_dict["total"])
        
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}/{args.steps} | LR: {current_lr:.2e} | Loss: {loss_dict['total']:.4f}")
    
    print("-" * 80)
    print("LR range test completed\n")
    
    # Smooth losses
    smoothed_losses = exponential_moving_average(losses, args.smoothing)
    
    # Find optimal LR
    optimal_lr, suggested_min, suggested_max = find_optimal_lr(lrs, smoothed_losses)
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nRecommended Learning Rate: {optimal_lr:.2e}")
    print(f"  (Point of steepest descent before divergence)")
    print(f"\nSuggested LR range to explore:")
    print(f"  Minimum: {suggested_min:.2e}")
    print(f"  Maximum: {suggested_max:.2e}")
    
    # Detect if loss diverged
    if smoothed_losses[-1] > smoothed_losses[0] * 2:
        print(f"\n⚠️  Loss diverged at high LR (final loss is {smoothed_losses[-1]/smoothed_losses[0]:.1f}x initial)")
        print(f"   This is expected and helps identify the upper bound.")
    
    print("\n" + "=" * 80)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw and smoothed losses
    ax.plot(lrs, losses, alpha=0.3, color='blue', label='Raw Loss')
    ax.plot(lrs, smoothed_losses, linewidth=2, color='blue', label='Smoothed Loss')
    
    # Mark optimal LR
    optimal_idx = lrs.index(optimal_lr)
    ax.axvline(optimal_lr, color='red', linestyle='--', linewidth=2, label=f'Recommended: {optimal_lr:.2e}')
    ax.scatter([optimal_lr], [smoothed_losses[optimal_idx]], color='red', s=100, zorder=5)
    
    # Mark safe range
    ax.axvspan(suggested_min, suggested_max, alpha=0.2, color='green', label='Suggested Range')
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Rate Finder (Leslie Smith Method)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Plot saved to: {output_path}")
    
    # Display
    if not args.no_plot:
        try:
            plt.show()
        except:
            pass  # Failed to display, but already saved
    
    plt.close()
    
    print("\nNext steps:")
    print(f"  1. Use --learning-rate {optimal_lr:.2e} for your next training run")
    print(f"  2. Or experiment within range [{suggested_min:.2e}, {suggested_max:.2e}]")
    print(f"  3. Consider using a learning rate schedule (warmup + cosine decay)")
    print("")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
