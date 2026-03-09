"""Training script for MaskGIT terrain model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generative.dataset import TokenSequenceDataset
from src.generative.maskgit import (
    TerrainMaskGIT,
    create_maskgit_model,
    MASKGIT_SMALL_CONFIG,
    MASKGIT_MEDIUM_CONFIG,
    MASKGIT_SMALL_PLUS_CONFIG,
    MASKGIT_LARGE_CONFIG,
)
from src.generative.masking import scheduled_masking, get_mask_ratio_for_epoch
from src.utils.device import get_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MaskGIT terrain model")
    
    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="data/token_sequences.pt",
        help="Path to token sequences dataset",
    )
    
    # Model architecture
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "small-plus", "medium", "large"],
        help="Model size preset (default: small)",
    )
    parser.add_argument("--d-model", type=int, help="Hidden dimension (overrides preset)")
    parser.add_argument("--n-layers", type=int, help="Number of layers (overrides preset)")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads (overrides preset)")
    parser.add_argument("--d-ff", type=int, help="Feedforward dimension (overrides preset)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    
    # Masking strategy
    parser.add_argument("--mask-ratio-start", type=float, default=0.6, help="Starting mask ratio")
    parser.add_argument("--mask-ratio-end", type=float, default=0.3, help="Ending mask ratio")
    parser.add_argument("--mask-schedule", type=str, default="cosine", 
                        choices=["linear", "cosine", "constant"], help="Mask ratio schedule")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/transformer", 
                        help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate during eval")
    parser.add_argument("--num-iterations", type=int, default=12, help="Number of iterations for generation")
    
    # Other
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser


def train_epoch(
    model: TerrainMaskGIT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
    mask_ratio_start: float,
    mask_ratio_end: float,
    mask_schedule: str,
) -> dict:
    """Train for one epoch with masked token prediction."""
    model.train()
    
    total_loss = 0.0
    total_masked_tokens = 0
    
    # Get mask ratio for this epoch
    mask_ratio = get_mask_ratio_for_epoch(
        epoch, total_epochs, mask_ratio_start, mask_ratio_end, mask_schedule
    )
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (mask={mask_ratio:.1%})")
    for batch_idx, tokens in enumerate(pbar):
        tokens = tokens.to(device)  # (batch, 64)
        
        # Apply masking
        masked_tokens, mask, targets = scheduled_masking(
            tokens=tokens,
            mask_ratio=mask_ratio,
            mask_token_id=model.mask_token_id,
            epoch=epoch,
            total_epochs=total_epochs,
            start_ratio=mask_ratio_start,
            end_ratio=mask_ratio_end,
            schedule=mask_schedule,
        )
        
        # Forward pass and compute loss on masked positions only
        optimizer.zero_grad()
        loss = model.compute_loss(masked_tokens, targets, mask)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        batch_loss = loss.item()
        num_masked = mask.sum().item()
        total_loss += batch_loss * num_masked
        total_masked_tokens += num_masked
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_loss = total_loss / total_masked_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'train_loss': avg_loss,
        'train_perplexity': perplexity,
        'mask_ratio': mask_ratio,
    }


@torch.no_grad()
def evaluate(
    model: TerrainMaskGIT,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_masked_tokens = 0
    
    for tokens in tqdm(dataloader, desc="Evaluating"):
        tokens = tokens.to(device)
        
        # Apply same masking ratio as training
        from src.generative.masking import random_masking
        masked_tokens, mask, targets = random_masking(
            tokens, mask_ratio, model.mask_token_id
        )
        
        # Compute loss
        loss = model.compute_loss(masked_tokens, targets, mask)
        
        # Track metrics
        batch_loss = loss.item()
        num_masked = mask.sum().item()
        total_loss += batch_loss * num_masked
        total_masked_tokens += num_masked
    
    avg_loss = total_loss / total_masked_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity,
    }


def save_checkpoint(
    model: TerrainMaskGIT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'n_layers': model.n_layers,
            'n_heads': model.n_heads,
            'grid_size': model.grid_size,
            'mask_token_id': model.mask_token_id,
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"transformer_epoch{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save latest
    latest_path = checkpoint_dir / "latest_model.pt"
    torch.save(checkpoint, latest_path)


def run(args: argparse.Namespace) -> None:
    """Main training function."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TokenSequenceDataset(args.data, split='train')
    val_dataset = TokenSequenceDataset(args.data, split='val')
    
    vocab_size = train_dataset.get_vocab_size()
    seq_len = train_dataset.get_sequence_length()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    
    # Select config based on size
    if args.model_size == 'small':
        config = MASKGIT_SMALL_CONFIG
    elif args.model_size == 'small-plus':
        config = MASKGIT_SMALL_PLUS_CONFIG
    elif args.model_size == 'medium':
        config = MASKGIT_MEDIUM_CONFIG
    elif args.model_size == 'large':
        config = MASKGIT_LARGE_CONFIG
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Override config with command line args
    if args.d_model is not None:
        config['d_model'] = args.d_model
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    if args.n_heads is not None:
        config['n_heads'] = args.n_heads
    if args.d_ff is not None:
        config['d_ff'] = args.d_ff
    config['dropout'] = args.dropout
    
    model = create_maskgit_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        **config
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.1,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    checkpoint_dir = Path(args.checkpoint_dir)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            total_epochs=args.epochs,
            mask_ratio_start=args.mask_ratio_start,
            mask_ratio_end=args.mask_ratio_end,
            mask_schedule=args.mask_schedule,
        )
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                mask_ratio=train_metrics['mask_ratio'],
            )
        else:
            val_metrics = {}
        
        # Update learning rate
        scheduler.step()
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics, 'lr': scheduler.get_last_lr()[0]}
        
        # Print epoch summary
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}, Perplexity: {metrics['train_perplexity']:.2f}")
        if val_metrics:
            print(f"  Val Loss: {metrics['val_loss']:.4f}, Perplexity: {metrics['val_perplexity']:.2f}")
        print(f"  Mask Ratio: {metrics['mask_ratio']:.1%}")
        print(f"  Learning Rate: {metrics['lr']:.2e}")
        
        # Generate samples
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nGenerating {args.num_samples} samples...")
            samples = model.generate(
                num_samples=args.num_samples,
                num_iterations=args.num_iterations,
                temperature=1.0,
                device=device,
            )
            print(f"Sample token range: [{samples.min()}, {samples.max()}]")
            print(f"Sample shape: {samples.shape}")
        
        # Save checkpoint
        is_best = val_metrics and val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
            )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
