"""Training script for terrain transformer."""

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
from src.generative.transformer import (
    TerrainTransformer,
    create_model,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    LARGE_CONFIG,
)
from src.utils.device import get_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train terrain transformer")
    
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
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size preset (default: medium)",
    )
    parser.add_argument("--d-model", type=int, help="Hidden dimension (overrides preset)")
    parser.add_argument("--n-layers", type=int, help="Number of layers (overrides preset)")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads (overrides preset)")
    parser.add_argument("--d-ff", type=int, help="Feedforward dimension (overrides preset)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/transformer", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate during eval")
    
    # Other
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, tokens in enumerate(pbar):
        tokens = tokens.to(device)  # (batch, 64)
        
        # Teacher forcing: predict next token given previous tokens
        # Input: tokens[:, :-1], Target: tokens[:, 1:]
        # But we can also just use the full sequence and shift in the loss
        input_tokens = tokens
        target_tokens = tokens
        
        # Forward pass
        logits, loss = model(input_tokens, targets=target_tokens)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update metrics
        batch_size = tokens.size(0)
        seq_len = tokens.size(1)
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'ppl': f"{torch.exp(loss):.2f}",
        })
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for tokens in tqdm(dataloader, desc="Evaluating"):
        tokens = tokens.to(device)
        
        # Forward pass
        logits, loss = model(tokens, targets=tokens)
        
        # Update metrics
        batch_size = tokens.size(0)
        seq_len = tokens.size(1)
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item(),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    checkpoint_path: Path,
    model_config: dict,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_config': model_config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def run(args: argparse.Namespace) -> None:
    """Execute training."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("LOADING DATASETS")
    print(f"{'=' * 80}\n")
    
    train_dataset = TokenSequenceDataset(args.data, split='train')
    val_dataset = TokenSequenceDataset(args.data, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("CREATING MODEL")
    print(f"{'=' * 80}\n")
    
    # Get model configuration
    if args.model_size == "small":
        config = SMALL_CONFIG.copy()
    elif args.model_size == "medium":
        config = MEDIUM_CONFIG.copy()
    else:
        config = LARGE_CONFIG.copy()
    
    # Override with command-line arguments
    if args.d_model:
        config['d_model'] = args.d_model
    if args.n_layers:
        config['n_layers'] = args.n_layers
    if args.n_heads:
        config['n_heads'] = args.n_heads
    if args.d_ff:
        config['d_ff'] = args.d_ff
    if args.dropout:
        config['dropout'] = args.dropout
    
    vocab_size = train_dataset.get_vocab_size()
    seq_len = train_dataset.get_sequence_length()
    
    model = create_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        **config
    )
    model = model.to(device)
    
    # -------------------------------------------------------------------------
    # Create optimizer and scheduler
    # -------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Cosine annealing scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    # Simple cosine schedule (could add warmup later)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_metrics']['loss']
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best val loss: {best_val_loss:.4f}\n")
    
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("TRAINING")
    print(f"{'=' * 80}\n")
    
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clipping: {args.grad_clip}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print()
    
    # Save training config
    training_config = {
        'model_config': config,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'grad_clip': args.grad_clip,
    }
    
    with open(checkpoint_dir / 'training_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Training history
    history = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
    }
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.grad_clip, epoch + 1
        )
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_perplexity'].append(train_metrics['perplexity'])
        
        # Evaluate
        val_metrics = None
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_perplexity'].append(val_metrics['perplexity'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Perplexity: {train_metrics['perplexity']:.2f}")
        if val_metrics:
            print(f"  Val Loss:   {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = checkpoint_dir / 'best_model.pt'
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    train_metrics, val_metrics, best_path, config
                )
                print(f"  ✓ New best model saved (val loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = checkpoint_dir / f'checkpoint_epoch{epoch+1}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, val_metrics or {}, ckpt_path, config
            )
        
        # Update scheduler
        scheduler.step()
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1,
        train_metrics, val_metrics or {}, final_path, config
    )
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("✓ TRAINING COMPLETE")
    print(f"{'=' * 80}\n")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for command-line execution."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
