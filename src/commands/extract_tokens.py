"""Extract token sequences from VQVAE for transformer training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.autoencoder.dataset import PreprocessedTileDataset
from src.autoencoder.vqvae import VQVAE, DEFAULT_MODEL_CONFIG
from src.utils.checkpoint import load_model_for_inference, read_checkpoint_config
from src.utils.device import get_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract VQVAE token sequences for transformer training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained VQVAE checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.pt",
        help="Path to preprocessed tile dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/token_sequences.pt",
        help="Output path for token sequences",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (larger = faster but more VRAM)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    return parser


def extract_tokens_from_vqvae(
    model: VQVAE,
    chunks: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode chunks through VQVAE and extract token indices.
    
    Args:
        model: Trained VQVAE model
        chunks: (B, 8, 32, 32) tile tensors
        device: Device to run on
        
    Returns:
        tokens: (B, 64) token indices (flattened 8×8 grids)
    """
    chunks = chunks.to(device)
    
    with torch.no_grad():
        # Encode through tile encoder and conv encoder
        tile_encoded = model.tile_encoder(chunks)
        z_e = model.conv_encoder(tile_encoded)
        
        # Quantize to get token indices
        _, z_q, _, _, encoding_indices = model.vq(z_e)
        
        # encoding_indices shape: (B*H*W, 1) where H=8, W=8
        # Reshape to (B, 64) - flattened 8×8 grid
        batch_size = chunks.shape[0]
        tokens = encoding_indices.view(batch_size, -1).squeeze(-1)  # (B, 64)
        
    return tokens


def run(args: argparse.Namespace) -> None:
    """Execute token extraction."""
    device = get_device()
    print(f"Using device: {device}\n")
    
    # -------------------------------------------------------------------------
    # Load VQVAE model
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("LOADING VQVAE MODEL")
    print(f"{'=' * 80}\n")
    
    if args.model:
        model_path = args.model
    else:
        checkpoints_dir = Path("checkpoints")
        best = checkpoints_dir / "best_model.pt"
        latest = checkpoints_dir / "latest_model.pt"
        
        if best.exists():
            model_path = str(best)
        elif latest.exists():
            model_path = str(latest)
        else:
            print("❌ No checkpoint found. Please train VQVAE first.")
            print("   Expected: checkpoints/best_model.pt or checkpoints/latest_model.pt")
            return
    
    print(f"Loading checkpoint: {model_path}")
    
    ckpt_config = read_checkpoint_config(model_path, str(device))
    if not ckpt_config:
        print("  [warn] No config in checkpoint; using default architecture.")
    model_config = {**DEFAULT_MODEL_CONFIG, **ckpt_config}
    
    print(f"  Codebook size: {model_config['n_embeddings']}")
    print(f"  Hidden dim: {model_config['h_dim']}")
    print(f"  Using EMA: {model_config['use_ema']}")
    
    model = VQVAE(**model_config)
    model = load_model_for_inference(model, model_path, str(device))
    
    print("✓ Model loaded successfully\n")
    
    # -------------------------------------------------------------------------
    # Load tile dataset
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("LOADING TILE DATASET")
    print(f"{'=' * 80}\n")
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run 'terrain data prepare' first to create the dataset.")
        return
    
    print(f"Loading dataset: {data_path}")
    dataset = PreprocessedTileDataset(data_path)
    
    print(f"  Total chunks: {len(dataset)}")
    print(f"  Chunk shape: {dataset[0].shape}")  # Should be (8, 32, 32)
    
    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    print(f"\nSplit:")
    print(f"  Training: {train_size:,} chunks ({100*(1-args.val_split):.1f}%)")
    print(f"  Validation: {val_size:,} chunks ({100*args.val_split:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Extract token sequences
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("EXTRACTING TOKEN SEQUENCES")
    print(f"{'=' * 80}\n")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep original order for reproducible splits
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    all_tokens = []
    
    print(f"Processing {len(dataloader)} batches...")
    for batch_chunks in tqdm(dataloader, desc="Encoding"):
        tokens = extract_tokens_from_vqvae(model, batch_chunks, device)
        all_tokens.append(tokens.cpu())
    
    # Concatenate all batches
    all_tokens = torch.cat(all_tokens, dim=0)  # (N, 64)
    
    print(f"\n✓ Extracted {all_tokens.shape[0]:,} token sequences")
    print(f"  Shape: {all_tokens.shape}  (num_sequences, 64)")
    print(f"  Token range: [{all_tokens.min()}, {all_tokens.max()}]")
    print(f"  Expected range: [0, {model_config['n_embeddings']-1}]")
    
    # Verify tokens are in valid range
    if all_tokens.min() < 0 or all_tokens.max() >= model_config['n_embeddings']:
        print(f"⚠️  Warning: Some tokens are out of range!")
    
    # Split sequences
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]
    
    # -------------------------------------------------------------------------
    # Compute statistics
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("TOKEN STATISTICS")
    print(f"{'=' * 80}\n")
    
    unique_tokens = torch.unique(all_tokens).numel()
    print(f"Unique tokens used: {unique_tokens} / {model_config['n_embeddings']} "
          f"({100*unique_tokens/model_config['n_embeddings']:.1f}%)")
    
    # Most common tokens
    token_counts = torch.bincount(all_tokens.flatten(), minlength=model_config['n_embeddings'])
    top_k = 10
    top_tokens = torch.topk(token_counts, k=top_k)
    
    print(f"\nTop {top_k} most common tokens:")
    total_count = all_tokens.numel()
    for i, (count, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices), 1):
        pct = 100 * count.item() / total_count
        print(f"  {i:2d}. Token {token_id:4d}: {count:8d} occurrences ({pct:5.2f}%)")
    
    # -------------------------------------------------------------------------
    # Save to disk
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("SAVING TOKEN SEQUENCES")
    print(f"{'=' * 80}\n")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Package everything together
    data_to_save = {
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'config': {
            'vocab_size': model_config['n_embeddings'],
            'sequence_length': 64,  # 8×8 flattened
            'grid_size': (8, 8),
            'original_chunk_size': (32, 32),
            'num_train': train_size,
            'num_val': val_size,
        },
        'model_config': model_config,
        'source_dataset': str(data_path),
        'checkpoint': model_path,
    }
    
    torch.save(data_to_save, output_path)
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"✓ Saved token sequences to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Train tokens: {train_tokens.shape}")
    print(f"  Val tokens: {val_tokens.shape}")
    
    print(f"\n{'=' * 80}")
    print("✓ TOKEN EXTRACTION COMPLETE")
    print(f"{'=' * 80}\n")
    
    print("Next steps:")
    print("  1. Train transformer: terrain gen train --data", output_path)
    print(f"  2. Generate terrain: terrain gen sample --model <checkpoint>\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for command-line execution."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
