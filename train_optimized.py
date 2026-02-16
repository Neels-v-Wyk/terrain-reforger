#!/usr/bin/env python3
"""
Train optimized VQ-VAE on Terraria natural world data (9 channels).

This version uses findings from world analysis:
- 218 block types (31.5% of total)
- 77 wall types (22.2% of total)
- No paint, illuminants, or echo coating
- Simplified liquids (binary + type only)
- 9 channels instead of 17 (47% reduction)
- 745 fewer embedding table entries
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import os

from src.autoencoder.vqvae_optimized import VQVAEOptimized, compute_optimized_loss
from src.autoencoder.dataset_optimized import OptimizedTerrariaTileDataset, PreprocessedTileDataset
from src.autoencoder.dataset_cached import CachedTileDataset
from src.autoencoder.samplers import InterleavedFileSampler
from src.utils.checkpoint import CheckpointManager, save_final_model
from src.utils.visualization import plot_training_results, compare_optimized_tiles
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train VQ-VAE on Terraria chunks")
    parser.add_argument('--data', type=str, help='Path to preprocessed .pt dataset')
    parser.add_argument('--world', type=str, help='Path to single .wld file (if not using preprocessed data)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--one-pass', action='store_true', help='Train for exactly one pass over the data (epochs=1)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--disk-mode', action='store_true', help='Use disk-based loading with LRU cache (saves RAM)')
    parser.add_argument('--cache-size', type=int, default=5, help='LRU cache size (files) for disk mode')
    
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration (optimized)
    model_config = {
        'embedding_dim': 32,      # Per-category embedding size
        'h_dim': 128,            # Hidden dimension
        'res_h_dim': 64,         # Residual block hidden dimension
        'n_embeddings': 512,     # Codebook size
        'beta': 0.25             # VQ loss weight
    }
    
    # Training configuration
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1 if args.one_pass else args.epochs
    CHECKPOINT_INTERVAL = 500  # Save every N updates
    
    # Dataset loading
    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
             # Use new cached dataset
             print(f"Loading cached dataset from directory {data_path}...")
             preload_mode = not args.disk_mode
             
             training_data = CachedTileDataset(
                 data_dir=data_path, 
                 preload=preload_mode,
                 max_cache_size=args.cache_size
             )
             CHUNK_SIZE = 32 # Assuming default
             print(f"  Chunk size: {CHUNK_SIZE} (assumed)")
        else:
            # Use preprocessed dataset file
            print(f"Loading preprocessed dataset from {args.data}...")
            training_data = PreprocessedTileDataset(args.data)
            CHUNK_SIZE = training_data.config['chunk_size']
            print(f"  Chunk size: {CHUNK_SIZE}")
    else:
        # Use single world extraction
        WORLD_FILE = args.world if args.world else "worldgen/World_20260211_213447_mBvegqrN.wld"
        REGION = (0, 0, 8360, 2360)  # Use larger region
        CHUNK_SIZE = 32
        OVERLAP = 16
        
        print(f"Creating optimized training dataset from {WORLD_FILE}...")
        training_data = OptimizedTerrariaTileDataset(
            world_path=WORLD_FILE,
            region=REGION,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            use_diversity_filter=True,
            min_diversity=0.20,
            deduplicate=True
        )
    
    # Configure Sampler for Disk Mode to prevent thrashing
    sampler = None
    shuffle = True
    
    if args.data and Path(args.data).is_dir() and args.disk_mode:
        if isinstance(training_data, CachedTileDataset):
            print(f"Using InterleavedFileSampler to optimize disk access (Cache Size: {args.cache_size})...")
            sampler = InterleavedFileSampler(
                training_data, 
                max_cache_size=args.cache_size,
                shuffle_files=True
            )
            shuffle = False # Sampler handles the shuffling logic
        
    # Create data loader
    train_loader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=shuffle, # Must be False if sampler is provided
        sampler=sampler,
        num_workers=0  # Single worker for now
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total training samples: {len(training_data)}")
    print(f"  Model: Optimized VQ-VAE (9 channels, 218 blocks, 77 walls)")
    print(f"  Embedding dim: {model_config['embedding_dim']}")
    print(f"  Codebook size: {model_config['n_embeddings']}")
    
    # Create model
    model = VQVAEOptimized(**model_config).to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
    
    # Track results
    results = {
        'loss_vals': [],
        'perplexities': [],
        'block_loss': [],
        'wall_loss': [],
        'liquid_loss': [],
        'continuous_loss': []
    }
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    resume_path = args.resume or os.environ.get('RESUME_CHECKPOINT')
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from checkpoint: {resume_path}")
        model, loaded_optimizer, results, start_epoch = checkpoint_manager.load_checkpoint(
            model, resume_path, optimizer, str(device)
        )
        if loaded_optimizer is not None:
            optimizer = loaded_optimizer
        best_loss = min(results.get('loss_vals', [float('inf')]))
        print(f"  Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    model.train()
    update_count = 0
    last_batch: torch.Tensor | None = None
    last_reconstruction: torch.Tensor | None = None
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'categorical': [],
            'block': [],
            'wall': [],
            'liquid': [],
            'continuous': [],
            'embedding': []
        }
        epoch_perplexities = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            last_batch = batch
            
            # Forward pass
            embedding_loss, x_hat, perplexity, logits = model(batch)
            last_reconstruction = x_hat
            
            # Compute loss
            total_loss, loss_dict = compute_optimized_loss(
                batch, x_hat, logits, embedding_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            epoch_perplexities.append(perplexity.item())
            
            # Add to global results
            results['loss_vals'].append(loss_dict['total'])
            results['perplexities'].append(perplexity.item())
            results['block_loss'].append(loss_dict['block'])
            results['wall_loss'].append(loss_dict['wall'])
            results['liquid_loss'].append(loss_dict['liquid'])
            results['continuous_loss'].append(loss_dict['continuous'])
            
            update_count += 1
            
            # Periodic checkpoint
            if update_count % CHECKPOINT_INTERVAL == 0:
                avg_loss = sum(epoch_losses['total'][-CHECKPOINT_INTERVAL:]) / min(CHECKPOINT_INTERVAL, len(epoch_losses['total']))
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                    
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    results=results,
                    epoch=epoch,
                    config=model_config,
                    is_best=is_best
                )
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                recent_loss = sum(epoch_losses['total'][-10:]) / len(epoch_losses['total'][-10:])
                recent_perp = sum(epoch_perplexities[-10:]) / len(epoch_perplexities[-10:])
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {recent_loss:.4f} Perplexity: {recent_perp:.2f}")
        
        # End of epoch summary
        avg_losses = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        avg_perplexity = sum(epoch_perplexities) / len(epoch_perplexities)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_losses['total']:.4f}")
        print(f"  Reconstruction: {avg_losses['reconstruction']:.4f}")
        print(f"    Categorical: {avg_losses['categorical']:.4f}")
        print(f"      Block: {avg_losses['block']:.4f}")
        print(f"      Wall: {avg_losses['wall']:.4f}")
        print(f"      Liquid: {avg_losses['liquid']:.4f}")
        print(f"    Continuous: {avg_losses['continuous']:.4f}")
        print(f"  Embedding: {avg_losses['embedding']:.4f}")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Codebook usage: {(avg_perplexity / model_config['n_embeddings']) * 100:.1f}%")
        print(f"{'='*80}\n")
        
        # Plot metrics
        plot_training_results(results, "checkpoints")
        
        # Visualize a few blocks from the last batch
        if last_batch is not None and last_reconstruction is not None:
            print("Block Comparison (Last Batch):")
            # Get last sample in batch
            orig_sample = last_batch[-1].cpu().detach().numpy()  # (9, H, W)
            recon_sample = last_reconstruction[-1].cpu().detach().numpy() # (9, H, W)
            
            # Compare 3 consecutive blocks from the middle
            mid_h, mid_w = CHUNK_SIZE // 2, CHUNK_SIZE // 2
            for i in range(3):
                # Extract single tile vector (9,)
                orig_tile = orig_sample[:, mid_h, mid_w + i]
                recon_tile = recon_sample[:, mid_h, mid_w + i]
                print("-" * 40)
                print(compare_optimized_tiles(orig_tile, recon_tile, i+1))
            print("-" * 40 + "\n")
    
    # Save final model
    print("\nTraining complete! Saving final model...")
    final_path = save_final_model(
        model, optimizer,
        results,
        model_config,
        "checkpoints"
    )
    print(f"Final model saved to: {final_path}")
    
    print("\nTo load this model for inference:")
    print("  from src.utils.checkpoint import CheckpointManager")
    print(f"  manager = CheckpointManager('checkpoints')")
    print(f"  model, _, _, _ = manager.load_checkpoint(model, '{final_path}')")


if __name__ == "__main__":
    main()
