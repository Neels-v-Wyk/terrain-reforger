"""
Training script using the embedding-based VQ-VAE architecture.

This version treats categorical features (block types, wall types) as proper
categories with embedding layers, which is more appropriate than continuous normalization.
"""

from src.terraria import load_world
from src.terraria.tensor_utils import (
    decode_tile_from_tensor, 
    format_tile_readable, 
    summarize_chunk_comparison
)
from src.autoencoder.vqvae import (
    VQVAEEmbedded,
    compute_categorical_loss,
    compute_continuous_loss
)
from src.autoencoder.dataset import TerrariaTileDataset
from src.utils.checkpoint import CheckpointManager, save_final_model
import numpy as np
import torch
import os

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

N_UPDATES = 10000
CHECKPOINT_INTERVAL = 500  # Save checkpoint every N updates

# Model configuration (save this for reloading)
model_config = {
    'h_dim': 128,
    'res_h_dim': 64,
    'n_res_layers': 3,
    'n_embeddings': 512,
    'embedding_dim': 64,
    'beta': 0.25,
    'categorical_embedding_dim': 16,
}

model = VQVAEEmbedded(**model_config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'embedding_losses': [],
    'codebook_usage': [],
    'categorical_losses': [],
    'continuous_losses': [],
}

world = load_world('worldgen/World_20260211_213447_mBvegqrN.wld')

print("Creating training dataset with diversity filtering...")
training_data = TerrariaTileDataset(
    world, 
    region=(0, 0, 8360, 2360),  # Use larger region for training
    chunk_size=32, 
    overlap=8,
    use_diversity_filter=True,  # Filter out boring/repetitive chunks
    min_diversity=0.20,          # Minimum diversity score to accept
    deduplicate=True,            # Remove near-duplicate chunks
    verbose=True                 # Show statistics
)

training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

# Optional: Load from checkpoint to resume training
RESUME_FROM_CHECKPOINT = os.getenv('RESUME_CHECKPOINT')  # Set env var to resume
start_epoch = 0

if RESUME_FROM_CHECKPOINT:
    print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
    model, optimizer, results, start_epoch = checkpoint_manager.load_checkpoint(
        model, RESUME_FROM_CHECKPOINT, optimizer, str(device)
    )
    print(f"Resumed from epoch {start_epoch}")
    print(f"Previous loss: {results['loss_vals'][-1] if results['loss_vals'] else 'N/A'}")


def train():
    global start_epoch
    best_loss = float('inf')
    
    for i in range(start_epoch, N_UPDATES):
        x = next(iter(training_loader))
        x = x.to(device)
        
        optimizer.zero_grad()

        embedding_loss, (categorical_logits, continuous_pred), perplexity, encoding_indices = \
            model(x, return_indices=True)
        
        # Compute separate losses for categorical and continuous features
        categorical_loss = compute_categorical_loss(categorical_logits, x)
        continuous_loss = compute_continuous_loss(continuous_pred, x)
        
        # Total reconstruction loss
        recon_loss = categorical_loss + continuous_loss
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        # Track metrics
        unique_codes = len(torch.unique(encoding_indices))
        results["recon_errors"].append(recon_loss.cpu().detach().item())
        results["perplexities"].append(perplexity.cpu().detach().item())
        results["loss_vals"].append(loss.cpu().detach().item())
        results["embedding_losses"].append(embedding_loss.cpu().detach().item())
        results["codebook_usage"].append(unique_codes)
        results["categorical_losses"].append(categorical_loss.cpu().detach().item())
        results["continuous_losses"].append(continuous_loss.cpu().detach().item())
        results["n_updates"] = i

        print(f"Update {i+1}/{N_UPDATES} - Loss: {loss.item():.4f}, "
              f"Cat: {categorical_loss.item():.4f}, Cont: {continuous_loss.item():.4f}, "
              f"Embed: {embedding_loss.item():.4f}, Perp: {perplexity.item():.2f}, "
              f"Codes: {unique_codes}/512")
        
        # Save checkpoint periodically
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            is_best = loss.item() < best_loss
            if is_best:
                best_loss = loss.item()
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                results=results,
                epoch=i+1,
                config=model_config,
                is_best=is_best
            )
            
            # Cleanup old checkpoints (keep last 3)
            checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
        
        # Show detailed comparison every 100 updates
        if (i + 1) % 100 == 0:
            print("\n" + "="*80)
            print(f"Detailed Analysis at Update {i+1}")
            print("="*80)
            
            # Reconstruct full tensor for analysis
            x_recon = model.reconstruct_tensor(categorical_logits, continuous_pred)
            
            # Get first sample from batch
            orig = x[0].cpu().detach().numpy()  # (C, H, W)
            recon = x_recon[0].cpu().detach().numpy()  # (C, H, W)
            
            # Show summary
            print(summarize_chunk_comparison(orig, recon, sample_size=5))
            print("\n" + "="*80 + "\n")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        x = next(iter(training_loader))
        x = x.to(device)
        _, (categorical_logits, continuous_pred), _ = model(x)
        
        x_recon = model.reconstruct_tensor(categorical_logits, continuous_pred)
        
        # Compare last sample in batch
        original_chunk = x[-1].cpu().detach().numpy()
        reconstructed_chunk = x_recon[-1].cpu().detach().numpy()

        # Compare first 5 tiles
        print("\nFinal Comparison (first 5 tiles):")
        for i in range(5):
            orig_tile = decode_tile_from_tensor(original_chunk[:, i//32, i%32])
            recon_tile = decode_tile_from_tensor(reconstructed_chunk[:, i//32, i%32])
            print(f"\nTile {i}:")
            print(f"  Original:      {format_tile_readable(orig_tile)}")
            print(f"  Reconstructed: {format_tile_readable(recon_tile)}")
    
    # Save final model
    print("\n" + "="*80)
    print("Training complete! Saving final model...")
    print("="*80)
    
    save_final_model(
        model=model,
        optimizer=optimizer,
        results=results,
        config=model_config,
        save_dir="models"
    )
    
    # Also save as final checkpoint
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        results=results,
        epoch=N_UPDATES,
        config=model_config,
        tag="final"
    )
    
    print("\nTraining artifacts saved:")
    print("  - models/vqvae_terraria_*.pt (full model)")
    print("  - models/vqvae_weights_*.pt (weights only)")
    print("  - checkpoints/best_model.pt (best checkpoint)")
    print("  - checkpoints/latest_model.pt (latest checkpoint)")
    print("\nTo resume training:")
    print("  RESUME_CHECKPOINT=checkpoints/latest_model.pt python3 src/main.py")


if __name__ == "__main__":
    print("="*80)
    print("Training VQ-VAE with Categorical Embeddings")
    print("="*80)
    print(f"Device: {device}")
    print(f"Training samples: {len(training_data)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*80)
    print()
    
    train()
