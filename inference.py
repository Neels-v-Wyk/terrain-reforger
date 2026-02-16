#!/usr/bin/env python3
"""Load a trained optimized model and run reconstruction inference."""

import argparse
import torch
from pathlib import Path

from src.autoencoder.dataset_optimized import OptimizedTerrariaTileDataset
from src.autoencoder.vqvae_optimized import VQVAEOptimized, compute_optimized_loss
from src.utils.checkpoint import load_model_for_inference
from src.utils.visualization import compare_optimized_tiles


def main():
    parser = argparse.ArgumentParser(description="Run inference with optimized VQ-VAE")
    parser.add_argument("model", nargs="?", help="Checkpoint/model path")
    parser.add_argument("--world", default="worldgen/World_20260211_213447_mBvegqrN.wld", help="Path to .wld file")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), default=[1500, 1000, 1600, 1100], help="Inference region")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size")
    parser.add_argument("--samples", type=int, default=3, help="Number of chunks to inspect")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            latest = checkpoints_dir / "latest_model.pt"
            best = checkpoints_dir / "best_model.pt"
            if latest.exists():
                model_path = str(latest)
            elif best.exists():
                model_path = str(best)
            else:
                print("No optimized checkpoints found in checkpoints/")
                return
        else:
            print("checkpoints directory not found. Please train an optimized model first.")
            return
    
    print(f"\n{'='*80}")
    print("LOADING MODEL FOR INFERENCE")
    print(f"{'='*80}")
    
    # Create optimized model instance (must match training config)
    model_config = {
        "embedding_dim": 32,
        "h_dim": 128,
        "res_h_dim": 64,
        "n_embeddings": 512,
        "beta": 0.25,
    }

    model = VQVAEOptimized(**model_config)
    
    # Load trained weights
    model = load_model_for_inference(model, model_path, str(device))
    
    print(f"{'='*80}\n")
    
    # Load test data
    print("Loading test data...")
    test_data = OptimizedTerrariaTileDataset(
        world_path=args.world,
        region=tuple(args.region),
        chunk_size=args.chunk_size,
        overlap=0,
        use_diversity_filter=False,
        deduplicate=False,
    )
    
    print(f"Test chunks: {len(test_data)}")
    
    # Run inference on a few chunks
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print(f"{'='*80}\n")
    
    model.eval()
    with torch.no_grad():
        for i in range(min(args.samples, len(test_data))):
            print(f"\nTest Chunk {i+1}")
            print("-" * 80)
            
            x = test_data[i].unsqueeze(0).to(device)
            embedding_loss, x_hat, perplexity, logits = model(x)
            total_loss, loss_dict = compute_optimized_loss(x, x_hat, logits, embedding_loss)

            print(f"Perplexity: {perplexity.item():.2f}")
            print(f"Total loss: {total_loss.item():.4f}")
            print(
                f"Component losses - block: {loss_dict['block']:.4f}, "
                f"wall: {loss_dict['wall']:.4f}, liquid: {loss_dict['liquid']:.4f}, "
                f"continuous: {loss_dict['continuous']:.4f}"
            )

            orig_sample = x[0].cpu().detach().numpy()
            recon_sample = x_hat[0].cpu().detach().numpy()
            mid_h, mid_w = args.chunk_size // 2, args.chunk_size // 2
            for tile_idx in range(3):
                orig_tile = orig_sample[:, mid_h, mid_w + tile_idx]
                recon_tile = recon_sample[:, mid_h, mid_w + tile_idx]
                print(compare_optimized_tiles(orig_tile, recon_tile, tile_idx + 1))
            print()
    
    print(f"{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}\n")
    
    print("✓ Model loaded and tested successfully")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Device: {device}")
    print(f"✓ Codebook size: {model_config['n_embeddings']}")


if __name__ == "__main__":
    main()
