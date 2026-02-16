"""Training command for optimized VQ-VAE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.autoencoder.dataset_cached import CachedTileDataset
from src.autoencoder.dataset_optimized import OptimizedTerrariaTileDataset, PreprocessedTileDataset
from src.autoencoder.samplers import InterleavedFileSampler
from src.autoencoder.vqvae_optimized import VQVAEOptimized, compute_optimized_loss
from src.utils.checkpoint import CheckpointManager, save_final_model
from src.utils.visualization import compare_optimized_tiles, plot_training_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VQ-VAE on Terraria chunks")
    parser.add_argument("--data", type=str, help="Path to preprocessed .pt dataset")
    parser.add_argument("--world", type=str, help="Path to single .wld file (if not using preprocessed data)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--one-pass", action="store_true", help="Train for exactly one pass over the data (epochs=1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--disk-mode", action="store_true", help="Use disk-based loading with LRU cache (saves RAM)")
    parser.add_argument("--cache-size", type=int, default=5, help="LRU cache size (files) for disk mode")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    model_config = {
        "embedding_dim": 32,
        "h_dim": 128,
        "res_h_dim": 64,
        "n_embeddings": 512,
        "beta": 0.25,
    }

    batch_size = args.batch_size
    learning_rate = 2e-4
    num_epochs = 1 if args.one_pass else args.epochs
    checkpoint_interval = 500

    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
            print(f"Loading cached dataset from directory {data_path}...")
            preload_mode = not args.disk_mode
            training_data = CachedTileDataset(
                data_dir=data_path,
                preload=preload_mode,
                max_cache_size=args.cache_size,
            )
            chunk_size = 32
            print(f"  Chunk size: {chunk_size} (assumed)")
        else:
            print(f"Loading preprocessed dataset from {args.data}...")
            training_data = PreprocessedTileDataset(args.data)
            chunk_size = training_data.config["chunk_size"]
            print(f"  Chunk size: {chunk_size}")
    else:
        world_file = args.world if args.world else "worldgen/World_20260211_213447_mBvegqrN.wld"
        region = (0, 0, 8360, 2360)
        chunk_size = 32
        overlap = 16

        print(f"Creating optimized training dataset from {world_file}...")
        training_data = OptimizedTerrariaTileDataset(
            world_path=world_file,
            region=region,
            chunk_size=chunk_size,
            overlap=overlap,
            use_diversity_filter=True,
            min_diversity=0.20,
            deduplicate=True,
        )

    sampler = None
    shuffle = True

    if args.data and Path(args.data).is_dir() and args.disk_mode:
        if isinstance(training_data, CachedTileDataset):
            print(f"Using InterleavedFileSampler to optimize disk access (Cache Size: {args.cache_size})...")
            sampler = InterleavedFileSampler(
                training_data,
                max_cache_size=args.cache_size,
                shuffle_files=True,
            )
            shuffle = False

    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )

    print("\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total training samples: {len(training_data)}")
    print("  Model: Optimized VQ-VAE (9 channels, 218 blocks, 77 walls)")
    print(f"  Embedding dim: {model_config['embedding_dim']}")
    print(f"  Codebook size: {model_config['n_embeddings']}")

    model = VQVAEOptimized(**model_config).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

    results = {
        "loss_vals": [],
        "perplexities": [],
        "block_loss": [],
        "wall_loss": [],
        "liquid_loss": [],
        "continuous_loss": [],
    }

    start_epoch = 0
    best_loss = float("inf")

    resume_path = args.resume or os.environ.get("RESUME_CHECKPOINT")
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from checkpoint: {resume_path}")
        model, loaded_optimizer, results, start_epoch = checkpoint_manager.load_checkpoint(
            model,
            resume_path,
            optimizer,
            str(device),
        )
        if loaded_optimizer is not None:
            optimizer = loaded_optimizer
        best_loss = min(results.get("loss_vals", [float("inf")]))
        print(f"  Resumed from epoch {start_epoch}")

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    model.train()
    update_count = 0
    last_batch: Optional[torch.Tensor] = None
    last_reconstruction: Optional[torch.Tensor] = None

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = {
            "total": [],
            "reconstruction": [],
            "categorical": [],
            "block": [],
            "wall": [],
            "liquid": [],
            "continuous": [],
            "embedding": [],
        }
        epoch_perplexities = []

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            last_batch = batch

            embedding_loss, x_hat, perplexity, logits = model(batch)
            last_reconstruction = x_hat

            total_loss, loss_dict = compute_optimized_loss(batch, x_hat, logits, embedding_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            epoch_perplexities.append(perplexity.item())

            results["loss_vals"].append(loss_dict["total"])
            results["perplexities"].append(perplexity.item())
            results["block_loss"].append(loss_dict["block"])
            results["wall_loss"].append(loss_dict["wall"])
            results["liquid_loss"].append(loss_dict["liquid"])
            results["continuous_loss"].append(loss_dict["continuous"])

            update_count += 1

            if update_count % checkpoint_interval == 0:
                avg_loss = sum(epoch_losses["total"][-checkpoint_interval:]) / min(
                    checkpoint_interval,
                    len(epoch_losses["total"]),
                )
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss

                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    results=results,
                    epoch=epoch,
                    config=model_config,
                    is_best=is_best,
                )
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)

            if batch_idx % 10 == 0:
                recent_loss = sum(epoch_losses["total"][-10:]) / len(epoch_losses["total"][-10:])
                recent_perp = sum(epoch_perplexities[-10:]) / len(epoch_perplexities[-10:])
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {recent_loss:.4f} Perplexity: {recent_perp:.2f}"
                )

        avg_losses = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        avg_perplexity = sum(epoch_perplexities) / len(epoch_perplexities)

        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1} Summary:")
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
        print(f"{'=' * 80}\n")

        plot_training_results(results, "checkpoints")

        if last_batch is not None and last_reconstruction is not None:
            print("Block Comparison (Last Batch):")
            orig_sample = last_batch[-1].cpu().detach().numpy()
            recon_sample = last_reconstruction[-1].cpu().detach().numpy()

            mid_h, mid_w = chunk_size // 2, chunk_size // 2
            for index in range(3):
                orig_tile = orig_sample[:, mid_h, mid_w + index]
                recon_tile = recon_sample[:, mid_h, mid_w + index]
                print("-" * 40)
                print(compare_optimized_tiles(orig_tile, recon_tile, index + 1))
            print("-" * 40 + "\n")

    print("\nTraining complete! Saving final model...")
    final_path = save_final_model(model, optimizer, results, model_config, "checkpoints")
    print(f"Final model saved to: {final_path}")

    print("\nTo load this model for inference:")
    print("  from src.utils.checkpoint import CheckpointManager")
    print("  manager = CheckpointManager('checkpoints')")
    print(f"  model, _, _, _ = manager.load_checkpoint(model, '{final_path}')")


if __name__ == "__main__":
    main()
