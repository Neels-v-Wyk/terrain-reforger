"""Training command for VQ-VAE."""

from __future__ import annotations

import argparse
import os
import math
import random
from pathlib import Path
from typing import Optional, Sequence, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.autoencoder.dataset_cached import CachedTileDataset
from src.autoencoder.dataset import TerrariaTileDataset, PreprocessedTileDataset
from src.autoencoder.samplers import InterleavedFileSampler
from src.autoencoder.vqvae import VQVAE, compute_loss
from src.utils.checkpoint import CheckpointManager, save_final_model
from src.utils.device import get_device
from src.utils.visualization import compare_tiles, plot_training_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VQ-VAE on Terraria chunks")
    parser.add_argument("--data", type=str, help="Path to preprocessed .pt dataset directory or file")
    parser.add_argument("--world", type=str, help="Path to single .wld file or directory of worlds")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--one-pass", action="store_true", help="Train for exactly one pass over the data (epochs=1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data to use for validation (default: 0.1)")
    parser.add_argument("--disk-mode", action="store_true", help="Use disk-based loading with LRU cache (saves RAM)")
    parser.add_argument("--cache-size", type=int, default=5, help="LRU cache size (files) for disk mode")
    # EMA quantizer options
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA codebook updates (use vanilla VQ-VAE)")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay rate (default: 0.99)")
    parser.add_argument(
        "--ema-reset-multiplier",
        type=float,
        default=0.5,
        help="Dead-code reset threshold as multiplier of uniform usage (default: 0.5)",
    )
    parser.add_argument(
        "--ema-reset-interval",
        type=int,
        default=500,
        help="How often (updates) to check/reset dead codes (default: 500)",
    )
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight (default: 0.25)")
    parser.add_argument(
        "--block-loss-weighted",
        action="store_true",
        help="Use inverse-frequency class weighting for block-type cross entropy",
    )
    parser.add_argument(
        "--block-weight-min",
        type=float,
        default=0.5,
        help="Minimum class weight clamp for weighted block loss (default: 0.5)",
    )
    parser.add_argument(
        "--block-weight-max",
        type=float,
        default=5.0,
        help="Maximum class weight clamp for weighted block loss (default: 5.0)",
    )
    parser.add_argument(
        "--metrics-stride",
        type=int,
        default=50,
        help="Store metrics every N updates to reduce checkpoint JSON size (default: 50)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute training from a pre-populated Namespace. Called by main() and the CLI."""
    device = get_device()
    print(f"Using device: {device}")

    model_config = {
        "embedding_dim": 32,
        "h_dim": 128,
        "res_h_dim": 64,
        "n_embeddings": 512,
        "beta": args.beta,
        # EMA settings to prevent codebook collapse
        "use_ema": not args.no_ema,
        "ema_decay": args.ema_decay,
        # Reinitialize codes used less than (multiplier / n_embeddings)
        "ema_reset_threshold": args.ema_reset_multiplier,
        "ema_reset_interval": args.ema_reset_interval,
    }

    batch_size = args.batch_size
    learning_rate = 2e-4
    num_epochs = 1 if args.one_pass else args.epochs
    checkpoint_interval = 500

    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
            print(f"Loading cached dataset from directory {data_path}...")
            all_files = sorted(list(data_path.glob("*.pt")))
            if not all_files:
                raise FileNotFoundError(f"No .pt files found in {data_path}")
            
            # Split files
            random.shuffle(all_files)
            val_count = max(1, int(len(all_files) * args.val_split))
            val_files = all_files[:val_count]
            train_files = all_files[val_count:]
            
            print(f"  Total files: {len(all_files)}")
            print(f"  Training files: {len(train_files)}")
            print(f"  Validation files: {len(val_files)}")
            
            preload_mode = not args.disk_mode
            train_dataset = CachedTileDataset(
                file_paths=train_files,
                preload=preload_mode,
                max_cache_size=args.cache_size,
                verbose=True
            )
            val_dataset = CachedTileDataset(
                file_paths=val_files,
                preload=preload_mode, 
                max_cache_size=args.cache_size,
                verbose=False
            )
            chunk_size = 32
            print(f"  Chunk size: {chunk_size} (assumed)")
        else:
            print(f"Loading preprocessed dataset from {args.data}...")
            full_dataset = PreprocessedTileDataset(args.data)
            chunk_size = full_dataset.config["chunk_size"]
            print(f"  Chunk size: {chunk_size}")
            
            # Split indices
            indices = list(range(len(full_dataset)))
            random.shuffle(indices)
            val_count = int(len(indices) * args.val_split)
            val_indices = indices[:val_count]
            train_indices = indices[val_count:]
            
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")

    else:
        # World loading logic
        if args.world:
            world_path = Path(args.world)
            if world_path.is_dir():
                worlds = sorted(list(world_path.glob("*.wld")))
            elif world_path.is_file():
                worlds = [world_path]
            else:
                raise FileNotFoundError(f"World path not found: {world_path}")
        else:
            worldgen_dir = Path("worldgen")
            worlds = sorted(worldgen_dir.glob("*.wld")) if worldgen_dir.exists() else []
            if not worlds:
                print("No .wld files found. Specify --world or run 'terrain data worldgen' first.")
                return
        
        print(f"Found {len(worlds)} worlds.")
        random.shuffle(worlds)
        
        # Split worlds
        if len(worlds) > 1:
            val_count = max(1, int(len(worlds) * args.val_split))
            val_worlds = worlds[:val_count]
            train_worlds = worlds[val_count:]
        else:
            print("Warning: Only 1 world found. Using it for both training and validation (data leakage!).")
            train_worlds = worlds
            val_worlds = worlds
            
        print(f"  Training worlds: {len(train_worlds)}")
        print(f"  Validation worlds: {len(val_worlds)}")

        region = (0, 0, 8360, 2360)
        chunk_size = 32
        overlap = 16

        def create_dataset_from_worlds(world_files):
            datasets = []
            for w in world_files:
                try:
                    ds = TerrariaTileDataset(
                        world_path=w,
                        region=region,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        use_diversity_filter=True,
                        min_diversity=0.20,
                        deduplicate=True,
                    )
                    datasets.append(ds)
                except Exception as e:
                    print(f"Error loading world {w}: {e}")
            if not datasets:
                raise ValueError("Could not load any datasets from provided worlds")
            return ConcatDataset(datasets)

        print("Creating training dataset...")
        train_dataset = create_dataset_from_worlds(train_worlds)
        print("Creating validation dataset...")
        val_dataset = create_dataset_from_worlds(val_worlds)

    train_sampler = None
    train_shuffle = True

    # Use specialized sampler only for CachedTileDataset in disk mode
    if isinstance(train_dataset, CachedTileDataset) and args.disk_mode:
        print(f"Using InterleavedFileSampler for training (Cache Size: {args.cache_size})...")
        train_sampler = InterleavedFileSampler(
            train_dataset,
            max_cache_size=args.cache_size,
            shuffle_files=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total training samples: {len(train_dataset)}")
    print(f"  Total validation samples: {len(val_dataset)}")
    print("  Model: VQ-VAE (8 channels)")
    print(f"  Embedding dim: {model_config['embedding_dim']}")
    print(f"  Codebook size: {model_config['n_embeddings']}")
    print(f"  Quantizer: {'EMA' if model_config['use_ema'] else 'Vanilla'} (beta={model_config['beta']})")
    print(
        f"  Block loss weighting: {'enabled' if args.block_loss_weighted else 'disabled'} "
        f"(min={args.block_weight_min}, max={args.block_weight_max})"
    )
    if model_config['use_ema']:
        dead_usage_rate = model_config['ema_reset_threshold'] / model_config['n_embeddings']
        print(f"  EMA decay: {model_config['ema_decay']}")
        print(
            f"  Dead code reset threshold: {model_config['ema_reset_threshold']:.2f} x uniform "
            f"({dead_usage_rate * 100:.4f}% usage rate)"
        )
        print(f"  Dead code reset interval: every {model_config['ema_reset_interval']} updates")
    print(f"  Metrics stride: every {args.metrics_stride} updates")

    model = VQVAE(**model_config).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

    results = {
        "loss_vals": [],
        "perplexities": [],
        "block_loss": [],
        "shape_loss": [],
        "wall_loss": [],
        "liquid_loss": [],
        "continuous_loss": [],
        "n_updates": 0,
    }

    start_epoch = 0
    best_loss = float("inf")
    update_count = 0

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
        update_count = int(results.get("n_updates", 0))
        results.setdefault("shape_loss", [])
        results.setdefault("n_updates", update_count)
        print(f"  Resumed from epoch {start_epoch}")

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    model.train()
    last_batch: Optional[torch.Tensor] = None
    last_reconstruction: Optional[torch.Tensor] = None

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = {
            "total": [],
            "reconstruction": [],
            "categorical": [],
            "block": [],
            "shape": [],
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

            total_loss, loss_dict = compute_loss(
                batch,
                x_hat,
                logits,
                embedding_loss,
                block_loss_weighted=args.block_loss_weighted,
                block_weight_min=args.block_weight_min,
                block_weight_max=args.block_weight_max,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            epoch_perplexities.append(perplexity.item())

            update_count += 1
            results["n_updates"] = update_count

            should_store_metrics = (update_count % args.metrics_stride == 0) or (batch_idx == len(train_loader) - 1)
            if should_store_metrics:
                results["loss_vals"].append(loss_dict["total"])
                results["perplexities"].append(perplexity.item())
                results["block_loss"].append(loss_dict["block"])
                results["shape_loss"].append(loss_dict["shape"])
                results["wall_loss"].append(loss_dict["wall"])
                results["liquid_loss"].append(loss_dict["liquid"])
                results["continuous_loss"].append(loss_dict["continuous"])

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
        print(f"      Shape: {avg_losses['shape']:.4f}")
        print(f"      Wall: {avg_losses['wall']:.4f}")
        print(f"      Liquid: {avg_losses['liquid']:.4f}")
        print(f"    Continuous: {avg_losses['continuous']:.4f}")
        print(f"  Embedding: {avg_losses['embedding']:.4f}")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Codebook usage (perplexity-based): {(avg_perplexity / model_config['n_embeddings']) * 100:.1f}%")
        
        # Show detailed codebook stats if using EMA quantizer
        usage_stats = model.get_codebook_usage_stats()
        if usage_stats:
            print(f"  Codebook stats (EMA tracked):")
            print(f"    Active codes: {usage_stats['active_codes']}/{model_config['n_embeddings']} ({usage_stats['usage_percent']:.1f}%)")
            print(f"    Dead codes: {usage_stats['dead_codes']}")
            print(
                f"    Active > 1/n: {usage_stats['active_codes_above_uniform']}/{model_config['n_embeddings']} | "
                f"Active > 0.5/n: {usage_stats['active_codes_above_half_uniform']}/{model_config['n_embeddings']}"
            )
            print(f"    Usage entropy (normalized): {usage_stats['entropy_normalized']:.3f}")
            print(
                f"    Top code share: {usage_stats['top1_share'] * 100:.2f}% | "
                f"Top-5: {usage_stats['top5_share'] * 100:.2f}% | "
                f"Top-10: {usage_stats['top10_share'] * 100:.2f}%"
            )
        print(f"{'=' * 80}\n")

        # Validation Loop
        print("Running Validation...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                emb_loss_val, x_hat_val, perp_val, logits_val = model(val_batch)
                
                loss_val, loss_dict_val = compute_loss(
                    val_batch, 
                    x_hat_val, 
                    logits_val, 
                    emb_loss_val,
                    block_loss_weighted=False 
                )
                val_losses.append(loss_dict_val['total'])
        
        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        results.setdefault("val_loss", []).append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"{'=' * 80}\n")
        
        model.train() # Switch back to train mode

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
                print(compare_tiles(orig_tile, recon_tile, index + 1))
            print("-" * 40 + "\n")

    print("\nTraining complete! Saving final model...")
    final_path = save_final_model(model, optimizer, results, model_config, "checkpoints")
    print(f"Final model saved to: {final_path}")

    print("\nTo load this model for inference:")
    print("  from src.utils.checkpoint import CheckpointManager")
    print("  manager = CheckpointManager('checkpoints')")
    print(f"  model, _, _, _ = manager.load_checkpoint(model, '{final_path}')")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
