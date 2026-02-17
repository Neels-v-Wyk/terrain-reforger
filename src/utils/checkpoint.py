"""
Model checkpoint saving and loading utilities.

Handles saving/loading of:
- Model state (weights, architecture)
- Optimizer state (for resuming training)
- Training metrics and history
- Configuration/hyperparameters
"""

import torch
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    
    Creates organized checkpoint directory with:
    - Model weights
    - Optimizer state
    - Training history
    - Configuration metadata
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Args:
            checkpoint_dir: Base directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        results: Dict[str, Any],
        epoch: int,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: str = ""
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            results: Training results/metrics dictionary
            epoch: Current epoch/update number
            config: Model configuration dictionary
            is_best: Whether this is the best model so far
            tag: Optional tag for checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint name
        if tag:
            checkpoint_name = f"checkpoint_{tag}_epoch{epoch}_{timestamp}.pt"
        else:
            checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'timestamp': timestamp,
        }
        
        if config:
            checkpoint['config'] = config
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # If best model, save a copy as "best_model.pt"
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        # Also save latest as "latest_model.pt"
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        # Save training metrics as JSON (human-readable)
        metrics_path = checkpoint_path.with_suffix('.json')
        self._save_metrics_json(results, metrics_path, epoch, config)
        
        return checkpoint_path
    
    def _save_metrics_json(
        self,
        results: Dict[str, Any],
        path: Path,
        epoch: int,
        config: Optional[Dict[str, Any]]
    ):
        """Save metrics in human-readable JSON format."""
        # Convert numpy arrays to lists
        metrics_json = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0 and isinstance(value[0], (np.ndarray, np.number)):
                    metrics_json[key] = [float(v) for v in value]
                else:
                    metrics_json[key] = value
            else:
                metrics_json[key] = value
        
        data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_json,
            'n_updates': int(results.get('n_updates', 0)),
        }
        
        if config:
            data['config'] = config
        
        # Calculate summary statistics
        if 'loss_vals' in results and len(results['loss_vals']) > 0:
            recent = results['loss_vals'][-100:]  # Last 100 updates
            data['summary'] = {
                'final_loss': float(results['loss_vals'][-1]),
                'avg_loss_last_100': float(np.mean(recent)),
                'min_loss': float(np.min(results['loss_vals'])),
                'final_perplexity': float(results['perplexities'][-1]) if 'perplexities' in results else None,
                'n_updates': int(results.get('n_updates', 0)),
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str | Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any], int]:
        """
        Load a checkpoint.
        
        Args:
            model: Model instance to load weights into
            checkpoint_path: Path to checkpoint file (or None for latest)
            optimizer: Optimizer to load state into (optional)
            device: Device to map tensors to
            
        Returns:
            Tuple of (model, optimizer, results, epoch)
        """
        # Use latest if no path specified
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest_model.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model state loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        
        # Return model, optimizer, results, and epoch
        results = checkpoint.get('results', {})
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Checkpoint loaded successfully")
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")
        
        return model, optimizer, results, epoch
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Don't delete best or latest
        special_checkpoints = {
            self.checkpoint_dir / "best_model.pt",
            self.checkpoint_dir / "latest_model.pt"
        }
        
        # Sort by modification time (newest first)
        checkpoints = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Delete old checkpoints
        deleted = 0
        for checkpoint in checkpoints[keep_last_n:]:
            if checkpoint not in special_checkpoints:
                checkpoint.unlink()
                # Also delete corresponding JSON
                json_path = checkpoint.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                deleted += 1
        
        if deleted > 0:
            print(f"Cleaned up {deleted} old checkpoint(s)")


def save_final_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    results: Dict[str, Any],
    config: Dict[str, Any],
    save_dir: str | Path = "models"
) -> Path:
    """
    Save final trained model with metadata.
    
    This is a simplified save for production/inference use.
    
    Args:
        model: Trained model
        optimizer: Optimizer (for potential resume)
        results: Training results
        config: Model configuration
        save_dir: Directory to save to
        
    Returns:
        Path to saved model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"vqvae_terraria_{timestamp}.pt"
    model_path = save_dir / model_name
    
    # Save complete package
    save_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'training_results': results,
        'timestamp': timestamp,
        'version': '1.0',
    }
    
    torch.save(save_data, model_path)
    
    # Also save just model weights for lighter loading
    weights_path = save_dir / f"vqvae_weights_{timestamp}.pt"
    torch.save(model.state_dict(), weights_path)
    
    # Save metadata as JSON
    metadata_path = save_dir / f"vqvae_metadata_{timestamp}.json"
    metadata = {
        'timestamp': timestamp,
        'config': config,
        'final_metrics': {
            'loss': float(results['loss_vals'][-1]) if results['loss_vals'] else None,
            'perplexity': float(results['perplexities'][-1]) if results['perplexities'] else None,
            'total_updates': results.get('n_updates', 0),
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL MODEL SAVED")
    print(f"{'='*80}")
    print(f"Full model:    {model_path}")
    print(f"Weights only:  {weights_path}")
    print(f"Metadata:      {metadata_path}")
    print(f"{'='*80}\n")
    
    return model_path


def load_model_for_inference(
    model: torch.nn.Module,
    model_path: str | Path,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load a trained model for inference.
    
    Args:
        model: Model instance (with correct architecture)
        model_path: Path to saved model file
        device: Device to load to
        
    Returns:
        Loaded model in eval mode
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load model data
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict (handle both checkpoint and weights-only files)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Print config if available
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print(f"Config: {checkpoint['config']}")
    
    return model
