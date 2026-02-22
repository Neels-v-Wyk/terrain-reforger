import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List

import lihzahrd.enums as enums

from src.terraria.natural_ids import BLOCK_INDEX_TO_ID, WALL_INDEX_TO_ID


def _fmt(name: str) -> str:
    """Convert UPPER_SNAKE_CASE enum name to Title Case."""
    return name.replace("_", " ").title()


# Human-readable name lookups (keyed by game ID, not our compressed index)
BLOCK_NAMES: Dict[int, str] = {int(bt.value): _fmt(bt.name) for bt in enums.BlockType}
BLOCK_NAMES[0] = "Air"

WALL_NAMES: Dict[int, str] = {int(wt.value): _fmt(wt.name.replace("_UNSAFE", "")) for wt in enums.WallType}
WALL_NAMES[0] = "No Wall"

LIQUID_NAMES: Dict[int, str] = {int(lt.value): _fmt(lt.name.replace("NO_LIQUID", "None")) for lt in enums.LiquidType}

BLOCK_SHAPES: Dict[int, str] = {
    0: "Full",
    1: "Half",
    2: "Top-Right Slope",
    3: "Top-Left Slope",
    4: "Bottom-Right Slope",
    5: "Bottom-Left Slope",
}

def plot_training_results(results: Dict[str, List[float]], save_dir: str):
    """
    Plot training metrics and save to file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Total Loss and Perplexity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(results['loss_vals'], label='Total Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Updates')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Perplexity
    ax2.plot(results['perplexities'], label='Perplexity', color='orange')
    ax2.set_title('Codebook Perplexity')
    ax2.set_xlabel('Updates')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()
    
    # Plot Component Losses
    plt.figure(figsize=(10, 6))
    if 'block_loss' in results:
        plt.plot(results['block_loss'], label='Block Loss', alpha=0.7)
    if 'shape_loss' in results:
        plt.plot(results['shape_loss'], label='Shape Loss', alpha=0.7)
    if 'wall_loss' in results:
        plt.plot(results['wall_loss'], label='Wall Loss', alpha=0.7)
    if 'liquid_loss' in results:
        plt.plot(results['liquid_loss'], label='Liquid Loss', alpha=0.7)
    if 'continuous_loss' in results:
        plt.plot(results['continuous_loss'], label='Continuous Loss', alpha=0.7)
        
    plt.title('Component Losses')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'component_losses.png'))
    plt.close()

def decode_tile(tensor_slice: np.ndarray) -> Dict[str, Any]:
    """
    Convert a single tile (8-channel) to human-readable format.
    
    Args:
        tensor_slice: Array of shape (8,)
        
    Returns:
        Dictionary with human-readable tile information
    """
    # 0: Block Type Index (needs mapping)
    block_idx = int(round(tensor_slice[0]))
    block_id = BLOCK_INDEX_TO_ID.get(block_idx, 0)
    
    # 1: Block Shape (categorical index 0-5)
    block_shape = int(round(tensor_slice[1]))
    block_shape = max(0, min(5, block_shape))
    
    # 2: Wall Type Index
    wall_idx = int(round(tensor_slice[2]))
    wall_id = WALL_INDEX_TO_ID.get(wall_idx, 0)
    
    # 3: Liquid Type (Categorical)
    liquid_type_idx = int(round(tensor_slice[3]))
    
    # 4-7: Wires and Actuator
    wire_red = tensor_slice[4] > 0.5
    wire_blue = tensor_slice[5] > 0.5
    wire_green = tensor_slice[6] > 0.5
    actuator = tensor_slice[7] > 0.5
    
    # Format names
    block_name = BLOCK_NAMES.get(block_id, f"Unknown Block ({block_id})")
    wall_name = WALL_NAMES.get(wall_id, f"Unknown Wall ({wall_id})")
    shape_name = BLOCK_SHAPES.get(block_shape, "Full")
    
    # Liquid name mapping (0=None, 1=Water, 2=Lava, 3=Honey, 4=Shimmer)
    liquid_map = {0: "None", 1: "Water", 2: "Lava", 3: "Honey", 4: "Shimmer"}
    liquid_name = liquid_map.get(liquid_type_idx, "None")
    
    
    # Format names - Reusing logic from above, but cleaner
    # ... actually the above simple return is sufficient for basic debug.
    # But let's keep the structured dictionary return for existing consumers.
    
    result = {
        'block': {
            'index': block_idx,
            'id': block_id,
            'name': block_name,
            'shape': shape_name,
            'active': block_id > 0
        },
        'wall': {
            'index': wall_idx,
            'id': wall_id,
            'name': wall_name,
            'active': wall_id > 0
        },
        'liquid': {
            'present': liquid_type_idx > 0,
            'type_idx': liquid_type_idx,
            'name': liquid_name
        },
        'wiring': {
            'red': wire_red,
            'blue': wire_blue,
            'green': wire_green,
            'actuator': actuator
        }
    }
    
    return result

def format_tile(tile_data: Dict[str, Any]) -> str:
    """Format tile data into string."""
    parts = []
    
    # Handle the fact that decode_tile might return flattened or structured dict
    # If flattened (my first edit), convert to structured
    if "block_id" in tile_data:
        # Convert flat to structured
        tile_data = {
             'block': {'active': tile_data['block_id'] > 0, 'name': tile_data['block'], 'shape': tile_data['shape']},
             'wall': {'active': tile_data['wall_id'] > 0, 'name': tile_data['wall']},
             'liquid': {'present': tile_data['liquid'] != 'None', 'name': tile_data['liquid']},
             'wiring': {'red': 'R:1' in tile_data['wires'], 'blue': 'B:1' in tile_data['wires'], 'green': 'G:1' in tile_data['wires']}
        }
    
    if tile_data['block']['active']:
        s = f"Block: {tile_data['block']['name']}"
        if tile_data['block']['shape'] != 'Full':
            s += f" ({tile_data['block']['shape']})"
        parts.append(s)

        
    if tile_data['wall']['active']:
        parts.append(f"Wall: {tile_data['wall']['name']}")
        
    if tile_data['liquid']['present']:
        parts.append(f"Liquid: {tile_data['liquid']['name']}")
        
    wires = []
    if tile_data['wiring']['red']: wires.append('Red')
    if tile_data['wiring']['blue']: wires.append('Blue')
    if tile_data['wiring']['green']: wires.append('Green')
    if wires:
        parts.append(f"Wires: {'+'.join(wires)}")
        
    if tile_data['wiring']['actuator']:
        parts.append("Actuator")
        
    return " | ".join(parts) if parts else "Empty Air"

def compare_tiles(original: np.ndarray, reconstructed: np.ndarray, index: int) -> str:
    """
    Compare original and reconstructed 8-channel tiles.
    """
    orig_data = decode_tile(original)
    recon_data = decode_tile(reconstructed)
    
    orig_str = format_tile(orig_data)
    recon_str = format_tile(recon_data)
    
    status = "✓" if orig_str == recon_str else "✗"
    
    return (f"Tile {index}: [{status}]\n"
            f"  Orig : {orig_str}\n"
            f"  Recon: {recon_str}")
