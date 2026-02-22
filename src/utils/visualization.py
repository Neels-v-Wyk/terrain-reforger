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

def decode_optimized_tile(tensor_slice: np.ndarray) -> Dict[str, Any]:
    """
    Convert a single optimized tile (9-channel) to human-readable format.
    
    Args:
        tensor_slice: Array of shape (9,)
        
    Returns:
        Dictionary with human-readable tile information
    """
    # 0: Block Type Index (needs mapping)
    block_idx = int(round(tensor_slice[0]))
    block_id = BLOCK_INDEX_TO_ID.get(block_idx, 0)
    
    # 1: Block Shape (continuous/categorical)
    block_shape = int(round(float(tensor_slice[1]) * 5.0))
    
    # 2: Wall Type Index (needs mapping)
    wall_idx = int(round(tensor_slice[2]))
    wall_id = WALL_INDEX_TO_ID.get(wall_idx, 0)
    
    # 3: Liquid Present (binary)
    liquid_present = tensor_slice[3] > 0.5
    
    # 4: Liquid Type Index (0-4)
    # 0=None, 1=Water, 2=Lava, 3=Honey, 4=Shimmer
    liquid_type = int(round(tensor_slice[4]))
    
    # 5-8: Wires/Actuator
    wire_red = tensor_slice[5] > 0.5
    wire_blue = tensor_slice[6] > 0.5
    wire_green = tensor_slice[7] > 0.5
    actuator = tensor_slice[8] > 0.5
    
    result = {
        'block': {
            'index': block_idx,
            'id': block_id,
            'name': BLOCK_NAMES.get(block_id, f"Unknown Block {block_id}"),
            'shape': BLOCK_SHAPES.get(block_shape, str(block_shape)),
            'active': block_id > 0
        },
        'wall': {
            'index': wall_idx,
            'id': wall_id,
            'name': WALL_NAMES.get(wall_id, f"Unknown Wall {wall_id}"),
            'active': wall_id > 0
        },
        'liquid': {
            'present': liquid_present,
            'type_idx': liquid_type,
            'name': LIQUID_NAMES.get(liquid_type, "None") if liquid_present else "None"
        },
        'wiring': {
            'red': wire_red,
            'blue': wire_blue,
            'green': wire_green,
            'actuator': actuator
        }
    }
    
    return result

def format_optimized_tile(tile_data: Dict[str, Any]) -> str:
    """Format optimized tile data into string."""
    parts = []
    
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

def compare_optimized_tiles(original: np.ndarray, reconstructed: np.ndarray, index: int) -> str:
    """
    Compare original and reconstructed 9-channel tiles.
    """
    orig_data = decode_optimized_tile(original)
    recon_data = decode_optimized_tile(reconstructed)
    
    orig_str = format_optimized_tile(orig_data)
    recon_str = format_optimized_tile(recon_data)
    
    status = "✓" if orig_str == recon_str else "✗"
    
    return (f"Tile {index}: [{status}]\n"
            f"  Orig : {orig_str}\n"
            f"  Recon: {recon_str}")
