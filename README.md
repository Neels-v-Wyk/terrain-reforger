# terrain-reforger

VQ-VAE model for Terraria terrain generation. Trains on world data and exports reconstructions as TEdit schematics.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Requires tModLoader/Terraria for world generation (see `src/scripts/worldgen.sh`).

## Usage

### Generate and prepare data

```bash
# Generate worlds
terrain data worldgen --num-worlds 20

# Prepare training dataset
terrain data prepare

# Optional: analyze worlds to regenerate natural ID mappings
terrain data analyze
```

### Train model

```bash
# Basic training
terrain model train --data data/dataset.pt

# With recommended options
terrain model train --data data/dataset.pt --lr 2e-4 --block-loss-weighted --epochs 50

# Resume from checkpoint
terrain model train --data data/dataset.pt --resume checkpoints/latest_model.pt
```

### Diagnostic tools

```bash
# Quick validation on subset of data
terrain model diagnose --data data/dataset.pt

# Find optimal learning rate
terrain model lr-find --data data/dataset.pt
```

### Export results

```bash
# View reconstruction quality
terrain model infer

# Export as TEdit schematics
terrain model export --x 1500 --y 500 --width 64 --height 64
```

## Commands

**Data preparation:**
- `terrain data worldgen` - Generate Terraria worlds
- `terrain data prepare` - Extract and prepare training dataset
- `terrain data analyze` - Analyze worlds and update ID mappings

**Model:**
- `terrain model train` - Train VQ-VAE model
- `terrain model diagnose` - Run diagnostics on data subset
- `terrain model lr-find` - Learning rate range test
- `terrain model infer` - Test reconstruction quality
- `terrain model export` - Export to TEdit schematics

Run any command with `--help` for options.

## Files

- `checkpoints/` - Model checkpoints and training plots
- `worldgen/` - Generated world files
- `data/` - Prepared datasets
- `exports/` - TEdit schematic exports

