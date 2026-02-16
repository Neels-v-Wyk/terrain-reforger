# terrain-reforger (WIP)
A Terraria terrain generation tool using Discrete Diffusion and Content Aware Inpainting with PyTorch

## Notes

- Checkpoints, latest model, and training plots are written under `checkpoints/`.
- Natural block/wall ID mappings are in `src/terraria/natural_ids.py`.
- Generated worlds, datasets, and checkpoints are intentionally ignored by git (see `.gitignore` entries for `worldgen/`, `data/`, and `checkpoints/`).

## Reproducibility

- Generate worlds: `terrain data worldgen --num-worlds 20`
- Refresh natural ID mapping: `terrain data analyze --source-dir worldgen --output src/terraria/natural_ids.py`
- Build per-world cache: `terrain data prepare --mode chunked --source worldgen --output-dir data/cache`
- Train from cache: `terrain model train --data data/cache --disk-mode --cache-size 5 --epochs 50`
