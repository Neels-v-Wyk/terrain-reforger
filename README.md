# terrain-reforger (WIP)
A Terraria terrain generation tool using Discrete Diffusion and Content Aware Inpainting with PyTorch

## Canonical commands

```bash
uv pip install -e .
terrain data prepare --mode consolidated --source worldgen --output data/dataset_optimized.pt
terrain data prepare --mode chunked --source worldgen --output-dir data/cache
terrain data analyze --source-dir worldgen --output src/terraria/natural_ids.py
terrain model train --data data/dataset_optimized.pt --epochs 50
terrain model infer checkpoints/best_model.pt
```

## Notes

- Checkpoints, latest model, and training plots are written under `checkpoints/`.
- Natural block/wall ID mappings are in `src/terraria/natural_ids.py`.
- Generated worlds, datasets, and checkpoints are intentionally ignored by git (see `.gitignore` entries for `worldgen/`, `data/`, and `checkpoints/`).

## Reproducibility

- Generate worlds: `terrain data worldgen --num-worlds 20`
- Refresh natural ID mapping: `terrain data analyze --source-dir worldgen --output src/terraria/natural_ids.py`
- Build per-world cache: `terrain data prepare --mode chunked --source worldgen --output-dir data/cache`
- Train from cache: `terrain model train --data data/cache --disk-mode --cache-size 5 --epochs 50`
