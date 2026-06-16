# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERDES is a retinal pathology diagnostic system using 3D deep learning models on ocular ultrasound videos. It implements a two-stage pipeline: (1) Retinal Detachment (RD) detection, (2) Macula status classification if RD is detected.

## Common Commands

```bash
# Train a model (default unetplusplus, override via model=unet3d, etc.)
python erdes/train.py

# Train with specific experiment config
python erdes/train.py experiment=non_rd_vs_rd/unet3d

# Train on GPU
python erdes/train.py trainer=gpu

# Evaluate checkpoint
python erdes/eval.py ckpt_path=/path/to/checkpoint.ckpt

# Run tests
pytest

# Run specific test file
pytest tests/test_train.py

# Run tests excluding slow tests
pytest -k "not slow"
```

## Architecture

### Training System (Hydra-based)
- **Entry point**: `erdes/train.py` uses Hydra for config-driven training
- **Configs**: `configs/train.yaml` is the main config, composed from `configs/data/`, `configs/model/`, `configs/trainer/`, etc.
- **Model instantiation**: `configs/model/*.yaml` uses `_target_: erdes.models.model_module.ModelModule` with `net._target_: erdes.models.components.factory.build_3d_architecture`
- **5 classification tasks**: non_rd_vs_rd, normal_vs_rd, pvd_vs_rd, macula_detached_vs_intact, normal_vs_pvd

### Model Factory Pattern
All 3D architectures are built via `erdes/models/components/factory.py:build_3d_architecture()`. Supported models:
- `unet3d`, `unetplusplus`, `vnet`, `swinunetr`, `unetr` (encoder-decoder/ViT-based)
- `resnet3d`, `senet` (CNN-based)
- `vit` (Vision Transformer)

### Data Module
`erdes/data/erdes_datamodule.py:ERDESDataModule` loads video data from CSVs specifying train/val/test splits. Videos are resized to configurable `size: [depth, height, width]`.

### Diagnostic Pipeline
`pipe/diagnostic_pipeline.py` provides a two-stage inference pipeline:
- Stage 1: RD detection (has_rd binary classification)
- Stage 2: Macula status (macula_detached) if RD detected

Expects checkpoints in `logs/{rd,md}/{model_name}/checkpoints/` structure.

## Key File Locations
- `erdes/models/model_module.py` - Lightning module wrapping the model
- `erdes/models/components/factory.py` - Architecture factory
- `erdes/data/erdes_datamodule.py` - Data loading
- `pipe/diagnostic_pipeline.py` - Inference pipeline
- `configs/experiment/` - Task-specific configs (one subfolder per task)