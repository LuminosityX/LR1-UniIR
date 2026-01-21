# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniIR (Universal Multimodal Information Retriever) - ECCV 2024 paper implementation. Trains and evaluates universal multimodal information retrieval models that handle heterogeneous queries (text, image, or both) to retrieve from heterogeneous candidate pools across modalities.

## Architecture

```
/src/
├── models/                    # Model implementations
│   ├── uniir_clip/           # CLIP-based models
│   │   ├── clip_scorefusion/ # Score-level fusion (CLIP-SF)
│   │   ├── clip_featurefusion/ # Feature-level fusion (CLIP-FF)
│   │   └── engine.py
│   ├── uniir_blip/           # BLIP-based models
│   │   ├── backbone/         # BLIP backbone (med.py, blip.py, vit.py)
│   │   ├── blip_featurefusion/
│   │   ├── blip_scorefusion/
│   │   └── engine.py
│   └── jina_v4t/             # Jina V4 model variant
├── data/                      # Data handling
│   ├── mbeir_dataset.py      # Dataset loading (JSONL format)
│   ├── mbeir_data_utils.py
│   └── preprocessing/        # Dataset-specific preprocessors (11 datasets)
└── common/                    # Shared utilities
    ├── mbeir_embedder.py     # Distributed embedding generation
    ├── mbeir_retriever.py    # FAISS indexing & retrieval
    ├── config_updater.py     # YAML config management
    └── dist_utils.py         # Distributed training utilities
```

## Environment Setup

```bash
# Training environment
conda env create -f src/models/uniir_env.yml
conda activate uniir

# Evaluation environment (for FAISS)
conda env create -f src/common/faiss_env.yml
conda activate faiss
```

## Common Commands

### Training

Training scripts are located in: `src/models/{MODEL}/configs_scripts/{SIZE}/train/{EXP_NAME}/`

Example CLIP-SF Large training:
```bash
cd src/models/uniir_clip/clip_scorefusion/configs_scripts/large/train/inbatch/
# Edit run_inbatch.sh to set UNIIR_DIR, MBEIR_DATA_DIR, SRC_DIR
bash run_inbatch.sh
```

Entry points:
- `src/models/uniir_clip/clip_scorefusion/train.py`
- `src/models/uniir_clip/clip_featurefusion/train.py`
- `src/models/uniir_blip/train.py`
- `src/models/jina_v4t/train.py`

### Evaluation Pipeline

Evaluation scripts: `src/models/{MODEL}/configs_scripts/{SIZE}/eval/{EXP_NAME}/`

Three-step process:
1. Generate embeddings: `mbeir_embedder.py` (distributed)
2. Create FAISS index: `mbeir_retriever.py --enable_create_index`
3. Run retrieval: `mbeir_retriever.py --enable_retrieval`

```bash
cd src/models/uniir_clip/clip_scorefusion/configs_scripts/large/eval/inbatch/
bash run_eval_pipeline_inbatch.sh
```

## Configuration

All models use OmegaConf YAML configuration. Key sections:
- `experiment`: instruct_status, exp_name, WandB settings
- `data_config`: image_size, negatives config, paths to JSONL files
- `trainer_config`: learning rate, epochs, warmup, gradient accumulation
- `model`: architecture params (e.g., `clip_vision_model_name: "ViT-L/14"`)
- `dist_config`: distributed training setup

Config path pattern: `models/{MODEL}/configs_scripts/{SIZE}/{MODE}/{EXP_NAME}/`

## Required Path Configuration

All training/eval scripts require editing three paths:
1. `UNIIR_DIR` - checkpoint and embedding storage
2. `MBEIR_DATA_DIR` - M-BEIR dataset location
3. `SRC_DIR` - repository src/ directory

Environment variables needed in `.env`:
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`

## Data Pipeline

Dataset: M-BEIR benchmark (5.6M heterogeneous candidates)
- Download from: https://huggingface.co/datasets/TIGER-Lab/M-BEIR
- Format: JSONL for queries and candidate pools
- Modes: "Instruct" and "NoInstruct"
- Splits: train, val, test

Dataset class: `MBEIRDatasetBase` in `src/data/mbeir_dataset.py`

## Key Patterns

- Distributed training with PyTorch DDP
- Mixed precision training with `torch.cuda.amp`
- Cosine annealing LR scheduler with warmup
- WandB for experiment tracking
- Checkpoints: `{UNIIR_DIR}/checkpoint/{MODEL}/{SIZE}/{instruct_status}/{exp_name}/`
