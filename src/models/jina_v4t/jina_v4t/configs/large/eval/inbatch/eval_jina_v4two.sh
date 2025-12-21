#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

# Path to the codebase and config file
SRC="/data/UniIR/UniIR-main/src"  # Absolute path to codebse /UniIR/src # <--- Change this to the path of your UniIR/src

# Path to common dir
COMMON_DIR="/data/UniIR/UniIR-main/src/common"

# Path to MBEIR data and UniIR directory where we store the checkpoints, embeddings, etc.
UNIIR_DIR="/data/UniIR/UniIR-main" # <--- Change this to the UniIR directory
MBEIR_DATA_DIR="/data/M-BEIR" # <--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
MODEL="jina_v4t/jina_v4t"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=7  # <--- Change this to the CUDA devices you want to use
NPROC=1 # <--- Change this to the number of GPUs you want to use
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# CD to script directory
cd $COMMON_DIR

# Activate conda environment
# conda activate blip
# conda activate uniir # <--- Change this to the name of your conda environment

# Run Embedding command
CONFIG_PATH="$CONFIG_DIR/embed.yaml"
SCRIPT_NAME="mbeir_embedder.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct False

export MASTER_PORT=29588
export MASTER_ADDR=127.0.0.1
export WORLD_SIZE=1
    # --nproc_per_node=1 \
    # --nnodes=1 \
    # --node_rank=0 \
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
# python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
#     --master_port=$MASTER_PORT \
#     --master_addr=$MASTER_ADDR \
#     --config_path "$CONFIG_PATH" \
#     --uniir_dir "$UNIIR_DIR" \
#     --mbeir_data_dir "$MBEIR_DATA_DIR"

# python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
#     --master_port=$MASTER_PORT \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29506 \
#     --config_path "$CONFIG_PATH" \
#     --uniir_dir "$UNIIR_DIR" \
#     --mbeir_data_dir "$MBEIR_DATA_DIR"

# Activate faiss environment
#conda activate faiss # <--- Change this to the name of your conda environment

# Run Index command
CONFIG_PATH="$CONFIG_DIR/index.yaml"
SCRIPT_NAME="mbeir_retriever.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct False

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_create_index

# Run retrieval command
CONFIG_PATH="$CONFIG_DIR/retrieval.yaml"
SCRIPT_NAME="mbeir_retriever.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct False

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_retrieval