#!/bin/bash
#SBATCH --job-name=nex_p3_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=logs/0001_nex_patch3_smoke_%j.log
#SBATCH --error=logs/0001_nex_patch3_smoke_%j.err

set -euo pipefail

cd /home/vivekfei/EXP/NeX
source ~/.bashrc
conda activate nex

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.

echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "PWD=$(pwd)"
echo "CONDA_PREFIX=$CONDA_PREFIX"
nvidia-smi || true

python train.py \
  -scene dataset/full_data_4x_fresh_colmap/CanClip \
  -model_dir 0001_CanClip_NeX_PATCH3_smoke_val5 \
  -restart \
  -val_image_interval 5 \
  -epochs 1 \
  -tb_savempi 1 \
  -checkpoint 1 \
  -tb_saveimage 0 \
  -tb_toc 20 \
  -ray 1024 \
  -layers 4 \
  -sublayers 2 \
  -hidden 64 \
  -mlp 1 \
  -llff_width 504 \
  -num_workers 0 \
  -cv2resize \
  -no_video \
  -no_webgl

echo
echo "================================================================================"
echo "RESULT TREE"
echo "================================================================================"
tree results/0001_CanClip_NeX_PATCH3_smoke_val5 -L 4

echo
echo "================================================================================"
echo "METRICS"
echo "================================================================================"
cat results/0001_CanClip_NeX_PATCH3_smoke_val5/metrics.csv

echo
echo "================================================================================"
echo "VAL PER IMAGE"
echo "================================================================================"
cat results/0001_CanClip_NeX_PATCH3_smoke_val5/val_per_image_metrics.csv

echo
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
cat results/0001_CanClip_NeX_PATCH3_smoke_val5/logs/summary.json

echo
echo "================================================================================"
echo "VALIDATION IMAGES"
echo "================================================================================"
find results/0001_CanClip_NeX_PATCH3_smoke_val5/images -type f | sort

echo "DONE"
