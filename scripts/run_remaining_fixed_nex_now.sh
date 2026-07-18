#!/usr/bin/env bash

cd /home/vivekfei/EXP/NeX

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nex

export PYTHONPATH=.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export NEX_DISABLE_BOTTOM_BAR=1

PARENT="new_fixed_nex_run"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${PARENT}_REMAINING6_NOW_${STAMP}.log"

mkdir -p "results/$PARENT" logs

{
echo "================================================================================"
echo "REMAINING 6 FIXED NeX STARTING NOW"
echo "================================================================================"
echo "DATE=$(date)"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG=$LOG"
echo

for SCENE in flower fortress horns Ollie room trex; do
  MODEL_DIR="${PARENT}/${SCENE}_FIXED_COLMAP_PATCH3c_PAPERDEFAULT_l16_sl12_h384_mlp4_ray8000_w1008_e300_eval5_ckpt10_valimgint5_cv2"
  RUN_DIR="results/${MODEL_DIR}"
  SUMMARY="${RUN_DIR}/logs/summary.json"

  echo
  echo "================================================================================"
  echo "START SCENE=$SCENE"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "DATE=$(date)"
  echo "================================================================================"

  if [ -f "$SUMMARY" ]; then
    DONE=$(python - "$SUMMARY" <<'PY'
import json, sys
try:
    s=json.load(open(sys.argv[1]))
    print("YES" if s.get("pipeline_complete") is True else "NO")
except Exception:
    print("NO")
PY
)
    if [ "$DONE" = "YES" ]; then
      echo "SKIP already finished: $SCENE"
      continue
    fi
  fi

  echo "PRECHECK DATASET FILES"
  for F in "dataset/$SCENE" "dataset/$SCENE/poses_bounds.npy" "dataset/$SCENE/hwf_cxcy.npy" "dataset/$SCENE/train_image.txt" "dataset/$SCENE/val_image.txt"; do
    if [ -e "$F" ]; then
      echo "OK: $F"
    else
      echo "MISSING: $F"
      exit 20
    fi
  done

  echo
  echo "GPU BEFORE $SCENE"
  nvidia-smi || true

  echo
  echo "TRAIN $SCENE"
  python -u train.py \
    -scene "dataset/${SCENE}" \
    -model_dir "$MODEL_DIR" \
    -restart \
    -val_image_interval 5 \
    -epochs 300 \
    -tb_savempi 5 \
    -checkpoint 10 \
    -tb_saveimage 0 \
    -tb_toc 20 \
    -ray 8000 \
    -layers 16 \
    -sublayers 12 \
    -hidden 384 \
    -mlp 4 \
    -llff_width 1008 \
    -num_workers 8 \
    -cv2resize \
    -no_video \
    -no_webgl

  ECODE=$?
  echo
  echo "SCENE=$SCENE EXIT_CODE=$ECODE DATE=$(date)"

  if [ "$ECODE" != "0" ]; then
    echo "STOPPING BECAUSE FAILED SCENE=$SCENE"
    exit "$ECODE"
  fi

  echo "FINISHED SCENE=$SCENE"
  nvidia-smi || true
done

echo
echo "================================================================================"
echo "REMAINING 6 FINISHED"
echo "================================================================================"
echo "DATE=$(date)"
} 2>&1 | tee -a "$LOG"
