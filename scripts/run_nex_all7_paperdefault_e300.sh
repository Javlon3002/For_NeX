#!/usr/bin/env bash
set -eo pipefail

cd /home/vivekfei/EXP/NeX

RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/NEX_ALL7_PATCH3c_PAPERDEFAULT_E300_${RUN_ID}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================================"
echo "NeX ALL7 PATCH3c PAPERDEFAULT E300"
echo "================================================================================"
echo "START_DATE=$(date)"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "MASTER_LOG=$MASTER_LOG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nex

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.
export NEX_DISABLE_BOTTOM_BAR=1

echo
echo "================================================================================"
echo "ENV CHECK"
echo "================================================================================"
echo "CONDA_PREFIX=$CONDA_PREFIX"
python - <<'PY'
import torch, sys
print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

echo
echo "================================================================================"
echo "GIT CHECK"
echo "================================================================================"
git log --oneline --decorate -6
git status --short

if [ -n "$(git status --short)" ]; then
  echo "ERROR: git working tree is not clean inside run script."
  exit 1
fi

DATA_ROOT="dataset/full_data_4x_fresh_colmap"
SCENES=(CanClip flower fortress horns Ollie room trex)

echo
echo "================================================================================"
echo "DATASET PREFLIGHT"
echo "================================================================================"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="$DATA_ROOT/$SCENE"
  IMG_DIR="$SCENE_DIR/images"

  echo
  echo "----- $SCENE -----"

  test -d "$SCENE_DIR" || { echo "ERROR: missing $SCENE_DIR"; exit 1; }
  test -d "$IMG_DIR" || { echo "ERROR: missing $IMG_DIR"; exit 1; }
  test -f "$SCENE_DIR/poses_bounds.npy" || { echo "ERROR: missing poses_bounds.npy for $SCENE"; exit 1; }
  test -f "$SCENE_DIR/hwf_cxcy.npy" || { echo "ERROR: missing hwf_cxcy.npy for $SCENE"; exit 1; }
  test -f "$SCENE_DIR/sparse/0/cameras.bin" || { echo "ERROR: missing sparse/0/cameras.bin for $SCENE"; exit 1; }
  test -f "$SCENE_DIR/sparse/0/images.bin" || { echo "ERROR: missing sparse/0/images.bin for $SCENE"; exit 1; }
  test -f "$SCENE_DIR/sparse/0/points3D.bin" || { echo "ERROR: missing sparse/0/points3D.bin for $SCENE"; exit 1; }

  NIMG="$(find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  echo "images=$NIMG"

  if [ "$NIMG" -lt 10 ]; then
    echo "ERROR: too few images for $SCENE"
    exit 1
  fi
done

echo
echo "================================================================================"
echo "REFRESH FAIR SPLITS: val_image_interval=5"
echo "================================================================================"
python tools/make_fair_split_txt.py \
  --root "$DATA_ROOT" \
  --val_image_interval 5 \
  --scenes "${SCENES[@]}"

echo
echo "================================================================================"
echo "SPLIT SUMMARY"
echo "================================================================================"
python - <<'PY'
import json
from pathlib import Path

root = Path("dataset/full_data_4x_fresh_colmap")
scenes = ["CanClip", "flower", "fortress", "horns", "Ollie", "room", "trex"]

for scene in scenes:
    info = json.loads((root / scene / "split_info.json").read_text())
    print(f"{scene}: total={info['n_images']} train={len(info['train_names'])} val={len(info['val_names'])}")
    print("  val:", ", ".join(info["val_names"]))
PY

echo
echo "================================================================================"
echo "RUN CONFIG"
echo "================================================================================"
echo "epochs=300"
echo "val_image_interval=5"
echo "eval_every/tb_savempi=5"
echo "checkpoint_every=10"
echo "tb_saveimage=0"
echo "save validation pred/gt/error images for every eval epoch"
echo "paper/default NeX capacity: layers=16 sublayers=12 hidden=384 mlp=4 ray=8000 width=1008"
echo "no_video=true no_webgl=true cv2resize=true"

START_ALL="$(date +%s)"

for SCENE in "${SCENES[@]}"; do
  SCENE_START="$(date +%s)"
  SCENE_DIR="$DATA_ROOT/$SCENE"
  EXP_NAME="0007_${SCENE}_NeX_PATCH3c_PAPERDEFAULT_l16_sl12_h384_mlp4_ray8000_w1008_e300_eval5_ckpt10_valimgint5_cv2_login"

  echo
  echo "################################################################################"
  echo "START SCENE: $SCENE"
  echo "EXP_NAME: $EXP_NAME"
  echo "DATE: $(date)"
  echo "################################################################################"

  rm -rf "results/$EXP_NAME"

  python train.py \
    -scene "$SCENE_DIR" \
    -model_dir "$EXP_NAME" \
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

  echo
  echo "================================================================================"
  echo "POST-RUN VERIFY: $SCENE"
  echo "================================================================================"

  python - "$EXP_NAME" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

exp_name = sys.argv[1]
exp = Path("results") / exp_name

summary = json.loads((exp / "logs/summary.json").read_text())
metrics = pd.read_csv(exp / "metrics.csv")
val = pd.read_csv(exp / "val_per_image_metrics.csv")

errors = []

if summary.get("pipeline_complete") is not True:
    errors.append("pipeline_complete is not true")
if summary.get("total_epochs") != 300:
    errors.append(f"total_epochs expected 300, got {summary.get('total_epochs')}")
if summary.get("latest_epoch") != 300:
    errors.append(f"latest_epoch expected 300, got {summary.get('latest_epoch')}")
if summary.get("latest_eval_epoch") != 300:
    errors.append(f"latest_eval_epoch expected 300, got {summary.get('latest_eval_epoch')}")

n_train = int(summary["n_train_images"])
n_val = int(summary["n_val_images"])
expected_steps = 300 * n_train
if int(summary.get("latest_step")) != expected_steps:
    errors.append(f"latest_step expected {expected_steps}, got {summary.get('latest_step')}")

train_epochs = metrics.loc[metrics["phase"] == "train", "epoch"].astype(int).tolist()
val_epochs = metrics.loc[metrics["phase"] == "val", "epoch"].astype(int).tolist()
expected_train_epochs = list(range(1, 301))
expected_val_epochs = list(range(5, 301, 5))

if train_epochs != expected_train_epochs:
    errors.append(f"train epochs wrong: first/last/count = {train_epochs[:3]} ... {train_epochs[-3:]} / {len(train_epochs)}")
if val_epochs != expected_val_epochs:
    errors.append(f"val epochs wrong: first/last/count = {val_epochs[:3]} ... {val_epochs[-3:]} / {len(val_epochs)}")

expected_val_rows = n_val * len(expected_val_epochs)
if len(val) != expected_val_rows:
    errors.append(f"val_per_image rows expected {expected_val_rows}, got {len(val)}")

if "lpips" not in val.columns:
    errors.append("lpips column missing")
elif val["lpips"].isna().any():
    errors.append("lpips contains NaN")

ckpts = sorted(p.name for p in (exp / "checkpoints/periodic").glob("epoch_*.pt"))
expected_ckpts = [f"epoch_{e:04d}.pt" for e in range(10, 301, 10)]
if ckpts != expected_ckpts:
    errors.append(f"periodic checkpoints wrong: count={len(ckpts)} first={ckpts[:3]} last={ckpts[-3:]}")

for e in expected_val_epochs:
    imgdir = exp / "images" / f"epoch_{e:04d}"
    files = sorted(imgdir.glob("*.png"))
    expected_imgs = n_val * 3
    if len(files) != expected_imgs:
        errors.append(f"{imgdir} expected {expected_imgs} png files, got {len(files)}")

if errors:
    print("FAIL:", exp_name)
    for err in errors:
        print(" -", err)
    sys.exit(1)

print("PASS:", exp_name)
print(" best_epoch:", summary.get("best_epoch"))
print(" best_val_mean_psnr:", summary.get("best_val_mean_psnr"))
print(" best_val_mean_ssim:", summary.get("best_val_mean_ssim"))
print(" best_val_mean_lpips:", summary.get("best_val_mean_lpips"))
print(" final_val_psnr:", summary.get("latest_val_mean_psnr"))
print(" final_val_ssim:", summary.get("latest_val_mean_ssim"))
print(" final_val_lpips:", summary.get("latest_val_mean_lpips"))
PY

  echo
  echo "SUMMARY JSON:"
  cat "results/$EXP_NAME/logs/summary.json"

  echo
  echo "METRICS TAIL:"
  tail -n 20 "results/$EXP_NAME/metrics.csv"

  echo
  echo "CHECKPOINTS:"
  find "results/$EXP_NAME/checkpoints" -type f | sort | tail -n 40

  SCENE_END="$(date +%s)"
  SCENE_SEC="$((SCENE_END - SCENE_START))"
  echo
  echo "DONE SCENE: $SCENE"
  echo "SECONDS: $SCENE_SEC"
  echo "HOURS: $(python - <<PY
print(round($SCENE_SEC / 3600, 3))
PY
)"
done

END_ALL="$(date +%s)"
ALL_SEC="$((END_ALL - START_ALL))"

echo
echo "================================================================================"
echo "ALL7 FINISHED"
echo "================================================================================"
echo "END_DATE=$(date)"
echo "TOTAL_SECONDS=$ALL_SEC"
echo "TOTAL_HOURS=$(python - <<PY
print(round($ALL_SEC / 3600, 3))
PY
)"

echo
echo "================================================================================"
echo "AGGREGATE SUMMARY"
echo "================================================================================"
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

rows = []
for exp in sorted(Path("results").glob("0007_*_NeX_PATCH3c_PAPERDEFAULT_*")):
    s = json.loads((exp / "logs/summary.json").read_text())
    rows.append({
        "experiment": exp.name,
        "scene": s.get("scene"),
        "n_train": s.get("n_train_images"),
        "n_val": s.get("n_val_images"),
        "best_epoch": s.get("best_epoch"),
        "best_psnr": s.get("best_val_mean_psnr"),
        "best_ssim": s.get("best_val_mean_ssim"),
        "best_lpips": s.get("best_val_mean_lpips"),
        "final_psnr": s.get("latest_val_mean_psnr"),
        "final_ssim": s.get("latest_val_mean_ssim"),
        "final_lpips": s.get("latest_val_mean_lpips"),
        "pipeline_complete": s.get("pipeline_complete"),
    })

df = pd.DataFrame(rows)
out = Path("results/NEX_ALL7_PATCH3c_PAPERDEFAULT_E300_summary.csv")
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("wrote:", out)
PY

echo
echo "DONE ALL7"
