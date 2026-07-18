#!/usr/bin/env python3
from pathlib import Path
import json
import csv
import sys

ROOT = Path("/home/vivekfei/EXP/NeX")
BRANCH = ROOT / "results" / "new_fixed_nex_run"

SCENES = ["flower", "fortress", "horns", "room", "trex", "CanClip", "Ollie"]

FILES_TO_DUMP = [
    "config.json",
    "logs/summary.json",
    "run_info.txt",
    "metrics.csv",
    "val_per_image_metrics.csv",
]

def banner(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def file_banner(path):
    print("\n" + "-" * 100)
    print(f"FILE: {path}")
    print("-" * 100)

def read_text(path):
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"

def find_scene_dir(scene):
    scene_l = scene.lower()
    matches = []
    for d in sorted(BRANCH.iterdir()):
        if not d.is_dir():
            continue
        name_l = d.name.lower()
        if name_l.startswith(scene_l + "_"):
            matches.append(d)
    return matches

def load_json(path):
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}

def best_row_from_metrics(metrics_path):
    best = {}
    if not metrics_path.exists():
        return best
    try:
        with metrics_path.open(newline="", errors="replace") as f:
            for r in csv.DictReader(f):
                if (r.get("phase") == "val") and ((r.get("is_best") or "").strip() == "1"):
                    best = r
    except Exception:
        return {}
    return best

banner("NEX / new_fixed_nex_run / ALL 7 SCENES DUMP")
print(f"ROOT:   {ROOT}")
print(f"BRANCH: {BRANCH}")
print(f"EXISTS: {BRANCH.exists()}")

banner("COMPACT SUMMARY FIRST")
print("scene,run_dir,pipeline_complete,status,best_epoch,best_psnr,best_ssim,best_lpips,final_epoch,final_psnr,final_ssim,final_lpips,n_train,n_val,created_at,finished_at")

scene_dirs = {}

for scene in SCENES:
    matches = find_scene_dir(scene)
    if not matches:
        print(f"{scene},MISSING,,,,,,,,,,,,,,")
        continue

    d = matches[0]
    scene_dirs[scene] = d

    cfg = load_json(d / "config.json")
    summary = load_json(d / "logs" / "summary.json")
    best_metrics = best_row_from_metrics(d / "metrics.csv")

    args = cfg.get("args", {}) if isinstance(cfg, dict) else {}

    best_ssim = summary.get("best_val_mean_ssim", "")
    best_lpips = summary.get("best_val_mean_lpips", "")

    if best_ssim in ("", None):
        best_ssim = best_metrics.get("val_mean_ssim", "")
    if best_lpips in ("", None):
        best_lpips = best_metrics.get("val_mean_lpips", "")

    row = [
        scene,
        d.name,
        summary.get("pipeline_complete", ""),
        summary.get("status", ""),
        summary.get("best_epoch", ""),
        summary.get("best_val_mean_psnr", ""),
        best_ssim,
        best_lpips,
        summary.get("final_epoch", summary.get("latest_epoch", "")),
        summary.get("latest_val_mean_psnr", ""),
        summary.get("latest_val_mean_ssim", ""),
        summary.get("latest_val_mean_lpips", ""),
        summary.get("n_train_images", cfg.get("n_train_images", "")),
        summary.get("n_val_images", cfg.get("n_val_images", "")),
        cfg.get("created_at", ""),
        summary.get("finished_at", ""),
    ]
    print(",".join(str(x).replace("\n", " ") for x in row))

for scene in SCENES:
    banner(f"SCENE: {scene}")

    d = scene_dirs.get(scene)
    if not d:
        print(f"[MISSING SCENE DIR FOR {scene}]")
        continue

    print(f"RUN_DIR: {d}")

    for rel in FILES_TO_DUMP:
        path = d / rel
        file_banner(path)
        if path.exists():
            print(read_text(path))
        else:
            print(f"[MISSING FILE: {path}]")

banner("END OF NEX DUMP")
