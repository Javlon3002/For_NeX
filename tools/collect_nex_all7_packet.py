#!/usr/bin/env python3
from pathlib import Path
import csv, json, math
from datetime import datetime

ROOT = Path("/home/vivekfei/EXP/NeX")
BRANCH = ROOT / "results" / "new_fixed_nex_run"
OUT = ROOT / "dumps" / "NEX_NEW_FIXED_RUN_ALL7_ANALYSIS_PACKET.txt"

SCENES = ["flower", "fortress", "horns", "room", "trex", "CanClip", "Ollie"]

def is_num(x):
    try:
        return x not in ("", None, "nan", "NaN") and not math.isnan(float(x))
    except Exception:
        return False

def fnum(x, nd=6):
    if not is_num(x):
        return ""
    return f"{float(x):.{nd}f}"

def read_json(p):
    try:
        return json.loads(p.read_text(errors="replace"))
    except Exception:
        return {}

def read_csv(p):
    if not p.exists():
        return []
    with p.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def find_scene_dir(scene):
    scene_l = scene.lower()
    matches = []
    for d in sorted(BRANCH.iterdir()):
        if d.is_dir() and d.name.lower().startswith(scene_l + "_"):
            matches.append(d)
    return matches[0] if matches else None

def last_val_row(rows):
    last = {}
    for r in rows:
        if r.get("phase") == "val" and is_num(r.get("val_mean_psnr")):
            last = r
    return last

def best_val_row(rows):
    best = {}
    for r in rows:
        if r.get("phase") == "val" and str(r.get("is_best", "")).strip() == "1":
            best = r
    return best

def val_row_for_epoch(rows, epoch):
    for r in rows:
        if r.get("phase") == "val" and str(r.get("epoch")) == str(epoch):
            return r
    return {}

def per_image_rows(rows, epoch):
    return [r for r in rows if str(r.get("epoch")) == str(epoch)]

def runtime_min(created_at, finished_at):
    try:
        a = datetime.fromisoformat(created_at)
        b = datetime.fromisoformat(finished_at)
        return (b - a).total_seconds() / 60.0
    except Exception:
        return None

lines = []
def emit(x=""):
    lines.append(str(x))

emit("=" * 120)
emit("NEX ANALYSIS PACKET: new_fixed_nex_run")
emit("=" * 120)
emit(f"ROOT   : {ROOT}")
emit(f"BRANCH : {BRANCH}")
emit(f"EXISTS : {BRANCH.exists()}")
emit("")

summary_rows = []
scene_data = {}

for scene in SCENES:
    d = find_scene_dir(scene)
    if not d:
        summary_rows.append([scene, "MISSING"] + [""] * 19)
        continue

    cfg = read_json(d / "config.json")
    summ = read_json(d / "logs" / "summary.json")
    metrics = read_csv(d / "metrics.csv")
    perimg = read_csv(d / "val_per_image_metrics.csv")

    args = cfg.get("args", {}) if isinstance(cfg.get("args", {}), dict) else {}

    best_epoch = summ.get("best_epoch", "")
    final_epoch = summ.get("final_epoch", summ.get("latest_epoch", ""))
    best_row = val_row_for_epoch(metrics, best_epoch) or best_val_row(metrics)
    final_row = val_row_for_epoch(metrics, final_epoch) or last_val_row(metrics)

    rt = runtime_min(cfg.get("created_at", ""), summ.get("finished_at", ""))

    summary_rows.append([
        scene,
        d.name,
        summ.get("pipeline_complete", ""),
        summ.get("status", ""),
        cfg.get("model_name", ""),
        cfg.get("branch", ""),
        args.get("epochs", ""),
        args.get("ray", ""),
        args.get("layers", ""),
        args.get("sublayers", ""),
        args.get("hidden", ""),
        args.get("mlp", ""),
        args.get("llff_width", ""),
        summ.get("n_train_images", cfg.get("n_train_images", "")),
        summ.get("n_val_images", cfg.get("n_val_images", "")),
        best_epoch,
        fnum(summ.get("best_val_mean_psnr")),
        fnum(summ.get("best_val_mean_ssim")),
        fnum(summ.get("best_val_mean_lpips")),
        final_epoch,
        fnum(summ.get("latest_val_mean_psnr")),
        fnum(summ.get("latest_val_mean_ssim")),
        fnum(summ.get("latest_val_mean_lpips")),
        fnum(rt, 3) if rt is not None else "",
    ])

    scene_data[scene] = {
        "dir": d,
        "cfg": cfg,
        "args": args,
        "summary": summ,
        "metrics": metrics,
        "perimg": perimg,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_row": best_row,
        "final_row": final_row,
    }

emit("COMPACT CSV SUMMARY")
emit("-" * 120)
emit(",".join([
    "scene","run_dir","complete","status","model","branch",
    "epochs","ray","layers","sublayers","hidden","mlp","llff_width",
    "n_train","n_val",
    "best_epoch","best_psnr","best_ssim","best_lpips",
    "final_epoch","final_psnr","final_ssim","final_lpips","runtime_min"
]))
for r in summary_rows:
    emit(",".join(str(x).replace("\n"," ") for x in r))

for scene in SCENES:
    emit("")
    emit("=" * 120)
    emit(f"SCENE: {scene}")
    emit("=" * 120)

    data = scene_data.get(scene)
    if not data:
        emit("MISSING")
        continue

    d = data["dir"]
    cfg = data["cfg"]
    args = data["args"]
    summ = data["summary"]
    best_row = data["best_row"]
    final_row = data["final_row"]
    perimg = data["perimg"]
    best_epoch = data["best_epoch"]
    final_epoch = data["final_epoch"]

    emit(f"RUN_DIR: {d}")
    emit("")
    emit("CONFIG ESSENTIALS")
    emit("-" * 120)
    emit(f"model_name: {cfg.get('model_name', '')}")
    emit(f"branch: {cfg.get('branch', '')}")
    emit(f"git_commit: {cfg.get('git_commit', '')}")
    emit(f"host: {cfg.get('host', '')}")
    emit(f"user: {cfg.get('user', '')}")
    emit(f"scene: {cfg.get('scene', '')}")
    emit(f"scene_path: {cfg.get('scene_path', '')}")
    emit(f"created_at: {cfg.get('created_at', '')}")
    emit(f"no_validation_leakage: {cfg.get('no_validation_leakage', '')}")
    emit(f"n_train_images: {cfg.get('n_train_images', '')}")
    emit(f"n_val_images: {cfg.get('n_val_images', '')}")
    for k in [
        "epochs","ray","layers","sublayers","hidden","mlp","llff_width",
        "num_workers","cv2resize","val_image_interval","checkpoint",
        "tb_savempi","no_video","no_webgl","lr"
    ]:
        emit(f"args.{k}: {args.get(k, '')}")

    split = cfg.get("split_info", {})
    emit(f"val_ids_0based: {split.get('val_ids_0based', '')}")
    emit(f"val_names: {split.get('val_names', '')}")

    emit("")
    emit("SUMMARY JSON ESSENTIALS")
    emit("-" * 120)
    for k in [
        "pipeline_complete","status","finished_at","total_epochs",
        "best_epoch","best_val_mean_psnr","best_val_mean_ssim","best_val_mean_lpips",
        "final_epoch","latest_epoch","latest_val_mean_psnr","latest_val_mean_ssim","latest_val_mean_lpips"
    ]:
        emit(f"{k}: {summ.get(k, '')}")

    emit("")
    emit("BEST EPOCH VAL METRICS ROW")
    emit("-" * 120)
    if best_row:
        emit(",".join(best_row.keys()))
        emit(",".join(str(best_row.get(k, "")) for k in best_row.keys()))
    else:
        emit("MISSING BEST ROW")

    emit("")
    emit("FINAL EPOCH VAL METRICS ROW")
    emit("-" * 120)
    if final_row:
        emit(",".join(final_row.keys()))
        emit(",".join(str(final_row.get(k, "")) for k in final_row.keys()))
    else:
        emit("MISSING FINAL ROW")

    emit("")
    emit(f"PER-IMAGE METRICS AT BEST EPOCH = {best_epoch}")
    emit("-" * 120)
    rows = per_image_rows(perimg, best_epoch)
    if rows:
        emit(",".join(rows[0].keys()))
        for r in rows:
            emit(",".join(str(r.get(k, "")) for k in rows[0].keys()))
    else:
        emit("MISSING PER-IMAGE BEST EPOCH ROWS")

    emit("")
    emit(f"PER-IMAGE METRICS AT FINAL EPOCH = {final_epoch}")
    emit("-" * 120)
    rows = per_image_rows(perimg, final_epoch)
    if rows:
        emit(",".join(rows[0].keys()))
        for r in rows:
            emit(",".join(str(r.get(k, "")) for k in rows[0].keys()))
    else:
        emit("MISSING PER-IMAGE FINAL EPOCH ROWS")

emit("")
emit("=" * 120)
emit("END OF NEX ANALYSIS PACKET")
emit("=" * 120)

OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
