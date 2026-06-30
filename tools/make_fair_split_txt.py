#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset/full_data_4x_fresh_colmap")
    ap.add_argument("--val_image_interval", "--val_interval", type=int, default=5)
    ap.add_argument("--scenes", nargs="+", default=["CanClip", "flower", "fortress", "horns", "Ollie", "room", "trex"])
    args = ap.parse_args()

    interval = int(args.val_image_interval)
    if interval <= 1:
        raise ValueError("val_image_interval must be >= 2")

    root = Path(args.root)

    for scene in args.scenes:
        scene_dir = root / scene
        img_dir = scene_dir / "images"

        imgs = sorted([
            p.name for p in img_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        ids = list(range(len(imgs)))
        val_ids = [i for i in ids if i % interval == 0]
        train_ids = [i for i in ids if i % interval != 0]

        train = [imgs[i] for i in train_ids]
        val = [imgs[i] for i in val_ids]

        if not train:
            raise ValueError(f"{scene}: empty train split")
        if not val:
            raise ValueError(f"{scene}: empty val split")

        (scene_dir / "train_image.txt").write_text("\n".join(train) + "\n")
        (scene_dir / "val_image.txt").write_text("\n".join(val) + "\n")

        info = {
            "scene": scene,
            "val_image_interval": interval,
            "val_split_rule": "start_from_first_image_zero_based_ids_mod_interval_eq_0",
            "n_images": len(imgs),
            "train_ids_0based": train_ids,
            "val_ids_0based": val_ids,
            "train_names": train,
            "val_names": val,
        }
        (scene_dir / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")

        print(f"[OK] {scene}: interval={interval} total={len(imgs)} train={len(train)} val={len(val)}")
        print("     val:", ", ".join(val))

if __name__ == "__main__":
    main()
