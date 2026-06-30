from pathlib import Path

root = Path("dataset/full_data_4x_fresh_colmap")
scenes = ["CanClip", "flower", "fortress", "horns", "Ollie", "room", "trex"]

for scene in scenes:
    scene_dir = root / scene
    img_dir = scene_dir / "images"

    imgs = sorted([
        p.name for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    val = [name for i, name in enumerate(imgs) if i % 5 == 0]
    train = [name for i, name in enumerate(imgs) if i % 5 != 0]

    (scene_dir / "train_image.txt").write_text("\n".join(train) + "\n")
    (scene_dir / "val_image.txt").write_text("\n".join(val) + "\n")

    print(f"[OK] {scene}: total={len(imgs)} train={len(train)} val={len(val)}")
    print("     val:", ", ".join(val))
