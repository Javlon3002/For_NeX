from pathlib import Path
from utils.colmap_runner import load_colmap_data, save_poses

root = Path("dataset/full_data_4x_fresh_colmap")
scenes = ["CanClip", "flower", "fortress", "horns", "Ollie", "room", "trex"]

for scene in scenes:
    d = root / scene
    print(f"=== {scene} ===")
    poses, pts3d, perm, hwf_cxcy = load_colmap_data(str(d))
    save_poses(str(d), poses, pts3d, perm, hwf_cxcy)
    print("created:", d / "poses_bounds.npy")
    print("created:", d / "hwf_cxcy.npy")
