"""
Splits image+label pool into train/val and writes data.yaml.

Usage:
    python prepare_dataset.py                  # original camera labels
    python prepare_dataset.py --source merged  # merged Roboflow + camera
"""

import argparse
import random
import shutil
from pathlib import Path

PROJECT   = Path(__file__).parent
VAL_RATIO = 0.15
SEED      = 42

random.seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="original",
                        choices=["original", "merged"],
                        help="'original' = labeling_data/images, 'merged' = labeling_data/merged")
    args = parser.parse_args()

    if args.source == "merged":
        img_dir  = PROJECT / "labeling_data" / "merged" / "images"
        lbl_dir  = PROJECT / "labeling_data" / "merged" / "labels"
        dset_dir = PROJECT / "labeling_data" / "merged_dataset"
    else:
        img_dir  = PROJECT / "labeling_data" / "images"
        lbl_dir  = PROJECT / "labeling_data" / "labels"
        dset_dir = PROJECT / "labeling_data" / "dataset"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    paired = []
    skipped_empty = 0
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.read_text().strip():
            paired.append((img, lbl))
        else:
            skipped_empty += 1

    random.shuffle(paired)
    n_val   = max(1, int(len(paired) * VAL_RATIO))
    n_train = len(paired) - n_val

    splits = {"train": paired[:n_train], "valid": paired[n_train:]}

    for split, pairs in splits.items():
        (dset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy(img, dset_dir / split / "images" / img.name)
            shutil.copy(lbl, dset_dir / split / "labels" / lbl.name)

    yaml_content = f"""path: {dset_dir}
train: train/images
val: valid/images

nc: 2
names:
  - person
  - car
"""
    (dset_dir / "data.yaml").write_text(yaml_content)

    print(f"Dataset prepared → {dset_dir}")
    print(f"  Train : {n_train} images")
    print(f"  Val   : {n_val} images")
    print(f"  Skipped (empty labels): {skipped_empty}")
    print(f"\nRun next: python finetune_on_camera.py")


if __name__ == "__main__":
    main()
