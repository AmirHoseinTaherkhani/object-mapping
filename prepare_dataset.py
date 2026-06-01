"""
Splits labeling_data/images + labels into train/val and writes data.yaml.
Run this after reviewing auto-labels (and any manual fixes).
Then run: python finetune_on_camera.py
"""

import random
import shutil
from pathlib import Path

PROJECT   = Path(__file__).parent
IMG_DIR   = PROJECT / "labeling_data" / "images"
LBL_DIR   = PROJECT / "labeling_data" / "labels"
DSET_DIR  = PROJECT / "labeling_data" / "dataset"
VAL_RATIO = 0.15
SEED      = 42

random.seed(SEED)


def main():
    images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

    # only keep images that have a matching non-empty label file
    paired = []
    skipped_empty = 0
    for img in images:
        lbl = LBL_DIR / (img.stem + ".txt")
        if lbl.exists() and lbl.read_text().strip():
            paired.append((img, lbl))
        else:
            skipped_empty += 1

    random.shuffle(paired)
    n_val   = max(1, int(len(paired) * VAL_RATIO))
    n_train = len(paired) - n_val

    splits = {"train": paired[:n_train], "valid": paired[n_train:]}

    for split, pairs in splits.items():
        (DSET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DSET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy(img, DSET_DIR / split / "images" / img.name)
            shutil.copy(lbl, DSET_DIR / split / "labels" / lbl.name)

    # write data.yaml
    yaml_content = f"""path: {DSET_DIR}
train: train/images
val: valid/images

nc: 2
names:
  - person
  - car
"""
    (DSET_DIR / "data.yaml").write_text(yaml_content)

    print(f"Dataset prepared → {DSET_DIR}")
    print(f"  Train : {n_train} images")
    print(f"  Val   : {n_val} images")
    print(f"  Skipped (empty labels): {skipped_empty}")
    print(f"\nRun next: python finetune_on_camera.py")


if __name__ == "__main__":
    main()
