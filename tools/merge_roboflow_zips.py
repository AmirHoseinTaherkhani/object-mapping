"""
Extract the manually-downloaded Roboflow zips and merge into trainingData.

Class maps are hard-coded based on each zip's data.yaml (already inspected).
Run from repo root:
    python tools/merge_roboflow_zips.py
"""

import random
import shutil
import zipfile
from pathlib import Path

random.seed(42)

ROOT      = Path(__file__).parent.parent
LDS       = ROOT / "labeling_data" / "large_datasets"
TRAIN_DIR = ROOT / "labeling_data" / "trainingData"

MAX_PER_DATASET = 5000   # cap per split per dataset

# ── class maps: {roboflow_class_id: our_class_id | None=skip} ─────────────────

MAPS = {
    "crowdhuman": {
        # nc=2: ['head', 'person']
        0: None,   # head   → skip
        1: 0,      # person → person
    },
    "widerperson": {
        # nc=3: ['partially-visible persons', 'pedestrians', 'riders']
        0: 0,      # partially-visible → person
        1: 0,      # pedestrians       → person
        2: 0,      # riders (cyclists) → person
    },
    "ua_detrac_10k": {
        # nc=4: ['bus', 'car', 'truck', 'van']
        0: 1,      # bus   → car
        1: 1,      # car   → car
        2: 1,      # truck → car
        3: 1,      # van   → car
    },
    "cowc": {
        # nc=1: ['Car']
        0: 1,      # Car → car
    },
}

ZIP_TO_KEY = {
    "crowdhuman.v2i.yolov11.zip":                         "crowdhuman",
    "WiderPerson.v1i.yolov11.zip":                        "widerperson",
    "UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.yolov11.zip": "ua_detrac_10k",
    "COWC.v6-32.5cm-426p-.yolov11.zip":                  "cowc",
    # skipped: UA-DETRAC.v1i.yolov11.zip (unnamed classes, redundant)
    # skipped: Visdrone.v2-v11.yolov11.zip (already have VisDrone merged)
}


# ── helpers ───────────────────────────────────────────────────────────────────

def remap_label(src: Path, class_map: dict) -> list[str]:
    lines = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            continue
        dst_cls = class_map.get(src_cls)
        if dst_cls is None:
            continue
        lines.append(f"{dst_cls} " + " ".join(parts[1:]))
    return lines


def merge_split(src_img: Path, src_lbl: Path,
                dst_img: Path, dst_lbl: Path,
                class_map: dict, tag: str,
                max_images: int = MAX_PER_DATASET) -> int:
    if not src_img.exists():
        return 0
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png")) + \
           sorted(src_img.glob("*.jpeg"))
    if len(imgs) > max_images:
        imgs = random.sample(imgs, max_images)

    n = 0
    for img in imgs:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        remapped = remap_label(lbl, class_map)
        if not remapped:
            continue
        stem = f"{tag}_{n:07d}"
        dst_p = dst_img / (stem + img.suffix)
        if not dst_p.exists():
            shutil.copy(img, dst_p)
            (dst_lbl / (stem + ".txt")).write_text("\n".join(remapped))
        n += 1
    return n


# ── 1. remove old empty placeholder directories ───────────────────────────────

STALE_DIRS = ["sdd", "ua_detrac", "cowc", "crowdhuman", "widerperson"]
for d in STALE_DIRS:
    p = LDS / d
    if p.exists() and not (p / "done.txt").exists():
        shutil.rmtree(p)
        print(f"Removed stale dir: {p.name}/")


# ── 2. handle partial COCO download ──────────────────────────────────────────

coco_done = LDS / "coco" / "done.txt"
coco_train = LDS / "coco" / "train" / "images"
if coco_train.exists() and not coco_done.exists():
    n_coco = len(list(coco_train.glob("*.jpg")))
    if n_coco > 0:
        coco_done.write_text(f"Total: {n_coco} (partial download)\n")
        print(f"Marked {n_coco} partial COCO images as ready.")


# ── 3. extract + merge each zip ──────────────────────────────────────────────

split_map = {
    "train": ("train", "train"),
    "valid": ("valid", "valid"),
    "test":  ("test",  "test"),
}

td = TRAIN_DIR
dst_splits = {
    "train": (td / "train" / "images", td / "train" / "labels"),
    "valid": (td / "valid" / "images", td / "valid" / "labels"),
    "test":  (td / "test"  / "images", td / "test"  / "labels"),
}

for zip_name, key in ZIP_TO_KEY.items():
    zip_path = LDS / zip_name
    if not zip_path.exists():
        print(f"\n[SKIP] {zip_name} not found.")
        continue

    out_dir = LDS / key
    done_marker = out_dir / "done.txt"
    if done_marker.exists():
        print(f"\n[{key}] Already extracted — merging directly.")
        extract_root = out_dir
    else:
        print(f"\n[{key}] Extracting {zip_name}…")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        # Roboflow zips usually extract to a subfolder named after the project
        # Find the actual root (where train/ valid/ test/ live)
        extract_root = out_dir
        # If the zip extracted into a single subfolder, descend into it
        subdirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name not in ("train","valid","test")]
        if subdirs and not (out_dir / "train").exists():
            extract_root = subdirs[0]
            print(f"  Detected zip root: {extract_root.name}/")

    class_map = MAPS[key]
    total = 0
    for rf_split, (split_name, _) in split_map.items():
        src_img = extract_root / rf_split / "images"
        src_lbl = extract_root / rf_split / "labels"
        dst_img, dst_lbl = dst_splits[split_name]
        n = merge_split(src_img, src_lbl, dst_img, dst_lbl, class_map, tag=key)
        if n:
            print(f"  {rf_split}: +{n} images → {split_name}/")
        total += n

    done_marker.write_text(f"Total: {total}\n")
    print(f"  Done: {total} images merged.")


# ── 4. merge any COCO images not yet in training ─────────────────────────────

coco_dir = LDS / "coco"
if (coco_dir / "done.txt").exists():
    print(f"\n[coco] Merging partial COCO images…")
    # COCO images already have unique coco_* names; just copy if not present
    coco_total = 0
    for split_name in ("train", "valid"):
        src_img = coco_dir / split_name / "images"
        src_lbl = coco_dir / split_name / "labels"
        dst_img, dst_lbl = dst_splits[split_name]
        if not src_img.exists():
            continue
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in sorted(src_img.glob("*.jpg")):
            lbl = src_lbl / (img.stem + ".txt")
            if not lbl.exists():
                continue
            if not (dst_img / img.name).exists():
                shutil.copy(img, dst_img / img.name)
                shutil.copy(lbl, dst_lbl / lbl.name)
                n += 1
        if n:
            print(f"  {split_name}: +{n} COCO images")
        coco_total += n


# ── 5. final count ────────────────────────────────────────────────────────────

print("\n── Final training data sizes ───────────────────────────────────")
for split in ("train", "valid", "test"):
    p = TRAIN_DIR / split / "images"
    print(f"  {split:<6}: {len(list(p.glob('*'))) if p.exists() else 0:>6} images")

print("\nDone. Ready to sync to HPC and submit jobs.")
