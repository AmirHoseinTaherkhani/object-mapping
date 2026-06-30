"""
Merge all Roboflow datasets + existing camera labels into one unified pool.

Class remapping → our 2-class schema:
  0 = person
  1 = car  (covers car, bus, truck, motorcycle, van, pickup, etc.)

Datasets used:
  cars cars cars.v2i.yolov8   — all classes → car
  cars.v1i.yolov8             — all classes → car
  cars.v3-2024-02-14          — all classes → car  (capped at MAX_PER_DS)
  cars.v3i.yolov8             — all classes → car  (capped at MAX_PER_DS)
  vehicleDataSetV02           — all classes → car
  people.v2i.yolov8 (1)      — class 0 (people) → person
  people.v3i.yolov8           — class 1 (person) → person, class 0 (bagpack) skipped
  labeling_data/              — our corrected camera frames (already 2-class)

People.v2i.yolov8 is intentionally excluded (Violence detection, wrong task).

Output: labeling_data/merged/images/  +  labeling_data/merged/labels/
"""

import random
import shutil
from pathlib import Path

PROJECT     = Path(__file__).parent
NEW_DS_DIR  = PROJECT / "newDatasets"
CAM_IMG_DIR = PROJECT / "labeling_data" / "images"
CAM_LBL_DIR = PROJECT / "labeling_data" / "labels"
OUT_DIR     = PROJECT / "labeling_data" / "merged"
OUT_IMG     = OUT_DIR / "images"
OUT_LBL     = OUT_DIR / "labels"

MAX_PER_DS  = 2000   # cap large datasets so car class doesn't overwhelm person
SEED        = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Dataset definitions: (folder, class_map)
# class_map: {src_class_id -> dst_class_id}  — None means skip
# ---------------------------------------------------------------------------
DATASETS = [
    # ── vehicles ──
    ("cars cars cars.v2i.yolov8", {0: 1, 1: 1, 2: 1, 3: 1}),
    ("cars.v1i.yolov8",           {0: 1, 1: 1, 2: 1}),
    # capped — large Indian traffic dataset
    ("cars.v3-2024-02-14-2-20pm.yolov8", {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1}),
    # capped — large overhead vehicle dataset
    ("cars.v3i.yolov8",           {0: 1, 1: 1, 2: 1, 3: 1}),
    ("vehicleDataSetV02",         {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
    # ── persons ──
    ("people.v2i.yolov8 (1)",     {0: 0}),
    # people.v3i: class 0=bagpack (skip), class 1=person
    ("people.v3i.yolov8",         {0: None, 1: 0}),
]

LARGE_DS = {
    "cars.v3-2024-02-14-2-20pm.yolov8",
    "cars.v3i.yolov8",
}


def collect_pairs(ds_dir: Path):
    """Return (image_path, label_path) pairs from all splits."""
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def remap_label(src_lbl: Path, class_map: dict) -> list[str]:
    """Read a YOLO label file, apply class_map, return remapped lines."""
    lines = []
    for line in src_lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(parts[0])
        except ValueError:
            continue
        dst_cls = class_map.get(src_cls)
        if dst_cls is None:
            continue
        lines.append(f"{dst_cls} " + " ".join(parts[1:]))
    return lines


def copy_pair(img: Path, lbl: Path, class_map: dict, idx: int, ds_tag: str):
    """Remap labels and copy image+label to output directory."""
    remapped = remap_label(lbl, class_map)
    if not remapped:
        return False   # skip images where all boxes were filtered out

    safe_tag = ds_tag.replace(" ", "_").replace(".", "_")
    stem = f"{safe_tag}_{idx:06d}"
    suffix = img.suffix

    out_img = OUT_IMG / (stem + suffix)
    out_lbl = OUT_LBL / (stem + ".txt")

    shutil.copy(img, out_img)
    out_lbl.write_text("\n".join(remapped))
    return True


def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    total_written = 0
    stats = {"person": 0, "car": 0}

    # ── 1. Roboflow datasets ────────────────────────────────────────────
    for ds_name, class_map in DATASETS:
        ds_dir = NEW_DS_DIR / ds_name
        if not ds_dir.exists():
            print(f"  [SKIP] not found: {ds_dir}")
            continue

        pairs = collect_pairs(ds_dir)
        if ds_name in LARGE_DS and len(pairs) > MAX_PER_DS:
            pairs = random.sample(pairs, MAX_PER_DS)
            print(f"  [CAP]  {ds_name}: sampled {MAX_PER_DS}/{len(pairs)+MAX_PER_DS}")

        n = 0
        for img, lbl in pairs:
            if copy_pair(img, lbl, class_map, total_written, ds_name):
                total_written += 1
                n += 1
        print(f"  {ds_name}: {n} images")

    # ── 2. Our own camera labels (already 2-class, copy as-is) ─────────
    cam_pairs = []
    for img in sorted(CAM_IMG_DIR.glob("*.jpg")) + sorted(CAM_IMG_DIR.glob("*.png")):
        lbl = CAM_LBL_DIR / (img.stem + ".txt")
        if lbl.exists() and lbl.read_text().strip():
            cam_pairs.append((img, lbl))

    for img, lbl in cam_pairs:
        stem = f"camera_{total_written:06d}"
        out_img = OUT_IMG / (stem + img.suffix)
        out_lbl = OUT_LBL / (stem + ".txt")
        shutil.copy(img, out_img)
        shutil.copy(lbl, out_lbl)
        total_written += 1

    print(f"  camera frames: {len(cam_pairs)} images")

    # ── 3. Count class distribution and collect all person images ───────
    # Priority order for oversampling:
    #   1. Camera frames (most domain-relevant — copied with "camera_" prefix)
    #   2. Roboflow person datasets
    person_imgs_camera = []   # from our intersection camera
    person_imgs_other  = []   # from Roboflow person datasets

    for lbl in sorted(OUT_LBL.glob("*.txt")):
        has_person = False
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                c = int(parts[0])
                if c == 0:
                    stats["person"] += 1
                    has_person = True
                if c == 1:
                    stats["car"] += 1

        if has_person:
            img_candidates = list(OUT_IMG.glob(lbl.stem + ".*"))
            if img_candidates:
                entry = (img_candidates[0], lbl)
                if lbl.stem.startswith("camera_"):
                    person_imgs_camera.append(entry)
                else:
                    person_imgs_other.append(entry)

    # Combine: camera frames first (get repeated more often)
    person_imgs = person_imgs_camera + person_imgs_other
    ratio = stats["car"] / max(stats["person"], 1)

    # ── 4. Oversample person images until car:person ≤ TARGET_RATIO ───
    TARGET_RATIO = 8.0
    if ratio > TARGET_RATIO and person_imgs:
        copies_needed  = int(stats["car"] / TARGET_RATIO - stats["person"])
        copies_per_img = max(1, copies_needed // len(person_imgs))
        print(f"\n  Oversampling: {len(person_imgs_camera)} camera + "
              f"{len(person_imgs_other)} Roboflow person images ×{copies_per_img}")
        print(f"  (ratio {ratio:.1f}:1 → target ≤{TARGET_RATIO:.0f}:1)")

        oversample_count = 0
        for img, lbl in person_imgs:
            for _ in range(copies_per_img):
                stem = f"person_over_{total_written:06d}"
                shutil.copy(img, OUT_IMG / (stem + img.suffix))
                shutil.copy(lbl, OUT_LBL / (stem + ".txt"))
                total_written += 1
                oversample_count += 1
                for line in lbl.read_text().splitlines():
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        c = int(parts[0])
                        if c == 0: stats["person"] += 1
                        if c == 1: stats["car"] += 1

        print(f"  Added {oversample_count} person copies")
        ratio = stats["car"] / max(stats["person"], 1)

    print(f"\n{'='*50}")
    print(f"Merged dataset → {OUT_DIR}")
    print(f"  Total images : {total_written}")
    print(f"  Person boxes : {stats['person']}")
    print(f"  Car boxes    : {stats['car']}")
    print(f"  Car:Person ratio : {ratio:.1f}:1")
    print(f"{'='*50}")
    print(f"\nNext: python prepare_dataset.py --source merged")


if __name__ == "__main__":
    main()
