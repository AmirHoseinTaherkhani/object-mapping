"""
Merge test splits from all newDatasets into a single unified test set.

Uses the same 2-class remapping as merge_datasets.py:
  0 = person
  1 = car  (car, bus, truck, motorcycle, van, pickup …)

People.v2i.yolov8 is excluded (Violence-detection task, wrong domain).

Output:
  labeling_data/trainingData/test/images/
  labeling_data/trainingData/test/labels/

Run:
    python tools/merge_test_sets.py
"""

import shutil
from pathlib import Path

HERE     = Path(__file__).parent.parent
DS_DIR   = HERE / "labeling_data" / "trainingData" / "newDatasets"
OUT_DIR  = HERE / "labeling_data" / "trainingData" / "test"
OUT_IMG  = OUT_DIR / "images"
OUT_LBL  = OUT_DIR / "labels"

# Identical to merge_datasets.py — all datasets → our 2-class schema.
DATASETS = [
    # ── vehicles ──
    ("cars cars cars.v2i.yolov8",           {0: 1, 1: 1, 2: 1, 3: 1}),
    ("cars.v1i.yolov8",                     {0: 1, 1: 1, 2: 1}),
    ("cars.v3-2024-02-14-2-20pm.yolov8",   {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1}),
    ("cars.v3i.yolov8",                     {0: 1, 1: 1, 2: 1, 3: 1}),
    ("vehicleDataSetV02",                   {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
    # ── persons ──
    ("people.v2i.yolov8 (1)",               {0: 0}),
    ("people.v3i.yolov8",                   {0: None, 1: 0}),   # bagpack→skip, person→0
    # People.v2i.yolov8 intentionally excluded (Violence detection, wrong task)
]


def remap_label(src_lbl: Path, class_map: dict) -> list[str]:
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


def main():
    if OUT_DIR.exists():
        print(f"Output already exists: {OUT_DIR}")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        shutil.rmtree(OUT_DIR)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    total = 0
    stats = {"person": 0, "car": 0}

    for ds_name, class_map in DATASETS:
        ds_dir = DS_DIR / ds_name
        if not ds_dir.exists():
            print(f"  [SKIP] not found: {ds_dir}")
            continue

        test_img_dir = ds_dir / "test" / "images"
        test_lbl_dir = ds_dir / "test" / "labels"

        if not test_img_dir.exists():
            print(f"  [NO-TEST] {ds_name}")
            continue

        images = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))
        n = 0
        for img in images:
            lbl = test_lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            remapped = remap_label(lbl, class_map)
            if not remapped:
                continue

            safe_tag = ds_name.replace(" ", "_").replace(".", "_")
            stem     = f"{safe_tag}_{total:06d}"
            out_img  = OUT_IMG / (stem + img.suffix)
            out_lbl  = OUT_LBL / (stem + ".txt")
            shutil.copy(img, out_img)
            out_lbl.write_text("\n".join(remapped))

            for line in remapped:
                c = int(line.split()[0])
                if c == 0: stats["person"] += 1
                if c == 1: stats["car"] += 1

            total += 1
            n += 1

        print(f"  {ds_name}: {n} test images")

    print(f"\n{'='*48}")
    print(f"Test set → {OUT_DIR}")
    print(f"  Total images : {total}")
    print(f"  Person boxes : {stats['person']}")
    print(f"  Car boxes    : {stats['car']}")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
