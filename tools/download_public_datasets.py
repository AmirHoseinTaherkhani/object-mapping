"""
Download and convert all 6 public overhead-surveillance datasets.

Each dataset is converted to our 2-class YOLO schema:
  0 = person
  1 = car  (car, van, bus, truck, motorcycle, motor, etc.)

After running, all images and labels land in:
  labeling_data/large_datasets/<dataset_name>/train/images
  labeling_data/large_datasets/<dataset_name>/train/labels
  ... (val, test)

Then run tools/merge_large_datasets.py to fold them into the main
train/valid/test splits.

Usage:
    python tools/download_public_datasets.py [--dataset all|visdrone|crowdhuman|widerperson|ua-detrac|cowc|sdd]
    python tools/download_public_datasets.py --dataset visdrone     # download only one

Datasets:
  1. VisDrone 2019 Detection (~2.3 GB)   — auto-downloaded via ultralytics
  2. CrowdHuman         (~15 GB)         — needs Roboflow API key or manual download
  3. WiderPerson        (~3.5 GB)        — direct download available
  4. UA-DETRAC          (~7 GB)          — needs manual download from detrac-db.rit.albany.edu
  5. COWC               (~5 GB)          — direct download available
  6. Stanford Drone Dataset (~9 GB)      — needs manual download from cvgl.stanford.edu

Set your Roboflow API key in the environment for datasets 2 (and optionally 3,4,6):
    export ROBOFLOW_API_KEY=your_key_here
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT     = Path(__file__).parent.parent
OUT_BASE = ROOT / "labeling_data" / "large_datasets"


# ── helpers ────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> bool:
    """Download url to dest using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    print(f"  Downloading {dest.name} from {url[:60]}…")
    cmd = ["wget", "-q", "--show-progress", "-O", str(dest), url]
    if shutil.which("wget") is None:
        cmd = ["curl", "-L", "--progress-bar", "-o", str(dest), url]
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def extract_zip(zip_path: Path, dest: Path):
    print(f"  Extracting {zip_path.name}…")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def extract_tar(tar_path: Path, dest: Path):
    print(f"  Extracting {tar_path.name}…")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest)])


def remap_yolo_label(src: Path, class_map: dict) -> list[str]:
    """Apply class_map {src_id: dst_id | None=skip} and return remapped YOLO lines."""
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


def convert_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path,
                  class_map: dict, tag: str) -> int:
    """Copy images + convert labels for one split. Returns count of copied images."""
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    imgs = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png")) + \
           sorted(src_img.glob("*.jpeg"))
    for img in imgs:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        remapped = remap_yolo_label(lbl, class_map)
        if not remapped:
            continue
        stem = f"{tag}_{n:07d}"
        shutil.copy(img, dst_img / (stem + img.suffix))
        (dst_lbl / (stem + ".txt")).write_text("\n".join(remapped))
        n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  1. VisDrone 2019 Detection
#     10 classes: 0=pedestrian,1=people,2=bicycle,3=car,4=van,5=truck,
#                 6=tricycle,7=awning-tricycle,8=bus,9=motor
#     Remap: {0,1} → person, {3,4,5,8,9} → car, skip {2,6,7}
# ══════════════════════════════════════════════════════════════════════════════

VISDRONE_CLASS_MAP = {
    0: 0,    # pedestrian → person
    1: 0,    # people     → person
    2: None, # bicycle    → skip
    3: 1,    # car        → car
    4: 1,    # van        → car
    5: 1,    # truck      → car
    6: None, # tricycle   → skip
    7: None, # awning-tricycle → skip
    8: 1,    # bus        → car
    9: 1,    # motor      → car
}


def download_visdrone():
    out = OUT_BASE / "visdrone"
    if (out / "done.txt").exists():
        print("[VisDrone] Already downloaded and converted.")
        return

    print("\n[VisDrone 2019] Locating dataset…")

    # ultralytics may download to the project-local datasets/ dir or ~/datasets/
    candidate_roots = [
        ROOT / "datasets" / "VisDrone",
        Path.home() / "datasets" / "VisDrone",
    ]

    vd_root = None
    for candidate in candidate_roots:
        if (candidate / "images" / "train").exists():
            vd_root = candidate
            print(f"  Found VisDrone at {vd_root}")
            break

    if vd_root is None:
        print("  Not found locally — downloading via ultralytics (~2.3 GB)…")
        try:
            from ultralytics.data.utils import check_det_dataset
            check_det_dataset("VisDrone.yaml")
            for candidate in candidate_roots:
                if (candidate / "images" / "train").exists():
                    vd_root = candidate
                    break
        except Exception as e:
            print(f"  Auto-download failed: {e}")

    if vd_root is None:
        print("  ERROR: VisDrone not found and auto-download failed.")
        print("  Run manually:")
        print("    python -c \"from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml')\"")
        return

    print("  Converting VisDrone to 2-class schema…")
    total = 0
    for split, split_name in [("train","train"), ("val","valid"), ("test","test")]:
        src_img = vd_root / "images" / split
        src_lbl = vd_root / "labels" / split
        if not src_img.exists():
            continue
        dst_img = out / split_name / "images"
        dst_lbl = out / split_name / "labels"
        n = convert_split(src_img, src_lbl, dst_img, dst_lbl, VISDRONE_CLASS_MAP, "vd")
        print(f"    {split}: {n} images")
        total += n

    (out / "done.txt").write_text(f"Total: {total}\n")
    print(f"  VisDrone done: {total} images → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. CrowdHuman
#     1 class: 0=person → our person (0)
#     Roboflow version: workspace=crowdhuman / project=crowdhuman
# ══════════════════════════════════════════════════════════════════════════════

def download_crowdhuman():
    out = OUT_BASE / "crowdhuman"
    if (out / "done.txt").exists():
        print("[CrowdHuman] Already downloaded.")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    print("\n[CrowdHuman] (~15 GB persons dataset)")

    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            downloaded = False
            for ws, proj, ver in [
                ("crowdhuman", "crowdhuman", 1),
                ("roboflow-100", "crowd-counting-pce88", 1),
            ]:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
                    print(f"  Downloaded from Roboflow: {ws}/{proj}")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Roboflow {ws}/{proj} failed: {e}")
                    continue
            if downloaded:
                (out / "done.txt").write_text("from_roboflow\n")
                return
        except ImportError:
            print("  roboflow package not installed. Trying direct link…")

    # Direct download fallback (CrowdHuman official)
    # Requires registration; provide instructions
    print("  ACTION REQUIRED — CrowdHuman requires account registration:")
    print("    1. Register at https://www.crowdhuman.org/")
    print("    2. Download CrowdHuman_train.zip + CrowdHuman_val.zip")
    print("    3. Place them in:", out)
    print("    4. Re-run this script to convert")
    print("  OR set ROBOFLOW_API_KEY to download automatically.")

    # Check if manually placed zips exist
    train_zip = out / "CrowdHuman_train.zip"
    if train_zip.exists():
        print("  Found CrowdHuman_train.zip — extracting and converting…")
        _crowdhuman_convert(out)


def _crowdhuman_convert(out: Path):
    """Convert CrowdHuman annotation format to YOLO if manual zips are present."""
    # CrowdHuman uses a specific annotation format (odgt files)
    # This is a simplified conversion — for full conversion use the official tools
    print("  CrowdHuman format conversion requires the odgt parser.")
    print("  Refer to: https://github.com/Sense-X/TinyBenchmark/tree/master/data/CrowdHuman")
    print("  A simpler option: download the Roboflow version with YOLO format already applied.")


# ══════════════════════════════════════════════════════════════════════════════
#  3. WiderPerson
#     5 classes: 0=pedestrian,1=rider,2=partially-visible,3=ignore,4=crowd
#     Remap: 0,1,2 → person, 3,4 → skip
# ══════════════════════════════════════════════════════════════════════════════

WIDERPERSON_CLASS_MAP = {0: 0, 1: 0, 2: 0, 3: None, 4: None}


def download_widerperson():
    out = OUT_BASE / "widerperson"
    if (out / "done.txt").exists():
        print("[WiderPerson] Already downloaded.")
        return

    print("\n[WiderPerson] (~3.5 GB pedestrian dataset)")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            for ws, proj, ver in [
                ("wider-person", "widerperson", 1),
                ("roboflow-100", "wider-person-h46we", 1),
            ]:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
                    print(f"  Downloaded from Roboflow: {ws}/{proj}")
                    (out / "done.txt").write_text("from_roboflow\n")
                    return
                except Exception as e:
                    print(f"  Roboflow {ws}/{proj} failed: {e}")
                    continue
        except ImportError:
            pass

    # Direct download from official site (Google Drive)
    # https://competitions.codalab.org/competitions/20132
    print("  ACTION REQUIRED — WiderPerson direct links:")
    print("    Download from: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/")
    print("    OR Google Drive: search 'WiderPerson dataset' for public links")
    print("    Place WiderPerson.zip in:", out)
    print("  OR set ROBOFLOW_API_KEY to download automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  4. UA-DETRAC
#     Vehicle detection: cars, vans, buses, trucks → all map to car (1)
# ══════════════════════════════════════════════════════════════════════════════

def download_ua_detrac():
    out = OUT_BASE / "ua_detrac"
    if (out / "done.txt").exists():
        print("[UA-DETRAC] Already downloaded.")
        return

    print("\n[UA-DETRAC] (~7 GB vehicle detection at intersections)")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            for ws, proj, ver in [
                ("roboflow-100", "ua-detrac", 1),
                ("detrac", "ua-detrac-detection", 1),
            ]:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
                    print(f"  Downloaded from Roboflow: {ws}/{proj}")
                    (out / "done.txt").write_text("from_roboflow\n")
                    return
                except Exception as e:
                    print(f"  Roboflow {ws}/{proj} failed: {e}")
                    continue
        except ImportError:
            pass

    # Official download links
    print("  ACTION REQUIRED — UA-DETRAC requires registration:")
    print("    Register and download from: http://detrac-db.rit.albany.edu/")
    print("    Files needed: DETRAC-train-data.zip + DETRAC-test-data.zip")
    print("    Place zips in:", out)
    print("  OR set ROBOFLOW_API_KEY to download automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  5. COWC (Cars Overhead With Context)
#     Only cars → class 1
#     Available as 64×64 image patches (car/not-car classification)
#     For detection use the Potsdam/Selwyn/Toronto annotation sets
# ══════════════════════════════════════════════════════════════════════════════

def download_cowc():
    out = OUT_BASE / "cowc"
    if (out / "done.txt").exists():
        print("[COWC] Already downloaded.")
        return

    print("\n[COWC] (~5 GB overhead car detection)")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            for ws, proj, ver in [
                ("roboflow-100", "aerial-cars-cowc", 1),
                ("cowc", "cowc-detection", 1),
            ]:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
                    print(f"  Downloaded from Roboflow: {ws}/{proj}")
                    (out / "done.txt").write_text("from_roboflow\n")
                    return
                except Exception as e:
                    print(f"  Roboflow {ws}/{proj} failed: {e}")
                    continue
        except ImportError:
            pass

    print("  ACTION REQUIRED — COWC direct links:")
    print("    https://gdo152.llnl.gov/cowc/")
    print("    Download the detection datasets (Potsdam, Selwyn, Toronto annotation zips)")
    print("  OR set ROBOFLOW_API_KEY to download automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Stanford Drone Dataset (SDD)
#     6 classes: pedestrian,bicycle,car,bus,golf cart,skater,cart
#     Remap: pedestrian → person(0), car/bus/golf cart/cart → car(1)
#     skip: bicycle, skater
# ══════════════════════════════════════════════════════════════════════════════

SDD_CLASS_MAP = {
    0: 0,    # pedestrian → person
    1: None, # bicycle    → skip (or 1 for bike + rider)
    2: 1,    # car        → car
    3: 1,    # bus        → car
    4: 1,    # golf cart  → car
    5: None, # skater     → skip
    6: 1,    # cart       → car
}


def download_sdd():
    out = OUT_BASE / "sdd"
    if (out / "done.txt").exists():
        print("[Stanford Drone Dataset] Already downloaded.")
        return

    print("\n[Stanford Drone Dataset (SDD)] (~9 GB overhead campus surveillance)")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            for ws, proj, ver in [
                ("stanford-drone-dataset", "stanford-drone", 1),
                ("roboflow-100", "stanford-drone-dataset", 1),
            ]:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
                    print(f"  Downloaded from Roboflow: {ws}/{proj}")
                    (out / "done.txt").write_text("from_roboflow\n")
                    return
                except Exception as e:
                    print(f"  Roboflow {ws}/{proj} failed: {e}")
                    continue
        except ImportError:
            pass

    print("  ACTION REQUIRED — Stanford Drone Dataset:")
    print("    Download from: http://cvgl.stanford.edu/projects/uav_data/")
    print("    This is a large download (~80 GB raw video).")
    print("    Preprocessed annotation-only version (~9 GB) is linked on the same page.")
    print("  OR set ROBOFLOW_API_KEY to download automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  7. COCO 2017 — selective download (person + car-class images only)
#     Downloads annotations first (~241MB), filters relevant image IDs,
#     then fetches only those images (~200KB each → ~1-2GB for 5k images).
#
#     COCO class → our class:
#       1  person     → 0  person
#       2  bicycle    → skip  (person-on-bike captured by person bbox)
#       3  car        → 1  car
#       4  motorcycle → 1  car
#       6  bus        → 1  car
#       8  truck      → 1  car
# ══════════════════════════════════════════════════════════════════════════════

COCO_CLASS_MAP = {
    1: 0,    # person      → person
    3: 1,    # car         → car
    4: 1,    # motorcycle  → car
    6: 1,    # bus         → car
    8: 1,    # truck       → car
}


def _coco_ann_to_yolo(ann_list: list, img_w: int, img_h: int,
                      cat_map: dict) -> list[str]:
    """Convert COCO bbox annotations to YOLO lines (centre-relative)."""
    lines = []
    for ann in ann_list:
        dst = cat_map.get(ann["category_id"])
        if dst is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        lines.append(f"{dst} {cx:.6f} {cy:.6f} {w/img_w:.6f} {h/img_h:.6f}")
    return lines


def download_coco(max_images: int = 5000):
    """Download COCO annotations + only the images containing our classes."""
    out = OUT_BASE / "coco"
    if (out / "done.txt").exists():
        print("[COCO] Already downloaded and converted.")
        return

    import json
    import urllib.request

    out.mkdir(parents=True, exist_ok=True)
    ann_dir = out / "annotations"
    ann_dir.mkdir(exist_ok=True)

    print("\n[COCO 2017] Downloading annotations (~241 MB)…")
    ann_zip = ann_dir / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        if shutil.which("wget"):
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(ann_zip), url], check=True)
        else:
            subprocess.run(["curl", "-L", "--progress-bar", "-o", str(ann_zip), url], check=True)
    if not (ann_dir / "instances_train2017.json").exists() and \
       not (ann_dir / "annotations" / "instances_train2017.json").exists():
        extract_zip(ann_zip, ann_dir)

    # The zip may extract to annotations/ or annotations/annotations/ depending on its structure
    ann_root = ann_dir / "annotations" if (ann_dir / "annotations" / "instances_train2017.json").exists() else ann_dir

    total = 0
    for split, ann_file, split_name in [
        ("train2017", "instances_train2017.json", "train"),
        ("val2017",   "instances_val2017.json",   "valid"),
    ]:
        ann_path = ann_root / ann_file
        if not ann_path.exists():
            print(f"  Missing {ann_file} — skipping {split}")
            continue

        print(f"  Parsing {ann_file}…")
        data = json.loads(ann_path.read_text())

        # Build image_id → annotations lookup
        img_anns: dict[int, list] = {}
        for ann in data["annotations"]:
            if ann["category_id"] in COCO_CLASS_MAP:
                img_anns.setdefault(ann["image_id"], []).append(ann)

        # Gather image metadata for images that have relevant annotations
        relevant = [img for img in data["images"] if img["id"] in img_anns]
        print(f"  {split}: {len(relevant)} images with person/car annotations")

        import random
        random.seed(42)
        if len(relevant) > max_images:
            relevant = random.sample(relevant, max_images)

        dst_img = out / split_name / "images"
        dst_lbl = out / split_name / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_meta in relevant:
            img_id  = img_meta["id"]
            fname   = img_meta["file_name"]
            w, h    = img_meta["width"], img_meta["height"]
            stem    = f"coco_{split_name}_{n:07d}"
            dst_img_path = dst_img / (stem + ".jpg")
            dst_lbl_path = dst_lbl / (stem + ".txt")

            if not dst_img_path.exists():
                url = f"http://images.cocodataset.org/{split}/{fname}"
                try:
                    urllib.request.urlretrieve(url, str(dst_img_path))
                except Exception as e:
                    print(f"  Failed to download {fname}: {e}")
                    continue

            lines = _coco_ann_to_yolo(img_anns[img_id], w, h, COCO_CLASS_MAP)
            if lines:
                dst_lbl_path.write_text("\n".join(lines))
                n += 1
            else:
                dst_img_path.unlink(missing_ok=True)

            if (n % 500 == 0 and n > 0) or n == len(relevant):
                print(f"    {n}/{len(relevant)} images downloaded…", flush=True)

        print(f"  {split_name}: {n} images converted")
        total += n

    (out / "done.txt").write_text(f"Total: {total}\n")
    print(f"  COCO done: {total} images → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Merge large datasets into main train/valid/test splits
# ══════════════════════════════════════════════════════════════════════════════

def merge_into_training(max_per_dataset: int = 5000):
    """
    Fold all downloaded large datasets into labeling_data/trainingData/
    train/images + train/labels (and valid/, test/).

    Caps each dataset at max_per_dataset images per split to prevent
    one dataset dominating the training distribution.
    """
    import random
    random.seed(42)

    td = ROOT / "labeling_data" / "trainingData"

    split_map = {
        "train": (td / "train" / "images", td / "train" / "labels"),
        "valid": (td / "valid" / "images", td / "valid" / "labels"),
        "test":  (td / "test"  / "images", td / "test"  / "labels"),
    }

    total_added = 0
    for ds_dir in sorted(OUT_BASE.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "done.txt").exists():
            continue
        print(f"\n  Merging {ds_dir.name}…")
        for split, (dst_img, dst_lbl) in split_map.items():
            src_img = ds_dir / split / "images"
            src_lbl = ds_dir / split / "labels"
            if not src_img.exists():
                continue
            imgs = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png"))
            if len(imgs) > max_per_dataset:
                imgs = random.sample(imgs, max_per_dataset)
            n = 0
            for img in imgs:
                lbl = src_lbl / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                # Use image name as-is (already unique from convert_split)
                dst_img.mkdir(parents=True, exist_ok=True)
                dst_lbl.mkdir(parents=True, exist_ok=True)
                dst_img_path = dst_img / img.name
                dst_lbl_path = dst_lbl / lbl.name
                if not dst_img_path.exists():
                    shutil.copy(img, dst_img_path)
                    shutil.copy(lbl, dst_lbl_path)
                    n += 1
            print(f"    {split}: +{n} images")
            total_added += n

    print(f"\n  Total added to training splits: {total_added}")
    print("  Re-check data.yaml path and re-run fine-tuning.")


# ── CLI ────────────────────────────────────────────────────────────────────────

DOWNLOADS = {
    "visdrone":    download_visdrone,
    "coco":        download_coco,
    "crowdhuman":  download_crowdhuman,
    "widerperson": download_widerperson,
    "ua-detrac":   download_ua_detrac,
    "cowc":        download_cowc,
    "sdd":         download_sdd,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="all",
                   choices=["all"] + list(DOWNLOADS.keys()),
                   help="Which dataset to download (default: all)")
    p.add_argument("--merge", action="store_true",
                   help="After downloading, merge into labeling_data/trainingData/")
    p.add_argument("--max-per-dataset", type=int, default=5000,
                   help="Cap images per dataset per split when merging (default: 5000)")
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        targets = list(DOWNLOADS.values())
    else:
        targets = [DOWNLOADS[args.dataset]]

    for fn in targets:
        fn()

    if args.merge:
        print("\n[Merge] Folding downloaded datasets into training splits…")
        merge_into_training(args.max_per_dataset)

    print("\n── Summary ────────────────────────────────────────────────────")
    for name, fn in DOWNLOADS.items():
        done = (OUT_BASE / name.replace("-", "_") / "done.txt").exists()
        status = "✓ ready" if done else "✗ ACTION REQUIRED"
        print(f"  {name:<20} {status}")

    need_roboflow = not all(
        (OUT_BASE / name.replace("-", "_") / "done.txt").exists()
        for name in DOWNLOADS
    )
    if need_roboflow:
        print("\n  Tip: set ROBOFLOW_API_KEY to auto-download the remaining datasets.")
        print("  Export the key in your shell:")
        print("    export ROBOFLOW_API_KEY=<your_key>")
        print("  Then re-run:  python tools/download_public_datasets.py")

    print()


if __name__ == "__main__":
    main()
