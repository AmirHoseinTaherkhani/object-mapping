"""
Evaluate pretrained YOLO11 models on our 2-class test set with COCO class remapping.

Standard model.val() won't work because pretrained models output 80 COCO classes
while our GT labels use 2 classes (person=0, car=1). This script:
  1. Runs inference on each test image
  2. Remaps COCO predictions: person(0)→0, car(2)/motorcycle(3)/bus(5)/truck(7)→1
  3. Matches against our GT labels and computes standard mAP50 / mAP50-95

Usage:
    python tools/eval_pretrained_map.py --weights models/weights/yolo11s.pt
    python tools/eval_pretrained_map.py --weights models/weights/yolo11m.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT     = Path(__file__).parent.parent
TEST_DIR = ROOT / "labeling_data" / "trainingData" / "test"
RESULTS  = ROOT / "experiments" / "yolo11_experiments" / "results"

# COCO class IDs → our 2-class scheme
COCO_TO_OURS = {0: 0, 2: 1, 3: 1, 5: 1, 7: 1}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
NC = 2
CLASS_NAMES = ["person", "car"]


# ── geometry ──────────────────────────────────────────────────────────────────

def box_iou(b1, b2):
    """IoU between [N,4] and [M,4] boxes in xyxy format. Returns [N,M]."""
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    iw  = np.maximum(0, ix2 - ix1)
    ih  = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


def load_gt(label_path, w, h):
    """Load YOLO label file → (cls int[], xyxy float[N,4]) in pixel coords."""
    if not label_path.exists():
        return np.empty(0, int), np.empty((0, 4))
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return np.empty(0, int), np.empty((0, 4))
    arr = np.array([[float(v) for v in l.split()] for l in lines])
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = (cx - bw / 2) * w;  y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w;  y2 = (cy + bh / 2) * h
    return cls, np.stack([x1, y1, x2, y2], 1)


# ── AP computation ────────────────────────────────────────────────────────────

def compute_ap(recall, precision):
    """All-points interpolated AP."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def per_class_ap(tp_all, conf_all, pred_cls_all, target_cls_all):
    """
    Returns (ap50, ap50_95, best_prec, best_rec) arrays of length NC.
    tp_all: [N_det, n_iou_thresh] bool
    """
    n_iou = len(IOU_THRESHOLDS)
    ap50     = np.zeros(NC)
    ap50_95  = np.zeros(NC)
    prec_out = np.zeros(NC)
    rec_out  = np.zeros(NC)

    for c in range(NC):
        n_gt_c = int((target_cls_all == c).sum())
        mask = pred_cls_all == c
        if not mask.any() or n_gt_c == 0:
            continue

        tp_c   = tp_all[mask]            # [n_det_c, n_iou]
        conf_c = conf_all[mask]
        order  = np.argsort(-conf_c)
        tp_c   = tp_c[order]

        aps = []
        for ti in range(n_iou):
            ctp = np.cumsum(tp_c[:, ti]).astype(float)
            cfp = np.arange(1, len(ctp) + 1) - ctp
            p   = ctp / (ctp + cfp + 1e-7)
            r   = ctp / n_gt_c
            aps.append(compute_ap(r, p))

        ap50[c]    = aps[0]
        ap50_95[c] = float(np.mean(aps))

        # precision / recall at best F1
        ctp = np.cumsum(tp_c[:, 0]).astype(float)
        cfp = np.arange(1, len(ctp) + 1) - ctp
        p   = ctp / (ctp + cfp + 1e-7)
        r   = ctp / n_gt_c
        f1  = 2 * p * r / (p + r + 1e-7)
        best = int(np.argmax(f1))
        prec_out[c] = p[best]
        rec_out[c]  = r[best]

    return ap50, ap50_95, prec_out, rec_out


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(weights: str, device: str = "mps", conf: float = 0.001, iou: float = 0.6):
    weights_path = Path(weights)
    print(f"\n{'='*60}")
    print(f"Evaluating: {weights_path.name}  (pretrained COCO → 2-class remap)")
    print(f"{'='*60}")

    model  = YOLO(str(weights_path))
    images = sorted((TEST_DIR / "images").glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    n      = len(images)
    print(f"Test images: {n}")

    tp_list  = []
    conf_list, pred_cls_list, target_cls_list = [], [], []
    n_iou = len(IOU_THRESHOLDS)
    BATCH = 32
    t0 = time.time()

    for batch_start in range(0, n, BATCH):
        batch_paths = images[batch_start: batch_start + BATCH]
        if batch_start % 512 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(batch_start, 1) * (n - batch_start)
            print(f"  {batch_start:>4}/{n}  ETA {eta:.0f}s")

        results = model.predict(batch_paths, verbose=False, conf=conf, iou=iou,
                                device=device)

        for result, img_path in zip(results, batch_paths):
            h, w = result.orig_shape

            label_path = TEST_DIR / "labels" / (img_path.stem + ".txt")
            gt_cls, gt_boxes = load_gt(label_path, w, h)
            target_cls_list.extend(gt_cls.tolist())

            # ── predictions ──────────────────────────────────────────────────
            if result.boxes is not None and len(result.boxes):
                pcoco = result.boxes.cls.cpu().numpy().astype(int)
                pconf = result.boxes.conf.cpu().numpy()
                pxyxy = result.boxes.xyxy.cpu().numpy()

                keep = np.array([c in COCO_TO_OURS for c in pcoco])
                if keep.any():
                    pcls  = np.array([COCO_TO_OURS[c] for c in pcoco[keep]])
                    pconf = pconf[keep]
                    pxyxy = pxyxy[keep]
                else:
                    pcls = pconf = pxyxy = np.empty(0)
            else:
                pcls = pconf = pxyxy = np.empty(0)

            n_det = len(pconf)
            n_gt  = len(gt_cls)

            if n_det == 0:
                continue

            conf_list.extend(pconf.tolist())
            pred_cls_list.extend(pcls.tolist())

            tp = np.zeros((n_det, n_iou), bool)

            if n_gt > 0:
                iou_mat = box_iou(pxyxy, gt_boxes)  # [n_det, n_gt]
                for ti, iou_t in enumerate(IOU_THRESHOLDS):
                    matched = set()
                    for di in np.argsort(-pconf):
                        dc = int(pcls[di])
                        best_val, best_gi = iou_t, -1
                        for gi in range(n_gt):
                            if gi in matched or gt_cls[gi] != dc:
                                continue
                            if iou_mat[di, gi] >= best_val:
                                best_val, best_gi = iou_mat[di, gi], gi
                        if best_gi >= 0:
                            tp[di, ti] = True
                            matched.add(best_gi)

            tp_list.append(tp)

    elapsed = time.time() - t0
    print(f"Inference + matching done in {elapsed:.1f}s")

    if not conf_list:
        print("No detections — check model and dataset paths.")
        return None

    tp_all       = np.concatenate(tp_list, 0)
    conf_all     = np.array(conf_list)
    pred_cls_all = np.array(pred_cls_list, int)
    target_cls_all = np.array(target_cls_list, int)

    ap50, ap50_95, prec, rec = per_class_ap(tp_all, conf_all, pred_cls_all, target_cls_all)

    mAP50    = float(np.mean(ap50))
    mAP50_95 = float(np.mean(ap50_95))
    mPrec    = float(np.mean(prec))
    mRec     = float(np.mean(rec))

    print(f"\nResults ({weights_path.name}):")
    print(f"  mAP50      = {mAP50:.4f}")
    print(f"  mAP50-95   = {mAP50_95:.4f}")
    print(f"  Precision  = {mPrec:.4f}")
    print(f"  Recall     = {mRec:.4f}")
    for c in range(NC):
        print(f"  AP50[{CLASS_NAMES[c]:6s}] = {ap50[c]:.4f}   AP50-95 = {ap50_95[c]:.4f}  "
              f"P={prec[c]:.4f}  R={rec[c]:.4f}")

    result_dict = {
        "model": weights_path.stem,
        "mode": "pretrained",
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "precision": mPrec,
        "recall": mRec,
        "per_class": {
            CLASS_NAMES[c]: {
                "ap50": float(ap50[c]), "ap50_95": float(ap50_95[c]),
                "precision": float(prec[c]), "recall": float(rec[c]),
            }
            for c in range(NC)
        },
        "coco_remap": COCO_TO_OURS,
        "n_test_images": n,
    }

    out_path = RESULTS / f"pretrained_{weights_path.stem}_map.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=2))
    print(f"\nSaved: {out_path}")
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device",  default="mps")
    parser.add_argument("--conf",    type=float, default=0.001)
    parser.add_argument("--iou",     type=float, default=0.6)
    args = parser.parse_args()
    evaluate(args.weights, args.device, args.conf, args.iou)
