"""
Extract ~200 diverse frames from ANMR0006.mp4 for Roboflow labeling.

Strategy:
  1. Scan the full video in uniform segments.
  2. Within each segment pick the frame with the most detections (using best.pt).
  3. Skip frames too visually similar to the previous saved frame.
  4. Save to labeling_data/images/ ready for Roboflow upload.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT    = Path(__file__).parent
VIDEO      = PROJECT / "Demo/ANMR0006.mp4"
OUT_DIR    = PROJECT / "labeling_data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES   = 200
MIN_DIFF_THRESH = 2.0    # fixed camera — even small changes are meaningful
CONF            = 0.20


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two frames (downsampled for speed)."""
    a_s = cv2.resize(a, (320, 180))
    b_s = cv2.resize(b, (320, 180))
    return float(np.mean(np.abs(a_s.astype(np.int16) - b_s.astype(np.int16))))


def main():
    cap   = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames  ({total/fps:.0f}s at {fps:.0f}fps)")

    model = YOLO(str(PROJECT / "models/weights/best.pt"))

    # Divide video into TARGET_FRAMES equal segments; pick best frame per segment
    segment_size = total // TARGET_FRAMES
    print(f"Segment size: {segment_size} frames  (~{segment_size/fps:.1f}s each)")
    print(f"Scoring frames with best.pt (conf={CONF})...\n")

    saved       = 0
    prev_frame  = None
    skipped_sim = 0

    for seg in range(TARGET_FRAMES):
        seg_start = seg * segment_size
        seg_end   = min(seg_start + segment_size, total)

        # Sample up to 6 candidate frames within the segment
        candidates = np.linspace(seg_start, seg_end - 1, min(6, seg_end - seg_start), dtype=int)

        best_frame = None
        best_score = -1

        for fi in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue

            results  = model.predict(frame, conf=CONF, verbose=False)
            n_detect = len(results[0].boxes) if results[0].boxes is not None else 0

            # Score = detections + small bonus for visual variety vs previous saved
            variety = mean_diff(frame, prev_frame) if prev_frame is not None else 999
            score   = n_detect * 10 + min(variety, 50)

            if score > best_score:
                best_score = score
                best_frame = (frame, int(fi), n_detect, variety)

        if best_frame is None:
            continue

        frame, fi, n_det, variety = best_frame

        # Skip if too similar to last saved frame
        if prev_frame is not None and variety < MIN_DIFF_THRESH:
            skipped_sim += 1
            continue

        fname = OUT_DIR / f"frame_{fi:06d}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        prev_frame = frame
        saved += 1

        if saved % 20 == 0 or saved == 1:
            print(f"  [{saved:3d}] frame {fi:6d}  detections={n_det}  diff={variety:.1f}")

    cap.release()

    print(f"\nDone.")
    print(f"  Saved  : {saved} frames → {OUT_DIR}")
    print(f"  Skipped: {skipped_sim} (too similar)")
    print()
    print("Next steps:")
    print("  1. Go to https://roboflow.com → New Project → Object Detection")
    print("     Classes: person, car")
    print(f"  2. Upload the {saved} images from: {OUT_DIR}")
    print("  3. Label each image (draw boxes around persons and cars)")
    print("  4. Export as 'YOLOv8 PyTorch' format → download ZIP")
    print("  5. Unzip into: labeling_data/dataset/")
    print("  6. Run: python finetune_on_camera.py")


if __name__ == "__main__":
    main()
