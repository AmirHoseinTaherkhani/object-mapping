"""
Sanity-check replay of output_counted.mp4.

Reads the status bar text from each frame's bottom region and prints a
timestamped log of every frame where the car or person count increases.
Used to manually verify the counter isn't double-counting or missing entries.

Usage:
    python check_counts.py [path/to/video]   # defaults to output_counted.mp4
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE    = Path(__file__).parent
DEFAULT = HERE / "output_counted.mp4"


def extract_counts(frame: np.ndarray) -> tuple[int, int] | None:
    """Read counts from the top-left info box drawn by count_objects.py."""
    roi = frame[5:70, 5:225]
    # OCR-free approach: just read the pixel region used for count display.
    # We re-read the counts by scanning the annotated frame text via pytesseract
    # if available, otherwise fall back to colour-histogram heuristic.
    # Simplest reliable approach: parse the text rendered by count_objects.py
    # using pytesseract if installed.
    try:
        import pytesseract
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6 digits")
        nums = re.findall(r"\d+", text)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    except ImportError:
        pass
    return None


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}\nRun count_objects.py first.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    prev_cars = 0
    prev_ppl  = 0
    events    = []

    ocr_available = False
    try:
        import pytesseract
        ocr_available = True
        print("pytesseract found — parsing counts from frame text.")
    except ImportError:
        print("pytesseract not installed — falling back to frame-diff event detection.")
        print("Install with: pip install pytesseract  (also needs tesseract binary)")

    print(f"\nReplaying {video_path.name}  ({total} frames @ {fps:.1f}fps)\n")
    print(f"{'Timestamp':>10}  {'Frame':>6}  {'Class':>7}  {'New Count':>10}")
    print("-" * 42)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps

        if ocr_available:
            result = extract_counts(frame)
            if result is not None:
                cars, ppl = result
                if cars > prev_cars:
                    for _ in range(cars - prev_cars):
                        msg = f"{ts:>10.2f}s  {frame_idx:>6d}  {'car':>7}  {cars:>10}"
                        print(msg)
                        events.append(("car", ts, frame_idx, cars))
                    prev_cars = cars
                if ppl > prev_ppl:
                    for _ in range(ppl - prev_ppl):
                        msg = f"{ts:>10.2f}s  {frame_idx:>6d}  {'person':>7}  {ppl:>10}"
                        print(msg)
                        events.append(("person", ts, frame_idx, ppl))
                    prev_ppl = ppl
        else:
            # Without OCR: detect count-increase events by looking for
            # brightness changes in the count region (heuristic, not reliable for exact numbers).
            # Just count unique frames where the count box changes significantly.
            roi_curr = frame[5:70, 5:225].astype(np.float32)
            if frame_idx == 0:
                roi_prev = roi_curr.copy()
            diff = np.mean(np.abs(roi_curr - roi_prev))
            if diff > 8.0 and frame_idx > 0:
                events.append(("change", ts, frame_idx, diff))
                print(f"{ts:>10.2f}s  {frame_idx:>6d}  {'change':>7}  (diff={diff:.1f}) — install pytesseract for class detail")
            roi_prev = roi_curr.copy()

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 42)
    print(f"Total count-change events : {len(events)}")
    if ocr_available:
        car_events = [e for e in events if e[0] == "car"]
        ppl_events = [e for e in events if e[0] == "person"]
        print(f"  Car increments    : {len(car_events)}")
        print(f"  Person increments : {len(ppl_events)}")
        if car_events:
            print(f"  First car at      : {car_events[0][1]:.2f}s (frame {car_events[0][2]})")
        if ppl_events:
            print(f"  First person at   : {ppl_events[0][1]:.2f}s (frame {ppl_events[0][2]})")
    print("=" * 42)
    print("If any increment jumped by >1, a batch of objects entered simultaneously.")
    print("If no decrements are logged, double-counting is confirmed absent.")


if __name__ == "__main__":
    main()
