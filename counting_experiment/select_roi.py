"""
ROI selector — click polygon points on the first video frame.
  Enter  : close polygon and save to roi.json
  Escape : reset and start over
  Q      : quit without saving
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO   = Path(__file__).parent.parent / "Demo" / "ANMR0006.mp4"
ROI_OUT = Path(__file__).parent / "roi.json"

points: list[list[int]] = []
confirmed = False


def draw(frame_clean: np.ndarray) -> np.ndarray:
    img = frame_clean.copy()
    if len(points) >= 2:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
    for pt in points:
        cv2.circle(img, tuple(pt), 5, (0, 200, 255), -1)
    instructions = [
        "Click to add points",
        "Enter: close & save",
        "Esc: reset",
        "Q: quit",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if len(points) > 0:
        cv2.putText(img, f"{len(points)} points", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return img


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    global points, confirmed

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {VIDEO}")

    ret, frame_clean = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read first frame.")

    h, w = frame_clean.shape[:2]
    print(f"Video frame size: {w}x{h}")
    print("Click to define ROI polygon. Press Enter to save, Esc to reset, Q to quit.")

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", mouse_cb)

    while True:
        img = draw(frame_clean)
        cv2.imshow("ROI Selector", img)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("Quit without saving.")
            break

        elif key == 27:  # Escape — reset
            points = []
            print("Reset.")

        elif key == 13 or key == 10:  # Enter — close and save
            if len(points) < 3:
                print("Need at least 3 points to form a polygon.")
                continue

            # close the polygon visually for confirmation
            closed = points + [points[0]]
            confirm_img = frame_clean.copy()
            cv2.fillPoly(confirm_img,
                         [np.array(points, dtype=np.int32)],
                         (0, 255, 255))
            blended = cv2.addWeighted(frame_clean, 0.6, confirm_img, 0.4, 0)
            cv2.polylines(blended,
                          [np.array(points, dtype=np.int32)],
                          True, (0, 220, 0), 2)
            for pt in points:
                cv2.circle(blended, tuple(pt), 5, (0, 200, 255), -1)
            cv2.putText(blended, "ROI confirmed — press any key to save",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
            cv2.imshow("ROI Selector", blended)
            cv2.waitKey(0)

            ROI_OUT.write_text(json.dumps(points, indent=2))
            print(f"Saved {len(points)}-point ROI → {ROI_OUT}")
            confirmed = True
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
