"""
Generate two PowerPoint decks for the ObjectMapping project.

  docs/slides_limitations.pptx   — limitations, priority order, solutions
  docs/slides_engineering.pptx   — technical challenges, fixes, reverted attempts

Run:
    python tools/make_slides.py
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

OUT = Path(__file__).parent.parent / "docs"

# ── colour palette ─────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1A, 0x23, 0x3A)   # slide background / title bar
BLUE   = RGBColor(0x27, 0x5E, 0xA0)   # accent / headers
TEAL   = RGBColor(0x1A, 0x8A, 0x8A)   # secondary accent
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LGREY  = RGBColor(0xF2, 0xF4, 0xF7)   # table row alt
MGREY  = RGBColor(0xCC, 0xD1, 0xD9)   # divider / border
GREEN  = RGBColor(0x27, 0xAE, 0x60)
AMBER  = RGBColor(0xE6, 0x7E, 0x22)
RED    = RGBColor(0xC0, 0x39, 0x2B)
DKTEXT = RGBColor(0x1A, 0x23, 0x3A)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── helpers ────────────────────────────────────────────────────────────────────
def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def fill_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, fill=None, line=None):
    shape = slide.shapes.add_shape(1, x, y, w, h)   # MSO_SHAPE_TYPE.RECTANGLE = 1
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, x, y, w, h, size=18, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return txb


def title_slide(prs, title, subtitle):
    slide = blank_slide(prs)
    fill_bg(slide, NAVY)
    # accent bar
    add_rect(slide, Inches(0), Inches(3.1), SLIDE_W, Inches(0.08), fill=TEAL)
    add_text(slide, title,    Inches(1), Inches(1.5), Inches(11.3), Inches(1.4),
             size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, subtitle, Inches(1), Inches(3.3), Inches(11.3), Inches(0.7),
             size=22, color=MGREY, align=PP_ALIGN.CENTER)
    add_text(slide, "ObjectMapping Project  |  UNH",
             Inches(1), Inches(6.8), Inches(11.3), Inches(0.4),
             size=13, color=MGREY, align=PP_ALIGN.CENTER)
    return slide


def section_slide(prs, heading):
    slide = blank_slide(prs)
    fill_bg(slide, BLUE)
    add_rect(slide, Inches(0), Inches(3.4), SLIDE_W, Inches(0.06), fill=TEAL)
    add_text(slide, heading, Inches(1.5), Inches(2.6), Inches(10.3), Inches(1.2),
             size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    return slide


def content_slide(prs, title, body_fn):
    """White slide with a navy title bar; body_fn(slide) draws the content."""
    slide = blank_slide(prs)
    fill_bg(slide, WHITE)
    # title bar
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.1), fill=NAVY)
    add_text(slide, title, Inches(0.35), Inches(0.15), Inches(12.6), Inches(0.8),
             size=24, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    body_fn(slide)
    return slide


def add_table_row(slide, cols, y, row_h, col_widths, col_xs,
                  font_sizes, bolds, colors, bg=None, text_color=DKTEXT):
    if bg:
        add_rect(slide, col_xs[0], y, sum(col_widths), row_h, fill=bg)
    for i, (txt, sz, bold, x, w) in enumerate(
            zip(cols, font_sizes, bolds, col_xs, col_widths)):
        tc = DKTEXT if text_color == DKTEXT else text_color
        add_text(slide, txt, x + Inches(0.08), y + Inches(0.05),
                 w - Inches(0.16), row_h - Inches(0.05),
                 size=sz, bold=bold, color=tc, align=PP_ALIGN.LEFT)


# ══════════════════════════════════════════════════════════════════════════════
#  DECK 1 — LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
def make_limitations_deck():
    prs = new_prs()

    # ── TITLE ──────────────────────────────────────────────────────────────────
    title_slide(prs,
                "Model Limitations & Roadmap",
                "Current weaknesses, priority order, and path to a general-purpose model")

    # ── SLIDE 2: Goal shift ────────────────────────────────────────────────────
    def goal_body(slide):
        boxes = [
            (AMBER, "Previous goal",
             "A model fine-tuned for\none specific camera.\nHigh accuracy on that\ncamera only."),
            (BLUE,  "New goal",
             "A general overhead\nsurveillance model.\nDeploys on any fixed\ncamera without retraining."),
            (GREEN, "Why it matters",
             "Camera-specific tuning\nis a dead end.\nGeneralisation unlocks\nreal-world deployment."),
        ]
        xs = [Inches(0.5), Inches(4.67), Inches(8.84)]
        for (color, heading, body), x in zip(boxes, xs):
            add_rect(slide, x, Inches(1.3), Inches(3.8), Inches(5.5), fill=color)
            add_text(slide, heading, x+Inches(0.2), Inches(1.5), Inches(3.4),
                     Inches(0.6), size=18, bold=True, color=WHITE)
            add_text(slide, body,   x+Inches(0.2), Inches(2.2), Inches(3.4),
                     Inches(4.3), size=15, color=WHITE)

    content_slide(prs, "Goal Has Changed — This Changes Everything", goal_body)

    # ── SLIDE 3: Priority table ────────────────────────────────────────────────
    def priority_body(slide):
        headers = ["Priority", "Limitation", "Effect on System", "Effort"]
        col_ws  = [Inches(0.9), Inches(3.5), Inches(5.5), Inches(2.9)]
        col_xs  = [Inches(0.15)]
        for w in col_ws[:-1]:
            col_xs.append(col_xs[-1] + w)
        row_h = Inches(0.52)
        y0    = Inches(1.25)

        # header row
        add_rect(slide, col_xs[0], y0, sum(col_ws), row_h, fill=NAVY)
        for txt, x, w in zip(headers, col_xs, col_ws):
            add_text(slide, txt, x+Inches(0.08), y0+Inches(0.07), w-Inches(0.16),
                     row_h-Inches(0.07), size=14, bold=True, color=WHITE)

        rows = [
            ("1", "No accuracy metric",       "Cannot measure any improvement",          "Low",    GREEN),
            ("2", "Training data too narrow",  "Root cause of most other failures",       "Medium", RED),
            ("3", "Model too small (YOLOv8s)", "Insufficient capacity to generalise",     "Low",    AMBER),
            ("4", "No multi-camera config",    "Blocks general deployment",               "Low",    AMBER),
            ("5", "Tracker is IoU-only",       "ID switches on crossing paths",           "Low",    AMBER),
            ("6", "Person detection weak",     "Resolved by #2; add class weighting if needed", "Low", GREEN),
            ("7", "Car false positives",       "Threshold search as quick fix",           "Low",    GREEN),
            ("8", "Ghost zone edge cases",     "Self-resolve with better detector",       "Medium", MGREY),
        ]
        for i, (pri, lim, effect, effort, dot_color) in enumerate(rows):
            bg = LGREY if i % 2 == 0 else WHITE
            y  = y0 + row_h * (i + 1)
            add_rect(slide, col_xs[0], y, sum(col_ws), row_h, fill=bg)
            # priority circle
            add_rect(slide, col_xs[0]+Inches(0.15), y+Inches(0.1),
                     Inches(0.32), Inches(0.32), fill=dot_color)
            add_text(slide, pri, col_xs[0]+Inches(0.18), y+Inches(0.06),
                     Inches(0.28), Inches(0.38), size=13, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER)
            for txt, x, w in zip([lim, effect, effort], col_xs[1:], col_ws[1:]):
                add_text(slide, txt, x+Inches(0.08), y+Inches(0.07),
                         w-Inches(0.16), row_h-Inches(0.1), size=12.5, color=DKTEXT)

    content_slide(prs, "Limitations — Priority Order", priority_body)

    # ── SLIDE 4: The single biggest fix ────────────────────────────────────────
    def data_body(slide):
        # left: problem
        add_rect(slide, Inches(0.3), Inches(1.3), Inches(5.8), Inches(5.6), fill=LGREY)
        add_text(slide, "Current Training Data",
                 Inches(0.5), Inches(1.4), Inches(5.4), Inches(0.5),
                 size=16, bold=True, color=BLUE)
        items = [
            "~14 000 images from ONE camera",
            "One intersection, one city",
            "Limited lighting variation",
            "Cars: strongly over-represented",
            "Result: brittle, camera-specific model",
        ]
        for j, item in enumerate(items):
            add_text(slide, f"✗  {item}",
                     Inches(0.6), Inches(2.0)+Pt(22)*j*2, Inches(5.2), Inches(0.45),
                     size=13.5, color=RED)

        # right: solution
        add_rect(slide, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.6), fill=NAVY)
        add_text(slide, "Target Training Data (~230k images)",
                 Inches(6.7), Inches(1.4), Inches(6.1), Inches(0.5),
                 size=16, bold=True, color=WHITE)
        datasets = [
            ("VisDrone 2019 (full)",       "Diverse drone overhead, multi-city"),
            ("Stanford Drone Dataset",     "Fixed overhead — closest to our camera"),
            ("UA-DETRAC",                  "140k+ frames, 100+ real intersections"),
            ("COWC",                       "Aerial cars, 6 cities"),
            ("CrowdHuman",                 "Dense, occluded pedestrians"),
            ("WiderPerson",                "Overhead + elevated pedestrian scenes"),
        ]
        for j, (name, desc) in enumerate(datasets):
            y = Inches(2.05) + Pt(18) * j * 2.6
            add_text(slide, f"✓  {name}", Inches(6.7), y, Inches(6.0), Inches(0.35),
                     size=13, bold=True, color=GREEN)
            add_text(slide, f"   {desc}", Inches(6.7), y+Inches(0.28), Inches(6.0),
                     Inches(0.28), size=11.5, color=MGREY)

    content_slide(prs, "Priority #2 — Diverse Training Data (Root Cause Fix)", data_body)

    # ── SLIDE 5: Model + Tracker upgrades ──────────────────────────────────────
    def upgrade_body(slide):
        for x, color, heading, items in [
            (Inches(0.3), BLUE, "Model Upgrade  (Priority #3)",
             ["YOLOv8s → YOLOv8m",
              "11M → 25M parameters",
              "Fits within ANE real-time budget",
              "Better generalisation capacity",
              "One line change in finetune_on_camera.py",
              "Re-export to CoreML after training"]),
            (Inches(6.8), NAVY, "Tracker Upgrade  (Priority #5)",
             ["ByteTrack → BoT-SORT",
              "ByteTrack: IoU matching only",
              "BoT-SORT: IoU + appearance features",
              "Maintains correct IDs when two objects cross",
              "Already in BoxMOT — few lines to swap",
              "Test with tools/test_bytetrack.py"]),
        ]:
            add_rect(slide, x, Inches(1.3), Inches(6.1), Inches(5.6), fill=color)
            add_text(slide, heading, x+Inches(0.25), Inches(1.45), Inches(5.6),
                     Inches(0.55), size=17, bold=True, color=WHITE)
            for j, item in enumerate(items):
                add_text(slide, f"→  {item}",
                         x+Inches(0.25), Inches(2.15)+Inches(0.57)*j,
                         Inches(5.6), Inches(0.5), size=13.5, color=WHITE)

    content_slide(prs, "Quick Wins — Model Size and Tracker", upgrade_body)

    # ── SLIDE 6: Ghost zone edge cases ─────────────────────────────────────────
    def ghost_body(slide):
        cases = [
            ("Close car double-counted",
             "Ghost radius (180px) too small for centroid jump during fragmentation",
             "Elliptical ghost zone oriented in direction of motion  (~20 lines)"),
            ("Person double-counted on ROI re-entry",
             "Ghost at exit point doesn't cover re-entry from different angle",
             "Boundary-segment ghost covers a stretch of the ROI edge"),
            ("Stopped car double-counted",
             "Ghost expires (~0.75 s) before stopped car resumes — no suppression",
             "Stationary-object memory dict: keeps position suppressed while still (~50 lines)"),
        ]
        add_text(slide, "Lower priority — a stronger detector fragments less, reducing these naturally.",
                 Inches(0.3), Inches(1.2), Inches(12.7), Inches(0.4),
                 size=13, color=AMBER, bold=True)
        for j, (title, cause, fix) in enumerate(cases):
            y = Inches(1.75) + Inches(1.75)*j
            add_rect(slide, Inches(0.3), y, Inches(12.7), Inches(1.6), fill=LGREY,
                     line=MGREY)
            add_text(slide, title, Inches(0.5), y+Inches(0.08), Inches(12.3),
                     Inches(0.38), size=15, bold=True, color=NAVY)
            add_text(slide, f"Cause:  {cause}", Inches(0.5), y+Inches(0.48),
                     Inches(12.3), Inches(0.38), size=12.5, color=DKTEXT)
            add_text(slide, f"Fix:      {fix}", Inches(0.5), y+Inches(0.88),
                     Inches(12.3), Inches(0.48), size=12.5, color=BLUE)

    content_slide(prs, "Priority #8 — Ghost Zone Edge Cases (Defer Until After Retraining)", ghost_body)

    # ── SLIDE 7: Accuracy metric + multi-camera ─────────────────────────────────
    def infra_body(slide):
        for x, color, pri, heading, items in [
            (Inches(0.3), TEAL, "#1", "Accuracy Metric First",
             ["Manually count one 60-second clip (two videos)",
              "Produce ground-truth JSON per video",
              "Write tools/evaluate.py  (~30 lines)",
              "Any count change > ±1 flags for review",
              "Without this, no improvement can be measured"]),
            (Inches(6.8), BLUE, "#4", "Multi-Camera YAML Config",
             ["Move ROI + thresholds into YAML file",
              "configs/cameras/intersection_A.yaml",
              "run_coreml.py loads via --camera-config",
              "New camera = new YAML file, no code change",
              "~40 lines; assign to student immediately"]),
        ]:
            add_rect(slide, x, Inches(1.3), Inches(6.1), Inches(5.6), fill=color)
            add_rect(slide, x, Inches(1.3), Inches(0.7), Inches(0.5), fill=NAVY)
            add_text(slide, pri, x+Inches(0.05), Inches(1.35), Inches(0.6),
                     Inches(0.38), size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            add_text(slide, heading, x+Inches(0.8), Inches(1.4), Inches(5.1),
                     Inches(0.5), size=17, bold=True, color=WHITE)
            for j, item in enumerate(items):
                add_text(slide, f"•  {item}",
                         x+Inches(0.25), Inches(2.1)+Inches(0.6)*j,
                         Inches(5.6), Inches(0.52), size=13, color=WHITE)

    content_slide(prs, "Infrastructure — Start Here (Student Tasks)", infra_body)

    prs.save(OUT / "slides_limitations.pptx")
    print(f"Saved: {OUT / 'slides_limitations.pptx'}")


# ══════════════════════════════════════════════════════════════════════════════
#  DECK 2 — ENGINEERING CHALLENGES
# ══════════════════════════════════════════════════════════════════════════════
def make_engineering_deck():
    prs = new_prs()

    # ── TITLE ──────────────────────────────────────────────────────────────────
    title_slide(prs,
                "Real-Time Pipeline: Engineering Challenges",
                "What broke, how we fixed it, and what we had to revert")

    # ── SLIDE 2: Overview ──────────────────────────────────────────────────────
    def overview_body(slide):
        add_text(slide,
                 "Adapting the PyTorch reference counter (12–18 fps) to the Apple Neural Engine (30–45 fps) "
                 "required solving a chain of interdependent problems. Each fix exposed the next issue.",
                 Inches(0.4), Inches(1.2), Inches(12.5), Inches(0.8),
                 size=14, color=DKTEXT)
        stages = [
            (GREEN, "CoreML\nExport", "NMS bug\nConf shift"),
            (BLUE,  "Frame\nSkipping", "np.empty bug\nKalman revert"),
            (TEAL,  "Async\nCapture", "None frame\nSIGABRT crash"),
            (AMBER, "ByteTrack\nCalibration", "track_thresh\ntrack_buffer"),
            (RED,   "Ghost Zone\nSystem", "6+ iterations\nPer-class tuning"),
            (NAVY,  "Arm-Opening\nFix", "Live-track\nproximity check"),
        ]
        n  = len(stages)
        bw = Inches(1.9)
        bh = Inches(2.8)
        gap = Inches(0.18)
        x0 = Inches(0.3)
        y0 = Inches(2.15)
        for i, (color, heading, detail) in enumerate(stages):
            x = x0 + (bw + gap) * i
            add_rect(slide, x, y0, bw, bh, fill=color)
            add_text(slide, heading, x+Inches(0.1), y0+Inches(0.2),
                     bw-Inches(0.2), Inches(1.0), size=14, bold=True, color=WHITE,
                     align=PP_ALIGN.CENTER)
            add_text(slide, detail, x+Inches(0.1), y0+Inches(1.2),
                     bw-Inches(0.2), Inches(1.4), size=11.5, color=WHITE,
                     align=PP_ALIGN.CENTER)
            if i < n - 1:
                add_text(slide, "→", x+bw+Inches(0.02), y0+Inches(1.1),
                         gap+Inches(0.1), Inches(0.5), size=18, bold=True,
                         color=MGREY, align=PP_ALIGN.CENTER)

    content_slide(prs, "Overview — Six Problem Chains", overview_body)

    # ── SLIDE 3: CoreML export ─────────────────────────────────────────────────
    def coreml_body(slide):
        for x, color, heading, rows in [
            (Inches(0.3), RED, "Problem 1 — NMS Baked Into Model",
             [("Symptom", "Every inference call returned empty boxes"),
              ("Cause",   "nms=True bakes NMS into the CoreML graph; Ultralytics 8.x\ncannot parse the resulting tensor shape"),
              ("Fix",     "Export without nms=True — NMS stays in Python, negligible overhead"),
              ("Outcome", "✓ Fixed completely")]),
            (Inches(6.8), AMBER, "Problem 2 — Confidence Distribution Shift",
             [("Symptom", "No car tracks created; cars filtered before reaching tracker"),
              ("Cause",   "CoreML shifts car confidence from PyTorch's 0.45+ range\ndown to 0.15–0.37 — three layers all filtered cars out"),
              ("Fix",     "COREML_CONF_FLOOR=0.10  |  conf_car=0.15\nByteTrack track_thresh=0.15"),
              ("Outcome", "✓ Fixed — but requires careful threshold management")]),
        ]:
            add_rect(slide, x, Inches(1.25), Inches(6.1), Inches(5.8), fill=color)
            add_text(slide, heading, x+Inches(0.2), Inches(1.35), Inches(5.7),
                     Inches(0.55), size=15, bold=True, color=WHITE)
            for j, (label, text) in enumerate(rows):
                y = Inches(2.1) + Inches(1.12)*j
                add_rect(slide, x+Inches(0.15), y, Inches(5.8), Inches(0.95),
                         fill=RGBColor(0, 0, 0))
                add_text(slide, label, x+Inches(0.25), y+Inches(0.05), Inches(1.2),
                         Inches(0.35), size=11, bold=True, color=MGREY)
                add_text(slide, text, x+Inches(0.25), y+Inches(0.38), Inches(5.5),
                         Inches(0.5), size=12, color=WHITE)

    content_slide(prs, "CoreML Export Bugs", coreml_body)

    # ── SLIDE 4: Frame skipping ────────────────────────────────────────────────
    def skip_body(slide):
        # Three columns: wrong approach / why it failed / correct approach
        cols = [
            (RED,   "Attempt 1  — np.empty\non skip frames",
             ["Passed np.empty((0,6)) to ByteTrack\non every skipped frame",
              "Reasoning: signal 'nothing seen'",
              "Result: ALL tracks entered 'lost'\nstate every other frame",
              "track_age stalled → MIN_TRACK_AGE\nnever reached → nothing counted",
              "✗  Broke counting entirely"]),
            (AMBER, "Attempt 2  — Kalman\nvelocity prediction",
             ["Tried np.empty so ByteTrack\nwould predict via Kalman",
              "Theory: Kalman estimates next\nposition from velocity",
              "Reality: diverged after 2–3 frames\n→ IoU mismatch on next detection",
              "Track fragmentation got WORSE\nthan the original bug",
              "✗  Reverted after testing"]),
            (GREEN, "Final Fix  — pass\nlast_dets every frame",
             ["Pass the most recent real\ndetection on ALL skip frames",
              "ByteTrack stays confirmed,\ntracks never enter 'lost'",
              "Kalman updates correctly;\nno position divergence",
              "track_age increments every\nframe → counting works",
              "✓  Stable; in production"]),
        ]
        bw = Inches(4.1)
        for i, (color, heading, items) in enumerate(cols):
            x = Inches(0.3) + (bw + Inches(0.25))*i
            add_rect(slide, x, Inches(1.25), bw, Inches(5.9), fill=color)
            add_text(slide, heading, x+Inches(0.2), Inches(1.35), bw-Inches(0.3),
                     Inches(0.75), size=15, bold=True, color=WHITE)
            for j, item in enumerate(items):
                add_text(slide, item, x+Inches(0.2), Inches(2.25)+Inches(0.92)*j,
                         bw-Inches(0.3), Inches(0.82), size=12.5, color=WHITE)

    content_slide(prs, "Frame Skipping — Three Attempts to Get It Right", skip_body)

    # ── SLIDE 5: Async capture issues ──────────────────────────────────────────
    def async_body(slide):
        issues = [
            (AMBER, "None-Frame Bug",
             "AsyncCapture returns (True, None) when buffer is momentarily empty — "
             "video is alive but no frame produced yet.",
             "Original guard 'if not ret' missed it. None passed to CoreML, "
             "which silently returned empty predictions → near-zero detections + "
             "artificially high throughput.",
             "Add explicit 'if frame is None: sleep; continue' guard.",
             "✓  Fixed — one line"),
            (RED,   "Long-Run SIGABRT Crash",
             "Process aborted (OS-level, not Python-catchable) after ~1 800 frames "
             "of inference on the 1920×1080 @ 60 fps custom video. Short clips never triggered it.",
             "Background reader thread churns frames into a ring buffer at full decode "
             "speed while the ANE processes at ~40 fps. Over hundreds of calls, the thread–ANE "
             "interaction accumulated until the ANE runtime aborted.",
             "Replace AsyncCapture with SyncCapture for file sources. "
             "Async kept only for live/RTSP where dropping frames is correct.",
             "✓  Fixed — no more crash on long videos"),
        ]
        y0 = Inches(1.25)
        for i, (color, title, symptom, cause, fix, outcome) in enumerate(issues):
            y = y0 + Inches(2.9)*i
            add_rect(slide, Inches(0.3), y, Inches(12.7), Inches(2.7), fill=LGREY,
                     line=MGREY)
            add_rect(slide, Inches(0.3), y, Inches(2.2), Inches(2.7), fill=color)
            add_text(slide, title, Inches(0.4), y+Inches(0.1), Inches(2.0),
                     Inches(0.9), size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            for label, text, ty in [
                ("Symptom:", symptom, y+Inches(0.1)),
                ("Cause:",   cause,   y+Inches(0.85)),
                ("Fix:",     fix,     y+Inches(1.65)),
            ]:
                add_text(slide, label, Inches(2.65), ty, Inches(1.1), Inches(0.35),
                         size=11, bold=True, color=BLUE)
                add_text(slide, text, Inches(3.6), ty, Inches(9.2), Inches(0.65),
                         size=11.5, color=DKTEXT)
            add_text(slide, outcome, Inches(2.65), y+Inches(2.28), Inches(9.8),
                     Inches(0.3), size=12, bold=True, color=GREEN)

    content_slide(prs, "Async Capture — Two Hidden Bugs", async_body)

    # ── SLIDE 6: Ghost zone evolution ──────────────────────────────────────────
    def ghost_body(slide):
        add_text(slide,
                 "CoreML confidence jitter fragments tracks, causing double-counts. "
                 "Ghost zones suppress re-detection of the same object. "
                 "Getting the parameters right took six major iterations.",
                 Inches(0.35), Inches(1.2), Inches(12.6), Inches(0.65),
                 size=13, color=DKTEXT)
        iterations = [
            (GREEN, "Flat 80px / 90f",    "Worked for basic dedup"),
            (RED,   "Tighten to 40px/30f","✗ Blocked real pedestrians walking same path"),
            (AMBER, "Per-class params",   "Cars: 80px/90f  Persons: 80px/120f — resolved conflict"),
            (RED,   "Size-adaptive 0.8×", "✗ Adjacent-lane car suppressed — reverted"),
            (GREEN, "Size-adaptive 0.6×", "180px for close car, 30f timeout — current balance"),
            (RED,   "Stop-car extension", "✗ Blocked new cars in same lane — reverted"),
        ]
        bw  = Inches(2.05)
        gap = Inches(0.08)
        y0  = Inches(2.0)
        for i, (color, label, outcome) in enumerate(iterations):
            x = Inches(0.25) + (bw+gap)*i
            add_rect(slide, x, y0, bw, Inches(0.45), fill=color)
            add_text(slide, f"v{i+1}", x+Inches(0.05), y0+Inches(0.05),
                     Inches(0.3), Inches(0.35), size=12, bold=True, color=WHITE)
            add_text(slide, label, x+Inches(0.38), y0+Inches(0.05),
                     bw-Inches(0.45), Inches(0.35), size=11, bold=True, color=WHITE)
            if i < len(iterations)-1:
                add_text(slide, "→", x+bw+Inches(0.01), y0+Inches(0.05),
                         gap+Inches(0.05), Inches(0.35), size=14, bold=True,
                         color=MGREY, align=PP_ALIGN.CENTER)
            add_rect(slide, x, y0+Inches(0.55), bw, Inches(0.55),
                     fill=LGREY, line=MGREY)
            add_text(slide, outcome, x+Inches(0.05), y0+Inches(0.6),
                     bw-Inches(0.1), Inches(0.45), size=10.5, color=DKTEXT)

        # Key insights
        insights = [
            ("Per-class parameters",
             "Cars need short MIN_TRACK_AGE (fragments quickly); persons need long "
             "(filters arm-opening). One flat value cannot satisfy both."),
            ("First centroid for persons",
             "With MIN_TRACK_AGE=20, a person walks ~80px before the ghost check fires. "
             "Current centroid drifts to the radius edge and escapes. "
             "First centroid (where the track appeared) is the correct comparison point."),
            ("Died-young asymmetry",
             "Car died-young → ghost MUST be created (prevents same physical car being "
             "recounted after a confidence dip). Person died-young → NO ghost "
             "(short person fragments at crosswalk entries block the next legitimate pedestrian)."),
        ]
        y0b = Inches(3.25)
        for i, (title, text) in enumerate(insights):
            y = y0b + Inches(1.35)*i
            add_rect(slide, Inches(0.25), y, Inches(12.8), Inches(1.2),
                     fill=NAVY)
            add_text(slide, title, Inches(0.45), y+Inches(0.1), Inches(4.0),
                     Inches(0.42), size=13, bold=True, color=TEAL)
            add_text(slide, text, Inches(0.45), y+Inches(0.55), Inches(12.3),
                     Inches(0.55), size=12, color=WHITE)

    content_slide(prs, "Ghost Zone System — Six Iterations", ghost_body)

    # ── SLIDE 7: Arm-opening + in-car ──────────────────────────────────────────
    def arm_body(slide):
        for x, color, heading, paras in [
            (Inches(0.3), BLUE, "Arm-Opening Double-Count",
             [("Scenario",
               "Person opens arms → YOLO loses original box, creates new one.\n"
               "ByteTrack assigns new track ID (T2) while T1 still alive.\n"
               "No ghost exists yet (T1 hasn't died) → ghost-zone check cannot help."),
              ("Why standard fix failed",
               "Cannot use ghost zones for a living track.\n"
               "Lowering MIN_TRACK_AGE to catch it faster also counted real new people."),
              ("Solution",
               "Live-track proximity check at counting time:\n"
               "Look up T1's centroid from 'age' frames ago — its position when T2 first appeared.\n"
               "If distance < GHOST_RADIUS, suppress T2 as a duplicate of a living T1."),
              ("Outcome", "✓  Arm-opening no longer double-counted")]),
            (Inches(6.8), TEAL, "In-Car Person Suppression",
             [("Scenario",
               "Overhead camera sees through windshield.\n"
               "Driver / passengers detected as persons inside a car bounding box.\n"
               "These are real people but must not be counted as pedestrians."),
              ("Why it matters",
               "Without suppression, every occupied vehicle adds\n"
               "phantom person counts to the total."),
              ("Solution",
               "After each inference frame: remove any person detection\n"
               "whose centroid falls inside a car bounding box,\n"
               "before detections reach ByteTrack.\n"
               "No in-car person track is ever created."),
              ("Outcome", "✓  In-car persons suppressed cleanly")]),
        ]:
            add_rect(slide, x, Inches(1.25), Inches(6.1), Inches(6.1), fill=color)
            add_text(slide, heading, x+Inches(0.2), Inches(1.35), Inches(5.7),
                     Inches(0.5), size=16, bold=True, color=WHITE)
            y = Inches(2.0)
            for label, text in paras:
                add_text(slide, label, x+Inches(0.2), y, Inches(5.7),
                         Inches(0.32), size=11.5, bold=True, color=MGREY)
                add_text(slide, text, x+Inches(0.2), y+Inches(0.32), Inches(5.7),
                         Inches(0.72), size=12, color=WHITE)
                y += Inches(1.1)

    content_slide(prs, "Arm-Opening Double-Count & In-Car Person Suppression", arm_body)

    # ── SLIDE 8: What we couldn't fix ──────────────────────────────────────────
    def unfixed_body(slide):
        add_text(slide,
                 "Three attempts were developed, tested, and reverted because every fix overcorrected.",
                 Inches(0.35), Inches(1.2), Inches(12.6), Inches(0.45),
                 size=14, color=DKTEXT)
        cases = [
            ("Ghost radius 0.8×",
             "Close-car fragment at >180px from ghost zone centre → double-count.",
             "Widened ghost radius to 0.8× box width (e.g. 240px for 300px car).",
             "Ghost now spanned the adjacent incoming lane → suppressed legitimate new cars."),
            ("Stopped-car ghost extension",
             "Car stopped at red light; track dies during wait; ghost expires; restart recounted.",
             "Track consecutive stationary frames; add that count to ghost timeout on death.",
             "Extension was too long in stop-and-go traffic → blocked the next real car in same lane."),
            ("np.empty on skip frames",
             "Stale last_dets re-fed to ByteTrack each skip → Kalman velocity zeroed.",
             "Pass np.empty so ByteTrack uses its own Kalman prediction forward.",
             "Kalman diverged after 2–3 skip frames → worse fragmentation than the original bug."),
        ]
        for i, (title, problem, attempt, why_failed) in enumerate(cases):
            y = Inches(1.8) + Inches(1.82)*i
            add_rect(slide, Inches(0.3), y, Inches(12.7), Inches(1.7),
                     fill=LGREY, line=MGREY)
            add_rect(slide, Inches(0.3), y, Inches(0.18), Inches(1.7), fill=RED)
            add_text(slide, title, Inches(0.6), y+Inches(0.08), Inches(3.8),
                     Inches(0.42), size=14, bold=True, color=NAVY)
            for label, text, tx, tw in [
                ("Problem:",  problem,     Inches(0.6),  Inches(3.6)),
                ("Attempted:", attempt,    Inches(4.6),  Inches(3.9)),
                ("Why reverted:", why_failed, Inches(8.7), Inches(4.1)),
            ]:
                add_text(slide, label, tx, y+Inches(0.55), tw, Inches(0.3),
                         size=10.5, bold=True, color=BLUE)
                add_text(slide, text, tx, y+Inches(0.88), tw, Inches(0.65),
                         size=11.5, color=DKTEXT)

    content_slide(prs, "What We Could Not Fix — Three Reverted Attempts", unfixed_body)

    prs.save(OUT / "slides_engineering.pptx")
    print(f"Saved: {OUT / 'slides_engineering.pptx'}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_limitations_deck()
    make_engineering_deck()
    print("Done.")
