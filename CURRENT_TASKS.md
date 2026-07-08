# Current Tasks

Shared task board between **Hermes** (master planning agent) and **Claude Code** (local
execution agent). Both agents read and write this file, committing and pushing after every
meaningful update so the other stays in sync.

---

## Conventions

### Status markers
| Marker | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[→]` | In progress — who is working on it and since when |
| `[x]` | Done — result summarized inline |
| `[!]` | Blocked — reason stated, waiting on human or external resource |

### Who writes what
- **Hermes** writes new tasks, sets priorities, clarifies scope, marks `[!]` when blocked.
- **Claude Code** updates status to `[→]` when it starts, fills in results and marks `[x]` when done.
- Either agent may add notes under a task.
- After every update: `git add CURRENT_TASKS.md && git commit -m "..." && git push`.

### Task format
```
## T<id> — <title>
**Status**: [ ] / [→] / [x] / [!]
**Priority**: high / medium / low
**Assigned to**: Hermes / Claude Code / Human

<description — enough context for Claude Code to execute without asking follow-up questions>

**Result**: (filled in by Claude Code when done)
**Notes**: (optional — either agent)
```

---

## Active tasks

## T001 — Tracking inference batch (no ReID)
**Status**: [x] Done — 2026-07-07
**Priority**: high
**Assigned to**: Claude Code

Run all 8 models (4 pretrained COCO + 4 fine-tuned) × all 13 Demo/ videos through
`tools/run_tracking_inference.py --batch --device mps --resize 1280`. Output goes to
`experiments/yolo11_experiments/tracking_inference/<model>/`. Skip-existing logic is in
place — safe to interrupt and resume with the same command.

**Result**: 104 videos written (8 models × 13 videos). All at ~480MB each for ANMR0006,
smaller for the short benchmark clips. Location:
`experiments/yolo11_experiments/tracking_inference/<pretrained|finetuned>_yolo11<s|m|l|x>/`

---

## T002 — Tracking inference batch (with ReID, ANMR0006 only)
**Status**: [x] Done — 2026-07-07
**Priority**: medium
**Assigned to**: Claude Code

Run all 8 models on `Demo/ANMR0006.mp4` with OSNet x0.25 appearance re-ID enabled.
Output goes to `experiments/yolo11_experiments/tracking_inference/reid/<model>/ANMR0006.mp4`.
Command:
```bash
python tools/run_tracking_inference.py --batch --reid --device mps --resize 1280 --reid-device cpu
```

**Result**: 8 videos written, all ~480–494 MB. Location:
`experiments/yolo11_experiments/tracking_inference/reid/<model>/ANMR0006.mp4`

---

## T003 — Review tracking videos and pick best model
**Status**: [ ]
**Priority**: high
**Assigned to**: Human (with Hermes support)

Once T001 and T002 are done, review the output videos in
`experiments/yolo11_experiments/tracking_inference/` to decide which model + ReID
configuration produces the most stable tracks on `ANMR0006.mp4`. Key questions:
- Which model size gives the best car/person track continuity?
- Does ReID meaningfully reduce ID switches vs no-ReID?
- Is the speed difference (s vs x) worth the quality gain?

**Notes**: Hermes should ask the user for their observations after watching the videos,
then update T004 with the chosen model.

---

## T004 — Tune counting parameters for YOLO11 (PyTorch confidence range)
**Status**: [ ]
**Priority**: high
**Assigned to**: Claude Code (after T003 decides the model)

The ghost-zone constants in `tools/run_yolo11_experiment.py` are copied from
`run_coreml.py` and tuned for CoreML's compressed confidence range (cars: 0.15–0.37).
PyTorch produces stable confidence (cars: 0.45+), so the same constants produce wrong
counts — ANMR0006 shows 0 cars / 1 person across all 4 models, which is clearly wrong.

Steps:
1. Run `--verbose` on ANMR0006.mp4 with the chosen model, pipe through grep to see
   COUNTED / GHOST-SUPP / DIED-YOUNG tags.
2. Compare against `counting_experiment/count_objects.py` (ground truth) on the same video.
3. Adjust `MIN_TRACK_AGE`, `GHOST_RADIUS`, `GHOST_TIMEOUT` in `run_yolo11_experiment.py`
   until counts match the reference baseline within ~5%.

**Notes**: The diagnostic workflow for run_coreml.py (documented in CLAUDE.md) applies
directly here. Start by checking whether ghost zones are over-suppressing or under-suppressing.

---

## T005 — Web app integration
**Status**: [ ]
**Priority**: low
**Assigned to**: Hermes (planning) → Claude Code (execution)

The branch is named `dev-webapp-integration` but no web front-end exists yet.
Hermes should clarify scope with the user before Claude Code starts any implementation:
- What framework? (FastAPI + React? Streamlit? Flask?)
- What does the UI show? (live feed, count overlays, historical charts?)
- Does it connect to the CoreML pipeline or the YOLO11 pipeline?
- Is this for local demo or deployed somewhere?

**Notes**: No code written yet. Hermes to gather requirements first.

---

## Completed tasks

- **T001** — Tracking inference batch (no ReID): 104 videos, all 8 models × 13 videos ✓
- **T002** — Tracking inference batch (ReID): 8 videos, all 8 models on ANMR0006 ✓
