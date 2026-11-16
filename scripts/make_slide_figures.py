"""Render slide figures from the real InspectFlow pipeline (no mock-ups).

Usage: python scripts/make_slide_figures.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import camera_noise  # noqa: E402
from inspectflow.config import load_config  # noqa: E402
from inspectflow.cv.detection import detect_parts  # noqa: E402
from inspectflow.data.synthetic import make_frame  # noqa: E402
from inspectflow.pipeline import StationPipeline  # noqa: E402

COLORS = {"accept": (60, 160, 60), "review": (30, 130, 230), "reject": (50, 50, 200)}
BANNER = {"accept": "ACCEPT", "review": "REVIEW", "reject": "REJECT"}


def annotate(image: np.ndarray, result, pipeline: StationPipeline, reason: str) -> np.ndarray:
    canvas = image[:300].copy()
    cfg = pipeline.config.detector
    scratched = {(d.bbox.x, d.bbox.y) for d in result.defects}
    for det in detect_parts(image, min_area=cfg.min_area, max_area=cfg.max_area):
        b = det.bbox
        bad = (b.x, b.y) in scratched
        color = (40, 40, 220) if bad else (70, 190, 70)
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.width, b.y + b.height), color, 2)
        tag = "scratch" if bad else "ok"
        cv2.putText(canvas, tag, (b.x, b.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    decision = result.decision.value
    cv2.rectangle(canvas, (330, 8), (632, 62), COLORS[decision], -1)
    cv2.putText(canvas, f"{BANNER[decision]}", (342, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, reason, (342, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    qty = f"count {result.counted_quantity}/{result.expected_quantity}"
    ocr = f"OCR {result.work_order_id} (expected {result.expected_work_order_id})"
    cv2.putText(canvas, f"{qty} | {ocr} | {result.latency_ms:.1f} ms", (16, 288),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return canvas


def find_noise_case(pipeline: StationPipeline) -> tuple[np.ndarray, object]:
    """First frame where camera noise breaks the work-order OCR (the weak point)."""
    for seed in range(2000, 2200):
        rng = np.random.default_rng(seed)
        frame = make_frame(work_order_id="WO-2048", ok_count=3, scratch_count=0, seed=seed)
        noisy = camera_noise(frame.image, rng)
        res = pipeline.inspect_frame(noisy, expected_quantity=3, expected_work_order_id="WO-2048")
        if res.work_order_id != "WO-2048":
            return noisy, res
    raise SystemExit("no OCR failure found under noise")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "slide_assets" / "project1_inspectflow"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "pipeline.yaml")
    cfg.experiments.runs_path = "experiments/slide-runs.jsonl"
    cfg.experiments.inspections_path = "experiments/slide-inspections.jsonl"
    pipe = StationPipeline(cfg, ROOT)

    clean = make_frame(work_order_id="WO-2048", ok_count=3, scratch_count=0, seed=1234)
    r_clean = pipe.inspect_frame(clean.image, expected_quantity=3, expected_work_order_id="WO-2048")
    a = annotate(clean.image, r_clean, pipe, "count, code match")

    noisy_img, r_noisy = find_noise_case(pipe)
    b = annotate(noisy_img, r_noisy, pipe, "OCR mismatch -> planner")

    scr = make_frame(work_order_id="WO-3370", ok_count=2, scratch_count=1, seed=1500)
    r_scr = pipe.inspect_frame(scr.image, expected_quantity=3, expected_work_order_id="WO-3370")
    c = annotate(scr.image, r_scr, pipe, "scratch found")

    raw = out / "raw_frames"
    raw.mkdir(exist_ok=True)
    cv2.imwrite(str(raw / "accept_clean.png"), clean.image)
    cv2.imwrite(str(raw / "review_camera_noise.png"), noisy_img)
    cv2.imwrite(str(raw / "reject_scratch.png"), scr.image)
    for name, img in (("01_accept_clean", a), ("02_review_camera_noise", b), ("03_reject_scratch", c)):
        cv2.imwrite(str(out / f"{name}.png"), img)
    triple = np.hstack([a, b, c])
    cv2.imwrite(str(out / "04_decision_rules_triptych.png"), triple)

    # clean vs noise, same scene
    rng = np.random.default_rng(7)
    noisy_same = camera_noise(clean.image, rng)
    r_same = pipe.inspect_frame(noisy_same, expected_quantity=3, expected_work_order_id="WO-2048")
    pair = np.hstack([a, annotate(noisy_same, r_same, pipe, "same scene + camera noise")])
    cv2.imwrite(str(out / "05_clean_vs_noise_same_scene.png"), pair)

    # benchmark chart from results/benchmark.json produced by scripts/evaluate.py
    bench = json.loads((ROOT / "results" / "benchmark.json").read_text())
    rows = [("Detection F1 @ IoU 0.5", "detection_f1_iou50"), ("Exact count", "count_exact_match"),
            ("Scratch macro-F1", "defect_f1_macro"), ("Work-order OCR", "ocr_exact_match"),
            ("Lot decision acc.", "decision_accuracy")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=200)
    ax.bar(x - 0.2, [bench["clean"][k] for _, k in rows], 0.4, label="Clean (80 frames)", color="#1f5fd1")
    ax.bar(x + 0.2, [bench["camera_noise"][k] for _, k in rows], 0.4, label="Camera noise (80 frames)", color="#e08a2c")
    for i, (_, k) in enumerate(rows):
        ax.text(i - 0.2, bench["clean"][k] + 0.01, f"{bench['clean'][k]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, bench["camera_noise"][k] + 0.01, f"{bench['camera_noise'][k]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_title("InspectFlow held-out benchmark (synthetic frames)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "06_benchmark_clean_vs_noise.png")
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
