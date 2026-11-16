"""Post raw station frames to a running API and save the ERP inbox + a result card.

Start the server first:  uvicorn inspectflow.api.main:app --app-dir src --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slide_assets" / "project1_inspectflow"
BASE = "http://127.0.0.1:8000"
CASES = [
    ("accept_clean.png", 3, "WO-2048"),
    ("review_camera_noise.png", 3, "WO-2048"),
    ("reject_scratch.png", 3, "WO-3370"),
]
COLORS = {"accept": (60, 160, 60), "review": (30, 130, 230), "reject": (50, 50, 200)}


def main() -> None:
    rows = []
    with httpx.Client(base_url=BASE, timeout=30) as client:
        print(client.get("/health").json())
        for name, qty, wo in CASES:
            data = (OUT / "raw_frames" / name).read_bytes()
            r = client.post(
                "/api/v1/inspections/frame",
                files={"image": (name, data, "image/png")},
                data={"expected_quantity": qty, "expected_work_order_id": wo},
            )
            r.raise_for_status()
            body = r.json()
            rows.append((name, body))
            print(name, "->", body["erp_document"]["decision"], f"{body['inspection']['latency_ms']:.1f} ms")
        inbox = client.get("/api/v1/erp/inspections").json()
    (OUT / "api_erp_inbox.json").write_text(json.dumps(inbox, indent=2))

    card = np.full((60 + 150 * len(rows), 900, 3), 250, np.uint8)
    cv2.putText(card, "POST /api/v1/inspections/frame  ->  ERP quality-inspection document", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
    for i, (name, body) in enumerate(rows):
        y = 60 + 150 * i
        doc, ins = body["erp_document"], body["inspection"]
        thumb = cv2.imread(str(OUT / "raw_frames" / name))[:300, :640]
        thumb = cv2.resize(thumb, (300, 140))
        card[y : y + 140, 16:316] = thumb
        cv2.rectangle(card, (330, y), (510, y + 36), COLORS[doc["decision"]], -1)
        cv2.putText(card, doc["decision"].upper(), (344, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        lines = [
            f"station {doc['station_id']}   work order {doc['work_order_id']}",
            f"counted {doc['counted_quantity']} / expected {doc['expected_quantity']}   defects {len(doc['defects'])}",
            f"latency {ins['latency_ms']:.1f} ms   {ins['model_versions']['detector']}",
        ]
        for j, text in enumerate(lines):
            cv2.putText(card, text, (330, y + 62 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.imwrite(str(OUT / "07_api_erp_documents.png"), card)
    print("saved", OUT / "07_api_erp_documents.png")


if __name__ == "__main__":
    sys.exit(main())
