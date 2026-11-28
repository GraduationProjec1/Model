#!/usr/bin/env bash
# Script chạy end-to-end kiến trúc Decoupled Continual Object Detection trên Vast.ai (1x RTX 4090, ~2 giờ):
#   1. Tạo dataset 1 lớp (0 = product) chống rò rỉ calib/Val-B
#   2. Train Tầng 1 Class-agnostic RT-DETR-l (hoặc YOLO11m) ở 960px + xuất proposal boxes trên Val
#   3. Train Tầng 2 Continual Crop Recognizer tuần tự 5 task (Mode A: Prototype 0% quên & Mode B: Replay+KD)
#   4. Nối Tầng 1 + Tầng 2 -> chấm Val-B chuẩn (al_score) + đo Stage-1 Recall@0.5
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA="${AC_DATA:-/workspace/autocheckout-data}"
OUT="${DECOUPLED_OUT:-$REPO/data/training/update-model/decoupled_cl}"
DET_MODEL="${DET_MODEL:-rtdetr-l.pt}"     # hoặc yolo11m.pt
DET_IMGSZ="${DET_IMGSZ:-960}"
DET_EPOCHS="${DET_EPOCHS:-25}"
ROI_ARCH="${ROI_ARCH:-convnext_tiny}"     # hoặc dinov2_vits14

cd "$REPO"
mkdir -p "$OUT"

echo "=== BƯỚC 1/4: TẠO DATASET 1 LỚP CHO TẦNG 1 (CHỐNG RÒ RỈ) ==="
python -m tools.make_agnostic_dataset --data "$DATA" --selection margin_k_B500_r3 \
    --include-synth --out "$OUT/agnostic_data"

echo "=== BƯỚC 2/4: TRAIN TẦNG 1 ($DET_MODEL @ ${DET_IMGSZ}px) & XUẤT PROPOSALS ==="
python -c "import ultralytics" 2>/dev/null || pip install -q ultralytics
python - << EOF
from pathlib import Path
import json, numpy as np
from ultralytics import RTDETR, YOLO

det_model = "${DET_MODEL}"
out_dir = Path("${OUT}")
data_yaml = str(out_dir / "agnostic_data/dataset.yaml")
cls = RTDETR if "rtdetr" in det_model.lower() else YOLO
model = cls(det_model)
model.train(data=data_yaml, imgsz=${DET_IMGSZ}, epochs=${DET_EPOCHS}, batch=8, project=str(out_dir), name="stage1_det", exist_ok=True)
best = cls(str(out_dir / "stage1_det/weights/best.pt"))

data_root = Path("${DATA}")
img_root = data_root / "rpc"
if not (img_root / "val2019").is_dir():
    img_root = data_root / "archive"
val_full = json.load(open(data_root / "eval/val_full.json"))
calib_ids = set(json.load(open(data_root / "relearn/active/splits/val_a_split.json"))["calib"])
valb_ids = {im["id"] for im in json.load(open(data_root / "relearn/meta/val_b.json"))["images"]}
eval_ids = calib_ids | valb_ids

img_ids, boxes, scores = [], [], []
for im in val_full["images"]:
    if im["id"] not in eval_ids:
        continue
    p = img_root / im["file_name"]
    if not p.is_file():
        p = img_root / "val2019" / im["file_name"]
    res = best.predict(str(p), imgsz=${DET_IMGSZ}, conf=0.10, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        for b, s in zip(xyxy, conf):
            img_ids.append(int(im["id"]))
            boxes.append(b.astype(np.float32))
            scores.append(float(s))

np.savez_compressed(
    out_dir / "stage1_proposals_val.npz",
    image_id=np.asarray(img_ids, dtype=np.int64),
    boxes=np.stack(boxes, axis=0) if boxes else np.zeros((0, 4), dtype=np.float32),
    score=np.asarray(scores, dtype=np.float32),
)
print(f"[Stage1] Saved {len(img_ids)} proposals to {out_dir / 'stage1_proposals_val.npz'}")
EOF

echo "=== BƯỚC 3/4: TRAIN TẦNG 2 CONTINUAL RECOGNIZER (5 TASK: MODE A & MODE B) ==="
python -m tools.continual_roi_train --data "$DATA" --mode prototype --arch "$ROI_ARCH" \
    --out "$OUT/stage2_prototype"
python -m tools.continual_roi_train --data "$DATA" --mode replay_kd --arch "$ROI_ARCH" \
    --exemplars-per-class 20 --out "$OUT/stage2_replay_kd"

echo "=== BƯỚC 4/4: ĐÁNH GIÁ END-TO-END TRÊN VAL-B ==="
python -m tools.eval_decoupled --data "$DATA" --proposals "$OUT/stage1_proposals_val.npz" \
    --checkpoint "$OUT/stage2_prototype/continual_roi.pth" --name decoupled_prototype --out "$OUT/eval"
python -m tools.eval_decoupled --data "$DATA" --proposals "$OUT/stage1_proposals_val.npz" \
    --checkpoint "$OUT/stage2_replay_kd/continual_roi.pth" --name decoupled_replay_kd --out "$OUT/eval"
echo "=== HOÀN TẤT DECOUPLED CL ==="
