# Runbook — học lại (mode improve) với thư mục dữ liệu `relearn_fix_v2` (02/11/2026)

> **Cập nhật 02/11 (đêm):** dữ liệu giờ nằm trong một thư mục chung `../autocheckout-data` (xem
> `autocheckout-data/README.md`, `autocheckout/datapaths.py`); `relearn_fix_v2` cũ = `autocheckout-data/relearn` + các
> phần dùng chung. Lệnh chạy giữ nguyên (`bash scripts/run_relearn_fix.sh ...`); runbook GPU mới: `docs/al/RUNBOOK.md`.

**Bản dùng chính: `relearn_fix_v2`** (13 GB) = ảnh ghép đủ mặt của task 6–7 (`rpc_synth_relearn_v1`, 600 vật/SKU)
+ ảnh ghép v2 (300 vật/SKU, 600 nếu SKU chưa có ảnh đủ mặt) + ảnh quầy thật Val-A x3 + replay 30%, nhãn thật 100%.

| Biến thể | SKU | Dòng ảnh (thật / đủ mặt / v2) | Vật | Vật/SKU yếu | Lớp ít nhất |
|---|---|---|---|---|---|
| pilot | 17 | 7.576 (2.541 / 1.767 / 3.268) | 107.676 | 1.279–1.796 | 263 |
| e1 | 63 (13 chưa có ảnh đủ mặt) | 15.377 (4.431 / 2.879 / 8.067) | 222.752 | 1.159–2.022 | 636 |

`relearn_fix_v1` (12 GB, không ảnh đủ mặt) giữ để so sánh tác dụng của ảnh đủ mặt:
`RELEARN_DATA_ROOT=../relearn_fix_v1 SETTINGS=configs/training/update_model_relearn_fix_v1.json bash scripts/run_relearn_fix.sh`

Chỉ cần hai thư mục, đặt **cạnh nhau**:
```
~/AutoCheckout-CL/      code (nhánh feat/update-model + các sửa ngày 02/11)
~/relearn_fix_v2/       dữ liệu tự đủ (13 GB), tạo bằng build_relearn_data.py --self-contained
```
Đặt nơi khác thì `export RELEARN_DATA_ROOT=/duong/dan/relearn_fix_v2`.

## Trong `relearn_fix_v2/` (v1 giống, trừ `images/relearn`)
| Thư mục | Nội dung |
|---|---|
| `variants/pilot`, `variants/e1` | `train_task_5_fix.json` (ảnh phản hồi Val-A x3 + ảnh ghép v2 theo hạn mức + replay), `train_rows.json`, `manifest.json` |
| `images/real`, `images/relearn`, `images/v2` | ảnh train đã chép sẵn |
| `eval/` | `rpc_eval_official_v1`: `val_full.json`, `test_full.json`, `task_config.json`, ... |
| `archive/val2019`, `archive/test2019` | ảnh val (5.792) và test (24.000) để chấm |
| `pretrained/hf-cache` | Deformable DETR (SenseTime) + ConvNeXt-V2 base, đúng revision code ghim |
| `seed/full-syn-r1` | model round 1: `task_5/task_final.pth` (md5 fe510a4c…), `pred_val.npz`, FSA `hf_model`, `plan.json` |
| `meta/` | Val-A / Val-B, cặp nhầm, SKU yếu |
| `SHA256SUMS` | kiểm toàn vẹn |

## Chạy (máy GPU, env `pdp` như handoff M5 mục 3.3)
```bash
tmux new -s relearn
conda activate pdp && cd ~/AutoCheckout-CL
bash scripts/run_relearn_fix.sh data        # kiểm dữ liệu + split + baseline + in số liệu pilot (CPU, vài phút)
bash scripts/run_relearn_fix.sh             # pilot: 4 epoch (~1 giờ trên 4090, ước tính)
```
Lần đầu script kiểm `SHA256SUMS` (bỏ qua bằng `CHECK=0`). Bị ngắt thì chạy lại đúng lệnh.

Kết quả: `data/training/update-model/relearn-fix-v2/`
- `stages/baseline.json`: cAcc-M2 Val-B của round 1 (đo trên Mac: 0,4308), 1.633 ảnh Val-A đếm sai.
- `stages/pilot.json`: cổng go/stop, `data.rows_differ_from_baseline` (SKU khác giữa gói và code; pilot: không có).
- `scored/pilot/sku_regression.json`: từng SKU, cặp nhầm trước/sau.

Pilot `go` → `bash scripts/run_relearn_fix.sh e1` (~2,5 giờ). Chỉ khi nhóm chốt cấu hình cuối:
`bash scripts/run_relearn_fix.sh final_test --confirm-test` (chấm test2019 một lần).

## Giới hạn
- 13 SKU của E1 không thuộc 50 SKU đã làm ảnh đủ mặt (15_dried_fruit, 67_dessert, 122/131/133_chocolate,
  46/47_instant_noodles, 115_canned_food, 4/7/10_puffed_food, 159_seasoner, 41_instant_drink): chỉ có ảnh v2 + ảnh thật.
- `round2` dựng dữ liệu từ release v2 gốc (không có trong thư mục này): cần giải nén `full_syn_rpc_synth_v2_v1_*.tar`
  và đặt `synth_v2_root`, nếu không code báo lỗi rõ.
- E1 của gói có 63 SKU; code tính lại trên Val-B ra 64 (thêm label 168). Gói train đúng 63 SKU của nó, chấm theo đó.

## Sửa trong code (chưa commit)
- `autocheckout/feedback_fix.py`: `select_by_quota()` (+ test trong `tests/test_feedback_fix.py`).
- `autocheckout/update_model.py`: `data_root` / `RELEARN_DATA_ROOT`; `prebuilt_release`; hạn mức `quota_*`;
  `--stage data`; chấm theo `train_rows.json` thật sự train; round2 báo lỗi rõ khi thiếu v2.
- `configs/training/update_model_relearn_fix_v{1,2}.json`, `scripts/run_relearn_fix.sh` (mặc định v2), file này.
