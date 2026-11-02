# Chi phí: học tăng dần so với train lại từ đầu (B4)

Số đo trên dữ liệu và log có sẵn, không ước đoán (02/11/2026).

## Thêm 25 SKU mới (task 5) vào model 175 SKU

| | Train lại từ đầu (FSA + task 1→5) | Học tăng dần (chỉ task 5) |
|---|---|---|
| Ảnh phải giữ để train | ảnh ghép task 1–5: **71.642 ảnh, 16,3 GB** | ảnh ghép task 5: **13.377 ảnh, 3,1 GB** (19%) |
| Giờ GPU (4090) | 5,04 + 5,96 + 5,36 + 5,23 + 5,18 + 5,90 = **32,7 giờ** | **5,9 giờ** (18%) |

Nguồn: `data/processed/rpc_synth_v2_v1/images/task_{1..5}` (du); `relearn_fix_v2/seed/full-syn-r1/stages/{fsa,task1,tasks}.json` (`train_hours`).
Học tăng dần của r1 (PDP) không replay ảnh cũ: kiến thức cũ giữ bằng prompt riêng đã đóng băng + nhãn giả do model cũ gán.

## Học lại 17 SKU yếu (mode improve, một gói active learning B = 500)

| Phần dữ liệu | Ảnh | Dung lượng |
|---|---|---|
| Ảnh quầy thật được gán nhãn (B) | 500 | 0,12 GB |
| Ảnh ghép đủ mặt của SKU yếu | 1.767 | 0,46 GB |
| Ảnh ghép v2 (gồm replay 30%: 2.273 ảnh của task cũ) | 3.268 | 0,81 GB |
| **Tổng** | **5.535** | **1,39 GB** (≈ 8,5% của 16,3 GB) |

Nguồn: `relearn_fix_v2/variants/al_margin_B500/train_task_5_fix.json`, kích thước file thật.
Giờ GPU của một gói: chưa đo (runbook ước ~1 giờ cho pilot 4 epoch; sẽ ghi sau đợt GPU 1).

## Số tham số được train
Chưa đếm (máy này không có torch). Đếm trên máy GPU: tổng tham số của checkpoint so với `prompts.private_*` + các hàng
`class_embed` của `train_rows` (mode improve) / prompt riêng task mới + `class_embed` (task mới).

## Ghi chú trung thực
Gói học lại có replay 30% ảnh ghép của task cũ → đây là học tăng dần **có replay nhỏ** (2.273 ảnh, ~3% số ảnh của train
lại từ đầu), không phải hoàn toàn không dùng dữ liệu cũ.
