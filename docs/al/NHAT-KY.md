# Nhật ký thực hiện (active learning + học tăng dần)

## 02/11/2026 — các bước chạy được trên CPU (không GPU, không push)

**Khác với plan:** nhánh thật là `relearn-task6` (đã có trên origin), không phải `feat/update-model`. Code mới để chưa
commit (git trong máy ảo không xoá được file lock khi chưa cho quyền xoá); danh sách file ở cuối.

### Bước 0 — khám phá
- `pred_val.npz` (r1 task 5): top-100 cặp (query, lớp, điểm) mỗi ảnh, 5.792 ảnh val, `ann_md5` khớp `eval/val_full.json`.
  Có lớp thứ hai của cùng query → **tính được margin, không cần dự đoán lại (bỏ A2b)**. Điểm là sigmoid, không có đủ
  200 lớp → entropy tính trên tối đa 5 lớp đã lưu của query (ghi trong `uncertainty.py`).
- `meta/val_a.json` 2.895 ảnh, `val_b.json` 2.897 ảnh; `orig_id` ảnh thật trong gói = id ảnh val_full.
- Gói pilot: ảnh thật = ảnh Val-A **đếm sai** (chọn bằng nhãn) có SKU của 17 hàng, lặp x3 → không phải active learning;
  giữ làm tham chiếu `oracle`.
- Ma trận AP50 theo task và độ quên đã có trong `training/full-syn-r1/runs/EXP-FULL-SYN/metrics_cl_{val,test}.json`.
- Mốc dưới kiểu E1 (`--use_private 0 --pseudo none`) có từ tháng 9 nhưng khác release; chạy lại trên r1 cần thêm
  `pdp_experiment` mới trong `full_syn.py` + GPU (⛔ B2).
- `update_model._train` đặt cứng `STOP_AFTER_STEPS=0`.

### A1–A3 — chia Val-A, điểm bất định, chọn ảnh
- Pool 2.314 ảnh / calib 581 ảnh (nguyên nhóm cảnh, cân theo mức dễ/vừa/khó, seed 0).
- 23 tập chọn: {random s0–s2, margin, margin_k, entropy, margin_bal} × B {250, 500, 850} + oracle (680 ảnh trong pool;
  167 ảnh oracle cũ rơi sang calib) + all.
- **Tín hiệu sớm (chỉ phân tích, nhãn không dùng để chọn):** tỉ lệ ảnh đếm sai trong ảnh được chọn

  | B | random (3 seed) | margin | margin_k | entropy | margin_bal |
  |---|---|---|---|---|---|
  | 250 | 0,56–0,60 | 0,82 | **0,88** | 0,78 | 0,79 |
  | 500 | 0,56–0,57 | 0,80 | **0,86** | 0,79 | 0,76 |
  | 850 | 0,55–0,58 | 0,77 | **0,78** | 0,75 | 0,74 |
  (cả pool: 0,56). Ảnh chọn theo margin cũng nhiều box hơn: B=500 margin_k 7.308 box so với random ~6.200 box.

### A4 — gói dữ liệu đợt 1 (B = 500)
`variants/al_{random_B500_s0,s1,s2, margin_B500, margin_k_B500, margin_bal_B500, oracle}`: phần cố định giống hệt
pilot (`fixed_part_md5` trùng), ảnh thật = ảnh chọn x3. 6.535 ảnh / 89.7–93.2 nghìn vật mỗi gói (oracle 7.075 ảnh).
Thêm 472−342 = 130 MB ảnh vào `images/real/val2019`. Kiểm rò rỉ Val-B / calib / test: 0.

### A5 — chấm với ngưỡng chọn trên calib
Model r1 trên Val-B: **cAcc-M2 = 0,4232** (ngưỡng chọn trên Val-A-calib) so với 0,4308 khi ngưỡng chọn ngay trên Val-B.
Đây là mốc so sánh chính của mọi run AL (`active/scored/baseline_r1.json`).

### A6–A7
`update_model --stage al --variant al_<tên>` (học như pilot, cổng pilot); `--stage data --variant al_<tên>` chạy được
cho cả 7 gói (17 hàng train, khớp baseline). `scripts/run_al.sh dot1 [--dry-run]`; `tools/al_report.py` (bảng + biểu đồ).

### B1 — độ quên của r1 (ma trận AP50 có sẵn)
Val: ACC cuối 95,0%, độ quên −0,24 điểm, BWT +1,44 điểm; test2019 (đã chấm từ trước): 94,8%, −0,33, +1,54.
→ r1 **không quên** (AP nhóm cũ còn tăng nhẹ). `ket-qua/forgetting.md`, heatmap `ket-qua/forgetting_r1_*.png`.

### B4 — chi phí: `ket-qua/chi-phi.md`
Học tăng dần task 5: 19% dung lượng, 18% giờ GPU so với train lại từ đầu. Gói AL B=500: 1,39 GB.

### Kiểm thử
`tests/test_active_learning.py` (18), `tests/test_al_build_variant.py` (4), `tests/test_forgetting_report.py` (2): qua.
`test_feedback_fix.py`, `test_cl_metrics.py`: qua. `test_update_model.py`: 1 test cần torch (máy ảo không cài được) —
chạy lại trên máy GPU.

### File mới / sửa (chưa commit)
Mới: `autocheckout/active/{__init__,splits,uncertainty,select}.py`, `tools/{al_select,al_build_variant,al_score,al_report,
forgetting_report}.py`, `scripts/run_al.sh`, `tests/{test_active_learning,test_al_build_variant,test_forgetting_report}.py`,
`docs/al/*`. Sửa: `autocheckout/update_model.py` (`_kind`, `--stage al`, `--variant` nhận `al_*`).
Dữ liệu (ngoài repo): `relearn_fix_v2/active/`, `relearn_fix_v2/variants/al_*`, ảnh thêm trong `images/real/val2019`.

### ⛔ Chờ quyết định
1. Thuê GPU cho đợt 1 (7 run × ~1 giờ ước tính) + push nhánh / gửi dữ liệu mới (thư mục `relearn_fix_v2` đã thay đổi:
   cần zip lại hoặc chỉ gửi thêm `active/`, `variants/al_*`, ảnh mới).
2. B2 (fine-tune thẳng trên r1): cần sửa `full_syn.py` + GPU.
