# Nhật ký thực hiện (active learning + học tăng dần)

## 02/11/2026 — các bước chạy được trên CPU (không GPU, không push)

**Khác với plan:** nhánh thật là `relearn-task6` (đã có trên origin), không phải `feat/update-model`. Code ở nhánh
`feat/active-learning` (commit 9bb0100 và commit sau).

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

## 02/11/2026 (tối) — hoàn tất code còn lại để máy GPU chỉ việc chạy

- **B2 fine-tune thẳng:** `configs/exp/native/EXP-FULL-SYN-NAIVE.sh` (= EXP-FULL-SYN + `--use_private 0 --pseudo none`,
  như E1 tháng 9), `full_syn.NAIVE` (seed FSA + task 1 của r1 như M5, bỏ audit nhãn giả, không chấm test),
  `configs/training/full_syn_naive.json` (cùng 640 px, 16 epoch, batch 8 như r1), `tools/run_naive.py` (train task
  2..last rồi in bảng quên r1 cạnh NAIVE). Test cấu hình + test dry run (slow, cần torch).
- **A8 đợt 2:** dựng sẵn 15 gói (B=250, 850 × random×3, margin, margin_k, margin_bal, entropy + entropy_B500).
- **A9 vòng 2:** `tools/al_round2.py` (model vòng 1 chấm phần pool chưa gán nhãn → chọn thêm B → gói `al_<tên>_r2`, tập
  gán nhãn cộng dồn); `update_model`: `al_*_r2` học theo công thức round2 từ model vòng 1, so với run vòng 1.
- **Test2019 một lần cho run AL:** `update_model --stage final_test --variant al_<tên> --confirm-test` (stage riêng
  `final_test_<tên>`), `tools/al_score.py --test-pred` (ngưỡng calib áp lên test, không ghi đè lần 2).
- **B4 tham số:** `tools/count_params.py` đọc checkpoint không cần torch: 131,5 triệu tham số; đóng băng 77,4%; khi
  thêm task chỉ 0,55% train ở `lr` (bảng trong `ket-qua/chi-phi.md`).
- **Đóng gói:** `scripts/pack_al_data.sh` → `packages_al/relearn_fix_v2_al_20261102.tar` (785 MB) và
  `seed_full_syn_r1_b2_20261102.tar` (993 MB), md5 trong `packages_al/MD5SUMS_20261102`.
- **Máy GPU:** `scripts/check_al_setup.sh` (kiểm dữ liệu, md5, torch/CUDA, chạy test có torch, dựng thử 1 gói),
  `docs/al/RUNBOOK.md` (lệnh từng bước).
- `scripts/run_al.sh`: thêm `dot2`, tự tạo gói vòng 2, chế độ `test ... --confirm-test`.

## 02/11/2026 (đêm) — gom dữ liệu vào một thư mục `autocheckout-data`

- Kiểm tra: `full_syn_rpc_synth_v2_v1_20261025.tar` (người dùng tải lại) = `data/` đã giải nén (cùng 101.681 file, mẫu 400
  file khớp SHA256 biên nhận); `relearn_fix_v2/eval` = `rpc_eval_official_v1`; ảnh `relearn_fix_v2/images/{v2,real}`
  trùng từng byte với `synth_v2/images` và `rpc/val2019`; model r1 hai nơi cùng md5.
- Dựng `Finalprpject/autocheckout-data/` (30 GB, 118.889 file) bằng hard link, không tốn thêm ổ: rpc/, eval/, synth_v2/,
  pretrained/, models/full-syn-r1/ (bỏ log/cache), relearn/ (bỏ phần trùng). `SHA256SUMS` toàn thư mục + `relearn/SHA256SUMS`.
- Code: `autocheckout/datapaths.py` (một chỗ khai báo đường dẫn; `AC_DATA` để đổi chỗ); `update_model.with_data_root`
  hiểu cấu trúc mới (và cấu trúc cũ relearn_fix_v1); tools `al_*` nhận `--data` thay cho `--release`; `run_naive` lấy
  đường dẫn từ datapaths; `run_al.sh`, `run_relearn_fix.sh`, `check_al_setup.sh` (thêm `FULL=1` kiểm SHA256) theo
  cấu trúc mới; bỏ `pack_al_data.sh` (thay bằng tar cả `autocheckout-data`).
- Kiểm: chọn lại ảnh trên cấu trúc mới ra đúng các tập cũ; r1 0,4232 (calib) / 0,4308 (Val-B); `--stage data` dựng
  được `al_margin_k_B500` với mọi đường dẫn trỏ vào `autocheckout-data`.
- Không còn dùng: `gpu_upload/`, `packages_al/` (hard link / gói cũ — xoá được).
