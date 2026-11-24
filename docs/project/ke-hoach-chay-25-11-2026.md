# Kế hoạch chạy ngày 25/11/2026: E1 (hướng A) + G0 (đo P1) + C0

Nhánh code: `feat/attn-prompt` (đã push, HEAD `4232630`).
Plan tổng: `attention-prompt-plan-v2-2026-11-24.md`.

## 0. Tóm tắt

| Việc | Ai | Máy | Thời gian ước tính | Kết quả |
|---|---|---|---|---|
| **Chuẩn bị code (P-1..P-3)** | Claude, tối 24/11 hoặc sáng 25/11 | CPU | 1–2 giờ | Một lệnh duy nhất chạy E1 + G0 trên GPU |
| **C0 + smoke test A + A-T7** | Claude | CPU (máy Mac) | 1–2 giờ | Số vật sót sửa được bằng chính sách đếm; A chạy được với r4 thật |
| **Thuê GPU, cài, chạy `attn_e1`** | **Bạn** | GPU (RTX 4090 trên Vast, như r4) | Khoảng 0,5 giờ cài + 2,5–3,5 giờ chạy | E1 xong, kết quả tự đồng bộ về Drive |
| Phân tích E1, so với r4 | Claude | CPU | 1 giờ | Bảng quyết định (Điểm dừng 3 của plan) |

Phát hiện khi lập kế hoạch:
- **E1 không có teacher.** r4 chạy `EXP-FB-FT.sh` với `--pseudo none`, mà `pdp/main.py:423` chỉ tạo teacher khi `pseudo != 'none'`. Câu hỏi "teacher khi xuất phát từ r3_real" vì vậy không còn. P2 (cache nhãn giả) không có tác dụng với E1, chỉ P1 có ích.
- **C0 không cần GPU.** `pred_val.npz` của r4 còn trên Drive: `gdrive:AutoCheckout/ket_qua_gpu/al_r4_20261114_165406/update-model/relearn-fix-v2/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5/pred_val.npz` (6,7 MB).
- **Tên lượt chạy quyết định siêu tham số và điểm xuất phát** (`autocheckout/update_model.py:90–137`). Muốn E1 dùng đúng dữ liệu, siêu tham số và checkpoint xuất phát của r4 mà không ghi đè r4, cần sửa nhỏ (P-1).

---

## 1. Việc Claude làm trước (cần bạn đồng ý)

### P-1. Lượt chạy `al_margin_k_B500_r4_pq` (sửa `autocheckout/update_model.py`)

| Sửa gì | Như thế nào | Mẫu làm theo |
|---|---|---|
| `_kind` | Tên kết thúc bằng `_r4_pq` trả về `r4_pq` | Cách vòng 3 xử lý tag (`_r3_real_s1`, `SEED_TAGS`) |
| `VARIANTS['r4_pq']` | `{**VARIANTS['r4'], 'per_query_prompt': 1}`: cùng epoch, lr, augment với r4 | `VARIANTS['r3_prompt']` (thêm `tune_old_private`) |
| `GATES['r4_pq']` | Như `r4` | `GATES` của tag seed |
| `_al_start` | `_r4_pq` xuất phát từ `<base>_r3_real`, so sánh với `<base>_r3_real`, đúng như r4 | Nhánh `_r4` hiện có |
| `_train` | Truyền `PER_QUERY_PROMPT=hp.get('per_query_prompt', 0)` vào môi trường | `TUNE_OLD_PRIVATE` |
| Dữ liệu | `_r4_pq` dùng thư mục variant của `_r4` (cùng ảnh, cùng `train_rows`) | `r3_data_tag` của lượt lặp seed |

Lý do ghi cờ vào siêu tham số của lượt chạy thay vì đặt biến môi trường bằng tay: cờ được lưu vào `resolved_cli.json` và `run_info`, nên không thể quên bật hay bật nhầm.

**Test:** `_kind`, `_al_start`, biến môi trường của `_train` khi chạy thử (dry run), và đường dẫn dữ liệu dùng chung với `_r4`.

### P-2. `scripts/run_al.sh`

Hàm `build`: tên `*_r4_pq` dựng dữ liệu qua `al_round2` giống `_r4`, hoặc dùng lại variant `_r4` nếu đã có. Thêm test vào `tests/test_al_r4_scripts.py`.

### P-3. Case `attn_e1` trong `scripts/vast_train_and_sync.sh`

| Bước | Lệnh bên trong | Thời gian |
|---|---|---|
| Kiểm tra đầu vào | Như case `al_r4`: r3_real `task_final.pth`, md5 weights, tập chọn `margin_k_B500_r3.json`, split Val-A | 1 phút |
| **G0-a: đo tốc độ** | `pdp/benchmark.py` với tham số mô hình của r4, `--bench_mode stages/infer/train`, chạy lần lượt `--reuse_features 0` và `1` | Khoảng 20 phút |
| **G0-b: kiểm P1 không đổi kết quả** | Xuất `pred_val.npz` của r4 hai lần (`--reuse_features 0`/`1`), so điểm (lệch ≤ 1e-4) và danh sách top-k (giống hệt) | Khoảng 15 phút |
| **E1** | `REUSE_FEATURES=1 bash scripts/run_al.sh al_margin_k_B500_r4_pq` (train, chấm Val-B, `al_score`) | Khoảng 2,5–3 giờ (r4: khoảng 3 giờ chưa có P1) |
| Phân rã lỗi | `python -m tools.error_breakdown` cho E1 | 2 phút |
| Đồng bộ | Phần kết thúc có sẵn của script tự `rclone copy` log, runs, scored lên `gdrive:AutoCheckout_Runs/attn_e1_<thời điểm>` | 5–10 phút |

**Luật dừng:**
- G0-b không đạt (điểm lệch > 1e-4 hoặc top-k khác): E1 chạy với `REUSE_FEATURES=0`, ghi lý do vào log.
- E1 lỗi: máy chỉ dừng (STOP), không bị hủy (destroy), như các case cũ.

### P-4. Việc CPU của Claude (không phụ thuộc GPU)

| Việc | Cách làm | Ra gì |
|---|---|---|
| **C0** | Tải `pred_val.npz` r4 từ Drive. Thử NMS IoU {0,45; 0,5; 0,55; 0,6}, NMS theo lớp, ngưỡng theo SKU co về ngưỡng nhóm. Chọn trên Val-A-calib, chấm trên Val-B bằng `tools/error_breakdown.py` | Giảm được bao nhiêu trong `miss_nms` (85) và `miss_threshold` (246), `extra_*` tăng bao nhiêu, cAcc mới |
| **Smoke test A trên r4 thật** | Nạp `models/AL-margin-k-r4/task_final.pth` với `--per_query_prompt 1`, suy luận 20 ảnh Val-A | Chạy không lỗi; thời gian/ảnh so với cách cũ |
| **A-T7** | Trên r4: IoU giữa box của cùng query ở lượt 1 và lượt 2, 50 ảnh Val-A | Giả định "query q nhắm cùng vật ở hai lượt" đúng hay sai |

---

## 2. Việc bạn làm ngày mai (GPU)

> Chỉ làm sau khi Claude báo P-1..P-3 đã xong, test xanh và đã push.

### Bước 1. Thuê máy (khoảng 10 phút)
- Vast.ai: **RTX 4090**, ổ đĩa ≥ 150 GB (như đợt r4).
- Trên Mac: `bash ~/Desktop/Finalprpject/vast.sh`, dán lệnh SSH của Vast, đặt tên `vast2`.
- Kiểm tra: `ssh vast2 nvidia-smi`.

### Bước 2. Cài code và dữ liệu (khoảng 20–30 phút)
```bash
ssh vast2
cd /workspace
git clone -b feat/attn-prompt git@github.com:GraduationProjec1/Model.git AutoCheckout-CL
cd AutoCheckout-CL
bash scripts/setup_vm.sh                   # môi trường Python, như các đợt trước
# gói r3 như đợt r4 (gpu_bundle_r3/install.sh), rồi:
BRANCH=feat/attn-prompt bash scripts/install_r4.sh
```
`install_r4.sh` kiểm md5 của r3_real và của tập chọn. Nếu báo `[LỖI]` thì dừng lại và gửi log cho Claude.

### Bước 3. Chạy (khoảng 3–3,5 giờ, chạy trong `tmux` để không mất khi rớt mạng)
```bash
tmux new -s attn
cd /workspace/AutoCheckout-CL
REUSE_FEATURES=1 INSTANCE_ID=<id máy Vast> VAST_API_KEY=<key> \
  bash scripts/vast_train_and_sync.sh attn_e1
# thoát tmux: Ctrl-b rồi d; vào lại: tmux attach -t attn
```

### Bước 4. Theo dõi
| Xem gì | Lệnh |
|---|---|
| Log tổng | `tail -f data/training/update-model/vast_logs_attn_e1_*/master_execution.log` |
| Tiến độ train | `cat data/training/update-model/relearn-fix-v2/runs/EXP-FB-FT-al_margin_k_B500_r4_pq/task_5/progress.json` |
| GPU có bận không | `nvidia-smi dmon -s u` (GPU-Util nên ≥ 80%) |
| Cờ A đã bật chưa | `grep -c per_query_prompt .../EXP-FB-FT-al_margin_k_B500_r4_pq/resolved_cli.json` phải ≥ 1; log có dòng `A: --per_query_prompt 1 drops ... query_tf` |

### Bước 5. Khi xong
- Script tự đồng bộ lên `gdrive:AutoCheckout_Runs/attn_e1_<thời điểm>/` rồi hủy hoặc dừng máy (`AUTO_DESTROY=1` mặc định).
- Báo Claude tên thư mục trên Drive để phân tích.

---

## 3. Claude phân tích sau E1

| Chỉ số | So với r4 | Nguồn |
|---|---|---|
| cAcc calib Val-B | 68,28% | `al_score` |
| Nhầm mã, tỉ lệ xuyên task | 564, 86,5% | `error_breakdown` |
| 6 cặp nặng nhất (98/97_milk…) | 64, 23, 17, 14, 11, 10 | `error_breakdown` |
| Sót / thừa | 628 / 214 | `error_breakdown` |
| Tham số train được | r4 | `run_info` |
| Giây/bước, tổng giờ GPU | r4 (khoảng 3 giờ) | `progress.json`, log |

Quyết định theo bảng ở §7 của plan v2. Nếu giữ A thì tiếp tục code B và C1 (E2, E4).

## 4. Rủi ro trong ngày

| Rủi ro | Xử lý |
|---|---|
| Máy Vast mới thiếu dữ liệu của đợt r4 | `install_r4.sh` kiểm từng file và báo `[LỖI]` sớm |
| P1 lệch trên GPU | G0-b phát hiện; E1 tự chạy lại với `REUSE_FEATURES=0` |
| E1 lỗi giữa chừng | Checkpoint mỗi 2 phút (`CKPT_EVERY_MINUTES`); chạy lại cùng lệnh sẽ tiếp tục (resume) |
| Hết giờ thuê | Không cần chạy B0/test2019. Chỉ G0 + E1, tổng khoảng 3,5 giờ |
