# Kế hoạch chạy GPU: E1 (A) → điểm dừng → E2 (A + B), E4 (A + C1)

Cập nhật: 26/11/2026. Nhánh: `feat/attn-prompt` (HEAD `7330d5f`, đã push).

Plan liên quan (cùng thư mục `docs/ai/planning/`):
- `2026-11-24-attention-prompt-plan-v2.md`: plan tổng, bảng quyết định (§7).
- `2026-11-25-plan-B-C1.md`: plan chi tiết B và C1, tiến độ (§9).
- `2026-11-24-huong-A-B-giai-thich.md`: giải thích ý tưởng A, B.
- `2026-11-24-query-attention-plan.md` (plan Q): **chưa code, có chủ ý.** Ngày 25/11 đã chốt gộp vào plan B/C1: lấy các phát hiện về code, còn Q1 (two-stage đầy đủ) và Q2 (tách hai luồng) đều phải chạy lại FSA nên để sau; Q3 (SAM) cần cổng G0-SAM trước. Xem `2026-11-25-plan-B-C1.md` §7.

---

## 0. Trạng thái

### 0.1 Code: xong cả 3 hướng

| Hướng | Nội dung | Commit | Kiểm trên r4 thật (CPU) |
|---|---|---|---|
| A | Mỗi query tự chọn hỗn hợp prompt, bỏ `query_tf` (−23,0M tham số) | `4232630` | Chạy được; 4,38 so với 4,48 giây/ảnh; **A-T7: IoU box cùng query giữa 2 lượt, trung vị 0,989** |
| Lệnh GPU + G0 | `attn_e1`, `tools/p1_check.py`, tên lượt chạy `_r4_pq` | `7294162` | `p1_check`: nhanh hơn 47,9%, giống hệt từng bit |
| B | Prompt hiệu chỉnh offset và trọng số của deformable cross-attn (W = 0 lúc đầu) | `7330d5f` | **Giống hệt r4 lúc bắt đầu (5/5 ảnh)**; không chậm hơn |
| C1 | K = 100 query đề xuất từ encoder, loss phụ `loss_prop` | `7330d5f` | 400 query, logits hữu hạn, lượt 2 dùng lại đề xuất lượt 1; không chậm hơn |

Toàn bộ bộ test: chỉ còn 3 lỗi có sẵn từ trước (`test_naive_lower_bound...`, `test_serial_and_parallel_writes...`, `test_resume_does_not_launch...`).

### 0.2 Việc CPU còn lại (Claude, không chặn GPU)

| Việc | Trạng thái |
|---|---|
| Smoke A, A-T7, smoke B, smoke C1 trên r4 thật | ✅ Xong |
| **C0**: chính sách đếm (NMS IoU {0,45; 0,5; 0,55; 0,6}, NMS theo lớp, ngưỡng theo SKU) trên `pred_val.npz` của r4 tải từ Drive, chọn trên Val-A-calib, chấm Val-B | ⏳ Chưa làm. Nguồn: `gdrive:AutoCheckout/ket_qua_gpu/al_r4_20261114_165406/update-model/relearn-fix-v2/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5/pred_val.npz` |

### 0.3 Phát hiện ảnh hưởng tới cách chạy

- **Không có teacher** trong các lượt E (r4 chạy `EXP-FB-FT.sh` với `--pseudo none`; `pdp/main.py` chỉ tạo teacher khi `pseudo != 'none'`). P2 (cache nhãn giả) không có tác dụng; chỉ P1 giúp giảm thời gian.
- **Tên lượt chạy quyết định mọi thứ** (`autocheckout/update_model.py`, `R4_TAGS`): dữ liệu, siêu tham số, điểm xuất phát (r3_real), gate đều giống r4; chỉ khác cờ của hướng tương ứng. Không phải đặt biến môi trường bằng tay.

---

## 1. Các lệnh GPU

| Lệnh | Lượt | Tên lượt chạy | Cờ bật | Module học với lr 2e-5 | Khi nào chạy |
|---|---|---|---|---|---|
| `attn_e1` | **E1** | `al_margin_k_B500_r4_pq` | A | — | **Đợt 1** |
| `attn_e2` | **E2** | `al_margin_k_B500_r4_pqx` | A + B | `prompts.xattn` | Đợt 2, sau Điểm dừng 3 |
| `attn_e4` | **E4** | `al_margin_k_B500_r4_pqc` | A + C1 (K = 100) | `proposals` | Đợt 2, sau Điểm dừng 3 |
| `attn_e3` | E3 | `al_margin_k_B500_r4_x` | Chỉ B | `prompts.xattn` | Đợt 3, chỉ khi cần (xem §5) |
| `attn_e5` | E5 | `al_margin_k_B500_r4_pqxc` | A + B + C1 | `prompts.xattn,proposals` | Đợt 3, chỉ khi E2 và E4 đều có ích |

Mỗi lệnh làm hết một lượt:
1. Kiểm đầu vào; thiếu thì thoát mã 3, **trước khi tốn GPU**.
2. Kiểm md5 weights.
3. **G0**: `p1_check` trên r3_real, khoảng 15 phút.
4. Train lượt chạy với `REUSE_FEATURES=1`, tự đổi sang 0 nếu G0 không đạt. Khoảng 2,5–3 giờ.
5. Chấm Val-B (`al_score`).
6. Phân rã lỗi (`error_breakdown`).
7. Đồng bộ lên Drive.

**Mỗi lượt khoảng 3–3,5 giờ.**

---

## 2. Cài máy (làm một lần cho mỗi máy)

Theo `Hướng dẫn chạy vòng 3 trên GPU (09 11 2026).md`, chỉ khác những chỗ **in đậm**.

### Bước 1. Thuê máy (hướng dẫn §2–3, khoảng 10 phút)
- Vast.ai: 1× RTX 4090 24 GB, RAM ≥ 32 GB, ≥ 16 CPU, template **PyTorch (Vast)** kiểu SSH, **Disk 150 GB**.
- Trên Mac: `cd ~/Desktop/Finalprpject && bash vast.sh`, rồi `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` và `ssh vast2`.

### Bước 2. Môi trường (hướng dẫn §4, khoảng 10 phút)
```bash
tmux new -s training
nvidia-smi && df -h /workspace && ssh -T git@github.com
curl https://rclone.org/install.sh | bash
pip install -U vastai
mkdir -p ~/.config/rclone && nano ~/.config/rclone/rclone.conf   # dán phần [gdrive] từ Mac
rclone lsd gdrive:AutoCheckout
cd /workspace
git clone git@github.com:GraduationProjec1/Model.git AutoCheckout-CL
cd AutoCheckout-CL && git checkout feat/attn-prompt          # <-- nhánh mới
pip install -U uv
uv venv --seed -p 3.10 /workspace/venv-pdp
source /workspace/venv-pdp/bin/activate
uv pip install --index-strategy unsafe-best-match -r configs/environment/requirements-lock-cu121.txt
uv pip install --no-deps -e .
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.2.2+cu121 True
```

### Bước 3. Dữ liệu 32 GB (hướng dẫn §5, khoảng 10–20 phút)
Làm đúng như §5: tải `autocheckout-data.tar`, kiểm md5 `5a1efb48209d998012b01dd1dfa9674c`, giải nén, `SHA_OK`.

### Bước 4. Gói vòng 3, rồi gói vòng 4 trên nhánh mới (khoảng 15 phút)
```bash
bash /workspace/gpu_bundle_r3/install.sh                      # hướng dẫn §6 (ghi đè code bằng commit cũ)
cd /workspace/AutoCheckout-CL
BRANCH=feat/attn-prompt bash scripts/install_r4.sh            # <-- BRANCH bắt buộc: đưa code về nhánh mới
git log -1 --oneline                                         # phải là commit mới nhất của feat/attn-prompt
```
`install_r4.sh` kiểm từng file và md5. Có `[LỖI]` thì dừng và gửi log cho Claude.

### Bước 5. Kiểm trước khi tốn GPU (khoảng 5 phút)
```bash
source /workspace/venv-pdp/bin/activate
cd /workspace/AutoCheckout-CL
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 AC_DATA=/workspace/autocheckout-data
env -u AC_DATA python -m pytest -q -p no:cacheprovider tests/test_update_model.py tests/test_al_r4_scripts.py \
    tests/test_pdp_a_per_query_prompt.py tests/test_pdp_b_prompt_xattn.py tests/test_pdp_c1_query_proposals.py \
    tests/test_p1_check.py
bash scripts/run_al.sh al_margin_k_B500_r4_pq --dry-run      # đổi tên theo lượt sắp chạy
```
**Đúng khi:** test `passed`; dry-run in số ảnh của lượt chạy và không báo thiếu file. Lần đầu, dry-run dựng dữ liệu `_r4` bằng `al_round2`; các lượt sau dùng lại.

---

## 3. Đợt 1: E1 (hướng A)

```bash
INSTANCE_ID=<id máy Vast> VAST_API_KEY=<key> bash scripts/vast_train_and_sync.sh attn_e1
# thoát tmux: Ctrl-b rồi d; vào lại: tmux a -t training
```
- Không cần đặt `REUSE_FEATURES`: lệnh tự bật P1 khi G0 đạt.
- Xong thì tự đồng bộ lên `gdrive:AutoCheckout_Runs/attn_e1_<thời điểm>/` rồi **hủy máy** (`AUTO_DESTROY=1` mặc định). Có bước lỗi thì máy chỉ dừng (STOP).
- **Báo Claude tên thư mục trên Drive.**

---

## 4. Theo dõi (dùng cho mọi lượt)

Đặt tên lượt đang chạy trước:
```bash
V=al_margin_k_B500_r4_pq          # E1; E2: _r4_pqx; E4: _r4_pqc; E3: _r4_x; E5: _r4_pqxc
RUN=data/training/update-model/relearn-fix-v2/runs/EXP-FB-FT-$V
```

| Xem gì | Lệnh |
|---|---|
| Log tổng | `tail -f data/training/update-model/vast_logs_attn_e*/master_execution.log` |
| G0 đạt chưa | `cat data/training/update-model/vast_logs_attn_e*/g0_p1_check/p1_check_summary.json` (`"passed": true`, `saving_total`) |
| Tiến độ train | `cat $RUN/task_5/progress.json` |
| GPU có bận không | `nvidia-smi dmon -s u` (GPU-Util nên ≥ 80%) |
| **Cờ đã bật đúng chưa** | `python -c "import json,sys; a=json.load(open(sys.argv[1]))['arguments']; print({k: a[i+1] for i, k in enumerate(a[:-1]) if k in ('--per_query_prompt', '--local_query', '--prompt_cross_attn', '--query_proposals', '--new_params')})" $RUN/resolved_cli.json` |

**Kết quả đúng của lệnh "cờ đã bật đúng chưa":**

| Lượt | `--per_query_prompt` | `--local_query` | `--prompt_cross_attn` | `--query_proposals` | `--new_params` |
|---|---|---|---|---|---|
| E1 | 1 | 0 | 0 | 0 | `class_embed,prompts.private` |
| E2 | 1 | 0 | 1 | 0 | `class_embed,prompts.private,prompts.xattn` |
| E4 | 1 | 0 | 0 | 100 | `class_embed,prompts.private,proposals` |
| E3 | 0 | 1 | 1 | 0 | `class_embed,prompts.private,prompts.xattn` |
| E5 | 1 | 0 | 1 | 100 | `class_embed,prompts.private,prompts.xattn,proposals` |

**Dòng phải có trong `$RUN/task_5/train.log`:**

| Lượt | Dòng | Ý nghĩa |
|---|---|---|
| E1, E2, E4, E5 | `A: --per_query_prompt 1 drops ... query_tf` | Checkpoint r3_real được chuyển sang chọn prompt theo query |
| E2, E3, E5 | `D3: the checkpoint has no [...xattn...]` | Module B bắt đầu từ 0 |
| E2, E3, E5 | `B: epoch N {'alpha': [...], 'w_off': [...], 'w_att': [...]}` mỗi epoch | `w_off`, `w_att` phải **tăng dần từ 0**. Đứng yên ở 0 thì B không học. Tăng vọt (> 0,1) thì báo Claude |
| E4, E5 | `D3: the checkpoint has no [...proposals...]` | Module C1 bắt đầu từ khởi tạo |
| E4, E5 | `PROP=` (loss phụ) trên thanh tiến độ trong `$(find data/training/update-model/relearn-fix-v2/attempts -name "EXP-FB-FT-${V}_val_runner.log")` | Phải **giảm dần**. Đứng yên thì đầu đề xuất không học |

---

## 5. Điểm dừng 3: đọc E1 rồi chọn đợt 2 (Claude phân tích, bạn quyết định)

Chỉ số so với r4:

| Chỉ số | r4 | Nguồn |
|---|---|---|
| cAcc calib Val-B | 68,28% | `al_score` |
| Nhầm mã, tỉ lệ xuyên task | 564, 86,5% | `error_breakdown` |
| 6 cặp nặng nhất (98/97_milk…) | 64, 23, 17, 14, 11, 10 | `error_breakdown` |
| Sót (`miss` / `miss_nms` / `miss_threshold`) | 297 / 85 / 246 | `error_breakdown` |
| Thừa | 214 | `error_breakdown` |
| Quên (gate: `pair_drop`, AP50 SKU khác) | gate của r4 | `stages/<V>.json` |
| Giây/bước, tổng giờ GPU | khoảng 3 giờ | `progress.json`, log |

| Kết quả E1 | Đợt 2 |
|---|---|
| cAcc không giảm | **E2 + E4** trên nền A (như bảng §1) |
| cAcc giảm | Bỏ A. **E3** (chỉ B, nền r4). C1 cần `--local_query 0`, nên Claude thêm tên lượt `_r4_c` (C1 với prompt theo ảnh dạng trung bình) trước khi chạy |

---

## 6. Đợt 2: E2 và E4

**Cách 1: hai máy song song (nhanh nhất, khoảng 3,5 giờ).** Mỗi máy làm §2 (cài máy) rồi:
```bash
INSTANCE_ID=<id> VAST_API_KEY=<key> bash scripts/vast_train_and_sync.sh attn_e2   # máy 1
INSTANCE_ID=<id> VAST_API_KEY=<key> bash scripts/vast_train_and_sync.sh attn_e4   # máy 2
```

**Cách 2: một máy, chạy liên tiếp (khoảng 7 giờ).** Lượt đầu **phải tắt tự hủy máy**, nếu không máy bị xóa sau E2:
```bash
AUTO_DESTROY=0 bash scripts/vast_train_and_sync.sh attn_e2 && \
  INSTANCE_ID=<id> VAST_API_KEY=<key> bash scripts/vast_train_and_sync.sh attn_e4
```
Mỗi lượt đều chạy lại G0 (khoảng 15 phút), chấp nhận để mỗi lượt tự kiểm.

Theo dõi theo §4 với `V=al_margin_k_B500_r4_pqx` (E2) và `V=al_margin_k_B500_r4_pqc` (E4).

**Phân tích thêm (Claude):**
- E2: nhầm mã cùng nhóm hàng (41,5% của r4) và các cặp 98/97_milk có giảm không; `w_off`/`w_att` theo epoch.
- E4: `miss` (297 của r4) có giảm không; `extra_dup`/`extra_shift` có tăng không; tỉ lệ GT được ghép với query đề xuất.

Quyết định đợt 3 theo bảng §7 của plan v2: E2 và E4 đều có ích thì chạy E5 (`attn_e5`); cần đối chứng B thì chạy E3 (`attn_e3`).

### 6.1 Chạy hết trên một máy (cài một lần)

Cả A, B, C1 đều trên nhánh `feat/attn-prompt`, nên **một máy cài theo §2 chạy được mọi lượt**. Các lượt độc lập (lượt nào cũng xuất phát từ r3_real), nên thứ tự không làm sai kết quả.

```bash
AUTO_DESTROY=0 bash scripts/vast_train_and_sync.sh attn_e1 && \
AUTO_DESTROY=0 bash scripts/vast_train_and_sync.sh attn_e2 && \
AUTO_DESTROY=0 bash scripts/vast_train_and_sync.sh attn_e4 && \
  INSTANCE_ID=<id> VAST_API_KEY=<key> bash scripts/vast_train_and_sync.sh attn_e5
```

| Điểm | Giải thích |
|---|---|
| `AUTO_DESTROY=0` cho mọi lượt trừ lượt cuối | Không có thì máy bị xóa ngay sau lượt đầu |
| `&&` | Một lượt lỗi thì dừng chuỗi; máy vẫn còn để xem log (không truyền `VAST_API_KEY` ở các lượt giữa nên máy không bị dừng hay xóa) |
| Thời gian | Khoảng 3–3,5 giờ mỗi lượt; 4 lượt khoảng 13–14 giờ |
| Kết quả | Mỗi lượt đồng bộ lên một thư mục riêng `gdrive:AutoCheckout_Runs/attn_eN_<thời điểm>/` |
| Đánh đổi | Chạy hết một lần thì **bỏ qua Điểm dừng 3** (§5): nếu E1 cho thấy A làm cAcc giảm, giờ GPU của E2/E4/E5 (đều có A) coi như phí. Muốn an toàn: chạy `attn_e1` trước, đọc kết quả, rồi mới chạy phần còn lại trên cùng máy (nhớ `AUTO_DESTROY=0` cho E1) |

---

## 7. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Máy mới thiếu dữ liệu của đợt r4 | `install_r4.sh` và bước kiểm đầu vào của lệnh `attn_e*` báo `[THIẾU]`/`[LỖI]` trước khi dùng GPU |
| P1 lệch trên GPU | G0 phát hiện; lượt tự chạy với `REUSE_FEATURES=0`; máy chỉ STOP để xem lại |
| Lượt lỗi giữa chừng | Checkpoint mỗi 2 phút; chạy lại cùng lệnh sẽ tiếp tục (resume) |
| Chạy hai lượt trên một máy mà quên `AUTO_DESTROY=0` | Máy bị hủy sau lượt đầu, phải cài lại. Xem §6 cách 2 |
| B không học (`w_off` đứng ở 0) hoặc hiệu chỉnh tăng vọt | Theo dõi dòng `B: epoch` (§4). Báo Claude |
| C1 làm vật thừa tăng | Phân rã lỗi có `extra_dup`, `extra_shift`; so với r4 |
| P1 lúc **train** chưa có A/B trên GPU (Task 10 của plan perf) | Unit test chứng minh P1 train dùng chung backbone đóng băng (xác định). So `progress.json` với r4 để phát hiện bất thường |
| Hết giờ thuê | Không chạy B0/test2019. Chỉ G0 + lượt E |
