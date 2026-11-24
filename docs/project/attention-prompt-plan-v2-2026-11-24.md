# Plan v2: giảm nhầm mã (A, B), giảm sót vật (C) trên nền tối ưu chi phí P1/P2 (24/11/2026)

Trạng thái: **đã duyệt 24/11** (người dùng: "yes"). Code làm trên nhánh `feat/attn-prompt` tách từ `perf/two-pass-reuse` @ `4df908d`, trong worktree `AutoCheckout-CL-perf`.
Thay thế: `attention-prompt-plan-2026-11-24.md` (v1).
Tài liệu ý tưởng: `huong-A-B-giai-thich-2026-11-24.md`.
Nguồn tối ưu chi phí: worktree `AutoCheckout-CL-perf`, nhánh `perf/two-pass-reuse` (`7630316` P1/P2 + `4df908d` xoá comment), và thiết kế `docs/design/2026-11-22-toi-uu-2-luot/` (ở thư mục gốc `Finalprpject`).

Ký hiệu: **[PERF]** đánh dấu nội dung mới so với v1, lấy từ hoặc dựa trên `AutoCheckout-CL-perf`.

---

## Thay đổi so với v1 (thêm gì, ở đâu)

| # | Thêm gì | Ở mục nào | Vì sao |
|---|---|---|---|
| 1 | **Đổi nhánh nền**: code A/B/C dựng trên `perf/two-pass-reuse` @ `4df908d`, không dựng trên `feat/dot-gpu-1` @ `f50bc66` | §2 | Thừa hưởng P1 (backbone + encoder chạy 1 lần cho 2 lượt) và P2 (cache nhãn giả teacher) |
| 2 | **Cổng GPU G0**: chạy Task 2, 9, 10 của plan perf trước mọi thí nghiệm A/B/C | §2.3, §8 | P1/P2 mới kiểm trên CPU. Trên GPU chưa đo tốc độ và chưa có A/B train |
| 3 | Mọi lượt E chạy với `REUSE_FEATURES=1`, `CACHE_PSEUDO=auto` | §7 | Giảm giờ GPU cho cả 3–6 lượt train |
| 4 | Ràng buộc thiết kế **"encoder không được phụ thuộc prompt"** cho A, B, C | §2.2 | Điều kiện để P1 còn đúng. A, B, C đều thỏa |
| 5 | Hướng A sửa ở **2 chỗ** thay vì 3: `two_pass.prompted_outputs` (eval, teacher, suy luận) và `common_step` (train) | §4 A3 | P1 đã gom đường tạo query vào `pdp/two_pass.py:29` |
| 6 | Hướng A bỏ luôn bước matcher ở lượt 1 khi train | §4 A3 | Khi không còn `query_loss` thì không cần ghép cặp ở lượt 1. Rẻ hơn và đơn giản hơn |
| 7 | Hướng C1 lấy đề xuất từ **đầu ra encoder đã dùng chung** của P1. Lượt 2 dùng lại đề xuất của lượt 1 | §6 C1 | Không tính encoder thêm lần nào cho đề xuất |
| 8 | Hướng C2 (SAM) **tính đề xuất một lần offline rồi cache theo `image_id`**, theo cách P2 cache nhãn giả | §6 C2 | Không chạy SAM trong vòng lặp train |
| 9 | Bảng chi phí: báo cả **giây/bước train** và **giây/ảnh suy luận**, đo bằng `pdp/benchmark.py --bench_mode stages/train/infer` của nhánh perf | §7 | Đo chi phí bằng cùng công cụ với perf |
| 10 | Test mới phải chạy cả khi **bật và tắt** `--reuse_features`, `--cache_pseudo`. Kết quả 2 chế độ phải khớp nhau | §4, §5, §6 | Giữ đảm bảo "giống hệt từng bit" của P1 |
| 11 | Quy ước code lấy từ perf: cờ mặc định tắt, ghi vào `training_contract` chỉ khi khác mặc định, chặn cấu hình sai ngay lúc khởi động | §3 | Checkpoint cũ vẫn resume được; lỗi cấu hình lộ sớm |
| 12 | 204 file xoá comment trong worktree perf: **đã kiểm (AST .py giống hệt, test chỉ 3 lỗi có sẵn), commit `4df908d` và push** theo yêu cầu 24/11. Mọi số dòng trong plan tính theo `4df908d` | §2.1 | Code và plan làm trên nền perf |
| 13 | Mục "để sau": bỏ hẳn lượt 1 khi dùng A (chọn prompt theo trạng thái ngay trong lượt) | §10 | Giảm tiếp chi phí train, nhưng đổi ngữ nghĩa nên không làm ngay |

Phần **không đổi** so với v1: kết quả Bước 0 (§1), ý tưởng và thiết kế toán học của A, B, C, tài liệu tham khảo, quy tắc quyết định.

---

## 0. Yêu cầu (nhắc lại)

- **Mục tiêu:**
  - Giảm **nhầm mã** (564 vật, 86,5% là nhầm xuyên task).
  - Giảm **sót vật** (628 vật = 331 vật bị lọc mất + 297 vật mô hình không thấy).
  - Mức quên không tăng.
- **[PERF] Ràng buộc chi phí:**
  - Không làm mất phần tiết kiệm của P1/P2.
  - Phần thêm của A/B/C phải được đo bằng cùng công cụ.
- **Ngân sách:** 3 lượt GPU chính (E1, E2, E4), thêm lượt nếu kết quả đòi hỏi.
- **Sửa trích dẫn:** PDP = arXiv 2603.02286, P²IOD = arXiv 2510.27316. Sửa ở `Report copy.md:32` và `Report copy 2.md:30`, chờ đồng ý.

---

## 1. Kết quả Bước 0 (giữ nguyên từ v1)

| Lỗi | Số liệu chính |
|---|---|
| Nhầm mã 564 | 86,5% xuyên task (ngẫu nhiên: 76,8%). 41,5% cùng nhóm hàng. Cặp nặng nhất 98_milk (t4) ↔ 97_milk (t5): 64 vụ. 6 cặp nặng nhất đều xuyên task |
| Sót 628 | `miss_threshold` 246 + `miss_nms` 85 (mô hình đã thấy) và `miss` 297 (không thấy) |
| Ảnh | Trung bình 14 vật mỗi ảnh có nhầm mã. Gần như mọi ảnh đều chứa SKU của nhiều task |

---

## 2. [PERF] Nền code và cổng chi phí

### 2.1 Nhánh và worktree

| Việc | Chi tiết |
|---|---|
| Nhánh mới | `feat/attn-prompt`, tách từ `perf/two-pass-reuse` @ `4df908d` (đã tạo) |
| Worktree | Làm ngay trong `AutoCheckout-CL-perf` (đã sạch sau khi commit `4df908d`). Plan và tài liệu nằm ở `AutoCheckout-CL-perf/docs/project/` |
| Khi P1/P2 được gộp vào `feat/dot-gpu-1` | Rebase `feat/attn-prompt` lên đó |

### 2.2 Ràng buộc để P1 còn đúng

P1 dùng lại **backbone ở mọi chỗ** và **encoder ở chế độ eval** cho lượt 2 (`pdp/two_pass.py:30`). Điều đó chỉ đúng khi **encoder không phụ thuộc prompt**:

| Hướng | Chạm vào | Encoder có đổi theo prompt không | P1 còn đúng không |
|---|---|---|---|
| A | Chọn prompt, self-attn decoder | Không | Đúng |
| B | Cross-attn decoder (`MultiscaleDeformableAttention` của decoder) | Không. Encoder dùng cùng class nhưng **không bao giờ** nhận delta (test B-T4) | Đúng |
| C1 | Đầu chấm điểm đề xuất trên đầu ra encoder, query thêm cho decoder | Không. Đầu đề xuất đọc encoder, không ghi vào encoder | Đúng |
| C2 | Đề xuất từ SAM (ngoài mô hình) | Không | Đúng |

Ghi nhận: P1 **chặn** khi backbone được train (`engine.py:778–789`, cấu hình M5). Các lượt E đều để backbone đóng băng.

### 2.3 Cổng G0: kiểm P1/P2 trên GPU (Task 2, 9, 10 của plan perf)

| Task perf | Việc | Đạt khi |
|---|---|---|
| 2 | Đo trước: tỉ lệ backbone/encoder/decoder, thời gian đọc dữ liệu, GPU-Util | Có số đo |
| 9 | Đo sau | Train đợt ≥ 2 giảm ≥ 25% giây/bước (epoch đầu), ≥ 35% khi cache trúng. Suy luận giảm ≥ 30% |
| 10 | A/B train cũ vs mới | Kết quả trong mức nhiễu (tiêu chí mục 11 của `01-plan.md`) |

**Nếu G0 không đạt:** vẫn làm A/B/C trên nhánh mới nhưng chạy các lượt E với `REUSE_FEATURES=0`, `CACHE_PSEUDO=0`. Khi đó E vẫn so sánh công bằng với r4, chỉ mất phần tiết kiệm.

---

## 3. [PERF] Quy ước code phải theo

| Loại | Lấy theo | Quy ước |
|---|---|---|
| Cờ | `pdp/main.py` (perf: `--reuse_features`, `--cache_pseudo`) | Mặc định **tắt**, hành vi cũ giữ nguyên khi tắt |
| Bật qua config | `configs/exp/native/common.sh:30` | `--per_query_prompt "${PER_QUERY_PROMPT:-0}"`, và tương tự cho `PROMPT_XATTN`, `QUERY_PROPOSALS` |
| Ghi vào checkpoint | Commit `c61f068`: `training_contract` chỉ chứa cờ khi khác mặc định | Checkpoint r4 và cũ hơn vẫn resume được |
| Chặn cấu hình sai | `engine.py:62–72` (P2), `:778–789` (P1) | `ValueError` kèm thông điệp rõ ngay lúc khởi động |
| Log | `print(..., file=self.args.log_file)` | Như P1/P2 |
| Test | `tests/test_pdp_p1_reuse_features.py`, `tests/pdp_helpers.py` (model tí hon có `image_id`) | File mới `tests/test_pdp_a_per_query_prompt.py`, `test_pdp_b_prompt_xattn.py`, `test_pdp_c1_query_proposals.py`. Chạy trên CPU |

---

## 4. Hướng A: chọn prompt theo từng query

**Mục đích:** mỗi vật dùng hỗn hợp prompt của chính nó, để giảm nhầm xuyên task.
**Cơ sở:** Bước 0 (86,5% nhầm xuyên task) và P²IOD ("prompts pool confusion").

| # | File (dòng ở `4df908d`) | Sửa gì |
|---|---|---|
| A1 | `pdp/models/prompt.py:229–` (`forward`; `query_tf` tạo ở `:28`) | Nhánh `per_query`: x (B,300,256), không qua `query_tf`. Tính `w = cos(x⊙A, K)` theo từng query bằng cách tách đại số: `x @ (A⊙K̂)ᵀ / sqrt(x² @ (A²)ᵀ)`. Ra Ek, Ev (B,300,5,256). Không tạo `query_tf` |
| A2 | `pdp/models/modeling_deformable_detr.py:733–736` | Self-attn: nếu `pk` 4 chiều thì tính điểm của mỗi query với prefix riêng, ghép với `QKᵀ`, softmax, nhân với `[Ev_q ; V]`. Nếu 3 chiều thì chạy code cũ |
| **A3 [PERF]** | `pdp/two_pass.py:29` | Thêm `per_query` vào `prompted_outputs`: `query = first.last_hidden_state` (B,300,256). **Một chỗ sửa này phủ cả teacher (`engine.py:206–215`), validation (`:666`), `evaluate` và suy luận (`inference.py:42–48`, `:96`)** |
| **A3 [PERF]** | `pdp/engine.py:453–481` (train) | Lượt 1: lấy `last_hidden_state` làm query. **Bỏ matcher và `one_hot_proposals` ở lượt 1** (`:464–470`), vì chúng chỉ phục vụ `query_loss`. Bỏ `query_tf` và `query_loss` (`:475–478`, `:571–574`) |
| A4 | `pdp/main.py`, `engine.py` (truyền vào `detr_config`), `configs/exp/native/common.sh` | Cờ `--per_query_prompt`. Chặn khi bật cùng `--local_query 1`. Biến môi trường `PER_QUERY_PROMPT` |

**Test:**
- A-T1: cờ tắt thì đầu ra giống hệt.
- A-T2: 300 query giống nhau thì kết quả trùng với bản chọn theo ảnh.
- A-T3: kích thước và bộ nhớ đúng thiết kế.
- A-T4: private của task cũ không có gradient.
- A-T5: tham số giảm đúng 23.040.300.
- A-T6: bật cùng `--local_query` thì báo lỗi.
- A-T7: IoU box của cùng query giữa lượt 1 và lượt 2 (chạy trên checkpoint r4).
- **A-T8 [PERF]:** bật A, chạy `--reuse_features 0` và `1`: đầu ra eval **giống hệt từng bit**. Teacher qua `prompted_outputs` cho cùng nhãn giả với và không có `--cache_pseudo`.

**Chi phí:**
- −23,0M tham số.
- **[PERF]** Bỏ matcher ở lượt 1 khi train.
- Phần tính thêm: (B,300,K)·(K,2560) mỗi lớp, nhỏ so với encoder.

---

## 5. Hướng B: prompt điều biến deformable cross-attention

**Mục đích:** prompt hiệu chỉnh vị trí và trọng số lấy mẫu, nhắm vào nhầm mã cùng nhóm hàng (41,5%) và cặp 97/98_milk.

| # | File | Sửa gì |
|---|---|---|
| B1 | `pdp/models/prompt.py` | `xattn_delta_off_{l}`: Linear(256→256), `xattn_delta_att_{l}`: Linear(256→128), **khởi tạo 0**. `α_l = α_max·tanh(β_l)`. `p_q = mean(Ek_q, Ev_q)` |
| B2 | `modeling_deformable_detr.py:633–644` (deformable attn, class ở `:549`), `:944` (decoder layer), `:1325` (decoder loop) | `offset += α·Δoff`, `weights = softmax(w(h) + α·Δatt)`. Nếu `xattn_delta=None` thì chạy code cũ. **Encoder không nhận delta** |
| B3 | `engine.py:95` (`SHARED_AFTER_TASK1`) và logic khóa | Đóng băng `xattn_delta` sau task 1 (`--freeze_xattn_after_task1`, mặc định 1) |
| B4 | `main.py`, `common.sh` | `--prompt_cross_attn`, `--xattn_alpha_max`. Log `mean|α·Δoff|`. Biến môi trường `PROMPT_XATTN` |

**Test:**
- B-T1: Δ = 0 thì đầu ra giống hệt.
- B-T2: `|α| ≤ α_max`.
- B-T3: từ task 2 không có gradient vào `xattn_delta`.
- B-T4: đầu ra encoder không đổi khi bật B.
- B-T5: ở task 1 có gradient vào `xattn_delta`.
- B-T6: tham số thêm khoảng 0,59M.
- **B-T7 [PERF]:** bật B, chạy `--reuse_features 0` và `1`: giống hệt từng bit ở eval. B-T4 là điều kiện để P1 dùng lại encoder.

**Chi phí:** +0,59M tham số. Thêm 2 Linear nhỏ mỗi lớp decoder, không đáng kể.

---

## 6. Hướng C: giảm sót vật

### C0. Chính sách đếm (chỉ CPU)

Thử NMS IoU cao hơn 0,45, NMS theo lớp, và ngưỡng theo SKU có co về ngưỡng nhóm. Chọn trên Val-A-calib, đo bằng `tools/error_breakdown.py`.
**Điều kiện:** tìm lại `pred_val.npz` của r4 và file nhãn Val-B/Val-A-calib, vì đường dẫn cũ đã mất.

### C1. Query đề xuất từ encoder

| # | File | Sửa gì |
|---|---|---|
| C1.1 | `pdp/models/prompt.py` (module mới đặt trong `prompts` để được train) | Đầu chấm "có vật" trên đầu ra encoder (theo khung `enc_output` của two-stage, `modeling_deformable_detr.py:1572–1576`). Top-K (K = 100) cho ra điểm tham chiếu, MLP cho ra query_pos |
| C1.2 | `modeling_deformable_detr.py` (đoạn tạo query cho decoder) | Ghép K query đề xuất với 300 query học được. Hungarian trên 300 + K |
| C1.3 | `engine.py` (loss) | Loss phụ cho đầu đề xuất: focal "có vật" + L1/GIoU như encoder loss của two-stage |
| **C1.4 [PERF]** | `pdp/two_pass.py`, `engine.py` (train) | Đề xuất tính từ đầu ra encoder **của lượt 1** rồi truyền sang lượt 2 cùng `encoder_outputs`. Ở eval (P1), encoder chạy 1 lần nên đề xuất cũng tính 1 lần |

**Test:**
- C1-T1: cờ tắt thì giống hệt.
- C1-T2: K = 0 thì giống hệt.
- C1-T3: đề xuất ở lượt 1 và lượt 2 trùng nhau.
- **C1-T4 [PERF]:** giống hệt từng bit khi `--reuse_features 0` và `1`.

**Chi phí:** khoảng +0,3M tham số. Decoder xử lý 300 + K query: tăng khoảng 33% phần decoder, mà decoder chỉ khoảng 2% tổng thời gian (số đo CPU của perf: backbone + encoder = 97,9%).

### C2. Đề xuất từ SAM (chỉ làm nếu C1 không đủ)

- **[PERF]** Tính SAM 2.1 automatic mask cho mọi ảnh train/val **một lần, offline**, lưu theo `image_id`, cùng cách khóa theo `image_id` của cache P2. Khi train chỉ đọc cache.
- Suy luận thật vẫn phải chạy SAM. Đo FPS riêng trước khi cân nhắc dùng ở quầy.

---

## 7. Ma trận thí nghiệm

**[PERF]** Mọi lượt E chạy `REUSE_FEATURES=1 CACHE_PSEUDO=auto`, trừ khi G0 không đạt.
Điều kiện công bằng như v1: cùng điểm xuất phát r3_real, cùng seed, cùng dữ liệu và lịch học, chọn cấu hình trên Val-B/Val-A-calib.

| Mã | Cấu hình | GPU | Ưu tiên |
|---|---|---|---|
| **G0 [PERF]** | Task 2, 9, 10 của plan perf | Ngắn | 0 (cổng) |
| C0 | r4 + chính sách đếm | Không | 1 |
| **E1** | A | 1 lượt | 2 |
| **E2** | A + B | 1 lượt | 3 |
| **E4** | A + C1 | 1 lượt | 3 |
| E3 | chỉ B | 1 lượt | Nếu cần |
| E5 | A + B + C1 | 1 lượt | Nếu E2 và E4 đều có ích |
| E6 | A + C2 | 1 lượt | Nếu C1 không đủ |

**Báo cáo cho mỗi lượt:**
1. cAcc calib Val-B.
2. `swap` (xuyên task / cùng task, cùng nhóm hàng / khác nhóm), `miss_*`, `extra_*`, 6 cặp nặng nhất.
3. Mức quên trên task 1–4.
4. Tham số train được (theo nhóm).
5. **[PERF] Chi phí:**

| Chỉ số | Công cụ | So với |
|---|---|---|
| Giây/bước train (đợt 1, đợt ≥ 2, cache trúng) | `pdp/benchmark.py --bench_mode train` | r4 có P1/P2 (từ G0) |
| Giây/ảnh suy luận | `--bench_mode infer`, batch 1 | r4 có P1 |
| Tỉ lệ backbone / encoder / decoder | `--bench_mode stages` | Để thấy phần A/B/C thêm vào decoder |
| Tổng giờ GPU của lượt | Log runner | Ghi vào bảng tổng hợp |

**Quy tắc quyết định:** như v1, thêm một dòng:

| Kết quả | Quyết định |
|---|---|
| **[PERF]** A/B/C làm giây/ảnh suy luận tăng > 10% so với r4 + P1 | Ghi rõ đánh đổi. Cân nhắc bỏ nếu cAcc tăng < 1 điểm |

---

## 8. Thứ tự công việc và điểm dừng

| Bước | Việc | Máy | Điểm dừng |
|---|---|---|---|
| 1 | Sửa trích dẫn PDP/P²IOD (sau khi đồng ý) | — | — |
| 2 | **[PERF]** ~~Tạo worktree~~ Nhánh `feat/attn-prompt` từ `4df908d` trong `AutoCheckout-CL-perf` (đã tạo). Mốc test: 3 lỗi có sẵn như `7630316` (đã chạy 24/11) | CPU | — |
| 3 | Tìm lại `pred_val.npz` r4 và file nhãn. Chạy C0 | CPU | **Dừng 1:** C0 sửa được bao nhiêu |
| 4 | Viết test A-T1..T8 (đỏ), cài A1–A4 (xanh) | CPU | — |
| 5 | **[PERF] G0:** chạy Task 2, 9, 10 của plan perf | GPU | **Dừng 2:** P1/P2 có dùng được cho E không |
| 6 | Chạy E1 | GPU | **Dừng 3:** đọc E1 |
| 7 | Viết test và cài B, C1. Chạy E2, E4 (có thể song song) | CPU → GPU | **Dừng 4:** quyết định E3, E5, E6 |
| 8 | Viết kết quả vào báo cáo kèm trích dẫn và bảng chi phí | — | — |

Bước 4 (code A) làm được song song với bước 5 (G0) vì không phụ thuộc nhau.

---

## 9. Rủi ro

| Rủi ro | Khả năng | Giảm thiểu |
|---|---|---|
| H1 sai: A không giảm nhầm mã | Trung bình | A vẫn bớt 23M tham số. Quyết định theo bảng ở §7 |
| B làm tăng mức quên | Trung bình | Khởi tạo 0, α có cận, đóng băng sau task 1. Đo quên riêng |
| C1: decoder đóng băng không dùng tốt query đề xuất | Trung bình | C1-T2 (K = 0 giống hệt). Dự phòng C2 |
| **[PERF]** G0 chưa chạy, P1/P2 lệch trên GPU | Thấp | Cổng G0. Nếu lệch, chạy E với cờ tắt |
| **[PERF]** 204 file xoá comment | Đã xử lý | Commit `4df908d`, đã kiểm AST và test, đã push |
| **[PERF]** Hai nhánh `feat/dot-gpu-1` và `perf/two-pass-reuse` lệch nhau khi một bên có commit mới | Trung bình | Rebase trước mỗi lượt GPU. Ghi commit vào `run_info` |
| Mất dữ liệu đầu vào (pred r4, nhãn Val-B) | Đã xảy ra | Bước 3 tìm lại. Nếu mất hẳn thì chấm lại r4 bằng P1 (rẻ: suy luận giảm khoảng 49%) |

**Độ phức tạp:**
- A: **Trung bình**.
- B: **Trung bình**.
- C1: **Cao** (thêm loss và matching).
- C0: **Thấp**.
- Tích hợp perf: **Thấp**, vì P1/P2 đã có test và cờ.

---

## 10. Để sau (không thuộc plan này)

- **[PERF]** **Bỏ hẳn lượt 1 khi dùng A:** mỗi lớp decoder chọn prompt từ trạng thái của chính query ngay trong lượt 2.
  - Lợi: bỏ thêm 1 lần encoder + decoder của student khi train (P1 giữ encoder student chạy 2 lần vì dropout).
  - Hại: đổi ngữ nghĩa (prompt phụ thuộc trạng thái đã có prompt), nên cần ablation riêng.
- Loss tương phản với prototype (hướng L): đang chờ người dùng chọn thêm thành E7 hay thay E2.
- bf16, ONNX/TensorRT: ngoài phạm vi, như plan perf.

---

## 11. Nhật ký thực hiện (24/11)

| Mục | Đã làm | Khác plan ở đâu, vì sao |
|---|---|---|
| Comment | `4df908d` (xoá comment, 204 file) đã push lên `origin/perf/two-pass-reuse` | Theo yêu cầu người dùng |
| Nhánh | `feat/attn-prompt` từ `4df908d` trong `AutoCheckout-CL-perf`, plan ở `docs/project/` | Không tạo worktree mới, theo yêu cầu "làm trên nền perf" |
| C0 | **Dời sang phiên GPU G0** | `pred_val.npz` của r4 không còn trên máy. Chấm lại 6000 ảnh val tốn khoảng 5–6 giờ CPU nhưng chỉ vài phút GPU. Đã tìm lại nhãn Val-B (`ket_qua_gpu/_phan_tich/data/relearn/meta/val_b.json`, 2897 ảnh), split Val-A (`.../active/splits/val_a_split.json`), checkpoint r4 (`models/AL-margin-k-r4/task_final.pth`) và ảnh (`data/archive/val2019`, 6000 ảnh) |
| A3 | `two_pass.prompted_outputs` tự đọc `model.config.per_query_prompt` | **Gọn hơn plan:** không phải sửa teacher, validation, `evaluate`, `predict_batch` (hợp đồng chữ ký với `checkout-system`), benchmark. Chỉ sửa `pdp/two_pass.py` và `common_step` |
| **A5 (mới)** | Cho phép chạy per-query từ checkpoint chọn prompt theo ảnh (r3_real, r4): `runtime.validate_runtime` bỏ qua kiểm `local_query`/`per_query_prompt` khi chuyển sang per-query; `engine.resume` bỏ trọng số `query_tf` và ghi log. Chiều ngược lại vẫn báo lỗi | **Plan thiếu bước này:** E1 phải xuất phát từ r3_real như r4, mà checkpoint đó có `local_query=1` và `query_tf`, code cũ từ chối nạp |
| Config | `configs/exp/native/common.sh`: `PER_QUERY_PROMPT=1` bật A và tự đặt `--local_query 0` | Theo quy ước biến môi trường của P1/P2 |
| Test | `tests/test_pdp_a_per_query_prompt.py` (11 test) + 2 test config | A-T7 (IoU query giữa 2 lượt trên checkpoint r4) chưa làm: cần model thật, làm cùng G0 |
| Smoke test A trên r4 thật (25/11) | `models/AL-margin-k-r4/task_final.pth`, bật per-query (bỏ `query_tf`), 10 ảnh Val-A, CPU, có P1: chạy không lỗi, logits hữu hạn, cùng kích thước. **4,38 giây/ảnh so với 4,48 của cách cũ**: A không làm chậm suy luận | Kiểm bằng script tạm `/tmp/a_smoke/run.py` (không đưa vào repo) |
| **A-T7** (25/11) | 65 query tự tin (điểm > 0,3) trên 10 ảnh: IoU giữa box của cùng query ở lượt 1 và lượt 2, **trung vị 0,989, 100% ≥ 0,5** | Giả định "query q nhắm cùng vật ở hai lượt" đúng, nên chọn prompt bằng đầu ra lượt 1 là hợp lệ. Lưu ý: mới đo trên r4 (prompt học theo ảnh); trên model đã train per-query cần đo lại sau E1 |
| Plan Q (`docs/ai/planning/2026-11-24-query-attention-plan.md`) | Người dùng đồng ý gộp: B và C1 giữ cách thí nghiệm rẻ (fine-tune từ r3_real), lấy các phát hiện của plan Q (`--new_params` không mở khóa; two-stage HF có sẵn nhưng cờ chưa vào config; cổng G0-SAM). Q1/Q2 bản đầy đủ (chạy lại FSA) để sau | Plan chi tiết B/C1: `plan-B-C1-2026-11-25.md` |
