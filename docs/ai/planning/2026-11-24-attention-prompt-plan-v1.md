# Kế hoạch: giảm nhầm mã (hướng A, B) và giảm sót vật (hướng C) cho PDP (24/11/2026)

> **Đã thay bằng v2:** `2026-11-24-attention-prompt-plan-v2.md` (dựng trên nhánh `perf/two-pass-reuse`).

Trạng thái: **chờ duyệt**. Chưa sửa code.
Tài liệu giải thích ý tưởng: `2026-11-24-huong-A-B-giai-thich.md`.

Quyết định của người dùng (24/11):
1. Làm cả hai hướng: nhầm mã và sót vật.
2. Đủ GPU cho 3 lượt train đủ 5 task, thêm lượt nếu cần.
3. Tìm tài liệu: đã xong, xem mục 2.

---

## 0. Sửa trích dẫn (làm ngay, không phụ thuộc plan)

- **arXiv 2510.27316 là P²IOD** ("Parameterized Prompt for Incremental Object Detection", An và cộng sự, CVPR 2026). Đã kiểm tra bằng metadata của `2510.27316v4.pdf`.
- **PDP là arXiv 2603.02286** ("Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection", CVPR 2026, code `zyt95579/PDP_IOD`). Đã kiểm tra bằng `2603.02286v1.pdf`.
- `Report copy.md:32` và `Report copy 2.md:30` đang ghi PDP kèm tên bài và số arXiv của P²IOD. **Cần sửa.** Số tham số prompt từng so với "paper 2510.27316" là so với P²IOD, không phải PDP.

---

## 1. Kết quả Bước 0 (từ `results/error_breakdown_r4`, không tốn GPU)

### 1.1 Nhầm mã: 564 vật

| Chỉ số | Giá trị |
|---|---|
| Nhầm **xuyên task** (SKU thật và SKU đoán thuộc 2 task khác nhau) | **488 / 564 = 86,5%** |
| Mức kỳ vọng nếu nhầm ngẫu nhiên giữa 200 lớp | 76,8% |
| Nhầm cùng nhóm hàng (supercategory) | 41,5% |
| Số vật trung bình mỗi ảnh có nhầm mã | 14,0 |
| Cặp nặng nhất | 98_milk (task 4) ↔ 97_milk (task 5): **64 vụ** |
| 5 cặp tiếp theo | 81_drink ↔ 92_alcohol (23), 117 ↔ 109 canned_food (17), 123 ↔ 122 chocolate (14), 137 ↔ 139 gum (11), 152_seasoner ↔ 65_dessert (10). **Cả 6 cặp đều xuyên task** |
| Chiều nhầm | Về task cũ hơn 268, về task mới hơn 220 |

**Diễn giải:**
- Nhầm xuyên task cao hơn mức ngẫu nhiên khoảng 10 điểm. Ranh giới task là một nguồn gây nhầm, điều này ủng hộ H1.
- Ảnh trung bình có 14 vật, nên một hỗn hợp prompt cho cả ảnh gần như luôn bị trộn nhiều task. Vì vậy phép so "ảnh 1 task với ảnh nhiều task" **không phân biệt được**. Thay vào đó dùng chỉ số **tỉ lệ nhầm xuyên task** để đo hướng A.
- 58,5% vụ nhầm là **giữa các nhóm hàng khác nhau**, không thuần do bao bì giống nhau. Hướng B (nhìn chi tiết) chỉ nhắm trúng phần cùng nhóm hàng.

### 1.2 Sót vật: 628 vật, chia làm hai loại khác hẳn nhau

| Loại | Số vật | Nghĩa | Hướng sửa |
|---|---|---|---|
| `miss_threshold` | 246 | **Có** box đúng SKU, nhưng điểm dưới ngưỡng | Chính sách đếm (C0) |
| `miss_nms` | 85 | **Có** box đúng SKU, bị NMS xóa vì nằm sát vật khác | Chính sách đếm (C0) |
| `miss` | 297 | **Không** có box nào đúng SKU trên vật | Cấu trúc query (C1, C2) |

Hơn một nửa số vật sót (331) **mô hình đã thấy**, chỉ bị bước lọc loại. Hướng sót vật vì vậy tách thành C0 (rẻ, chỉ CPU) và C1/C2 (cần train).

---

## 2. Tài liệu: ý tưởng nào đã có, ý tưởng nào mới

| Hướng | Công trình liên quan | Đánh giá độ mới |
|---|---|---|
| **A**: prompt theo từng query | **P²IOD** (2510.27316) chỉ ra "prompts pool confusion": ảnh có đối tượng của nhiều task thì chọn prompt kiểu L2P/CODA không hợp, nhưng P²IOD vẫn dùng prompt theo ảnh. **MD-DETR** (ECCV'24): gộp 300 proposal thành 1 query cho cả ảnh. **iDPA** (ICML'25): "instance-level" nhưng thực ra là prompt theo task. **DyQ-DETR**: query riêng theo task, không phải prompt | **Gần giống, chưa thấy ai làm đúng.** Động cơ đã có trong P²IOD, nên A là bước mở rộng tự nhiên có trích dẫn chống lưng |
| **B**: prompt điều biến deformable attention | P²IOD viết: *"there is currently no research integrating prompts into deformable attention mechanisms"*. **Stable-SAM** (2311.15776): plugin học dịch vị trí lấy mẫu trên SAM đóng băng (0,08M tham số). LoRA trên offset của Mask DINO (MAKE 2024, **chưa kiểm chứng**). Cổng khởi tạo 0: ControlNet, LLaMA-Adapter | **Chưa thấy cho IOD.** Có câu trích từ P²IOD xác nhận khoảng trống. Adapter offset nói chung thì đã có |
| **C**: query/điểm tham chiếu từ proposal | **DAMM** (2508.04868): query từ mask SAM, +3,1 AP (Cityscapes). **HQP-DETR** (2511.08018): anchor/query từ SAM automatic mask. **Efficient DETR**, **DINO** (mixed query selection), **DDQ** (93,8 AP CrowdHuman): khởi tạo từ dense prior hoặc encoder | **Đã có người làm.** Điểm mới chỉ là áp vào IOD dựa trên prompt, backbone đóng băng, khay checkout dày đặc |

Lưu ý của agent tìm tài liệu: MD-DETR ghi bộ nhớ được đưa vào "deformable attention module of the decoder", còn P²IOD nói prompt chỉ ở self-attention. Code PDP của dự án xác nhận prompt **chỉ ở self-attention** (`modeling_deformable_detr.py:783–786`).

**Danh sách trích dẫn:** PDP, P²IOD, MD-DETR, iDPA, DyQ-DETR, Deformable DETR, Stable-SAM, ControlNet, LLaMA-Adapter, DAMM, HQP-DETR, Efficient DETR, DINO, RPC.

---

## 3. Hướng A: chọn prompt theo từng query

**Mục đích:** mỗi vật dùng hỗn hợp prompt của riêng nó, để giảm **nhầm xuyên task**.

**Cơ sở:**
- 86,5% nhầm mã là xuyên task, cao hơn mức ngẫu nhiên 76,8%.
- Ảnh trung bình có 14 vật.
- P²IOD đã nêu vấn đề "prompts pool confusion".

| # | Sửa gì | Như thế nào | Tại sao |
|---|---|---|---|
| A1 | `pdp/models/prompt.py` | Nhánh `per_query`: nhận x (B,300,256), không qua `query_tf`. `w = cos(x ⊙ A, K)` tính cho từng query bằng cách tách đại số (`x @ (A⊙K̂)ᵀ / sqrt(x² @ (A²)ᵀ)`), không tạo tensor (B,300,K,256). `P_ = w @ P` cho ra Ek, Ev có kích thước (B,300,5,256). Khi bật nhánh này thì không tạo `query_tf` | Thay đổi tối thiểu: kho prompt, công thức, cơ chế đóng băng và loss DDL giữ nguyên, chỉ đổi "ai chọn prompt" |
| A2 | `modeling_deformable_detr.py`, self-attn (`:783–786`) | Nếu `pk` có 4 chiều: `s_prefix = einsum(Q, Ek_q)` (B,H,300,5), ghép với `QKᵀ` (B,H,300,300), softmax, `out = attn[:5]·Ev_q + attn[5:]@V`. Nếu `pk` có 3 chiều thì chạy code cũ | Prefix chung không dùng được khi mỗi query có prefix khác nhau. Khi mọi prefix giống nhau, cách này tương đương toán học với cách cũ |
| A3 | `engine.py:182` (teacher), `:391–417` (train), `:514–517` (loss), `inference.py:43–45` | Truyền `last_hidden_state` (B,300,256) thẳng vào. Bỏ `query_tf` và `query_loss` | Train, teacher và suy luận phải dùng cùng cơ chế |
| A4 | `main.py`, `engine.py:84–92`, `configs/exp/` | Cờ `--per_query_prompt`. Báo lỗi nếu bật cùng `--local_query 1`. Thêm `ATT-A.sh` | Bật/tắt cho ablation |

**Test (viết trước, chạy trên CPU):**
- A-T1: cờ tắt thì đầu ra giống hệt code cũ.
- A-T2: 300 query giống nhau thì `w` và đầu ra self-attn trùng với bản theo ảnh.
- A-T3: kích thước và bộ nhớ đúng như thiết kế.
- A-T4: private của task cũ không có gradient.
- A-T5: số tham số giảm đúng 23.040.300.
- A-T6: bật cùng `--local_query` thì báo lỗi.

**Đem lại:**
- Kỳ vọng giảm nhầm xuyên task. Đây là giả thuyết.
- **Bớt 23,0M tham số** (chắc chắn).
- Mức quên không tăng.
- Bớt siêu tham số `λ_query`.

**Rủi ro:**
- Query nhận sai ngay ở lượt 1 sẽ chọn sai prompt, củng cố lỗi.
- Query số q ở lượt 1 và lượt 2 phải nhắm cùng một vật. Cần đo IoU giữa box của hai lượt (A-T7, chạy trên checkpoint r4).

---

## 4. Hướng B: prompt điều biến deformable cross-attention

**Mục đích:** prompt hiệu chỉnh vị trí và trọng số lấy mẫu, để task học nhìn vào chi tiết phân biệt SKU. Nhắm vào **phần nhầm mã cùng nhóm hàng (41,5%)** và các cặp như 98_milk ↔ 97_milk.

| # | Sửa gì | Như thế nào | Tại sao |
|---|---|---|---|
| B1 | `pdp/models/prompt.py` | Thêm `xattn_delta_off_{l}`: Linear(256→256) và `xattn_delta_att_{l}`: Linear(256→128) cho 6 lớp, **khởi tạo bằng 0**. `α_l = 0,5·tanh(β_l)`. `p_q = mean(Ek_q, Ev_q)` | Đặt trong `prompts` vì decoder đóng băng và chỉ `prompts`, `class_embed` được train. Khởi tạo 0 nên lúc đầu mô hình y hệt r4 (cách làm của ControlNet, LLaMA-Adapter) |
| B2 | `MultiscaleDeformableAttention.forward` (`:677–684`), decoder layer, decoder loop (`:1400`) | `offset += α·Δoff`, `trọng số = softmax(w(h) + α·Δatt)`. Nếu `xattn_delta=None` thì chạy code cũ. Encoder không nhận delta | Deformable attention không có K/V. Offset và trọng số là hai nút vặn duy nhất (giống cách làm của Stable-SAM) |
| B3 | `engine.py:61, :718` | Thêm `xattn_delta` vào nhóm đóng băng sau task 1 (cờ `--freeze_xattn_after_task1`, mặc định 1) | Chống quên: từ task 2, tính riêng theo task chỉ đi qua prompt |
| B4 | Cờ, log, config | `--prompt_cross_attn`, `--xattn_alpha_max`. Log `mean|α·Δoff|` theo lớp. Thêm `ATT-AB.sh`, `ATT-B.sh` | Ablation và chẩn đoán |

**Test:**
- B-T1: khi Δ = 0, đầu ra giống hệt lúc tắt B.
- B-T2: `|α| ≤ α_max`.
- B-T3: từ task 2, `xattn_delta` không có gradient.
- B-T4: encoder không đổi.
- B-T5: ở task 1, gradient đến được `xattn_delta`.
- B-T6: tham số thêm khoảng 0,59M, không phụ thuộc số SKU.

**Đem lại:**
- Kỳ vọng giảm nhầm mã cùng nhóm hàng. Đây là giả thuyết.
- +0,59M tham số cố định.
- Có thể vẽ điểm lấy mẫu trước và sau trên cặp sữa 97/98 để minh họa trong báo cáo.

**Rủi ro:**
- Mức quên tăng nếu không đóng băng.
- Đóng băng sau task 1 có thể làm B kém tác dụng ở task 2–5.

---

## 5. Hướng C: giảm sót vật

### C0. Chính sách đếm: 331 vật sót do ngưỡng/NMS (chỉ CPU, làm trước)

| # | Sửa gì | Như thế nào | Tại sao |
|---|---|---|---|
| C0.1 | NMS khi đếm | Thử ngưỡng IoU NMS cao hơn 0,45, và NMS **theo lớp** thay cho NMS không phân biệt lớp | 85 vật bị NMS xóa vì sát vật khác, mà khay dày đặc thì các vật hay chồng nhau |
| C0.2 | Ngưỡng điểm | Thử ngưỡng theo SKU (có co về ngưỡng của nhóm task để tránh quá khớp), chọn trên **Val-A-calib** | 246 vật có box đúng nhưng điểm thấp. Hiện chỉ có 4 ngưỡng cho 4 nhóm task |
| C0.3 | Đo | Chạy lại `tools/error_breakdown.py` để xem `miss_*` giảm bao nhiêu và `extra_*` tăng bao nhiêu | Hạ ngưỡng thì vật thừa tăng. Phải xem tổng cAcc |

- **Điều kiện:** cần tìm lại `pred_val.npz` của r4 và file nhãn Val-B/Val-A-calib. Các đường dẫn trong `error_breakdown_r4/error_breakdown.json` đã không còn ở máy.
- **Đem lại:** không tốn GPU và áp được ngay cho mọi mô hình, kể cả mô hình sau khi làm A, B, C1.
- **Giới hạn:** chỉ tác động lên 331 vật. Có rủi ro quá khớp nếu chọn trên tập nhỏ.

### C1. Khởi tạo query từ đề xuất của encoder (rẻ, không cần mô hình ngoài)

| Sửa gì | Như thế nào | Tại sao |
|---|---|---|
| Thêm **K query đề xuất** (ví dụ K = 100) cạnh 300 query học được | Chấm điểm "có vật" cho mỗi vị trí trên đầu ra encoder (đầu `enc_output` đã có trong code two-stage, `modeling_deformable_detr.py:1636–1641`). Lấy top-K làm điểm tham chiếu, rồi một MLP nhỏ sinh query_pos từ box đề xuất | Cơ chế của two-stage Deformable DETR / DINO / Efficient DETR: query bắt đầu **đúng chỗ có vật**. Vật nằm trong vùng đông được "neo" sẵn một query |
| Đặt module mới trong `prompts` (hoặc thêm vào `new_params`) | Hungarian matching trên 300 + K query | Decoder vẫn đóng băng. Chỉ phần đề xuất được train |
| Prompt cho K query mới | Dùng hướng A (theo từng query) | Query mới cũng cần prompt đúng task |

- **Tham số:** thêm khoảng 0,3M.
- **Tốc độ:** gần như không đổi.
- **Rủi ro:** decoder đóng băng chưa từng thấy loại query này, nên có thể cần train thêm phần query_pos.

### C2. Khởi tạo query từ mask SAM (chỉ làm nếu C1 không đủ)

- **Như C1**, nhưng đề xuất lấy từ **SAM 2.1 automatic mask** (repo có sẵn `sam2.1_b.pt`). Box của mask cho điểm tham chiếu. Theo DAMM và HQP-DETR.
- **Tính trước offline** đề xuất SAM cho mọi ảnh train/val rồi lưu cache, để không chạy SAM trong vòng lặp train.
- **Đem lại:** đề xuất độc lập với detector, có thể bắt vật mà encoder bỏ sót. DAMM: +3,1 AP khi thêm query SAM (miền giao thông).
- **Chi phí:** suy luận mỗi ảnh phải chạy thêm SAM, cần đo FPS. Đây là trở ngại lớn cho quầy thanh toán thật.

---

## 6. Ma trận thí nghiệm

**Điều kiện công bằng:** cùng điểm xuất phát với r4 (r3_real), cùng seed, cùng dữ liệu và lịch học, chỉ khác cờ. Chọn cấu hình trên Val-B / Val-A-calib, **không dùng test/holdout**.

| Mã | Cấu hình | GPU | Ưu tiên | Trả lời câu hỏi |
|---|---|---|---|---|
| E0 | r4 (có sẵn) | — | — | Số gốc |
| C0 | r4 + chính sách đếm mới | Không | 1 | Sửa được bao nhiêu trong 331 vật sót do lọc |
| **E1** | A | 1 lượt | 2 | H1: prompt theo query có giảm nhầm xuyên task không. Tham số −23M |
| **E2** | A + B | 1 lượt | 3 | H2: cross-attn có giảm thêm nhầm cùng nhóm hàng không |
| **E4** | A + C1 | 1 lượt | 3 | Query đề xuất có giảm 297 vật sót thật không |
| E3 | chỉ B | 1 lượt | Nếu cần | B có cần A không |
| E5 | A + B + C1 | 1 lượt | Nếu E2, E4 đều có ích | Hai hướng có cộng dồn không |
| E6 | A + C2 | 1 lượt | Nếu C1 không đủ | SAM có hơn đề xuất của encoder không |

**3 lượt GPU cốt lõi: E1, E2, E4.** E3, E5, E6 là lượt thêm, chỉ chạy khi kết quả trước cho thấy cần.
Lưu ý: tôi đã đổi từ "A, A+B, chỉ B" ban đầu sang "A, A+B, A+C1", vì người dùng chọn làm cả hướng sót vật. "Chỉ B" chuyển xuống mục "nếu cần".

**Báo cáo cho mỗi cấu hình:**
1. cAcc calib Val-B (quy ước chính thức qua `al_score`).
2. `swap` (tách xuyên task / cùng task, cùng nhóm hàng / khác nhóm hàng), `miss_threshold`, `miss_nms`, `miss`, `extra_*`.
3. Riêng 6 cặp SKU nặng nhất ở mục 1.1.
4. Mức quên: cAcc/AP của SKU task 1–4 ở cuối task 5.
5. Tham số train được (theo nhóm) và thời gian suy luận mỗi ảnh.

**Quy tắc quyết định:**

| Kết quả | Quyết định |
|---|---|
| E1 giảm nhầm xuyên task, cAcc không giảm | Giữ A, làm nền cho E2 và E4 |
| E1 không đổi nhầm mã, cAcc không giảm | Vẫn giữ A (bớt 23M tham số). H1 sai |
| E1 làm cAcc giảm | Bỏ A. E2 và E4 chạy trên nền r4. Thêm E3 |
| E2 giảm thêm nhầm cùng nhóm hàng, quên không tăng | Giữ B. Đây là đóng góp mới (P²IOD xác nhận chưa ai làm) |
| E2 giảm nhầm nhưng quên tăng | Thử α_max = 0,25. Báo cáo đánh đổi |
| E4 giảm `miss` | Giữ C1. Cân nhắc E5 |
| E4 không giảm `miss` | Cân nhắc E6 (SAM) nếu chi phí suy luận chấp nhận được |

---

## 7. Thứ tự công việc và điểm dừng

1. **Sửa trích dẫn PDP/P²IOD** trong `Report copy*.md`, sau khi người dùng đồng ý.
2. **Tìm lại** `pred_val.npz` của r4 và file nhãn Val-B/Val-A-calib. Chạy **C0** (CPU).
   **Điểm dừng 1:** xem C0 sửa được bao nhiêu vật sót.
3. **A:** viết test A-T1..T7 (đỏ), cài A1–A4 (xanh), chạy toàn bộ `tests/`. Chạy **E1**.
   **Điểm dừng 2:** đọc E1 theo bảng quyết định.
4. **B** và **C1:** viết test, cài code, chạy **E2** và **E4** (có thể chạy song song).
   **Điểm dừng 3:** quyết định E3, E5, E6.
5. Viết kết quả vào báo cáo kèm danh sách trích dẫn ở mục 2.
