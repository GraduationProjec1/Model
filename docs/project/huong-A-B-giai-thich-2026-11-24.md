# Hướng A và hướng B: mục đích, ý tưởng, cách làm (24/11/2026)

Tài liệu để **đọc và duyệt ý tưởng** trước khi lên plan triển khai. Chưa có dòng code nào bị sửa.
Bản plan kỹ thuật nháp: `attention-prompt-plan-2026-11-24.md`. Plan đó sẽ được viết lại sau khi bạn duyệt tài liệu này.

---

## 1. Xuất phát điểm

### 1.1 Lỗi của r4 (Val-B, cAcc calib)

| Loại lỗi | Số vật | Ví dụ |
|---|---|---|
| **Nhầm mã** | 566 | Có lon nước, mô hình khoanh đúng lon nhưng gọi sai mã SKU |
| Sót vật | 654 | Có gói mì, mô hình không khoanh |
| Thừa vật | 240 | Một lon bị đếm 2 lần, hoặc khoanh vào nền |

Hai hướng A và B **chỉ nhắm vào nhầm mã**. Sót vật là việc của hướng khác (SAM, DINO, ghép một-nhiều).

### 1.2 Nguồn gốc hai hướng

- A và B **không lấy từ bài báo nào**, kể cả DAMM. Đây là **giả thuyết** rút ra từ việc đọc code PDP và đối chiếu với loại lỗi của r4.
- Kỹ thuật phụ trợ trong B (khởi tạo bằng 0, cổng tanh) là kỹ thuật phổ biến: ControlNet, LoRA, LLaMA-Adapter, Flamingo.
- DAMM chỉ là động lực chung: can thiệp vào cross-attention có thể tăng độ chính xác.
- Trong báo cáo, A và B phải được trình bày là **giả thuyết + kiểm chứng bằng thí nghiệm**.

---

## 2. Nhắc lại: prompt hiện tại hoạt động thế nào

### 2.1 Decoder và hai loại attention

Decoder có **300 query**. Mỗi query là một "ô" sẽ thành một dự đoán (một vật hoặc "không có gì"). Mỗi lớp decoder (6 lớp) làm:

```
① Self-attention : các query trao đổi với nhau  → phối hợp, tránh hai query nhận cùng một vật
② Cross-attention: mỗi query nhìn vào ảnh       → lấy thông tin vật: hình dạng, màu, logo
③ FFN
```

Cross-attention ở đây là **deformable**: mỗi query chỉ lấy mẫu 128 điểm (8 head × 4 mức tỉ lệ × 4 điểm) quanh vị trí của nó. **Vị trí và trọng số các điểm đó tính thẳng từ query** qua 2 lớp Linear. Không có ma trận K/V của ảnh.

### 2.2 Prompt hiện tại (CODA-Prompt chuyển sang DETR)

```
Lượt 1 (không prompt): ảnh → DETR → 300 query đầu ra
          query_tf gộp 300 query → 1 vector x cho CẢ ẢNH                  (~23,0M tham số)

Lượt 2 (có prompt), mỗi lớp decoder:
   Kho thành phần prompt = 100 dùng chung + 1 riêng cho mỗi SKU
   Mỗi thành phần được chấm điểm theo mức "hợp" với x
   Trộn các thành phần theo điểm → 1 bộ prompt (5 token K + 5 token V)
   Nối bộ prompt vào self-attention ①
```

**Hai đặc điểm là gốc của A và B:**

| | Đặc điểm | Hướng |
|---|---|---|
| ① | **Một bộ prompt cho cả ảnh.** Mọi vật trong ảnh dùng cùng hỗn hợp prompt | → **A** |
| ② | **Prompt chỉ ở self-attention.** Phần nhìn ảnh (cross-attention) không nhận prompt | → **B** |

---

## 3. Hướng A: chọn prompt cho từng vật

### 3.1 Vấn đề

CODA-Prompt gốc làm cho **phân loại ảnh**: một ảnh là một lớp, nên một bộ prompt cho cả ảnh là hợp lý. Phát hiện vật thì khác: **một khay có nhiều vật, thuộc nhiều task**.

**Ví dụ:** khay có 3 vật.

| Vật | SKU | Học ở task |
|---|---|---|
| 1 | Lon nước ngọt đỏ | 1 |
| 2 | Gói mì vàng | 3 |
| 3 | Chai nước suối | 5 |

Hiện tại `x` là trung bình có trọng số của cả khay, nên hỗn hợp prompt là **một chút task 1 + một chút task 3 + một chút task 5**. Cả 3 vật nhận **cùng** hỗn hợp này:
- Query của lon nước nhận cả kiến thức về gói mì và chai nước. Đó là nhiễu.
- Kiến thức đúng của lon nước bị pha loãng.

**Giả thuyết H1:** sự pha loãng này góp phần gây nhầm mã, nhất là ở ảnh có SKU của nhiều task.

### 3.2 Ý tưởng

> Mỗi query (mỗi vật) **tự chọn** hỗn hợp prompt dựa trên chính nó, không dựa trên trung bình cả ảnh.

Với ví dụ trên, query của lon nước chọn chủ yếu thành phần prompt của task 1, query của gói mì chọn của task 3, và cứ thế.

### 3.3 Cách làm, ở mức ý tưởng

| Bước | Hiện tại | Hướng A |
|---|---|---|
| "Câu hỏi" để chọn prompt | 1 vector x cho cả ảnh (qua `query_tf`) | **300 vector**, mỗi query dùng chính đầu ra lượt 1 của nó |
| Chấm điểm thành phần | 1 bộ điểm / ảnh | **1 bộ điểm / query** (cùng công thức cosine) |
| Bộ prompt | 1 bộ (5 + 5 token) / ảnh | **1 bộ / query** |
| Gắn vào self-attention | Nối chung vào K/V, mọi query thấy cùng 5 token | Mỗi query thấy **5 token của riêng nó** + 300 query khác |
| `query_tf` + `query_loss` | Cần | **Bỏ** |

**Những gì giữ nguyên:** kho prompt, công thức chấm điểm, việc đóng băng prompt của task cũ, các loss phụ (DDL), backbone/encoder/decoder đóng băng, cách chạy 2 lượt.

**Nơi sửa trong code:**
- `pdp/models/prompt.py`: chọn prompt theo query.
- `pdp/models/modeling_deformable_detr.py`: self-attention với prefix riêng cho từng query.
- `pdp/engine.py` và `pdp/inference.py`: train, teacher và suy luận dùng chung cơ chế mới.

### 3.4 A đem lại gì

| Mặt | Kỳ vọng | Mức chắc chắn |
|---|---|---|
| Nhầm mã | Giảm, rõ nhất ở ảnh nhiều task | Giả thuyết, cần thí nghiệm |
| Tham số | **Giảm khoảng 23,0M** vì bỏ `query_tf` | Chắc chắn (tính từ code) |
| Mức quên | Không tăng | Khá chắc, vì prompt cũ vẫn đóng băng |
| Tốc độ | Gần như không đổi | Khá chắc |
| Siêu tham số | Bớt `λ_query` | Chắc chắn |

**Điểm hay kể cả khi H1 sai:** nếu độ chính xác giữ nguyên, A vẫn là bản rẻ hơn khoảng 23M tham số. Đó cũng là một kết quả có giá trị.

### 3.5 Rủi ro của A

1. **Query chọn sai prompt thì sai càng thêm.** Prompt được chọn theo đầu ra lượt 1. Nếu lượt 1 đã tưởng lon nước là SKU khác, query sẽ chọn prompt của SKU sai và có thể **củng cố lỗi**. Trong cách hiện tại, trung bình cả ảnh làm lỗi này bị "pha loãng".
2. **Query ở lượt 1 và lượt 2 có thể không cùng một vật.** A ngầm giả định query số q ở lượt 1 và lượt 2 nhắm cùng một vật. Giả định này hợp lý (cùng điểm tham chiếu học được, decoder đóng băng) nhưng **chưa được đo**. Có thể kiểm tra bằng IoU giữa box của query q ở hai lượt.
3. **Thừa vật có thể tăng nhẹ.** Hai query cùng nhắm một vật nhưng có prefix khác nhau thì có thể khó "nhường" nhau hơn.

### 3.6 Kiểm chứng A

- **Trước khi train (CPU, dữ liệu r4 có sẵn):** tách số nhầm mã của r4 theo ảnh **1 task** và ảnh **nhiều task**. Nếu ảnh nhiều task không nhầm nhiều hơn rõ rệt, H1 yếu ngay từ đầu.
- **Sau khi train:** nhầm mã giảm **tập trung ở ảnh nhiều task** thì H1 đúng. Giảm đều cả hai loại ảnh thì A có ích nhưng vì lý do khác.

---

## 4. Hướng B: prompt điều khiển việc nhìn ảnh

### 4.1 Vấn đề

Nhầm mã nghĩa là mô hình **khoanh đúng chỗ nhưng nhận sai SKU**. SKU trong cửa hàng thường chỉ khác nhau ở **chi tiết nhỏ**:

| Cặp dễ nhầm | Khác nhau ở |
|---|---|
| Hai vị của cùng một dòng nước ngọt | Màu dải nhãn, chữ vị |
| Hai dung tích của cùng một chai | Chiều cao, chữ "330ml"/"500ml" |
| Hai gói mì cùng hãng | Hình minh họa, màu góc gói |

Muốn phân biệt thì phải **nhìn đúng chỗ**: dải nhãn, góc gói, nắp. Trong decoder, việc "nhìn vào đâu" do **cross-attention** quyết định: 128 điểm lấy mẫu quanh vật, vị trí và trọng số tính từ query.

**Prompt hiện tại không chạm vào phần này.** Nó chỉ đổi query qua self-attention, rồi query đã đổi làm điểm lấy mẫu đổi theo một cách **gián tiếp**. Phần tính vị trí lấy mẫu là trọng số pretrained bị đóng băng, **không biết gì về SKU mới**.

**Giả thuyết H2:** vì prompt không điều khiển trực tiếp được chỗ nhìn, mô hình khó học cách nhìn vào chi tiết phân biệt SKU mới.

### 4.2 Ý tưởng

> Cho prompt của mỗi query **"vặn" nhẹ** vị trí và trọng số các điểm lấy mẫu, để mỗi task học được nhìn vào đâu trên vật.

Không thay cách tính cũ. Chỉ **cộng thêm một lượng hiệu chỉnh nhỏ**:

```
Hiện tại:   vị trí lấy mẫu = điểm tham chiếu + offset(query)
            trọng số       = softmax(w(query))

Hướng B:    vị trí lấy mẫu = điểm tham chiếu + offset(query) + α · Δoffset(prompt)
            trọng số       = softmax(w(query) + α · Δw(prompt))
```

Ví dụ: với nhóm nước ngọt, prompt của task học đẩy vài điểm lấy mẫu lên **dải nhãn giữa lon**, nơi ghi vị. Với nhóm mì gói, prompt đẩy điểm lấy mẫu về **góc có hình minh họa**.

### 4.3 Cách làm, ở mức ý tưởng

| Thành phần | Làm gì | Tại sao |
|---|---|---|
| **Vector prompt cho mỗi query** | Lấy từ bộ prompt mà query đã chọn (trung bình 10 token) | Tận dụng kho prompt CODA sẵn có: kiến thức theo task đã nằm trong đó, không cần kho mới |
| **2 lớp Linear nhỏ mỗi lớp decoder** | Biến vector prompt thành Δoffset và Δw | Đây là "tay vặn" duy nhất được thêm vào |
| **Khởi tạo bằng 0** | Lúc bắt đầu, Δ = 0, mô hình chạy y hệt r4 | Không phá trọng số đã học. Hiệu chỉnh chỉ lớn dần nếu có ích |
| **Hệ số α có giới hạn** (≤ 0,5) | Chặn độ lớn hiệu chỉnh | Không để điểm lấy mẫu chạy ra ngoài vật |
| **Đóng băng 2 lớp Linear sau task 1** | Từ task 2, chỉ prompt mới được học | Chống quên: cách "vặn" cho SKU cũ không bị task mới thay đổi |
| **Đặt trong module `prompts`** | Không đặt trong decoder | Decoder đang đóng băng. Chỉ `prompts` và `class_embed` được train |

**Nơi sửa trong code:**
- `pdp/models/prompt.py`: 2 lớp Linear và α.
- `pdp/models/modeling_deformable_detr.py`: deformable cross-attention nhận Δ.
- `pdp/engine.py`: đóng băng sau task 1, theo cơ chế `SHARED_AFTER_TASK1` đã có.

### 4.4 B đem lại gì

| Mặt | Kỳ vọng | Mức chắc chắn |
|---|---|---|
| Nhầm mã | Giảm, rõ ở các cặp SKU hay nhầm | Giả thuyết, cần thí nghiệm |
| Tham số | **Thêm khoảng 0,59M**, cố định, không tăng theo số SKU | Chắc chắn |
| Mức quên | Không tăng nếu đóng băng sau task 1 | Có rủi ro, cần đo |
| Tốc độ | Gần như không đổi | Khá chắc |
| Sót vật / thừa vật | Gần như không đổi | Không phải mục tiêu |

### 4.5 Rủi ro của B

1. **Quên.** Cross-attention quyết định cách nhìn cho **mọi** SKU. Nếu bỏ bước đóng băng, task mới có thể làm hỏng cách nhìn của SKU cũ.
2. **Đóng băng quá sớm.** 2 lớp Linear chỉ học ở task 1, nên có thể chưa đủ "biết cách vặn" cho SKU ở task sau. Nếu xảy ra, B sẽ cho mức tăng nhỏ ở task 2–5.
3. **Bị A giới hạn.** Nếu không có A, vector prompt vẫn là hỗn hợp của cả ảnh, nên mọi vật bị "vặn" giống nhau. Đây là lý do B nên đi sau A và có cấu hình đối chứng "chỉ B".
4. **Khó giải thích nếu thất bại.** B không tăng có thể do H2 sai, do α quá nhỏ, hoặc do đóng băng quá sớm. Cần log độ lớn hiệu chỉnh để chẩn đoán.

### 4.6 Kiểm chứng B

- So **A + B** với **A**: nhầm mã giảm thêm thì H2 đúng.
- So **chỉ B** với **A + B**: biết B có cần A không.
- Đo mức quên trên SKU task 1–4. Nếu tăng thì B đang đánh đổi.
- Có thể **vẽ điểm lấy mẫu** trước và sau B trên vài cặp SKU hay nhầm. Nếu điểm dịch về vùng phân biệt (nhãn, nắp), đó là bằng chứng trực quan tốt cho báo cáo.

---

## 5. So sánh A, B và DAMM

| | Hướng A | Hướng B | DAMM (tham khảo) |
|---|---|---|---|
| Can thiệp vào | Cách **chọn** prompt | **Cross-attention** (vị trí, trọng số lấy mẫu) | **Cross-attention** (cách tính điểm) + nguồn query |
| Nhắm lỗi | Nhầm mã do pha loãng kiến thức | Nhầm mã do nhìn sai chỗ | Định vị vật nhỏ, che khuất |
| Giữ deformable attention | Có | Có | Phải đổi sang attention thường |
| Train lại decoder | Không | Không | Có |
| Tham số | **−23,0M** | +0,59M | Thêm cả Grounding DINO + SAM |
| Có bài báo chống lưng | Không | Không (dùng kỹ thuật phụ trợ phổ biến) | Có (chính nó) |
| Rủi ro quên | Thấp | Trung bình | Không áp dụng (không phải CL) |
| Chi phí làm | Thấp | Vừa | Cao |

---

## 6. Thứ tự đề xuất

```
Bước 0 (CPU, không tốn GPU)
   Tách nhầm mã r4 theo ảnh 1 task / nhiều task
        │
        ├─ ảnh nhiều task nhầm nhiều hơn rõ  ──► làm A trước
        └─ không khác biệt                    ──► cân nhắc bỏ A, làm thẳng B
                                                  (hoặc vẫn làm A chỉ để giảm tham số)
Bước 1: A  ──► đo
Bước 2: A + B, và chỉ B (đối chứng)  ──► đo
```

---

## 7. Câu hỏi cần bạn quyết định trước khi lên plan thật

1. **Mục tiêu ưu tiên** là giảm nhầm mã (A, B) hay giảm sót vật (hướng SAM / DINO, lỗi lớn hơn về số lượng)?
2. **Ngân sách GPU:** mỗi cấu hình là 1 lượt train đủ 5 task như r4. Cả A, A+B, chỉ B là 3 lượt. Có đủ không?
3. **Lợi ích giảm 23M tham số của A** có được tính là đóng góp riêng trong báo cáo không, kể cả khi độ chính xác chỉ giữ nguyên?
4. Có muốn tôi **tìm tài liệu** xem đã ai làm "prompt theo từng vật" hoặc "prompt trong deformable attention" cho phát hiện vật tăng dần chưa, để biết ý tưởng có mới không và có gì để trích dẫn?
