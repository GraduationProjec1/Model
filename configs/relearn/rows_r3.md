# rows_r3.json — 66 hàng classifier mở ở vòng 3

Thiết kế: `Finalprpject/docs/design/2026-11-07-vong-3/01-plan.md` (mục 6.3, quyết định nhóm 07/11/2026).

**Quy tắc (chỉ dùng Val-A):** danh sách `e1` (63 SKU, `relearn/variants/e1/train_rows.json`: cặp nhầm ≥ 10 lần trên Val-A của model r1 ∪ 50 SKU của `selection.json`) cộng các SKU **ngoài e1** là đối tác của một cặp nhầm **≥ 8 lần trên Val-A** (pool + calib, 2.895 ảnh) với một SKU đã có trong e1, đo trên model `al_margin_k_B500_r2` (5 ngưỡng nhóm chọn trên calib, NMS 0,45).

| Label | SKU | Cặp nhầm trên Val-A (vòng 2) | Recall / precision Val-A |
|---|---|---|---|
| 40 | 81_drink | ↔ 185 92_alcohol (e1): 11 lần | 0,99 / 0,90 |
| 168 | 144_candy | ↔ 164 119_canned_food (e1): 13 lần | 0,93 / 0,91 |
| 167 | 139_gum | ↔ 68 136_gum (e1): 8 lần | 0,93 / 0,93 |

Đã loại (07/11): 147_candy, 156_seasoner, 126_chocolate, 172_personal_hygiene — từng được đề xuất từ phân tích Val-B; trên Val-A chỉ nhầm ≤ 6 lần, recall và precision ≥ 0,85.

Script tính: `Finalprpject/docs/design/2026-11-07-vong-3/assets/r3_evidence.py` (chạy từ gốc repo).
