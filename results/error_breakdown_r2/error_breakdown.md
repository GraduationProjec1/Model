# Phân rã lỗi: 1280 / 2897 ảnh sai (cAcc 0.5582)

| Loại lỗi | Số vật | Ảnh chỉ có loại này |
|---|---|---|
| swap | 848 | 439 |
| miss_nms | 119 | 39 |
| miss_threshold | 241 | 126 |
| miss | 341 | 104 |
| extra_dup | 6 | 1 |
| extra_shift | 346 | 107 |
| extra_bg | 225 | 78 |

| Nhóm lỗi | Ảnh chỉ có nhóm này |
|---|---|
| swap (swap) | 439 |
| miss_or_extra_policy (extra_dup, miss_nms, miss_threshold) | 172 |
| miss (miss) | 104 |
| extra_other (extra_bg, extra_shift) | 201 |

| Số lỗi trong ảnh sai | Ảnh |
|---|---|
| 1 | 773 |
| 2 | 311 |
| 3 | 111 |
| 4+ | 85 |

| Độ khó | Ảnh | Sai | cAcc |
|---|---|---|---|
| easy | 953 | 269 | 0.7177 |
| hard | 981 | 569 | 0.4200 |
| medium | 963 | 442 | 0.5410 |

Ảnh sai có dính 17 SKU học lại: 325; ảnh sai chỉ do 17 SKU: 113; tỷ lệ lỗi dính 17 SKU: 18.53%

| SKU | Trong 17 SKU | Lỗi | Ảnh chỉ sai vì SKU này | swap | miss_nms | miss_threshold | miss | extra_dup | extra_shift | extra_bg |
|---|---|---|---|---|---|---|---|---|---|---|
| 162 98_milk | có | 133 | 57 | 118 | 0 | 5 | 2 | 0 | 1 | 7 |
| 141 141_gum |  | 164 | 51 | 12 | 0 | 3 | 5 | 0 | 29 | 115 |
| 186 97_milk | có | 119 | 50 | 113 | 0 | 1 | 2 | 0 | 3 | 0 |
| 164 119_canned_food |  | 57 | 25 | 30 | 0 | 11 | 16 | 0 | 0 | 0 |
| 167 139_gum |  | 48 | 24 | 36 | 0 | 4 | 0 | 0 | 7 | 1 |
| 68 136_gum |  | 55 | 23 | 8 | 0 | 0 | 7 | 0 | 2 | 38 |
| 153 35_instant_drink | có | 43 | 23 | 24 | 0 | 12 | 1 | 0 | 3 | 3 |
| 185 92_alcohol |  | 47 | 22 | 44 | 0 | 1 | 0 | 0 | 2 | 0 |
| 118 140_gum |  | 48 | 21 | 24 | 0 | 6 | 10 | 0 | 1 | 7 |
| 188 112_canned_food | có | 37 | 19 | 26 | 0 | 1 | 4 | 0 | 6 | 0 |
| 191 151_candy |  | 39 | 17 | 29 | 2 | 0 | 4 | 1 | 3 | 0 |
| 40 81_drink |  | 33 | 17 | 31 | 0 | 1 | 0 | 0 | 1 | 0 |
| 166 130_chocolate |  | 24 | 16 | 4 | 3 | 4 | 11 | 1 | 1 | 0 |
| 69 137_gum |  | 35 | 15 | 29 | 0 | 0 | 2 | 0 | 4 | 0 |
| 190 135_gum |  | 29 | 15 | 19 | 0 | 1 | 2 | 0 | 4 | 3 |
| 77 156_seasoner |  | 28 | 15 | 17 | 3 | 1 | 7 | 0 | 0 | 0 |
| 165 126_chocolate |  | 27 | 15 | 15 | 0 | 3 | 5 | 0 | 3 | 1 |
| 98 198_stationery |  | 29 | 14 | 7 | 2 | 1 | 9 | 0 | 9 | 1 |
| 168 144_candy |  | 43 | 13 | 18 | 0 | 10 | 5 | 0 | 5 | 5 |
| 85 172_personal_hygiene |  | 20 | 13 | 12 | 4 | 1 | 3 | 0 | 0 | 0 |

| Cặp nhầm | Trong 17 SKU | Lần nhầm | Ảnh chỉ sai vì cặp này |
|---|---|---|---|
| 162 98_milk ↔ 186 97_milk | có | 100 | 44 |
| 40 81_drink ↔ 185 92_alcohol |  | 30 | 15 |
| 69 137_gum ↔ 167 139_gum |  | 20 | 10 |
| 61 123_chocolate ↔ 140 122_chocolate |  | 17 | 8 |
| 59 117_canned_food ↔ 163 109_canned_food |  | 17 | 7 |
| 18 36_instant_drink ↔ 129 34_instant_drink | có | 12 | 4 |
| 17 33_instant_drink ↔ 153 35_instant_drink | có | 11 | 6 |
| 77 156_seasoner ↔ 191 151_candy |  | 11 | 7 |
| 60 121_canned_food ↔ 188 112_canned_food | có | 10 | 4 |
| 164 119_canned_food ↔ 168 144_candy |  | 10 | 4 |
| 11 23_dried_food ↔ 101 17_dried_fruit |  | 10 | 7 |
| 109 73_drink ↔ 133 74_drink | có | 10 | 4 |
| 33 67_dessert ↔ 143 159_seasoner |  | 9 | 5 |
| 118 140_gum ↔ 190 135_gum |  | 9 | 5 |
| 70 138_gum ↔ 162 98_milk | có | 8 | 1 |
| 182 64_dessert ↔ 191 151_candy |  | 8 | 3 |
| 79 160_seasoner ↔ 164 119_canned_food |  | 8 | 4 |
| 84 171_personal_hygiene ↔ 85 172_personal_hygiene |  | 8 | 6 |
| 167 139_gum ↔ 195 167_personal_hygiene |  | 8 | 2 |
| 41 83_drink ↔ 183 82_drink |  | 8 | 2 |
