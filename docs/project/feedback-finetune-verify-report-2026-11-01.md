# Kết quả kiểm tra plan "cập nhật model" — 01/11/2026

Kiểm trên `origin/feat/task6-relearn` (chỉ đọc) và dữ liệu trên Mac của người dùng. **Không mục nào FAIL.** Dữ liệu train (`_gt_full` và ảnh của hai release) không có trên máy này → V18 BLOCKED; Task 4 và Task 6–7 chờ dữ liệu.

| Mục | Kết quả | Bằng chứng | Ảnh hưởng tới plan |
|---|---|---|---|
| V1 | PASS | `pdp/engine.py:536-540`: mọi head `class_embed[i]`, grad hàng `< PREV_INTRODUCED_CLS` = 0 | Đúng F1 |
| V2 | PASS | `configs/tasks_100-4x25_seed0.json`: task 5 offset 175, 25 class → PREV 175, seen 200 (`engine.py:58-60, 102`) | Chạy lại task 5 không có `--train_rows` thì hàng 0–174 bị khoá |
| V3 | PASS | `common.sh`: `--n_classes 225`; config `num_slots` 224, task 6 = 24 slot dự phòng; `invalid_cls_logits = range(200, 224)`, hàng 224 = nền | Đúng G4 |
| V4 | PASS | `main.py:382`: `prev_ckpt = args.prev_ckpt` khi `task_id == start_task`; `engine.resume` nạp `model` + `class_query_cache/class_prototypes/class_cache_count`, không nạp optimizer | Đúng I1 |
| V5 | PASS | `main.py:389-391`: `train_task_{t}{train_suffix}.json`, val = `val_task_{t}.json` | Task dir cần `val_task_5.json` |
| V6 | PASS | `main.py:240` (`--init_new_prompts`, mặc định 1); `main.py:445` chỉ gọi `init_task_prompts()` khi khác 0; `set_task_id(task_id-1)` | Đúng F3 |
| V7 | PASS | `main.py:439`: không `set_teacher()` khi `--pseudo none`; `engine.common_step`: teacher None → không nhãn giả | Đúng F4 |
| V8 | PASS | Tên tham số từ checkpoint task 5 + `match_name_keywords`: `class_embed` (12) và `prompts.private_*` (18) → `lr`; `prompts.shared_*` + `query_tf` (20), `input_proj`, `bbox_embed`, `level_embed`, `reference_points`, `query_position_embeddings` → `lr_old` | Đúng F5 |
| V9 | PASS, có ghi chú | Chuỗi `EXP-FULL-SYN.sh → full_syn.sh → EXP-B2.sh → common.sh`; argparse lấy giá trị cuối. Khai báo hai lần sẵn có: `--pseudo_dedup_iou` (0 rồi 0,5), `--repo_name` (detector rồi FSA_HF). `common.sh` **đã tự thêm** `--prev_ckpt` khi có biến `PREV_CKPT` | Sửa plan: `EXP-FB-FT.sh` không thêm `--prev_ckpt` lần nữa, chỉ đặt biến `PREV_CKPT` |
| V10 | PASS | `engine.configure_optimizers`: `StepLR(optimizer, lr_drop)` theo epoch; `main.py:89` mặc định 40 | Đúng F6 |
| V11 | PASS (rủi ro xác nhận) | `--weight_decay` mặc định 1e-4; AdamW co tham số dù grad = 0. Ở float32, `1 − lr·wd` = 1.0 với lr 1e-4, 2e-5, 5e-6 nên hàng khoá không đổi; với lr 1e-1 thì đổi | Plan đúng khi yêu cầu chép lại hàng bị khoá sau mỗi bước (Task 1 Step 4) |
| V12 | PASS | `--freeze backbone,encoder,decoder` theo phần tên; `class_embed.*` không bị khoá → grad không None | — |
| V13 | PASS | `run_exp.sh:44-61`: `--start_task t --n_tasks t`; bỏ qua task đã có `task_final.pth` + dự đoán | Run dir mới không cần `task_4` |
| V14 | PASS, có ghi chú | `engine.py` (khoảng 400-460): prototype chỉ cập nhật cho label `≥ PREV_INTRODUCED_CLS` (175–199) | Chạy lại task 5 làm mới prototype của 25 class task 5; class khác giữ nguyên. Ghi vào báo cáo; ảnh hưởng nhỏ tới PPG của mode new sau này |
| V15 | PASS | `full_syn.score_val` đọc `task_dir/val_full.json`; `subset_run(run_dir, ann_path, target, split)` | Đúng I10 |
| V16 | PASS | `compare_r1_vs_t6.py`: 0,4368093922651934 và 0,4469958563535912 | — |
| V17 | PASS | md5 `fe510a4c4dc2c81a328299c4009d31ff` | — |
| V18 | **BLOCKED** | Mac không có `rpc_synth_v2_v1`, `rpc_synth_relearn_v1` (chỉ là symlink tới `/root/AutoCheckout-CL/...` của máy GPU); `AutoCheckout-data-2026-10-11.tar.zst` là bản 11/10, không có ảnh `train2019` hay hai release này. Có `data/archive/val2019` (6.000 ảnh) | Task 4, 6, 7 chờ người dùng lấy dữ liệu từ máy GPU |
| V19 | PASS theo code, dữ liệu BLOCKED | `relearn._relearn_fixture` + `TaskConfig.rpc_to_label_in(6)`: SKU task 6 mang alias 200–224, SKU khác (kể cả SKU task 7) mang label gốc | Đổi alias → gốc bằng `alias_to_label()` là đủ |
| V20 | PASS một phần | Id ảnh v2 = t×10⁶ + n, relearn task 6 bắt đầu 6.000.000, val 1–5.792 → không trùng giữa nguồn; ảnh phản hồi lặp ×3 sẽ trùng id | Vẫn đánh lại id (Task 4) |
| V21 | PASS | `pdp/expand.py`: chép hàng < `learned`, ô mới N(0, init_std), prompt mới 0, từ chối slot sau `learned` đã train | Đúng N1 |
| V22 | PASS | `main.py:430-446`: `resume` → `set_teacher` → `init_task_prompts` | Đúng N4 |
| V23 | PASS | `pdp/ppg.py:64-93`: `gt_iou > 0` bỏ nhãn giả trùng GT | Đúng N3 |
| V24 | PASS | `taskcfg.py:185-190`: RPC id trùng chỉ hợp lệ với `alias_of` | Đúng Task 8 |

## Trả lời phần D

1. F1–F6 làm đúng như bảng. Một chỉnh nhỏ: `--prev_ckpt` đã do `common.sh` thêm qua biến `PREV_CKPT` (V9).
2. Rủi ro đã xác nhận: weight decay (V11, plan đã xử lý); prototype class task 5 bị làm mới (V14, ghi báo cáo).
3. Bị chặn vì dữ liệu: Task 4 (dựng dữ liệu), Task 6–7 (chạy GPU). Task 1–3 làm được.
4. Giờ GPU chưa kiểm được với số ảnh thật (cần V18).
