# Kiểm tra plan "cập nhật model" trước khi code và chạy — cho agent

**Việc của bạn:** kiểm từng mục dưới đây trên code và dữ liệu thật, **không sửa file nào, không train, không dùng GPU**. Plan cần kiểm: `docs/project/feedback-finetune-plan-2026-11-01.md`. Code gốc: nhánh `origin/feat/task6-relearn` (đọc bằng `git show origin/feat/task6-relearn:<file>` hoặc worktree tạm, không checkout đè lên thư mục làm việc).

**Báo cáo:** một bảng `| Mục | Kết quả (PASS / FAIL / BLOCKED) | Bằng chứng (file:dòng, lệnh, số) | Ảnh hưởng tới plan |`, rồi danh sách việc cần sửa trong plan nếu có. FAIL = plan sai so với code; BLOCKED = thiếu dữ liệu hoặc không chạy được ở máy này. Không tự sửa plan.

## A. Mode improve — các chỗ sửa F1–F6 có làm đúng điều plan nói không

| Mục | Kiểm gì | Cách kiểm | Kỳ vọng |
|---|---|---|---|
| V1 | Luật khoá ô hiện nay (F1) | `pdp/engine.py`, `on_after_backward` | Đặt 0 grad mọi hàng `class_embed[i]` < `PREV_INTRODUCED_CLS`, mọi head |
| V2 | Task 5 có PREV / seen là gì | `pdp/engine.py` `__init__` (`task_map`), `TaskConfig` round 1 | PREV = 175, seen = 200 → không có `--train_rows` thì hàng 0–174 bị khoá khi chạy lại task 5 |
| V3 | `n_classes` và logit bị che | `data/training/full-syn-r1/runs/EXP-FULL-SYN/resolved_cli.json`; `invalid_cls_logits` trong `engine.py` | `n_classes` = 225; task 5 che logit 200..223; hàng cuối 224 là nền |
| V4 | `--prev_ckpt` được dùng cho task đầu (F1) | `pdp/main.py` quanh dòng 382 | `prev_ckpt = args.prev_ckpt` khi `task_id == start_task`; `trainer.resume(prev_ckpt)` nạp `model` + bộ nhớ prototype, **không** nạp optimizer |
| V5 | `--train_suffix _fix` đọc đúng file | `pdp/main.py` quanh dòng 389 | Đọc `train_task_5_fix.json`; val đọc `val_task_5.json` |
| V6 | `--init_new_prompts 0` giữ prompt (F3) | `pdp/main.py` quanh dòng 445; `pdp/models/prompt.py` `init_task_prompts` | Với 0 thì không gọi `init_task_prompts()`; `set_task_id(task_id-1)` = 4 → slot đang train là slot của task 5 |
| V7 | `--pseudo none` tắt thầy và nhãn giả (F4) | `pdp/main.py` (`set_teacher`), `engine.common_step` | Không dựng thầy; `old_model_outputs` = None; không gọi `add_pseudo_labels` |
| V8 | `--new_params class_embed,prompts.private` chia lr đúng (F5) | In tên mọi tham số có `requires_grad` của model fixture, áp `match_name_keywords` | `class_embed.*` và `*.prompts.private_*` → `lr`; `*.prompts.query_tf.*`, `*.prompts.shared_*`, `input_proj`, `bbox_embed` → `lr_old` |
| V9 | Cờ sau ghi đè cờ trước | `configs/exp/native/EXP-FULL-SYN.sh` và các file nó `source`; argparse | Các cờ trong `EXP-FB-FT.sh` (`--pseudo`, `--new_params`, `--lr`, `--lr_old`, `--lr_drop`) đứng sau và argparse lấy giá trị cuối; liệt kê cờ nào bị khai báo hai lần |
| V10 | `--lr_drop` (F6) | `engine.configure_optimizers` | `StepLR(optimizer, lr_drop)` bước theo epoch; mặc định 40 |
| V11 | Weight decay với hàng bị khoá (I9) | `pdp/main.py` `--weight_decay` (mặc định 1e-4); `torch.optim.AdamW` | Grad = 0 **không** chặn weight decay; tính `1 − lr·wd` ở float32 với lr 2e-5 và 5e-6: có làm tròn về 1.0 không. Kết luận: plan đúng khi yêu cầu chép lại hàng bị khoá sau mỗi bước (Task 1 Step 4) |
| V12 | Hàng không có grad | `class_embed` có `requires_grad` ở task 5 không (`--freeze` chỉ khoá backbone, encoder, decoder) | `weight.grad` không phải None trong `on_after_backward` |
| V13 | `run_exp.sh` với START_TASK = N_TASKS = 5 trong run dir mới | `scripts/run_exp.sh` | Gọi `main.py --start_task 5 --n_tasks 5`; không cần `task_4` vì có `--prev_ckpt`; task đã có `task_final.pth` + `pred_val.npz` thì bỏ qua |
| V14 | Bộ nhớ prototype khi chạy lại task 5 | `engine.py` (`class_query_cache`, `missing_prototypes`) | Ghi rõ: train lại task 5 cập nhật prototype của class 175–199; có ảnh hưởng gì tới lượt `new` sau này (PPG dùng prototype) |
| V15 | Chấm trên Val-B | `autocheckout/full_syn.py` `score_val`, `subset_run` | `score_val` đọc `task_dir/val_full.json`; `subset_run(..., split='val')` lọc được `pred_val.npz` theo id ảnh |

## B. Số liệu nền

| Mục | Kiểm gì | Cách kiểm | Kỳ vọng |
|---|---|---|---|
| V16 | Script so sánh tái tạo số của gate | `python3 docs/project/feedback-finetune/compare_r1_vs_t6.py` từ thư mục có `training/` và `full_syn_relearn_task6_20261101/` (hoặc sửa đường dẫn trong bản sao tạm) | cAcc 0,4368093922651934 và 0,4469958563535912 |
| V17 | Checkpoint điểm bắt đầu | `md5sum data/training/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth` | `fe510a4c4dc2c81a328299c4009d31ff` |
| V18 | Dữ liệu cho I6, I7 | Tồn tại và đếm ảnh: `rpc_synth_v2_v1/tasks/synth_v2/train_task_{1..5}_gt_full.json`, `rpc_synth_relearn_v1/tasks/synth_v2/train_task_{6,7}_gt_full.json`, thư mục ảnh của hai release, `data/archive/val2019` | Có đủ; thiếu → BLOCKED, ghi rõ file nào |
| V19 | Nhãn trong `_gt_full` của relearn | Đọc `categories` và tập `category_id` của `train_task_6_gt_full.json` | Có label alias 200–224 (cần đổi về gốc) và label gốc của SKU khác; không có label ≥ 250 |
| V20 | Id ảnh có trùng giữa các nguồn không | Tập `images.id` của v2 (task 1–5), relearn (6–7), `val_full.json` | Ghi số id trùng; plan yêu cầu đánh lại id (Task 4) |

## C. Mode new

| Mục | Kiểm gì | Cách kiểm | Kỳ vọng |
|---|---|---|---|
| V21 | `expand.py` dùng được cho class mới | `pdp/expand.py` | Chép hàng 0..learned-1, ô mới N(0, init_std), prompt mới = 0; từ chối checkpoint đã train slot sau `learned` |
| V22 | Khởi tạo prompt cho task mới | `main.py` + `init_task_prompts` | Với `--init_new_prompts 1`, slot của task mới được khởi tạo sau khi nạp trọng số |
| V23 | PPG và loại nhãn giả trùng GT | `engine.add_pseudo_labels`, `ppg.select_pseudo_labels`, `--pseudo_gt_iou` | Nhãn giả của vật cũ trùng box GT của vật mới bị bỏ |
| V24 | `TaskConfig` nhận task class mới không alias | `autocheckout/taskcfg.py` `_validate` | RPC id mới hợp lệ; RPC id trùng mà không có `alias_of` bị từ chối |

## D. Kết luận cần trả lời

1. Sáu chỗ sửa F1–F6 có làm đúng điều bảng "Nguyên nhân và chỗ sửa" nói không? Mục nào sai, sửa plan thế nào.
2. Có rủi ro nào plan chưa tính (ví dụ V11, V14, cờ khai báo hai lần ở V9)?
3. Task nào bị chặn vì dữ liệu (V18)?
4. Ước tính giờ GPU (pilot khoảng 2 giờ, E1 khoảng 8,7 giờ) có hợp lý với số ảnh thật không (cần V18).
