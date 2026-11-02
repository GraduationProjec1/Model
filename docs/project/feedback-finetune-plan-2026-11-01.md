# Cập nhật model: cải thiện vật nhận sai (improve) và học vật mới (new) — kế hoạch cho agent

> **Trạng thái 01/11:** verify xong (không FAIL, xem `feedback-finetune-verify-report-2026-11-01.md`); Task 0–5 xong trên nhánh `feat/update-model` (cờ `--train_rows`, `tools/sku_regression.py`, `autocheckout/feedback_fix.py`, `autocheckout/update_model.py`, `EXP-FB-FT.sh`, notebook 08; dry run notebook chạy hết). Còn lại: Task 4 Step 3 (dựng dữ liệu thật, cần `_gt_full` từ máy GPU), Task 6–7 (GPU), Phần 2.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Trước khi code: chạy hết file kiểm tra `feedback-finetune-verify-2026-11-01.md` và báo kết quả cho người dùng.** Đọc hết "Nguyên nhân và chỗ sửa", "Quyết định", "Global Constraints" trước khi làm task nào.

**Goal:** Một điểm vào `autocheckout/update_model.py` với hai chế độ dùng chung code train, chấm và cổng:
- `--mode improve`: sửa các vật model đang nhận sai bằng cách train tiếp **trọng số cũ** của các cặp SKU hay nhầm (thay cho relearn task 6 bằng alias, đã `stop` ở cổng với +1,02 điểm cAcc-M2).
- `--mode new`: học sản phẩm chưa có trong model bằng **ô mới**, giữ đúng cách continual learning của PDP.

**Thứ tự:** Phần 1 (improve) làm và chạy trước. Phần 2 (new) code sau khi improve có kết quả; chạy thật khi có dữ liệu sản phẩm mới.

**Tech Stack:** như nhánh `feat/task6-relearn` (Python 3.10, PyTorch 2.2.2, PDP Deformable DETR + ConvNeXt-V2, pycocotools, scipy 1.10.1), env `pdp`.

## Nguyên nhân lỗi task 6 và chỗ sửa (spec của mode improve)

| # | Nguyên nhân lỗi task 6 | Chỗ sửa | Bằng chứng / cách kiểm |
|---|---|---|---|
| F1 | Tạo ô mới, không sửa ô cũ | `--train_rows`, chạy lại task 5 từ checkpoint task 5 (`--prev_ckpt`, `--train_suffix _fix`); không `expand.py`, không alias | Hàng `class_embed` 0–199 của checkpoint relearn: Δ = 0 so với round 1 |
| F2 | Ô SKU anh em bị khoá, không phân biệt được | `--train_rows` gồm **cả hai phía** của mỗi cặp nhầm | 122→123_chocolate 17→36; vật SKU cũ bị gọi thành SKU task 6: 292→530 |
| F3 | Học lại từ đầu (ô và prompt mới ngẫu nhiên) | `--init_new_prompts 0`, giữ trọng số cũ | `expand.py`: ô mới N(0, 0,02); `init_task_prompts()` ép prompt mới vuông góc với prompt cũ |
| F4 | Nhãn giả mang lỗi của model cũ | `--pseudo none` + nhãn thật (`_gt_full`, ảnh phản hồi) | Nhánh prototype của PPG task 6: precision 0,468 |
| F5 | Phần dùng chung bị kéo lệch | `--new_params class_embed,prompts.private` để `query_tf` và prompt dùng chung xuống `lr_old`; thêm replay 30% | `query_tf` đổi 22,8% (lr 1e-4); 150 SKU khác 95,5% → 94,9% vật đúng |
| F6 | Learning rate cao, không giảm | `--lr 2e-5 --lr_old 5e-6` + `--lr_drop` | `--lr_drop` mặc định 40 > 16 epoch |

Bằng chứng tái tạo được bằng `docs/project/feedback-finetune/compare_r1_vs_t6.py` (ra đúng cAcc 0,4368 / 0,4470).

## Quyết định (spec)

Giá trị mặc định đủ để chạy; người dùng đổi thì cập nhật bảng trong cùng commit.

### Chung

| # | Quyết định | Giá trị |
|---|---|---|
| G1 | Điểm vào | `python -m autocheckout.update_model --mode {improve,new} --settings <json>`; mỗi mode là một chuỗi `stage_*` ghi `stages/<stage>.json`, bước xong thì bỏ qua, `GateStop` khi cổng dừng (khuôn `autocheckout/relearn.py`) |
| G2 | Chung giữa hai mode | nạp checkpoint, `full_syn._train` qua `scripts/run_exp.sh`, chấm `full_syn.score_val`, `tools/sku_regression.py`, ghi cổng, `stage_final_test` |
| G3 | Một đợt có cả vật mới và vật sai | Chạy **lần lượt**: `new` → qua cổng → `improve` (có thể đưa ô của vật mới vào `--train_rows` nếu nó hay nhầm với vật cũ). Không gộp vào một lần train |
| G4 | `n_classes` | 225 như round 1 (200 class đã học + slot dự phòng + nền). Mode `new` mở rộng khi cần thêm slot |

### Mode improve

| # | Quyết định | Giá trị |
|---|---|---|
| I1 | Điểm bắt đầu | `data/training/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth`, md5 `fe510a4c4dc2c81a328299c4009d31ff`; FSA của round 1 được link như `relearn.stage_expand` |
| I2 | Chia val | `rpc_eval_official_v1/val_full.json` (5.792 ảnh) chia 50/50 theo nhóm cảnh (`autocheckout.groups.coco_group_keys`), seed 20261101, phân tầng theo `level`. Val-A = luồng phản hồi, Val-B = chấm cổng |
| I3 | Ảnh phản hồi | Ảnh Val-A model round 1 đếm sai (policy stage 5 của `calibration_count_m2.json` round 1), kèm toàn bộ nhãn thật; lặp ×3 |
| I4 | Cặp nhầm | Từ Val-A, ghép vật–box IoU ≥ 0,5, cộng hai chiều. Pilot: 10 cặp nhiều nhất. E1: mọi cặp ≥ 10 lần ∪ 50 SKU của `configs/relearn/selection.json` |
| I5 | `--train_rows` | Mọi SKU trong cặp (F1, F2). E2 (chỉ khi E1 trượt và người dùng đồng ý): `all` |
| I6 | Ảnh ghép nhãn đầy đủ | Ảnh có ≥ 1 vật thuộc `train_rows` từ `rpc_synth_relearn_v1/tasks/synth_v2/train_task_{6,7}_gt_full.json` (alias → label gốc bằng `TaskConfig.alias_to_label()`) và `rpc_synth_v2_v1/tasks/synth_v2/train_task_{1..5}_gt_full.json` |
| I7 | Replay (F5) | `rpc_synth_v2_v1` `train_task_{1..5}_gt_full.json`, ảnh không chọn ở I6, 30% số ảnh file train; mỗi SKU ≥ 100 vật rồi bù ngẫu nhiên (seed 20261101) |
| I8 | Cờ train | `--pseudo none --init_new_prompts 0 --new_params class_embed,prompts.private --lr 2e-5 --lr_old 5e-6`, giữ `--freeze backbone,encoder,decoder`, 640 px, batch 8. Pilot 4 epoch `--lr_drop 3`; E1 6 epoch `--lr_drop 4`; vòng 2: 2 epoch từ model E1, `--lr 1e-5 --lr_old 2.5e-6` |
| I9 | Hàng bị khoá và weight decay | Hàng ngoài `train_rows` phải giữ nguyên **từng bit** sau train. AdamW `--weight_decay 1e-4` vẫn áp lên tham số có grad = 0; hiện chỉ không đổi nhờ `1 − lr·wd` làm tròn về 1.0 ở float32. Code phải đảm bảo bằng cách khác (xem Task 1 Step 4), không dựa vào làm tròn |
| I10 | Chấm | `score_val` trên task dir có `val_full.json` = Val-B. Mốc = round 1 lọc về Val-B (như `full_syn.subset_run`, `split='val'`). "SKU khác" = SKU không thuộc `train_rows` |

**Cổng mode improve** (so với mốc trên Val-B; `fixed`/`broken` = ảnh sai→đúng / đúng→sai; sign test một phía `scipy.stats.binom.sf(fixed - 1, fixed + broken, 0.5)`):

| Điều kiện | Pilot | E1 | Vòng 2 (so với E1) |
|---|---|---|---|
| cAcc-M2 Val-B | > mốc | ≥ mốc + 0,0165 | ≥ E1 |
| Sign test | p < 0,05 | p < 0,05 | — |
| Tổng lần nhầm của các cặp (hai chiều) | giảm ≥ 30% | giảm ≥ 30% | không tăng |
| AP50 SKU khác | giảm ≤ 0,01 | giảm ≤ 0,01 | giảm ≤ 0,005 |
| SKU (≥ 30 vật) tụt > 0,02 tỉ lệ vật đúng | ≤ 3 | 0 | 0 |

### Mode new

| # | Quyết định | Giá trị |
|---|---|---|
| N1 | Ô mới | Thêm task mới vào task config (class mới, label nối tiếp, **không** `alias_of`) bằng `tools/make_new_task_config.py`; mở rộng checkpoint bằng `pdp/expand.py` có sẵn khi `n_classes` không đủ slot |
| N2 | Ô được train | Mặc định của PDP: chỉ ô của task mới (không truyền `--train_rows`) |
| N3 | Nhãn vật cũ trong ảnh | PPG của thầy như round 1 (`--pseudo_gt_iou 0.5 --pseudo_dedup_iou 0.5`) |
| N4 | Prompt | `--init_new_prompts 1` (slot của task mới) |
| N5 | Bảo vệ phần dùng chung (bài học F5, F6) | `--new_params class_embed,prompts.private`, `--lr 1e-4 --lr_old 1e-5`, `--lr_drop` = 75% số epoch; `--epochs 12` (task 6 chững từ khoảng epoch 11) |
| N6 | Dữ liệu | Ảnh chụp sản phẩm mới → cutout → ảnh ghép bằng `tools/synth_v2.py` (như `notebooks/data_preprocessing/06_synthetic_relearn.ipynb`); file train chỉ có nhãn của SKU mới; `_gt_full` chỉ để audit |
| N7 | Cổng | AP50 val của SKU mới ≥ 0,90; AP50 của SKU cũ giảm ≤ 0,01; cAcc-M2 của SKU cũ giảm ≤ 0,01 |

**Giới hạn ghi vào báo cáo:** mode improve là fine-tune production, **không** phải CL đúng giao thức (train bằng `_gt_full`, trái ghi chú ở `docs/data_preprocessing/formats.md`, và có replay). Mode new giữ đúng giao thức. Val-B khoảng 2.900 ảnh, sai số chuẩn cAcc khoảng ±0,9 điểm, nên cổng có sign test.

## Global Constraints

- Nhánh mới `feat/update-model`, tách từ `origin/feat/task6-relearn`.
- **Không sửa** `data/training/full-syn-r1/`, `data/training/full-syn-relearn/`, `data/processed/rpc_synth_v2_v1/`, `data/processed/rpc_synth_relearn_v1/`, `data/processed/rpc_eval_official_v1/`. Output mới: `data/processed/rpc_feedback_fix_v1/`, `data/training/update-model/<mode>-<ngày>/`.
- Không truyền `--train_rows` → hành vi train y hệt hiện nay (test bắt buộc, Task 1).
- Chỉ chọn trên Val-A; Val-B chỉ để chấm; test2019 chấm một lần, sau khi người dùng đổi `CONFIRM_TEST = True`.
- Test: `python -m pytest <file> -q` từ repo root, env `pdp` đã `conda activate`, tuần tự.
- Code theo phong cách file xung quanh; tài liệu tiếng Việt. Mỗi task một commit, kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Người dùng tự push.
- Không chạy GPU khi người dùng chưa xác nhận máy và giờ chạy. Mỗi lượt train chạy trong tmux.
- Thiếu dữ liệu (I6, I7, N6) → dừng task đó, báo người dùng, không tự sinh lại.

## File Structure

| File | Mode | Trách nhiệm |
|---|---|---|
| `pdp/main.py` (sửa) | improve | Cờ `--train_rows` (JSON list label, hoặc `all`, mặc định rỗng) |
| `pdp/engine.py` (sửa) | improve | `on_after_backward` + optimizer tôn trọng `--train_rows` (F1, F2, I9) |
| `autocheckout/update_model.py` (mới) | chung | Điểm vào, `prepare`, chấm mốc, train, chấm Val-B, cổng, test |
| `tools/sku_regression.py` (mới) | chung | So hai file dự đoán theo SKU, cặp nhầm, fixed/broken, sign test |
| `autocheckout/feedback_fix.py` (mới) | improve | Chia val, ảnh phản hồi, cặp nhầm, replay, release ghép, task dir |
| `configs/exp/native/EXP-FB-FT.sh` (mới) | improve | Cờ I8 + `--train_rows` + `--train_suffix _fix` (`--prev_ckpt` từ biến `PREV_CKPT` qua `common.sh`) |
| `tools/make_new_task_config.py` (mới) | new | Thêm task class mới vào task config |
| `autocheckout/new_skus.py` (mới) | new | Mở rộng checkpoint, dựng task dir, val của SKU mới |
| `configs/exp/native/EXP-NEW.sh` (mới) | new | Cờ N3–N5 |
| `configs/training/update_model_{improve,new}.json` (mới) | chung | Settings |
| `notebooks/training/08_update_model.ipynb` (mới) | chung | Notebook chạy, một phần cho mỗi mode |
| `tests/test_train_rows.py`, `tests/test_sku_regression.py`, `tests/test_feedback_fix.py`, `tests/test_update_model.py`, `tests/test_make_new_task_config.py` (mới) | | Test |

---

## Phần 0 — Kiểm tra trước

### Task 0: Verify và nhánh

- [x] **Step 1:** Làm hết `docs/project/feedback-finetune-verify-2026-11-01.md`, gửi người dùng bảng kết quả. Có mục FAIL → dừng, chờ người dùng.
- [x] **Step 2:** `git fetch && git checkout -b feat/update-model origin/feat/task6-relearn`; commit plan, file verify, script gốc trong `docs/project/feedback-finetune/`.

## Phần 1 — Mode improve

### Task 1: `--train_rows` (F1, F2, I9)

- [x] **Step 1 (test trước)** `tests/test_train_rows.py`, dựng trainer nhỏ như `tests/pdp_helpers.py` và `tests/test_pdp_expand.py::test_task_after_expansion_trains_on_alias_labels`:
  - không cờ: sau `backward`, grad hàng `< PREV_INTRODUCED_CLS` = 0 (như cũ);
  - `--train_rows [1, 3]`: grad ≠ 0 ở hàng 1, 3 và hàng cuối; = 0 ở hàng khác;
  - `--train_rows all`: grad ≠ 0 ở mọi hàng `< seen_classes`;
  - sau 3 bước optimizer với `--lr 1e-1 --weight_decay 1e-1` (đủ lớn để làm tròn không che): hàng bị khoá **bằng từng bit** giá trị ban đầu.
- [x] **Step 2:** FAIL.
- [x] **Step 3:** `pdp/main.py`: thêm cờ. `pdp/engine.py` `__init__`: dựng `self.frozen_rows` (tensor bool, `True` = khoá; hàng cuối `False`), thuộc tính thường, không phải buffer.
- [x] **Step 4:** `on_after_backward`: có `frozen_rows` → đặt 0 grad weight và bias theo mask, không thì giữ code cũ. Thêm `on_before_optimizer_step` hoặc `optimizer_step` hook: lưu bản sao các hàng bị khoá trước bước và chép lại sau bước (bảo đảm I9, không phụ thuộc weight decay hay trạng thái Adam).
- [x] **Step 5:** PASS; chạy lại `tests/test_pdp_*` không cần CUDA. Commit `feat(pdp): --train_rows chooses the classifier rows a task trains`.

### Task 2: `tools/sku_regression.py`

- [x] **Step 1 (test trước):** fixture nhỏ biết trước fixed/broken; test `skipif(not data)` tái tạo 0,4368093922651934 / 0,4469958563535912.
- [x] **Step 2:** Viết từ `docs/project/feedback-finetune/compare_r1_vs_t6.py`: `per_object`, `per_image`, `compare(...) -> dict` (cAcc hai bên, fixed, broken, p, cặp nhầm trước/sau, tỉ lệ đúng từng SKU, SKU tụt > 0,02 có ≥ 30 vật). CLI in markdown và ghi JSON.
- [x] **Step 3:** PASS. Commit.

### Task 3: Khung `update_model.py` + `EXP-FB-FT.sh` (F3–F6)

- [x] **Step 1:** `configs/exp/native/EXP-FB-FT.sh`: `source EXP-FULL-SYN.sh`, rồi `ARGS+=(--pseudo none --init_new_prompts 0 --new_params class_embed,prompts.private --lr "$LR" --lr_old "$LR_OLD" --lr_drop "$LR_DROP" --train_rows "${TRAIN_ROWS:?}" --train_suffix _fix)` và yêu cầu `: "${PREV_CKPT:?}"`; `--prev_ckpt` do `common.sh` tự thêm từ biến `PREV_CKPT` (verify V9).
- [x] **Step 2:** `autocheckout/update_model.py`: `prepare(settings)` (kiểm md5 I1), `stage_split`, `stage_baseline`, `stage_train(variant)` (`pilot`/`e1`/`e2`), `stage_round2`, `stage_final_test`; `main()` với `--mode`. Train qua `full_syn._train(..., start=5, n=5, epochs=…)`, run dir `runs/EXP-FB-FT-<variant>/`.
- [x] **Step 3:** `tests/test_update_model.py` dry run trên fixture sáu label (như `relearn._relearn_fixture`, `full_syn.DRY_MODEL_ARGS`): `resolved_cli.json` có đúng các cờ F3–F6 ở **giá trị cuối**; hàng `class_embed` trước bước train đầu khớp checkpoint `--prev_ckpt`; không có dòng teacher/PPG trong log; chạy lại bỏ qua stage đã xong; cổng `stop` dừng đúng chỗ.
- [x] **Step 4:** PASS. Commit `feat(update): improve mode skeleton and EXP-FB-FT`.

### Task 4: Dữ liệu `feedback_fix.py` (code xong; Step 3 chờ `_gt_full`)

- [x] **Step 1 (test trước)** trên fixture: Val-A/B không giao nhau kể cả theo nhóm cảnh; level lệch ≤ 1 ảnh; file train không có label ≥ 200; mọi `file_name` tồn tại; id duy nhất; ảnh phản hồi ×3; replay đủ `min_per_class`; alias đổi về gốc.
- [x] **Step 2:** Hàm `split_val`, `count_errors`, `confusion_pairs`, `pair_rows`, `select_synthetic`, `select_replay`, `build_release` (symlink `v2`, `relearn`, `real`; viết lại `file_name`; đánh lại id; `manifest.json`), `build_task_dir` (`train_task_5_fix.json`, `val_full.json` = `val_task_5.json` = Val-B, `test_full.json` symlink).
- [ ] **Step 3:** Có dữ liệu thật → dựng release pilot và E1, in số ảnh mỗi nguồn và số vật mỗi SKU. Commit.

### Task 5: Notebook phần improve + dry run

- [x] **Step 1:** `notebooks/training/08_update_model.ipynb` phần "Cải thiện": markdown tiếng Việt, bảng nguyên nhân → chỗ sửa, bảng cổng; mỗi stage một cell; in bảng so mốc và `sku_regression` kể cả khi `stop`; `CONFIRM_TEST = False`.
- [x] **Step 2:** `python -m tools.execute_training_notebook --notebook notebooks/training/08_update_model.ipynb --settings configs/training/update_model_improve.json --live-notebook notebooks/training/08_update_model.ipynb` (dry run) → exit 0. Commit.

### Task 6: Pilot GPU (người dùng xác nhận trước)

- [ ] **Step 1:** Settings `dry_run: false`, chỉ tới stage `pilot`; chạy trong tmux. Ước tính khoảng 1,8 giờ train (khoảng 13k ảnh × 4 epoch, 0,96 s/bước) + 15 phút chấm.
- [ ] **Step 2:** Kiểm `train.log`: `Trainable parameters` đúng; loss hữu hạn; không teacher/PPG.
- [ ] **Step 3:** Báo người dùng: bảng cổng + `sku_regression`. `stop` → dừng.

### Task 7: E1, vòng 2, test (sau `go` pilot + người dùng đồng ý)

- [ ] **Step 1:** E1 qua đêm (khoảng 8,7 giờ). `stop` → hỏi người dùng có chạy E2 không.
- [ ] **Step 2:** `go` → vòng 2 (khoảng 1 giờ); giữ model tốt hơn theo cổng.
- [ ] **Step 3:** Ghi mục "10. Cải thiện có phản hồi" trong `docs/project/full-syn-r1-analysis-2026-10-29.md` + handoff mới; commit.
- [ ] **Step 4:** Chỉ khi `CONFIRM_TEST = True`: test2019 một lần, so với 44,0%; commit.

## Phần 2 — Mode new (sau Phần 1)

### Task 8: Config task mới

- [ ] **Step 1 (test trước)** `tests/test_make_new_task_config.py`: thêm task với RPC id mới → label nối tiếp, không `alias_of`; RPC id trùng class cũ → lỗi (dùng mode improve); config không đổi task cũ (digest các task cũ giữ nguyên); `n_classes` tăng khi hết slot.
- [ ] **Step 2:** `tools/make_new_task_config.py`. PASS. Commit.

### Task 9: `EXP-NEW.sh`, `new_skus.py`, dry run

- [ ] **Step 1:** `EXP-NEW.sh` theo N3–N5. `new_skus.py`: `stage_expand` (gọi `pdp/expand.py` khi cần slot), `stage_task` (train task mới với thầy + PPG), chấm theo N7.
- [ ] **Step 2:** `update_model.py --mode new` gọi các stage này; phần "Học vật mới" trong notebook 08.
- [ ] **Step 3:** Dry run trên fixture: thêm 1 SKU giả → ô mới train, ô cũ khoá (bằng từng bit), prompt slot mới khởi tạo, cổng N7 ra quyết định. Commit.

### Task 10: Chạy mode new thật (khi có dữ liệu sản phẩm mới)

- [ ] **Step 1:** Người dùng cung cấp ảnh sản phẩm mới. Sinh cutout và ảnh ghép theo N6 (notebook dữ liệu riêng, như `06_synthetic_relearn`).
- [ ] **Step 2:** Chạy `--mode new`; báo cổng N7. Nếu SKU mới hay nhầm với SKU cũ → chạy `--mode improve` với `--train_rows` gồm cả ô mới (G3).

### Ngoài phạm vi

- Nối vào `checkout-system`: route `relearn.py` (đang tạo lớp mới cho sản phẩm nhận sai) chuyển sang gọi `--mode improve`; route sản phẩm mới gọi `--mode new`; bỏ TD-04 trong `services/corrections.py`. Plan riêng sau khi Phần 1 `go`.
