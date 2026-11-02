# Runbook — chạy "cập nhật model" mode improve (01/11/2026)

Hướng dẫn chạy từ đầu tới cuối cho nhánh `feat/update-model`. Plan chi tiết: [docs/project/feedback-finetune-plan-2026-11-01.md](docs/project/feedback-finetune-plan-2026-11-01.md).

## 0. Code đã đủ chưa

| Phần | Trạng thái |
|---|---|
| Mode **improve** (sửa vật nhận sai): cờ `--train_rows`, dữ liệu, train, chấm, cổng, notebook 08 | **Đủ để chạy.** 112 test qua; dry run CPU chạy hết chuỗi split → baseline → pilot → e1 → round2 → test |
| Chạy trên dữ liệu thật và GPU | **Chưa chạy lần nào.** Lần chạy đầu là pilot (mục 5) |
| Mode **new** (học sản phẩm mới) | **Chưa viết** (Phần 2 của plan). `--mode new` báo lỗi `NotImplementedError` |

Việc của bạn: đưa code lên máy GPU, chuẩn bị dữ liệu, rồi chạy lần lượt các mục 1–9.

## 1. Đưa code lên GitHub (trên Mac)

```bash
cd ~/Desktop/Finalprpject/AutoCheckout-CL
git push -u origin feat/update-model
```

## 2. Chuẩn bị máy GPU (vast 4090)

```bash
cd ~/AutoCheckout-CL
git fetch origin && git checkout feat/update-model
conda activate pdp
```

Máy mới thì cài env như [handoff M5](docs/project/HANDOFF_M5_PILOT_2026-10-30.md) mục 3.3, rồi giải nén hai gói dữ liệu ở **gốc repo**, không phải trong `data/`, và không dùng `--progress`:

```bash
rclone cat drive:AutoCheckout/2026-10-25/full_syn_rpc_synth_v2_v1_20261025.tar | tar -x -C ~/AutoCheckout-CL
rclone cat drive:AutoCheckout/2026-10-31/relearn_rpc_synth_relearn_v1_20261031.tar | tar -x -C ~/AutoCheckout-CL
sha256sum -c --quiet data/packages/relearn_rpc_synth_relearn_v1_20261031/SHA256SUMS && echo RELEARN_OK
```

## 3. Kiểm dữ liệu (bắt buộc, khoảng 1 phút)

```bash
md5sum data/training/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth   # phải là fe510a4c4dc2c81a328299c4009d31ff
ls data/training/full-syn-r1/plan.json \
   data/training/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz \
   data/training/full-syn-r1/runs/EXP-FULL-SYN/calibration_count_m2.json \
   data/training/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model/config.json
ls data/processed/rpc_eval_official_v1/{val_full,test_full,task_config}.json
ls data/processed/rpc_synth_v2_v1/tasks/synth_v2/train_task_{1,2,3,4,5}_gt_full.json
ls data/processed/rpc_synth_relearn_v1/tasks/synth_v2/train_task_{6,7}_gt_full.json
ls data/archive/val2019 | wc -l                                                # 6000
ls data/pretrained/hf-cache
```

Thiếu file nào thì **dừng**, lấy lại gói tương ứng; không tự sinh lại dữ liệu.

## 4. Test và settings (khoảng 10 phút, CPU)

```bash
python -m pytest tests/test_train_rows.py tests/test_sku_regression.py tests/test_feedback_fix.py \
  tests/test_update_model.py -q
cp configs/training/update_model_improve.json data/training/settings_update_model.json
sed -i 's/"dry_run": true/"dry_run": false/' data/training/settings_update_model.json
grep -E 'dry_run|campaign_id|seed_campaign|synth_v2_root|relearn_release' data/training/settings_update_model.json
```

Phải thấy `"dry_run": false`, `"campaign_id": "update-improve-1"`, `"seed_campaign": "data/training/full-syn-r1"`. Kết quả của campaign nằm ở `data/training/update-model/update-improve-1/`; dữ liệu train ghép nằm ở `data/processed/rpc_feedback_fix_v1/<pilot|e1|round2>/`.

## 5. Mốc và pilot (khoảng 2–2,5 giờ)

Chạy trong tmux. Mỗi lệnh tự chạy các bước trước nó còn thiếu, bước đã xong thì bỏ qua:

```bash
tmux new -s fbft            # Ctrl+B rồi D để thoát ra; tmux attach -t fbft để vào lại
conda activate pdp && cd ~/AutoCheckout-CL
S=data/training/settings_update_model.json
python -m autocheckout.update_model --mode improve --settings $S --stage baseline 2>&1 | tee fbft_baseline.log
python -m autocheckout.update_model --mode improve --settings $S --stage pilot    2>&1 | tee fbft_pilot.log
```

Sau `baseline` (vài phút), kiểm `data/training/update-model/update-improve-1/stages/baseline.json`:

- `cAcc_m2` của round 1 trên Val-B khoảng 0,43–0,44;
- `val_a_wrong_images` khoảng 1.600;
- `rows.pilot` gồm khoảng 10–20 label; xem các cặp trong `pairs_top50`.

Số lệch xa thì dừng lại hỏi trước khi chạy pilot.

**Trong lúc pilot chạy:**

| Xem gì | Ở đâu |
|---|---|
| Số ảnh mỗi nguồn (phản hồi / ảnh ghép / replay) | `data/processed/rpc_feedback_fix_v1/pilot/manifest.json` |
| Tiến độ, loss | `.../runs/EXP-FB-FT-pilot/task_5/train.log`, `progress.json` |
| Cờ thật sự dùng | `.../runs/EXP-FB-FT-pilot/resolved_cli.json`: phải có `--pseudo none`, `--train_rows`, `--prev_ckpt` |
| GPU | `nvidia-smi` |

Bị ngắt thì chạy lại đúng lệnh: task đang dở chạy tiếp từ `last.ckpt`.

## 6. Đọc kết quả pilot

Cuối log có dòng `GATE improve pilot: GO/STOP - …`. Chi tiết:

- `stages/pilot.json`: cAcc-M2 Val-B trước/sau, số ảnh được sửa (fixed) / bị làm sai (broken), p của sign test, AP50 của SKU khác, các SKU bị tụt;
- `scored/pilot/sku_regression.json`: tỉ lệ đúng từng SKU, các cặp nhầm trước/sau.

| Kết quả | Làm gì |
|---|---|
| `GO` | Sang mục 7 |
| `STOP` | **Dừng.** Gửi `stages/pilot.json` và `scored/pilot/sku_regression.json` để phân tích; không tự chạy E1/E2 |

## 7. E1 (khoảng 9 giờ, chạy qua đêm), rồi vòng 2 (khoảng 1 giờ)

```bash
python -m autocheckout.update_model --mode improve --settings $S --stage e1     2>&1 | tee fbft_e1.log
python -m autocheckout.update_model --mode improve --settings $S --stage round2 2>&1 | tee fbft_round2.log
```

- E1 `STOP` → hỏi nhóm có chạy E2 (mở cả 200 ô) không. Đồng ý thì chạy `--stage e2`; vòng 2 chỉ chạy sau `GO` của E1.
- Vòng 2 so với E1. `STOP` nghĩa là giữ model E1.

Xem lại mọi bảng bằng notebook (các bước đã xong được bỏ qua, chỉ in kết quả):

```bash
python -m tools.execute_training_notebook --notebook notebooks/training/08_update_model.ipynb \
  --settings $S --live-notebook notebooks/training/08_update_model.ipynb
```

Lưu ý: chạy notebook khi E1 chưa xong thì nó sẽ train tiếp E1 và vòng 2. Chỉ dùng notebook để xem lại, hoặc khi muốn chạy liền một mạch.

## 8. Test2019 (chỉ một lần, sau khi nhóm đã đọc hết cổng)

```bash
python -m autocheckout.update_model --mode improve --settings $S --stage final_test --confirm-test
```

Lệnh này lấy model mới nhất đã `GO` (vòng 2 → E1 → E2), dùng ngưỡng chọn trên Val-B. So với round 1 task 5 trên test: cAcc 44,0% (ngưỡng nhóm). Đã chấm rồi thì lệnh chỉ in lại kết quả, không chấm lại.

## 9. Mang kết quả về và ghi lại

```bash
cd ~/AutoCheckout-CL
zip -r update-improve-1_$(date +%Y%m%d).zip data/training/update-model/update-improve-1 \
    data/processed/rpc_feedback_fix_v1/*/manifest.json -x '*.ckpt'
rclone copy -P update-improve-1_*.zip drive:AutoCheckout/$(date +%F)/
```

Ghi kết quả vào mục "10. Cải thiện có phản hồi" của `docs/project/full-syn-r1-analysis-2026-10-29.md`, rồi commit.

## Khi gặp lỗi

| Thông báo | Nghĩa / cách xử lý |
|---|---|
| `GateStop: improve <x>: …` | Cổng dừng có chủ đích, không phải bug. Đọc lý do ở mục 6 |
| `settings or inputs changed in an existing campaign` | Settings hoặc dữ liệu đã đổi so với lần trước: đặt `campaign_id` mới trong settings |
| `... has md5 ..., the settings expect fe510a4c...` | Checkpoint task 5 của round 1 không đúng bản: lấy lại `training.zip` |
| `pinned pretrained weights missing` | Thiếu `data/pretrained/hf-cache`: giải nén lại gói v2 |
| `has no predictions for every image of Val-B` | File dự đoán của run thiếu ảnh: xoá `runs/EXP-FB-FT-<x>/task_5/pred_val.npz` rồi chạy lại lệnh (chỉ dự đoán lại) |
| `another notebook is using this campaign` | Một lệnh khác đang chạy cùng campaign: chờ, hoặc dừng nó trong tmux |
| `the round-1 model confuses no pair` | Không tìm được cặp nhầm trên Val-A: kiểm lại `val_full.json` và `pred_val.npz` của round 1 |
