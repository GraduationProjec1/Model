# Runbook GPU: active learning + mốc fine-tune thẳng (B2)

Code: nhánh `feat/active-learning`. Mọi thứ đã dựng sẵn trên Mac; máy GPU chỉ cần **lấy code + dữ liệu rồi chạy**.
Quy trình thuê máy / SSH / Kaggle: như runbook Vast-GitHub-Kaggle (tài liệu dự án `claude/runbook-vast-github-kaggle-2026-11-02.md`).

## 1. Trước buổi GPU (Mac, một lần)
```bash
cd ~/Desktop/Finalprpject/AutoCheckout-CL
git push -u origin feat/active-learning
# 2 gói mới đã tạo sẵn ở ~/Desktop/Finalprpject/packages_al/ (tạo lại: bash scripts/pack_al_data.sh)
#   relearn_fix_v2_al_20261102.tar  (785 MB)  md5 7d975beaeb55954f46d7dd02461be550
#   seed_full_syn_r1_b2_20261102.tar (993 MB) md5 dbb1a9d45afc23dac6090e1839b442bd
```
Upload lên Kaggle (Private) cùng các gói đã có: `relearn_fix_v2.zip` (13 GB), `full_syn_rpc_synth_v2_v1_20261025.tar`
(chỉ cần cho B2).

## 2. Trên máy GPU
```bash
git clone git@github.com:<user>/AutoCheckout-CL.git ~/AutoCheckout-CL
cd ~/AutoCheckout-CL && git checkout feat/active-learning && pip install -r requirements.txt
kaggle datasets download -d <user>/<dataset> -p ~/dl --unzip

# active learning: relearn_fix_v2 ĐẶT CẠNH repo, rồi đè phần AL lên
unzip -q ~/dl/relearn_fix_v2.zip -d ~/            # -> ~/relearn_fix_v2
tar -xf ~/dl/relearn_fix_v2_al_20261102.tar -C ~/ # thêm active/, variants/al_*, images/real
# B2 (chỉ khi chạy B2)
tar -xf ~/dl/full_syn_rpc_synth_v2_v1_20261025.tar -C ~/AutoCheckout-CL
tar -xf ~/dl/seed_full_syn_r1_b2_20261102.tar -C ~/AutoCheckout-CL

bash scripts/check_al_setup.sh                    # phải in SẴN SÀNG
tmux new -s al
```

## 3. Active learning
| Bước | Lệnh | Thời gian (4090, ước tính) |
|---|---|---|
| Đợt 1: B=500 (random×3, margin, margin_k, margin_bal) + oracle | `bash scripts/run_al.sh dot1` | 7 run × ~1 giờ |
| Xem kết quả | `cat ../relearn_fix_v2/active/report/al_results.md` (+ `al_curve.png`) | |
| **Chỉ khi margin/margin_k/margin_bal > cả 3 random**: đợt 2 (B=250, 850, entropy) | `bash scripts/run_al.sh dot2` | 15 run × ~1 giờ |
| Vòng 2 cho chiến lược tốt nhất + random so sánh | `bash scripts/run_al.sh al_margin_k_B500_r2 al_random_B500_s0_r2` | 2 run × ~0,5 giờ |
| test2019, **một lần**, khi nhóm đã chốt | `bash scripts/run_al.sh test al_<tốt nhất> al_random_B500_s0 --confirm-test` | dự đoán ~0,5 giờ/run |

Số chính để báo cáo: `active/scored/<run>.json` → `cAcc_m2` (Val-B, ngưỡng chọn trên Val-A-calib). Mốc r1: **0,4232**.
`stages/<run>.json` có thêm số theo kiểu cũ (ngưỡng chọn trên Val-B), AP50 SKU khác, cặp nhầm, cổng go/stop.
Bị ngắt: chạy lại đúng lệnh (run xong được bỏ qua, run dở tiếp từ checkpoint).

## 4. B2: mốc fine-tune thẳng
```bash
python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 3   # ~10 giờ, đủ thấy quên
python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 5   # ~20 giờ, đủ 5 task
```
In ra và ghi `docs/al/ket-qua/forgetting.md`: bảng AP50 từng nhóm của r1 và NAIVE cạnh nhau, độ quên, BWT.

## 5. Mang kết quả về Mac
```bash
rsync -avP vast:~/relearn_fix_v2/active/ ~/Desktop/Finalprpject/relearn_fix_v2/active/
rsync -avP --exclude '*.pth' vast:~/AutoCheckout-CL/data/training/ ~/Desktop/Finalprpject/ket-qua-gpu/
rsync -avP vast:~/AutoCheckout-CL/docs/al/ket-qua/ ~/Desktop/Finalprpject/AutoCheckout-CL/docs/al/ket-qua/
# checkpoint cần giữ (vd. run AL tốt nhất): rsync thêm runs/EXP-FB-FT-<run>/task_5/task_final.pth
```
Nếu sửa code trên máy GPU: `git commit && git push` trước khi destroy máy.
