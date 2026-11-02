# Runbook GPU: active learning + mốc fine-tune thẳng (B2)

Code: nhánh `feat/active-learning`. Dữ liệu: **một thư mục `autocheckout-data`** (30 GB) đặt **cạnh** repo; cấu trúc và
nguồn gốc trong `autocheckout-data/README.md`, đường dẫn trong code ở `autocheckout/datapaths.py` (đổi chỗ: `AC_DATA`).
Thuê máy / SSH / Kaggle: như tài liệu dự án `claude/runbook-vast-github-kaggle-2026-11-02.md`.

```
~/                         (/root trên Vast)
├── AutoCheckout-CL/       git clone, nhánh feat/active-learning; kết quả ghi vào AutoCheckout-CL/data/training/
└── autocheckout-data/     rpc/ eval/ synth_v2/ pretrained/ models/full-syn-r1/ relearn/
```

## 1. Trước buổi GPU (Mac, một lần)
```bash
cd ~/Desktop/Finalprpject/AutoCheckout-CL && git push -u origin feat/active-learning
cd ~/Desktop/Finalprpject && tar -cf autocheckout-data.tar autocheckout-data && md5 autocheckout-data.tar   # ~15 phút
```
Upload `autocheckout-data.tar` (một file, ~30 GB) lên dataset Kaggle Private. Kaggle giữ nguyên file `.tar`.

## 2. Trên máy GPU (ổ ≥ 100 GB)
```bash
cd ~
git clone git@github.com:<user>/AutoCheckout-CL.git
cd AutoCheckout-CL && git checkout feat/active-learning && pip install -r requirements.txt && cd ~
kaggle datasets download -d <user>/<dataset> -p ~/dl --unzip
tar -xf ~/dl/autocheckout-data.tar -C ~/ && rm -rf ~/dl     # -> ~/autocheckout-data
cd ~/AutoCheckout-CL && FULL=1 bash scripts/check_al_setup.sh   # phải in SẴN SÀNG (FULL=1: kiểm SHA256, lần đầu)
tmux new -s al
```

## 3. Active learning
| Bước | Lệnh | Thời gian (4090, ước tính) |
|---|---|---|
| Đợt 1: B=500 (random×3, margin, margin_k, margin_bal) + oracle | `bash scripts/run_al.sh dot1` | 7 run × ~1 giờ |
| Xem kết quả | `cat ../autocheckout-data/relearn/active/report/al_results.md` (+ `al_curve.png`) | |
| **Chỉ khi margin/margin_k/margin_bal > cả 3 random**: đợt 2 (B=250, 850, entropy) | `bash scripts/run_al.sh dot2` | 15 run × ~1 giờ |
| Vòng 2 cho chiến lược tốt nhất + random so sánh | `bash scripts/run_al.sh al_margin_k_B500_r2 al_random_B500_s0_r2` | 2 run × ~0,5 giờ |
| test2019, **một lần**, khi nhóm đã chốt | `bash scripts/run_al.sh test al_<tốt nhất> al_random_B500_s0 --confirm-test` | ~0,5 giờ/run |

Số chính: `autocheckout-data/relearn/active/scored/<run>.json` → `cAcc_m2` (Val-B, ngưỡng chọn trên Val-A-calib).
Mốc r1: **0,4232**. `data/training/update-model/relearn-fix-v2/stages/<run>.json` có thêm số kiểu cũ (ngưỡng chọn trên
Val-B; r1 0,4308), AP50 SKU khác, cặp nhầm, cổng go/stop. Bị ngắt: chạy lại đúng lệnh.

## 4. B2: mốc fine-tune thẳng
```bash
python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 3   # ~10 giờ, đủ thấy quên
python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 5   # ~20 giờ, đủ 5 task
```
Ghi `docs/al/ket-qua/forgetting.md`: bảng AP50 từng nhóm của r1 và NAIVE cạnh nhau, độ quên, BWT.

## 5. Mang kết quả về Mac
```bash
rsync -avP vast:~/autocheckout-data/relearn/active/ ~/Desktop/Finalprpject/autocheckout-data/relearn/active/
rsync -avP --exclude '*.pth' vast:~/AutoCheckout-CL/data/training/ ~/Desktop/Finalprpject/ket-qua-gpu/
rsync -avP vast:~/AutoCheckout-CL/docs/al/ket-qua/ ~/Desktop/Finalprpject/AutoCheckout-CL/docs/al/ket-qua/
# checkpoint cần giữ (vd. run AL tốt nhất): rsync thêm runs/EXP-FB-FT-<run>/task_5/task_final.pth
```
Nếu sửa code trên máy GPU: `git commit && git push` trước khi destroy máy.
