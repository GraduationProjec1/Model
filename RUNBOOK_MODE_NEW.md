# Runbook: học 25 sản phẩm VN ở task 6 trên vast.ai (mode new, 20/11/2026)

Thiết kế, cổng và cách đọc kết quả: [docs/project/mode-new-2026-11-20.md](docs/project/mode-new-2026-11-20.md).
Nhánh code: `feat/mode-new`. Thời gian GPU ước tính 1,5–2,5 giờ trên RTX 4090, cộng 30–60 phút tải dữ liệu.

Trong tài liệu này, `DRIVE` là thư mục Drive chung của nhóm (folder id `14eR9fnA2Uu4TAhHVdKBnPyFA2I62Pn3t`), dùng
qua remote `gdrive` của rclone:

```bash
DRIVE='gdrive,root_folder_id=14eR9fnA2Uu4TAhHVdKBnPyFA2I62Pn3t:'
```

## 0. Trên Mac (trước khi thuê máy)

**0.1 Upload gói dữ liệu VN** (1,21 GB) vào `AN/new_products/` trên Drive chung:

```bash
cd "/Users/an/Documents/Do An/Ca/HIEN/upload"
DRIVE='gdrive,root_folder_id=14eR9fnA2Uu4TAhHVdKBnPyFA2I62Pn3t:'
rclone copy . "${DRIVE}AN/new_products" --include "vn25_20261120.tar*" --progress
rclone md5sum "${DRIVE}AN/new_products" --include "vn25_20261120.tar"   # phải trùng: cat vn25_20261120.tar.md5
```

Gói gồm `products.json`, `tasks_ac/task6/{train,val,test}_task_6.json`, 2.000 ảnh ghép, 140 ảnh quầy thật và
`SHA256SUMS` của 2.148 file. Model r4 và bộ dữ liệu RPC đã có sẵn trên Drive, không cần upload.

**0.2 Chuẩn bị khoá:**
- API key W&B (https://wandb.ai/authorize) để xem loss/AP trực tiếp.
- API key vast.ai (Account → API Keys) để script tự tắt máy.
- File cấu hình rclone ở `~/.config/rclone/rclone.conf`.

## 1. Thuê máy

- GPU: **1× RTX 4090** (24 GB). Ổ đĩa **≥ 80 GB**: dữ liệu RPC 32 GB, gói VN 1,2 GB, kết quả khoảng 3 GB.
- Image: một image có sẵn CUDA 12.x devel, ví dụ `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel`, hoặc template
  PyTorch mặc định của vast.
- Bật SSH. Ghi lại lệnh SSH (host, port) và **Instance ID**.

Từ Mac, chép cấu hình rclone lên máy (thay `<port>` và `<host>`):

```bash
ssh -p <port> root@<host> 'mkdir -p /root/.config/rclone'
scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf
```

## 2. Code và môi trường (trên máy GPU, khoảng 15 phút)

```bash
ssh -p <port> root@<host>
cd /workspace
git clone https://github.com/GraduationProjec1/Model.git AutoCheckout-CL   # repo private: dùng token
cd AutoCheckout-CL && git checkout feat/mode-new && git log -1 --oneline
command -v sudo >/dev/null || (apt-get update -qq && apt-get install -y -qq sudo)   # setup_vm.sh gọi sudo
command -v rclone >/dev/null || (curl -fsSL https://rclone.org/install.sh | bash)
VENV=/workspace/venv-pdp TORCH_CUDA_ARCH_LIST=8.9 bash scripts/setup_vm.sh   # torch 2.2.2 cu121, kernel, test
source /workspace/venv-pdp/bin/activate
pip install -q wandb==0.18.7
```

`setup_vm.sh` chạy cả bộ test. Trên máy GPU mọi test phải qua: các lỗi do đường dẫn có dấu cách chỉ xảy ra trên Mac.

## 3. Dữ liệu (30–60 phút, phần lớn là tải 32 GB)

```bash
DRIVE='gdrive,root_folder_id=14eR9fnA2Uu4TAhHVdKBnPyFA2I62Pn3t:'
DATA=/workspace/autocheckout-data
cd /workspace

# 3.1 Bộ dữ liệu RPC (eval, ảnh val/test, model round 1, pretrained, split Val-A): giải nén thẳng, không lưu tar
rclone cat "${DRIVE}autocheckout-data.tar" | tee >(md5sum > /workspace/acdata.md5) | tar -x -C /workspace
cat /workspace/acdata.md5                       # phải là 5a1efb48209d998012b01dd1dfa9674c
ls $DATA                                        # eval models pretrained relearn rpc synth_v2 ...

# 3.2 Model AL vòng 4 (điểm xuất phát)
R4="${DRIVE}ket_qua_gpu/al_r4_20261114_165406/update-model/relearn-fix-v2/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5"
rclone copy "$R4" $DATA/models/al-r4/task_5 --include "task_final.pth" --include "pred_val.npz" \
    --include "run_info.json" --include "runtime.json" --progress
md5sum $DATA/models/al-r4/task_5/task_final.pth  # phải là 64c57f9cbbd5ca056da1c50b7991b94b

# 3.3 Split Val-A (chọn ngưỡng hàng cũ trên calib); ghi đè nếu khác
rclone copyto "${DRIVE}ket_qua_gpu/al_r4_20261114_165406/relearn_active/splits/val_a_split.json" \
    $DATA/relearn/active/splits/val_a_split.json
md5sum $DATA/relearn/active/splits/val_a_split.json   # phải là 3d0e2c7a1648aed8cf2ed3752c782ff3

# 3.4 Gói VN
mkdir -p $DATA/new_products
rclone cat "${DRIVE}AN/new_products/vn25_20261120.tar" | tar -x -C $DATA/new_products
(cd $DATA/new_products/vn25 && sha256sum -c --quiet SHA256SUMS && echo "GÓI VN NGUYÊN VẸN")
```

## 4. Kiểm tra trước khi tốn GPU (khoảng 2 phút, CPU)

```bash
cd /workspace/AutoCheckout-CL && source /workspace/venv-pdp/bin/activate
export AC_DATA=/workspace/autocheckout-data
python -m pytest -q tests/test_new_skus.py tests/test_make_new_task_config.py   # cả dry run toàn chuỗi
python -m autocheckout.update_model --mode new --settings configs/training/update_model_new_vn25.json --stage data
python - <<'PY'
import json
s = json.load(open('data/training/update-model/new-vn25/inputs/splits.json'))
print({k: len(v) for k, v in s.items()})   # cần {'calib': 581, 'val_b': 2897, 'new_val': 28, 'new_test': 112}
PY
```

Dòng `--stage data` phải in `new task 6 = labels 200..224, n_classes 251; train 2000 images (pilot 300); seed md5
64c57f9c...`. Nếu báo lỗi (thiếu file, md5 lệch, Val-A không khớp) thì dừng ở đây và sửa trước khi chạy.

## 5. Chạy

Chạy trong tmux để SSH đứt vẫn chạy tiếp:

```bash
tmux new -s vn25          # Ctrl+B rồi D để thoát ra; tmux attach -t vn25 để vào lại
cd /workspace/AutoCheckout-CL && source /workspace/venv-pdp/bin/activate
export AC_DATA=/workspace/autocheckout-data CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
export AC_WANDB_PROJECT=autocheckout WANDB_API_KEY=<key W&B>
export RCLONE_REMOTE='gdrive,root_folder_id=14eR9fnA2Uu4TAhHVdKBnPyFA2I62Pn3t:AN/ket_qua_gpu'
export VAST_API_KEY=<key vast> INSTANCE_ID=<id máy>
export AUTO_DESTROY=0 AUTO_STOP=1      # lần đầu: chỉ STOP máy để còn vào xem; tự tin rồi mới để AUTO_DESTROY=1
bash scripts/vast_train_and_sync.sh new_vn25
```

Script tự làm theo thứ tự (bước đã xong được bỏ qua nếu chạy lại đúng lệnh):

| Bước | Làm gì | Thời gian ước tính |
|---|---|---|
| `data` | Dựng campaign `data/training/update-model/new-vn25` | < 1 phút |
| `expand` | r4 225 → 251 lớp, kiểm hàng lớp cũ giữ nguyên | < 1 phút |
| `baseline` | Model mở rộng (vẫn task 5) dự đoán 5.820 ảnh val: mốc của cổng | khoảng 10 phút |
| `pilot` | 1 epoch, 300 ảnh, rồi dự đoán val (W&B: `EXP-NEW-pilot/task_6`) | khoảng 15 phút |
| `task` | Task 6: 12 epoch, 2.000 ảnh, rồi dự đoán val và cổng (W&B: `EXP-NEW/task_6`) | khoảng 50 phút |
| `final_test` | Chỉ khi cổng go: 112 ảnh test quầy VN, một lần | vài phút |

Sau đó script upload `data/training/update-model` và log lên `AN/ket_qua_gpu/new_vn25_<giờ>/` (cứ 30 phút cũng
upload một lần), kiểm đủ file `.pth` trên Drive, rồi tắt máy. `FINAL_TEST=0` để tự chấm test sau.

## 6. Theo dõi trong lúc chạy

- **W&B** (project `autocheckout`): loss, AP val VN từng epoch của `EXP-NEW-pilot/task_6` và `EXP-NEW/task_6`. Đây
  cũng là chỗ thử phần giám sát realtime của Checkout-system (AI → Giám sát → Đang train).
- Log: `tail -f data/training/update-model/vast_logs_new_vn25_*/master_execution.log`
- Từng bước xong: `ls data/training/update-model/new-vn25/stages/`
- GPU: `nvidia-smi -l 5`

Pilot là chốt an toàn: loss phải giảm và `stages/pilot.json` phải có số (AP50 VN sau 1 epoch còn thấp là bình
thường). Pilot lỗi thì script dừng trước lượt dài.

## 7. Đọc kết quả

`data/training/update-model/new-vn25/stages/` (cũng có trên Drive):

| File | Xem gì |
|---|---|
| `baseline.json` | `old.cAcc_m2` ≈ 0,6828 (= `seed_val_b_cAcc_m2`), `old.AP50` ≈ 0,979, `new.cross_rate` ≈ 0,72 |
| `pilot.json` | Chuỗi chạy đúng; `new.AP50` sau 1 epoch |
| `task.json` | `gate.decision` (go/stop) và `gate.reason`; `new.AP50`, `new.AP50_per_sku`, `old.AP50`, `old.cAcc_m2`, `new.cross_rate`, `shop_must_filter_labels`, `comparison` (sign test), `train_hours` |
| `final_test.json` | Điểm cuối trên 112 ảnh test: `AP50`, `cAcc_m2` (mọi nhãn), `cAcc_m2_new_labels_only` (có lọc nhãn), `by_level`, `cross_rate` |

Cách hiểu go/stop và bước tiếp theo ở mục 4 của tài liệu thiết kế. Bị ngắt giữa chừng thì vào lại máy và chạy lại
đúng lệnh ở bước 5: task dở chạy tiếp từ `last.ckpt`.

## 8. Lấy model về cho Checkout-system

Model task 6: `runs/EXP-NEW/task_6/task_final.pth` (+ `runtime.json`), ngưỡng đếm trong `stages/task.json` →
`thresholds` (task_1..task_6), config `inputs/task_config.json` (251 lớp). Việc nạp vào backend (`slot_total`,
`task_skus`, lọc nhãn khi `shop_must_filter_labels`) làm sau khi có kết quả.
