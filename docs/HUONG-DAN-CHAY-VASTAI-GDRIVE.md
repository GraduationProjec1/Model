# Hướng Dẫn Huấn Luyện Tự Động Trên Vast.ai & Đồng Bộ Google Drive

> Tài liệu hướng dẫn chi tiết nằm tại gốc dự án: [HUONG-DAN-CHAY-VASTAI-GDRIVE.md](../../HUONG-DAN-CHAY-VASTAI-GDRIVE.md)
> File script tự động: [vast_train_and_sync.sh](../scripts/vast_train_and_sync.sh)

Tóm tắt lệnh chạy nhanh trên máy Vast.ai:
```bash
cd /workspace/AutoCheckout-CL
source /workspace/venv-pdp/bin/activate
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
export VAST_API_KEY="<VAST_API_KEY>"
export RCLONE_REMOTE="gdrive:AutoCheckout/ket_qua_gpu"
export AUTO_DESTROY=1

bash scripts/vast_train_and_sync.sh al_dot1
```
