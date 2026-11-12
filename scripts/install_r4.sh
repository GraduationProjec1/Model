#!/usr/bin/env bash
# Đợt GPU 1 sau vòng 3 (docs/design/2026-11-12-dot-gpu-1): bổ sung vào máy đã chạy gpu_bundle_r3/install.sh những thứ
# đợt này cần thêm, lấy thẳng từ Drive bằng rclone (không có zip riêng):
#   1. code nhánh feat/dot-gpu-1 (install.sh của gói r3 đã ghi đè code về commit 1d40e74);
#   2. weights, pred, stage, scored, inputs của al_margin_k_B500_r3_real và _r3_all (đợt al_r3 09/11);
#   3. weights_md5.txt của đợt al_r3 (case al_r4 của vast_train_and_sync.sh kiểm md5 trước khi train);
#   4. tập chọn vòng 4 margin_k_B500_r3.json (chọn trên Mac 12/11, kiểm rò rỉ 0).
# Chạy trên máy GPU, sau gpu_bundle_r3/install.sh:  bash /workspace/AutoCheckout-CL/scripts/install_r4.sh
# Chạy lại được: rclone chỉ tải file thiếu/khác.
set -euo pipefail
REPO=${REPO:-/workspace/AutoCheckout-CL}
DATA=${AC_DATA:-/workspace/autocheckout-data}
BRANCH=${BRANCH:-feat/dot-gpu-1}
SRC=${SRC:-gdrive:AutoCheckout/ket_qua_gpu/al_r3_20261109_021108}
SEL_SRC=${SEL_SRC:-gdrive:AutoCheckout/gpu_bundle_r4/selections}
SEL_MD5=757b1e93c23db2da7a0b48975104e265      # margin_k_B500_r3.json tạo trên Mac 12/11/2026
PRED_MD5=08fdb71ff7f7630227342b03c5c40f0c     # pred_val.npz của r3_real mà tập chọn dựa vào
CAMP=$REPO/data/training/update-model/relearn-fix-v2
RUNS=("al_margin_k_B500_r3_real" "al_margin_k_B500_r3_all")

fail() { echo "[LỖI] $*" >&2; exit 1; }
echo "== 0. Điều kiện: gói r3 đã cài"
[[ -f $CAMP/plan.json && -f $CAMP/stages/al_margin_k_B500_r2.json ]] || fail "chưa chạy gpu_bundle_r3/install.sh ($CAMP thiếu plan.json / stage r2)"
[[ -f $DATA/relearn/variants/e1/train_task_5_fix.json ]] || fail "thiếu gói e1 trong $DATA/relearn/variants (al_round2 dựng r4 từ base e1)"
[[ -f $DATA/relearn/variants/al_margin_k_B500_r3_real/train_task_5_fix.json ]] || fail "thiếu gói al_margin_k_B500_r3_real (gói r3)"
command -v rclone >/dev/null || fail "thiếu rclone"

echo "== 1. Code nhánh $BRANCH"
git -C "$REPO" fetch -q origin "$BRANCH"
# install.sh của gói r3 giải nén tar code 1d40e74 đè lên bản clone (không có .git trong tar): file theo dõi bị sửa,
# nên phải reset --hard về nhánh mới; data/ (gitignore) và file lạ không bị đụng.
git -C "$REPO" checkout -q -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || true
git -C "$REPO" reset -q --hard "origin/$BRANCH"
echo "   HEAD: $(git -C "$REPO" rev-parse --short HEAD)  $(git -C "$REPO" log -1 --format=%s)"
grep -q "al_r4)" "$REPO/scripts/vast_train_and_sync.sh" || fail "code chưa có case al_r4 (sai nhánh?)"

echo "== 2. Campaign: r3_real, r3_all"
for v in "${RUNS[@]}"; do
    rclone copy "$SRC/update-model/relearn-fix-v2/runs/EXP-FB-FT-$v/task_5" "$CAMP/runs/EXP-FB-FT-$v/task_5" \
        --include "task_final.pth" --include "pred_val.npz" --include "run_info.json" --progress
    rclone copy "$SRC/update-model/relearn-fix-v2/stages" "$CAMP/stages" --include "$v.json"
    rclone copy "$SRC/update-model/relearn-fix-v2/scored/$v" "$CAMP/scored/$v"
    rclone copy "$SRC/update-model/relearn-fix-v2/inputs/$v" "$CAMP/inputs/$v"
    for f in "runs/EXP-FB-FT-$v/task_5/task_final.pth" "runs/EXP-FB-FT-$v/task_5/pred_val.npz" "stages/$v.json" \
             "scored/$v/calibration_count_m2.json" "scored/$v/calibration_count_groups.json" "inputs/$v/train_task_5_fix.json"; do
        [[ -f $CAMP/$f ]] || fail "thiếu $CAMP/$f"
    done
done

echo "== 3. md5 weights (logs/weights_md5.txt của đợt al_r3)"
rclone copy "$SRC/logs" "$CAMP/.." --include "weights_md5.txt"
(cd "$CAMP/.." && md5sum -c --quiet --ignore-missing weights_md5.txt) || fail "md5 weights lệch"
echo "   md5 khớp cho: $(cd "$CAMP/.." && md5sum -c --ignore-missing weights_md5.txt 2>/dev/null | grep -c ': OK') file"

echo "== 4. Tập chọn vòng 4"
rclone copy "$SEL_SRC" "$DATA/relearn/active/selections" --include "margin_k_B500_r3.json"
SEL=$DATA/relearn/active/selections/margin_k_B500_r3.json
[[ $(md5sum "$SEL" | cut -d' ' -f1) == "$SEL_MD5" ]] || fail "$SEL md5 khác $SEL_MD5"
[[ $(md5sum "$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r3_real/task_5/pred_val.npz" | cut -d' ' -f1) == "$PRED_MD5" ]] || \
    fail "pred_val.npz của r3_real khác bản đã dùng để chọn ($PRED_MD5)"
python3 - "$SEL" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
ids, new = s['image_ids'], s['new_image_ids']
assert len(ids) == 1500 and len(new) == 500 and len(set(ids)) == 1500, (len(ids), len(new))
assert s['opts']['from'] == 'al_margin_k_B500_r3_real' and s['opts']['round'] == 3
print(f"   tập chọn: {len(ids)} ảnh ({len(new)} mới), strategy {s['strategy']}, từ {s['opts']['from']}")
PY
echo "  OK: sẵn sàng cho 'bash scripts/run_al.sh al_margin_k_B500_r4 --dry-run' rồi 'vast_train_and_sync.sh al_r4'"
