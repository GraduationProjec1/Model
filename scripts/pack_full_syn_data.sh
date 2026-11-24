#!/usr/bin/env bash
set -euo pipefail
RELEASE=${1:-rpc_synth_v2_v1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=full_syn_${RELEASE}_$(date +%Y%m%d)
RECEIPTS=data/packages/$NAME
PATHS=(
    data/processed/$RELEASE/images
    data/processed/$RELEASE/tasks
    data/processed/$RELEASE/manifest.json
    data/processed/$RELEASE/task_config.json
    data/processed/$RELEASE/verify_report.json
    data/processed/rpc_eval_official_v1
    data/archive/val2019
    data/archive/test2019
    data/pretrained/hf-cache
)

cd "$REPO"
mkdir -p "$RECEIPTS"
if [[ ! -s $RECEIPTS/SHA256SUMS ]]; then
    find "${PATHS[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$RECEIPTS/SHA256SUMS"
    echo "$NAME: $(wc -l < "$RECEIPTS/SHA256SUMS") files, $(du -sch "${PATHS[@]}" | tail -1 | cut -f1), git $(git rev-parse HEAD)" \
        > "$RECEIPTS/PACKAGE.txt"
fi
cat "$RECEIPTS/PACKAGE.txt"

tar -cf - "$RECEIPTS/SHA256SUMS" "$RECEIPTS/PACKAGE.txt" "${PATHS[@]}" \
    | tee "data/packages/$NAME.tar.part" | md5sum | cut -d' ' -f1 > "$RECEIPTS/tar.md5"
mv "data/packages/$NAME.tar.part" "data/packages/$NAME.tar"
echo "wrote $REPO/data/packages/$NAME.tar ($(du -h "data/packages/$NAME.tar" | cut -f1), md5 $(cat "$RECEIPTS/tar.md5"))"
cat <<EOF
Upload (DEST e.g. drive:AutoCheckout/$(date +%F)); Drive's md5 of the tar must be $(cat "$RECEIPTS/tar.md5"):
  rclone copy -P --drive-chunk-size 64M $REPO/data/packages/$NAME.tar DEST/
  rclone md5sum DEST/$NAME.tar
EOF
