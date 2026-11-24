#!/usr/bin/env bash
set -euo pipefail
DATA=${DATA:-/data/rpc}
RAW=$DATA/raw
URL=https://www.kaggle.com/api/v1/datasets/download/diyer22/retail-product-checkout-dataset
ZIP=retail-product-checkout-dataset.zip

if [[ -n ${KAGGLE_API_TOKEN:-} ]]; then
    AUTH=(-H "Authorization: Bearer $KAGGLE_API_TOKEN")
elif [[ -f ~/.kaggle/access_token ]]; then
    AUTH=(-H "Authorization: Bearer $(cat ~/.kaggle/access_token)")
elif [[ -f ~/.kaggle/kaggle.json ]]; then
    AUTH=(-u "$(python3 -c 'import json, os; d = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"))); print(d["username"] + ":" + d["key"])')")
else
    echo "no Kaggle credentials: put the API token in ~/.kaggle/access_token (see the guide, 6.1)" >&2
    exit 1
fi

mkdir -p "$RAW"
cd "$RAW"
if [[ ! -d retail_product_checkout/test2019 ]]; then
    if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
        curl -fL --retry 5 -C - "${AUTH[@]}" -o "$ZIP" "$URL"
        unzip -tq "$ZIP"
    fi
    unzip -q -o "$ZIP" \
        'retail_product_checkout/val2019/*' 'retail_product_checkout/test2019/*' \
        'retail_product_checkout/instances_val2019.json' 'retail_product_checkout/instances_test2019.json'
fi
n_val=$(ls retail_product_checkout/val2019 | wc -l)
n_test=$(ls retail_product_checkout/test2019 | wc -l)
echo "val2019:  $n_val images (expected 6000)"
echo "test2019: $n_test images (expected 24000)"
if [[ $n_val -eq 6000 && $n_test -eq 24000 ]]; then
    rm -f "$ZIP"
fi
df -h /
