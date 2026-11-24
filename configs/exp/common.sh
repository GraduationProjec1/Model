TASK_CFG=$REPO/configs/tasks_100-4x25_seed0.json
TASK_DIR=$DATA/tasks/100-4x25_seed0

BATCH_SIZE=${BATCH_SIZE:-4}
COMMON_ARGS=(
    --task_config "$TASK_CFG" --n_classes 225
    --task_ann_dir "$TASK_DIR" --train_img_dir "$DATA/checkout_800" --test_img_dir "$DATA/checkout_800"
    --repo_name SenseTime/deformable-detr --accelerator gpu --n_gpus 1 --require_kernel 1
    --batch_size "$BATCH_SIZE" --eff_batch_size 4 --num_workers 4 --tf32 1 --pseudo_dedup_iou 0.5
    --lr 1e-4 --lr_old 1e-5 --eval_epochs 100
    --use_prompts 1 --num_prompts 100 --prompt_len 10 --local_query 1 --lambda_query 0.1
    --freeze backbone,encoder,decoder --new_params class_embed,prompts
)
STANDARD_ARGS=(--train_suffix _capped --epochs 6)

FINETUNE_ARGS=(--use_prompts 0 --local_query 0 --pseudo none --freeze "" --optim_groups detr
    --lr_backbone_names backbone)

FSA_ARGS=(--repo_name "$RUNS/FSA/task_1/hf_model")

E4_ARGS=("${FSA_ARGS[@]}" --augment 1 --pseudo_gt_iou 0.5)

