
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.utils.data import DataLoader
from datetime import timedelta

import pytorch_lightning as pl
from datasets.coco_eval import CocoEvaluator
from engine import local_trainer, Evaluator, needs_teacher
from inference import write_predictions
from augment import TrainAugment
from checkpointing import ResumeCheckpoint, PlannedTrainingStop, ResumeAudit, TrainingProgress, remove_resume_checkpoints, resume_path, restore_best_epoch
import runtime
from autocheckout.io import save_json
from autocheckout.runinfo import RunInfo

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything
from datasets.coco_hug import CocoDetection, task_info_coco,task_info_voc,task_info_rpc, create_task_json 
from models.image_processing_deformable_detr import DeformableDetrImageProcessor 


def get_args_parser():
    parser = argparse.ArgumentParser('MD-DETR', add_help=False)

    parser.add_argument('--lr', default=2e-4, type=float, 
                        help='Initial learning rate for training')
    parser.add_argument('--new_params', default="", type=str, 
                        help='New parameters to add to the model')
    parser.add_argument('--freeze', default="", type=str, 
                        help='Parameters to freeze during training')
    parser.add_argument('--lr_old', default=2e-5, type=float, 
                        help='Learning rate for older layers')
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+', 
                        help='Names of backbone layers to apply learning rate to')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='Learning rate for backbone layers')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+', 
                        help='Layers to which a different learning rate multiplier is applied')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float, 
                        help='Multiplier for the learning rate on linear projection layers')

    parser.add_argument('--batch_size', default=6, type=int, 
                        help='Batch size for training')
    parser.add_argument('--n_classes', default=80, type=int, 
                        help='Number of object classes')
    parser.add_argument('--weight_decay', default=1e-4, type=float, 
                        help='Weight decay for optimizer regularization')

    parser.add_argument('--epochs', default=51, type=int, 
                        help='Total number of training epochs')
    parser.add_argument('--eval_epochs', default=2, type=int, 
                        help='Number of epochs between evaluations')
    parser.add_argument('--print_freq', default=500, type=int, 
                        help='Frequency of printing training information (in steps)')
    parser.add_argument('--repo_name', default="SenseTime/deformable-detr", type=str, 
                        help='Repository name for the model')
    parser.add_argument('--backbone', default='', help='Explicit timm backbone; empty preserves detector config')
    parser.add_argument('--backbone_pretrained_file', default='', help='Local timm backbone weights for cross-backbone transfer')
    parser.add_argument('--model_config', default='', help='Bootstrap model config (checkpoint runtime wins on restart)')
    parser.add_argument('--image_size', type=int, default=None, help='Processor shortest edge; also longest unless overridden')
    parser.add_argument('--max_image_size', type=int, default=None, help='Processor longest edge')
    parser.add_argument('--stop_after_steps', type=int, default=0,
                        help='Save and exit 75 at this optimizer step; rerun with 0 to continue')
    parser.add_argument('--verify_resume', type=int, default=0,
                        help='Hash-check restored model, optimizer, scheduler and PDP memory')

    parser.add_argument('--lr_drop', default=40, type=int, 
                        help='Epoch to drop learning rate')
    parser.add_argument('--save_epochs', default=10, type=int, 
                        help='Interval of epochs between saving model checkpoints')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+', 
                        help='List of epochs to drop learning rate')
    
    parser.add_argument('--clip_max_norm', default=0.1, type=float, 
                        help='Maximum norm for gradient clipping')
    parser.add_argument('--sgd', action='store_true', 
                        help='Use SGD optimizer instead of AdamW')

    parser.add_argument('--n_gpus', default=4, type=int, 
                        help="Number of GPUs available for training")

    parser.add_argument("--num_imgs_viz", type=int, default=1, 
                        help="Number of images for visualization during training")

    parser.add_argument("--use_prompts", type=int, default=1, 
                        help="Enable or disable use of prompt memory")
    parser.add_argument("--prompt_len", type=int, default=10, 
                        help="Length of the prompt memory")
    parser.add_argument("--num_prompts", type=int, default=10, 
                        help="Number of prompts in the prompt pool")
    parser.add_argument("--num_prompt_layers", type=int, default=1, 
                        help="Number of layers to which prompts are applied")
    parser.add_argument("--prompt_key", type=int, default=1, 
                        help="Use learnable prompt key")
    parser.add_argument("--prompt_pool_sz", type=int, default=10, 
                        help="Size of the prompt pool")

    parser.add_argument('--with_box_refine', default=False, action='store_true', 
                        help='Enable box refinement in the decoder')
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help='Enable two-stage Deformable DETR')

    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Coefficient for classification cost in matching")
    parser.add_argument('--set_cost_bbox', default=5, type=float, 
                        help="Coefficient for L1 box cost in matching")
    parser.add_argument('--set_cost_giou', default=2, type=float, 
                        help="Coefficient for GIoU box cost in matching")

    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="Coefficient for mask loss")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="Coefficient for dice loss")
    parser.add_argument('--cls_loss_coef', default=2, type=float, 
                        help="Coefficient for classification loss")
    parser.add_argument('--prompt_loss_coef', default=1, type=float, 
                        help="Coefficient for prompt loss")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="Coefficient for bounding box loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="Coefficient for GIoU loss")
    parser.add_argument('--focal_alpha', default=0.25, type=float, 
                        help="Alpha parameter for focal loss")

    parser.add_argument('--output_dir', default='', 
                        help='Directory to save outputs')
    parser.add_argument('--device', default='cuda', 
                        help='Device for training (default is CUDA)')
    parser.add_argument('--seed', default=42, type=int, 
                        help='Random seed')
    parser.add_argument('--resume', default=0, type=int, 
                        help='Resume training from a specific checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', 
                        help='Start training from a specific epoch')
    parser.add_argument('--eval', action='store_true', 
                        help='Run evaluation mode only')
    parser.add_argument('--viz', action='store_true', 
                        help='Run visualization only mode')
    parser.add_argument('--eval_every', default=1, type=int, 
                        help='Evaluate the model every N epochs')
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='Number of workers for data loading')
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='Cache dataset in memory for faster training')

    parser.add_argument('--n_tasks', default=4, type=int, 
                        help='Number of tasks for continual learning setup')
    parser.add_argument('--lambda_query', default=0, type=float, 
                        help='Lambda parameter for query-based continual learning')
    parser.add_argument('--local_query', default=0, type=int, 
                        help='Flag to enable localalized query')
    parser.add_argument('--per_query_prompt', default=0, type=int,
                        help='A: every object query selects its own prompt mixture (no query_tf); needs --local_query 0')
    parser.add_argument('--prompt_cross_attn', default=0, type=int,
                        help='B: the prompt of each query corrects the offsets and weights of the decoder cross-attention')
    parser.add_argument('--xattn_alpha_max', default=0.5, type=float,
                        help='B: bound of the scale alpha = xattn_alpha_max * tanh(beta) of that correction')
    parser.add_argument('--query_proposals', default=0, type=int,
                        help='C1: K extra object queries from the top-K encoder proposals (needs --local_query 0)')
    parser.add_argument('--lambda_prop', default=1.0, type=float,
                        help='C1: weight of the auxiliary loss of the proposal head')
    parser.add_argument('--start_task', default=1, type=int, 
                        help='Task to start training from in continual learning')
    parser.add_argument('--task_id', default=0, type=int, 
                        help='Task ID for continual learning')
    parser.add_argument('--reset_optim', default=1, type=int, 
                        help='Reset optimizer between tasks in continual learning')
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int, 
                        help='Number of classes introduced in previous tasks')
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int, 
                        help='Number of new classes introduced in the current task')
    parser.add_argument('--mask_gradients', default=1, type=int, 
                        help='Flag to mask gradients during continual learning')

    parser.add_argument('--checkpoint_dir', default='', 
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_base', default='', 
                        help='Base checkpoint directory')
    parser.add_argument('--checkpoint_next', default='', 
                        help='Next checkpoint to load')

    parser.add_argument('--num_classes', default=81, type=int, 
                        help='Number of classes including background')
    parser.add_argument('--train_img_dir', default='/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017', type=str, 
                        help='Training images directory')
    parser.add_argument('--test_img_dir', default='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', type=str, 
                        help='Validation images directory')
    parser.add_argument('--load_task_model', default='/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/hug_demo/checkpoint00.pth', type=str, 
                        help='Checkpoint path for loading task-specific model')
    parser.add_argument('--task_ann_dir', default='', type=str, 
                        help='Directory for task annotations')
    parser.add_argument('--split_point', default=0, type=int, 
                        help='Point to split training data for task setup')
    parser.add_argument('--task_config', default='', type=str,
                        help='Task config JSON (configs/tasks_*.json); replaces the hard-coded COCO task split')
    parser.add_argument('--train_suffix', default='', type=str,
                        help="Suffix of the training files, e.g. '_capped' reads train_task_<t>_capped.json")
    parser.add_argument('--prev_ckpt', default='', type=str,
                        help='F8: task_final.pth to start from when --start_task > 1 comes from another run')
    parser.add_argument('--accelerator', default='gpu', type=str, help="Lightning accelerator ('gpu' or 'cpu')")
    parser.add_argument('--eff_batch_size', default=32, type=int,
                        help='Effective batch size; gradients are accumulated to reach it (original: 32)')
    parser.add_argument('--tf32', default=0, type=int,
                        help='1: TF32 matrix multiplications on Ampere/Ada GPUs (about 10%% faster on the L4, 29/09)')
    parser.add_argument('--ckpt_every_minutes', default=30, type=float,
                        help='R1: minutes between resume checkpoints within an epoch (also saved every epoch)')
    parser.add_argument('--predict_only', default=0, type=int,
                        help='F9/V1: skip training, load task_<t>/task_final.pth and rewrite pred_{val,test}.npz')
    parser.add_argument('--eval_resolution', default=0, type=int,
                        help='M6: with --predict_only 1, predict at this image size (shortest = longest edge) '
                             'instead of the size saved in the checkpoint; 0 = the saved size')
    parser.add_argument('--dump_logits', default=0, type=int,
                        help='N0: with --predict_only 1, also save the raw logits of the 100 most confident queries '
                             'of each image as logits_<split>.npz next to pred_<split>.npz')
    parser.add_argument('--task_calib', default='', type=str,
                        help='N1: with --predict_only 1, a per-task logit calibration (json of '
                             'tools/task_calib_search.py) applied before the sigmoid; empty = none')
    parser.add_argument('--select_best_epoch', default=0, type=int,
                        help='FULL-SYN F0.4: after training, keep the epoch with the best val_task AP50 (FSA); '
                             'task_final.pth, hf_model and the predictions come from it')
    parser.add_argument('--pred_splits', default='val,test', type=str,
                        help="FULL-SYN F0.3: comma-separated splits written as pred_<split>.npz after each task; 'val' "
                             'keeps the gate rounds away from test (written later with --predict_only 1)')

    parser.add_argument('--init_new_prompts', default=1, type=int,
                        help='F2: Gram-Schmidt init of the private prompts of each new task')
    parser.add_argument('--ddl_lambda', default=0.15, type=float,
                        help='F3: weight of the directional decoupled loss L_DDL (paper: 0.15; 0 disables it)')
    parser.add_argument('--query_loss_grad', default=1, type=int,
                        help='F4: let the query loss L_Q back-propagate into query_tf')
    parser.add_argument('--teacher_prompts', default=1, type=int,
                        help='F5: teacher runs two passes with the prompts of earlier tasks (as at inference)')
    parser.add_argument('--reuse_features', default=0, type=int,
                        help='P1 (docs/design/2026-11-22-toi-uu-2-luot): compute the backbone once for both passes '
                             '(and the teacher); in eval also reuse the encoder output. Same results; 0 = as before')
    parser.add_argument('--cache_pseudo', default='0', choices=['0', '1', 'auto'],
                        help='P2: cache the teacher pseudo-labels of each image (batches without padding) for the '
                             'later epochs of a task. auto = on unless --augment, --feature_kd, --logit_kd or --ppg_legacy; '
                             '1 = on, error with those flags; 0 = as before')
    parser.add_argument('--pseudo', default='ppg', choices=['ppg', 'threshold', 'none'],
                        help='Pseudo-labels of earlier classes: PPG (paper), fixed threshold only, or none')
    parser.add_argument('--ppg_legacy', default=0, type=int,
                        help='F6: 1 = original candidate selection (top bg_thres_topk (query, class) pairs)')
    parser.add_argument('--pseudo_topk', default=50, type=int,
                        help='F6: teacher queries considered per image (ranked by best earlier-class score)')
    parser.add_argument('--pseudo_thresh_high', default=0.5, type=float, help='PPG tau_h (paper: 0.5)')
    parser.add_argument('--pseudo_thresh_low', default=0.2, type=float, help='PPG tau_l (paper: 0.2)')
    parser.add_argument('--prototype_sim_thresh', default=0.5, type=float, help='PPG theta_s (paper: 0.5)')
    parser.add_argument('--pseudo_gt_iou', default=0.0, type=float,
                        help='I3: drop pseudo-labels overlapping a current-task GT box with IoU >= this (0 = off)')
    parser.add_argument('--pseudo_dedup_iou', default=0.0, type=float,
                        help='F14: at most one pseudo-label per object, class-agnostic NMS at this IoU (0 = off, '
                             'as the paper; duplicates compound over the tasks)')
    parser.add_argument('--prototype_nearest', default=0, type=int,
                        help='I4: prototype-verified candidates must have their own class as nearest prototype')
    parser.add_argument('--freeze_shared_after_task1', default=0, type=int,
                        help='I5: from task 2 on, also freeze input_proj, query_tf, query_position_embeddings, '
                             'reference_points, level_embed and bbox_embed')
    parser.add_argument('--tune_after_task1', default='', type=str,
                        help='M5: from task 2 on, train the parameters under these name parts although --freeze '
                             'covers them (e.g. stages_3,encoder: last ConvNeXt stage and encoder), at lr_old')
    parser.add_argument('--train_rows', default='', type=str,
                        help='Update-model F1/F2: JSON list of labels whose class_embed rows this task trains, or '
                             '"all" (every seen class); the last row always trains and every other row keeps its '
                             'loaded values. Empty = the rows of the current task (original behaviour)')
    parser.add_argument('--feature_kd', default=0.0, type=float,
                        help='M5: weight of the MSE between the encoder output of the model and of the teacher '
                             '(previous task) on the same images, from task 2 on (0 = off)')
    parser.add_argument('--logit_kd', default=0.0, type=float,
                        help='N2: weight of the BCE between the student logits and the teacher scores (earlier classes) '
                             'of the queries the teacher is confident about, from task 2 on (0 = off)')
    parser.add_argument('--logit_kd_box', default=0.0, type=float,
                        help='N2: weight of the L1 between the student and teacher boxes of those queries (0 = off)')
    parser.add_argument('--logit_kd_topk', default=10, type=int,
                        help='N2: at most this many queries per image (above mean + std of the teacher scores)')
    parser.add_argument('--group_neg_weight', default=1.0, type=float,
                        help='N3: focal-loss weight (>= 1) of the negative logits of a matched query on the classes '
                             'that share the supercategory of its target (--task_config); 1 = off')
    parser.add_argument('--optim_groups', default='auto', choices=['auto', 'prompt', 'detr'],
                        help="B1: 'detr' = full fine-tuning learning rates; 'auto' = original rule")
    parser.add_argument('--joint', default=0, type=int,
                        help='B1/E0: one training on every data class of tasks 1..n_tasks (train_joint*.json), '
                             'written as task_<n_tasks>')
    parser.add_argument('--save_hf', default=0, type=int,
                        help='B1/I1: also save the final model and processor in Hugging Face format '
                             '(task_<t>/hf_model), usable as --repo_name (FSA)')
    parser.add_argument('--pred_ann_dir', default='', type=str,
                        help='Folder of val_full.json/test_full.json for the predictions (default: --task_ann_dir)')
    parser.add_argument('--augment', default=0, type=int,
                        help='I2: training augmentation (90-degree rotations, colour jitter, shortest edge 640-800)')
    parser.add_argument('--augment_flip', default=0, type=int, help='I2: also random horizontal flips')
    parser.add_argument('--tune_old_private', default=0, type=int,
                        help='round 3: the private prompts of earlier tasks train too (default 0: frozen, as PDP)')
    parser.add_argument('--use_shared', default=1, type=int, help='B2: use the shared prompt pool')
    parser.add_argument('--use_private', default=1, type=int, help='B2: use the private (per-class) prompt pool')
    parser.add_argument('--prior_init_classifier', default=1, type=int,
                        help='F13: focal-loss prior (p=0.01) for a classifier not loaded from the checkpoint')
    parser.add_argument('--proto_correct_only', default=1, type=int,
                        help='F7: prototypes only from queries whose predicted class is the GT class')
    parser.add_argument('--proto_margin_lambda', default=0.0, type=float,
                        help='M1: weight of the prototype margin loss on the matched queries; 0 = off')
    parser.add_argument('--proto_margin', default=0.1, type=float, help='M1: cosine margin of the true class')
    parser.add_argument('--proto_temp', default=0.1, type=float, help='M1: temperature of the margin loss')
    parser.add_argument('--proto_topk', default=20, type=int, help='M1: hardest other classes in the loss')
    parser.add_argument('--proto_anchor', default='prototype', choices=['prototype', 'class_embed'],
                        help='M1: anchors fixed at fit start, the class prototypes of the checkpoint or the '
                             'classifier rows (K0 verdict, tools/proto_check.py)')
    parser.add_argument('--shuffle', default=1, type=int,
                        help='F12: shuffle the training data every epoch')
    parser.add_argument('--require_kernel', default=0, type=int,
                        help='F11: fail if the CUDA kernel of deformable attention cannot be loaded')

    parser.add_argument('--bbox_thresh', default=0.3, type=float, 
                        help='Bounding box threshold for positive detections')
    parser.add_argument('--bg_thres', default=0.7, type=float, 
                        help='Threshold for considering a detection as background')
    parser.add_argument('--bg_thres_topk', default=5, type=int, 
                        help='Top-K background detections to consider')

    parser.add_argument('--big_pretrained', default="", type=str, 
                        help='Path to a larger pretrained model for initialization')
    
    return parser

def make_train_loader(dataset, args):
    return DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=args.batch_size,
                      num_workers=args.num_workers, pin_memory=True, shuffle=bool(args.shuffle))

def check_kernel(args):
    import models.modeling_deformable_detr as detr_module

    if detr_module.MultiScaleDeformableAttention is not None:
        print('Multi-scale deformable attention: CUDA kernel')
    else:
        message = f'Multi-scale deformable attention: PyTorch fallback ({detr_module.KERNEL_LOAD_ERROR})'
        if args.require_kernel:
            raise RuntimeError(message)
        print(message)

def setup_task_info(args):
    """Fill args.task_map, args.task_label2name and args.task_num_classes."""
    if args.task_config:
        args.task_map, args.task_label2name = task_info_rpc(args.task_config)
        args.task_num_classes = [args.task_map[task_id][2] for task_id in sorted(args.task_map)]
        n_slots = sum(args.task_num_classes)
        if args.n_classes != n_slots + 1:
            raise ValueError(f'--n_classes must be {n_slots + 1} (slots + 1) for {args.task_config}, got {args.n_classes}')
    else:
        args.task_map, args.task_label2name =  task_info_coco(split_point=args.split_point)
        args.task_num_classes =  [args.task_map[task_id][2] for task_id in range(1, args.n_tasks + 1)]
    args.task_label2name[args.n_classes-1] = "BG"

def task_dir(output_root, task_id):
    return os.path.join(output_root, f'task_{task_id}')

def make_pl_trainer(args):
    return pl.Trainer(devices=args.n_gpus, accelerator=args.accelerator, max_epochs=args.epochs,
                      gradient_clip_val=args.clip_max_norm, precision='32-true',
                      accumulate_grad_batches=runtime.accumulation_steps(args),
                      check_val_every_n_epoch=args.eval_epochs, enable_checkpointing=False,
                      callbacks=[ResumeAudit(bool(args.verify_resume)),
                                 TrainingProgress(args.output_dir),
                                 ResumeCheckpoint(args.output_dir, args.ckpt_every_minutes, args.stop_after_steps)],
                      log_every_n_steps=args.print_freq, num_sanity_val_steps=0,
                      logger=CSVLogger(save_dir=args.output_dir, name="lightning_logs"))

def write_task_predictions(args, trainer, task_id, processor):
    """F9/V1: predictions of the task's final model on the full files of --pred_splits (pred_<split>.npz)."""
    device = torch.device('cuda' if args.accelerator == 'gpu' else 'cpu')
    trainer.model.to(device)
    ann_dir = args.pred_ann_dir or args.task_ann_dir
    for split in args.pred_splits.split(','):
        write_predictions(trainer.model, args, ann_file=os.path.join(ann_dir, f'{split}_full.json'),
                          out_file=os.path.join(args.output_dir, f'pred_{split}.npz'), task_id=task_id,
                          seen_classes=trainer.seen_classes, split=split, processor=processor, device=device)

def run_task(args, task_id, output_root, processor=None):
    """Train task `task_id` (F8), then write its predictions (F9).

    Writes <output_root>/task_<t>/task_final.pth and pred_{val,test}.npz. With --predict_only 1 the
    task is not trained: task_final.pth is loaded and only the predictions are (re)written.
    """
    args.output_dir = task_dir(output_root, task_id)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_file = open(os.path.join(args.output_dir, 'train.log'), 'a', buffering=1)
    print('Logging: args ', args, file=args.log_file)
    args.task = str(task_id)
    final_path = os.path.join(args.output_dir, 'task_final.pth')
    ckpt_path = resume_path(args.output_dir) if not args.predict_only else None
    prev_ckpt = (args.prev_ckpt if task_id == args.start_task and args.prev_ckpt else
                 os.path.join(task_dir(output_root, task_id-1), 'task_final.pth')) if task_id > 1 and not args.joint else None
    construction_checkpoint = final_path if args.predict_only else ckpt_path or prev_ckpt
    args._runtime_snapshot = runtime.checkpoint_runtime(construction_checkpoint)
    runtime.validate_runtime(args, args._runtime_snapshot)
    processor = processor or runtime.processor(args, DeformableDetrImageProcessor)
    args._processor = processor
    train_name = f'train_joint{args.train_suffix}.json' if args.joint else f'train_task_{task_id}{args.train_suffix}.json'
    tr_ann = os.path.join(args.task_ann_dir, train_name)
    val_ann = os.path.join(args.task_ann_dir, 'val_full.json' if args.joint else f'val_task_{task_id}.json')

    run_info = RunInfo(os.path.join(args.output_dir, 'run_info.json'))
    run_info.start(vars(args), repo_root=REPO_ROOT, files={
        'task_config': args.task_config, 'train': None if args.predict_only else tr_ann, 'val_task': val_ann,
        'val_full': os.path.join(args.pred_ann_dir or args.task_ann_dir, 'val_full.json'),
        'test_full': os.path.join(args.pred_ann_dir or args.task_ann_dir, 'test_full.json'),
        'prev_ckpt': args.prev_ckpt if task_id == args.start_task else None})

    val_dataset = CocoDetection(img_folder=args.test_img_dir, ann_file=val_ann, processor=processor)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    train_dataloader = None
    if not args.predict_only:
        augment = TrainAugment(flip=bool(args.augment_flip)) if args.augment else None
        train_dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann, processor=processor,
                                      augment=augment)
        train_dataloader = make_train_loader(train_dataset, args)

    coco_evaluator = CocoEvaluator(val_dataset.coco, args.iou_types)
    local_evaluator = Evaluator(processor=processor, test_dataset=val_dataset, test_dataloader=val_dataloader,
                                coco_evaluator=coco_evaluator, args=args, task_label2name=args.task_label2name,
                                task_name='cur')
    trainer = local_trainer(train_loader=train_dataloader, val_loader=val_dataloader, test_dataset=val_dataset,
                            args=args, local_evaluator=local_evaluator, task_id=task_id)
    runtime.save_runtime(args.output_dir, trainer.runtime, trainer.loading_report)
    budget = dict(physical_batch=args.batch_size, effective_batch=args.eff_batch_size,
                  accumulate_grad_batches=runtime.accumulation_steps(args),
                  train_batches=0 if train_dataloader is None else len(train_dataloader))

    if args.use_prompts:
        print('previous task : ', trainer.model.model.prompts.task_count, file=args.log_file)
        trainer.model.model.prompts.set_task_id(task_id-1)
        print('current task : ', trainer.model.model.prompts.task_count, file=args.log_file)

    if args.predict_only:
        trainer.resume(final_path)
    else:
        if task_id > 1 and not args.joint:
            trainer.resume(prev_ckpt)
        else:
            trainer.resume()

        if needs_teacher(args, task_id):
            trainer.set_teacher()

        if args.use_prompts and args.init_new_prompts:
            trainer.model.model.prompts.init_task_prompts()

        if ckpt_path:
            print(f'Resuming task {task_id} from {ckpt_path}', file=args.log_file)
        lightning = make_pl_trainer(args)
        try:
            lightning.fit(trainer, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
            if getattr(lightning, 'interrupted', False):
                raise PlannedTrainingStop('Lightning interrupted; continue from the last saved checkpoint')
        except PlannedTrainingStop:
            run_info.finish(mode='interrupted', resumed_from=ckpt_path,
                            optimizer_steps=lightning.global_step,
                            resume_verification=getattr(trainer, 'resume_verification', {}), **budget)
            args.log_file.close()
            raise SystemExit(75)
        except BaseException as error:
            run_info.finish(mode='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error_type=type(error).__name__, error=str(error), resumed_from=ckpt_path,
                            optimizer_steps=lightning.global_step, **budget)
            args.log_file.close()
            raise
        budget.update(optimizer_steps=lightning.global_step,
                      optimizer_steps_this_session=lightning.global_step-trainer.resume_verification['restored_step'],
                      resume_verification=trainer.resume_verification)
        if args.select_best_epoch:
            budget['selected_epoch'] = restore_best_epoch(args.output_dir, trainer.model)
            print(f"Selected epoch {budget['selected_epoch']['epoch']} (val AP50 {budget['selected_epoch']['AP50']:.4f})",
                  file=args.log_file)
        trainer.save_task_final(final_path)
        remove_resume_checkpoints(args.output_dir)
        if args.save_hf:
            hf_dir = os.path.join(args.output_dir, 'hf_model')
            trainer.model.save_pretrained(hf_dir)
            processor.save_pretrained(hf_dir)
            if args.select_best_epoch:
                save_json(os.path.join(hf_dir, 'epoch_selection.json'), budget['selected_epoch'], indent=2)

    write_task_predictions(args, trainer, task_id, processor)
    run_info.finish(mode='predict' if args.predict_only else 'train', resumed_from=ckpt_path,
                    backbone=trainer.model.config.backbone, processor_size=processor.size, **budget)
    args.log_file.close()

def main(args):
    if args.eval:
        raise SystemExit('--eval is replaced by --predict_only 1 (predictions are written after each task, F9)')
    if args.eval_resolution and not args.predict_only:
        raise SystemExit('--eval_resolution only changes predictions: use it with --predict_only 1 (M6)')
    if args.dump_logits and not args.predict_only:
        raise SystemExit('--dump_logits only writes predictions: use it with --predict_only 1 (N0)')
    if args.task_calib and not args.predict_only:
        raise SystemExit('--task_calib only changes predictions: use it with --predict_only 1 (N1)')
    if args.task_calib and not os.path.isfile(args.task_calib):
        raise SystemExit(f'--task_calib {args.task_calib}: no such file')
    splits = args.pred_splits.split(',')
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'val', 'test'}:
        raise SystemExit(f'--pred_splits must list val and/or test once each, got {args.pred_splits!r}')
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high' if args.tf32 else 'highest')
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    check_kernel(args)
    args.iou_types = ['bbox']
    setup_task_info(args)
    runtime.accumulation_steps(args)

    if args.joint:
        names = [name for t in range(1, args.n_tasks + 1) for name in args.task_map[t][0]]
        args.task_map[args.n_tasks] = (names, 0, len(names))
        args.start_task = args.n_tasks

    output_root = args.output_dir
    for task_id in range(args.start_task, args.n_tasks+1):
        run_task(args, task_id, output_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
