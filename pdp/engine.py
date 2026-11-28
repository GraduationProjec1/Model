import os
import sys
import json
import math
import pdb
import tqdm
import torch
import utils
import numpy as np
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import matplotlib
from sklearn.cluster import KMeans
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets.coco_eval import CocoEvaluator
import torch.nn.functional as F
from datasets.coco_hug import CocoDetection, task_info_coco, create_task_json
from models.image_processing_deformable_detr import DeformableDetrImageProcessor 
from models.configuration_deformable_detr import DeformableDetrConfig
from models.modeling_deformable_detr import DeformableDetrForObjectDetection
from ppg import prototype_matrix, select_candidates, select_pseudo_labels
import proto_margin
from inference import predict_batch
from two_pass import prompted_outputs
from checkpointing import record_epoch
import runtime
from autocheckout.io import save_json
from autocheckout.model_acceptance import now
from autocheckout.taskcfg import TaskConfig

PRIOR_PROB = 0.01


def classifier_from_checkpoint(repo_name, num_labels):
	"""True if --repo_name holds a classifier of this size, i.e. from_pretrained loads it as it is."""
	if not repo_name:
		return False
	from models.configuration_deformable_detr import DeformableDetrConfig as Config
	return Config.from_pretrained(repo_name).num_labels == num_labels


def backbone_training_reason(model, args):
	"""P1: why a backbone parameter would train in some task of this run, or None when it stays frozen.

	Mirrors the freezing of local_trainer.resume(): a parameter is frozen when a part of its name is in --freeze,
	unless (from task 2 on) a part is in --tune_after_task1 (M5)."""
	freeze = set(args.freeze.split(',')) if args.freeze else set()
	tune = set(args.tune_after_task1.split(',')) if args.tune_after_task1 else set()
	later_tasks = args.n_tasks > 1 or args.start_task > 1
	for name, _ in model.named_parameters():
		parts = set(name.split('.'))
		if 'backbone' not in parts:
			continue
		if not parts & freeze:
			return f'{name} trains (--freeze {args.freeze!r})'
		if later_tasks and parts & tune:
			return f'{name} trains from task 2 on (--tune_after_task1 {args.tune_after_task1!r})'
	return None


def pseudo_cache_enabled(args):
	"""P2: whether the teacher pseudo-labels are cached by image_id (--cache_pseudo). They only repeat across epochs
	when the image, the teacher and the targets do: not with --augment, nor with --feature_kd or --logit_kd (which need
	the teacher's outputs every step), nor with the original pseudo-labelling (--ppg_legacy)."""
	if args.cache_pseudo == '0':
		return False
	checks = (('--augment', args.augment), ('--feature_kd', args.feature_kd > 0), ('--ppg_legacy', args.ppg_legacy),
			  ('--logit_kd', getattr(args, 'logit_kd', 0) > 0))
	blockers = [flag for flag, on in checks if on]
	if blockers and args.cache_pseudo == '1':
		raise ValueError(f'--cache_pseudo 1 cannot be used with {", ".join(blockers)}; use auto or 0')
	state = f'off ({", ".join(blockers)})' if blockers else 'on'
	print(f'P2: pseudo-label cache {state} (--cache_pseudo {args.cache_pseudo})', file=args.log_file)
	return not blockers


def needs_teacher(args, task_id: int) -> bool:
	"""Whether training ``task_id`` requires a frozen teacher (for pseudo-labels, feature KD, or logit KD)."""
	if task_id <= 1 or getattr(args, 'joint', False):
		return False
	return (
		getattr(args, 'pseudo', 'ppg') != 'none'
		or getattr(args, 'logit_kd', 0) > 0
		or getattr(args, 'feature_kd', 0) > 0
	)


def validate_logit_kd(args):
	"""N2: --logit_kd and --logit_kd_box are weights (>= 0), --logit_kd_topk at least one query."""
	if getattr(args, 'logit_kd', 0) < 0 or getattr(args, 'logit_kd_box', 0) < 0:
		raise ValueError('--logit_kd and --logit_kd_box must be >= 0')
	if getattr(args, 'logit_kd_topk', 10) < 1:
		raise ValueError(f'--logit_kd_topk must be >= 1, got {args.logit_kd_topk}')


def select_kd_queries(teacher_logits, seen_old, topk):
	"""N2 (ERD): [B, Q] bool, the queries whose best earlier-class teacher score is above mean + std of the image,
	at most topk per image (the most confident)."""
	best = teacher_logits[..., :seen_old].sigmoid().max(dim=-1).values
	above = best > best.mean(dim=1, keepdim=True) + best.std(dim=1, keepdim=True)
	top = torch.zeros_like(above)
	top.scatter_(1, best.topk(min(topk, best.shape[1]), dim=1).indices, True)
	return above & top


def logit_kd_loss(student_logits, teacher_logits, mask, seen_old):
	"""N2: BCE of the student logits of the selected queries against the teacher scores, earlier classes only,
	averaged over the selected queries and classes (0 when none is selected)."""
	if not mask.any():
		return student_logits.sum() * 0
	return F.binary_cross_entropy_with_logits(student_logits[mask][:, :seen_old],
											  teacher_logits[mask][:, :seen_old].sigmoid())


def box_kd_loss(student_boxes, teacher_boxes, mask):
	"""N2: L1 between the student and teacher boxes of the selected queries (0 when none is selected)."""
	if not mask.any():
		return student_boxes.sum() * 0
	return F.l1_loss(student_boxes[mask], teacher_boxes[mask])


def same_group_matrix(args):
	"""N3 (--group_neg_weight): [n_classes, n_classes] bool, True where two different classes share a supercategory
	in --task_config (classes without one are in no group); None when the flag is off (1.0)."""
	weight = getattr(args, 'group_neg_weight', 1.0)
	if weight < 1:
		raise ValueError(f'--group_neg_weight must be >= 1 (1 = off), got {weight}')
	if weight == 1:
		return None
	if not args.task_config:
		raise ValueError('--group_neg_weight needs --task_config (the supercategory of each class)')
	cfg = TaskConfig.load(args.task_config)
	names = [None] * args.n_classes
	for task in cfg.tasks:
		for c in task.classes:
			names[c.label] = c.supercategory
	same = torch.tensor([[a is not None and a == b for b in names] for a in names], dtype=torch.bool)
	same.fill_diagonal_(False)
	return same


def frozen_row_mask(train_rows, n_classes, seen_classes):
	"""--train_rows (update-model F1/F2): None when empty (the original rule, rows of earlier tasks frozen), else a
	bool tensor over the classifier rows, True = kept as loaded. The last row always trains."""
	if not train_rows:
		return None
	if train_rows == 'all':
		rows = list(range(seen_classes))
	else:
		with open(train_rows) as f:
			rows = [int(r) for r in json.load(f)]
	if not rows or not all(0 <= r < seen_classes for r in rows):
		raise ValueError(f'--train_rows must list labels in 0..{seen_classes - 1} (seen classes), got {rows}')
	frozen = torch.ones(n_classes, dtype=torch.bool)
	frozen[rows] = False
	frozen[-1] = False
	return frozen


SHARED_AFTER_TASK1 = ['input_proj', 'query_tf', 'query_position_embeddings', 'reference_points', 'level_embed', 'bbox_embed']

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, local_evaluator, task_id, eval_mode=False):
		super().__init__()


		if not hasattr(args, 'pseudo_thresh'):
			args.pseudo_thresh = 0.3  
		if not hasattr(args, 'use_distillation'):
			args.use_distillation = True  
		if not hasattr(args, 'bg_thres_topk'):
			args.bg_thres_topk = 5  

		detr_config = runtime.model_config(args, DeformableDetrConfig)
		detr_config.num_labels = args.n_classes
		detr_config.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
		detr_config.CUR_INTRODUCED_CLS = args.task_map[task_id][2]
		seen_classes = detr_config.PREV_INTRODUCED_CLS + detr_config.CUR_INTRODUCED_CLS
		self.__dict__['teacher'] = None
		self._old_prototypes = None
		validate_logit_kd(args)
		self.use_pseudo_cache = pseudo_cache_enabled(args)
		self.pseudo_cache = {}
		self.pseudo_cache_hits = 0
		detr_config.use_prompts = args.use_prompts
		detr_config.n_tasks = args.n_tasks
		detr_config.num_prompts = args.num_prompts
		detr_config.prompt_len = args.prompt_len
		detr_config.local_query = args.local_query
		detr_config.per_query_prompt = getattr(args, 'per_query_prompt', 0)
		detr_config.prompt_cross_attn = getattr(args, 'prompt_cross_attn', 0)
		detr_config.xattn_alpha_max = getattr(args, 'xattn_alpha_max', 0.5)
		detr_config.query_proposals = getattr(args, 'query_proposals', 0)
		detr_config.lambda_prop = getattr(args, 'lambda_prop', 1.0)
		detr_config.decoder_lora_rank = getattr(args, 'decoder_lora_rank', 0)
		detr_config.decoder_lora_alpha = getattr(args, 'decoder_lora_alpha', 16.0)
		detr_config.task_num_classes = args.task_num_classes
		detr_config.use_shared_pool = args.use_shared
		detr_config.use_private_pool = args.use_private
		detr_config.tune_old_private = bool(getattr(args, 'tune_old_private', 0))

		self.invalid_cls_logits = list(range(seen_classes, args.n_classes-1))
		self.seen_classes = seen_classes

		self.model, self.loading_report = runtime.build_model(args, detr_config, DeformableDetrForObjectDetection)
		self.model.same_group = same_group_matrix(args)
		self.model.group_neg_weight = getattr(args, 'group_neg_weight', 1.0)
		self.processor = getattr(args, '_processor', None) or runtime.processor(args, DeformableDetrImageProcessor)
		self.runtime = runtime.runtime_record(args, self.model, self.processor)

		if args.prior_init_classifier and not self.loading_report['classifier_loaded']:
			for head in self.model.class_embed:
				nn.init.constant_(head.bias, -math.log((1 - PRIOR_PROB) / PRIOR_PROB))
		
		self.task_id = task_id
		self.lr = args.lr
		self.lr_backbone = args.lr_backbone
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.args = args
		self.eval_mode = eval_mode
		self.print_count = 0
		self.evaluator = local_evaluator
		self.evaluator.model = self.model
		self.evaluator.invalid_cls_logits = self.invalid_cls_logits
		self.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
		self.frozen_rows = frozen_row_mask(getattr(args, 'train_rows', ''), args.n_classes, seen_classes)
		self._frozen_values = None


		self.prototype_cache_capacity = 100  
		self.total_classes = args.n_classes - 1 
		self.class_query_cache = {
			cls_idx: [] for cls_idx in range(self.total_classes)
		}
		self.class_prototypes = {
			cls_idx: torch.zeros(self.model.config.d_model, device=self.device)
			for cls_idx in range(self.total_classes)
		}
		self.class_cache_count = {cls_idx: 0 for cls_idx in range(self.total_classes)}
		
		self.prototype_update_frequency = 2  
		self.batch_counter = 0
		self.log_frequency = 5000  
		self.update_prototypes_last_epoch_only = getattr(args, 'update_prototypes_last_epoch_only', True)  


	def forward(self, pixel_values, pixel_mask):
		outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		return outputs

	def set_teacher(self):
		"""Freeze a copy of the current weights, i.e. the model at the end of the previous task (F5).

		Call after loading the previous task's weights and before initialising the prompts of the
		new task. The teacher only uses the prompts of earlier tasks (task_count = task_id - 2). It
		is stored outside the module tree on purpose: it must not be trained, counted as a
		parameter or saved in resume checkpoints (it is rebuilt from the previous task's weights).
		"""
		teacher = deepcopy(self.model)
		teacher.eval()
		for param in teacher.parameters():
			param.requires_grad = False
		if self.args.use_prompts:
			teacher.model.prompts.set_task_id(self.task_id - 2)
		self.__dict__['teacher'] = teacher
		self.pseudo_cache = {}

	def margin_anchors(self):
		"""M1: anchors of the seen classes, fixed for the whole run (a snapshot: the prototypes that the last epoch
		updates are not used), and the anchor row of each class."""
		seen = self.seen_classes
		if self.args.proto_anchor == 'class_embed':
			rows = self.model.class_embed[-1].weight
			vectors, valid = {c: rows[c] for c in range(seen)}, list(range(seen))
		else:
			vectors = self.class_prototypes
			valid = [c for c in range(seen) if self.class_cache_count.get(c, 0) > 0]
		if not valid:
			raise ValueError('--proto_margin_lambda > 0 needs class prototypes in the checkpoint (none of the '
							 f'{seen} seen classes has one); use --proto_anchor class_embed')
		anchors, slot = proto_margin.anchor_matrix(vectors, valid, seen)
		print(f'M1: {len(valid)}/{seen} classes anchored ({self.args.proto_anchor})', file=self.args.log_file)
		return anchors.to(self.device), slot.to(self.device)

	def prototype_margin_loss(self, outputs, labels):
		"""M1: margin loss of the decoder output of the queries the matcher pairs with a ground truth."""
		anchors, slot = self._margin_anchors
		without_aux = {k: v for k, v in outputs.items() if k not in ('auxiliary_outputs', 'enc_outputs')}
		with torch.no_grad():
			indices = self.model.matcher(without_aux, labels)
		features = torch.cat([outputs.last_hidden_state[b, q] for b, (q, _) in enumerate(indices)])
		classes = torch.cat([labels[b]['class_labels'][t.to(labels[b]['class_labels'].device)]
							 for b, (_, t) in enumerate(indices)])
		return proto_margin.margin_loss(features, classes, anchors, slot, margin=self.args.proto_margin,
										temp=self.args.proto_temp, topk=self.args.proto_topk)

	def on_fit_start(self):
		if self.teacher is not None:
			self.teacher.to(self.device)
		self._old_prototypes = None
		self._margin_anchors = self.margin_anchors() if getattr(self.args, 'proto_margin_lambda', 0) > 0 else None
		if self.frozen_rows is not None:
			self._frozen_values = [(h.weight[self.frozen_rows.to(h.weight.device)].detach().clone(),
									h.bias[self.frozen_rows.to(h.bias.device)].detach().clone())
								   for h in self.model.class_embed]

	@torch.no_grad()
	def teacher_outputs(self, pixel_values, pixel_mask, backbone_features=None):
		"""Teacher inference as at evaluation time: a first pass gives the query, the second pass
		uses the prompts (F5). --teacher_prompts 0 restores the original single pass without prompts."""
		self.teacher_forward_calls = getattr(self, 'teacher_forward_calls', 0) + 1
		self.teacher.eval()
		return prompted_outputs(
			self.teacher, pixel_values, pixel_mask,
			use_prompts=bool(self.args.use_prompts and self.args.teacher_prompts),
			local_query=self.args.local_query, reuse_features=bool(self.args.reuse_features),
			backbone_features=backbone_features, first_task_id=self.task_id - 1, second_task_id=self.task_id - 1)

	def update_class_cache(self, cls_idx, new_query):

		cache = self.class_query_cache[cls_idx]

		new_query_cpu = new_query.detach().cpu() 
		if len(cache) < self.prototype_cache_capacity:

			cache.append(new_query_cpu)
			self.class_cache_count[cls_idx] += 1
		else:
			cache.pop(0)
			cache.append(new_query_cpu)

	def compute_class_prototypes(self, cls_idx):

		cache = self.class_query_cache[cls_idx]
		if len(cache) == 0:

			return torch.zeros(self.model.config.d_model, device=self.device)

		device = self.device

		cache_on_device = [tensor.to(device, non_blocking=True) for tensor in cache]
		cache_tensor = torch.stack(cache_on_device, dim=0)
		prototype = cache_tensor.mean(dim=0)
		return prototype

	def print_prototype_space_stats(self, prefix=""):

		print(f"\n{prefix}======", file=getattr(self.args, 'log_file', None))
		print(f"{prefix}ID: {self.task_id}, CLASS: {self.PREV_INTRODUCED_CLS}", file=getattr(self.args, 'log_file', None))
		print(f"{prefix}: {self.total_classes}, NUM: {self.prototype_cache_capacity}", file=getattr(self.args, 'log_file', None))

		active_classes = []
		empty_classes = []
		total_cached_samples = 0
		
		for cls_idx in range(self.total_classes):
			cache_count = self.class_cache_count[cls_idx]
			total_cached_samples += cache_count
			if cache_count > 0:
				active_classes.append(cls_idx)
			else:
				empty_classes.append(cls_idx)
		
		print(f"{prefix} {len(active_classes)}/{self.total_classes}", file=getattr(self.args, 'log_file', None))
		print(f"{prefix} {total_cached_samples}", file=getattr(self.args, 'log_file', None))
		
		if active_classes:
			print(f"{prefix}:", file=getattr(self.args, 'log_file', None))
			for cls_idx in active_classes:
				prototype = self.class_prototypes[cls_idx]
				prototype_norm = torch.norm(prototype).item()
				cache_count = self.class_cache_count[cls_idx]
				is_current_task = "CURRENT TASK" if cls_idx > self.PREV_INTRODUCED_CLS else "PRE TASK"
				print(f"{prefix}  CALSS{cls_idx}: 缓存样本={cache_count}, PRO={prototype_norm:.4f}, TASK={is_current_task}", 
					  file=getattr(self.args, 'log_file', None))
		
		print(f"{prefix}========================\n", file=getattr(self.args, 'log_file', None))

	def print_prototype_details(self, cls_idx_list=None, show_vectors=False):

		if cls_idx_list is None:
			cls_idx_list = [cls_idx for cls_idx in range(self.total_classes) 
							if self.class_cache_count[cls_idx] > 0]
		
		for cls_idx in cls_idx_list:
			if cls_idx >= self.total_classes:
				continue
			
			prototype = self.class_prototypes[cls_idx]
			cache_count = self.class_cache_count[cls_idx]
			prototype_norm = torch.norm(prototype).item()
			
			
			if show_vectors and prototype_norm > 1e-8:
				vector_preview = prototype[:10].cpu().numpy() if prototype.is_cuda else prototype[:10].numpy()
			
			print("", file=getattr(self.args, 'log_file', None))

	def compute_prototype_similarity(self, feature, cls_label):

		if cls_label not in self.class_prototypes:
			return 0.0
		
		prototype = self.class_prototypes[cls_label]
		prototype_norm = torch.norm(prototype)
		if prototype_norm < 1e-8:
			return 0.0
		prototype = prototype.to(feature.device)

		similarity = F.cosine_similarity(feature, prototype, dim=0)
		return similarity.item()
   
	def generate_old_class_pseudo_labels(self, old_results, labels, old_features=None):
		high_thresh = getattr(self.args, 'pseudo_thresh_high', 0.5)
		low_thresh = getattr(self.args, 'pseudo_thresh_low', 0.2)
		prototype_sim_thresh = getattr(self.args, 'prototype_sim_thresh', 0.5)  
		use_prototype_filtering = getattr(self.args, 'use_prototype_filtering', True)


		for i in range(len(old_results)):
			scores = old_results[i]['scores']
			labels_tensor = old_results[i]['labels']
			boxes = old_results[i]['boxes']

			final_pseudo_boxes = []
			final_pseudo_labels = []
			valid_class_mask = labels_tensor <= self.PREV_INTRODUCED_CLS

			high_conf_mask = (scores > high_thresh) & valid_class_mask
			if high_conf_mask.any():
				final_pseudo_boxes.append(boxes[high_conf_mask])
				final_pseudo_labels.append(labels_tensor[high_conf_mask])

			medium_conf_mask = (scores > low_thresh) & (scores <= high_thresh) & valid_class_mask

			if medium_conf_mask.any() and use_prototype_filtering and old_features is not None and hasattr(old_features,
                                                                                                           'last_hidden_state'):

				candidate_indices = torch.nonzero(medium_conf_mask).squeeze(1)
				candidate_labels = labels_tensor[medium_conf_mask]
				candidate_boxes = boxes[medium_conf_mask]

				if candidate_indices.numel() > 0 and candidate_indices.max().item() < old_features.last_hidden_state.shape[1]:
					candidate_features = old_features.last_hidden_state[i, candidate_indices, :]

					unique_labels_in_candidates = torch.unique(candidate_labels)
					similarity_scores = torch.zeros_like(candidate_labels, dtype=torch.float32)

					for cls_id in unique_labels_in_candidates:

						if cls_id.item() in self.class_prototypes:
							cls_mask = candidate_labels == cls_id
							prototype = self.class_prototypes[cls_id.item()].to(candidate_features.device)

							if torch.norm(prototype) < 1e-8:
								continue

							cls_features = candidate_features[cls_mask]
							cls_similarities = F.cosine_similarity(cls_features, prototype.unsqueeze(0), dim=1)
							similarity_scores[cls_mask] = cls_similarities


					high_sim_mask = similarity_scores >= prototype_sim_thresh

					if high_sim_mask.any():
						final_pseudo_boxes.append(candidate_boxes[high_sim_mask])
						final_pseudo_labels.append(candidate_labels[high_sim_mask])

			if final_pseudo_labels:
				all_final_labels = torch.cat(final_pseudo_labels)
				all_final_boxes = torch.cat(final_pseudo_boxes)

				labels[i]['class_labels'] = torch.cat([labels[i]['class_labels'],all_final_labels.to(labels[i]['class_labels'].device)])
				labels[i]['boxes'] = torch.cat([labels[i]['boxes'],all_final_boxes.to(labels[i]['boxes'].device)])

		return labels

	def add_pseudo_labels(self, teacher_outputs, labels):
		"""Append pseudo-labels of earlier classes to each image's targets (PPG, fix F6; see ppg.py)."""
		pseudo = local_trainer.compute_pseudo_labels(self, teacher_outputs, labels)
		return local_trainer.attach_pseudo_labels(self, labels, pseudo)

	def compute_pseudo_labels(self, teacher_outputs, labels):
		"""PPG pseudo-labels of each image as (boxes, labels); does not change ``labels`` (P2 caches them)."""
		prev = self.PREV_INTRODUCED_CLS
		scores, cand_labels, queries, boxes = select_candidates(
			teacher_outputs.logits, teacher_outputs.pred_boxes, prev, self.args.pseudo_topk)
		if self._old_prototypes is None:
			self._old_prototypes = prototype_matrix(self.class_prototypes, prev, self.device)
		prototypes, valid = self._old_prototypes
		pseudo = []
		for i, target in enumerate(labels):
			features = teacher_outputs.last_hidden_state[i, queries[i]]
			pseudo.append(select_pseudo_labels(
				scores[i], cand_labels[i], boxes[i], features, prototypes, valid, mode=self.args.pseudo,
				tau_high=self.args.pseudo_thresh_high, tau_low=self.args.pseudo_thresh_low,
				sim_thresh=self.args.prototype_sim_thresh, nearest_prototype=bool(self.args.prototype_nearest),
				gt_boxes=target['boxes'], gt_iou=self.args.pseudo_gt_iou, dedup_iou=self.args.pseudo_dedup_iou))
		return pseudo

	def attach_pseudo_labels(self, labels, pseudo):
		"""Append each image's pseudo-labels to its targets, in place as before P2, and return ``labels``."""
		for target, (new_boxes, new_labels) in zip(labels, pseudo, strict=True):
			target['class_labels'] = torch.cat([target['class_labels'], new_labels.to(target['class_labels'].dtype)])
			target['boxes'] = torch.cat([target['boxes'], new_boxes.to(target['boxes'].dtype)])
			self.ppg_pseudo_labels_total = getattr(self, 'ppg_pseudo_labels_total', 0) + len(new_labels)
		return labels

	def cached_pseudo_labels(self, labels, pixel_mask):
		"""P2: the cached pseudo-labels of every image of the batch, or None (cache off, padding, or a miss).
		Only batches without padding: padding depends on the other images of the batch."""
		if not self.use_pseudo_cache or not bool(pixel_mask.all()):
			return None
		ids = [int(target['image_id']) for target in labels]
		if not all(image_id in self.pseudo_cache for image_id in ids):
			return None
		self.pseudo_cache_hits += len(ids)
		return [tuple(t.to(self.device) for t in self.pseudo_cache[image_id]) for image_id in ids]

	def store_pseudo_labels(self, labels, pixel_mask, pseudo):
		if self.use_pseudo_cache and bool(pixel_mask.all()):
			for target, (boxes, classes) in zip(labels, pseudo, strict=True):
				cached = (boxes.detach().cpu().clone(), classes.detach().cpu().clone())
				self.pseudo_cache[int(target['image_id'])] = cached

	def common_step(self, batch, batch_idx, return_outputs=None):
		pixel_values = batch["pixel_values"].to(self.device)
		pixel_mask = batch["pixel_mask"].to(self.device)
		labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
		orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

		backbone_features = self.shared_backbone_features(pixel_values, pixel_mask)
		shared = {} if backbone_features is None else {'backbone_features': backbone_features}

		old_model_outputs = None
		use_pseudo = getattr(self.args, 'pseudo', 'ppg') != 'none'
		if (not self.eval_mode and self.task_id > 1 and self.teacher is not None and
			hasattr(self.args, 'use_distillation') and self.args.use_distillation):
			pseudo = self.cached_pseudo_labels(labels, pixel_mask) if use_pseudo else None
			if pseudo is None:
				old_model_outputs = self.teacher_outputs(pixel_values, pixel_mask, **shared)
			else:
				labels = self.attach_pseudo_labels(labels, pseudo)
		if old_model_outputs is not None and use_pseudo:
			if self.args.ppg_legacy:
				old_results = self.processor.post_process(
					old_model_outputs,
					target_sizes=orig_target_sizes,
					bg_thres_topk=self.args.bg_thres_topk
				)
				labels = self.generate_old_class_pseudo_labels(old_results, labels, old_model_outputs)
			else:
				pseudo = self.compute_pseudo_labels(old_model_outputs, labels)
				self.store_pseudo_labels(labels, pixel_mask, pseudo)
				labels = self.attach_pseudo_labels(labels, pseudo)
		if self.args.use_prompts:
			with torch.no_grad():

				outputs = self.model(
					pixel_values=pixel_values, pixel_mask=pixel_mask, labels=None if shared else labels,
					train=False, task_id=self.task_id, **shared)

				if getattr(self.args, 'per_query_prompt', 0):
					query = outputs.last_hidden_state
				elif not self.args.local_query:
					query = outputs.last_hidden_state.mean(dim=1)
				else:
					query = outputs.last_hidden_state
					outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}
					indices = self.model.matcher(outputs_without_aux, labels)
					
					one_hot_proposals = torch.zeros((len(labels),300)).to(self.device)
					for i,ind in enumerate(indices):
						for j in ind[0]:
							one_hot_proposals[i][j] = 1

				if self.args.bg_thres and not return_outputs:
					results = self.processor.post_process(outputs, target_sizes=orig_target_sizes, bg_thres_topk=self.args.bg_thres_topk)

			if self.args.local_query:
				with torch.set_grad_enabled(torch.is_grad_enabled() and bool(self.args.query_loss_grad)):
					query_wt = self.model.model.prompts.query_tf(query.view(query.shape[0],-1))
					query_loss = F.cross_entropy(query_wt, one_hot_proposals)

		else:
			query = None

		# C1: the second pass reuses the proposals of the first (same queries, same order for the per-query prompts)
		given = getattr(outputs, 'proposal_boxes', None) if query is not None else None
		outputs = self.model(
			pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, query=query, train=True,
			task_id=self.task_id, proposals=given, **shared)

		if self.training and hasattr(outputs, 'last_hidden_state'):
			is_last_epoch = (self.current_epoch == self.args.epochs - 1) if hasattr(self.args, 'epochs') else False
			
			if self.update_prototypes_last_epoch_only and is_last_epoch:
				self.batch_counter += 1
				if self.batch_counter % self.prototype_update_frequency == 0:
					query_vectors = outputs.last_hidden_state

					outputs_without_aux = {k: v for k, v in outputs.items() if k not in ["auxiliary_outputs", "enc_outputs"]}
					indices = self.model.matcher(outputs_without_aux, labels)
					pred_classes = outputs.logits[..., :self.seen_classes].argmax(dim=-1)

					prototype_updates = {}
					
					for batch_idx, (pred_indices, target_indices) in enumerate(indices):
						current_labels = labels[batch_idx]['class_labels']
						current_query_vectors = query_vectors[batch_idx]

						for pred_idx, target_idx in zip(pred_indices, target_indices):
							if target_idx < len(current_labels):
								cls_label = current_labels[target_idx].item()
								correct = pred_classes[batch_idx, pred_idx].item() == cls_label
								if (0 <= cls_label < self.total_classes and cls_label >= self.PREV_INTRODUCED_CLS
										and (correct or not self.args.proto_correct_only)):
									matched_query = current_query_vectors[pred_idx]
									if cls_label not in prototype_updates:
										prototype_updates[cls_label] = []
									prototype_updates[cls_label].append(matched_query)
					
					for cls_label, query_list in prototype_updates.items():
						for query in query_list:
							self.update_class_cache(cls_label, query)

						self.class_prototypes[cls_label] = self.compute_class_prototypes(cls_label)
						
						if prototype_updates and self.batch_counter % 100 == 0:
							print(f"[Last Epoch] Batch {self.batch_counter}: Updated prototypes for {len(prototype_updates)} classes")
							
			elif not self.update_prototypes_last_epoch_only:

				self.batch_counter += 1
				if self.batch_counter % self.prototype_update_frequency == 0:
					query_vectors = outputs.last_hidden_state

					outputs_without_aux = {k: v for k, v in outputs.items() if k not in ["auxiliary_outputs", "enc_outputs"]}
					indices = self.model.matcher(outputs_without_aux, labels)
					pred_classes = outputs.logits[..., :self.seen_classes].argmax(dim=-1)
					prototype_updates = {}

					for batch_idx, (pred_indices, target_indices) in enumerate(indices):
						current_labels = labels[batch_idx]['class_labels']
						current_query_vectors = query_vectors[batch_idx]

						for pred_idx, target_idx in zip(pred_indices, target_indices):
							if target_idx < len(current_labels):
								cls_label = current_labels[target_idx].item()
								correct = pred_classes[batch_idx, pred_idx].item() == cls_label
								if (0 <= cls_label < self.total_classes and cls_label >= self.PREV_INTRODUCED_CLS
										and (correct or not self.args.proto_correct_only)):
									matched_query = current_query_vectors[pred_idx]
									if cls_label not in prototype_updates:
										prototype_updates[cls_label] = []
									prototype_updates[cls_label].append(matched_query)
					
					for cls_label, query_list in prototype_updates.items():
						for query in query_list:
							self.update_class_cache(cls_label, query)

						self.class_prototypes[cls_label] = self.compute_class_prototypes(cls_label)

		loss = outputs.loss
		loss_dict = outputs.loss_dict

		if (self.training and self.args.use_prompts and getattr(self.args, 'ddl_lambda', 0) > 0
				and self.args.use_shared and self.args.use_private):
			ddl_loss = self.model.model.prompts.ddl_loss_all_layers()
			loss_dict['loss_ddl'] = ddl_loss
			loss = loss + self.args.ddl_lambda * ddl_loss

		if self.training and getattr(self.args, 'proto_margin_lambda', 0) > 0:
			proto_loss = self.prototype_margin_loss(outputs, labels)
			loss_dict['loss_proto'] = proto_loss
			loss = loss + self.args.proto_margin_lambda * proto_loss

		if self.training and self.args.feature_kd > 0:
			if self.task_id > 1 and not getattr(self.args, 'joint', False) and old_model_outputs is None:
				raise RuntimeError(
					'--feature_kd > 0 on task_id > 1 requires a teacher (self.teacher is None or distillation off)'
				)
			if old_model_outputs is not None:
				kd_loss = F.mse_loss(outputs.encoder_last_hidden_state, old_model_outputs.encoder_last_hidden_state)
				loss_dict['loss_feature_kd'] = kd_loss
				loss = loss + self.args.feature_kd * kd_loss

		if self.training and getattr(self.args, 'logit_kd', 0) > 0:
			if self.task_id > 1 and not getattr(self.args, 'joint', False) and old_model_outputs is None:
				raise RuntimeError(
					'--logit_kd > 0 on task_id > 1 requires a teacher (self.teacher is None or distillation off)'
				)
			if old_model_outputs is not None:
				mask = select_kd_queries(old_model_outputs.logits, self.PREV_INTRODUCED_CLS, self.args.logit_kd_topk)
				logit_kd = logit_kd_loss(outputs.logits, old_model_outputs.logits, mask, self.PREV_INTRODUCED_CLS)
				loss_dict['loss_logit_kd'] = logit_kd
				loss = loss + self.args.logit_kd * logit_kd
				if self.args.logit_kd_box > 0:
					box_kd = box_kd_loss(outputs.pred_boxes, old_model_outputs.pred_boxes, mask)
					loss_dict['loss_box_kd'] = box_kd
					loss = loss + self.args.logit_kd_box * box_kd

		if self.args.use_prompts and self.args.local_query:
			loss_dict['query_loss'] = query_loss

			loss += self.args.lambda_query * query_loss

		if return_outputs:

			if self.args.mask_gradients:
				outputs.logits[:,:, self.invalid_cls_logits] = -10e10
				outputs.logits = outputs.logits[:,:,:self.args.n_classes-1]
		
			results = self.processor.post_process(outputs, target_sizes=orig_target_sizes)
			res = {target['image_id'].item(): output for target, output in zip(labels, results)}
			res = self.evaluator.prepare_for_coco_detection(res)
		
			return loss, loss_dict, res

		return loss, loss_dict
	
	def training_step(self, batch, batch_idx):
		loss, loss_dict = self.common_step(batch, batch_idx)
		short_map = {'loss_ce':'ce','loss_giou':'giou','cardinality_error':'car','training_loss':'tr','loss_bbox':'bbox', 'query_loss':'QL', 'loss_ddl':'DDL', 'loss_feature_kd':'FKD', 'loss_prop':'PROP', 'loss_proto':'PROTO', 'loss_logit_kd':'LKD', 'loss_box_kd':'BKD'}
		self.log("tr", loss, prog_bar=True)
		for k,v in loss_dict.items():
			self.log(short_map[k], v.item(), prog_bar=True)

		is_last_epoch = (self.current_epoch == self.args.epochs - 1) if hasattr(self.args, 'epochs') else False
		if is_last_epoch:
			if batch_idx % 100 == 0:
				self.print_prototype_space_stats(prefix=f"[Last Epoch {self.current_epoch}, Batch {batch_idx}] ")
		elif batch_idx % 500 == 0:
			self.print_prototype_space_stats(prefix=f"[Epoch {self.current_epoch}, Batch {batch_idx}] ")

		return loss

	def missing_prototypes(self):
		"""Classes of the current task without any cached query (F7): PPG cannot verify them later."""
		return [c for c in range(self.PREV_INTRODUCED_CLS, self.seen_classes) if self.class_cache_count[c] == 0]

	def on_train_end(self):
		missing = self.missing_prototypes()
		if missing:
			message = f'WARNING: task {self.task_id}: {len(missing)} classes have no prototype: {missing}'
			print(message)
			print(message, file=self.args.log_file)

	def on_after_backward(self, *args):
		for i in range(len(self.model.class_embed)):
			if self.frozen_rows is not None:
				mask = self.frozen_rows.to(self.model.class_embed[i].weight.device)
				self.model.class_embed[i].weight.grad[mask] = 0
				self.model.class_embed[i].bias.grad[mask] = 0
				continue
			self.model.class_embed[i].weight.grad[:self.PREV_INTRODUCED_CLS,:] = 0
			self.model.class_embed[i].bias.grad[:self.PREV_INTRODUCED_CLS] = 0
		return

	@torch.no_grad()
	def on_train_batch_end(self, outputs, batch, batch_idx):
		if self._frozen_values is None:
			return
		for head, (weight, bias) in zip(self.model.class_embed, self._frozen_values):
			mask = self.frozen_rows.to(head.weight.device)
			head.weight[mask] = weight
			head.bias[mask] = bias
	def is_last_epoch(self):
		if not hasattr(self.args, 'epochs'):
			return False
		return self.current_epoch == self.args.epochs - 1
	def on_train_epoch_end(self):
		prompts = getattr(self.model.model, 'prompts', None)
		if prompts is not None and prompts.cross_attn:
			print(f'B: epoch {self.current_epoch} {prompts.cross_attn_stats()}', file=self.args.log_file)

		is_last_epoch = self.is_last_epoch()
		if is_last_epoch:
			self.print_prototype_space_stats(prefix=f"[Last Epoch {self.current_epoch} End] ")

			current_task_classes = [cls_idx for cls_idx in range(self.PREV_INTRODUCED_CLS + 1, self.total_classes) 
									if self.class_cache_count[cls_idx] > 0]
			if current_task_classes:
				self.print_prototype_details(current_task_classes, show_vectors=False)
			print(f"\n[Performance] Prototype updating was limited to last epoch only for speed optimization", file=getattr(self.args, 'log_file', None))
		elif self.current_epoch % 5 == 0:  
			print(f"[Epoch {self.current_epoch} End] Training progress: {self.current_epoch}/{getattr(self.args, 'epochs', 'Unknown')}", 
				  file=getattr(self.args, 'log_file', None))

	def validation_step(self, batch, batch_idx):
		if batch_idx == 0:
			if not self.eval_mode:
				self.coco_evaluator = CocoEvaluator(self.test_dataset.coco, self.args.iou_types)
			else:
				self.coco_evaluator  = self.evaluator.coco_evaluator

		targets = batch['labels']
		sizes = torch.stack([t['orig_size'] for t in targets]).to(self.device)
		results = predict_batch(self.model, batch['pixel_values'].to(self.device), batch['pixel_mask'].to(self.device),
								sizes, use_prompts=bool(self.args.use_prompts), local_query=bool(self.args.local_query),
								seen_classes=self.seen_classes, reuse_features=bool(self.args.reuse_features))
		res = {t['image_id'].item(): r for t, r in zip(targets, results)}
		self.coco_evaluator.update(self.evaluator.prepare_for_coco_detection(res))

		if batch_idx == self.trainer.num_val_batches[0]-1:
			self.coco_evaluator.synchronize_between_processes()
			self.coco_evaluator.accumulate()
			self.coco_evaluator.summarize()
			stats = self.coco_evaluator.coco_eval[self.args.iou_types[0]].stats

			if self.trainer.global_rank == 0:
				metrics = {name: float(stats[i]) for i, name in enumerate(('AP', 'AP50', 'AP75'))}
				save_json(os.path.join(self.args.output_dir, f'val_epoch_{self.current_epoch:04d}.json'),
					{'timestamp': now(), 'epoch': self.current_epoch, 'optimizer_step': self.trainer.global_step,
					 'scope': 'val_task', 'metrics': metrics}, indent=2)
				for name, value in metrics.items():
					self.log('val_' + name, value, on_step=False, on_epoch=True)
				if getattr(self.args, 'select_best_epoch', 0):
					record_epoch(self.args.output_dir, self.current_epoch, self.trainer.global_step, metrics['AP50'], self.model)
				self.evaluator.print_coco_stats(self.current_epoch, stats, self.print_count)
				self.print_count = 1
				if self.args.viz:
					image_ids = self.evaluator.test_dataset.coco.getImgIds()
					for id in image_ids[0:self.args.num_imgs_viz]:
						self.evaluator.vizualize(id=id)
	
	def on_save_checkpoint(self, checkpoint):
		self.runtime.update(model_config=self.model.config.to_dict(), processor_config=self.processor.to_dict())
		checkpoint['runtime'] = self.runtime
		checkpoint['training_contract'] = runtime.training_contract(self.args)
		checkpoint['pdp_state'] = {
			'class_query_cache': self.class_query_cache,
			'class_prototypes': self.class_prototypes,
			'class_cache_count': self.class_cache_count,
			'batch_counter': self.batch_counter,
		}

	def on_load_checkpoint(self, checkpoint):
		if 'training_contract' in checkpoint and checkpoint['training_contract'] != runtime.training_contract(self.args):
			raise ValueError('Resume training contract differs from the checkpoint (batch, data or optimization policy)')
		state = checkpoint['pdp_state']
		self.class_query_cache = state['class_query_cache']
		self.class_prototypes = state['class_prototypes']
		self.class_cache_count = state['class_cache_count']
		self.batch_counter = state['batch_counter']
		self._old_prototypes = None

	def save_task_final(self, path):
		"""Weights and prototype memory at the end of the task, without optimizer state (F8).

		This file initialises the next task and becomes its teacher. Written to a temporary file
		first, then renamed, so an interruption never leaves a truncated file behind.
		"""
		self.runtime.update(model_config=self.model.config.to_dict(), processor_config=self.processor.to_dict())
		save_dict = {
			'runtime': self.runtime,
			'model': self.model.state_dict(),
			'task_id': self.task_id,
			'class_query_cache': self.class_query_cache,
			'class_prototypes': self.class_prototypes,
			'class_cache_count': self.class_cache_count,
		}
		tmp_path = path + '.tmp'
		torch.save(save_dict, tmp_path)
		os.replace(tmp_path, path)
		print(f'Model saved to {path}', file=self.args.log_file)
	
	def migrated_state(self, state):
		"""A checkpoint of a run without the new modules: A drops query_tf (no such module); B (D3 of
		docs/ai/planning/2026-11-25-plan-B-C1.md) keeps the initial weights of the modules the checkpoint lacks, which
		leave the outputs unchanged (W = 0). Any other missing weight still fails the strict load."""
		if getattr(self.args, 'per_query_prompt', 0):
			dropped = [k for k in state if '.prompts.query_tf.' in k]
			if dropped:
				print(f'A: --per_query_prompt 1 drops {len(dropped)} query_tf weights of the checkpoint: {dropped}',
					  file=self.args.log_file)
			state = {k: v for k, v in state.items() if k not in dropped}
		prefixes = ['.prompts.xattn_'] if getattr(self.args, 'prompt_cross_attn', 0) else []
		prefixes += ['.proposals.'] if getattr(self.args, 'query_proposals', 0) else []
		prefixes += ['.lora_A', '.lora_B'] if getattr(self.args, 'decoder_lora_rank', 0) > 0 else []
		current = self.model.state_dict()
		added = [k for k in current if k not in state and any(p in k for p in prefixes)]
		if added:
			print(f'D3: the checkpoint has no {added}; they keep their initial values', file=self.args.log_file)
		return {**state, **{k: current[k] for k in added}}

	def resume(self, load_path=''):
		print('\n Resuming model for task ', self.task_id, ' from : ',load_path, file=self.args.log_file)
		if load_path:
			checkpoint = torch.load(load_path, map_location='cpu')
			runtime.validate_runtime(self.args, checkpoint.get('runtime'))
			missing_keys, unexpected_keys = self.model.load_state_dict(
				self.migrated_state(checkpoint['model']), strict='runtime' in checkpoint)

			if 'class_query_cache' in checkpoint:
				self.class_query_cache = checkpoint['class_query_cache']
				print(f"\n Loaded class_query_cache with {len(self.class_query_cache)} classes", file=self.args.log_file)
			
			if 'class_prototypes' in checkpoint:
				self.class_prototypes = checkpoint['class_prototypes']
				print(f"\n Loaded class_prototypes with {len(self.class_prototypes)} classes", file=self.args.log_file)
			
			if 'class_cache_count' in checkpoint:
				self.class_cache_count = checkpoint['class_cache_count']
				print(f"\n Loaded class_cache_count with {len(self.class_cache_count)} classes", file=self.args.log_file)
			else:

				for cls_idx in range(self.total_classes):
					self.class_cache_count[cls_idx] = len(self.class_query_cache.get(cls_idx, []))
			
			self.print_prototype_space_stats(prefix="[Resume] ")

		if not self.args.eval and self.args.freeze:
			
			freeze = self.args.freeze.split(',')
			if self.task_id > 1 and self.args.freeze_shared_after_task1:
				freeze += SHARED_AFTER_TASK1
			tune = self.args.tune_after_task1.split(',') if self.task_id > 1 and self.args.tune_after_task1 else []
			for id, (name, params) in enumerate(self.model.named_parameters()):
				params.requires_grad = True
				flag = False
				parts = name.split('.')
				for n in parts:
					if n in freeze and not set(parts) & set(tune) and not ({'lora_A', 'lora_B'} & set(parts)):
						params.requires_grad = False
						flag = True
				if not flag:
					print ('Trainable ..', name, "  Req grad .. ",params.requires_grad, file=self.args.log_file)
			self.log_trainable_parameters()
		self.check_reuse_features()

	def check_reuse_features(self):
		"""P1: --reuse_features shares backbone features computed without gradient, so the backbone must stay frozen
		in every task of the run. Checked when task 1 starts, not when a later task opens the backbone (M5).
		Without prompts or teacher (FSA, NAIVE) there is nothing to share and the flag changes nothing."""
		if not self.args.reuse_features or self.args.eval or self.args.predict_only:
			return
		if not (self.args.use_prompts or self.args.pseudo != 'none'):
			print('P1: --reuse_features has no effect (no prompts, no teacher)', file=self.args.log_file)
			return
		reason = backbone_training_reason(self.model, self.args)
		if reason:
			raise ValueError(f'--reuse_features 1 needs a frozen backbone, but {reason}; run with --reuse_features 0')

	def shared_backbone_features(self, pixel_values, pixel_mask):
		"""P1: backbone output of the batch for the teacher and both student passes, or None when the flag is off
		or nothing is shared. No gradient needed: check_reuse_features() made sure the backbone is frozen."""
		if not (self.args.reuse_features and (self.args.use_prompts or self.teacher is not None)):
			return None
		with torch.no_grad():
			return self.model.model.extract_features(pixel_values, pixel_mask)

	def log_trainable_parameters(self):
		"""F10: number of trained parameters per group and learning rate, written to the task log.

		Not only prompts and the classifier are trained: input_proj, query_position_embeddings,
		reference_points, level_embed and bbox_embed are shared by every task and trained at lr_old.
		"""
		new_params = self.args.new_params.split(',')
		groups, total, trainable = {}, 0, 0
		for name, param in self.model.named_parameters():
			total += param.numel()
			if not param.requires_grad:
				continue
			trainable += param.numel()
			parts = name.split('.')
			group = parts[1] if parts[0] == 'model' else parts[0]
			lr = self.args.lr if self.match_name_keywords(name, new_params) else self.args.lr_old
			groups[(group, lr)] = groups.get((group, lr), 0) + param.numel()
		lines = [f'Trainable parameters: {trainable / 1e6:.2f}M of {total / 1e6:.2f}M']
		lines += [f'  {group:28s} lr {lr:g}: {count / 1e6:.3f}M' for (group, lr), count in sorted(groups.items())]
		print('\n'.join(lines), file=self.args.log_file)
		return total, trainable, groups

	def match_name_keywords(self, n, name_keywords):
		out = False
		for b in name_keywords:
			if b in n:
				out = True
				break
		return out
	
	def configure_optimizers(self):
		new_params = self.args.new_params.split(',')

		groups = self.args.optim_groups if self.args.optim_groups != 'auto' else ('prompt' if self.args.repo_name else 'detr')
		if groups == 'prompt':
			param_dicts = [
				{"params": [p for n, p in self.named_parameters()
					if self.match_name_keywords(n, new_params) and p.requires_grad],
					"lr":self.args.lr,
					},
				{
					"params": [p for n, p in self.named_parameters() if not self.match_name_keywords(n, new_params) and p.requires_grad],
					"lr": self.args.lr_old,
				},
			]
		else:
			param_dicts = [
			{
				"params":
					[p for n, p in self.named_parameters()
					if not self.match_name_keywords(n, self.args.lr_backbone_names) and not self.match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
				"lr": self.args.lr,
			},
			{
				"params": [p for n, p in self.named_parameters() if self.match_name_keywords(n, self.args.lr_backbone_names) and p.requires_grad],
				"lr": self.args.lr_backbone,
			},
			{
				"params": [p for n, p in self.named_parameters() if self.match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
				"lr": self.args.lr * self.args.lr_linear_proj_mult,
			}
			]

		self.optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
								weight_decay=self.weight_decay)

		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)

		return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.lr_scheduler, 'interval': 'epoch'}}

	def train_dataloader(self):
		return self.train_dataloader

	def val_dataloader(self):
		return self.val_dataloader

class Evaluator():
	def __init__(self, processor, test_dataset, test_dataloader, coco_evaluator, 
			  task_label2name, args, local_trainer=None, PREV_INTRODUCED_CLS=0, 
			  CUR_INTRODUCED_CLS=20, local_eval=0, task_id=0, task_name=None):
		
		self.processor = processor
		self.local_trainer = local_trainer
		if local_trainer:
			self.model = local_trainer.model
		else:
			self.model = None

		self.test_dataset = test_dataset
		self.test_dataloader = test_dataloader
		self.coco_evaluator = coco_evaluator
		self.task_label2name = task_label2name
		self.args = args
		self.local_eval = local_eval
		self.task_id = task_id
		self.task_name = task_name

		prev_intro_cls = PREV_INTRODUCED_CLS
		curr_intro_cls = CUR_INTRODUCED_CLS
		seen_classes = prev_intro_cls + curr_intro_cls
		self.invalid_cls_logits = list(range(seen_classes, self.args.n_classes-1))

	def convert_to_xywh(self, boxes):
		xmin, ymin, xmax, ymax = boxes.unbind(1)
		return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

	def prepare_for_coco_detection(self, predictions):
		coco_results = []
		for original_id, prediction in predictions.items():
			if len(prediction) == 0:
				continue

			boxes = prediction["boxes"]
			boxes = self.convert_to_xywh(boxes).tolist()
			scores = prediction["scores"].tolist()
			labels = prediction["labels"].tolist()

			coco_results.extend(
				[
					{
						"image_id": original_id,
						"category_id": labels[k],
						"bbox": box,
						"score": scores[k],
					}
					for k, box in enumerate(boxes)
				]
			)
		return coco_results
	
	def print_coco_stats(self, epoch, stats, print_count):

		if self.task_name == 'cur':
			task_name = 'Current Task (mAP@C): '+self.args.task
		elif self.task_name == 'prev':
			task_name = 'Previous Tasks (mAP@P): '+self.args.task
		else:
			task_name = 'All seen Tasks (mAP@A): '+self.args.task

		output = [task_name,
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '+'%0.3f'%stats[0],
		'\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = '+'%0.3f'%stats[1],
		'\nAverage Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = '+'%0.3f'%stats[2],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = '+'%0.3f'%stats[3],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = '+'%0.3f'%stats[4],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = '+'%0.3f'%stats[5],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = '+'%0.3f'%stats[6],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = '+'%0.3f'%stats[7],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '+'%0.3f'%stats[8],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = '+'%0.3f'%stats[9],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = '+'%0.3f'%stats[10],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = '+'%0.3f'%stats[11],
		'\n\n']
		
		if print_count == 0:
			print_format = 'w'
		else:
			print_format = 'a'
		with open(self.args.output_dir+'/stats.txt', print_format) as f:
			f.writelines(output)
		f.close()
	
	def plot_results(self, pil_img, ax, scores, labels, boxes):
		COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
		[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
		ax.imshow(pil_img)
		colors = COLORS * 100
		for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
			ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
									fill=False, color=c, linewidth=2))
			text = f'{self.task_label2name[label]}: {score:0.2f}'
			ax.text(xmin, ymin, text, fontsize=5,
					bbox=dict(facecolor='yellow', alpha=0.5))
		ax.grid('off')

	def evaluate(self):
		args = self.args
		model = self.model
		coco_evaluator = self.coco_evaluator
		print("\n Running Final Evaluation... \n", file=self.args.log_file)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		model.eval()

		for idx, batch in enumerate(tqdm.tqdm(self.test_dataloader)):
			
			pixel_values = batch["pixel_values"].to(device)
			pixel_mask = batch["pixel_mask"].to(device)
			labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
			orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
		
			with torch.no_grad():
				outputs = prompted_outputs(
					self.model, pixel_values, pixel_mask, use_prompts=bool(self.args.use_prompts),
					local_query=self.args.local_query, reuse_features=bool(self.args.reuse_features),
					first_task_id=self.task_id if self.args.use_prompts else None)

			if self.args.mask_gradients:
				outputs.logits[:,:, self.invalid_cls_logits] = -10e10
				outputs.logits = outputs.logits[:,:,:self.args.n_classes-1]
	
			
			results = self.processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes,
															threshold=0)
			res = {target['image_id'].item(): output for target, output in zip(labels, results)}
			res = self.prepare_for_coco_detection(res)
			coco_evaluator.update(res)

		coco_evaluator.synchronize_between_processes()
		coco_evaluator.accumulate()
		coco_evaluator.summarize()

		if self.local_eval:
			self.print_coco_stats(epoch=args.epochs+1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats, print_count=1)
		elif self.local_trainer.trainer.global_rank == 0:
			self.print_coco_stats(epoch=args.epochs+1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats, print_count=1)

		if args.viz:
			image_ids = self.test_dataset.coco.getImgIds()
			for id in image_ids[0:self.args.num_imgs_viz]:
				try:
					self.vizualize()
				except:
					continue

	def vizualize(self, id=None, score_threshold=0.18, device='cuda'):
		test_dataset = self.test_dataset
		image_ids = test_dataset.coco.getImgIds()
		
		if id == None:
			image_id = image_ids[np.random.randint(0, len(image_ids))]
		else:
			image_id = id
		print('Image n°{}'.format(image_id))
		image = test_dataset.coco.loadImgs(image_id)[0]
		image = Image.open(os.path.join(self.args.test_img_dir, image['file_name']))

		fig, ax = plt.subplots(1, 2, figsize=(14,6), dpi=220)

		annotations = test_dataset.coco.imgToAnns[image_id]
		cats = test_dataset.coco.cats
		id2label = {k: v['name'] for k,v in cats.items()}
		scores, labels, boxes = [],[],[]
		for annotation in annotations:
			box = annotation['bbox']
			class_idx = annotation['category_id']
			x,y,w,h = tuple(box)
			scores.append(1.0)
			labels.append(class_idx)
			boxes.append((x,y,x+w,y+h))
		
		ax[0].set_title('GT')
		self.plot_results(image,ax=ax[0],scores=np.array(scores),labels=np.array(labels),boxes=np.array(boxes))

		inputs = self.processor(images=image, return_tensors="pt")
		inputs['pixel_values'] = inputs['pixel_values'].to(device)
		inputs['pixel_mask'] = inputs['pixel_mask'].to(device)

		if self.args.use_prompts:
			with torch.no_grad():
				outputs = self.model(pixel_values=inputs['pixel_values'] , pixel_mask=inputs['pixel_mask'], train=False)
				
				if not self.args.local_query:
					query = outputs.last_hidden_state.mean(dim=1)
				else:
					query = outputs.last_hidden_state
		else:
			query = None

		with torch.no_grad():
			outputs = self.model(pixel_values=inputs['pixel_values'] , pixel_mask=inputs['pixel_mask'], query=query, train=False)

		if self.args.mask_gradients:
			outputs.logits[:,:, self.invalid_cls_logits] = -10e10
			outputs.logits = outputs.logits[:,:,:self.args.n_classes-1]
		
		task = 'cur'
		if len(self.args.task) > 1:
			task = 'prev'
			
		results = self.processor.post_process_object_detection(outputs,target_sizes=[image.size[::-1]],
															threshold=score_threshold)[0]

		ax[1].set_title('Prediction (Ours)')
		self.plot_results(image,ax=ax[1],scores=results['scores'],labels=results['labels'],boxes=results['boxes'])
		
		for i in range(2):
				ax[i].set_aspect('equal')
				ax[i].set_axis_off()

		plt.savefig(os.path.join(self.args.output_dir, f'{task}_img_{image_id}.jpg'), bbox_inches = 'tight',pad_inches = 0.1)
