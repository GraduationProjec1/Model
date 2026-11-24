"""Speed and memory of the PDP model on random images (plan 3.5 first measurement, R4 GPU smoke, V5).

    cd pdp && python benchmark.py <model arguments of an experiment> --bench_mode train --bench_steps 20
    cd pdp && python benchmark.py <model arguments> --bench_mode infer --bench_sizes 640 800

train: a full PDP step of a task >= 2 (teacher two passes + PPG, student query pass, training pass,
backward, optimizer step) on random 800x800 images with random boxes; reports seconds per image and
peak GPU memory, to choose --batch_size and to budget GPU hours.
infer: two-pass inference latency per image at each size, batch 1 (the checkout use case).
Also prints the storage added per class (private prompts, prototype memory).
"""

import argparse
import json
import time

import torch

import main as pdp_main
from engine import local_trainer
from inference import predict_batch


def random_batch(batch_size, size, n_objects, device, generator, first_id=1):
    """first_id: image_id of the first image; ids are first_id, first_id + 1, ... (P2 caches by image_id)."""
    boxes = torch.rand(batch_size, n_objects, 4, generator=generator) * 0.3
    boxes[..., :2] += 0.2
    labels = [{'class_labels': torch.randint(0, 100, (n_objects,), generator=generator).to(device),
               'boxes': b.to(device), 'orig_size': torch.tensor([size, size], device=device),
               'image_id': torch.tensor([first_id + i], device=device)} for i, b in enumerate(boxes)]
    pixels = torch.randn(batch_size, 3, size, size, generator=generator).to(device)
    return {'pixel_values': pixels, 'pixel_mask': torch.ones(batch_size, size, size, dtype=torch.long, device=device),
            'labels': labels}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench_train(args, device):
    """A task-2 step: teacher (copy of the model) + PPG + student, like training with pseudo-labels."""
    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=2)
    if args.use_prompts:
        trainer.model.model.prompts.set_task_id(1)
    trainer.resume()
    if args.pseudo != 'none':
        trainer.set_teacher()
    trainer.to(device)
    if trainer.teacher is not None:
        trainer.teacher.to(device)
    trainer.train()
    optimizer = trainer.configure_optimizers()['optimizer']
    generator = torch.Generator().manual_seed(0)
    batches = [random_batch(args.batch_size, args.bench_train_size, 12, device, generator,
                            first_id=1 + step * args.batch_size) for step in range(args.bench_steps)]
    timings = []
    for _ in range(args.bench_epochs):
        for step, batch in enumerate(batches):
            synchronize(device)
            start = time.perf_counter()
            loss, _ = trainer.common_step(batch, step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            synchronize(device)
            timings.append(time.perf_counter() - start)
    steady = timings[2:] or timings
    return {'mode': 'train', 'batch_size': args.batch_size, 'size': args.bench_train_size, 'steps': len(timings),
            'seconds_per_image': round(sum(steady) / len(steady) / args.batch_size, 4)}


def bench_infer(args, device):
    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=1)
    model = trainer.model.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    result = {'mode': 'infer'}
    for size in args.bench_sizes:
        timings = []
        for _ in range(args.bench_steps):
            batch = random_batch(1, size, 1, device, generator)
            synchronize(device)
            start = time.perf_counter()
            predict_batch(model, batch['pixel_values'], batch['pixel_mask'],
                          torch.tensor([[size, size]], device=device), use_prompts=bool(args.use_prompts),
                          local_query=bool(args.local_query), seen_classes=100)
            synchronize(device)
            timings.append(time.perf_counter() - start)
        steady = timings[2:] or timings
        result[f'ms_per_image_{size}'] = round(1000 * sum(steady) / len(steady), 1)
    return result


def stage_timer(model, device):
    """Forward hooks on the backbone, encoder and decoder of ``model``; returns (timings, remove).

    timings[name] collects the seconds of each call of that stage (P1 plan, Task 1).
    """
    timings, starts, handles = {}, {}, []

    def before(name):
        def hook(module, inputs):
            synchronize(device)
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(module, inputs, output):
            synchronize(device)
            timings.setdefault(name, []).append(time.perf_counter() - starts[name])
        return hook

    detr = model.model
    for name, module in (('backbone', detr.backbone), ('encoder', detr.encoder), ('decoder', detr.decoder)):
        handles.append(module.register_forward_pre_hook(before(name)))
        handles.append(module.register_forward_hook(after(name)))

    def remove():
        for handle in handles:
            handle.remove()

    return timings, remove


def mean_ms(values):
    return 1000 * sum(values) / len(values)


def bench_stages(args, device):
    """P1 plan, Task 1: time of the backbone (B), encoder (E) and decoder (D) inside the prompt-free pass,
    of the two passes, and the time the reuse of features would remove (plan section 6.3), batch 1.

    removed_ms_*: infer = B + E (pass 2 skips both); train with teacher, epoch 1 = 3B + E; train with the
    pseudo-label cache hit = 3B + 2E + 2D; train without teacher (task 1, mode improve) = B. Divide by the
    measured step time of --bench_mode train (per image) to get the real saving."""
    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=1)
    model = trainer.model.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    size = args.bench_sizes[0]
    passes = {'pass1': [], 'pass2': []}
    stages, remove = stage_timer(model, device)
    try:
        with torch.no_grad():
            for _ in range(args.bench_steps):
                batch = random_batch(1, size, 1, device, generator)
                synchronize(device)
                start = time.perf_counter()
                first = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'], train=False)
                synchronize(device)
                passes['pass1'].append(time.perf_counter() - start)
                query = first.last_hidden_state if args.local_query else first.last_hidden_state.mean(dim=1)
                start = time.perf_counter()
                model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'], query=query, train=False)
                synchronize(device)
                passes['pass2'].append(time.perf_counter() - start)
    finally:
        remove()
    skip = 2 if args.bench_steps > 2 else 0
    pass1, pass2 = mean_ms(passes['pass1'][skip:]), mean_ms(passes['pass2'][skip:])
    b, e, d = (mean_ms(stages[name][::2][skip:]) for name in ('backbone', 'encoder', 'decoder'))
    return {'mode': 'stages', 'size': size,
            'backbone_ms': round(b, 3), 'encoder_ms': round(e, 3), 'decoder_ms': round(d, 3),
            'pass1_ms': round(pass1, 3), 'pass2_ms': round(pass2, 3),
            'removed_ms_infer': round(b + e, 3), 'predicted_infer_saving': round((b + e) / (pass1 + pass2), 3),
            'removed_ms_train_epoch1': round(3 * b + e, 3), 'removed_ms_train_cached': round(3 * b + 2 * e + 2 * d, 3),
            'removed_ms_train_no_teacher': round(b, 3)}


def bench_data(args, processor=None):
    """P1 plan, Task 1: seconds to load --bench_steps batches of train_task_1<suffix>.json through the training
    DataLoader (same dataset class, processor and workers as main.py), without the model."""
    import os

    import runtime

    from datasets.coco_hug import CocoDetection
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor

    processor = processor or runtime.processor(args, DeformableDetrImageProcessor)
    ann = os.path.join(args.task_ann_dir, f'train_task_1{args.train_suffix}.json')
    dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=ann, processor=processor)
    loader = pdp_main.make_train_loader(dataset, args)
    timings, images = [], 0
    iterator = iter(loader)
    for _ in range(args.bench_steps):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        timings.append(time.perf_counter() - start)
        images += batch['pixel_values'].shape[0]
    if not timings:
        raise ValueError(f'no batch could be loaded from {ann}')
    return {'mode': 'data', 'batches': len(timings), 'num_workers': args.num_workers,
            'seconds_per_image': round(sum(timings) / images, 4)}


def storage_per_class(args):
    """Bytes added per class: private prompts (p, k, a of every layer) and the prototype memory (V5)."""
    layers = 6
    prompt = layers * (args.prompt_len * 256 + 256 + 256) * 4
    memory = 100 * 256 * 4
    return {'prompt_bytes_per_class': prompt, 'prototype_memory_bytes_per_class': memory}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(parents=[pdp_main.get_args_parser()])
    parser.add_argument('--bench_mode', choices=['train', 'infer', 'stages', 'data'], default='train')
    parser.add_argument('--bench_steps', type=int, default=20)
    parser.add_argument('--bench_epochs', type=int, default=1,
                        help='P2: train mode repeats the same --bench_steps batches this many times')
    parser.add_argument('--bench_sizes', type=int, nargs='+', default=[640, 800])
    parser.add_argument('--bench_train_size', type=int, default=800)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    pdp_main.check_kernel(args)
    pdp_main.setup_task_info(args)
    args.log_file = open('/dev/null', 'w')
    device = torch.device('cuda' if args.accelerator == 'gpu' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    modes = {'train': bench_train, 'infer': bench_infer, 'stages': bench_stages,
             'data': lambda args, device: bench_data(args)}
    result = modes[args.bench_mode](args, device)
    if device.type == 'cuda':
        result['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        result['gpu'] = torch.cuda.get_device_name(0)
    result.update(storage_per_class(args))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
