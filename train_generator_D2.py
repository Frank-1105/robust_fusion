# train_generator_D.py

# -*- coding: utf-8 -*-
# coding=utf-8
import argparse
import warnings
import itertools # 用于链接不同模型的参数
import os
import os.path as osp
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import sys
import torch.nn.functional as F
import mmcv
import torch
import torch.nn as nn # 引入torch.nn用于定义损失函数
import torchvision
import numpy as np
from mmcv.utils.parrots_wrapper import SyncBatchNorm
from tqdm import tqdm
from mmengine.dist import get_dist_info, init_dist, is_main_process
from mmengine import Config, DictAction
from mmseg.core import build_optimizer
from mmengine.runner import set_random_seed
from mmengine.dist import collect_results_cpu, collect_results_gpu
from mmseg.models import build_segmentor
# from mmengine.optim import build_ddp, build_dp
import itertools
from collections import defaultdict
from mmseg.utils import build_ddp, build_dp, get_root_logger

from utils.lr_policy import WarmUpPolyLR
from utils.generate_loss import calculate_quantization_loss, calculate_sparsity_loss, calculate_l1_range_loss
# from test_D import run_validation
from torchvision.utils import save_image
from models import FusionNet
from models.generator1 import GeneratorResnetWrapper as Generator
from models.generator1.G_F import generate_grad_box , create_patch_index_map


def custom_collate_fn(batch):
    """
    自定义的数据打包函数。
    'batch' 是一个列表，列表中的每个元素都是 EnhanceDataset.__getitem__ 返回的字典。
    例如: [
        {'modal_x': tensor1, 'modal_y': tensor1, 'label': tensor1, 'img_metas': dict1},
        {'modal_x': tensor2, 'modal_y': tensor2, 'label': tensor2, 'img_metas': dict2},
        ...
    ]
    """
    # 初始化用于存储各个部分的列表
    collated_batch = {}
    # 获取第一个样本的所有键
    keys = batch[0].keys()

    for key in keys:
        # 如果键是 'img_metas'，我们直接将所有字典收集到一个列表中
        if key == 'img_metas' or key == 'fn' or isinstance(batch[0][key], str):
            collated_batch[key] = [d[key] for d in batch]
        # 对于其他键（我们假设它们都是可以堆叠的 Tensor）
        else:
            # 将这个键对应的所有 Tensor 收集到一个列表中
            tensor_list = [d[key] for d in batch]
            # 使用 torch.stack 将 Tensor 列表堆叠成一个批次 Tensor
            try:
                collated_batch[key] = torch.stack(tensor_list, 0)
            except Exception as e:
                print(f"Error stacking tensor for key '{key}': {e}")
                # 如果堆叠失败，可以进行调试或进行其他处理
                # 例如，对于非 Tensor 类型的数据，可以简单地收集成列表
                collated_batch[key] = tensor_list

    return collated_batch



def evaluate_and_get_metrics(seg_model, enhanced_image, seg_label):

    # 1. 准备 mmseg 所需的输入格式
    # mmseg 的模型 forward 方法通常接受一个包含图像张量的列表
    # `data` 字典的格式也需要匹配 mmseg 的数据集格式
    data = {
        'img': [enhanced_image],
        'gt_semantic_seg': [seg_label]
    }

    # 2. 调用模型的验证步骤 `val_step`
    # 注意：mmseg 的模型在 `val_step` 中已经集成了前向传播和指标计算的逻辑
    with torch.no_grad():
        # 这里我们调用 val_step，它会返回一个包含指标的字典
        # 在真实的 mmseg 训练循环中，这是由 runner 自动调用的
        # 我们这里是手动调用
        metrics = seg_model.val_step(data)

    # 3. 返回指标字典
    return metrics

def get_current_stage_config(epoch, stages):
    """
    根据当前 epoch 和阶段配置列表，返回当前阶段的配置。

    Args:
        epoch (int): 当前的 epoch (从 1 开始)。
        stages (list[dict]): 包含各阶段配置的列表。

    Returns:
        dict: 当前阶段的配置字典。
        int: 当前阶段的编号 (从 1 开始)。
    """
    cumulative_epochs = 0
    for i, stage_config in enumerate(stages):
        cumulative_epochs += stage_config['epochs']
        if epoch <= cumulative_epochs:
            return stage_config, i + 1
    # 如果 epoch 超出总范围，则返回最后一个阶段的配置
    return stages[-1], len(stages)

def parse_args():
    parser = argparse.ArgumentParser(description='Jointly train a segmentation network and a generator')
    parser.add_argument('config', help='Path to the training configuration file')
    parser.add_argument('--work-dir', help='Directory to save logs and models')

    # --- Optional generator weights path ---
    parser.add_argument('--generator-ckpt', help='(Optional) Path to the pretrained generator network checkpoint')

    parser.add_argument(
        '--resume-from', help='Resume training from a checkpoint')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to perform evaluation during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='Number of GPUs to use (only for non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='IDs of GPUs to use (only for non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='Custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# ===================================================================================
# ====================== 使用下面的函数完整替换你原来的版本 ======================
# ===================================================================================
def run_validation(seg_model, generator_model, fusion_model, val_loader, val_dataset, cfg, distributed, logger, epoch, writer):
    """
    在训练期间运行验证的函数 (已修复 TypeError 并增加代码稳健性)。
    """
    # 1. 切换到评估模式
    seg_module = seg_model.module if distributed else seg_model
    gen_module = generator_model.module if distributed else generator_model

    seg_module.eval()
    gen_module.eval()

    results = []
    # 只在主进程上显示进度条
    if is_main_process():
        prog_bar = mmcv.ProgressBar(len(val_loader))

    for i, data in enumerate(val_loader):
        modal_x = data['modal_x'].to(cfg.device, non_blocking=True)
        modal_y = data['modal_y'].to(cfg.device, non_blocking=True)
        img_metas = data['img_metas']

        # 在验证期间，不需要计算梯度
        with torch.no_grad():
            fused_image = fusion_model(modal_x, modal_y)

            # 某些模型可能在推理时也需要梯度上下文，这里临时开启
            with torch.set_grad_enabled(True):
                index_tensor, _, _ = create_patch_index_map(
                    img_height=fused_image.shape[-2], img_width=fused_image.shape[-1],
                    filterSize=cfg.generator_filter_size, stride=cfg.generator_stride,
                    batch_size=fused_image.shape[0], device=fused_image.device
                )
                grad_tensor = generate_grad_box(
                    model_type='res50', img_tensor=fused_image, img_height=fused_image.shape[-2],
                    img_width=fused_image.shape[-1], index_tensor=index_tensor,
                    filterSize=cfg.generator_filter_size, use_tk_selection=True, tk_value=cfg.tk_value
                )

            _, _, _, _, maskD = gen_module(modal_x, modal_y, grad=grad_tensor)
            enhanced_image = fused_image + maskD

             # ==================== 新增代码块开始 ====================
            # 只在主进程上，并且只针对第一个验证批次保存图像，以供调试和可视化
            if i == 0 and is_main_process():
                # 1. 定义保存图像的根目录
                vis_dir = osp.join(cfg.work_dir, 'visualizations')
                enhanced_dir = osp.join(vis_dir, 'enhanced_images')
                fused_dir = osp.join(vis_dir, 'fused_images')
                maskd_dir = osp.join(vis_dir, 'maskd_images')

                # 2. 确保目录存在
                mmcv.mkdir_or_exist(enhanced_dir)
                mmcv.mkdir_or_exist(fused_dir)
                mmcv.mkdir_or_exist(maskd_dir)

                # 3. 构造文件名 (包含 epoch 信息)，将整个批次的图像保存为一个网格图
                enhanced_filename = osp.join(enhanced_dir, f'epoch_{epoch}.png')
                fused_filename = osp.join(fused_dir, f'epoch_{epoch}.png')
                maskd_filename = osp.join(maskd_dir, f'epoch_{epoch}.png')

                # 4. 使用 torchvision.utils.save_image 保存图像
                # normalize=True 会自动将图像的数值范围从任意范围安全地映射到 [0, 1]
                save_image(enhanced_image, enhanced_filename, normalize=True)
                save_image(fused_image, fused_filename, normalize=True)
                save_image(maskD.abs(), maskd_filename, normalize=True)
                logger.info(f"Saved visualization images for epoch {epoch} to {vis_dir}")
            # ===================== 新增代码块结束 =====================

            # MMSegmentation 的模型期望 img 和 img_metas 都是列表
            result = seg_model(img=[enhanced_image], img_metas=[img_metas], return_loss=False)

        results.extend(result)

        if is_main_process():
            prog_bar.update()

    # 2. 在 DDP 环境下，从所有进程安全地收集结果
    if distributed:
        # 在收集前确保所有进程都已完成推理
        torch.distributed.barrier()
        # 使用 mmengine 的函数收集所有 GPU 的结果到主进程
        gathered_results = collect_results_cpu(results, len(val_dataset))
    else:
        gathered_results = results

    # 3. 只在主进程 (rank 0) 上执行评估、日志记录和 TensorBoard 写入
    metrics = None
    if is_main_process():
        logger.info(f'\nEpoch [{epoch}] Validation: Evaluating segmentation metrics...')
        # 确保只在主进程上进行评估
        metrics = val_dataset.evaluate(gathered_results, metric='mIoU', logger=logger)

        # ==================== 修复问题的核心代码块开始 ====================

        summary_metrics = {}
        detail_metrics = {}

        # 更精确地划分指标：
        # - detail_metrics: 值为 NumPy 数组且维度 > 0 (即列表)
        # - summary_metrics: 值为单个浮点数或 0 维数组 (即标量)
        for k, v in metrics.items():
            if isinstance(v, np.ndarray) and v.ndim > 0:
                detail_metrics[k] = v
            else:
                # 将 Python 浮点数、numpy 浮点数和 0 维数组都归为摘要
                summary_metrics[k] = v

        # 记录摘要指标 (mIoU, mAcc, aAcc 等)
        # 使用 float(v) 确保兼容各种数值类型
        summary_str = ", ".join([f"{k}: {float(v):.4f}" for k, v in summary_metrics.items()])
        logger.info(f"Validation Summary @ Epoch {epoch}: {summary_str}")

        # 记录详细指标 (per-class IoU, per-class Acc 等)
        for metric_name, metric_array in detail_metrics.items():
            array_str = np.array2string(metric_array, formatter={'float_kind': lambda x: f"{x:.4f}"})
            logger.info(f"  - Per-class {metric_name}: {array_str}")

        # --- 将验证指标写入 TensorBoard ---
        if writer:
            # 1. 写入摘要指标
            for key, value in summary_metrics.items():
                writer.add_scalar(f'Validation_Summary/{key}', float(value), epoch)

            # 2. 写入详细的 Per-class 指标 (现在这里是安全的，不会报错)
            for metric_name, metric_array in detail_metrics.items():
                try:
                    class_names = val_dataset.CLASSES
                    if not isinstance(class_names, (list, tuple)) or len(class_names) != len(metric_array):
                        raise ValueError("Class names are invalid or length mismatch.")

                    for i, class_name in enumerate(class_names):
                        tag = f'Validation_Per_Class/{metric_name}/{class_name}'
                        writer.add_scalar(tag, metric_array[i], epoch)

                except (TypeError, AttributeError, ValueError):
                    # 安全回退到使用索引作为标签
                    for i in range(len(metric_array)):
                        tag = f'Validation_Per_Class/{metric_name}/class_{i}'
                        writer.add_scalar(tag, metric_array[i], epoch)

        # ===================== 修复问题的核心代码块结束 =====================

    # 4. 恢复模型的训练模式
    seg_module.train()
    gen_module.train()

    return metrics

def main():
    args = parse_args()

    ## 预设
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 初始化分布式环境
    cfg.device = 'cuda'
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # 初始化日志
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    writer = None
    if (distributed and (args.local_rank == 0)) or (not distributed):
        tf_log_dir = osp.join(cfg.work_dir, 'tf_logs')
        mmcv.mkdir_or_exist(tf_log_dir)
        writer = SummaryWriter(tf_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tf_log_dir}")

    # 记录元信息
    meta = dict()
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    logger.info("loading fusionnet...")
    fusion_model = FusionNet(cfg.img_size, cfg.fusion_model)
    fusion_checkpoint = torch.load(cfg.fusion_ckpt_path, map_location='cpu')['model']
    fusion_model.load_state_dict(fusion_checkpoint, strict=True)
    logger.info(f"successfully load fusionnet ckpt from {cfg.fusion_ckpt_path}..")
    fusion_model.to(cfg.device)
    for param in fusion_model.parameters():
        param.requires_grad = False
    fusion_model.eval()

    logger.info("loading segnet...")
    seg_model = build_segmentor(
        cfg.seg_model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    seg_checkpoint = torch.load(cfg.seg_ckpt_path, map_location='cpu')['state_dict']
    seg_model.load_state_dict(seg_checkpoint, strict=True)
    logger.info(f"successfully load segnet ckpt {cfg.seg_ckpt_path}")

    logger.info("initializing generator...")
    generator_model = Generator(eps=cfg.generator_eps)
    if cfg.generator_ckpt_path:
        generator_checkpoint = torch.load(cfg.generator_ckpt_path, map_location='cpu')['model']
        generator_model.load_state_dict(generator_checkpoint, strict=True)
        logger.info(f"successfully load generator ckpt{cfg.generator_ckpt_path} ")
    else:
        logger.info("Train the generator from scratch.")

    from datasets import Train_pipline
    if cfg.datasets.dataset_name == "MFNetEnhance":
        from datasets import MFNetEnhanceDataset as RGBXDataset
    elif cfg.datasets.dataset_name == "FMBEnhance":
        from datasets import FMBEnhanceDataset as RGBXDataset

    train_process = Train_pipline(cfg)
    train_dataset = RGBXDataset(cfg, train_process, stage="train")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    num_gpus = len(cfg.gpu_ids)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size if not distributed else cfg.train.batch_size // num_gpus,
        num_workers=cfg.train.num_workers,
        drop_last=True,
        shuffle=False if distributed else True,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=custom_collate_fn
        )
    if args.local_rank == 0:
        logger.info(f'num of training dataset: {len(train_dataset)}')

    val_process = Train_pipline(cfg)
    val_dataset = RGBXDataset(cfg, val_process, stage="val")
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size if not distributed else cfg.train.batch_size // num_gpus,
        num_workers=cfg.train.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=custom_collate_fn
    )
    if args.local_rank == 0:
        logger.info(f'num of val dataset: {len(val_dataset)}')

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        seg_model = build_ddp(
            seg_model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False, find_unused_parameters=find_unused_parameters)
        generator_model = build_ddp(
            generator_model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False, find_unused_parameters=find_unused_parameters)
        setattr(seg_model, '_use_replicated_tensor_module', False)
        setattr(generator_model, '_use_replicated_tensor_module', False)
    else:
        if len(cfg.gpu_ids) > 1:
            logger.info(f"Using MMDataParallel for {len(cfg.gpu_ids)} GPUs.")
            seg_model = build_dp(seg_model, cfg.device, device_ids=cfg.gpu_ids)
            generator_model = build_dp(generator_model, cfg.device, device_ids=cfg.gpu_ids)
        else:
            logger.info("Using single GPU, skipping DataParallel wrapper.")
            seg_model.to(cfg.device)
            generator_model.to(cfg.device)

    # ====================【修改 - 优化器初始化】====================
    # 为 seg_model 和 generator_model 设置不同的学习率
    base_lr = cfg.optimizer.lr
    seg_multiplier = cfg.get('seg_lr_multiplier', 1.0) # 从配置读取倍率，如果不存在则默认为1.0
    seg_lr = base_lr * seg_multiplier
    
    if is_main_process():
        logger.info(f"Setting up optimizer with differential learning rates:")
        logger.info(f"  - Generator LR: {base_lr}")
        logger.info(f"  - SegNet LR: {seg_lr} (Base LR * {seg_multiplier})")

    optimizer = torch.optim.AdamW(
        [
            {'params': generator_model.parameters(), 'lr': base_lr},
            {'params': seg_model.parameters(), 'lr': seg_lr}
        ],
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    # =================================================================

    # === 【修改后】 ===
    # 1. 从配置文件获取总轮数，这里直接用 T_max
    total_epochs = cfg.lr_scheduler.T_max
    niters_per_epoch = len(train_dataset) // (cfg.train.batch_size)
    logger.info(f"Total training epochs set to: {total_epochs}")

    # 2. 初始化 PyTorch 的余弦退火调度器
    # 注意：T_max 是以 epoch 为单位的
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=cfg.lr_scheduler.eta_min
    )

    state_epoch = 0
    if args.resume_from is not None:
        state_dict = torch.load(args.resume_from, map_location=torch.device('cpu'))
        state_epoch = state_dict['epoch'] + 1
        seg_model.load_state_dict(state_dict['seg_model'], strict=True)
        generator_model.load_state_dict(state_dict['generator_model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        if args.local_rank == 0:
            logger.info(f"from {args.resume_from} start resume training")

    seg_model.train()
    generator_model.train()

    best_miou = 0.0
    previous_best_ckpt_path = None

    if args.local_rank == 0:
        logger.info('starting joint training')

    ## ----------------- 训练循环 -----------------
    for epoch in range(state_epoch, total_epochs + 1): # MODIFIED: 使用 total_epochs
        if distributed:
            train_sampler.set_epoch(epoch)

        # === NEW: 在每个 epoch 开始时获取当前阶段的配置 ===
        current_stage_config, current_stage_num = get_current_stage_config(epoch, cfg.training_stages)
        if is_main_process():
            logger.info(f"--- Entering Epoch {epoch}/{total_epochs} | Training Stage {current_stage_num} ---")
            logger.info(f"Stage Config: sparsity_weight={current_stage_config['sparsity_weight']}, quantization_weight={current_stage_config['quantization_weight']}")
        # ======================================================

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        sum_losses = defaultdict(float)

        if epoch < cfg.warm_up_epoch:
            # 计算 warmup 学习率
            warmup_lr_g = cfg.optimizer.lr * (epoch + 1) / cfg.warm_up_epoch
            warmup_lr_s = warmup_lr_g * seg_multiplier
            optimizer.param_groups[0]['lr'] = warmup_lr_g
            optimizer.param_groups[1]['lr'] = warmup_lr_s

        for idx in pbar:
            minibatch = next(dataloader)
            modal_x = minibatch['modal_x'].cuda(non_blocking=True)
            modal_y = minibatch['modal_y'].cuda(non_blocking=True)
            seg_label = minibatch['label'].cuda(non_blocking=True)
            img_metas = minibatch['img_metas']

            if seg_label.ndim == 3:
                seg_label = seg_label.unsqueeze(1)

            with torch.no_grad():
                fused_image = fusion_model(modal_x, modal_y)

            index_tensor, _ , _  = create_patch_index_map(
                img_height=fused_image.shape[-2], img_width=fused_image.shape[-1],
                filterSize=cfg.generator_filter_size, stride=cfg.generator_stride,
                batch_size=fused_image.shape[0], device=fused_image.device
            )
            grad_tensor =  generate_grad_box(
                model_type='res50', img_tensor= fused_image, img_height=fused_image.shape[-2],
                img_width=fused_image.shape[-1], index_tensor=index_tensor,
                filterSize=cfg.generator_filter_size, use_tk_selection=True, tk_value=0.6)

            x_inf , adv_0, adv_00 , grad_img , maskD = generator_model(modal_x, modal_y, grad = grad_tensor)
            enhanced_image = fused_image + maskD

            quantization_loss = calculate_sparsity_loss(adv_0)
            # sparsity_loss = calculate_l1_range_loss(adv_0, 5000, 10000)
            min_l1, max_l1 = cfg.l1_norm_target_range
            sparsity_loss = calculate_l1_range_loss(adv_0, min_target=min_l1, max_target=max_l1)
            # quantization_loss = calculate_quantization_loss(adv_00, grad_tensor)
            seg_losses = seg_model(img=enhanced_image, img_metas=img_metas, gt_semantic_seg=seg_label)
            seg_loss_final = sum(v for k, v in seg_losses.items() if 'loss' in k)

            # === MODIFIED: 使用当前阶段的权重计算总损失 ===
            total_loss = (current_stage_config['sparsity_weight'] * sparsity_loss +
                          current_stage_config['quantization_weight'] * quantization_loss +
                          cfg.seg_weight * seg_loss_final)
            # ===============================================

            loss_dict = {
                'total_loss': total_loss.item(),
                'sparsity_loss': sparsity_loss.item(),
                'quantization_loss': quantization_loss.item(),
                'segmentation_loss': seg_loss_final.item(),
                }
            for k, v in seg_losses.items():
                if 'aux' in k:
                    log_key = f"{k.replace('_aux', '')}"
                elif 'decode' in k :
                    log_key = f"{k.replace('_decode', '')}"
                loss_dict[log_key] = v.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            if writer:
                global_step = epoch * niters_per_epoch + idx
                for key, value in loss_dict.items():
                    tag_prefix = 'Metrics' if 'acc' in key else 'Loss'
                    writer.add_scalar(f'{tag_prefix}/{key}', value, global_step)
                writer.add_scalar('Params/learning_rate', lr, global_step)
                # ====================【新增】====================
                # 同时记录分割网络的学习率
                writer.add_scalar('Params/segnet_learning_rate', optimizer.param_groups[1]['lr'], global_step)
                # ===============================================
                avg_loss = sum_losses["total_loss"] / (idx + 1)
                writer.add_scalar('Loss/avg_total_loss_epoch', avg_loss, global_step)

            decode_loss_ce = loss_dict.get('decode.loss_ce', 0.0)
            aux_loss_ce = loss_dict.get('aux.loss_ce', 0.0)

            for key, value in loss_dict.items():
                sum_losses[key] += value
            print_str = f'Epoch {epoch}/{total_epochs}' \
                    + f' Iter {idx + 1}/{niters_per_epoch}:' \
                    + f' lr={lr:.4e}' \
                    + f' total_loss={loss_dict["total_loss"]:.4f}' \
                    + f' avg_loss={sum_losses["total_loss"] / (idx + 1):.4f}' \
                    + f' (sp={loss_dict["sparsity_loss"]:.4f}, q={loss_dict["quantization_loss"]:.4f}, seg={loss_dict["segmentation_loss"]:.4f})'\
                    + f'decode.loss_ce: {decode_loss_ce:.4f} ' \
                    + f'aux.loss_ce: {aux_loss_ce:.4f}'
            pbar.set_description(print_str, refresh=False)

        # metrics = None
        # validation_interval = cfg.validation_interval
        # if (not args.no_validate) and ((epoch % validation_interval == 0) or (epoch == total_epochs)): # MODIFIED: 使用 total_epochs
        #     metrics = run_validation(
        #         seg_model=seg_model, generator_model=generator_model, fusion_model=fusion_model,
        #         val_loader=val_loader, val_dataset=val_dataset, cfg=cfg, distributed=distributed,
        #         logger=logger, epoch=epoch, writer=writer)
        #     if distributed:
        #         torch.distributed.barrier()

        if epoch >= cfg.warm_up_epoch:
            scheduler.step()

        metrics = None
        is_normal_val_epoch = (epoch % cfg.validation_interval == 0)
        is_last_epoch = (epoch == total_epochs)
        
        # 检查是否处于密集验证区间
        is_intensive_val_epoch = False
        intensive_val_cfg = cfg.get('intensive_validation') # 安全地获取配置
        if intensive_val_cfg:
            start_epoch = intensive_val_cfg.get('start', float('inf'))
            end_epoch = intensive_val_cfg.get('end', float('-inf'))
            if start_epoch <= epoch <= end_epoch:
                is_intensive_val_epoch = True
                if is_main_process():
                    # 在首次进入密集验证区间时打印日志，方便观察
                    if epoch == start_epoch:
                        logger.info(f"--- Entering intensive validation mode from epoch {start_epoch} to {end_epoch} ---")

        # 如果满足以下任一条件，则执行验证：
        # 1. 达到常规验证间隔 (is_normal_val_epoch)
        # 2. 是训练的最后一轮 (is_last_epoch)
        # 3. 处于密集验证区间内 (is_intensive_val_epoch)
        if (not args.no_validate) and (is_normal_val_epoch or is_last_epoch or is_intensive_val_epoch):
            metrics = run_validation(
                seg_model=seg_model, generator_model=generator_model, fusion_model=fusion_model,
                val_loader=val_loader, val_dataset=val_dataset, cfg=cfg, distributed=distributed,
                logger=logger, epoch=epoch, writer=writer)
            if distributed:
                torch.distributed.barrier()

        # if (distributed and (args.local_rank == 0)) or (not distributed):
        #     if metrics is not None:
        #         current_miou = metrics.get('mIoU', 0.0)
        #         if current_miou > best_miou:
        #             best_miou = current_miou
        #             logger.info(f"🚀 New best mIoU: {best_miou:.4f} at epoch {epoch}. Saving checkpoint...")
        #             checkpoint_best = {
        #                 'seg_model': seg_model.module.state_dict() if distributed else seg_model.state_dict(),
        #                 'generator_model': generator_model.module.state_dict() if distributed else generator_model.state_dict(),
        #                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'mIoU': best_miou
        #             }
        #             best_ckpt_filename = f'best_mIoU_epoch_{epoch}.pth'
        #             current_best_ckpt_path = os.path.join(cfg.work_dir, best_ckpt_filename)
        #             torch.save(checkpoint_best, current_best_ckpt_path)
        #             logger.info(f"Successfully saved new best checkpoint to: {current_best_ckpt_path}")
        #             if previous_best_ckpt_path and os.path.exists(previous_best_ckpt_path):
        #                 os.remove(previous_best_ckpt_path)
        #                 logger.info(f"Removed previous best checkpoint: {previous_best_ckpt_path}")
        #             previous_best_ckpt_path = current_best_ckpt_path
        #             del checkpoint_best

        #     latest_epoch_checkpoint = os.path.join(cfg.work_dir, f'latest.pth')
        #     checkpoint = {
        #         'seg_model': seg_model.module.state_dict() if distributed else seg_model.state_dict(),
        #         'generator_model': generator_model.module.state_dict() if distributed else generator_model.state_dict(),
        #         'optimizer': optimizer.state_dict(), 'epoch': epoch
        #     }
        #     torch.save(checkpoint, latest_epoch_checkpoint)

        #     if (epoch >= cfg.checkpoint.start_epoch) and (epoch % cfg.checkpoint.step == 0) or (epoch == total_epochs): # MODIFIED: 使用 total_epochs
        #         current_epoch_checkpoint = os.path.join(cfg.work_dir, f'epoch-{epoch}.pth')
        #         torch.save(checkpoint, current_epoch_checkpoint)
        #         logger.info("Successfully saved periodic checkpoint to: {}".format(current_epoch_checkpoint))

        #     del checkpoint
        # torch.cuda.empty_cache()


        if (distributed and (args.local_rank == 0)) or (not distributed):
            if metrics is not None:
                current_miou = metrics.get('mIoU', 0.0)
                
                # --- 原有的“最佳mIoU”保存逻辑 ---
                if current_miou > best_miou:
                    best_miou = current_miou
                    logger.info(f"🚀 New best mIoU: {best_miou:.4f} at epoch {epoch}. Saving checkpoint...")
                    checkpoint_best = {
                        'seg_model': seg_model.module.state_dict() if distributed else seg_model.state_dict(),
                        'generator_model': generator_model.module.state_dict() if distributed else generator_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch, 'mIoU': best_miou
                    }
                    best_ckpt_filename = f'best_mIoU_epoch_{epoch}.pth'
                    current_best_ckpt_path = os.path.join(cfg.work_dir, best_ckpt_filename)
                    torch.save(checkpoint_best, current_best_ckpt_path)
                    logger.info(f"Successfully saved new best checkpoint to: {current_best_ckpt_path}")
                    if previous_best_ckpt_path and os.path.exists(previous_best_ckpt_path):
                        os.remove(previous_best_ckpt_path)
                        logger.info(f"Removed previous best checkpoint: {previous_best_ckpt_path}")
                    previous_best_ckpt_path = current_best_ckpt_path
                    del checkpoint_best

                # --- 【新增逻辑开始】当 mIoU 大于 0.61 时保存 ---
                # 注意：mIoU指标通常是0-1之间的小数，所以61%对应0.61
                if current_miou > 0.61:
                    logger.info(f"📈 mIoU ({current_miou:.4f}) exceeded 0.61! Saving special checkpoint...")
                    checkpoint_over_61 = {
                        'seg_model': seg_model.module.state_dict() if distributed else seg_model.state_dict(),
                        'generator_model': generator_model.module.state_dict() if distributed else generator_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'epoch': epoch, 
                        'mIoU': current_miou
                    }
                    over_61_filename = f'mIoU_over_61_epoch_{epoch}.pth'
                    over_61_ckpt_path = os.path.join(cfg.work_dir, over_61_filename)
                    torch.save(checkpoint_over_61, over_61_ckpt_path)
                    logger.info(f"Successfully saved special checkpoint to: {over_61_ckpt_path}")
                    del checkpoint_over_61 # 及时释放内存
                # --- 【新增逻辑结束】---

            # --- 原有的“最新”和“周期性”保存逻辑 ---
            latest_epoch_checkpoint = os.path.join(cfg.work_dir, f'latest.pth')
            checkpoint = {
                'seg_model': seg_model.module.state_dict() if distributed else seg_model.state_dict(),
                'generator_model': generator_model.module.state_dict() if distributed else generator_model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch
            }
            torch.save(checkpoint, latest_epoch_checkpoint)

            if (epoch >= cfg.checkpoint.start_epoch) and (epoch % cfg.checkpoint.step == 0) or (epoch == total_epochs):
                current_epoch_checkpoint = os.path.join(cfg.work_dir, f'epoch-{epoch}.pth')
                torch.save(checkpoint, current_epoch_checkpoint)
                logger.info("Successfully saved periodic checkpoint to: {}".format(current_epoch_checkpoint))

            del checkpoint
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    main()