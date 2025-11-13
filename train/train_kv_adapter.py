# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
训练 KeyValueAdapter：将理解空间映射到生成空间

训练流程：
1. 输入原图片 -> 理解模式 -> past_key_values
2. 应用 Adapter -> adapted_past_key_values  
3. 输入编辑指令 + adapted_kv -> 生成模式 -> 生成图片
4. 计算生成图片与 ground truth 的 MSE 损失
5. 仅更新 Adapter 参数
"""

import functools
import os
import wandb
import yaml
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, 
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


@dataclass
class AdapterModelArguments:
    """模型相关参数"""
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "预训练 BAGEL 模型路径"}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "预训练语言模型路径"}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "VAE 模型路径"}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "ViT 模型路径"}
    )
    
    # Adapter 配置
    use_kv_adapter: bool = field(
        default=True,
        metadata={"help": "启用 KV Adapter"}
    )
    num_adapter_layers: int = field(
        default=8,
        metadata={"help": "Adapter 层数（32层模型推荐8，可选：1/4/8/16/32）"}
    )
    
    # 模型架构参数
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "解码器层类型"}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "最大 latent 尺寸"}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Latent patch 大小"}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "ViT 每边最大 patch 数"}
    )


@dataclass
class AdapterTrainingArguments:
    """训练相关参数"""
    # 数据集配置
    dataset_config_file: str = field(
        default="./data/configs/adapter_train.yaml",
        metadata={"help": "数据集配置文件"}
    )
    
    # 训练超参数
    total_steps: int = field(
        default=50000,
        metadata={"help": "总训练步数"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "每个 GPU 的 batch size"}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "学习率（Adapter 可以用较大学习率）"}
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "学习率预热步数"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪"}
    )
    
    # 损失权重
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "MSE 损失权重"}
    )
    
    # EMA
    ema: float = field(
        default=0.999,
        metadata={"help": "EMA 衰减率"}
    )
    
    # 日志和保存
    results_dir: str = field(
        default="./results/adapter_training",
        metadata={"help": "结果保存目录"}
    )
    checkpoint_dir: str = field(
        default="./checkpoints/adapter",
        metadata={"help": "检查点保存目录"}
    )
    log_every: int = field(
        default=50,
        metadata={"help": "日志记录间隔"}
    )
    save_every: int = field(
        default=5000,
        metadata={"help": "保存间隔"}
    )
    
    # 分布式训练
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "数据加载器 worker 数"}
    )
    
    # FSDP 配置
    use_fsdp: bool = field(
        default=True,
        metadata={"help": "使用 FSDP"}
    )
    use_activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "使用激活检查点（Adapter 训练通常不需要）"}
    )
    
    # Wandb
    use_wandb: bool = field(
        default=True,
        metadata={"help": "使用 Wandb"}
    )
    wandb_project: str = field(
        default="bagel-adapter-training",
        metadata={"help": "Wandb 项目名"}
    )
    
    # 恢复训练
    resume_from: str = field(
        default=None,
        metadata={"help": "从检查点恢复训练"}
    )
    freeze_main_model: bool = field(
        default=True,
        metadata={"help": "冻结主模型参数，只训练 Adapter"}
    )


def freeze_main_model_parameters(model):
    """冻结主模型参数，只保留 Adapter 可训练"""
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'kv_adapter' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    return frozen_params, trainable_params


def main():
    assert torch.cuda.is_available()
    
    # 解析参数
    parser = HfArgumentParser((AdapterModelArguments, AdapterTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # 初始化分布式
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 创建日志和目录
    if rank == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir)
        logger.info("=" * 80)
        logger.info("KeyValueAdapter 训练")
        logger.info("=" * 80)
        logger.info(f"模型参数: {model_args}")
        logger.info(f"训练参数: {training_args}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Batch size per GPU: {training_args.batch_size}")
        logger.info(f"Total batch size: {training_args.batch_size * world_size}")
        
        if training_args.use_wandb:
            wandb.init(
                project=training_args.wandb_project,
                config={
                    "model_args": model_args.__dict__,
                    "training_args": training_args.__dict__,
                }
            )
    
    dist.barrier()
    
    # ==================== 加载模型 ====================
    if rank == 0:
        logger.info("加载模型...")
    
    # 加载 tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.llm_path)
    special_tokens = add_special_tokens(tokenizer)
    
    # 加载 VAE
    vae_model = load_ae(model_args.vae_path).to(device).to(torch.bfloat16)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    
    # 加载主模型
    if os.path.exists(model_args.model_path) and os.path.isdir(model_args.model_path):
        # 从已有的 BAGEL 模型加载
        if rank == 0:
            logger.info(f"从 {model_args.model_path} 加载预训练模型")
        
        bagel_config = BagelConfig.from_pretrained(model_args.model_path)
        # 启用 Adapter
        bagel_config.use_kv_adapter = model_args.use_kv_adapter
        bagel_config.num_adapter_layers = model_args.num_adapter_layers
        
        model = Bagel.from_pretrained(
            model_args.model_path,
            config=bagel_config,
        )
    else:
        # 从头创建模型
        if rank == 0:
            logger.info("创建新模型")
        
        # 加载 LLM
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
        llm_config.qk_norm = True
        llm_config.layer_module = model_args.layer_module
        language_model = Qwen2ForCausalLM.from_pretrained(
            model_args.llm_path, 
            config=llm_config
        )
        language_model.resize_token_embeddings(len(tokenizer))
        
        # 加载 ViT
        vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_model = SiglipVisionModel.from_pretrained(
            model_args.vit_path,
            config=vit_config
        )
        
        # 创建 BAGEL config
        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_model.config,
            latent_patch_size=model_args.latent_patch_size,
            max_latent_size=model_args.max_latent_size,
            vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            use_kv_adapter=model_args.use_kv_adapter,
            num_adapter_layers=model_args.num_adapter_layers,
        )
        
        model = Bagel(language_model, vit_model, bagel_config)
    
    model = model.to(device)
    
    # 冻结主模型参数
    if training_args.freeze_main_model:
        frozen_params, trainable_params = freeze_main_model_parameters(model)
        if rank == 0:
            logger.info(f"冻结参数: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
            logger.info(f"可训练参数 (Adapter): {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            logger.info(f"参数比例: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")
    
    # ==================== FSDP 包装 ====================
    if training_args.use_fsdp:
        if rank == 0:
            logger.info("应用 FSDP...")
        
        fsdp_config = FSDPConfig()
        fsdp_model = fsdp_wrapper(
            model,
            fsdp_config,
            device_id=device,
        )
        
        # EMA 模型
        ema_model = fsdp_ema_setup(fsdp_model)
    else:
        fsdp_model = model
        ema_model = None
    
    # ==================== 数据加载 ====================
    if rank == 0:
        logger.info("加载数据集...")
    
    with open(training_args.dataset_config_file, 'r') as f:
        dataset_config = DataConfig(**yaml.safe_load(f))
    
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer,
        special_tokens,
        vae_model,
        enable_vit=True,
        enable_vae=True,
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=training_args.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.batch_size,
        sampler=train_sampler,
        num_workers=training_args.num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True,
        drop_last=True,
    )
    
    if rank == 0:
        logger.info(f"数据集大小: {len(train_dataset)}")
        logger.info(f"每个 epoch 步数: {len(train_loader)}")
    
    # ==================== 优化器和调度器 ====================
    # 只优化 Adapter 参数
    trainable_params = [p for p in fsdp_model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=training_args.weight_decay
    )
    
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.total_steps,
        min_lr=training_args.lr * 0.1,
    )
    
    # ==================== 恢复训练 ====================
    start_step = 0
    if training_args.resume_from:
        if rank == 0:
            logger.info(f"从 {training_args.resume_from} 恢复训练")
        
        checkpoint = torch.load(training_args.resume_from, map_location=device)
        fsdp_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
        
        if rank == 0:
            logger.info(f"从步数 {start_step} 恢复")
    
    # ==================== 训练循环 ====================
    if rank == 0:
        logger.info("=" * 80)
        logger.info("开始训练")
        logger.info("=" * 80)
    
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()
    
    start_time = time()
    global_step = start_step
    
    data_iter = iter(train_loader)
    
    for step in range(start_step, training_args.total_steps):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data = next(data_iter)
        
        # 移动到 GPU
        data = data.cuda(device).to_dict()
        
        # 前向传播
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)
            
            # 计算 MSE 损失
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            
            loss = mse * training_args.mse_weight
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # 更新 EMA
        if ema_model is not None:
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
        
        # 日志记录
        if step % training_args.log_every == 0:
            elapsed_time = time() - start_time
            steps_per_sec = (step - start_step + 1) / elapsed_time
            
            if rank == 0:
                logger.info(
                    f"Step {step}/{training_args.total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"MSE: {mse.item():.4f} | "
                    f"GradNorm: {total_norm:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Steps/s: {steps_per_sec:.2f}"
                )
                
                if training_args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/mse": mse.item(),
                        "train/grad_norm": total_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/steps_per_sec": steps_per_sec,
                    }, step=step)
        
        # 保存检查点
        if (step + 1) % training_args.save_every == 0 or step == training_args.total_steps - 1:
            if rank == 0:
                checkpoint_path = os.path.join(
                    training_args.checkpoint_dir,
                    f"adapter_step_{step+1}.pt"
                )
                
                # 只保存 Adapter 参数
                adapter_state_dict = {
                    name: param for name, param in fsdp_model.state_dict().items()
                    if 'kv_adapter' in name
                }
                
                torch.save({
                    'step': step + 1,
                    'model': fsdp_model.state_dict(),
                    'adapter_only': adapter_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': model_args.__dict__,
                }, checkpoint_path)
                
                logger.info(f"保存检查点: {checkpoint_path}")
                logger.info(f"Adapter 参数数量: {sum(p.numel() for p in adapter_state_dict.values()):,}")
        
        global_step += 1
    
    # 训练完成
    if rank == 0:
        total_time = time() - start_time
        logger.info("=" * 80)
        logger.info("训练完成！")
        logger.info(f"总时间: {total_time/3600:.2f} 小时")
        logger.info(f"最终损失: {loss.item():.4f}")
        logger.info("=" * 80)
        
        if training_args.use_wandb:
            wandb.finish()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

