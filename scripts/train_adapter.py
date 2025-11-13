#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# KeyValueAdapter 训练脚本
# 
# 功能：训练 Adapter 将理解空间映射到生成空间
# 输入：原图片 + 编辑后图片 (ground truth)
# 损失：Flow Matching MSE 损失
# 优化：仅训练 Adapter 参数，主模型冻结
# ============================================================================

# ========== 环境变量配置 ==========
# 请根据实际情况修改以下变量

# 分布式训练配置
export num_nodes=1              # 节点数
export node_rank=0              # 当前节点编号 (0-based)
export master_addr="localhost"  # 主节点地址
export master_port=29500        # 主节点端口

# 模型路径
export vae_path="flux/vae/ae.safetensors"
export vit_path="hf/siglip-so400m-14-980-flash-attn2-navit/"
export llm_path="hf/Qwen2.5-0.5B-Instruct/"
export model_path="hf/BAGEL-7B-MoT"  # 预训练模型路径（可选）

# 数据配置
export dataset_config="./data/configs/adapter_train.yaml"

# 输出路径
export output_path="./results/adapter_training"
export ckpt_path="./checkpoints/adapter"

# 恢复训练（可选）
# export resume_from="./checkpoints/adapter/adapter_step_5000.pt"

# ========== Adapter 配置 ==========
# num_adapter_layers 控制参数量：
# - 8:  推荐配置，参数量减少 75%  (每4层共享)
# - 16: 参数量减少 50%  (每2层共享)
# - 4:  参数量减少 87.5% (每8层共享)
# - 1:  参数量减少 96.9% (所有层共享)
export num_adapter_layers=8

# ========== 训练超参数 ==========
export batch_size=4            # 每个GPU的batch size
export lr=1e-4                 # 学习率（Adapter可以用较大学习率）
export total_steps=50000       # 总训练步数
export warmup_steps=1000       # 预热步数
export weight_decay=0.01       # 权重衰减
export max_grad_norm=1.0       # 梯度裁剪

# ========== 日志和保存 ==========
export log_every=50            # 日志记录间隔
export save_every=5000         # 保存检查点间隔

# ========== Wandb（可选）==========
export use_wandb=true
export wandb_project="bagel-adapter-training"

# ========== 运行训练 ==========
echo "======================================================================"
echo "KeyValueAdapter 训练"
echo "======================================================================"
echo "配置信息："
echo "  - 节点数: $num_nodes"
echo "  - 每节点GPU数: 8"
echo "  - Batch size per GPU: $batch_size"
echo "  - 总 batch size: $((batch_size * 8 * num_nodes))"
echo "  - Adapter 层数: $num_adapter_layers"
echo "  - 学习率: $lr"
echo "  - 总步数: $total_steps"
echo "  - 输出目录: $output_path"
echo "======================================================================"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/train_kv_adapter.py \
  --dataset_config_file $dataset_config \
  --model_path $model_path \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_kv_adapter true \
  --num_adapter_layers $num_adapter_layers \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --batch_size $batch_size \
  --lr $lr \
  --total_steps $total_steps \
  --warmup_steps $warmup_steps \
  --weight_decay $weight_decay \
  --max_grad_norm $max_grad_norm \
  --mse_weight 1.0 \
  --ema 0.999 \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --log_every $log_every \
  --save_every $save_every \
  --use_wandb $use_wandb \
  --wandb_project $wandb_project \
  --freeze_main_model true \
  --num_workers 4 \
  --seed 42

echo "======================================================================"
echo "训练完成！"
echo "结果保存在: $output_path"
echo "检查点保存在: $ckpt_path"
echo "======================================================================"

