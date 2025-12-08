# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, 
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)
from train.utils import count_parameters
from train.config import ModelArguments, TrainingArguments


def create_model(model_args: ModelArguments, training_args: TrainingArguments, logger):
    """Create and initialize the Bagel model."""
    # Setup LLM
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    # Setup ViT
    vit_model = None
    vit_config = None
    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    # Setup VAE
    vae_model = None
    vae_config = None
    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    # Create Bagel model
    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        num_adapter_layers=model_args.num_adapter_layers,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    return model, vae_model, vae_config


def setup_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, model, logger):
    """Setup tokenizer and resize embeddings if needed."""
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_path if training_args.finetune_from_hf else model_args.llm_path
    )
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    return tokenizer, new_token_ids


def prepare_fsdp_config(training_args: TrainingArguments, logger):
    """Prepare FSDP configuration based on world size."""
    world_size = dist.get_world_size()
    if training_args.sharding_strategy == 'HYBRID_SHARD':
        max_shard = world_size // training_args.num_replicate
        if max_shard < 1:
            raise ValueError(f"num_replicate ({training_args.num_replicate}) must be <= world_size ({world_size})")
        adjusted_num_shard = min(training_args.num_shard, max_shard)
        if adjusted_num_shard != training_args.num_shard:
            logger.warning(f"Adjusting num_shard from {training_args.num_shard} to {adjusted_num_shard}")
        num_shard = adjusted_num_shard
        
        if training_args.num_replicate * num_shard != world_size:
            num_replicate = world_size // num_shard
            if num_replicate * num_shard != world_size:
                logger.warning("Cannot satisfy HYBRID_SHARD constraint. Switching to FULL_SHARD.")
                training_args.sharding_strategy = 'FULL_SHARD'
                num_replicate = training_args.num_replicate
            else:
                logger.warning(f"Adjusting num_replicate from {training_args.num_replicate} to {num_replicate}")
                training_args.num_replicate = num_replicate
        else:
            num_replicate = training_args.num_replicate
    else:
        num_shard = training_args.num_shard
        num_replicate = training_args.num_replicate
    
    return FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=num_replicate,
        num_shard=num_shard,
    )


def wrap_model_with_fsdp(model, fsdp_config: FSDPConfig, training_args: TrainingArguments, logger):
    """Wrap model with FSDP and apply activation checkpointing."""
    # Ensure model is on CPU
    try:
        first_param = next(model.parameters())
        if first_param.is_cuda:
            logger.warning("Model parameters are on CUDA before FSDP wrapping, moving to CPU")
            model = model.cpu()
            torch.cuda.empty_cache()
    except StopIteration:
        logger.warning("Model has no parameters")
    
    torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.synchronize()
    
    # Setup ignored modules for adapter-only training
    ignored_modules = []
    if training_args.freeze_all_except_adapter:
        if hasattr(model, 'adapter'):
            ignored_modules = [model.adapter]
            logger.info("Excluding adapter from FSDP sharding for adapter-only training")
    
    # Wrap with FSDP
    fsdp_model = fsdp_wrapper(model, fsdp_config, ignored_modules=ignored_modules)
    
    dist.barrier()
    torch.cuda.synchronize()
    
    # Apply activation checkpointing
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )
    
    dist.barrier()
    torch.cuda.synchronize()
    
    return fsdp_model


def freeze_parameters(fsdp_model, vae_model, training_args: TrainingArguments, logger):
    """Freeze model parameters according to training arguments."""
    if training_args.freeze_all_except_adapter:
        logger.info("Freezing all model parameters except adapter...")
        dist.barrier()
        torch.cuda.synchronize()
        
        # Freeze all FSDP-managed parameters
        for param in fsdp_model.parameters():
            param.requires_grad = False
        
        # Unfreeze adapter parameters
        adapter_param_count = 0
        adapter_total_elements = 0
        
        try:
            adapter_module = fsdp_model.module.adapter
            for name, param in adapter_module.named_parameters():
                param.requires_grad = True
                adapter_param_count += 1
                adapter_total_elements += param.numel()
                if dist.get_rank() == 0 and adapter_param_count <= 5:
                    logger.info(f"  Keeping adapter.{name} trainable (shape: {param.shape}, numel: {param.numel()})")
            
            if adapter_param_count == 0:
                raise ValueError("Adapter module found but has no parameters!")
            
            if dist.get_rank() == 0:
                logger.info(f"Found {adapter_param_count} adapter parameter groups with {adapter_total_elements / 1e6:.2f}M total elements")
        except AttributeError:
            raise ValueError("Adapter module not found in model!")
        
        # Freeze VAE
        if training_args.visual_gen:
            for param in vae_model.parameters():
                param.requires_grad = False
        
        # Set models to eval mode
        fsdp_model.module.language_model.eval()
        if training_args.visual_und:
            fsdp_model.module.vit_model.eval()
        
        # Count trainable parameters
        adapter_trainable_count = adapter_total_elements
        fsdp_trainable_count = sum(p.numel() for p in fsdp_model.parameters() if p.requires_grad)
        fsdp_total_params = sum(p.numel() for p in fsdp_model.parameters())
        
        fsdp_trainable_tensor = torch.tensor([fsdp_trainable_count], dtype=torch.long, 
                                            device=f'cuda:{dist.get_rank() % torch.cuda.device_count()}')
        fsdp_total_tensor = torch.tensor([fsdp_total_params], dtype=torch.long, 
                                        device=f'cuda:{dist.get_rank() % torch.cuda.device_count()}')
        dist.all_reduce(fsdp_trainable_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(fsdp_total_tensor, op=dist.ReduceOp.SUM)
        total_fsdp_trainable = fsdp_trainable_tensor.item()
        total_fsdp_params = fsdp_total_tensor.item()
        
        total_trainable_count = adapter_trainable_count + total_fsdp_trainable
        total_model_params = adapter_trainable_count + total_fsdp_params
        
        logger.info(f"Trainable parameters: {total_trainable_count / 1e6:.2f}M / {total_model_params / 1e9:.2f}B "
                   f"({100 * total_trainable_count / total_model_params:.4f}%)")
        logger.info(f"  - Adapter (not sharded): {adapter_trainable_count / 1e6:.2f}M")
        logger.info(f"  - FSDP-managed (sharded): {total_fsdp_trainable / 1e6:.2f}M / {total_fsdp_params / 1e9:.2f}B")
        
        if total_trainable_count == 0:
            raise ValueError("No trainable parameters found after freezing!")
        
        dist.barrier()
        torch.cuda.synchronize()
    else:
        # Original freezing logic
        if training_args.freeze_vae and training_args.visual_gen:
            for param in vae_model.parameters():
                param.requires_grad = False
        if training_args.freeze_llm:
            fsdp_model.module.language_model.eval()
            for param in fsdp_model.module.language_model.parameters():
                param.requires_grad = False
        if training_args.freeze_vit and training_args.visual_und:
            fsdp_model.module.vit_model.eval()
            for param in fsdp_model.module.vit_model.parameters():
                param.requires_grad = False

