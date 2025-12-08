# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import wandb
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import FSDPCheckpoint, FSDPConfig
from train.config import ModelArguments, DataArguments, TrainingArguments
from train.utils import detect_peak_tflops
from train.model_setup import (
    create_model, setup_tokenizer, prepare_fsdp_config, 
    wrap_model_with_fsdp, freeze_parameters
)
from train.train_loop import training_loop
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType


def setup_optimizer_and_scheduler(fsdp_model, training_args: TrainingArguments, logger, device):
    """Setup optimizer and scheduler."""
    if training_args.freeze_all_except_adapter:
        trainable_params = []
        if hasattr(fsdp_model.module, 'adapter'):
            adapter_params = list(fsdp_model.module.adapter.parameters())
            trainable_params.extend(adapter_params)
            if dist.get_rank() == 0:
                logger.info(f"Found {len(adapter_params)} adapter parameter groups")
        else:
            raise ValueError("Adapter module not found! Cannot set up optimizer.")
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        
        local_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Optimizer will update {local_count / 1e6:.2f}M parameters")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_args.lr, 
            betas=(training_args.beta1, training_args.beta2), 
            eps=training_args.eps, 
            weight_decay=0
        )
    else:
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(), 
            lr=training_args.lr, 
            betas=(training_args.beta1, training_args.beta2), 
            eps=training_args.eps, 
            weight_decay=0
        )
    
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")
    
    return optimizer, scheduler


def setup_ema_model(fsdp_model, model_args: ModelArguments, training_args: TrainingArguments, 
                    config, fsdp_config: FSDPConfig, resume_from, finetune_from_ema, logger):
    """Setup EMA model if needed."""
    from train.fsdp_utils import fsdp_ema_setup
    from modeling.bagel import Bagel, Qwen2ForCausalLM, SiglipVisionModel
    from modeling.bagel import Qwen2Config, SiglipVisionConfig
    import os
    
    ema_model = None
    if training_args.freeze_all_except_adapter:
        logger.info("EMA disabled for adapter-only training to save memory")
    elif training_args.ema <= 0:
        logger.info("EMA disabled (ema <= 0)")
    else:
        logger.info("Creating EMA model...")
        device = torch.device(f'cuda:{dist.get_rank() % torch.cuda.device_count()}')
        
        if training_args.finetune_from_hf:
            llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
            ema_language_model = Qwen2ForCausalLM(llm_config)
            ema_language_model = ema_language_model.to(device)
        else:
            ema_language_model = Qwen2ForCausalLM.from_pretrained(
                model_args.llm_path, config=config.llm_config, torch_dtype=torch.bfloat16
            )
            ema_language_model = ema_language_model.to(device)
        if training_args.copy_init_moe:
            ema_language_model.init_moe()
        
        ema_vit_model = None
        if training_args.visual_und:
            if training_args.finetune_from_hf:
                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
                ema_vit_model = SiglipVisionModel(vit_config)
                ema_vit_model = ema_vit_model.to(device)
            else:
                ema_vit_model = SiglipVisionModel.from_pretrained(
                    model_args.vit_path, config=config.vit_config, torch_dtype=torch.bfloat16
                )
                ema_vit_model = ema_vit_model.to(device)
        
        ema_model = Bagel(ema_language_model, ema_vit_model, config)
        if training_args.visual_und:
            ema_model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(config.vit_config)
        
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
        
        # Copy parameters from main model
        logger.info("Copying parameters from main model to EMA model...")
        try:
            with FSDP.state_dict_type(fsdp_model, StateDictType.LOCAL_STATE_DICT):
                main_state_dict = fsdp_model.state_dict()
            with FSDP.state_dict_type(ema_model, StateDictType.LOCAL_STATE_DICT):
                ema_model.load_state_dict(main_state_dict, strict=False)
            del main_state_dict
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Failed to copy parameters to EMA model: {e}")
        
        if resume_from is not None and os.path.exists(resume_from):
            _, ema_model = FSDPCheckpoint.try_load_ckpt(
                resume_from, logger, None, ema_model, resume_from_ema=finetune_from_ema
            )
    
    return ema_model


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    dist.barrier()
    torch.cuda.synchronize()
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
    else:
        logger = create_logger(None, dist.get_rank())
    
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # Resume logic
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False

    # Set seed
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Create model
    model, vae_model, vae_config = create_model(model_args, training_args, logger)
    
    # Setup tokenizer
    tokenizer, new_token_ids = setup_tokenizer(model_args, training_args, model, logger)
    
    # Prepare FSDP config
    fsdp_config = prepare_fsdp_config(training_args, logger)
    
    # Load checkpoint
    model, _ = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, None, resume_from_ema=finetune_from_ema
    )
    
    # Wrap model with FSDP
    fsdp_model = wrap_model_with_fsdp(model, fsdp_config, training_args, logger)
    
    # Freeze parameters
    freeze_parameters(fsdp_model, vae_model, training_args, logger)
    
    # Setup EMA model
    ema_model = setup_ema_model(
        fsdp_model, model_args, training_args, model.config, 
        fsdp_config, resume_from, finetune_from_ema, logger
    )

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(fsdp_model, training_args, logger, device)
    
    # Resume optimizer/scheduler
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup dataset
    dist.barrier()
    torch.cuda.synchronize()
    
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    
    dist.barrier()
    torch.cuda.synchronize()
    
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': data_args.num_workers,
        'pin_memory': True,
        'collate_fn': collate_wrapper(),
        'drop_last': True,
    }
    if data_args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = data_args.prefetch_factor
    
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    
    dist.barrier()
    torch.cuda.synchronize()

    # Prepare models for training
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # Critical: Warm up FSDP by doing a dummy forward pass to initialize FSDP state
    # This helps avoid NCCL errors on first real forward pass
    logger.info("Warming up FSDP with dummy forward pass...")
    try:
        # Create minimal dummy data for warmup
        dummy_data = {
            'packed_text_ids': torch.zeros(1, dtype=torch.long, device=device),
            'packed_text_indexes': torch.zeros(1, dtype=torch.long, device=device),
            'packed_position_ids': torch.zeros(1, dtype=torch.long, device=device),
            'sample_lens': [1],
            'sequence_length': 1,
        }
        if training_args.visual_und:
            dummy_data['packed_vit_tokens'] = torch.zeros(1, 588, device=device)
            dummy_data['packed_vit_token_indexes'] = torch.zeros(1, dtype=torch.long, device=device)
            dummy_data['packed_vit_position_ids'] = torch.zeros(1, dtype=torch.long, device=device)
            dummy_data['vit_token_seqlens'] = torch.tensor([1], dtype=torch.int, device=device)
        if training_args.visual_gen:
            dummy_data['padded_latent'] = torch.zeros(1, 16, 8, 8, device=device)
            dummy_data['packed_vae_token_indexes'] = torch.zeros(1, dtype=torch.long, device=device)
            dummy_data['packed_latent_position_ids'] = torch.zeros(1, dtype=torch.long, device=device)
            dummy_data['packed_timesteps'] = torch.zeros(1, dtype=torch.long, device=device)
        
        dist.barrier()
        torch.cuda.synchronize()

        with torch.no_grad():
            _ = fsdp_model(**dummy_data)
        
        dist.barrier()
        torch.cuda.synchronize()
        logger.info("FSDP warmup completed successfully")
    except Exception as e:
        logger.warning(f"FSDP warmup failed (may be non-critical): {e}")
        torch.cuda.empty_cache()
        dist.barrier()
        torch.cuda.synchronize()

    # Training loop
    training_loop(
        fsdp_model, vae_model, tokenizer, new_token_ids, train_loader,
        optimizer, scheduler, ema_model, training_args,
        train_step, data_status, fsdp_config, device, logger, model
    )
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
