# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import wandb
import torch
import torch.distributed as dist
from time import time
from torch.utils.data import DataLoader

from train.fsdp_utils import FSDPCheckpoint, fsdp_ema_update
from train.utils import qwen2_flop_coefficients
from train.config import TrainingArguments

ANALYSIS_PROMPT = """Please provide improvement suggestions for the following image, focusing only on the inaccuracies in quantity, spatial positioning, and physical laws:
Correct any discrepancies in the number of data points or objects depicted.
Adjust the spatial positioning of elements to align with the correct reference points.
Ensure the representation of physical laws is consistent with established principles."""


def ensure_data_on_device(data, device, logger=None):
    """Ensure all tensors in data dict are on the correct device."""
    device_errors = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.device != device:
                if logger and dist.get_rank() == 0:
                    logger.warning(f"Moving {key} from {value.device} to {device}")
                data[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and len(value) > 0:
            # Handle lists of tensors (like nested_attention_masks)
            for i, item in enumerate(value):
                if isinstance(item, torch.Tensor) and item.device != device:
                    if logger and dist.get_rank() == 0:
                        logger.warning(f"Moving {key}[{i}] from {item.device} to {device}")
                    value[i] = item.to(device, non_blocking=True)
    
    # Synchronize after moving data
    torch.cuda.synchronize()
    return data


def train_one_step(
    fsdp_model, vae_model, tokenizer, new_token_ids, data, 
    training_args: TrainingArguments, device, logger
):
    """Perform one training step."""
    # Ensure all data is on the correct device before processing
    data = ensure_data_on_device(data, device, logger)
    
    # Encode images with VAE
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        if training_args.visual_gen:
            with torch.no_grad():
                padded_images = data.pop('padded_images')
                # Ensure images are on the same device as VAE model
                if padded_images.device != device:
                    padded_images = padded_images.to(device, non_blocking=True)
                torch.cuda.synchronize()  # Sync before VAE encoding
                
                padded_latent = vae_model.encode(padded_images)
                # Ensure latent is on the correct device
                if padded_latent.device != device:
                    padded_latent = padded_latent.to(device, non_blocking=True)
                data['padded_latent'] = padded_latent
                torch.cuda.synchronize()  # Sync after VAE encoding
        
        # Ensure all data is still on device after VAE encoding
        data = ensure_data_on_device(data, device, logger)
        
        # Critical: Ensure all CUDA operations are complete before FSDP operations
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        
        try:
            past_key_values = None
            if training_args.visual_und and 'packed_vit_tokens' in data:
                analysis_prompt_ids = tokenizer.encode(ANALYSIS_PROMPT)
                original_prompt_ids = data['packed_text_ids'].tolist()
                analysis_prompt_ids = original_prompt_ids + analysis_prompt_ids + [new_token_ids['eos_token_id']]
                
                # Ensure data is on device before generating past_key_values
                data = ensure_data_on_device(data, device, logger)
                
                # Critical synchronization before first FSDP call
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.synchronize()
                
                # Ensure FSDP model is in correct state
                if hasattr(fsdp_model, 'module'):
                    # Touch adapter to ensure it's initialized
                    if hasattr(fsdp_model.module, 'adapter'):
                        _ = fsdp_model.module.adapter
                
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.synchronize()
                
                with torch.no_grad():
                    past_key_values = fsdp_model(
                        generate_analysis_cache=True,
                        analysis_prompt_ids=analysis_prompt_ids,
                        new_token_ids=new_token_ids,
                        **data
                    )        
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.synchronize()

            # Final check before main forward pass
            data = ensure_data_on_device(data, device, logger)
            
            # Critical synchronization before main forward
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            
            loss_dict = fsdp_model(past_key_values=past_key_values, **data)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM: {e}")
                torch.cuda.empty_cache()
            elif "illegal memory access" in str(e).lower() or "cuda error" in str(e).lower():
                logger.error(f"CUDA memory access error: {e}")
                logger.error("This may indicate data device mismatch. Checking data devices...")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        logger.error(f"  {key}: device={value.device}, shape={value.shape}")
                torch.cuda.empty_cache()
            raise e
    
    return loss_dict


def compute_loss(loss_dict, data, training_args: TrainingArguments, device):
    """Compute and aggregate loss across all processes."""
    loss = 0
    
    # CE loss
    ce = loss_dict["ce"]
    if ce is not None:
        total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
        dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        if training_args.ce_loss_reweighting:
            ce_loss_weights = data.get('ce_loss_weights')
            ce = ce * ce_loss_weights
            total_ce_loss_weights = ce_loss_weights.sum()
            dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
            ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
        else:
            ce = ce.sum() * dist.get_world_size() / total_ce_tokens
        loss_dict["ce"] = ce.detach()
        loss = loss + ce * training_args.ce_weight
    else:
        loss_dict["ce"] = torch.tensor(0, device=device)
    
    # MSE loss
    if training_args.visual_gen:
        mse = loss_dict["mse"]
        total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
        dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
        mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
        loss_dict["mse"] = mse.detach()
        loss = loss + mse * training_args.mse_weight
    else:
        loss_dict["mse"] = torch.tensor(0, device=device)
    
    return loss / training_args.gradient_accumulation_steps, loss_dict


def log_metrics(
    curr_step, loss_dict, token_window, seqlen_square_window, 
    start_time, training_args: TrainingArguments, device, logger, 
    total_norm, total_samples, total_ce_tokens, total_mse_tokens,
    dense_token_factor, attn_factor
):
    """Log training metrics."""
    torch.cuda.synchronize()
    end_time = time()
    elapsed = max(end_time - start_time, 1e-6)
    steps_per_sec = training_args.log_every / elapsed
    tokens_per_sec = token_window / elapsed
    tokens_per_step = token_window / training_args.log_every
    
    # Calculate MFU (Model FLOPS Utilization)
    flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
    actual_tflops = flops_all_token / elapsed / 1e12
    peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
    mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
    
    message = f"(step={curr_step:07d}) "
    wandb_log = {}
    for key, value in loss_dict.items():
        avg_loss = torch.tensor(value.item(), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
        message += f"Train Loss {key}: {avg_loss:.4f}, "
        wandb_log[key] = avg_loss
    
    message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%"
    logger.info(message)
    if dist.get_rank() == 0:
        print(message, flush=True)
    
    wandb_log['total_mse_tokens'] = total_mse_tokens.item() if isinstance(total_mse_tokens, torch.Tensor) else total_mse_tokens
    wandb_log['total_ce_tokens'] = total_ce_tokens.item() if isinstance(total_ce_tokens, torch.Tensor) else total_ce_tokens
    wandb_log['total_norm'] = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
    wandb_log['total_samples'] = total_samples.item() if isinstance(total_samples, torch.Tensor) else total_samples
    wandb_log['tokens_per_sec'] = tokens_per_sec
    wandb_log['tokens_per_step'] = tokens_per_step
    wandb_log['actual_tflops'] = actual_tflops
    wandb_log['mfu'] = mfu_value
    
    mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
    dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
    wandb_log['mem_allocated'] = mem_allocated
    mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
    dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
    wandb_log['mem_cache'] = mem_cache
    
    if dist.get_rank() == 0:
        wandb.log(wandb_log, step=curr_step)
    
    return time(), 0.0, 0.0  # Return new start_time, reset token_window, seqlen_square_window


def save_checkpoint(
    curr_step, fsdp_model, ema_model, optimizer, scheduler, 
    data_status, training_args: TrainingArguments, fsdp_config, logger
):
    """Save training checkpoint."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    if dist.get_rank() == 0:
        gather_list = [None] * dist.get_world_size()
    else:
        gather_list = None
    
    try:
        dist.gather_object(data_status, gather_list, dst=0)
    except RuntimeError as e:
        logger.error(f"Error during gather_object at step {curr_step}: {e}")
        gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
    
    FSDPCheckpoint.fsdp_save_ckpt(
        ckpt_dir=training_args.checkpoint_dir, 
        train_steps=curr_step, 
        model=fsdp_model, 
        ema_model=ema_model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        logger=logger,
        fsdp_config=fsdp_config,
        data_status=gather_list
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def training_loop(
    fsdp_model, vae_model, tokenizer, new_token_ids, train_loader,
    optimizer, scheduler, ema_model, training_args: TrainingArguments,
    train_step, data_status, fsdp_config, device, logger, model
):
    """Main training loop."""
    from train.fsdp_utils import fsdp_ema_update
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    
    dist.barrier()
    torch.cuda.synchronize()
    
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        
        # Move data to device and convert to dict
        data = data.cuda(device).to_dict()
        
        # Ensure all tensors are on the correct device after to_dict()
        data = ensure_data_on_device(data, device)
        
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        if ce_loss_weights is not None:
            # Ensure ce_loss_weights is on device
            if isinstance(ce_loss_weights, torch.Tensor) and ce_loss_weights.device != device:
                ce_loss_weights = ce_loss_weights.to(device)
            data['ce_loss_weights'] = ce_loss_weights
        
        # Track tokens (with error handling to avoid blocking training)
        try:
            tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            token_window += tokens_tensor.item()
            if data['sample_lens']:
                sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
                sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
                dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
                seqlen_square_window += sample_square.item()
        except RuntimeError as e:
            if "cuda error" in str(e).lower() or "illegal memory" in str(e).lower():
                logger.warning(f"Error in token tracking (non-critical): {e}")
            else:
                raise
        
        # Critical: Synchronize all processes before forward pass
        # This ensures all ranks have data ready and CUDA operations are complete
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        
        # Forward pass
        loss_dict = train_one_step(
            fsdp_model, vae_model, tokenizer, new_token_ids, data,
            training_args, device, logger
        )
        
        # Compute loss
        loss, loss_dict = compute_loss(loss_dict, data, training_args, device)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
        
        # Logging
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            
            total_ce_tokens = torch.tensor(
                len(data.get('ce_loss_indexes', [])), device=device
            )
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            
            total_mse_tokens = torch.tensor(
                len(data.get('mse_loss_indexes', [])), device=device
            )
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            
            start_time, token_window, seqlen_square_window = log_metrics(
                curr_step, loss_dict, token_window, seqlen_square_window,
                start_time, training_args, device, logger,
                total_norm, total_samples, total_ce_tokens, total_mse_tokens,
                dense_token_factor, attn_factor
            )
            
            # Update wandb log with lr
            if dist.get_rank() == 0:
                optimizer_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.0
                wandb.log({'lr': optimizer_lr}, step=curr_step)
        
        # Update data status
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']
        
        # Save checkpoint
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            save_checkpoint(
                curr_step, fsdp_model, ema_model, optimizer, scheduler,
                data_status, training_args, fsdp_config, logger
            )
    
    # Save final checkpoint
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        save_checkpoint(
            curr_step, fsdp_model, ema_model, optimizer, scheduler,
            data_status, training_args, fsdp_config, logger
        )
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    return curr_step

