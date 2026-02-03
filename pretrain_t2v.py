# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Megatron-FSDP Pretrain Script for Wan2.2-TI2V-5B (Mock Data Only)
# Simplified: No epochs, just iterations.

import argparse
import logging
import math
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Megatron-FSDP imports
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard

# Wan modules
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel, WanAttentionBlock
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE

logger = logging.getLogger(__name__)

# Device mesh dimension names
DP_SHARD = "dp_shard"
DP_OUTER = "dp_outer"
CP = "cp"
DP_SHARD_CP = "dp_shard_cp"
TP = "tp"
HSDP = "hsdp"


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Wan2.2-TI2V-5B with Megatron-FSDP")
    
    # Model
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/wan22_pretrain")
    
    # Training (iteration-based, no epochs)
    parser.add_argument("--max_steps", type=int, required=True, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # Mock data
    parser.add_argument("--num_samples", type=int, default=10000, help="Mock dataset size")
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 832])
    
    # Megatron-FSDP
    parser.add_argument("--dp_shard_size", type=int, default=None)
    parser.add_argument("--dp_outer_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--zero_dp_strategy", type=str, default="optim",
                        choices=["no_shard", "optim", "optim_grads", "optim_grads_params"])
    parser.add_argument("--outer_dp_strategy", type=str, default="no_shard")
    parser.add_argument("--preserve_fp32_weights", action="store_true", default=True)
    parser.add_argument("--grad_reduce_in_fp32", action="store_true", default=False)
    parser.add_argument("--init_on_meta_device", action="store_true", default=False)
    parser.add_argument("--overlap_grad_reduce", action="store_true", default=False)
    parser.add_argument("--overlap_param_gather", action="store_true", default=False)
    # TransformerEngine gradient accumulation fusion
    parser.add_argument("--use_te_linear", action="store_true", default=False,
                        help="Replace nn.Linear with TE Linear for gradient_accumulation_fusion")
    parser.add_argument("--te_fuse_wgrad", action="store_true", default=False,
                        help="Enable fuse_wgrad_accumulation in TE Linear (requires aligned dimensions)")
    
    # Mixed precision
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    
    # Checkpointing & logging
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    
    return rank, world_size, local_rank


def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def set_seed(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def build_device_mesh(args, world_size):
    if args.dp_shard_size is None:
        args.dp_shard_size = world_size // (args.dp_outer_size * args.cp_size * args.tp_size)
    
    mesh_shape = (args.dp_outer_size, args.dp_shard_size, args.cp_size, args.tp_size)
    mesh_dim_names = (DP_OUTER, DP_SHARD, CP, TP)
    
    logger.info(f"Device mesh: {mesh_shape}")
    
    device_mesh = init_device_mesh("cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)
    device_mesh[(DP_SHARD, CP)]._flatten(DP_SHARD_CP)
    device_mesh[(DP_OUTER, DP_SHARD, CP)]._flatten(HSDP)
    
    return device_mesh


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset with random video tensors."""
    
    def __init__(self, num_samples, frame_num, resolution):
        self.num_samples = num_samples
        self.frame_num = frame_num
        self.resolution = resolution
        self.captions = [
            "A beautiful sunset over the ocean.",
            "A cat playing with yarn.",
            "Time-lapse of clouds over mountains.",
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "video": torch.randn(3, self.frame_num, *self.resolution),
            "text": self.captions[idx % len(self.captions)]
        }


def collate_fn(batch):
    return {
        "video": torch.stack([x["video"] for x in batch]),
        "text": [x["text"] for x in batch]
    }


class FlowMatchingLoss:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
    
    def __call__(self, model, x_0, context, seq_len):
        batch_size = len(x_0)
        device = x_0[0].device
        
        t = torch.rand(batch_size, device=device) * self.num_timesteps
        noise = [torch.randn_like(x) for x in x_0]
        
        x_t = []
        for i, (x, n) in enumerate(zip(x_0, noise)):
            t_i = (t[i] / self.num_timesteps).view(-1, 1, 1, 1)
            x_t.append((1 - t_i) * x + t_i * n)
        
        target = [n - x for n, x in zip(noise, x_0)]
        pred = model(x_t, t=t, context=context, seq_len=seq_len)
        
        return sum(F.mse_loss(p, tgt) for p, tgt in zip(pred, target)) / len(pred)


def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, step, args, rank):
    """Save checkpoint using distributed checkpoint for FSDP compatibility."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter
    
    checkpoint_dir = Path(args.output_dir) / f"step-{step}"
    
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint: {checkpoint_dir}")
    
    # Barrier to ensure directory is created before all ranks try to save
    dist.barrier()
    
    # Use distributed checkpoint - all ranks participate
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    
    dcp.save(state_dict, storage_writer=FileSystemWriter(str(checkpoint_dir)))
    
    if rank == 0:
        logger.info(f"Checkpoint saved: {checkpoint_dir}")


def train(model, text_encoder, vae, data_iter, optimizer, scheduler, criterion,
          args, rank, device, param_dtype, config):
    """Main training loop - iteration based, no epochs."""
    
    model.train()
    patch_size = config.patch_size
    grad_accum = args.gradient_accumulation_steps
    
    progress = tqdm(range(1, args.max_steps + 1), desc="Training", disable=rank != 0)
    accumulated_loss = 0.0
    
    for step in progress:
        # Accumulate gradients over micro-batches
        for micro_idx in range(grad_accum):
            batch = next(data_iter)
            videos = batch["video"].to(device)
            texts = batch["text"]
            
            is_last = (micro_idx == grad_accum - 1)
            model.set_model_auto_sync(is_last)
            
            # Encode (frozen)
            with torch.no_grad():
                context = text_encoder(texts, device)
                latents = vae.encode([v for v in videos])
            
            # Compute seq_len
            shape = latents[0].shape
            seq_len = math.ceil((shape[2] * shape[3]) / (patch_size[1] * patch_size[2]) * shape[1])
            
            # Forward + backward
            with torch.amp.autocast("cuda", dtype=param_dtype):
                loss = criterion(model, latents, context, seq_len) / grad_accum
            
            loss.backward()
            accumulated_loss += loss.item()
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        if step % args.log_steps == 0 and rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Step {step}/{args.max_steps} | Loss: {accumulated_loss:.4f} | LR: {lr:.2e}")
            progress.set_postfix(loss=f"{accumulated_loss:.4f}")
        
        accumulated_loss = 0.0
        
        # Save checkpoint
        if step % args.save_steps == 0:
            if dist.is_initialized():
                dist.barrier()
            save_checkpoint(model, optimizer, step, args, rank)
    
    # Final checkpoint
    if dist.is_initialized():
        dist.barrier()
    save_checkpoint(model, optimizer, args.max_steps, args, rank)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    setup_logging(rank)
    set_seed(args.seed, rank)
    
    logger.info(f"World size: {world_size}, Rank: {rank}")
    logger.info(f"Training for {args.max_steps} iterations")
    
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    param_dtype = dtype_map[args.param_dtype]
    config = WAN_CONFIGS["ti2v-5B"]
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    device_mesh = build_device_mesh(args, world_size)
    use_hsdp = args.dp_outer_size > 1
    
    # Load frozen models
    logger.info("Loading T5...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, config.t5_tokenizer),
    )
    text_encoder.model.eval()
    for p in text_encoder.model.parameters():
        p.requires_grad = False
    
    logger.info("Loading VAE...")
    vae = Wan2_2_VAE(vae_pth=os.path.join(args.checkpoint_dir, config.vae_checkpoint), device=device)
    vae.model.eval()
    for p in vae.model.parameters():
        p.requires_grad = False
    
    # Load trainable model
    logger.info("Loading WanModel...")
    init_device = torch.device("meta") if args.init_on_meta_device else device
    with torch.device(init_device):
        model = WanModel.from_pretrained(args.checkpoint_dir)
    
    if not args.init_on_meta_device:
        model = model.to(device=device, dtype=param_dtype)
    
    # Replace nn.Linear with TE Linear for gradient_accumulation_fusion
    if args.use_te_linear:
        from te_utils import replace_linear_with_te, is_te_available
        if is_te_available():
            # fuse_wgrad_accumulation requires specific GEMM dimension alignment
            logger.info(f"Replacing nn.Linear with TE Linear (fuse_wgrad_accumulation={args.te_fuse_wgrad})...")
            model = replace_linear_with_te(model, fuse_wgrad_accumulation=args.te_fuse_wgrad, alignment=8)
        else:
            logger.warning("TransformerEngine not available, skipping TE Linear replacement")
    
    model.train()
    model.requires_grad_(True)
    
    # Optimizer + FSDP
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    logger.info("Wrapping with Megatron-FSDP...")
    model, optimizer = fully_shard(
        module=model, optimizer=optimizer, device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD_CP,
        dp_outer_dim=DP_OUTER if use_hsdp else None,
        tp_dim=TP,
        hybrid_fsdp_group=device_mesh[HSDP].get_group() if use_hsdp else None,
        fsdp_unit_modules=[WanAttentionBlock],
        zero_dp_strategy=args.zero_dp_strategy,
        outer_dp_sharding_strategy=args.outer_dp_strategy if use_hsdp else "no_shard",
        preserve_fp32_weights=args.preserve_fp32_weights,
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        init_model_with_meta_device=args.init_on_meta_device,
        sync_model_each_microbatch=False,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=args.overlap_param_gather,
    )
    logger.info("Megatron-FSDP ready")
    
    # Set up TE Linear gradient accumulation (FSDP-compatible)
    te_grad_accumulator = None
    if args.use_te_linear and args.te_fuse_wgrad:
        from te_utils import setup_main_grad_for_te_linear
        setup_main_grad_for_te_linear(model)
    
    # Dataset + infinite iterator
    dataset = MockDataset(args.num_samples, args.frame_num, tuple(args.resolution))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    
    # Infinite iterator - cycles through data
    data_iter = cycle(iter(dataloader))
    
    scheduler = get_cosine_scheduler(optimizer, args.warmup_steps, args.max_steps)
    criterion = FlowMatchingLoss(num_timesteps=config.num_train_timesteps)
    
    # Train
    logger.info("Starting training...")
    train(model, text_encoder, vae, data_iter, optimizer, scheduler, criterion,
          args, rank, device, param_dtype, config)
    
    logger.info("Training completed!")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
