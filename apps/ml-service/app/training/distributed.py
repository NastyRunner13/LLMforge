"""Distributed training configuration — DeepSpeed and FSDP.

Provides factory functions to build DeepSpeed ZeRO configs and
PyTorch FSDP configs for multi-GPU training. These are used
by the TrainingEngine when num_gpus > 1.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Unified distributed training configuration."""

    strategy: str = "none"  # "none", "deepspeed_zero2", "deepspeed_zero3", "fsdp"
    num_gpus: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"  # "bf16", "fp16", "fp32"

    # DeepSpeed-specific
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False

    # FSDP-specific
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_auto_wrap_policy: str = "transformer_based"
    fsdp_min_num_params: int = 1_000_000


def build_deepspeed_config(config: DistributedConfig) -> dict:
    """Build a DeepSpeed JSON configuration."""
    ds_config = {
        "bf16": {"enabled": config.mixed_precision == "bf16"},
        "fp16": {"enabled": config.mixed_precision == "fp16"},
        "zero_optimization": {
            "stage": config.zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "train_batch_size": (
            config.per_device_batch_size * config.gradient_accumulation_steps * config.num_gpus
        ),
        "train_micro_batch_size_per_gpu": config.per_device_batch_size,
        "wall_clock_breakdown": False,
    }

    # ZeRO-3: partition parameters across GPUs
    if config.zero_stage == 3:
        ds_config["zero_optimization"].update(
            {
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )

    # CPU offloading (for memory-constrained setups)
    if config.offload_optimizer:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if config.offload_param:
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    logger.info(
        "DeepSpeed config: ZeRO-%d, %d GPUs, batch=%d, grad_accum=%d",
        config.zero_stage,
        config.num_gpus,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
    )

    return ds_config


def build_fsdp_config(config: DistributedConfig) -> dict:
    """Build FSDP configuration for HuggingFace Trainer."""
    fsdp_config = {
        "fsdp": config.fsdp_sharding_strategy,
        "fsdp_config": {
            "fsdp_auto_wrap_policy": config.fsdp_auto_wrap_policy,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": config.offload_param,
            "fsdp_sharding_strategy": config.fsdp_sharding_strategy,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
        },
    }

    if config.fsdp_auto_wrap_policy == "size_based":
        fsdp_config["fsdp_config"]["fsdp_min_num_params"] = config.fsdp_min_num_params

    logger.info(
        "FSDP config: %s, %d GPUs, batch=%d",
        config.fsdp_sharding_strategy,
        config.num_gpus,
        config.per_device_batch_size,
    )

    return fsdp_config


def apply_distributed_config(training_args_dict: dict, config: DistributedConfig) -> dict:
    """
    Apply distributed training config to a TrainingArguments dict.

    Returns the modified dict with DeepSpeed or FSDP settings applied.
    Also writes DeepSpeed config to a temp file if needed.
    """
    if config.strategy == "none" or config.num_gpus <= 1:
        return training_args_dict

    if config.strategy.startswith("deepspeed"):
        # Determine ZeRO stage from strategy name
        if "zero3" in config.strategy:
            config.zero_stage = 3
        else:
            config.zero_stage = 2

        ds_config = build_deepspeed_config(config)

        # Write to temp file (DeepSpeed needs a file path)
        import tempfile

        ds_config_path = os.path.join(tempfile.gettempdir(), f"ds_config_{os.getpid()}.json")
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)

        training_args_dict["deepspeed"] = ds_config_path
        logger.info("DeepSpeed config written to %s", ds_config_path)

    elif config.strategy == "fsdp":
        fsdp_config = build_fsdp_config(config)
        training_args_dict.update(fsdp_config)

    return training_args_dict


def get_launch_command(config: DistributedConfig, script_path: str) -> list[str]:
    """
    Build the shell command for distributed training launch.

    Returns a list of command args suitable for subprocess.run().
    """
    if config.num_gpus <= 1:
        return ["python", script_path]

    if config.strategy.startswith("deepspeed"):
        return [
            "deepspeed",
            f"--num_gpus={config.num_gpus}",
            script_path,
        ]

    # FSDP / general multi-GPU via torchrun
    return [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        "--master_port=29500",
        script_path,
    ]
