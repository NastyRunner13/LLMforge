"""Training engine -- HuggingFace TRL + PEFT wrapper.

Provides a high-level TrainingEngine class that loads a base model,
configures LoRA/QLoRA adapters via PEFT, runs SFT training via TRL,
and publishes metrics to Redis for real-time streaming.
"""

from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Chat template tags
_SYS = "<|system|>"
_USR = "<|user|>"
_AST = "<|assistant|>"
_END = "<|end|>"


class TrainingEngine:
    """High-level wrapper around HuggingFace TRL SFTTrainer."""

    def __init__(self, run_id: str, config: dict, dataset_path: str):
        self.run_id = run_id
        self.config = config
        self.dataset_path = dataset_path
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self._output_dir = ""

    def setup(self):
        """Load model, tokenizer, dataset, and configure trainer."""
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        base_model = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        method = self.config.get("method", "lora")
        hf_token = os.environ.get("HF_TOKEN", None)

        logger.info("[%s] Loading tokenizer: %s", self.run_id, base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("[%s] Loading model: %s (method=%s)", self.run_id, base_model, method)
        model_kwargs: dict = {"token": hf_token, "trust_remote_code": True}

        if method == "qlora":
            try:
                import torch
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to fp32")

        self.model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        if method in ("lora", "qlora"):
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            if method == "qlora":
                self.model = prepare_model_for_kbit_training(self.model)

            lora_cfg = self.config.get("lora_config") or {}
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        logger.info("[%s] Loading dataset: %s", self.run_id, self.dataset_path)
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")

        self._output_dir = tempfile.mkdtemp(prefix=f"llmforge_{self.run_id}_")
        ctx_len = self.config.get("context_length", 2048)

        training_args_dict = dict(
            output_dir=self._output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            max_steps=self.config.get("max_steps", -1),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            lr_scheduler_type=self.config.get("scheduler", "cosine"),
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            bf16=self.config.get("mixed_precision", "bf16") == "bf16",
            fp16=self.config.get("mixed_precision") == "fp16",
            logging_steps=10,
            save_steps=self.config.get("checkpoint_steps", 500),
            save_total_limit=self.config.get("max_checkpoints", 3),
            report_to="none",
            max_seq_length=ctx_len,
            packing=True,
        )

        # Apply distributed training config (DeepSpeed / FSDP)
        dist_cfg = self.config.get("distributed")
        if dist_cfg and dist_cfg.get("strategy", "none") != "none":
            from app.training.distributed import DistributedConfig, apply_distributed_config

            dc = DistributedConfig(
                strategy=dist_cfg.get("strategy", "none"),
                num_gpus=dist_cfg.get("num_gpus", 1),
                per_device_batch_size=self.config.get("batch_size", 4),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
                mixed_precision=self.config.get("mixed_precision", "bf16"),
                zero_stage=dist_cfg.get("zero_stage", 2),
                offload_optimizer=dist_cfg.get("offload_optimizer", False),
                offload_param=dist_cfg.get("offload_param", False),
                fsdp_sharding_strategy=dist_cfg.get("fsdp_sharding_strategy", "FULL_SHARD"),
            )
            training_args_dict = apply_distributed_config(training_args_dict, dc)
            logger.info(
                "[%s] Distributed training: %s (%d GPUs)", self.run_id, dc.strategy, dc.num_gpus
            )

        training_args = SFTConfig(**training_args_dict)

        def formatting_func(examples):
            texts = []
            keys = list(examples.keys())
            n = len(examples[keys[0]]) if keys else 0
            for i in range(n):
                parts = []
                if "system" in examples:
                    parts.append(f"{_SYS}\n{examples['system'][i]}")
                if "user" in examples:
                    parts.append(f"{_USR}\n{examples['user'][i]}")
                if "assistant" in examples:
                    parts.append(f"{_AST}\n{examples['assistant'][i]}{_END}")
                if "text" in examples and not parts:
                    parts.append(examples["text"][i])
                texts.append("\n".join(parts))
            return texts

        from app.training.callbacks import CheckpointUploadCallback, MetricsCallback

        callbacks = [
            MetricsCallback(run_id=self.run_id),
            CheckpointUploadCallback(run_id=self.run_id),
        ]

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            formatting_func=formatting_func,
            callbacks=callbacks,
        )

    def train(self):
        """Run the training loop and return output directory."""
        if self.trainer is None:
            raise RuntimeError("Call setup() before train()")

        resume_path = self.config.get("resume_from_checkpoint")
        logger.info("[%s] Starting training...", self.run_id)
        self.trainer.train(resume_from_checkpoint=resume_path)
        logger.info("[%s] Training complete.", self.run_id)
        return self._output_dir

    def save_model(self, output_path: str | None = None):
        """Save the trained model/adapter weights."""
        if self.trainer is None:
            raise RuntimeError("No trainer to save")
        save_path = output_path or os.path.join(self._output_dir, "final_model")
        self.trainer.save_model(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        logger.info("[%s] Model saved to %s", self.run_id, save_path)
        return save_path

    @property
    def output_dir(self) -> str:
        return self._output_dir
