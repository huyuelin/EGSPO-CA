#!/usr/bin/env python3
"""
Training experiment for EGSPO-CA v2 reproduction.

Supports:
  - Single dataset training (--data <path> --steps N)
  - 3-phase curriculum training (--curriculum):
    Phase 1: GSM8K warmup (200 steps)
    Phase 2: GSM8K + NuminaMath mixed (300 steps)
    Phase 3: NuminaMath-CoT main (2500 steps)
  - torch.compile for speed optimization (--compile)
  - gradient checkpointing for memory (--gradient_checkpointing)

Usage:
  # Full curriculum with K=8 (paper default)
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/run_minimal_train.py --method grpo --K 8 --max_new_tokens 512 --curriculum

  # Single dataset
  CUDA_VISIBLE_DEVICES=0 python scripts/run_minimal_train.py --method grpo --data data/numina_cot_2500.jsonl --steps 2500

  # With torch.compile
  CUDA_VISIBLE_DEVICES=0 python scripts/run_minimal_train.py --method egspo_ca --curriculum --compile
"""

import argparse
import logging
import os
import sys

# Ensure code directory is in sys.path for package imports
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import torch
from transformers import AutoTokenizer

from data.numina_loader import MathProblemDataset
from training.grpo_trainer import GRPOTrainingConfig, GRPOTrainer
from training.egspo_ca_trainer import EGSPOCATrainingConfig, EGSPOCATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOGGER = logging.getLogger(__name__)


def build_trainer(method, model_name, output_dir, dataset, compile_model,
                  K, max_new_tokens, gradient_checkpointing=False,
                  pretrained_model=None):
    """Build trainer, optionally loading a pre-trained model from warmup."""
    if method == "grpo":
        config = GRPOTrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            K=K,
            max_new_tokens=max_new_tokens,
            learning_rate=5.0e-6,
            num_train_epochs=3,
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            logging_steps=5,
            save_steps=500,
        )
        trainer = GRPOTrainer(config, dataset)
    else:
        config = EGSPOCATrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            K=K,
            max_new_tokens=max_new_tokens,
            learning_rate=5.0e-6,
            num_train_epochs=3,
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            logging_steps=5,
            save_steps=500,
        )
        trainer = EGSPOCATrainer(config, dataset)

    # Load pretrained weights if continuing from a prior phase
    if pretrained_model is not None:
        LOGGER.info("Loading pretrained model weights from prior curriculum phase...")
        trainer.model.load_state_dict(pretrained_model.state_dict(), strict=False)
        # Also load optimizer state if available
        if hasattr(pretrained_model, 'optimizer_state') and pretrained_model.optimizer_state:
            try:
                trainer.optimizer.load_state_dict(pretrained_model.optimizer_state)
                LOGGER.info("Optimizer state loaded from prior phase.")
            except Exception:
                LOGGER.warning("Could not load optimizer state, using fresh optimizer.")
        LOGGER.info("Model weights transferred successfully.")
        del pretrained_model
        torch.cuda.empty_cache()

    if gradient_checkpointing:
        LOGGER.info("Enabling gradient checkpointing for memory efficiency...")
        trainer.model.gradient_checkpointing_enable()
        LOGGER.info("Gradient checkpointing enabled.")

    if compile_model:
        LOGGER.info("Applying torch.compile to model...")
        trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
        LOGGER.info("torch.compile applied.")

    return trainer


def compute_steps_for_phase(dataset_size, target_steps):
    """Convert target steps to num_train_epochs."""
    return max(1, target_steps // max(dataset_size, 1))


def run_curriculum(method, model_name, output_dir, compile_model, K, max_new_tokens,
                   warmup_steps, mixed_steps, main_steps, gradient_checkpointing):
    """Run 3-phase curriculum training with model transfer between phases."""
    data_base = os.path.join(CODE_DIR, "data")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pretrained_model = None  # Will hold the model from previous phase

    # ======================================================================
    # Phase 1: GSM8K warmup (teaching basic structure)
    # ======================================================================
    gsm8k_path = os.path.join(data_base, "gsm8k.jsonl")
    if not os.path.exists(gsm8k_path):
        LOGGER.error("GSM8K data not found at %s. Cannot run curriculum.", gsm8k_path)
        return

    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 1: GSM8K warmup (%d steps, K=%d)", warmup_steps, K)
    LOGGER.info("=" * 60)
    gsm8k_dataset = MathProblemDataset(
        data_path=gsm8k_path,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_response_length=max_new_tokens,
    )
    LOGGER.info("GSM8K loaded: %d problems", len(gsm8k_dataset))

    warmup_trainer = build_trainer(
        method, model_name, os.path.join(output_dir, "warmup"),
        gsm8k_dataset, compile_model, K, max_new_tokens,
        gradient_checkpointing=gradient_checkpointing,
    )
    warmup_trainer.config.num_train_epochs = compute_steps_for_phase(
        len(gsm8k_dataset), warmup_steps)
    warmup_trainer.config.save_steps = warmup_steps
    warmup_trainer.train()

    # Save model state for next phase
    pretrained_model = warmup_trainer.model
    # Save optimizer state
    pretrained_model.optimizer_state = warmup_trainer.optimizer.state_dict()
    LOGGER.info("Phase 1 complete. Model transferred to Phase 2.")

    # ======================================================================
    # Phase 2: Mixed GSM8K + NuminaMath (gradual difficulty increase)
    # ======================================================================
    numina_path = os.path.join(data_base, "numina_cot_2500.jsonl")
    if not os.path.exists(numina_path):
        numina_path = os.path.join(data_base, "numina_cot.jsonl")
    if not os.path.exists(numina_path):
        LOGGER.error("NuminaMath data not found. Cannot continue curriculum.")
        return

    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 2: Mixed GSM8K+NuminaMath (%d steps, K=%d)", mixed_steps, K)
    LOGGER.info("=" * 60)

    # Create mixed dataset (50/50 GSM8K + NuminaMath)
    mixed_dataset_path = os.path.join(output_dir, "mixed_dataset.jsonl")
    import json
    with open(gsm8k_path) as f:
        gsm8k_data = [json.loads(line) for line in f if line.strip()]
    with open(numina_path) as f:
        numina_data = [json.loads(line) for line in f if line.strip()]

    # Interleave to ensure balance
    mixed_data = []
    max_len = max(len(gsm8k_data), len(numina_data))
    for i in range(max_len):
        if i < len(gsm8k_data):
            mixed_data.append(gsm8k_data[i])
        if i < len(numina_data):
            mixed_data.append(numina_data[i])

    os.makedirs(os.path.dirname(mixed_dataset_path), exist_ok=True)
    with open(mixed_dataset_path, 'w') as f:
        for item in mixed_data:
            f.write(json.dumps(item) + '\n')

    LOGGER.info("Mixed dataset: %d problems (GSM8K + NuminaMath interleaved)", len(mixed_data))
    mixed_dataset = MathProblemDataset(
        data_path=mixed_dataset_path,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_response_length=max_new_tokens,
    )

    mixed_trainer = build_trainer(
        method, model_name, os.path.join(output_dir, "mixed"),
        mixed_dataset, compile_model, K, max_new_tokens,
        gradient_checkpointing=gradient_checkpointing,
        pretrained_model=pretrained_model,
    )
    mixed_trainer.config.num_train_epochs = compute_steps_for_phase(
        len(mixed_dataset), mixed_steps)
    mixed_trainer.config.save_steps = mixed_steps
    mixed_trainer.train()

    pretrained_model = mixed_trainer.model
    pretrained_model.optimizer_state = mixed_trainer.optimizer.state_dict()
    LOGGER.info("Phase 2 complete. Model transferred to Phase 3.")

    # ======================================================================
    # Phase 3: NuminaMath-CoT main training
    # ======================================================================
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 3: NuminaMath-CoT main (%d steps, K=%d)", main_steps, K)
    LOGGER.info("=" * 60)

    main_dataset = MathProblemDataset(
        data_path=numina_path,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_response_length=max_new_tokens,
    )
    LOGGER.info("Main dataset loaded: %d problems", len(main_dataset))

    main_trainer = build_trainer(
        method, model_name, os.path.join(output_dir, "main"),
        main_dataset, compile_model, K, max_new_tokens,
        gradient_checkpointing=gradient_checkpointing,
        pretrained_model=pretrained_model,
    )
    main_trainer.config.num_train_epochs = compute_steps_for_phase(
        len(main_dataset), main_steps)
    main_trainer.config.save_steps = 500
    main_trainer.train()
    LOGGER.info("Phase 3 complete. Full curriculum training done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["grpo", "egspo_ca"], default="grpo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/numina_cot_2500.jsonl")
    parser.add_argument("--output", default="results/training")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--K", type=int, default=8,
                        help="Number of rollouts per prompt (paper: K=8)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens per rollout (paper: 512)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for speed (~30% faster)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run 3-phase curriculum: GSM8K → mixed → NuminaMath")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="GSM8K warmup steps (Phase 1)")
    parser.add_argument("--mixed_steps", type=int, default=300,
                        help="Mixed GSM8K+NuminaMath steps (Phase 2)")
    parser.add_argument("--main_steps", type=int, default=2500,
                        help="NuminaMath-CoT main steps (Phase 3)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for K=8 memory")
    args = parser.parse_args()

    # Use /dev/shm for temp files to avoid root partition full errors
    os.environ.setdefault("TMPDIR", "/dev/shm")
    os.environ.setdefault("TMP", "/dev/shm")
    os.environ.setdefault("TEMP", "/dev/shm")

    assert torch.cuda.is_available(), "CUDA not available!"
    LOGGER.info("Device: %s, GPU: %s, Free mem: %.1f GB",
                args.device, torch.cuda.get_device_name(0),
                torch.cuda.mem_get_info()[0] / 1024**3)

    # Curriculum mode
    if args.curriculum:
        run_curriculum(
            method=args.method,
            model_name=args.model,
            output_dir=args.output,
            compile_model=args.compile,
            K=args.K,
            max_new_tokens=args.max_new_tokens,
            warmup_steps=args.warmup_steps,
            mixed_steps=args.mixed_steps,
            main_steps=args.steps if args.steps != 2500 else args.main_steps,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        LOGGER.info("Curriculum training complete.")
        return

    # Single dataset mode
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = os.path.join(CODE_DIR, args.data)
    dataset = MathProblemDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_response_length=args.max_new_tokens,
    )
    LOGGER.info("Loaded %d training problems", len(dataset))

    trainer = build_trainer(
        args.method, args.model, args.output, dataset,
        args.compile, args.K, args.max_new_tokens,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer.config.num_train_epochs = compute_steps_for_phase(len(dataset), args.steps)
    trainer.train()
    LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()
