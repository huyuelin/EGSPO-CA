#!/usr/bin/env python3
"""
Run baseline training experiments for Phase 3D.
Supports all 10 baseline methods.

Usage:
  # EGSPO (entropy-only gating)
  CUDA_VISIBLE_DEVICES=3 python baselines/run_baseline.py --method egspo

  # Dr.GRPO (length-corrected)
  CUDA_VISIBLE_DEVICES=4 python baselines/run_baseline.py --method drgrpo

  # All simple baselines in sequence
  python baselines/run_baseline.py --all-simple
"""

import argparse
import logging
import os
import sys

# Ensure code directory is in sys.path
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import torch
from transformers import AutoTokenizer

from data.numina_loader import MathProblemDataset
from training.egspo_ca_trainer import EGSPOCATrainingConfig
from training.grpo_trainer import GRPOTrainingConfig
from baselines.baseline_trainers import (
    BASELINE_NAMES,
    BASELINE_REGISTRY,
    DrGRPOTrainingConfig,
    DAPOTrainingConfig,
    create_baseline_trainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Methods that use EGSPO-CA config (need value head)
EGSPO_CA_METHODS = {"egspo", "gtpo", "tempo", "delta", "spo", "hapo", "capo", "cfcredit"}
# Methods that use GRPO config (no value head)
GRPO_METHODS = {"drgrpo", "dapo"}


def get_config(method: str, args):
    """Create method-specific training config."""
    if method in EGSPO_CA_METHODS:
        return EGSPOCATrainingConfig(
            model_name=args.model,
            output_dir=os.path.join(args.output, method),
            K=args.K,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.lr,
            num_train_epochs=max(1, args.steps // args.max_problems) if args.max_problems else args.num_epochs,
            per_device_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            logging_steps=args.log_steps,
            save_steps=args.steps,
        )
    else:
        base = dict(
            model_name=args.model,
            output_dir=os.path.join(args.output, method),
            K=args.K,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.lr,
            num_train_epochs=max(1, args.steps // args.max_problems) if args.max_problems else args.num_epochs,
            per_device_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            logging_steps=args.log_steps,
            save_steps=args.steps,
        )
        if method == "drgrpo":
            return DrGRPOTrainingConfig(length_penalty=args.length_penalty, **base)
        elif method == "dapo":
            return DAPOTrainingConfig(
                clip_epsilon_high=args.clip_high,
                clip_epsilon_low=args.clip_low,
                **base,
            )
        return GRPOTrainingConfig(**base)


def run_baseline(method: str, args):
    """Run a single baseline experiment."""
    LOGGER.info("Starting baseline: %s", method)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(CODE_DIR, data_path)

    dataset = MathProblemDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_response_length=args.max_new_tokens,
    )

    if args.max_problems:
        import random
        random.shuffle(dataset.data)
        dataset.data = dataset.data[:args.max_problems]

    LOGGER.info("Loaded %d training problems", len(dataset))

    # Config and trainer
    config = get_config(method, args)
    trainer = create_baseline_trainer(method, config, dataset)

    # Train
    trainer.train()
    LOGGER.info("Baseline %s complete.", method)


def main():
    parser = argparse.ArgumentParser(description="Run baseline training experiments")
    parser.add_argument("--method", choices=BASELINE_NAMES, help="Single baseline method")
    parser.add_argument("--all-simple", action="store_true", help="Run EGSPO, DrGRPO, GTPO, DAPO")
    parser.add_argument("--all-medium", action="store_true", help="Run TEMPO, DelTA")
    parser.add_argument("--all-complex", action="store_true", help="Run SPO, HAPO, CAPO, CF Credit")
    parser.add_argument("--all", action="store_true", help="Run all 10 baselines")

    # Training params
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/numina_cot_2500.jsonl")
    parser.add_argument("--output", default="results/phase3d")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=2500)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")

    # Method-specific params
    parser.add_argument("--length_penalty", type=float, default=0.1)
    parser.add_argument("--clip_high", type=float, default=0.28)
    parser.add_argument("--clip_low", type=float, default=0.2)

    args = parser.parse_args()

    # Determine which methods to run
    methods = []
    if args.all:
        methods = list(BASELINE_NAMES)
    elif args.all_simple:
        methods = ["egspo", "drgrpo", "gtpo", "dapo"]
    elif args.all_medium:
        methods = ["tempo", "delta"]
    elif args.all_complex:
        methods = ["spo", "hapo", "capo", "cfcredit"]
    elif args.method:
        methods = [args.method]
    else:
        parser.error("Must specify --method, --all-simple, --all-medium, --all-complex, or --all")

    for method in methods:
        run_baseline(method, args)


if __name__ == "__main__":
    main()
