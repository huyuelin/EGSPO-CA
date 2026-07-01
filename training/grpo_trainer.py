"""
GRPO baseline trainer for EGSPO-CA v2 reproduction.

Implements standard GRPO (uniform token-level credit).
Reference: shao2024deepseekmath, Section 5.1 (Baselines).

This is a minimal single-GPU implementation for proof-of-concept.
For multi-GPU training, use deepspeed/FSDP (or verl as in the paper).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from credit.tbicc import compute_entropy_signal
from data.numina_loader import MathProblemDataset, compute_outcome_returns
from models.value_head import ValueHead
from training.loss import GRPOLoss
from training.reward_utils import batch_exact_match_reward

LOGGER = logging.getLogger(__name__)


@dataclass
class GRPOTrainingConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "../results/phase1/grpo"
    K: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 5.0e-6
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"


class GRPOTrainer:
    """
    Minimal GRPO trainer.

    Algorithm:
      1. For each prompt, generate K rollouts
      2. Compute outcome reward (exact match or reward model)
      3. Compute GRPO advantages (group-normalized)
      4. Compute clipped surrogate loss (uniform credit)
      5. Backprop and update
    """

    def __init__(
        self,
        config: GRPOTrainingConfig,
        train_dataset: MathProblemDataset,
        eval_datasets: Optional[Dict[str, MathProblemDataset]] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets

        # Load model and tokenizer (single model, ref logits via eval+no_grad pass)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Loss function
        self.loss_fn = GRPOLoss(K=config.K, clip_epsilon=config.clip_epsilon)

        # Optimizer (8-bit Adam for memory efficiency, saves ~42GB)
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=config.learning_rate,
            )
            LOGGER.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
            )
            LOGGER.info("Using standard AdamW optimizer.")

        LOGGER.info("GRPOTrainer initialized. Model: %s", config.model_name)

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        dtype = getattr(torch, self.config.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        return model

    def generate_rollouts(
        self,
        prompts: List[str],
        K: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate K rollouts per prompt.

        Returns:
            rollouts: dict with keys:
              - token_ids: (N*K, seq_len)
              - logits: (N*K, seq_len, vocab_size)
              - attention_mask: (N*K, seq_len)
              - rewards: (N*K,) outcome rewards
        """
        N = len(prompts)
        all_token_ids = []

        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Generate K rollouts (temporarily restore use_cache for generation)
            _saved_cache = self.model.config.use_cache
            self.model.config.use_cache = True
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_return_sequences=K,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.model.config.use_cache = _saved_cache

            rollout_ids = outputs.sequences  # (K, seq_len)
            all_token_ids.append(rollout_ids)

        # Concatenate
        token_ids = torch.cat(all_token_ids, dim=0)  # (N*K, seq_len)
        attention_mask = (token_ids != self.tokenizer.pad_token_id).long()

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
        }

    def compute_reward(
        self,
        generated_texts: List[str],
        answers: List[str],
    ) -> torch.Tensor:
        """
        Compute outcome reward via exact match of numerical answer.
        Returns 1.0 if generated answer matches ground truth, 0.0 otherwise.

        Returns:
            rewards: (batch,) 0 or 1
        """
        rewards_list = batch_exact_match_reward(generated_texts, answers)
        return torch.tensor(rewards_list, dtype=torch.float)

    def train(self):
        """Main training loop."""
        self.model.eval()

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

        global_step = 0
        for epoch in range(self.config.num_train_epochs):
            for batch in train_loader:
                N = len(batch["prompts"])

                # Generate rollouts
                rollouts = self.generate_rollouts(batch["prompts"], K=self.config.K)

                # Decode generated texts
                generated_texts = self.tokenizer.batch_decode(
                    rollouts["token_ids"],
                    skip_special_tokens=True,
                )
                # Repeat answers K times (one per rollout)
                repeated_answers = [
                    ans for ans in batch["answers"] for _ in range(self.config.K)
                ]

                # Compute rewards
                rewards = self.compute_reward(generated_texts, repeated_answers)

                # Debug: print first rollout text and extracted answer
                if global_step <= 5 or global_step % 50 == 0:
                    sample_text = generated_texts[0]
                    from training.reward_utils import extract_answer_from_text
                    extracted = extract_answer_from_text(sample_text)
                    LOGGER.info(
                        "Step %d DEBUG: rollout[0] last 200 chars: %s",
                        global_step, sample_text[-200:].replace('\n','↵')
                    )
                    LOGGER.info(
                        "Step %d DEBUG: extracted=%s, expected=%s, reward=%.1f",
                        global_step, extracted, repeated_answers[0], rewards[0].item()
                    )
                
                rewards = rewards.to(self.model.device)

                # Reward diagnostics (critical for debugging training signal)
                if global_step % self.config.logging_steps == 0:
                    correct = rewards.sum().item()
                    total = rewards.numel()
                    LOGGER.info(
                        "Step %d, Rewards: mean=%.4f, correct=%.1f/%d (%.1f%%), std=%.4f",
                        global_step, rewards.mean().item(), correct, total,
                        100.0 * correct / max(total, 1), rewards.std().item()
                    )
                    if correct == 0:
                        LOGGER.warning(
                            "ALL ROLLOUTS INCORRECT at step %d — "
                            "advantages collapse to zero, no learning signal! "
                            "Consider: easier problems (GSM8K warmup) or larger K.",
                            global_step,
                        )

                # Reference forward (eval mode, no grad) - same model, frozen state
                self.model.eval()
                with torch.no_grad():
                    ref_logits = self.model(
                        rollouts["token_ids"],
                        attention_mask=rollouts["attention_mask"],
                    ).logits

                # Policy forward (train mode, with grad)
                self.model.train()
                policy_logits = self.model(
                    rollouts["token_ids"],
                    attention_mask=rollouts["attention_mask"],
                ).logits

                # Compute loss
                loss = self.loss_fn(
                    policy_logits,
                    ref_logits,
                    rollouts["token_ids"],
                    rollouts["attention_mask"],
                    rewards,
                )

                # Backprop
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Logging
                if global_step % self.config.logging_steps == 0:
                    LOGGER.info(
                        "Epoch %d, Step %d, Loss: %.4f",
                        epoch,
                        global_step,
                        loss.item(),
                    )

                global_step += 1

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        LOGGER.info("Checkpoint saved to %s", output_dir)

    def evaluate(self, eval_dataset: MathProblemDataset, num_samples: int = 100):
        """Evaluate model on benchmark."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(min(num_samples, len(eval_dataset))):
                item = eval_dataset[i]
                prompt = item["prompt"]

                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.6,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1,
                )

                generated = self.tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                # Check correctness (simplified)
                if self._check_correctness(generated, item["answer"]):
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)
        LOGGER.info("Eval accuracy: %.2f%% (%d/%d)", 100.0 * accuracy, correct, total)
        return accuracy

    def _check_correctness(self, generated: str, answer: str) -> bool:
        """Check if generated answer matches ground truth."""
        # Simplified: extract numerical answer and compare
        # (Full version needs robust answer extraction)
        return answer.strip() in generated


def main():
    """Run GRPO baseline training."""
    logging.basicConfig(level=logging.INFO)

    # Config
    config = GRPOTrainingConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="../results/phase1/grpo",
        K=8,
        learning_rate=5.0e-6,
        num_train_epochs=3,
    )

    # Data
    train_dataset = MathProblemDataset(
        data_path="data/numina_cot.jsonl",
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
        max_prompt_length=256,
        max_response_length=512,
    )

    # Trainer
    trainer = GRPOTrainer(config, train_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
