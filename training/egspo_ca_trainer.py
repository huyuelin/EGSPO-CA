"""
EGSPO-CA v2 trainer.

Extends GRPO with T-BICC token-level credit assignment.
Reference: Section 4 (Method: EGSPO-CA).

Key differences from GRPO:
  1. Value head V_phi for truncated reward estimation
  2. T-BICC scores for token-level causal credit
  3. Multiplicative dual gating (entropy x T-BICC)
  4. Weighted clipped surrogate loss

This is a minimal single-GPU implementation for proof-of-concept.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from credit.tbicc import (
    compute_tbicc_scores,
    compute_entropy_signal,
    multiplicative_dual_gate,
)
from data.numina_loader import MathProblemDataset, compute_outcome_returns
from models.value_head import ValueHead, ValueHeadTrainer
from training.loss import EGSPOCALoss
from training.reward_utils import exact_match_reward, batch_exact_match_reward

LOGGER = logging.getLogger(__name__)


@dataclass
class EGSPOCATrainingConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "../results/phase1/egspo_ca"
    # GRPO params
    K: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    # Training params
    learning_rate: float = 5.0e-6
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    # T-BICC params
    delta: int = 8
    s0: float = 0.5
    eta: float = 0.5
    beta: float = 0.6
    gamma: float = 0.1
    lambda_ema: float = 0.9
    k_min: int = 2
    # Value head params
    value_head_hidden_dim: int = 1024
    value_head_warmup_steps: int = 200
    value_head_r2_threshold: float = 0.6
    # Misc
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"


class EGSPOCATrainer:
    """
    EGSPO-CA v2 trainer with T-BICC credit assignment.

    Algorithm:
      1. For each prompt, generate K rollouts
      2. Compute hidden states (for value head and T-BICC)
      3. Compute T-BICC scores (token-level causal credit)
      4. Compute multiplicative dual gate weights
      5. Compute weighted clipped surrogate loss
      6. Backprop and update
    """

    def __init__(
        self,
        config: EGSPOCATrainingConfig,
        train_dataset: MathProblemDataset,
        eval_datasets: Optional[Dict[str, MathProblemDataset]] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets

        # Load model and tokenizer (single model, ref logits via eval+no_grad pass)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Value head (must match model's dtype before creating optimizer)
        hidden_dim = self.model.config.hidden_size
        self.value_head = ValueHead(
            input_dim=hidden_dim,
            hidden_dim=config.value_head_hidden_dim,
        ).to(device=self.model.device, dtype=self.model.dtype)
        self.value_head_trainer = ValueHeadTrainer(
            self.value_head,
            warmup_steps=config.value_head_warmup_steps,
            r2_threshold=config.value_head_r2_threshold,
        )

        # Loss function
        self.loss_fn = EGSPOCALoss(
            K=config.K, clip_epsilon=config.clip_epsilon
        )

        # Optimizer (policy + value head) - 8-bit Adam for memory efficiency
        # Adam state for 7B model = ~56GB in fp32, 8-bit reduces to ~14GB
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                list(self.model.parameters())
                + list(self.value_head.parameters()),
                lr=config.learning_rate,
            )
            LOGGER.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                list(self.model.parameters())
                + list(self.value_head.parameters()),
                lr=config.learning_rate,
            )
            LOGGER.info("Using standard AdamW optimizer.")

        # T-BICC EMA state
        self.previous_tbicc = None
        # Cached ref logits from compute_credit_weights (avoid separate forward)
        self._cached_ref_logits = None

        LOGGER.info("EGSPOCATrainer initialized. Model: %s", config.model_name)

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

    def generate_rollouts_with_hidden(
        self,
        prompts: List[str],
        K: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate K rollouts per prompt, returning hidden states.

        Returns:
            dict with keys:
              - token_ids: (N*K, seq_len)
              - hidden_states: (N*K, seq_len, hidden_dim)
              - logits: (N*K, seq_len, vocab_size)
              - attention_mask: (N*K, seq_len)
        """
        N = len(prompts)
        all_token_ids = []
        all_hidden_states = []
        all_logits = []
        all_attention_masks = []

        for i, prompt in enumerate(prompts):
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

            rollout_ids = outputs.sequences
            all_token_ids.append(rollout_ids)

            # Extract hidden states via forward pass
            # model.generate() doesn't reliably return full hidden states,
            # so we run a forward pass to get them for T-BICC computation
            attn_mask = (rollout_ids != self.tokenizer.pad_token_id).long()
            with torch.no_grad():
                fwd_outputs = self.model(
                    rollout_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                )
                last_hidden = fwd_outputs.hidden_states[-1]  # (K, seq_len, hidden_dim)
                all_hidden_states.append(last_hidden)

        # Concatenate
        token_ids = torch.cat(all_token_ids, dim=0)
        hidden_states = torch.cat(all_hidden_states, dim=0)
        attention_mask = (token_ids != self.tokenizer.pad_token_id).long()

        return {
            "token_ids": token_ids,
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        return_ref_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute T-BICC credit weights for all tokens.
        
        When return_ref_logits=True, also returns the logits from the
        no-grad forward pass, avoiding a separate reference forward pass.

        Returns:
            weights: (N*K, seq_len) credit weights w_t
            (optionally) ref_logits: (N*K, seq_len, vocab_size)
        """
        token_ids = rollouts["token_ids"]
        hidden_states = rollouts["hidden_states"]
        attention_mask = rollouts["attention_mask"]

        # Get logits (for choice divergence) and hidden states
        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Cache logits as ref_logits to avoid separate forward pass
        self._cached_ref_logits = logits

        # Compute truncated values
        truncated_values = self.value_head_trainer.get_truncated_value(
            model_hidden_states, delta=self.config.delta
        )

        # Compute T-BICC scores
        tbicc_scores, diagnostics = compute_tbicc_scores(
            truncated_values=truncated_values,
            token_ids=token_ids,
            hidden_states=model_hidden_states,
            attention_mask=attention_mask,
            logits=logits,
            delta=self.config.delta,
            s0=self.config.s0,
            eta=self.config.eta,
            lambda_ema=self.config.lambda_ema,
            k_min=self.config.k_min,
            previous_scores=self.previous_tbicc,
        )

        # Update EMA state
        self.previous_tbicc = tbicc_scores

        # Compute entropy signal
        entropy = compute_entropy_signal(logits, attention_mask)

        # Apply confidence-gated warmup
        if self.value_head_trainer.is_tbicc_active():
            # Use full multiplicative dual gate
            weights = multiplicative_dual_gate(
                entropy,
                tbicc_scores,
                attention_mask,
                beta=self.config.beta,
                gamma=self.config.gamma,
            )
        else:
            # Warmup: use entropy only (T-BICC gated off)
            LOGGER.info("Warmup active: using entropy gating only.")
            weights = entropy * attention_mask.float()

        if return_ref_logits:
            return weights, logits
        return weights

    def train(self):
        """Main training loop."""
        self.model.eval()
        self.value_head.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

        global_step = 0
        for epoch in range(self.config.num_train_epochs):
            for batch in train_loader:
                # Generate rollouts with hidden states
                rollouts = self.generate_rollouts_with_hidden(
                    batch["prompts"], K=self.config.K
                )

                # Compute rewards (repeat answers K times per rollout)
                repeated_answers = [
                    ans for ans in batch["answers"] for _ in range(self.config.K)
                ]
                rewards = self.compute_reward(
                    rollouts["token_ids"], repeated_answers
                )

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

                # Compute credit weights (T-BICC + gating)
                credit_weights = self.compute_credit_weights(rollouts, rewards)
                ref_logits = self._cached_ref_logits  # Reuse from T-BICC forward pass
                if ref_logits is None:
                    # Fallback: separate reference forward (for baselines that don't cache)
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

                # Compute value head loss (MSE against discounted returns)
                discounted_returns = compute_outcome_returns(
                    rewards, rollouts["attention_mask"]
                )
                value_loss = self.value_head_trainer.compute_value_loss(
                    rollouts["hidden_states"],
                    discounted_returns,
                    rollouts["attention_mask"],
                )

                # Compute EGSPO-CA loss
                policy_loss = self.loss_fn(
                    policy_logits,
                    ref_logits,
                    rollouts["token_ids"],
                    rollouts["attention_mask"],
                    rewards,
                    credit_weights,
                )

                # Total loss
                total_loss = policy_loss + value_loss
                total_loss = total_loss / self.config.gradient_accumulation_steps
                total_loss.backward()

                # Update value head warmup state
                if global_step % self.config.logging_steps == 0:
                    r2 = self.value_head_trainer.compute_r2(
                        rollouts["hidden_states"],
                        discounted_returns,
                        rollouts["attention_mask"],
                    )
                    self.value_head_trainer.update_step(r2_heldout=r2)

                # Optimizer step
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters())
                        + list(self.value_head.parameters()),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Logging
                if global_step % self.config.logging_steps == 0:
                    LOGGER.info(
                        "Epoch %d, Step %d, Policy Loss: %.4f, Value Loss: %.4f",
                        epoch,
                        global_step,
                        policy_loss.item(),
                        value_loss.item(),
                    )

                global_step += 1

                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)

    def compute_reward(
        self,
        token_ids: torch.Tensor,
        answers: List[str],
    ) -> torch.Tensor:
        """Compute outcome reward (0 or 1) via exact match of numerical answer."""
        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )

        # Compute exact match rewards
        rewards_list = batch_exact_match_reward(generated_texts, answers)
        rewards = torch.tensor(rewards_list, device=token_ids.device, dtype=self.model.dtype)
        return rewards

    def save_checkpoint(self, step: int):
        """Save model and value head checkpoint."""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(
            self.value_head.state_dict(),
            output_dir / "value_head.pt",
        )

        LOGGER.info("Checkpoint saved to %s", output_dir)


def main():
    """Run EGSPO-CA v2 training."""
    logging.basicConfig(level=logging.INFO)

    config = EGSPOCATrainingConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="../results/phase1/egspo_ca",
        K=8,
        learning_rate=5.0e-6,
        num_train_epochs=3,
    )

    train_dataset = MathProblemDataset(
        data_path="data/numina_cot.jsonl",
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
        max_prompt_length=256,
        max_response_length=512,
    )

    trainer = EGSPOCATrainer(config, train_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
