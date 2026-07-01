"""
Loss functions for GRPO and EGSPO-CA v2.

Reference:
  - GRPO: shao2024deepseekmath
  - EGSPO-CA v2: Eq. (4) (weighted clipped surrogate objective)
"""

import logging
from typing import Literal

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def compute_grpo_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute GRPO group-normalized advantages.

    A^(k) = (r^(k) - mean(r)) / (std(r) + eps)

    Args:
        rewards: (batch, K) rewards for K rollouts per prompt
        eps: numerical stability

    Returns:
        advantages: (batch, K)
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True, correction=int(rewards.size(-1) > 1))
    # Handle case where std is NaN (K=1) or zero (all rewards same)
    std_r = torch.nan_to_num(std_r, nan=0.0, posinf=0.0, neginf=0.0)
    advantages = (rewards - mean_r) / (std_r + eps)
    # When std=0, all advantages should be 0
    zero_std_mask = (std_r < eps)
    advantages = torch.where(zero_std_mask, torch.zeros_like(advantages), advantages)
    return advantages  # (batch, K)


def grpo_loss(
    logits_policy: torch.Tensor,
    logits_old: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """
    GRPO clipped surrogate loss (uniform credit to all tokens).

    L = -E[n,k] [ sum_t min(rho_t * A^(k), clip(rho_t, 1±eps) * A^(k)) ]

    Args:
        logits_policy: (batch, seq_len, vocab_size) current policy logits
        logits_old: (batch, seq_len, vocab_size) old policy logits
        attention_mask: (batch, seq_len) 1 for valid, 0 for pad
        advantages: (batch,) GRPO advantages per rollout
        clip_epsilon: PPO clip parameter

    Returns:
        loss: scalar
    """
    # Compute log probabilities
    log_probs_policy = F.log_softmax(logits_policy, dim=-1)
    log_probs_old = F.log_softmax(logits_old, dim=-1)

    # Importance ratio rho_t = pi_new / pi_old
    # For efficiency, compute per-token log prob of the actual token
    # This requires token IDs
    # For now, assume we have log_probs of selected tokens
    # (This is a simplified version; full version needs token IDs)

    # Placeholder: this needs token IDs to compute actual log probs
    # The full implementation is in the trainer

    raise NotImplementedError(
        "grpo_loss needs token IDs. Use GRPOTrainer.compute_loss instead."
    )


def egspo_ca_loss(
    logits_policy: torch.Tensor,
    logits_old: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    credit_weights: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """
    EGSPO-CA v2 weighted clipped surrogate loss.

    L = -E[n,k] [ sum_t w_t^(k) * min(rho_t * A^(k), clip(rho_t, 1±eps) * A^(k)) ]

    Reference: Eq. (4) in paper.

    Args:
        logits_policy: (batch, seq_len, vocab_size)
        logits_old: (batch, seq_len, vocab_size)
        token_ids: (batch, seq_len) token IDs of the rollout
        attention_mask: (batch, seq_len)
        advantages: (batch,) GRPO advantages per rollout
        credit_weights: (batch, seq_len) w_t^(k) from multiplicative dual gate
        clip_epsilon: PPO clip parameter

    Returns:
        loss: scalar
    """
    # Gather log probs for actual tokens (serial to reduce peak memory)
    log_probs_policy = F.log_softmax(logits_policy, dim=-1)
    log_p_policy = torch.gather(
        log_probs_policy, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len)
    del log_probs_policy  # Free (batch, seq_len, vocab) tensor

    log_probs_old = F.log_softmax(logits_old, dim=-1)
    log_p_old = torch.gather(
        log_probs_old, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len)
    del log_probs_old

    # Importance ratio
    log_ratio = log_p_policy - log_p_old
    ratio = torch.exp(log_ratio)  # (batch, seq_len)

    # Expand advantages to (batch, seq_len)
    adv_expanded = advantages.unsqueeze(-1).expand_as(log_p_policy)

    # Clipped objective
    clipped_ratio = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    )
    clipped_objective = torch.minimum(
        ratio * adv_expanded,
        clipped_ratio * adv_expanded,
    )  # (batch, seq_len)

    # Apply credit weights
    weighted_objective = clipped_objective * credit_weights

    # Mask out padding
    valid_mask = attention_mask.bool()
    loss = -weighted_objective[valid_mask].mean()

    return loss


class GRPOLoss:
    """
    Computes GRPO loss for a batch of rollouts.

    Usage:
        loss_fn = GRPOLoss(K=8, clip_epsilon=0.2)
        loss = loss_fn(policy_model, old_model, batch)
    """

    def __init__(self, K: int = 8, clip_epsilon: float = 0.2):
        self.K = K
        self.clip_epsilon = clip_epsilon

    def __call__(
        self,
        policy_logits: torch.Tensor,
        old_logits: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        Args:
            policy_logits: (batch, seq_len, vocab_size)
            old_logits: (batch, seq_len, vocab_size)
            token_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            rewards: (batch,) outcome rewards

        Returns:
            loss: scalar
        """
        # Reshape: batch = N * K (N prompts, K rollouts each)
        batch_size = policy_logits.shape[0]
        N = batch_size // self.K
        assert batch_size % self.K == 0, \
            f"Batch size {batch_size} not divisible by K={self.K}"

        # Compute advantages (group-normalized within each prompt)
        rewards_reshaped = rewards.view(N, self.K)
        advantages = compute_grpo_advantages(rewards_reshaped)  # (N, K)
        advantages_flat = advantages.view(-1)  # (N*K,)

        # Compute loss
        loss = egspo_ca_loss(
            policy_logits,
            old_logits,
            token_ids,
            attention_mask,
            advantages_flat,
            torch.ones_like(attention_mask.float()),  # Uniform weights for GRPO
            self.clip_epsilon,
        )

        return loss


class EGSPOCALoss:
    """
    Computes EGSPO-CA v2 loss with T-BICC credit weights.

    Usage:
        loss_fn = EGSPOCALoss(K=8, clip_epsilon=0.2)
        loss = loss_fn(policy_model, old_model, batch, credit_weights)
    """

    def __init__(self, K: int = 8, clip_epsilon: float = 0.2):
        self.K = K
        self.clip_epsilon = clip_epsilon

    def __call__(
        self,
        policy_logits: torch.Tensor,
        old_logits: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        credit_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute EGSPO-CA v2 loss.

        Args:
            policy_logits: (batch, seq_len, vocab_size)
            old_logits: (batch, seq_len, vocab_size)
            token_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            rewards: (batch,) outcome rewards
            credit_weights: (batch, seq_len) w_t from multiplicative dual gate

        Returns:
            loss: scalar
        """
        batch_size = policy_logits.shape[0]
        N = batch_size // self.K
        assert batch_size % self.K == 0, \
            f"Batch size {batch_size} not divisible by K={self.K}"

        # Compute advantages
        rewards_reshaped = rewards.view(N, self.K)
        advantages = compute_grpo_advantages(rewards_reshaped)
        advantages_flat = advantages.view(-1)

        # Compute loss with credit weights
        loss = egspo_ca_loss(
            policy_logits,
            old_logits,
            token_ids,
            attention_mask,
            advantages_flat,
            credit_weights,
            self.clip_epsilon,
        )

        return loss
