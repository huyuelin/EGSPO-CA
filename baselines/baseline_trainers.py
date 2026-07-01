"""
Baseline trainers for EGSPO-CA v2 reproduction.
Phase 3D: 10 baseline methods organized by complexity.

Simple baselines (override loss/credit/advantages):
  S1: EGSPO  - Entropy-only gating (no T-BICC)
  S2: Dr.GRPO - Length-corrected GRPO
  S3: GTPO    - Entropy-weighted token-level advantage
  S4: DAPO    - Decoupled asymmetric clipping

Medium baselines (need new training structure):
  S5: TEMPO   - Temporal Difference credit
  S6: DelTA   - Embedding-centric credit

Complex baselines (need external APIs / heavy compute):
  S7: SPO     - Segment-level MC estimation
  S8: HAPO    - CMI capacity-based credit
  S9: CAPO    - Generative PRM (needs API)
  S10: CF Credit - Counterfactual mask-forward

Usage:
  from baselines.baseline_trainers import create_baseline_trainer
  trainer = create_baseline_trainer("egspo", config, train_dataset)
  trainer.train()
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Union

import torch
import torch.nn.functional as F

from training.egspo_ca_trainer import (
    EGSPOCATrainingConfig,
    EGSPOCATrainer,
)
from training.grpo_trainer import GRPOTrainingConfig, GRPOTrainer
from training.loss import compute_grpo_advantages, egspo_ca_loss
from data.numina_loader import MathProblemDataset
from credit.tbicc import compute_entropy_signal

LOGGER = logging.getLogger(__name__)

BaselineName = Literal[
    "egspo", "drgrpo", "gtpo", "dapo",
    "tempo", "delta", "spo", "hapo", "capo", "cfcredit"
]

BASELINE_NAMES: List[str] = [
    "egspo", "drgrpo", "gtpo", "dapo",
    "tempo", "delta", "spo", "hapo", "capo", "cfcredit"
]


# ═══════════════════════════════════════════════════════════════
# S1: EGSPO - Entropy-Gated Group Relative Policy Optimization
# ═══════════════════════════════════════════════════════════════

class EGSPOTrainer(EGSPOCATrainer):
    """
    EGSPO: Entropy-only gating, no T-BICC.
    
    This is the simplest baseline. Replaces the full multiplicative
    dual gate with pure entropy signal as token-level credit weights.
    
    Equivalent to EGSPO-CA with beta=1.0, gamma=0.0 (T-BICC disabled).
    
    Reference: EGSPO (arXiv 2025) - entropy-gated GRPO variant
    """

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        return_ref_logits: bool = False,
    ) -> torch.Tensor:
        """Entropy-only credit weights, no T-BICC."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]

        # Forward pass for logits (entropy computation only)
        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Still train value head (useful for ablation comparison)
        truncated_values = self.value_head_trainer.get_truncated_value(
            model_hidden_states, delta=self.config.delta
        )
        discounted_returns = self._compute_returns(rewards, attention_mask)

        # Value head update (collect loss outside training loop)
        _ = self.value_head_trainer.compute_value_loss(
            model_hidden_states, discounted_returns, attention_mask,
        )

        # Pure entropy signal (no T-BICC)
        entropy_signal = compute_entropy_signal(logits, attention_mask)
        weights = entropy_signal * attention_mask.float()

        if return_ref_logits:
            return weights, logits
        return weights

    def _compute_returns(self, rewards, attention_mask):
        """Compute discounted returns for value head (placeholder)."""
        from data.numina_loader import compute_outcome_returns
        return compute_outcome_returns(rewards, attention_mask)


# ═══════════════════════════════════════════════════════════════
# S2: Dr.GRPO - Dropout-Regularized / Length-Corrected GRPO
# ═══════════════════════════════════════════════════════════════

@dataclass
class DrGRPOTrainingConfig(GRPOTrainingConfig):
    """Dr.GRPO config with length penalty parameter."""
    length_penalty: float = 0.1  # Penalty weight for response length


class DrGRPOTrainer(GRPOTrainer):
    """
    Dr.GRPO: Length-corrected GRPO.
    
    Divides advantages by the normalized sequence length to penalize
    verbose generations. Uses the standard GRPO clipped surrogate loss.
    
    Key modification: compute_length_corrected_advantages()
    
    Reference: Dr.GRPO (2025) - length normalization for math reasoning
    """

    def __init__(self, config: DrGRPOTrainingConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.length_penalty = config.length_penalty

    def compute_length_corrected_advantages(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Length-corrected GRPO advantages.
        
        A_len^(k) = A_grpo^(k) - lambda * (len_k / mean_len - 1)
        
        This penalizes rollouts that are longer than average.
        """
        N = rewards.shape[0]
        K = self.config.K
        batch_size = N * K
        assert batch_size % K == 0

        # Standard GRPO advantages
        rewards_reshaped = rewards.view(N, K)
        advantages = compute_grpo_advantages(rewards_reshaped)
        advantages_flat = advantages.view(-1)

        # Compute sequence lengths (non-padding tokens)
        seq_lengths = attention_mask.sum(dim=-1).float()  # (batch,)
        mean_len = seq_lengths.mean() + 1e-8

        # Length penalty: penalize sequences longer than average
        length_factor = (seq_lengths / mean_len) - 1.0  # zero for avg-length
        length_penalty = self.length_penalty * length_factor

        return advantages_flat - length_penalty


# ═══════════════════════════════════════════════════════════════
# S3: GTPO - Guided Token-level Policy Optimization
# ═══════════════════════════════════════════════════════════════

class GTPOTrainer(EGSPOCATrainer):
    """
    GTPO: Entropy-weighted token-level policy optimization.
    
    Uses per-token entropy as advantage modulation, replacing the
    standard GRPO outcome-based advantage with token-level entropy
    signal combined with the outcome advantage.
    
    Key difference from EGSPO: EGSPO uses entropy as credit weights
    multiplied with outcome advantage. GTPO uses entropy directly
    as a per-token advantage signal, combined multiplicatively
    with the outcome advantage.
    
    Reference: GTPO (2025) - entropy-guided token-level optimization
    """

    def compute_gate_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GTPO gate weights: entropy-modulated credit.
        
        w_t = alpha * H_t + (1-alpha) * (outcome_reward / seq_len)
        where H_t is the per-token entropy signal.
        """
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]

        # Forward for entropy
        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Still update value head
        truncated_values = self.value_head_trainer.get_truncated_value(
            model_hidden_states, delta=self.config.delta
        )

        # Entropy signal
        entropy = compute_entropy_signal(logits, attention_mask)

        # Token-level outcome reward spread
        seq_lengths = attention_mask.sum(dim=-1, keepdim=True).float()
        uniform_reward = rewards.unsqueeze(-1) / seq_lengths.clamp(min=1)
        uniform_reward = uniform_reward * attention_mask.float()

        # Blend: 50% entropy, 50% uniform outcome
        alpha = 0.5
        gate_weights = alpha * entropy + (1.0 - alpha) * uniform_reward

        return gate_weights


# ═══════════════════════════════════════════════════════════════
# S4: DAPO - Decoupled Alignment Policy Optimization
# ═══════════════════════════════════════════════════════════════

@dataclass
class DAPOTrainingConfig(GRPOTrainingConfig):
    """DAPO config with decoupled clip parameters."""
    clip_epsilon_high: float = 0.28  # Upper clip for positive advantages
    clip_epsilon_low: float = 0.2    # Lower clip for negative advantages


class DAPOTrainer(GRPOTrainer):
    """
    DAPO: Decoupled clipping for positive/negative advantages.
    
    Uses different clip ratios for positive and negative advantages:
    - Positive advantages: more conservative (clip_epsilon_high)
    - Negative advantages: more aggressive (clip_epsilon_low)
    
    Key: decoupled_clipped_loss()
    
    Reference: DAPO (2025) - decoupled PPO for alignment
    """

    def __init__(self, config: DAPOTrainingConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.clip_epsilon_high = config.clip_epsilon_high
        self.clip_epsilon_low = config.clip_epsilon_low

    def compute_decoupled_loss(
        self,
        policy_logits: torch.Tensor,
        old_logits: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        DAPO loss with asymmetric clipping.
        
        For A > 0: clip(rho, 1, 1+eps_high)
        For A < 0: clip(rho, 1-eps_low, 1)
        """
        batch_size = policy_logits.shape[0]
        N = batch_size // self.config.K
        K = self.config.K

        # Standard advantages
        rewards_reshaped = rewards.view(N, K)
        advantages = compute_grpo_advantages(rewards_reshaped)
        advantages_flat = advantages.view(-1)

        # Log probs (serial for memory)
        log_probs_policy = F.log_softmax(policy_logits, dim=-1)
        log_p_policy = torch.gather(
            log_probs_policy, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs_policy

        log_probs_old = F.log_softmax(old_logits, dim=-1)
        log_p_old = torch.gather(
            log_probs_old, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs_old

        # Ratio
        ratio = torch.exp(log_p_policy - log_p_old)

        # Expand advantages
        adv_expanded = advantages_flat.unsqueeze(-1).expand_as(log_p_policy)

        # Asymmetric clipping
        clip_high = 1.0 + self.clip_epsilon_high  # upper bound
        clip_low = 1.0 - self.clip_epsilon_low    # lower bound

        clipped_ratio_high = torch.clamp(ratio, max=clip_high)
        clipped_ratio_low = torch.clamp(ratio, min=clip_low)

        # Use different clipping for positive/negative advantages
        pos_mask = adv_expanded >= 0
        clipped = torch.where(
            pos_mask,
            clipped_ratio_high * adv_expanded,
            clipped_ratio_low * adv_expanded,
        )

        # DAPO objective
        unclipped = ratio * adv_expanded
        objective = torch.minimum(unclipped, clipped)

        # Mask padding
        valid_mask = attention_mask.bool()
        loss = -objective[valid_mask].mean()

        return loss


# ═══════════════════════════════════════════════════════════════
# S5: TEMPO - Temporal Difference Prefix-Tree Credit
# ═══════════════════════════════════════════════════════════════

class TEMPOTrainer(EGSPOCATrainer):
    """
    TEMPO: Temporal Difference credit assignment.
    
    Assigns credit sequentially along the token sequence using
    TD-style updates. Each token's credit is the difference
    between predicted future reward and past expected reward.
    
    Reference: TEMPO (2025) - TD credit for language model reasoning
    """

    def compute_td_credit(
        self,
        truncated_values: torch.Tensor,
        attention_mask: torch.Tensor,
        gamma_td: float = 0.99,
    ) -> torch.Tensor:
        """
        TD credit: credit_t = V(t+1) + gamma * TD_residual
        
        This is a simplified version using the value head's
        truncated predictions as the TD target.
        """
        batch_size, seq_len = truncated_values.shape

        # Shift values: V(t+1) for each position t
        values_shifted = torch.zeros_like(truncated_values)
        values_shifted[:, :-1] = truncated_values[:, 1:]

        # TD error: V(t+1) - V(t) 
        td_credit = (values_shifted - truncated_values) / (truncated_values.abs() + 1e-8)
        td_credit = td_credit * attention_mask.float()

        # Normalize along sequence
        td_sum = td_credit.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        td_credit = td_credit / td_sum

        return td_credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """TD credit with entropy gating."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]

        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Value head predictions
        truncated_values = self.value_head_trainer.get_truncated_value(
            model_hidden_states, delta=self.config.delta
        )

        # TD credit
        td_credit = self.compute_td_credit(truncated_values, attention_mask)

        # Entropy gate
        entropy = compute_entropy_signal(logits, attention_mask)

        # Combined: TD credit gated by entropy
        weights = td_credit * entropy * attention_mask.float()

        return weights


# ═══════════════════════════════════════════════════════════════
# S6: DelTA - Embedding-Centric Credit
# ═══════════════════════════════════════════════════════════════

class DelTATrainer(EGSPOCATrainer):
    """
    DelTA: Embedding-centric credit assignment.
    
    Assigns credit based on how "far" each token's hidden state
    is from the embedding center of successful rollouts.
    
    Reference: DelTA (2025) - embedding-based credit for LLMs
    """

    def compute_embedding_credit(
        self,
        hidden_states: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 2,
    ) -> torch.Tensor:
        """
        Compute embedding-centric credit.
        
        For each batch item, credit score reflects cosine distance
        to the center of top-K successful trajectories.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Find successful rollouts (reward > 0)
        success_mask = rewards > 0.5
        if success_mask.sum() < 1:
            # No successful rollouts: uniform credit
            return attention_mask.float() / attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        # Mean hidden state of successful trajectories (last token)
        success_hidden = hidden_states[success_mask, -1, :]  # (S, hidden_dim)
        success_center = success_hidden.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        # Cosine similarity for each token to success center
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        center_norm = F.normalize(success_center, p=2, dim=-1)

        cosine_sim = (hidden_norm * center_norm).sum(dim=-1)  # (batch, seq_len)
        credit = (cosine_sim + 1.0) / 2.0  # normalize to [0, 1]
        credit = credit * attention_mask.float()

        return credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Embedding-centric credit with entropy gate."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]
        hidden_states = rollouts["hidden_states"]

        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Embedding credit
        embed_credit = self.compute_embedding_credit(
            model_hidden_states, rewards, attention_mask
        )

        # Entropy gate
        entropy = compute_entropy_signal(logits, attention_mask)

        weights = embed_credit * entropy * attention_mask.float()
        return weights


# ═══════════════════════════════════════════════════════════════
# S7: SPO - Segment-Level Monte Carlo
# ═══════════════════════════════════════════════════════════════

class SPOTrainer(EGSPOCATrainer):
    """
    SPO: Segment-Level Monte Carlo credit assignment.
    
    Divides each rollout into segments and assigns credit
    via MC estimation over segment prefixes.
    
    Reference: SPO (2025) - segment-level policy optimization
    """

    def compute_segment_credit(
        self,
        hidden_states: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_size: int = 32,
    ) -> torch.Tensor:
        """
        Segment-level MC credit.
        
        Partitions sequence into segments of size `segment_size`.
        Each token in segment gets uniform share of segment credit.
        Segment credit = expected reward difference across segments.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        credit = torch.zeros(batch_size, seq_len, device=device)
        num_segments = (seq_len + segment_size - 1) // segment_size

        for s in range(num_segments):
            start = s * segment_size
            end = min(start + segment_size, seq_len)

            # Progressive: later segments get credit based on cumulative progress
            progress_weight = s / max(num_segments - 1, 1)
            token_weight = progress_weight / max(end - start, 1)
            credit[:, start:end] = token_weight

        # Modulate by outcome reward
        credit = credit * rewards.unsqueeze(-1) * attention_mask.float()
        return credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Segment credit weights."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]
        hidden_states = rollouts["hidden_states"]

        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # Segment credit
        seg_credit = self.compute_segment_credit(
            model_hidden_states, rewards, attention_mask
        )

        # Entropy gate
        entropy = compute_entropy_signal(logits, attention_mask)

        weights = seg_credit * entropy * attention_mask.float()
        return weights


# ═══════════════════════════════════════════════════════════════
# S8: HAPO - CMI Capacity-Based Credit
# ═══════════════════════════════════════════════════════════════

class HAPOTrainer(EGSPOCATrainer):
    """
    HAPO: CMI (Conditional Mutual Information) capacity-based credit.
    
    Uses information-theoretic measures to assign credit to tokens
    that contribute most to the outcome based on CMI between
    the token distribution and the final reward.
    
    Reference: HAPO (2025) - information-theoretic credit assignment
    """

    def compute_cmi_credit(
        self,
        logits: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        num_bins: int = 10,
    ) -> torch.Tensor:
        """
        Approximate CMI-based credit.
        
        For each position t, compute the mutual information between
        the token distribution at t and the final outcome.
        Uses a coarse discretization (bins) of the reward space.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Discretize rewards into bins
        reward_min = rewards.min()
        reward_max = rewards.max() + 1e-8
        reward_bins = ((rewards - reward_min) / reward_max * num_bins).long().clamp(0, num_bins - 1)

        # For each bin, compute average logit entropy
        credit = torch.zeros(batch_size, seq_len, device=logits.device)

        for b in range(num_bins):
            bin_mask = (reward_bins == b).float()
            if bin_mask.sum() < 1:
                continue

            # Entropy of token distribution for this bin
            bin_logits = logits * bin_mask.unsqueeze(-1).unsqueeze(-1)
            probs = F.softmax(bin_logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (batch, seq_len)

            # Higher entropy = more uncertainty = potentially less credit
            # Lower entropy = more confident = potentially more credit
            bin_credit = -entropy  # negation: lower entropy = higher credit
            credit = credit + bin_credit * bin_mask.unsqueeze(-1)

        # Normalize
        credit = F.softmax(credit, dim=-1) * attention_mask.float()
        return credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """CMI credit with entropy gate."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]

        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # CMI credit
        cmi_credit = self.compute_cmi_credit(logits, rewards, attention_mask)

        # Entropy gate
        entropy = compute_entropy_signal(logits, attention_mask)

        weights = cmi_credit * entropy * attention_mask.float()
        return weights


# ═══════════════════════════════════════════════════════════════
# S9: CAPO - Generative Process Reward Model
# ═══════════════════════════════════════════════════════════════

class CAPOTrainer(EGSPOCATrainer):
    """
    CAPO: Generative Process Reward Model.
    
    Uses an LLM API to score intermediate steps for credit assignment.
    Each step is judged on relevance and correctness toward the answer.
    
    NOTE: Requires API access (Hunyuan / DashScope) for step scoring.
    For Phase 3D proof-of-concept, uses a heuristic approximation.
    
    Reference: CAPO (2025) - generative PRM for process supervision
    """

    def compute_prm_credit_heuristic(
        self,
        logits: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        step_interval: int = 50,  # Evaluate every 50 tokens
    ) -> torch.Tensor:
        """
        Heuristic PRM credit (no API needed).
        
        Uses the model's own confidence (logit max) at step boundaries
        as an approximate process reward signal.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Token-level confidence
        max_logits = logits.max(dim=-1).values  # (batch, seq_len)
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        # Step boundaries: every `step_interval` tokens
        credit = torch.zeros(batch_size, seq_len, device=logits.device)
        
        for t in range(0, seq_len, step_interval):
            t_end = min(t + step_interval, seq_len)
            # Average confidence over the step
            step_conf = max_probs[:, t:t_end].mean(dim=-1, keepdim=True)
            credit[:, t:t_end] = step_conf

        # Modulate by outcome reward  
        outcome_factor = rewards.unsqueeze(-1)
        credit = credit * outcome_factor * attention_mask.float()

        return credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """PRM credit weights."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]

        with torch.no_grad():
            outputs = self.model(
                token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            model_hidden_states = outputs.hidden_states[-1]

        # PRM heuristic credit
        prm_credit = self.compute_prm_credit_heuristic(
            logits, rewards, attention_mask
        )

        # Entropy gate
        entropy = compute_entropy_signal(logits, attention_mask)

        weights = prm_credit * entropy * attention_mask.float()
        return weights


# ═══════════════════════════════════════════════════════════════
# S10: CF Credit - Counterfactual Mask-Forward
# ═══════════════════════════════════════════════════════════════

class CFCreditTrainer(EGSPOCATrainer):
    """
    CF Credit: Counterfactual Mask-Forward credit assignment.
    
    For each token position t, masks out tokens [t:] and re-runs
    the forward pass to see how the predicted outcome changes.
    The credit for token t is the difference in outcome prediction.
    
    Key: delta_score = V(masked_at_t) - V(full_sequence)
    
    NOTE: This is computationally intensive (O(seq_len) forward passes).
    For proof-of-concept, uses a sparse sampling strategy.
    
    Reference: CF Credit (2025) - counterfactual credit for LLMs
    """

    def compute_cf_credit(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_rate: float = 0.1,  # Sample 10% of positions
    ) -> torch.Tensor:
        """
        Counterfactual credit via sparse mask-forward.
        
        Only samples `sample_rate` of positions for efficiency.
        For unsampled positions, uses interpolation.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Full sequence value prediction
        full_value = self.value_head(hidden_states)  # (batch, seq_len, 1)

        credit = torch.zeros(batch_size, seq_len, device=device)

        # Sample positions for counterfactual evaluation
        num_samples = max(1, int(seq_len * sample_rate))
        sample_positions = torch.randperm(seq_len, device=device)[:num_samples]
        sample_positions = sorted(sample_positions.tolist())

        for t in sample_positions:
            # Create masked sequence (zero out from position t)
            cf_attention_mask = attention_mask.clone()
            cf_attention_mask[:, t:] = 0  # Mask from position t onward

            # Forward pass with masked sequence
            with torch.no_grad():
                outputs = self.model(
                    token_ids,
                    attention_mask=cf_attention_mask,
                    output_hidden_states=True,
                )
                cf_hidden = outputs.hidden_states[-1]
                cf_value = self.value_head(cf_hidden)  # (batch, seq_len, 1)

            # Credit = difference in value prediction at position t
            cf_contrib = full_value[:, t, 0] - cf_value[:, t, 0]  # (batch,)
            credit[:, t] = cf_contrib

            del outputs, cf_hidden, cf_value  # Free memory

        # Interpolate for unsampled positions
        if num_samples < seq_len:
            for b in range(batch_size):
                sampled = sorted(sample_positions)
                for i in range(len(sampled) - 1):
                    start, end = sampled[i], sampled[i + 1]
                    step = (credit[b, end] - credit[b, start]) / max(end - start, 1)
                    for t in range(start + 1, end):
                        credit[b, t] = credit[b, start] + step * (t - start)

        # Normalize
        credit = F.softmax(credit, dim=-1) * attention_mask.float()
        return credit

    def compute_credit_weights(
        self,
        rollouts: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """CF credit weights."""
        token_ids = rollouts["token_ids"]
        attention_mask = rollouts["attention_mask"]
        hidden_states = rollouts["hidden_states"]

        # CF credit
        cf_credit = self.compute_cf_credit(
            token_ids, hidden_states, attention_mask
        )

        # For CF, use credit directly (already modulated)
        weights = cf_credit * attention_mask.float()
        return weights


# ═══════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "egspo": EGSPOTrainer,
    "drgrpo": DrGRPOTrainer,
    "gtpo": GTPOTrainer,
    "dapo": DAPOTrainer,
    "tempo": TEMPOTrainer,
    "delta": DelTATrainer,
    "spo": SPOTrainer,
    "hapo": HAPOTrainer,
    "capo": CAPOTrainer,
    "cfcredit": CFCreditTrainer,
}


def create_baseline_trainer(
    method: BaselineName,
    config: Union[EGSPOCATrainingConfig, GRPOTrainingConfig, DrGRPOTrainingConfig, DAPOTrainingConfig],
    train_dataset: MathProblemDataset,
    eval_datasets: Optional[Dict[str, MathProblemDataset]] = None,
):
    """
    Create a trainer for the specified baseline method.
    
    Args:
        method: One of 'egspo', 'drgrpo', 'gtpo', 'dapo', 'tempo', 
                'delta', 'spo', 'hapo', 'capo', 'cfcredit'
        config: Training configuration (method-specific)
        train_dataset: Training data
        eval_datasets: Optional evaluation datasets
    
    Returns:
        Trainer instance ready for .train()
    """
    if method not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{method}'. Choose from: {BASELINE_NAMES}"
        )
    
    trainer_cls = BASELINE_REGISTRY[method]
    return trainer_cls(config, train_dataset, eval_datasets=eval_datasets)
