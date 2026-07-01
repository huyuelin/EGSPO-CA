"""
Value Head V_phi for EGSPO-CA v2.

A lightweight 2-layer MLP on the policy's final hidden layer.
~4M parameters for Qwen2.5-7B (4096 -> 1024 -> 1).

Trains via regression against discounted Monte-Carlo returns from GRPO rollouts.
Confidence-gated warmup: weights frozen until held-out R^2 > 0.6.

Reference:
  - Section 4.1 (Value Head and Adaptive Truncation)
  - Appendix A (Value-Head Training)
"""

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class ValueHeadConfig:
    hidden_dim: int = 1024
    dropout: float = 0.1
    warmup_steps: int = 200
    r2_threshold: float = 0.6
    loss_weight: float = 1.0


class ValueHead(nn.Module):
    """
    2-layer MLP value head.

    Attaches to any HuggingFace model by extracting the last hidden state.
    Shares the policy forward pass (zero additional computation for activations).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh activation as per paper Appendix A
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                torch.nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            values: (batch, seq_len, 1) or (batch, 1)
        """
        # Ensure value head is on the same device and dtype as the input
        target_device = hidden_states.device
        target_dtype = hidden_states.dtype
        if next(self.net.parameters()).device != target_device:
            self.net = self.net.to(device=target_device)
        if next(self.net.parameters()).dtype != target_dtype:
            self.net = self.net.to(dtype=target_dtype)

        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            bsz = original_shape[0]
            seq_len = original_shape[1]
            hidden_flat = hidden_states.reshape(bsz * seq_len, -1)
            values_flat = self.net(hidden_flat)
            values = values_flat.view(bsz, seq_len, 1)
        else:
            values = self.net(hidden_states)
        return values


class ValueHeadTrainer:
    """
    Manages value head training with confidence-gated warmup.

    - During warmup (first `warmup_steps`), only the value head trains
      (policy frozen). T-BICC weights are gated off (use entropy only).
    - After warmup, check held-out R^2. If > r2_threshold, activate T-BICC.
    - Value head loss: MSE against discounted Monte-Carlo returns.
    """

    def __init__(
        self,
        value_head: ValueHead,
        warmup_steps: int = 200,
        r2_threshold: float = 0.6,
        loss_weight: float = 1.0,
    ):
        self.value_head = value_head
        self.warmup_steps = warmup_steps
        self.r2_threshold = r2_threshold
        self.loss_weight = loss_weight
        self.current_step = 0
        self.warmup_active = True
        self.tbicc_active = False

    def compute_value_loss(
        self,
        hidden_states: torch.Tensor,
        target_returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted values and discounted MC returns.

        Args:
            hidden_states: (batch, seq_len, input_dim)
            target_returns: (batch, seq_len) discounted MC returns
            attention_mask: (batch, seq_len) 1 for valid, 0 for pad

        Returns:
            loss: scalar
        """
        predictions = self.value_head(hidden_states).squeeze(-1)  # (batch, seq_len)

        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            loss = nn.functional.mse_loss(
                predictions[valid_mask],
                target_returns[valid_mask],
            )
        else:
            loss = nn.functional.mse_loss(predictions, target_returns)

        return self.loss_weight * loss

    @torch.no_grad()
    def compute_r2(
        self,
        hidden_states: torch.Tensor,
        target_returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute held-out R^2 to decide warmup completion."""
        predictions = self.value_head(hidden_states).squeeze(-1)

        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            pred_flat = predictions[valid_mask]
            target_flat = target_returns[valid_mask]
        else:
            pred_flat = predictions.view(-1)
            target_flat = target_returns.view(-1)

        ss_res = torch.sum((target_flat - pred_flat) ** 2)
        ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return r2.item()

    def update_step(self, r2_heldout: Optional[float] = None):
        """
        Update warmup/active state based on current step and R^2.

        Call this once per training step.
        """
        self.current_step += 1

        if self.warmup_active and self.current_step >= self.warmup_steps:
            if r2_heldout is not None and r2_heldout >= self.r2_threshold:
                self.warmup_active = False
                self.tbicc_active = True
                LOGGER.info(
                    "Warmup complete at step %d, R^2=%.3f. T-BICC activated.",
                    self.current_step,
                    r2_heldout,
                )
            else:
                LOGGER.info(
                    "Warmup step %d/%d, R^2=%.3f. T-BICC still gated.",
                    self.current_step,
                    self.warmup_steps,
                    r2_heldout if r2_heldout is not None else float("nan"),
                )

    def is_tbicc_active(self) -> bool:
        return self.tbicc_active

    @torch.no_grad()
    def get_truncated_value(
        self,
        hidden_states: torch.Tensor,
        delta: int,
    ) -> torch.Tensor:
        """
        Get truncated value estimate V_phi(y_{<=t+Delta}).

        Args:
            hidden_states: (batch, seq_len, input_dim) full sequence hidden states
            delta: truncation window

        Returns:
            truncated_values: (batch, seq_len) value at t+Delta
        """
        batch_size, seq_len, _ = hidden_states.shape
        truncated_values = []

        for t in range(seq_len):
            t_end = min(t + delta, seq_len - 1)
            h_trunc = hidden_states[:, t_end, :]  # (batch, input_dim)
            v_trunc = self.value_head(h_trunc).squeeze(-1)  # (batch,)
            truncated_values.append(v_trunc)

        return torch.stack(truncated_values, dim=1)  # (batch, seq_len)
