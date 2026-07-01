"""
Adaptive Truncation Delta (Theorem 4)

Implements per-token optimal truncation window selection:
  Delta*(t) = argmin_Delta [ 2*R_max*gamma^Delta + epsilon_V(Delta; t) ]

The mediator bias term 2*R_max*gamma^Delta decreases exponentially with Delta,
while the value estimation error epsilon_V(Delta) increases with Delta.
The optimal Delta balances these two sources of error.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

LOGGER = logging.getLogger(__name__)


class AdaptiveDeltaTracker:
    """Tracks value head estimation error for adaptive Delta selection."""

    def __init__(
        self,
        delta_candidates: List[int] = None,
        gamma: float = 0.99,
        R_max: float = 1.0,
        ema_alpha: float = 0.95,
        min_samples: int = 10,
    ):
        self.delta_candidates = delta_candidates or [0, 4, 8, 16, 32]
        self.gamma = gamma
        self.R_max = R_max
        self.ema_alpha = ema_alpha
        self.min_samples = min_samples

        # Track empirical value error per delta
        self._error_sum: Dict[int, float] = {d: 0.0 for d in self.delta_candidates}
        self._error_count: Dict[int, int] = {d: 0 for d in self.delta_candidates}
        self._ema_error: Dict[int, float] = {d: float("inf") for d in self.delta_candidates}
        self._sample_count: int = 0

    def update(
        self,
        delta_errors: Dict[int, float],
    ) -> None:
        """Update error statistics with new observations."""
        self._sample_count += 1
        for delta, error in delta_errors.items():
            if delta not in self._error_sum:
                continue
            self._error_sum[delta] += error
            self._error_count[delta] += 1
            # EMA update
            if self._ema_error[delta] == float("inf"):
                self._ema_error[delta] = error
            else:
                self._ema_error[delta] = (
                    self.ema_alpha * self._ema_error[delta]
                    + (1 - self.ema_alpha) * error
                )

    def get_empirical_error(self, delta: int) -> float:
        """Get mean empirical error for a delta value."""
        if self._error_count[delta] < self.min_samples:
            return float("inf")
        return self._error_sum[delta] / self._error_count[delta]

    def get_ema_error(self, delta: int) -> float:
        """Get EMA-smoothed error for a delta value."""
        return self._ema_error.get(delta, float("inf"))

    def compute_optimal_delta(
        self,
        use_ema: bool = True,
    ) -> int:
        """Compute the optimal Delta using Theorem 4 objective."""
        if self._sample_count < self.min_samples:
            LOGGER.debug(
                "Not enough samples for adaptive Delta (%d/%d), using default Delta=8",
                self._sample_count, self.min_samples,
            )
            return 8  # Fallback default

        best_delta = 8  # Default fallback
        best_score = float("inf")

        for delta in self.delta_candidates:
            # Mediator bias term: 2 * R_max * gamma^Delta
            bias = 2.0 * self.R_max * (self.gamma ** delta)

            # Value estimation error term
            if use_ema:
                var_term = self._ema_error[delta]
            else:
                var_term = self.get_empirical_error(delta)

            if abs(var_term - float("inf")) < 1e-6:
                continue  # Skip deltas without enough data

            score = bias + var_term

            if score < best_score:
                best_score = score
                best_delta = delta

        LOGGER.debug(
            "Adaptive Delta: selected %d (score=%.6f, samples=%d)",
            best_delta, best_score, self._sample_count,
        )
        return best_delta

    def compute_per_token_delta(
        self,
        value_errors: torch.Tensor,  # (batch, seq_len) per-delta errors
        delta_indices: Dict[int, int],  # delta -> column index mapping
    ) -> torch.Tensor:
        """Compute per-token optimal Delta based on value errors.

        Args:
            value_errors: (batch, seq_len, num_deltas) tensor of errors per delta
            delta_indices: mapping from delta value to column index

        Returns:
            optimal_deltas: (batch, seq_len) tensor of optimal delta values
        """
        batch_size, seq_len, _ = value_errors.shape
        device = value_errors.device

        # Compute scores for each delta
        scores = torch.full(
            (batch_size, seq_len, len(self.delta_candidates)),
            float("inf"),
            device=device,
        )

        for idx, delta in enumerate(self.delta_candidates):
            if delta not in delta_indices:
                continue
            col = delta_indices[delta]
            bias = 2.0 * self.R_max * (self.gamma ** delta)
            scores[:, :, idx] = bias + value_errors[:, :, col]

        # Select delta with minimum score
        best_indices = scores.argmin(dim=-1)  # (batch, seq_len)
        optimal_deltas = torch.tensor(
            [self.delta_candidates[i] for i in best_indices.flatten().tolist()],
            device=device,
        ).reshape(batch_size, seq_len)

        return optimal_deltas

    def is_ready(self) -> bool:
        """Check if enough data collected for adaptive Delta."""
        return self._sample_count >= self.min_samples


def estimate_value_error(
    value_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    target_returns: torch.Tensor,
    delta: int,
    gamma: float = 0.99,
) -> float:
    """Estimate value head error for a given Delta.

    Computes MSE between truncated value estimates and
    discounted Monte-Carlo returns, measuring how well the
    value head predicts future reward at each truncation point.

    Args:
        value_head: The value head MLP
        hidden_states: (batch, seq_len, hidden_dim)
        target_returns: (batch, seq_len) discounted returns
        delta: truncation window size
        gamma: discount factor

    Returns:
        Mean squared error for this delta
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    # Collect truncated hidden states
    truncated_hidden = []
    truncated_targets = []

    for t in range(seq_len):
        t_end = min(t + delta, seq_len - 1)
        if t_end < seq_len:
            truncated_hidden.append(hidden_states[:, t_end, :])
            truncated_targets.append(target_returns[:, t])

    if not truncated_hidden:
        return float("inf")

    truncated_hidden = torch.stack(truncated_hidden, dim=1)  # (B, T', H)
    truncated_targets = torch.stack(truncated_targets, dim=1)  # (B, T')

    with torch.no_grad():
        predicted = value_head(truncated_hidden).squeeze(-1)
        mse = torch.nn.functional.mse_loss(predicted, truncated_targets)

    return mse.item()
