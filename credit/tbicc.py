"""
T-BICC (Truncated Batch-Implicit Counterfactual Credit).

Computes token-level causal scores by leveraging the natural multi-rollout
structure of GRPO as implicit contrastive experiments.

Reference:
  - Section 4.2 (T-BICC)
  - Eq. (1) T-BICC score
  - Eq. (2) Dual-channel prefix matching
  - Proposition 1 (Mediator Bound)
  - Proposition 2 (Soft-Prefix Confounding Bound)
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

# Divergence kernel type
KernelType = Literal["l1", "wasserstein", "binary"]


def compute_choice_divergence(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_m: int = 5,
) -> torch.Tensor:
    """
    Compute choice divergence D_choice between rollouts.

    Groups tokens by top-m probability clustering.
    D_choice(t, k, j) = 1 - 1[cluster(y_t^(k)) == cluster(y_t^(j))]

    Args:
        logits: (K, seq_len, vocab_size) pre-softmax logits
        token_ids: (K, seq_len) token IDs
        top_m: number of top tokens to consider for clustering

    Returns:
        divergence: (K, K, seq_len) binary divergence matrix
            1 if clusters differ, 0 if same
    """
    K, seq_len, V = logits.shape

    # Get top-m tokens for each position
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        _, top_tokens = torch.topk(probs, top_m, dim=-1)  # (K, seq_len, top_m)

    # Cluster: assign each token to its "cluster" = set of top-m tokens
    # Simplification: use the top-1 token as cluster ID
    # (full top-m set comparison is expensive, use top-1 as approximation)
    cluster_ids = top_tokens[:, :, 0]  # (K, seq_len)

    # Compute pairwise divergence
    divergence = torch.zeros((K, K, seq_len), device=logits.device)
    for k in range(K):
        for j in range(K):
            if k == j:
                continue
            # 1 if clusters differ
            divergence[k, j] = (cluster_ids[k] != cluster_ids[j]).float()

    return divergence


def dual_channel_similarity(
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    window_size: int = 32,
    eta: float = 0.5,
) -> torch.Tensor:
    """
    Compute dual-channel prefix similarity (lexical + semantic).

    sim = eta * Jaccard_n(y_(t-L:t)^(k), y_(t-L:t)^(j))
          + (1-eta) * cos(e_t^(k), e_t^(j))

    where e_t^(k) = MeanPool(h_(t-L:t)^(k))

    Args:
        token_ids: (K, seq_len) token IDs
        hidden_states: (K, seq_len, hidden_dim)
        attention_mask: (K, seq_len) 1 for valid, 0 for pad
        window_size: L, local window size
        eta: weight for lexical channel

    Returns:
        similarity: (K, K, seq_len) similarity matrix in [0, 1]
    """
    K, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    similarity = torch.zeros((K, K, seq_len), device=device)

    for t in range(seq_len):
        if not attention_mask[:, t].any():
            continue

        # Local window [t-window_size+1, t]
        t_start = max(0, t - window_size + 1)
        t_end = t + 1

        # Lexical channel: Jaccard similarity over local window
        lexical_sim = torch.zeros((K, K), device=device)
        for k in range(K):
            for j in range(K):
                if k == j:
                    lexical_sim[k, j] = 1.0
                    continue
                # Jaccard = |intersection| / |union|
                set_k = set(token_ids[k, t_start:t_end].tolist())
                set_j = set(token_ids[j, t_start:t_end].tolist())
                intersection = len(set_k & set_j)
                union = len(set_k | set_j)
                lexical_sim[k, j] = intersection / max(union, 1)

        # Semantic channel: cosine similarity of mean-pooled hidden states
        semantic_sim = torch.zeros((K, K), device=device)
        for k in range(K):
            # Mean pool over window
            mask_k = attention_mask[k, t_start:t_end].bool()
            if mask_k.sum() == 0:
                continue
            e_k = hidden_states[k, t_start:t_end][mask_k].mean(dim=0)  # (hidden_dim,)

            for j in range(K):
                if k == j:
                    semantic_sim[k, j] = 1.0
                    continue
                mask_j = attention_mask[j, t_start:t_end].bool()
                if mask_j.sum() == 0:
                    continue
                e_j = hidden_states[j, t_start:t_end][mask_j].mean(dim=0)
                cos_sim = F.cosine_similarity(
                    e_k.unsqueeze(0), e_j.unsqueeze(0)
                ).item()
                semantic_sim[k, j] = (cos_sim + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]

        # Combine
        similarity[:, :, t] = eta * lexical_sim + (1 - eta) * semantic_sim

    return similarity


def compute_tbicc_scores(
    truncated_values: torch.Tensor,
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
    delta: int = 8,
    s0: float = 0.5,
    eta: float = 0.5,
    lambda_ema: float = 0.9,
    k_min: int = 2,
    kernel: KernelType = "l1",
    previous_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute T-BICC token-level causal scores.

    Reference: Eq. (1) in paper.

    Args:
        truncated_values: (K, seq_len) V_phi(y_<=t+Delta) for each rollout
        token_ids: (K, seq_len) token IDs
        hidden_states: (K, seq_len, hidden_dim)
        attention_mask: (K, seq_len)
        logits: (K, seq_len, vocab_size)
        delta: truncation window
        s0: similarity threshold (prefix confounding bound)
        eta: dual-channel weight
        lambda_ema: EMA smoothing factor
        k_min: minimum divergent rollouts (fallback to entropy if insufficient)
        kernel: divergence kernel type
        previous_scores: (K, seq_len) previous T-BICC scores for EMA

    Returns:
        tbicc_scores: (K, seq_len) normalized T-BICC scores in [0, 1]
        diagnostics: dict with intermediate computations
    """
    K, seq_len = truncated_values.shape
    device = truncated_values.device

    # 1. Compute dual-channel prefix similarity
    sim = dual_channel_similarity(
        token_ids, hidden_states, attention_mask,
        window_size=32, eta=eta,
    )  # (K, K, seq_len)

    # 2. Compute choice divergence
    div = compute_choice_divergence(logits, token_ids, top_m=5)  # (K, K, seq_len)

    # 3. Compute prefix-similarity weight alpha
    # alpha_(t,k,j) = sim * D_choice  (only count pairs with divergent choices)
    alpha = sim * div  # (K, K, seq_len)

    # Apply similarity threshold s0 (soft thresholding)
    # Pairs with sim < s0 contribute zero (Proposition 2)
    alpha = alpha * (sim >= s0).float()

    # 4. Compute divergence kernel d(V_phi^(k), V_phi^(j))
    if kernel == "l1":
        # |V^(k) - V^(j)| for each position
        d_values = torch.zeros((K, K, seq_len), device=device)
        for k in range(K):
            for j in range(K):
                if k == j:
                    continue
                d_values[k, j] = torch.abs(
                    truncated_values[k] - truncated_values[j]
                )
    elif kernel == "binary":
        # Binary reward: d = |1(r^(k)=correct) - 1(r^(j)=correct)|
        # This requires reward labels, which are passed separately
        # For now, use L1 on values as proxy
        d_values = torch.zeros((K, K, seq_len), device=device)
        for k in range(K):
            for j in range(K):
                if k == j:
                    continue
                d_values[k, j] = torch.abs(
                    truncated_values[k] - truncated_values[j]
                )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # 5. Compute T-BICC scores (Eq. 1)
    tbicc_raw = torch.zeros((K, seq_len), device=device)
    num_divergent = torch.zeros((K, seq_len), device=device)

    for k in range(K):
        for t in range(seq_len):
            if not attention_mask[k, t]:
                continue

            # Weighted average over j != k
            weights = alpha[k, :, t]  # (K,)
            divergences = d_values[k, :, t]  # (K,)

            valid_mask = (torch.arange(K, device=device) != k) & (weights > 0)
            num_valid = valid_mask.sum().item()

            if num_valid < k_min:
                # Fallback: use entropy gating only
                tbicc_raw[k, t] = 0.0
                num_divergent[k, t] = 0
                continue

            weighted_sum = (weights[valid_mask] * divergences[valid_mask]).sum()
            weight_sum = weights[valid_mask].sum() + 1e-8

            tbicc_raw[k, t] = weighted_sum / weight_sum
            num_divergent[k, t] = num_valid

    # 6. EMA smoothing
    if previous_scores is not None:
        if previous_scores.shape == tbicc_raw.shape:
            tbicc_raw = lambda_ema * previous_scores + (1 - lambda_ema) * tbicc_raw
        # else: skip EMA when shapes differ (different sequence lengths)

    # 7. Normalize to [0, 1] within batch
    tbicc_min = tbicc_raw[attention_mask.bool()].min()
    tbicc_max = tbicc_raw[attention_mask.bool()].max()
    tbicc_norm = (tbicc_raw - tbicc_min) / (tbicc_max - tbicc_min + 1e-8)
    tbicc_norm = tbicc_norm * attention_mask.float()

    diagnostics = {
        "tbicc_raw": tbicc_raw,
        "num_divergent": num_divergent,
        "similarity": sim,
        "divergence": div,
        "alpha": alpha,
    }

    return tbicc_norm, diagnostics


def compute_entropy_signal(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level entropy signal H_t.

    H_t = -sum_v pi_theta(v | x, y_<t) * log pi_theta(v | x, y_<t)

    Reference: Eq. (3) in paper.

    Args:
        logits: (K, seq_len, vocab_size)
        attention_mask: (K, seq_len)

    Returns:
        entropy: (K, seq_len) normalized entropy in [0, 1]
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (K, seq_len)

        # Normalize by max entropy (= log(vocab_size))
        max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float))
        entropy = entropy / (max_entropy + 1e-8)

        entropy = entropy * attention_mask.float()

    return entropy


def multiplicative_dual_gate(
    entropy: torch.Tensor,
    tbicc_scores: torch.Tensor,
    attention_mask: torch.Tensor,
    beta: float = 0.6,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Compute joint credit weight via multiplicative dual gating.

    w_t^(k) = beta * H_t^(k) * C_t^(k) + (1-beta) * C_t^(k) + gamma

    Reference: Eq. (4) in paper, Theorem 3.

    Args:
        entropy: (K, seq_len) normalized entropy H_t in [0, 1]
        tbicc_scores: (K, seq_len) normalized T-BICC scores in [0, 1]
        attention_mask: (K, seq_len)
        beta: multiplicative weight
        gamma: additive floor

    Returns:
        weights: (K, seq_len) credit weights
    """
    # Normalize within batch (again, to ensure [0,1])
    # Entropy already normalized by max entropy
    # T-BICC already normalized to [0,1]

    H = entropy
    C = tbicc_scores

    # Multiplicative dual gate
    weights = beta * H * C + (1 - beta) * C + gamma

    # Zero out padding positions
    weights = weights * attention_mask.float()

    return weights
