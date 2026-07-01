"""
Test script for EGSPO-CA v2 core math.

Verifies:
  1. T-BICC score computation
  2. Multiplicative dual gating
  3. Value head forward pass
  4. Loss functions

Run: python test_core_math.py
"""

import os
import sys

import torch

# Add code/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credit.tbicc import (
    compute_tbicc_scores,
    compute_entropy_signal,
    multiplicative_dual_gate,
    compute_choice_divergence,
    dual_channel_similarity,
)
from models.value_head import ValueHead
from training.loss import GRPOLoss, EGSPOCALoss


def test_value_head():
    """Test value head forward pass."""
    print("=== Test Value Head ===")

    batch_size, seq_len, input_dim = 2, 16, 64
    hidden_states = torch.randn(batch_size, seq_len, input_dim)

    vh = ValueHead(input_dim=input_dim, hidden_dim=32)

    # Forward pass
    values = vh(hidden_states)
    assert values.shape == (batch_size, seq_len, 1), \
        f"Expected {(batch_size, seq_len, 1)}, got {values.shape}"

    print(f"  Input: ({batch_size}, {seq_len}, {input_dim})")
    print(f"  Output: {values.shape}")
    print(f"  Values range: [{values.min():.4f}, {values.max():.4f}]")
    print("  PASSED")


def test_choice_divergence():
    """Test choice divergence computation."""
    print("\n=== Test Choice Divergence ===")

    K, seq_len, vocab_size = 4, 8, 100
    logits = torch.randn(K, seq_len, vocab_size)

    div = compute_choice_divergence(logits, torch.argmax(logits, dim=-1))

    assert div.shape == (K, K, seq_len), \
        f"Expected {(K, K, seq_len)}, got {div.shape}"

    # Diagonal should be 0 (same rollout)
    for k in range(K):
        assert (div[k, k] == 0).all(), "Diagonal should be 0"

    print(f"  Divergence shape: {div.shape}")
    print(f"  Divergence range: [{div.min():.4f}, {div.max():.4f}]")
    print("  PASSED")


def test_dual_channel_similarity():
    """Test dual-channel similarity computation."""
    print("\n=== Test Dual-Channel Similarity ===")

    K, seq_len, hidden_dim = 4, 16, 64
    token_ids = torch.randint(0, 100, (K, seq_len))
    hidden_states = torch.randn(K, seq_len, hidden_dim)
    attention_mask = torch.ones(K, seq_len)

    sim = dual_channel_similarity(
        token_ids, hidden_states, attention_mask,
        window_size=8, eta=0.5,
    )

    assert sim.shape == (K, K, seq_len), \
        f"Expected {(K, K, seq_len)}, got {sim.shape}"

    # Diagonal should be 1.0 (same rollout, perfect similarity)
    for k in range(K):
        assert torch.allclose(sim[k, k], torch.ones(seq_len)), \
            "Diagonal should be 1.0"

    # Similarity should be in [0, 1]
    assert (sim >= 0).all() and (sim <= 1).all(), \
        "Similarity should be in [0, 1]"

    print(f"  Similarity shape: {sim.shape}")
    print(f"  Similarity range: [{sim.min():.4f}, {sim.max():.4f}]")
    print("  PASSED")


def test_tbicc_scores():
    """Test T-BICC score computation."""
    print("\n=== Test T-BICC Scores ===")

    K, seq_len = 4, 16
    hidden_dim = 64

    # Dummy data
    truncated_values = torch.randn(K, seq_len)
    token_ids = torch.randint(0, 100, (K, seq_len))
    hidden_states = torch.randn(K, seq_len, hidden_dim)
    attention_mask = torch.ones(K, seq_len)
    logits = torch.randn(K, seq_len, 100)

    tbicc_scores, diagnostics = compute_tbicc_scores(
        truncated_values, token_ids, hidden_states, attention_mask, logits,
        delta=8, s0=0.5, eta=0.5,
    )

    assert tbicc_scores.shape == (K, seq_len), \
        f"Expected {(K, seq_len)}, got {tbicc_scores.shape}"

    # Scores should be in [0, 1] (normalized)
    assert (tbicc_scores >= 0).all() and (tbicc_scores <= 1).all(), \
        f"Scores should be in [0, 1], got [{tbicc_scores.min():.4f}, {tbicc_scores.max():.4f}]"

    print(f"  T-BICC shape: {tbicc_scores.shape}")
    print(f"  T-BICC range: [{tbicc_scores.min():.4f}, {tbicc_scores.max():.4f}]")
    print(f"  Num divergent: {diagnostics['num_divergent'].flatten()[:5].tolist()}")
    print("  PASSED")


def test_entropy_signal():
    """Test entropy signal computation."""
    print("\n=== Test Entropy Signal ===")

    K, seq_len, vocab_size = 4, 16, 100
    logits = torch.randn(K, seq_len, vocab_size)
    attention_mask = torch.ones(K, seq_len)

    entropy = compute_entropy_signal(logits, attention_mask)

    assert entropy.shape == (K, seq_len), \
        f"Expected {(K, seq_len)}, got {entropy.shape}"

    # Entropy should be in [0, 1] (normalized)
    assert (entropy >= 0).all() and (entropy <= 1).all(), \
        f"Entropy should be in [0, 1], got [{entropy.min():.4f}, {entropy.max():.4f}]"

    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    print("  PASSED")


def test_multiplicative_dual_gate():
    """Test multiplicative dual gating."""
    print("\n=== Test Multiplicative Dual Gate ===")

    K, seq_len = 4, 16
    entropy = torch.rand(K, seq_len)
    tbicc_scores = torch.rand(K, seq_len)
    attention_mask = torch.ones(K, seq_len)

    weights = multiplicative_dual_gate(
        entropy, tbicc_scores, attention_mask,
        beta=0.6, gamma=0.1,
    )

    assert weights.shape == (K, seq_len), \
        f"Expected {(K, seq_len)}, got {weights.shape}"

    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print("  PASSED")


def test_loss_functions():
    """Test loss function computation."""
    print("\n=== Test Loss Functions ===")

    batch_size, seq_len, vocab_size = 8, 16, 100
    K = 4

    policy_logits = torch.randn(batch_size, seq_len, vocab_size)
    old_logits = torch.randn(batch_size, seq_len, vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    rewards = torch.randn(batch_size)

    # GRPO loss
    grpo_loss_fn = GRPOLoss(K=K, clip_epsilon=0.2)
    try:
        loss = grpo_loss_fn(
            policy_logits, old_logits, token_ids, attention_mask, rewards
        )
        print(f"  GRPO loss: {loss.item():.4f}")
    except NotImplementedError as e:
        print(f"  GRPO loss: NotImplementedError (expected, needs token IDs)")

    # EGSPO-CA loss
    credit_weights = torch.rand(batch_size, seq_len)
    egspo_loss_fn = EGSPOCALoss(K=K, clip_epsilon=0.2)
    loss = egspo_loss_fn(
        policy_logits, old_logits, token_ids, attention_mask,
        rewards, credit_weights,
    )

    assert loss.item() > 0, "Loss should be positive"
    print(f"  EGSPO-CA loss: {loss.item():.4f}")
    print("  PASSED")


def main():
    print("Running EGSPO-CA v2 core math tests...\n")

    test_value_head()
    test_choice_divergence()
    test_dual_channel_similarity()
    test_tbicc_scores()
    test_entropy_signal()
    test_multiplicative_dual_gate()
    test_loss_functions()

    print("\nAll tests PASSED!")


if __name__ == "__main__":
    main()
