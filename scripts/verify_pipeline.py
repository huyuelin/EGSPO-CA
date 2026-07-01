#!/usr/bin/env python3
"""
Minimal pipeline verification for EGSPO-CA v2 reproduction.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/verify_pipeline.py
"""

import argparse
import logging
import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.value_head import ValueHead, ValueHeadTrainer
from credit.tbicc import compute_tbicc_scores, compute_entropy_signal, multiplicative_dual_gate
from training.reward_utils import exact_match_reward

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def test_cuda():
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"[OK] CUDA available. Device: {torch.cuda.get_device_name(0)}")
    print(f"     GPUs: {torch.cuda.device_count()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def test_model_load(model_name):
    print(f"\n--- Loading model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"[OK] Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Test generation
    test_prompt = "What is 2 + 2? Answer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=2,
            return_dict_in_generate=True,
        )
    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"[OK] Generation works. Sample: {generated[:100]}...")
    return model, tokenizer


def test_value_head(model, hidden_dim):
    print("\n--- Testing Value Head ---")
    value_head = ValueHead(input_dim=hidden_dim, hidden_dim=1024).to(model.device)
    dummy_hidden = torch.randn(4, 32, hidden_dim, dtype=torch.bfloat16, device=model.device)
    attn_mask = torch.ones(4, 32, dtype=torch.long, device=model.device)
    values = value_head(dummy_hidden)
    print(f"[OK] Value head forward: shape {values.shape}")
    return value_head


def test_tbicc(value_head, model, tokenizer):
    print("\n--- Testing T-BICC ---")
    test_prompt = "Solve: x + 3 = 7. What is x?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    attn_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits

    # Value head forward: (1, seq_len, 1) -> squeeze to (1, seq_len)
    values = value_head(hidden_states).squeeze(-1)

    # Expand to simulate K=2 rollouts for T-BICC
    # T-BICC expects (K, seq_len) with K >= 2
    values = values.repeat(2, 1)  # (2, seq_len)
    token_ids = inputs["input_ids"].repeat(2, 1)  # (2, seq_len)
    hidden_states = hidden_states.repeat(2, 1, 1)  # (2, seq_len, dim)
    attn_mask = attn_mask.repeat(2, 1)  # (2, seq_len)
    logits = logits.repeat(2, 1, 1)  # (2, seq_len, vocab)

    # T-BICC
    tbicc_scores, diag = compute_tbicc_scores(
        truncated_values=values,
        token_ids=token_ids,
        hidden_states=hidden_states,
        attention_mask=attn_mask,
        logits=logits,
        delta=8,
        s0=0.5,
        eta=0.5,
        lambda_ema=0.9,
        k_min=2,
        previous_scores=None,
    )
    print(f"[OK] T-BICC scores computed. Shape: {tbicc_scores.shape}, Mean: {tbicc_scores.mean():.4f}")

    # Entropy signal
    entropy = compute_entropy_signal(logits, attn_mask)
    print(f"[OK] Entropy signal: {entropy.mean():.4f}")

    # Gate
    weights = multiplicative_dual_gate(entropy, tbicc_scores, attn_mask, beta=0.6, gamma=0.1)
    print(f"[OK] Multiplicative gate: {weights.mean():.4f}")


def test_reward():
    print("\n--- Testing Reward Computation ---")

    # Test exact match with boxed
    correct_text = "Therefore, the answer is \\boxed{42}."
    reward = exact_match_reward(correct_text, "42")
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"[OK] Boxed exact match: {reward}")

    # Test wrong answer
    wrong_text = "Therefore, the answer is \\boxed{43}."
    reward = exact_match_reward(wrong_text, "42")
    assert reward == 0.0, f"Expected 0.0, got {reward}"
    print(f"[OK] Wrong answer: {reward}")

    # Test no boxed
    no_boxed = "I don't know the answer."
    reward = exact_match_reward(no_boxed, "42")
    assert reward == 0.0, f"Expected 0.0, got {reward}"
    print(f"[OK] No boxed answer: {reward}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    args = parser.parse_args()

    test_cuda()
    model, tokenizer = test_model_load(args.model)

    hidden_dim = model.config.hidden_size
    test_value_head(model, hidden_dim)

    value_head = ValueHead(input_dim=hidden_dim, hidden_dim=1024).to(model.device)
    test_tbicc(value_head, model, tokenizer)

    test_reward()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
