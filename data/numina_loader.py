"""
Data loader for NuminaMath-CoT and evaluation benchmarks.

Supports:
  - NuminaMath-CoT (7.5K problems for training)
  - MATH-500, AIME 2024, AIME 2025, GSM8K, OlympiadBench, Minerva-Math

Expected format (JSONL):
  {"problem": "...", "solution": "...", "answer": "...", "type": "..."}

Reference:
  - Section 5.1 (Setup -> Training data)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

LOGGER = logging.getLogger(__name__)

# Benchmark files (expected names)
BENCHMARK_FILES = {
    "MATH-500": "math500.jsonl",
    "AIME24": "aime24.jsonl",
    "AIME25": "aime25.jsonl",
    "GSM8K": "gsm8k.jsonl",
    "OlympiadBench": "olympiadbench.jsonl",
    "Minerva-Math": "minerva_math.jsonl",
}


class MathProblemDataset(Dataset):
    """Dataset for math problems with optional CoT solutions."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 256,
        max_response_length: int = 512,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.split = split

        # Load data
        self.data = self._load_data(data_path)
        LOGGER.info("Loaded %d problems from %s", len(self.data), data_path)

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        path = Path(data_path)
        assert path.exists(), f"Data file not found: {data_path}"

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Format prompt
        prompt = self._format_prompt(item["problem"])

        return {
            "problem": item["problem"],
            "prompt": prompt,
            "solution": item.get("solution", ""),
            "answer": item.get("answer", ""),
            "type": item.get("type", "unknown"),
        }

    def _format_prompt(self, problem: str) -> str:
        """Format problem as chat prompt for instruction-tuned models.

        Appends a boxed-answer instruction so that the model outputs
        its final answer in \\boxed{} format, enabling reliable
        automatic evaluation via exact match on the boxed content.
        """
        # Append boxed-format instruction for reliable answer extraction
        augmented = (
            f"{problem}\n\n"
            f"Please solve step by step. Put your final answer in \\boxed{{}}."
        )
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": augmented}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = augmented

        return prompt

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for training."""
        prompts = [item["prompt"] for item in batch]
        problems = [item["problem"] for item in batch]
        solutions = [item["solution"] for item in batch]
        answers = [item["answer"] for item in batch]

        return {
            "prompts": prompts,
            "problems": problems,
            "solutions": solutions,
            "answers": answers,
        }


def load_numina_cot(
    data_dir: str,
    split: str = "train",
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load NuminaMath-CoT data.

    Expected file: {data_dir}/numina_cot.jsonl

    Args:
        data_dir: directory containing data files
        split: "train" or "test"
        num_samples: max number of samples (for debugging)

    Returns:
        list of problem dicts
    """
    file_path = Path(data_dir) / "numina_cot.jsonl"
    assert file_path.exists(), f"NuminaMath-CoT file not found: {file_path}"

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    LOGGER.info("Loaded %d problems from NuminaMath-CoT", len(data))
    return data


def load_benchmark(
    data_dir: str,
    benchmark: str,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load evaluation benchmark.

    Args:
        data_dir: directory containing benchmark files
        benchmark: benchmark name (must be in BENCHMARK_FILES)
        num_samples: max number of samples

    Returns:
        list of problem dicts
    """
    assert benchmark in BENCHMARK_FILES, \
        f"Unknown benchmark: {benchmark}. Valid: {list(BENCHMARK_FILES.keys())}"

    file_path = Path(data_dir) / BENCHMARK_FILES[benchmark]
    assert file_path.exists(), f"Benchmark file not found: {file_path}"

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    LOGGER.info("Loaded %d problems from %s", len(data), benchmark)
    return data


def compute_discounted_returns(
    rewards: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted Monte-Carlo returns for value head training.

    G_t = r_t + gamma * r_(t+1) + ... + gamma^(T-t) * r_T

    For RLHF, reward is only at the end (r_T = outcome reward).
    So G_t = gamma^(T-t) * r_T for all t.

    Args:
        rewards: (batch, seq_len) or (batch,) outcome rewards
        gamma: discount factor

    Returns:
        returns: (batch, seq_len) discounted returns
    """
    if rewards.dim() == 1:
        # Outcome reward only: broadcast to all positions
        batch_size, seq_len = rewards.shape[0], 1  # Unknown seq_len
        # This case needs seq_len passed separately
        raise ValueError("Outcome rewards need seq_len. Use compute_outcome_returns.")

    # rewards is (batch, seq_len) with non-zero only at last valid position
    returns = torch.zeros_like(rewards)
    for t in range(rewards.shape[1]):
        # G_t = sum_{s=t}^T gamma^(s-t) * r_s
        # For outcome reward at position T: G_t = gamma^(T-t) * r_T
        future_rewards = rewards[:, t:]
        discounts = torch.pow(gamma, torch.arange(future_rewards.shape[1], device=rewards.device).float())
        returns[:, t] = (future_rewards * discounts.unsqueeze(0)).sum(dim=1)

    return returns


def compute_outcome_returns(
    outcome_rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted returns when reward is only at the end.

    Args:
        outcome_rewards: (batch,) outcome rewards (0 or 1 for correctness)
        attention_mask: (batch, seq_len)
        gamma: discount factor

    Returns:
        returns: (batch, seq_len) discounted returns
    """
    batch_size, seq_len = attention_mask.shape
    returns = torch.zeros((batch_size, seq_len), device=outcome_rewards.device, dtype=outcome_rewards.dtype)

    for b in range(batch_size):
        # Find last valid position
        valid_positions = attention_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
        T = valid_positions[-1].item()

        # G_t = gamma^(T-t) * r_T for t <= T
        r_T = outcome_rewards[b]
        for t in range(T + 1):
            returns[b, t] = (gamma ** (T - t)) * r_T

    return returns
