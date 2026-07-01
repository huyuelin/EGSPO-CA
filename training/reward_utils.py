"""
Reward utilities for math problem evaluation.

Implements exact match with answer extraction from \boxed{} format
(standard for MATH-type datasets).
"""

import re
from typing import List, Optional


def _normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    s = answer.strip().lower()
    # Remove trailing period
    s = s.rstrip(".")
    # Remove spaces
    s = s.replace(" ", "")
    # Remove leading/trailing whitespace
    s = s.strip()
    return s


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} expression from generated text.

    Args:
        text: generated completion text

    Returns:
        extracted answer string, or None if not found
    """
    # Pattern: \boxed{<content>} - handles nested braces
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last occurrence
    last_match = matches[-1]
    start_pos = last_match.end()

    # Parse matching braces
    depth = 1
    end_pos = start_pos
    while end_pos < len(text) and depth > 0:
        if text[end_pos] == "{":
            depth += 1
        elif text[end_pos] == "}":
            depth -= 1
        end_pos += 1

    if depth == 0:
        answer = text[start_pos : end_pos - 1]
        return _normalize_answer(answer)

    return None


def extract_answer_from_text(text: str) -> str:
    """
    Extract final answer from generated math solution text.

    Priority:
      1. \boxed{...} format
      2. "Answer: ..." format
      3. Last line as fallback

    Returns:
        extracted answer string
    """
    # Try boxed format
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Try "Answer: <value>" pattern
    answer_patterns = [
        r"Answer:\s*(.+?)(?:\n|$)",
        r"answer\s*(?:is|=)\s*(.+?)(?:\n|$)",
        r"Therefore,?\s*(?:the\s*)?(?:answer|result)\s*(?:is|=)\s*(.+?)(?:\n|$)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_answer(match.group(1))

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return _normalize_answer(lines[-1])

    return ""


def exact_match_reward(
    generated_text: str,
    ground_truth_answer: str,
) -> float:
    """
    Compute exact match reward for math problem.

    1.0 if generated answer matches ground truth, 0.0 otherwise.

    Args:
        generated_text: model's generated response
        ground_truth_answer: ground truth answer string

    Returns:
        reward: 1.0 (correct) or 0.0 (incorrect)
    """
    extracted = extract_answer_from_text(generated_text)
    gt = _normalize_answer(ground_truth_answer)

    if not extracted:
        return 0.0

    return 1.0 if extracted == gt else 0.0


def batch_exact_match_reward(
    generated_texts: List[str],
    ground_truth_answers: List[str],
) -> List[float]:
    """
    Batch version of exact_match_reward.

    Returns:
        list of rewards (0.0 or 1.0)
    """
    assert len(generated_texts) == len(ground_truth_answers), \
        f"Mismatch: {len(generated_texts)} generated vs {len(ground_truth_answers)} answers"

    rewards = []
    for gen, gt in zip(generated_texts, ground_truth_answers):
        rewards.append(exact_match_reward(gen, gt))

    return rewards
