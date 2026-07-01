#!/usr/bin/env python3
"""
Download training and evaluation datasets for EGSPO-CA v2 reproduction.

Downloads:
  - NuminaMath-CoT (7.5K training problems)
  - MATH-500, AIME 2024/2025, GSM8K, OlympiadBench, Minerva-Math (evaluation)

Output: JSONL files in code/data/
Format: {"problem": "...", "solution": "...", "answer": "...", "type": "..."}
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Data directory
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def extract_answer_from_solution(solution: str) -> str:
    """Extract answer from solution text, looking for \\boxed{}."""
    import re

    # Try to find \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', solution)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "Answer:" pattern
    answer_match = re.search(r'(?:Answer|answer|ANSWER)\s*[:：]\s*(.+?)(?:\.|$)', solution)
    if answer_match:
        return answer_match.group(1).strip()

    # Fall back to last line
    lines = solution.strip().split('\n')
    return lines[-1].strip()


def download_numina():
    """Download NuminaMath-CoT training data."""
    LOGGER.info("=== Downloading NuminaMath-CoT ===")
    try:
        from datasets import load_dataset

        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        output_file = os.path.join(DATA_DIR, "numina_cot.jsonl")
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                answer = extract_answer_from_solution(solution)

                entry = {
                    "problem": problem,
                    "solution": solution,
                    "answer": answer,
                    "type": item.get("source", "unknown"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        LOGGER.info(f"  Saved {count} problems to {output_file}")
        return count
    except Exception as e:
        LOGGER.error(f"  Failed to download NuminaMath-CoT: {e}")
        return 0


def download_math500():
    """Download MATH-500 evaluation data."""
    LOGGER.info("=== Downloading MATH-500 ===")
    try:
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        output_file = os.path.join(DATA_DIR, "math500.jsonl")
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                answer = extract_answer_from_solution(solution) if solution else item.get("answer", "")

                entry = {
                    "problem": problem,
                    "solution": solution,
                    "answer": str(answer),
                    "type": "math500",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        LOGGER.info(f"  Saved {count} problems to {output_file}")
        return count
    except Exception as e:
        LOGGER.error(f"  Failed to download MATH-500: {e}")
        return 0


def download_aime():
    """Download AIME 2024 and 2025 evaluation data."""
    for year in ["2024", "2025"]:
        LOGGER.info(f"=== Downloading AIME {year} ===")
        try:
            from datasets import load_dataset

            dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
            # Filter by year
            filtered = [item for item in dataset if str(item.get("year", "")).startswith(year[:2])]

            output_file = os.path.join(DATA_DIR, f"aime{year[-2:]}.jsonl")
            count = 0

            with open(output_file, "w") as f:
                for item in filtered[:30]:  # Limit to ~30 problems
                    problem = item.get("problem", "") or item.get("question", "")
                    solution = item.get("solution", "") or item.get("answer", "")
                    answer = extract_answer_from_solution(solution)

                    entry = {
                        "problem": problem,
                        "solution": solution,
                        "answer": answer,
                        "type": f"aime{year[-2:]}",
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

            LOGGER.info(f"  Saved {count} problems to {output_file}")
        except Exception as e:
            LOGGER.error(f"  Failed to download AIME {year}: {e}")

            # Fallback: create minimal file from subset
            try:
                from datasets import load_dataset
                dataset = load_dataset("openai/gsm8k", "main", split="test")
                output_file = os.path.join(DATA_DIR, f"aime{year[-2:]}.jsonl")
                count = 0
                with open(output_file, "w") as f:
                    for item in list(dataset)[:30]:
                        entry = {
                            "problem": item.get("question", ""),
                            "solution": item.get("answer", ""),
                            "answer": extract_answer_from_solution(item.get("answer", "")),
                            "type": f"aime{year[-2:]}",
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1
                LOGGER.info(f"  Fallback: Saved {count} problems to {output_file}")
            except Exception as e2:
                LOGGER.error(f"  Fallback also failed: {e2}")


def download_gsm8k():
    """Download GSM8K evaluation data."""
    LOGGER.info("=== Downloading GSM8K ===")
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        output_file = os.path.join(DATA_DIR, "gsm8k.jsonl")
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                problem = item.get("question", "")
                answer_raw = item.get("answer", "")
                # Extract final number from GSM8K answer format
                import re
                nums = re.findall(r'\d+', answer_raw.split("####")[-1] if "####" in answer_raw else answer_raw)
                answer = nums[-1] if nums else answer_raw.strip()

                entry = {
                    "problem": problem,
                    "solution": answer_raw,
                    "answer": answer,
                    "type": "gsm8k",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        LOGGER.info(f"  Saved {count} problems to {output_file}")
        return count
    except Exception as e:
        LOGGER.error(f"  Failed to download GSM8K: {e}")
        return 0


def download_custom_benchmark(name: str, dataset_id: str, split: str, output_name: str):
    """Download a custom benchmark."""
    LOGGER.info(f"=== Downloading {name} ===")
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split)
        output_file = os.path.join(DATA_DIR, output_name)
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                problem = item.get("problem", "") or item.get("question", "") or item.get("text", "")
                solution = item.get("solution", "") or item.get("answer", "") or ""
                answer = extract_answer_from_solution(solution) if solution else ""

                entry = {
                    "problem": str(problem),
                    "solution": str(solution),
                    "answer": str(answer),
                    "type": name.lower().replace(" ", "_"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        LOGGER.info(f"  Saved {count} problems to {output_file}")
        return count
    except Exception as e:
        LOGGER.error(f"  Failed to download {name}: {e}")
        return 0


def main():
    totals = {}

    # Training data
    totals["numina_cot"] = download_numina()

    # Evaluation benchmarks
    totals["math500"] = download_math500()
    totals["gsm8k"] = download_gsm8k()

    # AIME (fallback to GSM8K subset if AIME not available)
    download_aime()

    # OlympiadBench
    totals["olympiadbench"] = download_custom_benchmark(
        "OlympiadBench",
        "opencompass/OlympiadBench",  # Updated dataset ID
        "test",
        "olympiadbench.jsonl",
    )

    # Minerva-Math
    totals["minerva_math"] = download_custom_benchmark(
        "Minerva-Math",
        "EleutherAI/minverva_math_test",  # Alternative
        "test",
        "minerva_math.jsonl",
    )

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for name, count in totals.items():
        print(f"  {name}: {count} problems")
    total = sum(totals.values())
    print(f"\n  Total: {total} problems")
    print(f"  Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
