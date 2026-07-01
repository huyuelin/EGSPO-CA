#!/usr/bin/env python3
"""
Create a minimal test dataset for pipeline verification.

Generates a small JSONL file with simple arithmetic problems
for quick verification of the training pipeline.
"""

import json
import os

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(OUTPUT_DIR, "data", "test_problems.jsonl")

PROBLEMS = [
    {
        "problem": "What is 15 + 27?",
        "solution": "15 + 27 = 42",
        "answer": "42",
        "type": "arithmetic",
    },
    {
        "problem": "What is 100 - 37?",
        "solution": "100 - 37 = 63",
        "answer": "63",
        "type": "arithmetic",
    },
    {
        "problem": "What is 8 * 7?",
        "solution": "8 * 7 = 56",
        "answer": "56",
        "type": "arithmetic",
    },
    {
        "problem": "What is 144 / 12?",
        "solution": "144 / 12 = 12",
        "answer": "12",
        "type": "arithmetic",
    },
    {
        "problem": "Solve for x: 3x + 5 = 20.",
        "solution": "3x + 5 = 20 => 3x = 15 => x = 5",
        "answer": "5",
        "type": "algebra",
    },
    {
        "problem": "What is the square root of 144?",
        "solution": "sqrt(144) = 12",
        "answer": "12",
        "type": "arithmetic",
    },
    {
        "problem": "What is 2^10?",
        "solution": "2^10 = 1024",
        "answer": "1024",
        "type": "arithmetic",
    },
    {
        "problem": "If a rectangle has length 5 and width 3, what is its area?",
        "solution": "Area = length * width = 5 * 3 = 15",
        "answer": "15",
        "type": "geometry",
    },
    {
        "problem": "What is the sum of the first 5 positive integers?",
        "solution": "1 + 2 + 3 + 4 + 5 = 15",
        "answer": "15",
        "type": "arithmetic",
    },
    {
        "problem": "What is 25% of 200?",
        "solution": "25% of 200 = 0.25 * 200 = 50",
        "answer": "50",
        "type": "arithmetic",
    },
    {
        "problem": "If f(x) = 2x + 1, what is f(3)?",
        "solution": "f(3) = 2*3 + 1 = 7",
        "answer": "7",
        "type": "algebra",
    },
    {
        "problem": "What is the circumference of a circle with radius 7? Use pi = 3.14.",
        "solution": "C = 2 * pi * r = 2 * 3.14 * 7 = 43.96",
        "answer": "43.96",
        "type": "geometry",
    },
    {
        "problem": "What is the median of [3, 1, 4, 1, 5, 9, 2]?",
        "solution": "Sorted: [1, 1, 2, 3, 4, 5, 9]. Median = 3",
        "answer": "3",
        "type": "statistics",
    },
    {
        "problem": "What is 50% of 100 plus 25% of 80?",
        "solution": "50% of 100 = 50. 25% of 80 = 20. Sum = 70",
        "answer": "70",
        "type": "arithmetic",
    },
    {
        "problem": "If a triangle has base 6 and height 4, what is its area?",
        "solution": "Area = (1/2) * base * height = (1/2) * 6 * 4 = 12",
        "answer": "12",
        "type": "geometry",
    },
    {
        "problem": "What is the LCM of 4 and 6?",
        "solution": "Multiples of 4: 4, 8, 12, ... Multiples of 6: 6, 12, ... LCM = 12",
        "answer": "12",
        "type": "number_theory",
    },
    {
        "problem": "What is the GCD of 24 and 36?",
        "solution": "24 = 2^3 * 3, 36 = 2^2 * 3^2. GCD = 2^2 * 3 = 12",
        "answer": "12",
        "type": "number_theory",
    },
    {
        "problem": "Solve for y: 2y - 7 = 11.",
        "solution": "2y - 7 = 11 => 2y = 18 => y = 9",
        "answer": "9",
        "type": "algebra",
    },
    {
        "problem": "What is the value of 6 factorial (6!)?",
        "solution": "6! = 6 * 5 * 4 * 3 * 2 * 1 = 720",
        "answer": "720",
        "type": "arithmetic",
    },
    {
        "problem": "What is the probability of rolling a 6 on a fair six-sided die?",
        "solution": "1 favorable outcome / 6 total outcomes = 1/6",
        "answer": "1/6",
        "type": "probability",
    },
]


def main():
    with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
        for prob in PROBLEMS:
            f.write(json.dumps(prob) + "\n")
    print(f"Created test dataset with {len(PROBLEMS)} problems at {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
