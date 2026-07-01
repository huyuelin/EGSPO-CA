# EGSPO-CA: Divergence-Based Implicit Credit Assignment with Formal Causal Guarantees for Policy Optimization

Official implementation for the AAAI 2027 submission: *EGSPO-CA: Divergence-Based Implicit Credit Assignment with Formal Causal Guarantees for Policy Optimization*.

## Overview

EGSPO-CA leverages the natural multi-rollout structure of Group Relative Policy Optimization (GRPO) as an implicit contrastive signal for token-level causal credit estimation. The method introduces:

- **Truncated-BICC (T-BICC)**: Adaptive truncation with dual-channel (lexical + semantic) prefix matching, achieving Spearman $\rho = 0.724$ against ground-truth causal effects at +1.6% wall-clock overhead.
- **Formal Causal Framework**: A structural causal model (SCM + PSCE decomposition) with machine-verified propositions (Lean 4) bounding the approximation error.
- **Multiplicative Dual Gating**: Combines entropy capacity with causal magnitude for optimal token reweighting.
- **Causal-Lean Bridge (CLB)**: Optional formal theorem proving as online reward shaping.

## Key Results

| Method | MATH-500 | AIME24 | AIME25 | GSM8K | Olympiad | Minerva | Avg |
|--------|----------|--------|--------|-------|----------|---------|-----|
| GRPO | 72.4 | 22.8 | 16.4 | 88.1 | 38.2 | 32.1 | 45.0 |
| CAPO | 75.8 | 27.9 | 21.3 | 89.8 | 41.2 | 35.3 | 48.6 |
| **EGSPO-CA v2** | **76.7** | **29.2** | **22.6** | **90.2** | **42.1** | **36.0** | **49.5** |
| **+ CLB** | **77.4** | **30.1** | **23.4** | **90.3** | **42.8** | **36.7** | **50.1** |

Results on Qwen2.5-7B-Instruct, mean over 5 seeds. EGSPO-CA v2 outperforms 10 baselines including CAPO (+0.9 avg), CF Credit, HAPO, DelTA, SPO, and TEMPO.

## Project Structure

```
code/
├── configs/
│   └── base.yaml                # Hyperparameters from paper (K=8, Delta=8, etc.)
├── credit/
│   └── tbicc.py                 # T-BICC core + dual-channel matching + dual gate
├── models/
│   └── value_head.py            # Lightweight value head (4M params, confidence-gated)
├── training/
│   ├── loss.py                  # GRPO + EGSPO-CA weighted PPO loss
│   ├── grpo_trainer.py          # GRPO baseline trainer
│   ├── egspo_ca_trainer.py      # EGSPO-CA v2 trainer
│   ├── adaptive_delta.py        # Adaptive truncation (Theorem 4)
│   └── reward_utils.py          # Reward computation utilities
├── baselines/
│   ├── baseline_trainers.py     # Reimplementations of 10 baselines
│   └── run_baseline.py          # Baseline launcher
├── eval/
│   └── eval_credit.py           # Five GT evaluation (COA metrics)
├── data/
│   └── numina_loader.py         # NuminaMath-CoT + benchmark loaders
├── scripts/
│   ├── run_phase1.sh            # Training launch script
│   ├── run_baselines_parallel.sh
│   ├── launch_8gpu_gsm8k.sh
│   ├── verify_pipeline.py       # Pipeline verification
│   └── download_data.py         # Data preparation
├── ablations/                   # Sanity-check ablation configs
├── lean/                        # Lean 4 formal proofs (~550 lines)
├── results/                     # Evaluation outputs
├── test_core_math.py            # Unit tests for T-BICC math
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, Transformers, DeepSpeed, vLLM.

## Quick Start

```bash
# Verify core math
python test_core_math.py

# Train GRPO baseline
bash scripts/run_phase1.sh grpo

# Train EGSPO-CA v2
bash scripts/run_phase1.sh egspo_ca
```

## Training Configuration

Default hyperparameters (from paper):
- K=8 rollouts per prompt
- Truncation window: Delta=8 (adaptive)
- Prefix similarity threshold: s_0=0.5
- Dual-channel weight: eta=0.5
- Gate parameters: beta=0.6, gamma=0.1
- Learning rate: 5e-6
- Clip epsilon: 0.2

## Formal Verification

The formal proofs in `lean/` verify Propositions 1-3 and Theorems 2-3 using Lean 4 with Seed-Prover 1.5 and BFS-Prover-V2. All proofs compile with zero `sorry` and zero `axiom` beyond Mathlib.

## Citation

```bibtex
@inproceedings{hu2027egspo-ca,
  title={EGSPO-CA: Divergence-Based Implicit Credit Assignment with Formal Causal Guarantees for Policy Optimization},
  author={Hu, Yuelin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

## License

This project is released under the MIT License.
