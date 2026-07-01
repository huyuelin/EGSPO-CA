# EGSPO-CA: Divergence-Based Implicit Credit Assignment with Formal Causal Guarantees for Policy Optimization

<p align="center">
  <img src="assets/framework.png" width="95%" alt="EGSPO-CA Framework Overview"/>
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the EGSPO-CA framework. The K=8 rollouts sampled by GRPO form <em>implicit divergence experiments</em>: pairs sharing a prefix but diverging at token <em>t</em> expose a reward gap that serves as a contrastive counterfactual signal. T-BICC converts this into a token causal score via dual-channel prefix matching and a truncated value head; a multiplicative dual gate fuses it with entropy to reweight the PPO objective.
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/AAAI-2027-blue.svg" alt="AAAI 2027"></a>
  <a href="https://github.com/huyuelin/EGSPO-CA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

## Authors

**Yuelin Hu<sup>1</sup>, Zhenbo Yu<sup>1</sup>, Zhengxue Cheng<sup>1</sup>, Wei Liu<sup>2</sup>, Li Song<sup>1</sup>**

<sup>1</sup> Shanghai Jiao Tong University &nbsp;&nbsp; <sup>2</sup> Shanghai Maritime University

{huyuelin51717221, yuzhenbo, zxcheng, songli}@sjtu.edu.cn

---

## Abstract

Token-level credit assignment remains a fundamental bottleneck in reinforcement learning for large language models. Existing approaches occupy two extremes: correlational methods (entropy, CMI) are efficient but conflate uncertainty with causal importance, while explicit counterfactual methods achieve principled attribution at prohibitive cost. We propose **EGSPO-CA**, which leverages the natural multi-rollout structure of Group Relative Policy Optimization as an implicit contrastive signal for token-level causal credit estimation. We formalize this through a structural causal model (SCM) that decomposes the estimation error into quantifiable mediator bias and soft-prefix confounding, with both terms provably bounded (verified via Lean 4). Our method, **Truncated-BICC (T-BICC)**, with adaptive truncation and dual-channel prefix matching, achieves Spearman ρ = 0.724 against ground-truth causal effects at only **+1.6% wall-clock overhead**. Experiments across four models and six benchmarks demonstrate gains of **+4.5** average points over GRPO and **+0.9** over CAPO.

## Highlights

- **Formal Causal Guarantees**: First provably-bounded token credit signal in RLHF, verified via Lean 4 with zero `sorry`.
- **Negligible Overhead**: +1.6% wall-clock cost vs. +38–45% for comparable methods (CAPO, CF Credit).
- **Five Orthogonal Ground Truths**: Eliminates circular self-validation with mask-and-forward, Monte-Carlo Shapley, LLM-as-Judge, Lean-grounded steps, and CMI capacity.
- **Causal-Lean Bridge**: First use of formal theorem proving as online reward shaping for RLHF.

## Main Results

| Method | MATH-500 | AIME'24 | AIME'25 | GSM8K | OlympiadBench | Minerva | **Avg** |
|:-------|:--------:|:-------:|:-------:|:-----:|:-------------:|:-------:|:-------:|
| GRPO | 72.4 | 22.8 | 16.4 | 88.1 | 38.2 | 32.1 | 45.0 |
| DAPO | 73.9 | 24.7 | 18.1 | 89.0 | 39.6 | 33.4 | 46.5 |
| CAPO | 75.8 | 27.9 | 21.3 | 89.8 | 41.2 | 35.3 | 48.6 |
| CF Credit | 75.6 | 27.5 | 20.9 | 89.9 | 41.0 | 35.1 | 48.3 |
| **EGSPO-CA v2** | **76.7** | **29.2** | **22.6** | **90.2** | **42.1** | **36.0** | **49.5** |
| **+ CLB** | **77.4** | **30.1** | **23.4** | **90.3** | **42.8** | **36.7** | **50.1** |

> Qwen2.5-7B-Instruct, trained on 7.5K NuminaMath-CoT problems, mean ± 95% CI over 5 seeds. All methods share the same verl-based infrastructure.

## Credit Quality (COA, Spearman ρ)

| Method | GT-A (Mask) | GT-B (Shapley) | GT-C (Judge) | GT-D (Lean) | GT-E (CMI) | **Avg** |
|:-------|:-----------:|:--------------:|:------------:|:-----------:|:----------:|:-------:|
| Entropy | 0.28 | 0.24 | 0.31 | 0.19 | 0.53 | 0.31 |
| CAPO | 0.64 | 0.62 | 0.63 | 0.53 | 0.55 | 0.59 |
| HAPO | 0.52 | 0.49 | 0.55 | 0.41 | **0.81** | 0.56 |
| **EGSPO-CA v2** | **0.72** | **0.74** | **0.71** | **0.66** | 0.62 | **0.69** |

## Method

<p align="center">
  <img src="assets/framework.png" width="90%"/>
</p>

The core idea is that GRPO's K=8 rollouts per prompt naturally create **implicit divergence experiments**. When two rollouts share a prefix $Y_{<t}$ but diverge at token $t$, the resulting reward gap isolates the causal effect of that token choice. We formalize this via a structural causal model and prove:

$$|\mathbb{E}[\tilde{\tau}_t] - \tau_t^{PSCE}| \le 2R_{\max} \cdot \gamma^\Delta + \epsilon_V + L_{\text{sem}} \cdot (1 - s_0)$$

where both error terms (mediator bias and prefix confounding) are bounded and verified in Lean 4.

**Key components:**
1. **T-BICC** — Truncated Batch-Implicit Credit with adaptive Δ window
2. **Dual-Channel Matching** — Lexical (Jaccard) + Semantic (cosine of hidden states) prefix similarity
3. **Multiplicative Dual Gate** — Combines entropy *capacity* with causal *magnitude*
4. **Causal-Lean Bridge** — Optional formal verification as reward shaping

## Repository Structure

```
├── configs/
│   └── base.yaml                 # All hyperparameters (Table 1 in paper)
├── credit/
│   └── tbicc.py                  # T-BICC, dual-channel matching, dual gate
├── models/
│   └── value_head.py             # Lightweight value head (4M params)
├── training/
│   ├── loss.py                   # GRPO + EGSPO-CA weighted PPO loss
│   ├── grpo_trainer.py           # GRPO baseline
│   ├── egspo_ca_trainer.py       # EGSPO-CA v2 trainer
│   ├── adaptive_delta.py         # Adaptive truncation (Theorem 4)
│   └── reward_utils.py           # Reward computation
├── baselines/
│   └── baseline_trainers.py      # 10 baseline reimplementations
├── eval/
│   └── eval_credit.py            # Five ground-truth COA evaluation
├── lean/                          # Lean 4 formal proofs (~550 lines)
├── data/
│   └── numina_loader.py          # Data loading utilities
├── scripts/                       # Training & evaluation scripts
├── ablations/                     # Sanity-check ablation configs
├── test_core_math.py             # Unit tests for core algorithms
└── requirements.txt
```

## Getting Started

### Installation

```bash
git clone https://github.com/huyuelin/EGSPO-CA.git
cd EGSPO-CA
pip install -r requirements.txt
```

### Verify Core Implementation

```bash
python test_core_math.py
```

### Training

```bash
# GRPO baseline
bash scripts/run_phase1.sh grpo

# EGSPO-CA v2
bash scripts/run_phase1.sh egspo_ca

# 8-GPU distributed training
bash scripts/launch_8gpu_gsm8k.sh
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 8 | Rollouts per prompt |
| Δ | 8 (adaptive) | Truncation window |
| s₀ | 0.5 | Prefix similarity threshold |
| η | 0.5 | Lexical/semantic balance |
| β | 0.6 | Multiplicative gate weight |
| γ | 0.1 | Gradient floor |
| lr | 5×10⁻⁶ | Learning rate |

## Formal Verification

All propositions are machine-verified in Lean 4 with zero `sorry` and zero `axiom` beyond Mathlib:

| Statement | Prover | Status |
|-----------|--------|--------|
| Prop 1 (Mediator Bound) | Seed-Prover 1.5 | ✓ Verified |
| Prop 2 (Prefix Confounding) | Seed-Prover 1.5 | ✓ Verified |
| Prop 3 (Continuous Extension) | Seed-Prover 1.5 | ✓ Verified |
| Thm 2 (Gate Optimality) | BFS-Prover-V2 | ✓ Verified |
| Thm 3 (Truncation Existence) | Seed-Prover 1.5 | ✓ Verified |

## Citation

```bibtex
@inproceedings{hu2027egspo-ca,
  title={EGSPO-CA: Divergence-Based Implicit Credit Assignment with Formal Causal Guarantees for Policy Optimization},
  author={Hu, Yuelin and Yu, Zhenbo and Cheng, Zhengxue and Liu, Wei and Song, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

## Acknowledgements

This work was supported by Shanghai Jiao Tong University and Shanghai Maritime University. We thank the developers of [verl](https://github.com/volcengine/verl), [Seed-Prover](https://github.com/bytedance/SeedProver), and [Mathlib4](https://github.com/leanprover-community/mathlib4).

## License

This project is released under the [MIT License](LICENSE).
