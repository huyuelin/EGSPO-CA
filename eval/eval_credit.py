"""
Credit evaluation framework with five orthogonal ground truths.

Reference: Table 3 in the EGSPO-CA v2 paper.

GT-A: Mask-and-Forward — Interventional: mask token t, measure ΔP_answer
GT-B: Monte-Carlo Shapley — Game-theoretic: marginal contributions over subsets
GT-C: LLM-as-Judge — Expert: GPT-4o/Claude 3.5 critical token intersection
GT-D: Lean-Grounded — Verifier: Lean 4 tactic-to-token mapping
GT-E: CMI Capacity — Information-theoretic: I(Y_t; R | Y_<t, X)

Evaluation metric: Spearman ρ between credit weights and GT labels per sequence,
averaged over all evaluated problems.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Metrics
# ═══════════════════════════════════════════════════════════════

def spearman_rho(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Spearman rank correlation between two tensors.
    
    Args:
        x: (seq_len,) credit weights or scores
        y: (seq_len,) ground truth scores
        mask: (seq_len,) optional valid token mask
    
    Returns:
        rho: correlation coefficient in [-1, 1]
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    
    # Filter to valid positions
    x_valid = x[mask].float()
    y_valid = y[mask].float()
    
    if len(x_valid) < 3:
        return 0.0
    
    # Rank transform
    x_rank = x_valid.argsort().argsort().float()
    y_rank = y_valid.argsort().argsort().float()
    
    # Pearson correlation on ranks
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    
    cov = (x_centered * y_centered).sum()
    denom = (x_centered.pow(2).sum() * y_centered.pow(2).sum()).sqrt()
    
    if denom < 1e-8:
        return 0.0
    
    return (cov / denom).item()


def compute_answer_logprob(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_tokens: List[int],
) -> float:
    """
    Compute log-probability of answer tokens after the prompt.
    
    Uses a single forward pass to extract probabilities.
    """
    with torch.no_grad():
        outputs = model(
            token_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (1, seq_len, vocab)
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Sum log probs of answer tokens (last positions)
    # Note: simplified — assumes answer is at the end of the sequence
    total_lp = 0.0
    for t in answer_tokens:
        probs = log_probs[:, -1, :]  # last position only
        if t < probs.shape[-1]:
            total_lp += probs[0, t].item()
    
    return total_lp


# ═══════════════════════════════════════════════════════════════
# GT-A: Mask-and-Forward
# ═══════════════════════════════════════════════════════════════

def compute_gt_mask_and_forward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generated_ids: torch.Tensor,
    answer_str: str,
    max_tokens: int = 512,
) -> torch.Tensor:
    """
    GT-A: Mask each token, re-run forward, measure ΔP_answer.
    
    For each token position t in the generated sequence:
    1. Mask token at position t (replace with pad/sep)
    2. Re-run forward pass through model
    3. Compute P(answer | masked_sequence)
    4. Credit[t] = P(answer | full_sequence) - P(answer | masked_at_t)
    
    Args:
        model: Policy model (eval mode)
        tokenizer: Tokenizer
        prompt: Original prompt string
        generated_ids: (seq_len,) token IDs of the full generation
        answer_str: Correct answer for computing answer probability
        max_tokens: Max tokens to evaluate (for efficiency)
    
    Returns:
        gt_credit: (seq_len,) ground truth credit scores
    """
    device = next(model.parameters()).device
    seq_len = generated_ids.shape[0]
    
    # Limit number of positions to evaluate
    eval_positions = min(seq_len, max_tokens)
    
    # Tokenize answer for probability computation
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    
    # Full sequence answer probability
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_input_ids = torch.cat([
        torch.tensor(prompt_ids, device=device),
        generated_ids,
    ]).unsqueeze(0)
    full_attn = torch.ones_like(full_input_ids)
    
    full_lp = compute_answer_logprob(model, full_input_ids, full_attn, answer_ids)
    
    # Per-token masking
    credit = torch.zeros(seq_len, device=device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    for t in range(eval_positions):
        # Create masked version: replace token at position t with pad
        masked_ids = full_input_ids.clone()
        # The generated tokens start after the prompt
        gen_start = len(prompt_ids)
        masked_pos = gen_start + t
        if masked_pos < masked_ids.shape[1]:
            masked_ids[0, masked_pos] = pad_token_id
        
        masked_lp = compute_answer_logprob(model, masked_ids, full_attn, answer_ids)
        
        # Credit = reduction in answer log prob when token is masked
        credit[t] = full_lp - masked_lp
    
    return credit


# ═══════════════════════════════════════════════════════════════
# GT-B: Monte-Carlo Shapley
# ═══════════════════════════════════════════════════════════════

def compute_gt_mc_shapley(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generated_ids: torch.Tensor,
    answer_str: str,
    num_subsets: int = 200,
    max_tokens: int = 256,
) -> torch.Tensor:
    """
    GT-B: Monte-Carlo Shapley values.
    
    For each token, estimate Shapley value by averaging marginal
    contribution over random subsets:
    
    phi_t = E_{S⊆N\\{t}} [ v(S ∪ {t}) - v(S) ]
    
    where v(S) = P(correct | tokens in S), estimated via answer probability.
    
    Args:
        num_subsets: Number of random subsets to sample
        max_tokens: Max tokens to evaluate
    
    Returns:
        gt_credit: (seq_len,) Shapley value estimates
    """
    device = next(model.parameters()).device
    seq_len = generated_ids.shape[0]
    eval_positions = min(seq_len, max_tokens)
    
    # Answer tokens
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    
    # Base sequence
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    gen_start = len(prompt_ids)
    full_input_ids = torch.cat([
        torch.tensor(prompt_ids, device=device),
        generated_ids,
    ]).unsqueeze(0)
    full_attn = torch.ones_like(full_input_ids)
    
    # Accumulate marginal contributions
    contributions = torch.zeros(seq_len, device=device)
    counts = torch.zeros(seq_len, device=device)
    
    for _ in range(num_subsets):
        # Sample a random subset size
        subset_size = torch.randint(1, eval_positions, (1,)).item()
        subset = torch.randperm(eval_positions, device=device)[:subset_size]
        
        # Value of subset: P(answer | tokens in subset)
        masked_ids = full_input_ids.clone()
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Mask out tokens NOT in subset
        for t in range(eval_positions):
            if t not in subset:
                masked_ids[0, gen_start + t] = pad_token_id
        
        v_subset = compute_answer_logprob(model, masked_ids, full_attn, answer_ids)
        
        # For each token NOT in subset, compute marginal contribution
        for t in range(eval_positions):
            if t not in subset:
                # Add token t to subset
                masked_with_t = masked_ids.clone()
                masked_with_t[0, gen_start + t] = full_input_ids[0, gen_start + t]
                v_with_t = compute_answer_logprob(model, masked_with_t, full_attn, answer_ids)
                
                contributions[t] += (v_with_t - v_subset) if v_subset != 0 else 0
                counts[t] += 1
    
    # Average
    credit = torch.zeros(seq_len, device=device)
    for t in range(eval_positions):
        if counts[t] > 0:
            credit[t] = contributions[t] / counts[t]
    
    return credit


# ═══════════════════════════════════════════════════════════════
# GT-C: LLM-as-Judge
# ═══════════════════════════════════════════════════════════════

def compute_gt_llm_judge(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generated_text: str,
    answer_str: str,
    api_client: Optional[object] = None,
) -> torch.Tensor:
    """
    GT-C: LLM-as-Judge critical token annotation.
    
    Sends the reasoning chain to GPT-4o and Claude 3.5,
    asks them to identify critical reasoning steps,
    and maps those back to token positions.
    
    For offline mode (no API), uses a heuristic:
    - Tokens near logical operators (if, then, therefore, because)
    - Tokens near numerical values matching the answer
    
    Args:
        api_client: Optional resilient_llm_client for API calls
    
    Returns:
        gt_mask: (seq_len,) binary critical token mask
    """
    generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    seq_len = len(generated_ids)
    
    if api_client is not None:
        # Real LLM-as-Judge via API
        return _llm_judge_api(tokenizer, prompt, generated_text, answer_str, api_client)
    else:
        # Heuristic fallback
        return _llm_judge_heuristic(tokenizer, generated_ids, generated_text, answer_str)


def _llm_judge_heuristic(
    tokenizer: PreTrainedTokenizer,
    generated_ids: List[int],
    generated_text: str,
    answer_str: str,
) -> torch.Tensor:
    """Heuristic critical token identification without API."""
    seq_len = len(generated_ids)
    mask = torch.zeros(seq_len)
    
    # Critical keywords (case-insensitive)
    critical_words = [
        "therefore", "thus", "hence", "because", "since",
        "finally", "conclude", "answer", "result", "so ",
        "compute", "calculate", "solve", "derive",
    ]
    
    text_lower = generated_text.lower()
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    
    # Mark tokens near critical keywords
    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([generated_ids[i]]).lower().strip()
        
        # Check if token is part of a critical word
        context_start = max(0, i - 5)
        context_end = min(seq_len, i + 5)
        context = tokenizer.decode(generated_ids[context_start:context_end]).lower()
        
        is_critical = any(kw in context for kw in critical_words)
        
        # Also mark tokens near the answer
        if answer_str.lower() in context:
            is_critical = True
        
        # Numerical tokens are often critical
        if any(c.isdigit() for c in token_str):
            is_critical = True
        
        if is_critical:
            # Mark window around critical token
            for j in range(max(0, i - 2), min(seq_len, i + 3)):
                mask[j] = 1.0
    
    return mask


def _llm_judge_api(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generated_text: str,
    answer_str: str,
    api_client,
) -> torch.Tensor:
    """
    Real LLM-as-Judge via API (GPT-4o + Claude 3.5 intersection).
    
    Returns:
        gt_mask: (seq_len,) binary mask (1.0 for critical tokens)
    """
    # This would call the actual APIs
    # For now, fall back to heuristic
    generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    LOGGER.warning("LLM-Judge API not fully implemented, using heuristic fallback")
    return _llm_judge_heuristic(tokenizer, generated_ids, generated_text, answer_str)


# ═══════════════════════════════════════════════════════════════
# GT-D: Lean-Grounded Critical Steps
# ═══════════════════════════════════════════════════════════════

def compute_gt_lean_grounded(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generated_text: str,
    lean_tactic_map: Optional[Dict[str, List[int]]] = None,
) -> torch.Tensor:
    """
    GT-D: Lean-Grounded Critical Steps.
    
    Maps Lean 4 tactic invocations back to natural language tokens.
    Requires Lean 4 proof data for the problem.
    
    For offline mode, uses a structured-step heuristic:
    identifies "Step 1:", "Step 2:", etc. as critical boundaries.
    
    Args:
        lean_tactic_map: {tactic_name: [token_positions...]}
            Maps Lean tactics to positions in the natural language text
    
    Returns:
        gt_mask: (seq_len,) binary mask
    """
    generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    seq_len = len(generated_ids)
    mask = torch.zeros(seq_len)
    
    if lean_tactic_map is not None:
        # Real Lean 4 tactic-to-token mapping
        for tactic, positions in lean_tactic_map.items():
            for pos in positions:
                if pos < seq_len:
                    mask[pos] = 1.0
    else:
        # Heuristic: identify step boundaries
        _lean_step_heuristic(tokenizer, generated_ids, generated_text, mask)
    
    return mask


def _lean_step_heuristic(
    tokenizer: PreTrainedTokenizer,
    generated_ids: List[int],
    generated_text: str,
    mask: torch.Tensor,
):
    """Heuristic step boundary detection."""
    seq_len = len(generated_ids)
    text_lower = generated_text.lower()
    
    # Step markers
    step_patterns = [
        "step", "first", "second", "third", "next", "finally",
        "therefore", "hence", "thus", "we have", "we get",
        "solve", "compute", "calculate", "rearrange", "substitute",
        "theorem", "lemma", "by definition",
    ]
    
    # Math operators indicate reasoning steps
    math_ops = ["=", "+", "-", "×", "÷", "∫", "∑", "lim", "→", "⇒"]
    
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    
    for i in range(seq_len):
        token = tokenizer.decode([generated_ids[i]]).strip()
        
        # Step boundary: newlines, colons, etc.
        if token in ["\n", ".", ":"]:
            # Mark next few tokens as step boundary
            for j in range(i, min(seq_len, i + 3)):
                mask[j] = 1.0
        
        # Math operators
        if any(op in token for op in math_ops):
            mask[i] = 1.0


# ═══════════════════════════════════════════════════════════════
# GT-E: CMI Capacity
# ═══════════════════════════════════════════════════════════════

def compute_gt_cmi_capacity(
    hidden_states: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    num_bins: int = 10,
) -> torch.Tensor:
    """
    GT-E: CMI Capacity.
    
    Compute approximate I(Y_t; R | Y_<t, X) using discretized
    hidden state distributions.
    
    For each position t:
    - Estimate H(Y_t) from the rollout group
    - Estimate H(Y_t | R) conditioning on reward bins
    - CMI[t] = H(Y_t) - H(Y_t | R)
    
    Where Y_t is the token distribution at position t.
    
    Args:
        hidden_states: (batch, seq_len, hidden_dim) from rollout group
        rewards: (batch,) outcome rewards
        attention_mask: (batch, seq_len)
        num_bins: Number of reward bins for conditioning
    
    Returns:
        cmi: (seq_len,) CMI per position
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device
    
    cmi = torch.zeros(seq_len, device=device)
    
    # Discretize rewards
    if rewards.max() - rewards.min() > 1e-8:
        reward_bins = ((rewards - rewards.min()) / 
                       (rewards.max() - rewards.min() + 1e-8) * num_bins).long().clamp(0, num_bins - 1)
    else:
        reward_bins = torch.zeros_like(rewards).long()
    
    for t in range(seq_len):
        # Only positions with valid attention
        valid_mask = attention_mask[:, t].bool()
        if valid_mask.sum() < 2:
            continue
        
        # Hidden states at position t (valid only)
        h_t = hidden_states[:, t, :][valid_mask]  # (N_valid, hidden_dim)
        r_bins_t = reward_bins[valid_mask]  # (N_valid,)
        
        N = h_t.shape[0]
        if N < 4:
            continue
        
        # Entropy of Y_t (via pair-wise distances in hidden space)
        # H(Y_t) ≈ log of mean pairwise distance
        h_centered = h_t - h_t.mean(dim=0, keepdim=True)
        
        # Unconditional entropy approximation
        h_norm = F.normalize(h_t, p=2, dim=-1)
        similarities = h_norm @ h_norm.T  # (N, N)
        diag_mask = ~torch.eye(N, device=device, dtype=torch.bool)
        
        if similarities[diag_mask].numel() > 0:
            mean_sim = similarities[diag_mask].mean().abs()
            h_uncond = -mean_sim * math.log(max(mean_sim.item(), 1e-8))
        else:
            continue
        
        # Conditional entropy: average entropy within each reward bin
        h_cond = 0.0
        total_weight = 0
        
        for b in range(num_bins):
            bin_mask = (r_bins_t == b)
            bin_count = bin_mask.sum().item()
            
            if bin_count < 2:
                continue
            
            bin_h = h_norm[bin_mask]  # (bin_count, hidden_dim)
            bin_sim = bin_h @ bin_h.T
            bin_diag = ~torch.eye(bin_count, device=device, dtype=torch.bool)
            
            if bin_sim[bin_diag].numel() > 0:
                bin_mean_sim = bin_sim[bin_diag].mean().abs()
                bin_entropy = -bin_mean_sim * math.log(max(bin_mean_sim.item(), 1e-8))
                h_cond += bin_entropy * bin_count
                total_weight += bin_count
        
        if total_weight > 0:
            h_cond = h_cond / total_weight
            cmi[t] = max(h_uncond - h_cond, 0.0)
    
    return cmi


# ═══════════════════════════════════════════════════════════════
# Unified Evaluation Runner
# ═══════════════════════════════════════════════════════════════

def evaluate_credit_quality(
    credit_weights: torch.Tensor,  # (batch, seq_len)
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    generated_texts: List[str],
    generated_ids_list: List[torch.Tensor],
    answers: List[str],
    hidden_states: Optional[torch.Tensor] = None,
    rewards: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    ground_truths: Optional[List[str]] = None,  # which GTs to compute
) -> Dict[str, float]:
    """
    Evaluate credit quality across specified ground truths.
    
    Args:
        credit_weights: (batch, seq_len) credit weights from the method
        model: Policy model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        generated_texts: List of generated text strings
        generated_ids_list: List of (seq_len,) token ID tensors
        answers: List of correct answer strings
        hidden_states: Optional (batch, seq_len, hidden_dim)
        rewards: Optional (batch,) outcome rewards
        attention_mask: Optional (batch, seq_len)
        ground_truths: List of GT methods to compute, e.g. ["A", "B", "E"]
    
    Returns:
        results: {"COA-A": float, "COA-B": float, ...} average Spearman ρ
    """
    if ground_truths is None:
        ground_truths = ["A", "B", "E"]  # Default: compute GTs without API/Lean
    
    results = {}
    batch_size = credit_weights.shape[0]
    
    for gt_name in ground_truths:
        correlations = []
        
        for i in range(batch_size):
            credit_i = credit_weights[i]  # (seq_len,)
            attn_i = attention_mask[i] if attention_mask is not None else torch.ones_like(credit_i)
            
            if gt_name == "A":
                # Mask-and-Forward (expensive — sample one)
                if i == 0:  # Only first for efficiency
                    gt_i = compute_gt_mask_and_forward(
                        model, tokenizer,
                        prompts[i], generated_ids_list[i], answers[i],
                    )
                else:
                    gt_i = torch.zeros_like(credit_i)
            
            elif gt_name == "B":
                # MC Shapley (expensive — approximate)
                if i < 3:  # First 3 for efficiency
                    gt_i = compute_gt_mc_shapley(
                        model, tokenizer,
                        prompts[i], generated_ids_list[i], answers[i],
                        num_subsets=50,  # Reduced for efficiency
                    )
                else:
                    gt_i = torch.zeros_like(credit_i)
            
            elif gt_name == "C":
                # LLM-as-Judge (needs API)
                gt_i = compute_gt_llm_judge(
                    tokenizer, prompts[i], generated_texts[i], answers[i],
                )
            
            elif gt_name == "D":
                # Lean-Grounded (needs Lean 4 data)
                gt_i = compute_gt_lean_grounded(
                    tokenizer, prompts[i], generated_texts[i],
                )
            
            elif gt_name == "E":
                # CMI Capacity
                if hidden_states is not None and rewards is not None:
                    gt_i = compute_gt_cmi_capacity(
                        hidden_states, rewards, attention_mask,
                    )
                else:
                    gt_i = torch.zeros_like(credit_i)
            
            else:
                raise ValueError(f"Unknown ground truth: {gt_name}")
            
            # Compute correlation (normalize GT to credit range)
            if gt_i.abs().sum() > 0:
                rho = spearman_rho(credit_i, gt_i, mask=attn_i.bool())
                correlations.append(rho)
        
        if correlations:
            results[f"COA-{gt_name}"] = sum(correlations) / len(correlations)
        else:
            results[f"COA-{gt_name}"] = 0.0
    
    return results


def evaluate_multi_method(
    method_credits: Dict[str, torch.Tensor],  # {method_name: credit_weights}
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    generated_texts: List[str],
    generated_ids_list: List[torch.Tensor],
    answers: List[str],
    hidden_states: Optional[torch.Tensor] = None,
    rewards: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple methods against all ground truths.
    
    Returns:
        {method_name: {"COA-A": float, "COA-B": float, ...}}
    """
    all_results = {}
    
    for method, credit in method_credits.items():
        results = evaluate_credit_quality(
            credit_weights=credit,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            generated_texts=generated_texts,
            generated_ids_list=generated_ids_list,
            answers=answers,
            hidden_states=hidden_states,
            rewards=rewards,
            attention_mask=attention_mask,
            ground_truths=ground_truths,
        )
        all_results[method] = results
    
    return all_results
