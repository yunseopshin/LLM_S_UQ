# Phase 1-3 — Entropy / Top-1 Offline Cache

Implement `src/features/cached_scalars.py`.

**Purpose**:
Compute per-token predictive entropy and top-1 probability from logits.
These are ψ-independent, so compute once and cache offline.

**Requirements**:

1. Function `compute_token_entropy_and_top1(logits)`:
   - Input: logits (T, vocab_size) Tensor
   - Compute:
     * probs = softmax(logits, dim=-1)
     * entropy = -Σ p * log(p), handling log(0) via torch.where or small eps
     * top1_prob = probs.max(dim=-1).values
   - Returns: entropy (T,), top1_prob (T,)

2. Function `cache_scalars_for_directory(generations_dir, cache_dir)`:
   - Iterate over all .pt files in generations_dir
   - Compute entropy and top1 for each file's logits
   - Save to {cache_dir}/{idx:05d}.pt:
     * "entropy": (T,) fp32 Tensor
     * "top1_prob": (T,) fp32 Tensor
     * "token_ids": (T,) Long Tensor (for sanity checks)
   - Keep original logits in generations/ (don't delete)

3. Function `load_scalars(idx, cache_dir)`:
   - Load and return saved entropy, top1_prob, token_ids

**Script `scripts/01b_cache_scalars.py`**:
- Read generations_dir, cache_dir from config
- Run cache_scalars_for_directory with tqdm progress

**Important**:
- Numerical stability: use log_softmax or subtract logits.max before softmax
- Handle 0 * log(0) = 0: use -(p * logp).nansum() or torch.where
- Cast fp16 logits to fp32 before computation
