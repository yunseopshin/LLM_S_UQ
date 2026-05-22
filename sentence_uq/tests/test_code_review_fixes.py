"""Regression tests for Phase 7-3 code-review fixes.

Each test pins down one invariant established in
``prompts/phase_7_3_code_review_fixes.md``:

1. Fix 1 — generation loop stores ``logits[t]`` for ``token_ids[t]``.
2. Fix 3 — epsilon-stabilised gradient keeps boundary sentences contributing.
3. Fix 4 — invalid binomial counts (``K > m``, ``K < 0``, ``m < 0``) raise.
4. Fix 5 — cache / generation mismatches raise on load.
5. Fix 6 — strict-factuality and error-detection AUROCs are equal by
   the ``AUROC(1-y, 1-s) == AUROC(y, s)`` identity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.generation import (  # noqa: E402
    generate_with_hidden_states,
    resolve_selected_layers,
)
from src.evaluation.metrics import compute_strict_metrics  # noqa: E402
from src.models.fisher_scoring import (  # noqa: E402
    _compute_grad_and_fisher,
    _last_diagnostics,
)
from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.utils.validation import validate_binomial_counts  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny fakes — kept local to avoid cross-test imports
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal HF-tokenizer surface used by ``generate_with_hidden_states``."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.chat_template = None  # bypass chat templating

    def _encode_str(self, text: str) -> list[int]:
        return [(ord(c) % (self.vocab_size - 2)) + 2 for c in text][:16] or [2]

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        ids = torch.tensor([self._encode_str(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(
            chr(int(t) + 32) for t in token_ids if int(t) != self.eos_token_id
        )

    def convert_tokens_to_ids(self, tok: str) -> int:
        return -1


class _FakeCausalLM(torch.nn.Module):
    """Tiny causal LM that returns deterministic logits + hidden states."""

    def __init__(
        self, hidden_dim: int = 8, num_hidden_layers: int = 2, vocab_size: int = 16
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            _name_or_path="fake",
        )
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
                for _ in range(num_hidden_layers)
            ]
        )
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.weight.copy_(torch.eye(hidden_dim) + 0.01 * (i + 1))
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[dict] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        states: list[torch.Tensor] = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            states.append(h)
        logits = self.lm_head(h)
        prev_len = int(past_key_values["seq_len"]) if past_key_values else 0
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(states),
            past_key_values={"seq_len": prev_len + int(input_ids.shape[1])},
        )


# ---------------------------------------------------------------------------
# Fix 1 — generation logits alignment
# ---------------------------------------------------------------------------


def test_generation_logits_are_current_token_logits() -> None:
    """Greedy decoding ⇒ argmax(logits[t]) must equal token_ids[t].

    Before Phase 7-3 fix 1 the loop stored ``step.logits[0, -1, :]``,
    i.e. the distribution conditioned on ``x_{≤t}`` (predicting
    ``x_{t+1}``). After the fix it stores the distribution that
    actually sampled ``x_t``. Under greedy decoding the latter has
    ``token_ids[t] == argmax(logits[t])`` for every ``t``.
    """
    torch.manual_seed(0)
    model = _FakeCausalLM(hidden_dim=8, num_hidden_layers=2, vocab_size=16)
    tok = _FakeTokenizer(vocab_size=16)
    selected = resolve_selected_layers(model.config.num_hidden_layers, None)

    rec = generate_with_hidden_states(
        model,
        tok,
        prompt="abc",
        selected_layers=selected,
        max_new_tokens=5,
        do_sample=False,  # greedy
    )

    token_ids = rec["token_ids"]
    logits = rec["logits"].to(torch.float32)
    assert token_ids.numel() > 0, "fixture must emit at least one token"
    pred = logits.argmax(dim=-1)
    assert torch.equal(pred, token_ids), (
        f"logits[t] should sample token_ids[t]; got argmax={pred.tolist()} "
        f"vs token_ids={token_ids.tolist()}"
    )


# ---------------------------------------------------------------------------
# Fix 3 — epsilon-stabilised gradient at clipping boundary
# ---------------------------------------------------------------------------


def test_clipped_gradient_boundary_behavior() -> None:
    """A sentence with ``μ ≤ ε`` still contributes to the gradient.

    The true clipped-objective gradient would zero such a sentence
    out (``mu_clamped`` is constant in ``θ`` past the clip, so autograd
    through :func:`_compute_clipped_objective` yields zero for that
    sentence). The implementation's epsilon-stabilised gradient keeps
    it, and ``_last_diagnostics`` reports it as a boundary sentence.

    Logit magnitudes are chosen so ``μ`` falls strictly below ``ε``
    while ``π(1 - π)`` is still numerically representable — pushing
    ``z @ θ`` further than this drives the contribution under fp32 noise
    even though the analytical statement still holds.
    """
    k = 3
    eps = 1e-6
    # σ(z @ θ) ≈ 3e-7 < ε, but π(1-π) ≈ 3e-7 is still well above fp32 noise.
    # Per-element value of -5 with θ = 1 gives z @ θ = -15.
    z_boundary = -5.0 * torch.ones(4, k, dtype=torch.float32)
    z_interior = torch.zeros(4, k, dtype=torch.float32)
    all_z = [z_boundary, z_interior]
    all_K = torch.tensor([2, 1], dtype=torch.long)
    all_m = torch.tensor([4, 4], dtype=torch.long)
    mu_0 = torch.zeros(k, dtype=torch.float32)
    Sigma_0_inv = torch.eye(k, dtype=torch.float32)
    theta = torch.ones(k, dtype=torch.float32)

    grad_both, _ = _compute_grad_and_fisher(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps=eps
    )
    diag = dict(_last_diagnostics)
    assert diag["total_sentences"] == 2
    assert diag["boundary_count"] >= 1, (
        f"boundary sentence must register in _last_diagnostics; got {diag}"
    )

    grad_interior_only, _ = _compute_grad_and_fisher(
        theta, [z_interior], all_K[1:], all_m[1:], mu_0, Sigma_0_inv, eps=eps
    )
    assert not torch.allclose(grad_both, grad_interior_only, atol=1e-3), (
        "boundary sentence's contribution should be non-zero in the "
        "epsilon-stabilised gradient"
    )

    # Compare with autograd through the clipped objective at the same
    # θ — they should DIFFER, confirming the analytic gradient is not
    # the true clipped gradient at the boundary (Phase 7-3 fix 2/3).
    from src.models.fisher_scoring import _compute_clipped_objective

    theta_grad = theta.clone().requires_grad_(True)
    obj = _compute_clipped_objective(
        theta_grad, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps=eps
    )
    (autograd_grad,) = torch.autograd.grad(obj, theta_grad)
    assert not torch.allclose(grad_both, autograd_grad, atol=1e-3), (
        "analytic ε-stabilised gradient and autograd-through-clipped "
        "gradient must DIFFER at the boundary"
    )


# ---------------------------------------------------------------------------
# Fix 4 — invalid binomial counts raise
# ---------------------------------------------------------------------------


def test_invalid_binomial_counts_raise() -> None:
    K_good = torch.tensor([0, 1, 2], dtype=torch.long)
    m_good = torch.tensor([0, 2, 3], dtype=torch.long)
    validate_binomial_counts(K_good, m_good, context="ok-tensor")

    K_good_np = np.array([0, 1, 2], dtype=np.int64)
    m_good_np = np.array([0, 2, 3], dtype=np.int64)
    validate_binomial_counts(K_good_np, m_good_np, context="ok-numpy")

    with pytest.raises(ValueError, match="K_j > m_j"):
        validate_binomial_counts(
            torch.tensor([3]), torch.tensor([2]), context="K>m"
        )
    with pytest.raises(ValueError, match="K_j < 0"):
        validate_binomial_counts(
            torch.tensor([-1]), torch.tensor([2]), context="K<0"
        )
    with pytest.raises(ValueError, match="m_j < 0"):
        validate_binomial_counts(
            torch.tensor([0]), torch.tensor([-1]), context="m<0"
        )
    with pytest.raises(ValueError, match="K_j > m_j"):
        validate_binomial_counts(
            np.array([3]), np.array([2]), context="np-K>m"
        )


# ---------------------------------------------------------------------------
# Fix 5 — cache verification
# ---------------------------------------------------------------------------


def _write_pair(
    tmp_path: Path,
    rel_path: str,
    token_ids: torch.Tensor,
    cache_source: str,
    cache_token_ids: torch.Tensor,
) -> tuple[Path, Path]:
    """Write a minimal (gen, cache) pair to ``tmp_path``."""
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "cache"
    (gen_dir / Path(rel_path).parent).mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    T = int(token_ids.shape[0])
    hidden = torch.zeros(T, 2, 4, dtype=torch.float16)
    torch.save(
        {
            "text": "x",
            "prompt": "p",
            "prompt_text": "p",
            "prompt_ids": torch.zeros(1, dtype=torch.long),
            "token_ids": token_ids,
            "hidden_states": hidden,
            "logits": torch.zeros(T, 4, dtype=torch.float16),
            "selected_layers": [0, 1],
            "dataset": "factscore_bio",
            "meta": {},
            "finished": True,
        },
        gen_dir / rel_path,
    )
    torch.save(
        {
            "entropy": torch.zeros(T, dtype=torch.float32),
            "top1_prob": torch.zeros(T, dtype=torch.float32),
            "token_ids": cache_token_ids,
            "source_path": cache_source,
        },
        cache_dir / "00000.pt",
    )
    return gen_dir, cache_dir


def test_cache_source_path_mismatch_raises(tmp_path: Path) -> None:
    token_ids = torch.arange(4, dtype=torch.long)
    gen_dir, cache_dir = _write_pair(
        tmp_path,
        rel_path="correct.pt",
        token_ids=token_ids,
        cache_source="wrong.pt",      # mismatch
        cache_token_ids=token_ids,
    )
    with pytest.raises(ValueError, match="Cache/source mismatch"):
        SentenceUQTrainer._load_prompt_tensors(
            gen_dir, cache_dir, rel_path="correct.pt", cache_idx=0
        )


def test_cache_token_ids_mismatch_raises(tmp_path: Path) -> None:
    token_ids = torch.arange(4, dtype=torch.long)
    other_ids = token_ids.clone()
    other_ids[0] = 99
    gen_dir, cache_dir = _write_pair(
        tmp_path,
        rel_path="correct.pt",
        token_ids=token_ids,
        cache_source="correct.pt",
        cache_token_ids=other_ids,  # content mismatch
    )
    with pytest.raises(ValueError, match="Cache token_ids mismatch"):
        SentenceUQTrainer._load_prompt_tensors(
            gen_dir, cache_dir, rel_path="correct.pt", cache_idx=0
        )


# ---------------------------------------------------------------------------
# Fix 6 — strict-factuality vs error-detection AUROC
# ---------------------------------------------------------------------------


def test_strict_vs_error_metric_direction() -> None:
    rng = np.random.default_rng(0)
    n = 50
    m = rng.integers(1, 6, size=n).astype(np.int64)
    K = np.minimum(m, rng.integers(0, 6, size=n)).astype(np.int64)
    mu_hat = rng.uniform(0.05, 0.95, size=n)

    out = compute_strict_metrics(K, m, mu_hat)
    assert "strict_factuality_auroc" in out
    assert "error_detection_auroc" in out
    # AUROC(1-y, 1-s) == AUROC(y, s).
    assert out["strict_factuality_auroc"] == pytest.approx(
        out["error_detection_auroc"], abs=1e-12
    )
    # Both must be valid probabilities.
    assert 0.0 <= out["strict_factuality_auroc"] <= 1.0
