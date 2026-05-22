"""Plain sklearn LogisticRegression baseline.

Phase 5-1 baseline #4. Predicts ``μ̂_j ∈ [0, 1]`` from a sentence-level
aggregate feature vector built from the *cached* generation-time
quantities (no re-encoding):

    ζ_j = [ mean_ℓ h̄_ℓ,  mean_ℓ H_ℓ,  mean_ℓ p^{(1)}_ℓ ],   ℓ ∈ s_j

where ``h̄_ℓ = Σ_l (1 / num_layers) · h_ℓ^{(l)}`` is the *unweighted*
layer-average hidden state (a deliberately cheap surrogate for the
learnable layer-mixing of the main model). The trailing entropy /
top-1 scalars match the per-token feature recipe of
:mod:`src.features.extractor` (Part VI §6.1) — they are the bottom
two coordinates of ``z_ℓ`` averaged over the sentence.

This baseline produces **point estimates only** — no uncertainty
decomposition. It exists so the Phase 5-1 evaluation can show that
the Bayesian head meaningfully improves calibration over the same
features fitted by a vanilla logistic regression.

Training target
---------------
The spec leaves the target unspecified. We default to the strict
factuality label ``A_j = 1{K_j = m_j}`` (matching the binary head used
in Han et al. and our secondary evaluation tier) but accept
``"ratio"`` for ``U_j = K_j / m_j`` regression-style targets — fitted
by clipping ``U_j`` to ``(eps, 1 - eps)`` and training the same
classifier with sample weights ``m_j`` so the binomial-like
likelihood ``K_j log p + (m_j - K_j) log (1 - p)`` is correctly
optimised (the sklearn ``log_loss`` reduces to this objective once
labels are augmented this way — see Hosmer & Lemeshow §1.6).

``m_j == 0`` rows are dropped (CLAUDE.md rule 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


__all__ = ["LogisticRegressionBaseline", "build_sentence_features"]


def build_sentence_features(
    hidden_states: Tensor,
    entropy: Tensor,
    top1: Tensor,
    token_range: tuple[int, int],
) -> Tensor:
    """Compute the sentence-level aggregate feature ``ζ_j``.

    Layer-averaged hidden state + scalar means of entropy / top-1::

        h̄_ℓ = mean over layers of h_ℓ^{(l)}     ∈ R^{hidden_dim}
        ζ_j = [ mean_ℓ h̄_ℓ , mean_ℓ H_ℓ , mean_ℓ p^{(1)}_ℓ ] ∈ R^{hidden_dim + 2}

    Parameters
    ----------
    hidden_states : Tensor of shape ``(T, num_layers, hidden_dim)``.
    entropy : Tensor of shape ``(T,)``.
    top1 : Tensor of shape ``(T,)``.
    token_range : tuple ``(start, end)`` — half-open.

    Returns
    -------
    Tensor of shape ``(hidden_dim + 2,)`` in fp32 on CPU.
    """
    start, end = int(token_range[0]), int(token_range[1])
    if end <= start:
        raise ValueError(
            f"token_range covers zero tokens: ({start}, {end})"
        )
    T = int(hidden_states.shape[0])
    if end > T:
        raise ValueError(
            f"token_range end={end} exceeds sequence length T={T}"
        )

    h = hidden_states[start:end].to(torch.float32)            # (L_j, L_lay, D)
    h_layer_avg = h.mean(dim=1)                               # (L_j, D)
    h_mean = h_layer_avg.mean(dim=0)                          # (D,)
    ent_mean = entropy[start:end].to(torch.float32).mean()    # scalar
    top1_mean = top1[start:end].to(torch.float32).mean()      # scalar

    return torch.cat([h_mean, ent_mean.view(1), top1_mean.view(1)], dim=0).cpu()


@dataclass(frozen=True)
class _FittedState:
    feature_dim: int
    target: str  # "strict" | "ratio"


class LogisticRegressionBaseline:
    """Wrapper around ``sklearn.linear_model.LogisticRegression``.

    Parameters
    ----------
    target : {"strict", "ratio"}, optional
        ``"strict"`` (default) trains on ``A_j = 1{K_j = m_j}``. ``"ratio"``
        trains on the augmented ``(K_j positive + (m_j - K_j) negative)``
        sample replication so the optimisation matches the binomial
        log-likelihood at the sentence level.
    C : float, optional
        Inverse L2 regularisation strength. Defaults to ``1.0`` (sklearn
        default). The Phase 5-1 spec does not constrain this — it
        appears only in the L1 Han et al. probe.
    max_iter : int, optional
        Forwarded to sklearn. Defaults to 1000 to avoid convergence
        warnings on the higher-dim hidden-state feature.
    random_state : int, optional
        Forwarded to sklearn.
    """

    def __init__(
        self,
        target: str = "strict",
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 0,
    ) -> None:
        if target not in ("strict", "ratio"):
            raise ValueError(
                f"target must be 'strict' or 'ratio'; got {target!r}"
            )
        self.target = target
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._clf: Any = None
        self._fitted: Optional[_FittedState] = None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        Z: Tensor,
        K: Tensor,
        m: Tensor,
    ) -> "LogisticRegressionBaseline":
        """Fit on ``(Z, K, m)``. Rows with ``m_j == 0`` are dropped.

        Parameters
        ----------
        Z : Tensor of shape ``(N, feature_dim)``.
        K, m : Tensor of shape ``(N,)`` (integer counts).

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression

        Z_np, y_np, w_np = self._prepare_training_arrays(Z, K, m)
        if Z_np.shape[0] == 0:
            raise ValueError("no usable training rows (all m_j == 0)")
        if len(np.unique(y_np)) < 2:
            raise ValueError(
                "training labels are degenerate (only one class present); "
                "logistic regression cannot be fit"
            )

        self._clf = LogisticRegression(
            penalty="l2",
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(Z_np, y_np, sample_weight=w_np)
        self._fitted = _FittedState(
            feature_dim=int(Z_np.shape[1]), target=self.target
        )
        return self

    def predict_proba(self, Z: Tensor) -> Tensor:
        """Return ``p(A = 1 | ζ)`` (or ``μ̂`` in ``"ratio"`` mode).

        Parameters
        ----------
        Z : Tensor of shape ``(N, feature_dim)`` or ``(feature_dim,)``.

        Returns
        -------
        Tensor of shape ``(N,)`` in fp32 on CPU.
        """
        if self._clf is None or self._fitted is None:
            raise RuntimeError("LogisticRegressionBaseline.fit must be called first")
        Z_np = self._coerce_to_numpy(Z)
        if Z_np.ndim == 1:
            Z_np = Z_np[None, :]
        if Z_np.shape[1] != self._fitted.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: model fit on {self._fitted.feature_dim} "
                f"dims, predict received {Z_np.shape[1]}"
            )
        # sklearn classes_ may be sorted as [0, 1]; locate the positive class.
        classes = list(self._clf.classes_)
        pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
        probs = self._clf.predict_proba(Z_np)[:, pos_idx]
        return torch.from_numpy(probs.astype(np.float32))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_numpy(Z: Tensor) -> np.ndarray:
        if torch.is_tensor(Z):
            return Z.detach().cpu().to(torch.float32).numpy()
        return np.asarray(Z, dtype=np.float32)

    def _prepare_training_arrays(
        self, Z: Tensor, K: Tensor, m: Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if Z.dim() != 2:
            raise ValueError(
                f"Z must be 2-D (N, feature_dim); got shape {tuple(Z.shape)}"
            )
        N = int(Z.shape[0])
        if K.shape != (N,):
            raise ValueError(
                f"K must have shape (N={N},); got {tuple(K.shape)}"
            )
        if m.shape != (N,):
            raise ValueError(
                f"m must have shape (N={N},); got {tuple(m.shape)}"
            )

        Z_np = Z.detach().cpu().to(torch.float32).numpy()
        K_np = K.detach().cpu().to(torch.int64).numpy()
        m_np = m.detach().cpu().to(torch.int64).numpy()

        mask = m_np > 0
        Z_np = Z_np[mask]
        K_np = K_np[mask]
        m_np = m_np[mask]

        if self.target == "strict":
            y_np = (K_np == m_np).astype(np.int64)
            w_np = np.ones_like(y_np, dtype=np.float64)
            return Z_np, y_np, w_np

        # "ratio" mode — replicate each sentence with K positives and
        # (m - K) negatives.  Sample weights = m_j keep the influence
        # of each *sentence* equal across rows of different m_j.
        keep_rows: List[int] = []
        ys: List[int] = []
        ws: List[float] = []
        for i in range(Z_np.shape[0]):
            mi = int(m_np[i])
            ki = int(K_np[i])
            if mi <= 0:
                continue
            if ki > 0:
                keep_rows.append(i)
                ys.append(1)
                ws.append(float(ki))
            if (mi - ki) > 0:
                keep_rows.append(i)
                ys.append(0)
                ws.append(float(mi - ki))
        if not keep_rows:
            return (
                np.zeros((0, Z_np.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
            )
        rep_Z = Z_np[np.asarray(keep_rows, dtype=np.int64)]
        rep_y = np.asarray(ys, dtype=np.int64)
        rep_w = np.asarray(ws, dtype=np.float64)
        return rep_Z, rep_y, rep_w


def collate_sentence_features(
    sentence_records: Iterable[Dict[str, Any]],
) -> Dict[str, Tensor]:
    """Convenience: turn :meth:`SentenceUQTrainer.prepare_data` rows into ``Z, K, m``.

    Each input record must expose ``hidden_states`` / ``entropy`` /
    ``top1`` / ``token_range`` / ``K_j`` / ``m_j`` (the shape produced
    by :meth:`src.train.trainer.SentenceUQTrainer.prepare_data`).

    Returns
    -------
    dict with keys ``"Z"`` (Tensor of shape ``(N, hidden_dim + 2)``),
    ``"K"`` and ``"m"`` (LongTensors of shape ``(N,)``).
    """
    Z_rows: List[Tensor] = []
    K_list: List[int] = []
    m_list: List[int] = []
    for rec in sentence_records:
        Z_rows.append(
            build_sentence_features(
                hidden_states=rec["hidden_states"],
                entropy=rec["entropy"],
                top1=rec["top1"],
                token_range=(int(rec["token_range"][0]), int(rec["token_range"][1])),
            )
        )
        K_list.append(int(rec.get("K_j", 0) or 0))
        m_list.append(int(rec.get("m_j", 0) or 0))
    if not Z_rows:
        return {
            "Z": torch.zeros((0, 0), dtype=torch.float32),
            "K": torch.zeros((0,), dtype=torch.long),
            "m": torch.zeros((0,), dtype=torch.long),
        }
    return {
        "Z": torch.stack(Z_rows, dim=0),
        "K": torch.tensor(K_list, dtype=torch.long),
        "m": torch.tensor(m_list, dtype=torch.long),
    }
