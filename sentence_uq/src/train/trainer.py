"""Bilevel trainer for the Bayesian sentence-level UQ model.

Phase 4-1 (binomial observation, setup-aware). Combines the feature
extractor (Phase 2-1) and the damped Fisher-scoring inner loop
(Phase 3-1) wrapped by :class:`BayesianSentenceUQ` (Phase 3-2) into the
outer optimisation loop that updates ``ψ = (W, α, μ_0, log σ_0)``.

Outer / inner structure (research_document_v8 Part VII §7.6)
------------------------------------------------------------
* **Inner loop** — at every outer step the differentiable Fisher
  scoring of Phase 3-1 returns ``θ̂(ψ)`` *with autograd hooked in*.
* **Outer loop** — Adam over ``ψ`` minimises the (sum) binomial NLL::

      L_outer(ψ) = Σ_{j: m_j > 0} [
          -K_j log μ̃_j(ψ) - (m_j - K_j) log(1 - μ̃_j(ψ))
      ],
      μ_j(ψ) = (1/L_j) Σ_ℓ σ(θ̂(ψ)ᵀ z_ℓ(ψ)),
      μ̃_j(ψ) = clip(μ_j(ψ), ε, 1 - ε)

``θ̂`` is recomputed every outer step (no warm-starting on the autograd
path) so gradients flow ``∂L/∂ψ`` cleanly.

Setup-aware data flow
---------------------
:meth:`SentenceUQTrainer.prepare_data` reads a pre-computed split
(``data/splits/setup_{N}.json`` from Phase 1-0) and joins:

* generation tensors from ``data/generations/{dataset}/...`` (Phase 1-1),
* cached per-token scalars from ``data/cache/{dataset}/...`` (Phase 1-3),
* annotation results (per-sentence ``K_j``, ``m_j``, token_range) from
  ``data/processed/{dataset}/...`` (Phase 1-4).

It flattens to a list of sentence-level records — one dict per
sentence — that the rest of the trainer consumes.

Notes
-----
* ``m_j = 0`` sentences are kept in the records but skipped by both
  ``model.compute_loss`` (CLAUDE.md rule 8) and the evaluation metrics.
* Hidden states arrive in fp16; ``extract_sentence_token_features`` casts
  to fp32 internally (CLAUDE.md rule 10).
* Multiple sentences from the same prompt share Python references to the
  same ``hidden_states`` / ``entropy`` / ``top1`` tensors, so the memory
  cost of "flattening" is constant per prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.features.extractor import extract_sentence_token_features
from src.models.bayesian_main import BayesianSentenceUQ, verify_local_pd
from src.utils.validation import validate_binomial_counts


__all__ = ["SentenceUQTrainer"]


_UNSAFE_FNAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(s: str, max_len: int = 200) -> str:
    """Mirror of ``src.data.generation._safe_filename`` for path lookup.

    Generation files for FActScore-Bio live at
    ``factscore_bio/{safe(entity)}.pt`` and LongFact files at
    ``longfact/{safe(topic)}/{idx:03d}.pt``. We need the same sanitiser
    here to map a prompt record back to its ``.pt`` path.
    """
    cleaned = _UNSAFE_FNAME_RE.sub("_", (s or "").strip()).strip("._")
    if not cleaned:
        cleaned = "unnamed"
    return cleaned[:max_len]


class SentenceUQTrainer:
    """Bilevel outer-loop trainer for :class:`BayesianSentenceUQ`.

    Owns an Adam optimiser over ``ψ = (W, α, μ_0, log σ_0)`` and exposes
    ``prepare_data`` / ``train_epoch`` / ``evaluate`` / ``fit`` as the
    Phase 4-1 spec requires.

    Parameters
    ----------
    model : BayesianSentenceUQ
        Wraps the Phase 2-1 feature extractor and the Phase 3-1 inner
        Fisher loop. ``model.parameters()`` must already expose every
        ψ-component (verified by Phase 3-2 tests).
    lr : float, optional
        Learning rate for Adam. Defaults to ``1e-3`` (spec).
    num_epochs : int, optional
        Maximum number of outer-loop epochs in :meth:`fit`. Defaults
        to 50 (spec).
    eval_every : int, optional
        Run :meth:`evaluate` on the val set every ``eval_every`` epochs.
        Set to 0 to disable.
    pd_check_every : int, optional
        Run :func:`verify_local_pd` every ``pd_check_every`` epochs. The
        true-Hessian check costs ``O(k²)`` autograd backward passes, so
        the spec recommends 5 (default). Set to 0 to disable.
    device : str | torch.device, optional
        Device for the model and the per-sentence feature extraction.
        Defaults to ``"cpu"``.
    log_fn : Callable[[str], None], optional
        Where to send progress lines. Defaults to ``print``.
    weight_decay : float, optional
        Adam weight decay. Defaults to 0.
    """

    def __init__(
        self,
        model: BayesianSentenceUQ,
        lr: float = 1e-3,
        num_epochs: int = 50,
        eval_every: int = 1,
        pd_check_every: int = 5,
        device: str | torch.device = "cpu",
        log_fn: Optional[Callable[[str], None]] = None,
        weight_decay: float = 0.0,
    ) -> None:
        if not isinstance(model, BayesianSentenceUQ):
            raise TypeError(
                "model must be a BayesianSentenceUQ instance; "
                f"got {type(model).__name__}"
            )
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if eval_every < 0:
            raise ValueError(f"eval_every must be non-negative, got {eval_every}")
        if pd_check_every < 0:
            raise ValueError(
                f"pd_check_every must be non-negative, got {pd_check_every}"
            )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.eval_every = int(eval_every)
        self.pd_check_every = int(pd_check_every)
        self.log_fn = log_fn if log_fn is not None else print

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        split_file: str | Path,
        generations_dirs: Dict[str, str | Path],
        cache_dirs: Dict[str, str | Path],
        processed_dirs: Optional[Dict[str, str | Path]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Flatten a Phase 1-0 split into per-sentence training records.

        Each output record carries the dataset-tag, a source identifier,
        the ``(token_range, K_j, m_j)`` triple from Phase 1-4 annotation,
        and references to the prompt-level ``hidden_states`` / ``entropy``
        / ``top1`` tensors needed by :func:`extract_sentence_token_features`.

        Parameters
        ----------
        split_file : path-like
            ``data/splits/setup_{N}.json`` (Phase 1-0 output).
        generations_dirs : dict
            ``{"factscore_bio": "data/generations/factscore_bio",
               "longfact":      "data/generations/longfact"}``.
        cache_dirs : dict
            Same structure for the Phase 1-3 entropy / top-1 cache.
        processed_dirs : dict, optional
            Same structure for the Phase 1-4 annotation outputs. Defaults
            to ``"data/processed/{dataset}"`` when omitted.

        Returns
        -------
        dict
            ``{"train": [...], "val": [...], "test": [...]}``. Each entry
            is a list of per-sentence dicts shaped as::

                {
                    "dataset":   "factscore_bio" | "longfact",
                    "source_id": entity_name | f"{topic}/{prompt_idx}",
                    "token_range": (int, int),
                    "K_j": int,
                    "m_j": int,
                    "hidden_states": fp16 Tensor (T, num_layers, hidden_dim),
                    "entropy": fp32 Tensor (T,),
                    "top1":    fp32 Tensor (T,),
                }
        """
        split_path = Path(split_file)
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

        if processed_dirs is None:
            processed_dirs = {
                ds: f"data/processed/{ds}" for ds in generations_dirs
            }

        gen_path_to_idx = {
            ds: self._build_generation_index(Path(gen_dir))
            for ds, gen_dir in generations_dirs.items()
        }
        annotations = {
            ds: self._load_annotations(ds, Path(processed_dirs[ds]))
            for ds in generations_dirs
        }

        prompt_cache: Dict[Tuple[str, str], Tuple[Tensor, Tensor, Tensor]] = {}

        out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for section in ("train", "val", "test"):
            for prompt in split.get(section) or []:
                ds = prompt.get("dataset")
                if ds not in generations_dirs:
                    continue

                rel_path, source_id, ann_key = self._prompt_keys(prompt, ds)
                if rel_path is None:
                    continue
                if rel_path not in gen_path_to_idx[ds]:
                    continue

                ann_record = annotations[ds].get(ann_key)
                if ann_record is None:
                    continue

                cache_key = (ds, rel_path)
                if cache_key not in prompt_cache:
                    cache_idx = gen_path_to_idx[ds][rel_path]
                    prompt_cache[cache_key] = self._load_prompt_tensors(
                        gen_dir=Path(generations_dirs[ds]),
                        cache_dir=Path(cache_dirs[ds]),
                        rel_path=rel_path,
                        cache_idx=cache_idx,
                    )
                hidden_states, entropy, top1 = prompt_cache[cache_key]
                T = int(hidden_states.shape[0])

                for sent in ann_record.get("sentences", []) or []:
                    tr_raw = sent.get("token_range")
                    if tr_raw is None or len(tr_raw) != 2:
                        continue
                    start, end = int(tr_raw[0]), int(tr_raw[1])
                    if end <= start or end > T:
                        continue

                    out[section].append(
                        {
                            "dataset": ds,
                            "source_id": source_id,
                            "token_range": (start, end),
                            "K_j": int(sent.get("K_j", 0) or 0),
                            "m_j": int(sent.get("m_j", 0) or 0),
                            "hidden_states": hidden_states,
                            "entropy": entropy,
                            "top1": top1,
                        }
                    )

        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self, train_data: Sequence[Dict[str, Any]]
    ) -> Dict[str, float]:
        """One full-batch outer-loop step.

        Recomputes ``z_tokens`` for every training sentence (so gradients
        flow through ``W`` and ``α``), runs the differentiable Fisher
        MAP, evaluates the binomial NLL, and steps Adam.

        Parameters
        ----------
        train_data : sequence of per-sentence dicts (from :meth:`prepare_data`).

        Returns
        -------
        dict
            ``{"loss": float, "num_sentences": int, "num_positive": int}``.
            ``num_positive`` counts sentences with ``m_j > 0`` (the ones
            that contribute to the binomial NLL).
        """
        if len(train_data) == 0:
            raise ValueError("train_data is empty")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        all_z, all_K, all_m = self._collate(train_data, requires_grad=True)
        loss = self.model.compute_loss(all_z, all_K, all_m)
        loss.backward()
        self.optimizer.step()

        num_positive = int((all_m > 0).sum().item())
        return {
            "loss": float(loss.detach().item()),
            "num_sentences": len(train_data),
            "num_positive": num_positive,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        train_data: Sequence[Dict[str, Any]],
        eval_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Ratio-level metrics on ``eval_data`` using ``θ̂`` learned from ``train_data``.

        Re-runs the (detached) Fisher MAP on ``train_data`` with the
        current ψ, then plug ``θ̂`` into ``μ_j = (1/L_j) Σ_ℓ σ(θ̂ᵀ z_ℓ)``
        for every evaluation sentence and compares against the observed
        ratio ``U_j = K_j / m_j`` (sentences with ``m_j = 0`` are skipped
        — CLAUDE.md rule 8). The reported metrics are the Phase 6-1
        ratio-level primary set: MAE, RMSE, Pearson r, mean binomial NLL.

        Parameters
        ----------
        train_data, eval_data : sequences of per-sentence dicts.

        Returns
        -------
        dict
            ``{"MAE", "RMSE", "Pearson_r", "binomial_NLL", "n"}``.
            ``n`` is the number of evaluated (``m_j > 0``) sentences.
        """
        self.model.eval()

        with torch.no_grad():
            train_z, train_K, train_m = self._collate(
                train_data, requires_grad=False
            )
            theta_hat, _ = self.model.compute_map(
                train_z, train_K, train_m, differentiable=False
            )

            eval_z, eval_K, eval_m = self._collate(eval_data, requires_grad=False)

            mu_hats = torch.empty(
                len(eval_z), dtype=torch.float32, device=self.device
            )
            for j, z in enumerate(eval_z):
                pi = torch.sigmoid(z @ theta_hat)
                mu_hats[j] = pi.mean()

        return self._ratio_metrics(mu_hats, eval_K, eval_m, eps=self.model.eps)

    # ------------------------------------------------------------------
    # End-to-end fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Optional[Sequence[Dict[str, Any]]] = None,
        test_data: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the full outer loop and return per-epoch + final tensors.

        Parameters
        ----------
        train_data, val_data, test_data : sequences of per-sentence dicts.
            ``val_data`` / ``test_data`` may be ``None``; metrics for the
            corresponding section are then omitted.

        Returns
        -------
        dict with keys:
            ``train_loss``    : list[float] — outer loss per epoch
            ``val_metrics``   : list[dict]  — metric dicts (when val_data given)
            ``pd_checks``     : list[dict]  — verify_local_pd outputs
            ``test_metrics``  : dict        — final test metrics (when test_data given)
            ``theta_hat``     : Tensor (k,) — final MAP from train
            ``Sigma_hat``     : Tensor (k, k) — Laplace covariance (inverse of Fisher precision)
        """
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_metrics": [],
            "pd_checks": [],
        }

        for epoch in range(1, self.num_epochs + 1):
            train_stats = self.train_epoch(train_data)
            history["train_loss"].append(train_stats["loss"])

            log_line = (
                f"[epoch {epoch:3d}/{self.num_epochs}] "
                f"loss={train_stats['loss']:.4f} "
                f"(N+={train_stats['num_positive']})"
            )

            do_eval = (
                val_data is not None
                and self.eval_every > 0
                and epoch % self.eval_every == 0
            )
            if do_eval:
                val_metrics = self.evaluate(train_data, val_data)
                history["val_metrics"].append({"epoch": epoch, **val_metrics})
                log_line += (
                    f"  val MAE={val_metrics['MAE']:.4f}"
                    f"  RMSE={val_metrics['RMSE']:.4f}"
                    f"  r={val_metrics['Pearson_r']:.4f}"
                    f"  NLL={val_metrics['binomial_NLL']:.4f}"
                )

            do_pd = self.pd_check_every > 0 and epoch % self.pd_check_every == 0
            if do_pd:
                pd_info = self._verify_pd(train_data)
                history["pd_checks"].append({"epoch": epoch, **pd_info})
                log_line += f"  pd={pd_info['laplace_valid_local']}"

            self.log_fn(log_line)

        # Final posterior at the end of training.
        with torch.no_grad():
            train_z, train_K, train_m = self._collate(
                train_data, requires_grad=False
            )
            theta_hat, H_fisher = self.model.compute_map(
                train_z, train_K, train_m, differentiable=False
            )
            Sigma_hat = self._safe_inverse(H_fisher)

        history["theta_hat"] = theta_hat.detach().cpu()
        history["Sigma_hat"] = Sigma_hat.detach().cpu()

        if test_data is not None:
            test_metrics = self.evaluate(train_data, test_data)
            history["test_metrics"] = test_metrics
            self.log_fn(
                "[test ] "
                f"MAE={test_metrics['MAE']:.4f} "
                f"RMSE={test_metrics['RMSE']:.4f} "
                f"r={test_metrics['Pearson_r']:.4f} "
                f"NLL={test_metrics['binomial_NLL']:.4f} "
                f"(n={test_metrics['n']})"
            )

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collate(
        self,
        data: Sequence[Dict[str, Any]],
        requires_grad: bool,
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        """Compute ``z_tokens`` per sentence + stack ``K`` and ``m``.

        With ``requires_grad=True`` the feature extraction runs inside
        ``torch.enable_grad`` so gradients propagate to ψ. With
        ``requires_grad=False`` it runs under ``torch.no_grad`` for
        cheap inference.
        """
        params = self.model.feature_params
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()

        all_z: List[Tensor] = []
        K_list: List[int] = []
        m_list: List[int] = []
        with ctx:
            for sent in data:
                hidden_states = sent["hidden_states"].to(self.device)
                entropy = sent["entropy"].to(self.device)
                top1 = sent["top1"].to(self.device)
                token_range = (
                    int(sent["token_range"][0]),
                    int(sent["token_range"][1]),
                )
                z = extract_sentence_token_features(
                    hidden_states=hidden_states,
                    entropy=entropy,
                    top1_prob=top1,
                    token_range=token_range,
                    params=params,
                )
                all_z.append(z)
                K_list.append(int(sent.get("K_j", 0) or 0))
                m_list.append(int(sent.get("m_j", 0) or 0))

        K_t = torch.tensor(K_list, dtype=torch.long, device=self.device)
        m_t = torch.tensor(m_list, dtype=torch.long, device=self.device)
        validate_binomial_counts(K_t, m_t, context="SentenceUQTrainer._collate")
        return all_z, K_t, m_t

    @staticmethod
    def _ratio_metrics(
        mu_hats: Tensor, all_K: Tensor, all_m: Tensor, eps: float
    ) -> Dict[str, float]:
        """Compute MAE / RMSE / Pearson r / mean binomial NLL on ``m_j > 0`` rows."""
        mask = all_m > 0
        n = int(mask.sum().item())
        if n == 0:
            nan = float("nan")
            return {
                "MAE": nan,
                "RMSE": nan,
                "Pearson_r": nan,
                "binomial_NLL": nan,
                "n": 0,
            }

        K_pos = all_K[mask].to(torch.float32)
        m_pos = all_m[mask].to(torch.float32)
        mu_pos = mu_hats[mask].to(torch.float32)
        U_true = K_pos / m_pos

        mae = float((mu_pos - U_true).abs().mean().item())
        rmse = float(((mu_pos - U_true).pow(2).mean()).sqrt().item())

        if n >= 2:
            mu_std = float(mu_pos.std(unbiased=False).item())
            u_std = float(U_true.std(unbiased=False).item())
            if mu_std > 0.0 and u_std > 0.0:
                cov = float(
                    ((mu_pos - mu_pos.mean()) * (U_true - U_true.mean()))
                    .mean()
                    .item()
                )
                pearson = cov / (mu_std * u_std)
            else:
                pearson = float("nan")
        else:
            pearson = float("nan")

        mu_clamped = mu_pos.clamp(eps, 1.0 - eps)
        nll_sum = float(
            (
                -K_pos * torch.log(mu_clamped)
                - (m_pos - K_pos) * torch.log(1.0 - mu_clamped)
            )
            .sum()
            .item()
        )
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Pearson_r": float(pearson),
            "binomial_NLL": nll_sum / float(n),
            "n": n,
        }

    def _verify_pd(self, train_data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Wrap :func:`verify_local_pd` with the trainer's MAP."""
        with torch.no_grad():
            train_z, train_K, train_m = self._collate(
                train_data, requires_grad=False
            )
            theta_hat, _ = self.model.compute_map(
                train_z, train_K, train_m, differentiable=False
            )
        return verify_local_pd(
            theta_hat,
            train_z,
            train_K,
            train_m,
            self.model.feature_params.mu_0,
            self.model.feature_params.get_Sigma_0_inv(),
            eps=self.model.eps,
        )

    @staticmethod
    def _safe_inverse(matrix: Tensor) -> Tensor:
        """Numerically robust inverse of a (near-)PSD matrix.

        Symmetrises ``matrix`` and adds adaptive jitter until inversion
        succeeds — mirrors the Cholesky helper in
        :mod:`src.inference.predict` but uses a direct ``linalg.inv``
        since the caller (Phase 4-1 fit / Phase 3-3 Predictor) only needs
        ``Σ̂ = H_fisher⁻¹``.
        """
        sym = 0.5 * (matrix + matrix.T)
        k = sym.shape[0]
        eye = torch.eye(k, dtype=sym.dtype, device=sym.device)
        jitter = 0.0
        for attempt in range(8):
            try:
                return torch.linalg.inv(sym + jitter * eye)
            except RuntimeError:
                jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
        raise RuntimeError(
            "Failed to invert Fisher precision even with jitter up to "
            f"{jitter:.2e}; check training stability."
        )

    # ------------------------------------------------------------------
    # File-system helpers for prepare_data
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generation_index(generations_dir: Path) -> Dict[str, int]:
        """Map each ``.pt`` relative path → its sorted-position index.

        The position is exactly the index used by
        :func:`src.features.cached_scalars.cache_scalars_for_directory`,
        so we can derive the cache filename ``{idx:05d}.pt`` from any
        generation file.
        """
        if not generations_dir.exists():
            return {}
        files = sorted(
            (p for p in generations_dir.rglob("*.pt") if p.is_file()),
            key=lambda p: p.relative_to(generations_dir).as_posix(),
        )
        return {
            p.relative_to(generations_dir).as_posix(): i
            for i, p in enumerate(files)
        }

    @staticmethod
    def _prompt_keys(
        prompt: Dict[str, Any], dataset: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return ``(rel_path, source_id, annotation_key)`` for a split entry.

        - FActScore-Bio: rel_path ``{safe(entity)}.pt``; both ids = original entity.
        - LongFact: rel_path ``{safe(topic)}/{idx:03d}.pt``;
          both ids ``f"{topic}/{idx}"``.
        """
        if dataset == "factscore_bio":
            entity = str(prompt.get("entity") or "")
            if not entity:
                return None, None, None
            return f"{_safe_filename(entity)}.pt", entity, entity
        if dataset == "longfact":
            topic = str(prompt.get("topic") or "")
            idx = prompt.get("prompt_idx", 0)
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                idx_int = 0
            if not topic:
                return None, None, None
            return (
                f"{_safe_filename(topic)}/{idx_int:03d}.pt",
                f"{topic}/{idx_int}",
                f"{topic}/{idx_int}",
            )
        return None, None, None

    @staticmethod
    def _load_prompt_tensors(
        gen_dir: Path,
        cache_dir: Path,
        rel_path: str,
        cache_idx: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Load ``(hidden_states, entropy, top1)`` for one prompt."""
        gen_path = gen_dir / rel_path
        gen_payload = torch.load(gen_path, map_location="cpu", weights_only=False)
        hidden_states = gen_payload["hidden_states"]

        cache_path = cache_dir / f"{cache_idx:05d}.pt"
        cache_payload = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        entropy = cache_payload["entropy"]
        top1 = cache_payload["top1_prob"]

        if int(hidden_states.shape[0]) != int(entropy.shape[0]):
            raise ValueError(
                f"hidden_states / entropy length mismatch for {rel_path}: "
                f"{tuple(hidden_states.shape)} vs {tuple(entropy.shape)}"
            )
        return hidden_states, entropy, top1

    @staticmethod
    def _load_annotations(
        dataset: str, processed_dir: Path
    ) -> Dict[str, Dict[str, Any]]:
        """Load every annotation record under ``processed_dir`` keyed by source_id.

        Uses the combined ``annotated.json`` when available; falls back
        to per-record JSON files otherwise. Returns ``{}`` if the
        directory does not exist (caller skips matching prompts).
        """
        if not processed_dir.exists():
            return {}

        records: List[Dict[str, Any]] = []
        combined = processed_dir / "annotated.json"
        if combined.exists():
            try:
                with open(combined, "r", encoding="utf-8") as f:
                    records = json.load(f) or []
            except (OSError, json.JSONDecodeError):
                records = []

        if not records:
            if dataset == "factscore_bio":
                for p in sorted(processed_dir.glob("*.json")):
                    if p.name == "annotated.json":
                        continue
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            records.append(json.load(f))
                    except (OSError, json.JSONDecodeError):
                        continue
            elif dataset == "longfact":
                for topic_dir in sorted(
                    p for p in processed_dir.iterdir() if p.is_dir()
                ):
                    if topic_dir.name == "knowledge":
                        continue
                    for p in sorted(topic_dir.glob("*.json")):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                records.append(json.load(f))
                        except (OSError, json.JSONDecodeError):
                            continue

        out: Dict[str, Dict[str, Any]] = {}
        for record in records:
            meta = record.get("meta") or {}
            if dataset == "factscore_bio":
                key = record.get("entity") or meta.get("entity")
            elif dataset == "longfact":
                topic = record.get("topic") or meta.get("topic")
                idx = record.get("prompt_idx")
                if idx is None:
                    idx = meta.get("prompt_idx", 0)
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    idx_int = 0
                key = f"{topic}/{idx_int}" if topic else None
            else:
                key = None
            if key:
                out[str(key)] = record
        return out
