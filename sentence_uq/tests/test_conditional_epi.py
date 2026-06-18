"""Tests for Phase 13 - conditional epistemic value + interval coverage.

Covers the spec (``prompts/phase_13_epi_conditional_coverage.md``):

1. **Band masking** - ``mu_hat = 0.52`` lands in ``[0.4, 0.6]`` / ``[0.5, 0.7]``
   but not ``[0.6, 0.95]``; empty / ``N < 25`` bands are skipped without error.
2. **Setting-1 synthetic** - when ``epi`` separates ``err`` within a fixed-``mu``
   band the narrow-band AUROC ~ 1 and the partial-corr gate is significant; a
   ``w``-leak construct (``epi := mu(1 - mu)``) is NOT significant after the
   ``[1, mu, mu^2]`` control. Pins the reading rule.
3. **Predictive interval** - for ``Sigma_hat -> 0`` variant (c) collapses to
   variant (b) within tolerance (no epistemic => same interval).
4. **Coverage monotonicity** - empirical coverage is non-decreasing in the
   nominal level, widths are non-decreasing, and all ``U`` intervals lie in
   ``[0, 1]``.
5. **Adaptivity** - on a construct where high-epi sentences are under-covered by
   (b), variant (c) raises their coverage toward nominal.
6. **No-retrain guard** - importing the script does not mutate the saved
   parameters (hash), and ``main`` is guarded behind ``__main__``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    binomial_equal_tailed_interval,
    equal_tailed_interval_from_samples,
    predictive_interval_coverage,
    sample_posterior_predictive_K,
)
from src.features.extractor import SentenceUQParams  # noqa: E402
from src.inference.predict import Predictor  # noqa: E402

_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "13_epi_conditional_coverage.py"


def _load_phase13() -> Any:
    """Import the digit-prefixed Phase-13 script as a module (no side effects)."""
    spec = importlib.util.spec_from_file_location("phase13_module", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_predictor(k: int, sigma_scale: float, seed: int = 0) -> Predictor:
    """A real :class:`Predictor` with random ``theta_hat`` and a PD ``Sigma_hat``.

    Only ``predict_sentence`` / posterior sampling are exercised, which depend on
    ``theta_hat`` / ``Sigma_hat`` (not on the feature extractor), so a minimal
    :class:`SentenceUQParams` with ``feature_dim == k`` suffices.
    """
    torch.manual_seed(seed)
    params = SentenceUQParams(hidden_dim=8, num_layers=2, projection_dim=k - 2)
    assert params.feature_dim == k
    theta = torch.randn(k)
    a = torch.randn(k, k)
    sigma = sigma_scale * (a @ a.T / k + torch.eye(k))
    return Predictor(theta_hat=theta, Sigma_hat=sigma, feature_params=params)


# ---------------------------------------------------------------------------
# Test 1 - band masking + skip
# ---------------------------------------------------------------------------


def test_band_masking_membership_and_skip() -> None:
    ph = _load_phase13()
    mu = np.array([0.52], dtype=np.float64)
    assert bool(ph.mu_band_mask(mu, 0.4, 0.6)[0]) is True
    assert bool(ph.mu_band_mask(mu, 0.5, 0.7)[0]) is True
    assert bool(ph.mu_band_mask(mu, 0.6, 0.95)[0]) is False

    # A tiny signal set: a band catching < 25 sentences is skipped (no error);
    # a band catching nothing is skipped with N = 0.
    rng = np.random.default_rng(0)
    n = 10
    sig = {
        "mu_hat": np.full(n, 0.52),
        "err": rng.integers(0, 2, n).astype(float),
        "epi_mu": rng.random(n),
        "epi_mc": rng.random(n),
        "mean_entropy": rng.random(n),
        "abs_ratio_err": rng.random(n),
    }
    row_small = ph.conditional_epi_band_report("g", (0.4, 0.6), sig, bootstrap_iters=50)
    assert row_small["skipped"] is True
    assert "N<25" in row_small["skip_reason"]
    assert row_small["N"] == 10

    row_empty = ph.conditional_epi_band_report("g", (0.9, 0.99), sig, bootstrap_iters=50)
    assert row_empty["skipped"] is True
    assert row_empty["N"] == 0


# ---------------------------------------------------------------------------
# Test 2 - Setting-1 synthetic: genuine signal vs w-leak
# ---------------------------------------------------------------------------


def test_setting1_signal_survives_and_wleak_does_not() -> None:
    ph = _load_phase13()
    rng = np.random.default_rng(1)
    n_half = 110
    n = 2 * n_half

    # All sentences sit inside the narrow band [0.45, 0.55] => mu_hat ~ const,
    # so w = mu(1-mu) is ~const and cannot explain any epi signal.
    mu_hat = rng.uniform(0.46, 0.54, n)
    err = np.concatenate([np.zeros(n_half), np.ones(n_half)])

    # (a) genuine signal: epi tracks err (independent of mu_hat).
    epi_signal = err * 1.0 + rng.normal(0.0, 0.05, n)
    sig_signal = {
        "mu_hat": mu_hat,
        "err": err,
        "epi_mu": epi_signal,
        "epi_mc": epi_signal + rng.normal(0.0, 0.05, n),
        "mean_entropy": rng.normal(0.0, 1.0, n),
        "abs_ratio_err": err * 0.3 + rng.normal(0.0, 0.02, n),
    }
    row = ph.conditional_epi_band_report(
        "narrow_symmetric", (0.4, 0.6), sig_signal, bootstrap_iters=100, seed=1
    )
    assert row["skipped"] is False
    assert 0.1 <= row["base_err_rate"] <= 0.9
    assert row["auroc_epi_mu"] > 0.9
    # The Setting-1 decision gate is the partial-Spearman ("significant
    # partial_corr"), NOT the broad OR-gate; the genuine signal must pass it.
    assert float(row["partial_pass"]) == 1.0
    assert row["partial_rho"] > 0.0 and row["partial_rho_p"] < 0.05

    # (b) w-leak: epi := mu(1-mu) over a range where w is MONOTONE in mu
    # (mu in [0.05, 0.45]) AND the labels are mu-driven, so the RAW (un-
    # residualised) epi<->err association is genuinely significant. A correct
    # residualisation on [1, mu, mu^2] must remove it (partial NOT significant);
    # a broken residualiser would (wrongly) keep it -- this is the discriminating
    # construct. We score it on a wide band [0.0, 0.5] so the full mu range is in.
    from scipy.stats import spearmanr

    mu_leak = rng.uniform(0.05, 0.45, n)
    epi_leak = mu_leak * (1.0 - mu_leak)              # monotone increasing here
    # err is driven by mu (high mu -> error), so it leaks through epi(mu).
    err_leak = (mu_leak > np.median(mu_leak)).astype(float)
    sig_leak = {
        "mu_hat": mu_leak,
        "err": err_leak,
        "epi_mu": epi_leak,
        "epi_mc": epi_leak,
        "mean_entropy": rng.normal(0.0, 1.0, n),
        "abs_ratio_err": np.abs(err_leak - mu_leak),
    }
    # The leak is real: raw epi<->err Spearman is significant and positive.
    raw_rho, raw_p = spearmanr(epi_leak, err_leak)
    assert raw_rho > 0.0 and raw_p < 0.05
    row_leak = ph.conditional_epi_band_report(
        "narrow_symmetric", (0.0, 0.5), sig_leak, bootstrap_iters=100, seed=1
    )
    assert row_leak["skipped"] is False
    # After the quadratic residualisation the partial_corr is NOT significant:
    # the spec's decision gate correctly rejects the w-leak.
    assert float(row_leak["partial_pass"]) == 0.0


# ---------------------------------------------------------------------------
# Test 3 - predictive interval: (c) collapses to (b) as Sigma -> 0
# ---------------------------------------------------------------------------


def test_predictive_interval_collapses_to_binomial() -> None:
    k = 4
    torch.manual_seed(3)
    theta = torch.randn(k)
    Sigma = 1e-12 * torch.eye(k)
    z = torch.randn(3, k)
    mu_hat = float(torch.sigmoid(z @ theta).mean())
    m_j = 12

    gen = torch.Generator().manual_seed(7)
    K_samples = sample_posterior_predictive_K(
        theta, Sigma, z, m_j, num_samples=4000, generator=gen
    )["K_samples"]
    for level in (0.50, 0.80, 0.90, 0.95):
        lo_c, hi_c = equal_tailed_interval_from_samples(K_samples, level)
        lo_b, hi_b = binomial_equal_tailed_interval(m_j, mu_hat, level)
        assert abs(lo_c - lo_b) <= 1
        assert abs(hi_c - hi_b) <= 1


# ---------------------------------------------------------------------------
# Test 4 - coverage / width monotonicity, U intervals in [0, 1]
# ---------------------------------------------------------------------------


def test_coverage_and_width_monotonic_in_level() -> None:
    ph = _load_phase13()
    k = 6
    predictor = _make_predictor(k, sigma_scale=0.3, seed=4)

    rng = np.random.default_rng(4)
    n = 30
    z_list = [torch.randn(rng.integers(2, 6), k) for _ in range(n)]
    m = rng.integers(3, 15, n).astype(np.float64)
    mu_hat = np.array(
        [float(torch.sigmoid(z @ predictor.theta_hat).mean()) for z in z_list]
    )
    K = np.array([rng.integers(0, int(mi) + 1) for mi in m], dtype=np.float64)
    sig = {"m": m, "mu_hat": mu_hat, "z_tokens_list": z_list}

    levels = (0.50, 0.80, 0.90, 0.95)
    intervals = ph._per_sentence_intervals(
        predictor, sig, levels, num_samples=300, seed=4
    )

    for variant in ("b", "c"):
        covs, widths = [], []
        for lvl in levels:
            lo = intervals[f"{variant}_lo_{lvl}"]
            hi = intervals[f"{variant}_hi_{lvl}"]
            # All U bounds in [0, 1].
            assert np.all(lo >= 0.0) and np.all(hi <= m)
            assert np.all(lo / m >= -1e-9) and np.all(hi / m <= 1.0 + 1e-9)
            res = predictive_interval_coverage(lo, hi, K, m)
            covs.append(res["coverage"])
            widths.append(res["mean_width"])
        # Non-decreasing in nominal level (nested equal-tailed bounds).
        assert np.all(np.diff(covs) >= -1e-9), f"{variant} coverage not monotone: {covs}"
        assert np.all(np.diff(widths) >= -1e-9), f"{variant} width not monotone: {widths}"


# ---------------------------------------------------------------------------
# Test 5 - adaptivity: (c) raises coverage of under-covered high-epi sentences
# ---------------------------------------------------------------------------


def test_adaptivity_c_widens_high_epi_toward_nominal() -> None:
    # Construct: mu_hat = sigmoid(3) ~ 0.953 (logit driven by feature dim 0).
    # High-epi sentences have a huge posterior variance on that direction, so the
    # posterior-predictive (c) spans almost all of [0, m]; the aleatoric-only (b)
    # interval stays tight near m. The truth K = m // 2 is a "surprising" outcome
    # that (b) misses but (c) covers.
    k = 2
    theta = torch.tensor([3.0, 0.0])
    z = torch.tensor([[1.0, 0.0]])
    m_j = 10
    K_true = 5
    level = 0.95
    mu_hat = float(torch.sigmoid(z @ theta).mean())

    Sigma_high = torch.diag(torch.tensor([25.0, 0.0]))
    n_sent = 25
    gen = torch.Generator().manual_seed(5)

    lo_b, hi_b = binomial_equal_tailed_interval(m_j, mu_hat, level)
    b_inside = []
    c_inside = []
    for _ in range(n_sent):
        b_inside.append(lo_b <= K_true <= hi_b)
        K_samples = sample_posterior_predictive_K(
            theta, Sigma_high, z, m_j, num_samples=2000, generator=gen
        )["K_samples"]
        lo_c, hi_c = equal_tailed_interval_from_samples(K_samples, level)
        c_inside.append(lo_c <= K_true <= hi_c)

    cov_b_high = float(np.mean(b_inside))
    cov_c_high = float(np.mean(c_inside))
    # (b) under-covers the surprising outcome; (c) widens and covers it.
    assert cov_b_high <= 0.1
    assert cov_c_high >= 0.9
    assert abs(cov_c_high - level) < abs(cov_b_high - level)


# ---------------------------------------------------------------------------
# Test 6 - no-retrain guard
# ---------------------------------------------------------------------------


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


_MODEL_PATH = _PROJECT_ROOT / "results" / "setup_2" / "trained_model.pt"


def test_no_retrain_guard_model_hash_unchanged() -> None:
    if not _MODEL_PATH.exists():
        pytest.skip("trained_model.pt not present; skipping hash guard")
    before = _sha256(_MODEL_PATH)
    module = _load_phase13()  # import must have no side effects on the model
    after = _sha256(_MODEL_PATH)
    assert before == after, "Phase-13 import mutated the trained model"
    # main() must be guarded (not executed at import time).
    assert hasattr(module, "main")
    assert callable(module.main)


def test_no_retrain_guard_real_run(tmp_path) -> None:
    """Stronger guard: actually RUN main() (read-only) and prove (a) the model is
    byte-unchanged and (b) outputs land only under <results-dir>/conditional_epi/.
    """
    split = _PROJECT_ROOT / "data" / "splits" / "setup_2.json"
    if not (_MODEL_PATH.exists() and split.exists()):
        pytest.skip("Setup-2 model/split not present; skipping real-run guard")
    module = _load_phase13()
    before = _sha256(_MODEL_PATH)
    cwd = Path.cwd()
    import os
    os.chdir(_PROJECT_ROOT)
    try:
        rc = module.main([
            "--setup", "2", "--device", "cpu", "--limit", "8", "--no-plots",
            "--pred-samples", "64", "--mc-samples", "32",
            "--results-dir", str(tmp_path),
            "--trained-model", str(_MODEL_PATH),
        ])
    finally:
        os.chdir(cwd)
    assert rc == 0
    assert _sha256(_MODEL_PATH) == before, "main() mutated the trained model"
    out = tmp_path / "conditional_epi"
    assert (out / "band_sweep.csv").exists()
    assert (out / "coverage.csv").exists()
    assert (out / "setting1_verdict.txt").exists()
    assert (out / "setting2_verdict.txt").exists()


# ---------------------------------------------------------------------------
# Test 7 - the verdict reading-rule functions (the phase deliverables)
# ---------------------------------------------------------------------------


def _band_row(**over: Any) -> dict:
    """A Setting-1 band-report row with safe defaults, overridable per test."""
    row = {
        "group": "narrow_symmetric", "band_lo": 0.4, "band_hi": 0.6, "N": 50,
        "base_err_rate": 0.5, "skipped": False, "skip_reason": "",
        "auroc_epi_mu": 0.5, "auroc_epi_mu_ci_lo": 0.3, "auroc_epi_mu_ci_hi": 0.7,
        "auroc_epi_mc": 0.5, "auroc_epi_mc_ci_lo": 0.3, "auroc_epi_mc_ci_hi": 0.7,
        "partial_rho": 0.0, "partial_rho_p": 0.9, "partial_sign": "+",
        "logit_epi_coef": 0.0, "logit_epi_p": 0.9,
        "partial_pass": 0.0, "logistic_pass": 0.0, "gate_passed": 0.0,
        "auroc_mean_entropy": 0.5, "auroc_neg_mu_hat": 0.5,
    }
    row.update(over)
    return row


def _verdict_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.startswith("VERDICT:"):
            return ln.split(":", 1)[1].strip()
    raise AssertionError("no VERDICT line")


def test_setting1_verdict_gates_on_partial_corr_not_or_gate() -> None:
    import pandas as pd
    ph = _load_phase13()

    # (a) genuine: narrow band, CI excludes 0.5 AND partial_corr significant.
    df_ok = pd.DataFrame([_band_row(
        auroc_epi_mu_ci_lo=0.60, partial_pass=1.0, partial_rho=0.4,
        partial_rho_p=0.01, gate_passed=1.0,
    )])
    assert _verdict_line(ph.setting1_verdict_text(df_ok)) == "ACCEPT"

    # (b) DIVERGENT (the fix): CI excludes 0.5 and the OR-gate fires, but ONLY via
    # the logistic channel (partial_pass=0). Must NOT be ACCEPT.
    df_div = pd.DataFrame([_band_row(
        auroc_epi_mu_ci_lo=0.60, partial_pass=0.0, logistic_pass=1.0,
        gate_passed=1.0, partial_rho=0.1, partial_rho_p=0.20,
    )])
    assert _verdict_line(ph.setting1_verdict_text(df_div)) != "ACCEPT"

    # (c) wide-signal-but-narrow-gone => REJECT (mu_hat residue branch).
    df_wide = pd.DataFrame([
        _band_row(group="asymmetric_lower_sweep", band_lo=0.3, band_hi=0.7,
                  auroc_epi_mu_ci_lo=0.60),
        _band_row(auroc_epi_mu_ci_lo=0.30, partial_pass=0.0),  # narrow fails
    ])
    assert _verdict_line(ph.setting1_verdict_text(df_wide)) == "REJECT"

    # (d) no scorable narrow band => INCONCLUSIVE.
    df_none = pd.DataFrame([
        _band_row(skipped=True, skip_reason="N<25 (N=3)"),
    ])
    assert _verdict_line(ph.setting1_verdict_text(df_none)) == "INCONCLUSIVE"


def test_setting2_verdict_collapsed_vs_accept() -> None:
    import pandas as pd
    ph = _load_phase13()

    def _cov_df(cb_all, cc_all):
        return pd.DataFrame([
            {"level": 0.95, "variant": "aleatoric_only", "coverage": cb_all,
             "mean_width": 0.5, "n": 100},
            {"level": 0.95, "variant": "aleatoric+epistemic", "coverage": cc_all,
             "mean_width": 0.5, "n": 100},
        ])

    def _terc_df(cb_high, cc_high):
        return pd.DataFrame([
            {"tercile": "high", "variant": "aleatoric_only", "level": 0.95,
             "n": 33, "mean_epi_mu": 0.01, "coverage": cb_high, "mean_width": 0.6},
            {"tercile": "high", "variant": "aleatoric+epistemic", "level": 0.95,
             "n": 33, "mean_epi_mu": 0.01, "coverage": cc_high, "mean_width": 0.65},
        ])

    n = 60
    mu = np.full(n, 0.5)
    # collapsed: epi is < 5% of total width even though (c) is "closer".
    sig_collapsed = {
        "aleatoric_U": np.full(n, 1.0), "epi_mu": np.full(n, 0.01), "mu_hat": mu,
    }
    v = ph.setting2_verdict_text(_cov_df(0.90, 0.93), _terc_df(0.99, 0.96), sig_collapsed)
    assert "NEGATIVE" in _verdict_line(v)

    # non-collapsed + (c) closer in high-epi and not worse overall => ACCEPT.
    sig_live = {
        "aleatoric_U": np.full(n, 0.9), "epi_mu": np.full(n, 0.1), "mu_hat": mu,
    }
    v2 = ph.setting2_verdict_text(_cov_df(0.90, 0.93), _terc_df(0.99, 0.96), sig_live)
    assert _verdict_line(v2) == "ACCEPT"


# ---------------------------------------------------------------------------
# Test 8 - base_err skip branch, collect_signals, _tercile_adaptivity
# ---------------------------------------------------------------------------


def test_band_skipped_when_base_err_out_of_range() -> None:
    ph = _load_phase13()
    n = 40
    sig = {
        "mu_hat": np.full(n, 0.52),
        "err": np.ones(n),            # base_err = 1.0 -> outside [0.1, 0.9]
        "epi_mu": np.random.default_rng(0).random(n),
        "epi_mc": np.random.default_rng(1).random(n),
        "mean_entropy": np.random.default_rng(2).random(n),
        "abs_ratio_err": np.random.default_rng(3).random(n),
    }
    row = ph.conditional_epi_band_report("g", (0.4, 0.6), sig, bootstrap_iters=10)
    assert row["skipped"] is True
    assert "outside" in row["skip_reason"]
    assert row["base_err_rate"] == 1.0


def test_tercile_adaptivity_strata_and_labels() -> None:
    ph = _load_phase13()
    n = 30
    epi = np.linspace(0.0, 1.0, n)          # clean terciles
    m = np.full(n, 10.0)
    K = np.full(n, 5.0)
    # (b) tight near m (misses K=5 everywhere); (c) wide (covers).
    intervals = {
        "b_lo_0.95": np.full(n, 8.0), "b_hi_0.95": np.full(n, 10.0),
        "c_lo_0.95": np.full(n, 0.0), "c_hi_0.95": np.full(n, 10.0),
    }
    df = ph._tercile_adaptivity(intervals, epi, K, m, 0.95)
    assert set(df["tercile"]) == {"low", "mid", "high"}
    # mean_epi_mu must increase low < mid < high (binning sanity).
    means = {t: float(df[df["tercile"] == t]["mean_epi_mu"].iloc[0])
             for t in ("low", "mid", "high")}
    assert means["low"] < means["mid"] < means["high"]
    # b/c labels not swapped: (c) covers (cov 1.0), (b) does not (cov 0.0).
    hi_b = df[(df["tercile"] == "high") & (df["variant"] == "aleatoric_only")]
    hi_c = df[(df["tercile"] == "high") & (df["variant"] == "aleatoric+epistemic")]
    assert float(hi_b["coverage"].iloc[0]) == 0.0
    assert float(hi_c["coverage"].iloc[0]) == 1.0


def test_collect_signals_contracts() -> None:
    ph = _load_phase13()
    k = 6
    predictor = _make_predictor(k, sigma_scale=0.2, seed=8)
    rng = np.random.default_rng(8)

    def _extract(rec, _params, _device):
        return rec["z"]

    records = []
    for _ in range(5):
        L = int(rng.integers(2, 6))
        m_j = int(rng.integers(2, 8))
        K_j = int(rng.integers(0, m_j + 1))
        T = L + 3
        records.append({
            "z": torch.randn(L, k),
            "m_j": m_j, "K_j": K_j,
            "token_range": (1, 1 + L),
            "entropy": torch.rand(T),
        })
    sig = ph.collect_signals(
        predictor, records, predictor.feature_params, torch.device("cpu"),
        _extract, mc_samples=16, seed=8,
    )
    n = len(records)
    for key in ("mu_hat", "epi_mu", "epi_mc", "aleatoric_U", "total_U",
                "mean_entropy", "K", "m", "U", "A", "err", "abs_ratio_err"):
        assert sig[key].shape == (n,), key
    assert np.all((sig["U"] >= 0.0) & (sig["U"] <= 1.0))
    assert np.all(np.isin(sig["A"], [0.0, 1.0]))
    assert np.allclose(sig["err"], 1.0 - sig["A"])
    assert np.allclose(sig["abs_ratio_err"], np.abs(sig["U"] - sig["mu_hat"]))
    assert np.all(sig["epi_mu"] >= 0.0)
