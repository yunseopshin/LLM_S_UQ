# Phase 10-1 — Beta-Binomial Observation Model (Switchable, Peer to Binomial)

Add a Beta-Binomial observation model as a **config-selectable peer** to the existing
Binomial model. This is an empirical validation step: we want to run Binomial and
Beta-Binomial side by side, on identical data and identical code paths, so that any
difference in results (epistemic statistics, ECE, AUROC) is attributable solely to
the likelihood.

**Prerequisite**: Phases 1-1 through 7-3 implemented; tests passing. Full run artifacts
(`N_train=1638`, `N_test=353`) available.

---

## 0. Hard Constraints (read first)

1. **Do NOT delete or change the behavior of the Binomial path.** All existing functions,
   their default arguments, and their numerical outputs must remain bit-identical. The
   existing `tests/test_fisher_scoring.py` and `tests/test_decomposition.py` must pass
   **unchanged**.
2. **Binomial and Beta-Binomial are peers selected by a single config switch**
   (`model.likelihood: binomial | beta_binomial`). They share the inner-loop, posterior,
   trainer, and prediction code; only the per-sentence scalars differ.
3. The Beta-Binomial **must reduce to the Binomial in the limit** $\phi \to \infty$. A test
   pins this.
4. English only, plain-ASCII math inside code/docstrings (`mu_j`, `g_j`, `phi`), no Unicode
   combining marks.

---

## 1. Model (mean-dispersion parameterization)

Per sentence $j$, introduce a latent rate and a global concentration $\phi > 0$:

$$p_j \sim \mathrm{Beta}(a_j, b_j), \quad a_j = \phi\,\mu_j(\theta), \quad b_j = \phi\,(1-\mu_j(\theta)), \qquad K_j \mid p_j \sim \mathrm{Binomial}(m_j, p_j).$$

The mean structure is **unchanged**:

$$\mu_j(\theta) = \frac{1}{L_j}\sum_{\ell \in s_j} \sigma(\theta^\top z_\ell), \qquad g_j = \frac{\partial \mu_j}{\partial \theta} = \frac{1}{L_j}\sum_{\ell \in s_j}\pi_\ell(1-\pi_\ell)\,z_\ell.$$

Marginal of $K_j$ is Beta-Binomial. The key moments (with $\rho := 1/(\phi+1)$):

$$\mathbb{E}[U_j \mid \theta] = \mu_j, \qquad \mathrm{Var}[U_j \mid \theta] = \frac{\mu_j(1-\mu_j)}{m_j}\big[\,1 + (m_j - 1)\rho\,\big].$$

The mean equals the Binomial mean; only the conditional variance gains the overdispersion
factor $1 + (m_j - 1)\rho$. As $\phi \to \infty$ ($\rho \to 0$) this collapses to the Binomial.

**Placement of $\phi$**: $\phi$ is a **global scalar** in the outer parameter set
$\psi = \{W, \alpha, \mu_0, \log\sigma_0, \log\phi\}$, learned by the outer Adam loop. The
inner Laplace posterior remains over $\theta$ **only** — its dimension $k$, $\hat{\Sigma}$,
and the epistemic term are all unchanged. From the inner loop's perspective $\phi$ is a
known constant evaluated at the current $\psi$.

---

## 2. Per-sentence scalars (digamma / trigamma)

Let $\Psi_0 = $ `torch.digamma`, $\Psi_1 = $ `torch.polygamma(1, .)`, and clamp
$\mu \in [\epsilon, 1-\epsilon]$ before any gamma call.

**Score scalar** $r_j = \partial \log P_j / \partial \mu_j$ (the residual that multiplies $g_j$):

$$r_j^{BB} = \phi\big[\,\Psi_0(K_j + \phi\mu_j) - \Psi_0(\phi\mu_j) - \Psi_0(m_j - K_j + \phi(1-\mu_j)) + \Psi_0(\phi(1-\mu_j))\,\big].$$

**Fisher weight** $w_j$ (the scalar that multiplies $g_j g_j^\top$ in the precision). Use the
**observed information** $I_j^\mu = -\partial^2 \log P_j / \partial \mu_j^2$, which is closed-form
for the Beta-Binomial (the expected Fisher has no closed form here):

$$w_j^{BB} = \phi^2\big[\,\Psi_1(\phi\mu_j) - \Psi_1(K_j + \phi\mu_j) + \Psi_1(\phi(1-\mu_j)) - \Psi_1(m_j - K_j + \phi(1-\mu_j))\,\big].$$

This is $> 0$ because $\Psi_1$ is positive and strictly decreasing. **Design note to record in
the docstring**: the Binomial path uses the *expected* Fisher weight $m_j/(\mu_j(1-\mu_j))$;
the Beta-Binomial uses *observed* information because no closed-form expected Fisher exists.
The two coincide in the $\phi\to\infty$ limit at the optimum (where $K_j \approx m_j\mu_j$).

**Negative log-pmf** (for the line-search objective and the outer NLL), with
$\log B(x,y) = \mathrm{lgamma}(x) + \mathrm{lgamma}(y) - \mathrm{lgamma}(x+y)$:

$$-\log P_j = -\Big[\log\tbinom{m_j}{K_j} + \log B\big(K_j+\phi\mu_j,\; m_j-K_j+\phi(1-\mu_j)\big) - \log B\big(\phi\mu_j,\; \phi(1-\mu_j)\big)\Big].$$

**Strict target** $A_j = \mathbf{1}[K_j = m_j]$:

$$\log P(A_j=1) = \mathrm{lgamma}(m_j + \phi\mu_j) - \mathrm{lgamma}(\phi\mu_j) + \mathrm{lgamma}(\phi) - \mathrm{lgamma}(m_j + \phi).$$

**$\phi \to \infty$ limits (must hold, used in tests)**: $r_j^{BB} \to (K_j - m_j\mu_j)/(\mu_j(1-\mu_j))$,
$\;w_j^{BB} \to K_j/\mu_j^2 + (m_j-K_j)/(1-\mu_j)^2$ (equals $m_j/(\mu_j(1-\mu_j))$ at the optimum),
$\;-\log P_j \to$ Binomial NLL, $\;\log P(A_j=1) \to m_j \log \mu_j$.

`m_j = 0` sentences are skipped, exactly as in the Binomial path.

---

## 3. File-by-file requirements

### 3.1 NEW: `src/models/likelihood.py`

Define a small interface and the two peer implementations. This is the only place the two
models differ.

```python
class Likelihood:
    """Per-sentence scalar interface shared by the inner loop, the outer NLL,
    and the predictive decomposition. All methods operate elementwise on
    1-D tensors of clamped mu, integer K, integer m. mu is assumed already
    clamped to [eps, 1-eps] by the caller for the gamma-based variant."""
    def score_mu(self, K, m, mu): ...        # r_j  = d log P / d mu
    def fisher_weight(self, K, m, mu): ...    # w_j  (scalar on g g^T)
    def neg_log_pmf(self, K, m, mu): ...      # -log P_j  (sum-ready, per sentence)
    def log_prob_all_correct(self, m, mu): ...# log P(A_j = 1)
    @property
    def rho(self): ...                        # within-sentence correlation for decomposition


class BinomialLikelihood(Likelihood):
    """Reproduces the CURRENT epsilon-stabilized Binomial formulas EXACTLY.
    Copy the existing expressions from fisher_scoring.py verbatim:
      score_mu      = (K - m*mu) / clamp(mu*(1-mu), min=eps)
      fisher_weight = m / clamp(mu*(1-mu), min=eps)
      neg_log_pmf   = current binomial NLL (with the same log C(m,K) handling)
      log_prob_all_correct = m * log(mu)
      rho           = 0.0
    """

class BetaBinomialLikelihood(Likelihood):
    """phi is a live tensor (params.phi) so gradients flow to log_phi.
    Implements the Section 2 formulas with torch.digamma / polygamma(1,.) / lgamma.
      rho = 1.0 / (phi + 1.0)
    """
```

- `eps` matches the inner-loop default (`1e-6`).
- All methods must be autograd-friendly (no `.detach()`, no in-place on tracked tensors), so
  the unrolled inner loop and the outer NLL both backprop into `log_phi`.

### 3.2 `src/features/extractor.py` — add gated `log_phi`

- `SentenceUQParams.__init__` gains `likelihood: str = "binomial"`, `phi_init: float = 50.0`,
  `learn_phi: bool = True`.
- When `likelihood == "beta_binomial"`: register `self.log_phi = nn.Parameter(torch.tensor(math.log(phi_init)), requires_grad=learn_phi)`. Expose `@property phi -> exp(log_phi)`.
- When `likelihood == "binomial"`: do **not** register `log_phi` (keep the parameter set and
  `state_dict` identical to today). `phi` property raises or returns `None`.
- Do not touch `extract_token_features`, `get_Sigma_0(_inv)`, or feature dims.

### 3.3 `src/models/fisher_scoring.py` — route scalars through a likelihood

- Add a keyword arg `likelihood: Likelihood = None` to `_compute_grad_and_fisher`,
  `_compute_clipped_objective`, `fisher_scoring_map`, and `fisher_scoring_map_detached`.
  `None` constructs a `BinomialLikelihood()` internally so **every existing call site and test
  is byte-for-byte unchanged**.
- Replace the three hard-coded binomial expressions with:
  - residual: `R = likelihood.score_mu(K, m, mu)`
  - precision weight: `w = likelihood.fisher_weight(K, m, mu)`  (used as `w * g g^T`)
  - line-search objective: `-sum(likelihood.neg_log_pmf(K, m, mu)) - prior_penalty`
- Keep the existing `_last_diagnostics` (boundary_fraction etc.) working. Add `phi` and `rho`
  to it when a Beta-Binomial likelihood is in use.
- The damped update, adaptive `lambda`, `m_j = 0` skip, and differentiability requirements are
  unchanged.

### 3.4 `src/models/bayesian_main.py` — construct the likelihood from config

- Read `model.likelihood` from config. Build the likelihood object each forward pass so that
  `BetaBinomialLikelihood(phi=params.phi, eps=...)` carries a live gradient.
- Pass `likelihood=` into the inner loop and into the outer NLL.
- $\hat{\Sigma} = (H_{\text{final}})^{-1}$ is unchanged.

### 3.5 `src/inference/predict.py` — decomposition with overdispersion factor

Pull `rho` from the likelihood (Binomial -> `rho = 0`, so the formulas reduce exactly to today).
Let `f = 1 + (m_star - 1) * rho`.

- Epistemic (unchanged): `Epi_mu = g_hat @ Sigma_hat @ g_hat`.
- Ratio aleatoric: `Alea_U = f / m_star * max(0, mu_hat*(1-mu_hat) - Epi_mu)`.
- Count aleatoric: `Alea_K = m_star * f * max(0, mu_hat*(1-mu_hat) - Epi_mu)`.
- Count epistemic (unchanged): `Epi_K = m_star**2 * Epi_mu`.
- Totals: `Total_U = Alea_U + Epi_mu`, `Total_K = Alea_K + Epi_K`.
- Strict: `P(A_star = 1) = exp(likelihood.log_prob_all_correct(m_star, mu_hat))`
  (replaces the plug-in `mu_hat ** m_star`; for Binomial it equals `mu_hat ** m_star`).

Confirm with a unit test that `rho = 0` reproduces the current numbers exactly.

### 3.6 `src/train/trainer.py` — include `log_phi` in the outer optimizer

- When `model.likelihood == "beta_binomial"` and `learn_phi`, add `params.log_phi` to the Adam
  parameter group (same lr group is fine; expose `optim.phi_lr` if a separate lr is desired).
- Outer NLL uses the likelihood's `neg_log_pmf` (Beta-Binomial NLL with `lgamma`), so
  `log_phi` gets gradient both directly (via the NLL) and indirectly (through the unrolled
  `theta_hat(psi)`).
- Log `phi_hat` and `rho_hat = 1/(phi_hat+1)` each epoch.

### 3.7 `configs/` — the switch

Add to the model block (defaults preserve current behavior):

```yaml
model:
  likelihood: binomial      # binomial | beta_binomial
  phi_init: 50.0            # beta_binomial only: initial concentration (near-Binomial)
  learn_phi: true           # beta_binomial only: if false, phi fixed at phi_init
```

Create `configs/setup_2_betabinomial.yaml` as a thin override of `configs/setup_2.yaml`
with `likelihood: beta_binomial`. The existing `configs/setup_2.yaml` is its Binomial peer.
Do the same for setups 1 and 3 if cheap.

---

## 4. Pre-flight diagnostics (run before trusting any fitted phi)

Add a helper `src/utils/dispersion.py` (or a notebook cell) that, given `all_K, all_m`:

1. Reports the distribution of `m_j` — especially `frac(m_j == 0)`, `frac(m_j == 1)`,
   `frac(m_j >= 2)`. **`phi` is only identified from `m_j >= 2` sentences**; if that fraction
   is small, `phi` is weakly identified and the Beta-Binomial may not move.
2. Computes a method-of-moments estimate of `rho` (pooled observed overdispersion of
   `U_j = K_j/m_j` relative to the Binomial `mu(1-mu)/m_j`, restricted to `m_j >= 2`), as an
   independent sanity check on the learned `phi_hat`.

Print these before training so we know up front whether overdispersion is even present.

---

## 5. Tests (`tests/test_beta_binomial.py`)

1. **Binomial regression**: `BinomialLikelihood` injected into the inner loop yields outputs
   identical (atol=0 / exact) to the no-`likelihood` path. Existing `test_fisher_scoring.py`
   and `test_decomposition.py` still pass unchanged.
2. **phi -> infinity limit**: with `phi = 1e6`, `BetaBinomialLikelihood.score_mu`,
   `fisher_weight`, `neg_log_pmf`, and `log_prob_all_correct` match `BinomialLikelihood`
   within `atol=1e-3` on random `(K, m, mu)` with `m in {1..8}`.
3. **Decomposition reduction**: `predict` with `rho = 0` reproduces current numbers; with
   `rho > 0`, `Alea_U` scales by `1 + (m-1)*rho` and all terms stay non-negative.
4. **Gradient flow**: `torch.autograd.gradcheck` on the Beta-Binomial scalars w.r.t. `mu` and
   `phi`; and an end-to-end check that one outer step produces a finite, non-zero gradient on
   `log_phi` through `fisher_scoring_map`.
5. **Positivity**: `fisher_weight > 0` for random valid `(K, m, mu, phi)`.
6. **m_j = 0 skip**: unaffected.

---

## 6. Acceptance / Go-No-Go

After a Beta-Binomial run on Setup 2 (peer to the existing Binomial run), report a single
comparison table:

| metric | binomial | beta_binomial |
|---|---|---|
| fitted `phi_hat` / `rho_hat` | (n/a) | ... |
| saturation fraction (`pi_l` extreme) | ... | ... |
| mean `\|g_hat\|` | ... | ... |
| mean `Epi_mu` | ... | ... |
| ratio-level ECE | ... | ... |
| strict-level AUROC | ... | ... |

**Expectation to state honestly in the writeup**: the Beta-Binomial does not change the
*form* of `Epi_mu = g_hat^T Sigma_hat g_hat`, so it does not fix the epistemic collapse
directly. The hypothesized mechanism is indirect: if the data support overdispersion
(`phi_hat` small / `rho_hat` non-trivial), the likelihood no longer needs to push `mu_j` to
the boundary to match all-correct / all-wrong sentences, so `pi_l` saturation drops and
`g_hat` recovers. The effect is real only if `phi_hat` is meaningfully finite. If `phi_hat`
stays large, the Beta-Binomial is correctly telling us overdispersion is absent, and we should
look elsewhere (token-saliency pooling, focal/calibration loss).

Keep the existing `logit L2 penalty`; if it interacts with the new likelihood, re-tune the
`lambda` grid for the Beta-Binomial run rather than removing it.

---

## 7. Out of scope (do not touch)

- `src/models/bayesian_aux.py` (logit-Gaussian auxiliary model).
- Prior hyperparameter treatment of `mu_0` / `log_sigma_0` (separate decision).
- Per-sentence `phi_j` regression — the interface above already accepts a per-sentence `phi`
  tensor, so a future dispersion head can fill it via the outer loop without further inner-loop
  changes, but that is a later phase.
