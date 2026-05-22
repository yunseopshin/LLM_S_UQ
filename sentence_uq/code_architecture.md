# `sentence_uq` — Code Architecture

Bayesian sentence-level factuality UQ from LLM hidden states. Reference doc for
the codebase under `/home/ys971217/LLM_S_UQ/sentence_uq`. Companion to
`README.md` (project intent) and `CLAUDE.md` (project rules).

---

## 1. Directory tree

```
sentence_uq/
├── README.md
├── requirements.txt
├── code_architecture.md                    # ← this file
├── configs/
│   ├── default.yaml                        # base config (Setup 2)
│   ├── pilot.yaml                          # 5-entity smoke test
│   ├── setup_1.yaml                        # cross-domain (Han reproduction)
│   ├── setup_2.yaml                        # in-domain FActScore-Bio (default)
│   └── setup_3.yaml                        # multi-domain LongFact
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                      # raw dataset prep + setup splits
│   │   ├── generation.py                   # LLM load + per-token hidden states
│   │   ├── sentence_split.py               # spaCy sentences + token ranges
│   │   └── annotation.py                   # atomic facts → (K_j, m_j)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py                    # ψ = (W, α, μ_0, log σ_0); z_ℓ
│   │   └── cached_scalars.py               # entropy H_ℓ, top-1 p^(1)_ℓ
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fisher_scoring.py               # damped Fisher MAP θ̂(ψ) (diff'ble)
│   │   ├── bayesian_main.py                # BayesianSentenceUQ + verify_local_pd
│   │   └── bayesian_aux.py                 # closed-form BayesianLogitRegression
│   ├── train/
│   │   ├── __init__.py                     # re-exports SentenceUQTrainer
│   │   └── trainer.py                      # bilevel outer-loop trainer
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py                      # Predictor + BatchPredictor + I/O
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── token_entropy.py                # mean per-token entropy
│   │   ├── semantic_entropy.py             # Kuhn et al. 2023 (NLI clustering)
│   │   ├── luq.py                          # Zhang et al. 2024 (per-sentence NLI)
│   │   ├── logistic_regression.py          # sklearn LR over ζ_j
│   │   └── factuality_probe.py             # Han et al. 2025 (L1 probe)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                      # ratio + strict + calibration
│   └── utils/
│       ├── __init__.py
│       ├── debug.py                        # gradient/PD/Fisher diagnostics
│       ├── io.py                           # (empty — reserved)
│       └── logging.py                      # (empty — reserved)
├── scripts/
│   ├── 00_prepare_dataset.py               # Phase 1-0
│   ├── 01_generate_data.py                 # Phase 1-1/1-2
│   ├── 01b_cache_scalars.py                # Phase 1-3
│   ├── 02_annotate_factuality.py           # Phase 1-4
│   ├── 03_train.py                         # Phase 4-1
│   ├── 04_train_aux.py                     # Phase 4-2
│   ├── 04_evaluate.py                      # Phase 6
│   ├── 05_baselines.py                     # Phase 5
│   ├── run_experiment.sh                   # setup-N orchestration
│   ├── run_pilot.sh                        # resumable pilot (5 entities)
│   └── run_full.sh                         # resumable full run
├── tests/                                  # 14 pytest modules (one per source)
└── data/
    ├── raw/{factscore_bio,longfact}/       # entities.json, prompts.json
    ├── splits/setup_{1,2,3}.json           # produced by Phase 1-0
    ├── generations/{factscore_bio,longfact}/   # per-prompt .pt (Phase 1-1)
    ├── cache/{factscore_bio,longfact}/     # {idx:05d}.pt entropy/top-1
    └── processed/{factscore_bio,longfact}/ # annotated.json + per-record JSON
```

---

## 2. File-by-file summary

### `configs/`
| File | Purpose |
|---|---|
| `default.yaml` | Base config: model, dataset dirs, generation/cache/processed paths, training hyperparams, `results_dir`. |
| `pilot.yaml` | 5-entity smoke test (overrides `max_new_tokens=256`, adds `pilot_size`). |
| `setup_{1,2,3}.yaml` | Inherit defaults, override `dataset.setup`, `dataset.split_file`, `results_dir`. |

### `src/data/`
| File | Role |
|---|---|
| `dataset.py` | Materialise FActScore-Bio entities + LongFact prompts; build the 3 train/val/test splits. |
| `generation.py` | Load HF causal LM (any model), apply chat template, manual generation loop, save per-prompt `.pt` with hidden states + logits. |
| `sentence_split.py` | spaCy sentences → per-sentence `(char_start, char_end, token_range)` via offset mapping. |
| `annotation.py` | GPT-4o-mini decompose → revise → subjectivity filter → Wikipedia retrieval → judge → `(K_j, m_j)`. |

### `src/features/`
| File | Role |
|---|---|
| `extractor.py` | `SentenceUQParams(nn.Module)` (ψ) + `z_ℓ = [W·Σ α_l h^(l), H_ℓ, p^(1)_ℓ]`. |
| `cached_scalars.py` | One-pass cache of `H_ℓ`, `p^(1)_ℓ` from stored logits (ψ-independent). |

### `src/models/`
| File | Role |
|---|---|
| `fisher_scoring.py` | Differentiable damped Fisher-scoring MAP for the binomial latent model. |
| `bayesian_main.py` | `BayesianSentenceUQ` outer-loop module wrapping ψ + Fisher MAP + binomial NLL. Plus `verify_local_pd`. |
| `bayesian_aux.py` | Closed-form `BayesianLogitRegression` over sentence-level ζ_j with logit-Gaussian target. |

### `src/train/`
| File | Role |
|---|---|
| `trainer.py` | `SentenceUQTrainer` — `prepare_data` / `train_epoch` / `evaluate` / `fit`; Adam over ψ. |

### `src/inference/`
| File | Role |
|---|---|
| `predict.py` | `Predictor` (4-level uncertainty + token attribution + MC), `BatchPredictor`, `save_trained_model`/`load_trained_model`. |

### `src/baselines/`
| File | Role |
|---|---|
| `token_entropy.py` | Mean cached entropy over sentence span. |
| `semantic_entropy.py` | Kuhn et al. 2023: sample, NLI-cluster, Shannon entropy. Shared `NLIScorer`. |
| `luq.py` | Zhang et al. 2024: per-sentence consistency via NLI vs samples. |
| `logistic_regression.py` | `LogisticRegressionBaseline` on layer-averaged ζ_j with strict/ratio targets. |
| `factuality_probe.py` | Han et al. 2025 L1 probe (original re-encode and adapted cached-states variants). |

### `src/evaluation/`
| File | Role |
|---|---|
| `metrics.py` | Ratio-level (MAE/RMSE/r/NLL), strict (AUROC/AUPRC/Brier), ECE, PRR, bootstrap CIs, reliability diagram, MC-vs-linear comparison, `full_evaluation` aggregator. |

### `src/utils/`
| File | Role |
|---|---|
| `debug.py` | `check_gradient_flow`, `visualize_feature_distribution`, `diagnose_fisher_scoring`, `sanity_check_boundary_fraction`, `check_m_j_distribution`. |
| `io.py`, `logging.py` | Empty placeholders. |

### `scripts/`
| File | Role |
|---|---|
| `00_prepare_dataset.py` | Phase 1-0 — calls `prepare_all_and_split(setup)`. |
| `01_generate_data.py` | Phase 1-1 — `load_model` + `batch_generate` per dataset of the chosen setup. |
| `01b_cache_scalars.py` | Phase 1-3 — `cache_scalars_for_directory` per dataset. |
| `02_annotate_factuality.py` | Phase 1-4 — sentence-split + `annotate_batch` driver. |
| `03_train.py` | Phase 4-1 — assemble `SentenceUQTrainer`, run `fit`, save trained model. |
| `04_train_aux.py` | Phase 4-2 — fit `BayesianLogitRegression` against offline `u_star`. |
| `04_evaluate.py` | Phase 6 — Predictor + baselines + ablations + plots + summary DataFrame. |
| `05_baselines.py` | Phase 5 — run all 5 baselines on the same split. |
| `run_experiment.sh` | One-shot chain `00 → 01 → 01b → 02 → 03 → 04` for a setup. |
| `run_pilot.sh` / `run_full.sh` | Resumable wrappers with per-phase stamps + logs (calls 00,01,01b,02,03,[04_aux],05,04_eval). |

### `tests/`
14 pytest modules — one per non-trivial source file. Tests parameterise over
multiple `(hidden_dim, num_layers)` tuples to enforce the model-agnostic rule.

---

## 3. Core class & function signatures

### `src/features/extractor.py`
```python
class SentenceUQParams(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, projection_dim: int = 64) -> None
    @property
    def feature_dim(self) -> int                         # = projection_dim + 2
    def get_Sigma_0(self) -> torch.Tensor                # (k, k)
    def get_Sigma_0_inv(self) -> torch.Tensor            # (k, k)

def extract_token_features(
    hidden_states: torch.Tensor,    # (T, num_layers, hidden_dim)
    entropy: torch.Tensor,          # (T,)
    top1_prob: torch.Tensor,        # (T,)
    params: SentenceUQParams,
) -> torch.Tensor                   # (T, k)

def extract_sentence_token_features(
    hidden_states, entropy, top1_prob,
    token_range: Tuple[int, int],
    params: SentenceUQParams,
) -> torch.Tensor                   # (L_j, k)

def extract_sentence_aggregate_feature(z_tokens: torch.Tensor) -> torch.Tensor  # (3k,)
```

### `src/features/cached_scalars.py`
```python
def compute_token_entropy_and_top1(logits: Tensor) -> tuple[Tensor, Tensor]    # (T,), (T,)
def cache_scalars_for_directory(generations_dir, cache_dir, *, progress=True) -> dict
def load_scalars(idx: int, cache_dir) -> dict
```

### `src/models/fisher_scoring.py`
```python
def _compute_grad_and_fisher(
    theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]              # grad (k,), H_fisher (k,k)

def _compute_clipped_objective(theta, ..., eps=1e-6) -> torch.Tensor          # scalar

def fisher_scoring_map(
    all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv,
    num_iters=15, eps=1e-6, lambda_init=1e-4, verbose=False,
) -> Tuple[torch.Tensor, torch.Tensor]              # θ̂ (k,), H_fisher_final (k,k)

def fisher_scoring_map_detached(...) -> Tuple[torch.Tensor, torch.Tensor]     # no_grad
```

### `src/models/bayesian_main.py`
```python
class BayesianSentenceUQ(nn.Module):
    def __init__(self, feature_params: SentenceUQParams, num_fisher_iters=10, eps=1e-6)
    def compute_map(
        self, all_z_tokens, all_K, all_m, differentiable=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]          # θ̂, H_fisher
    def compute_loss(self, all_z_tokens, all_K, all_m) -> torch.Tensor     # scalar binomial NLL
    def predict(self, z_tokens, m_j=None) -> Dict   # raises NotImplementedError (use Predictor)

def verify_local_pd(
    theta_hat, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6,
) -> Dict[str, float]
    # {"fisher_min_eig", "true_min_eig", "fisher_pd", "true_pd", "laplace_valid_local"}
```

### `src/models/bayesian_aux.py`
```python
def safe_logit(u: torch.Tensor, eps: float = 1e-3) -> torch.Tensor

class BayesianLogitRegression:
    def __init__(self, feature_dim: int, prior_mu=None, prior_sigma=1.0, noise_sigma=0.1)
    def fit(self, Z: torch.Tensor, U_star: torch.Tensor) -> "BayesianLogitRegression"
    def predict(self, z_new: torch.Tensor) -> Dict[str, torch.Tensor]
        # {"p_factual", "logit_mean", "logit_var", "epistemic_logit", "aleatoric_logit"}
    def estimate_noise_variance(self, Z, U_star) -> float
    def set_noise_sigma(self, noise_sigma: float) -> None
```

### `src/train/trainer.py`
```python
class SentenceUQTrainer:
    def __init__(self, model: BayesianSentenceUQ, lr=1e-3, num_epochs=50,
                 eval_every=1, pd_check_every=5, device="cpu",
                 log_fn=None, weight_decay=0.0)
    def prepare_data(
        self, split_file, generations_dirs, cache_dirs, processed_dirs=None,
    ) -> Dict[str, List[Dict[str, Any]]]            # {"train", "val", "test"}
    def train_epoch(self, train_data) -> Dict[str, float]                   # {"loss","num_sentences","num_positive"}
    def evaluate(self, train_data, eval_data) -> Dict[str, float]           # MAE/RMSE/Pearson_r/binomial_NLL/n
    def fit(self, train_data, val_data=None, test_data=None) -> Dict[str, Any]
        # history with train_loss, val_metrics, pd_checks, theta_hat, Sigma_hat, test_metrics
```

### `src/inference/predict.py`
```python
class Predictor:
    def __init__(self, theta_hat, Sigma_hat, feature_params, use_probit_shrinkage=False)
    def predict_sentence(self, z_tokens, m_j=None) -> Dict[str, Optional[Union[float, Tensor]]]
        # mu_hat, p_factual_probit, epi_mu, aleatoric_U, total_U,
        # epi_K, aleatoric_K, p_strict_factual, token_pi, token_attr, token_local_epi
    def predict_from_hidden_states(self, hidden_states, entropy, top1, token_range, m_j=None) -> Dict
    def predict_mc_epistemic(self, z_tokens, num_samples=100, m_j=None, generator=None) -> Dict

class BatchPredictor:
    def __init__(self, predictor: Predictor)
    def predict(self, all_z_tokens, all_m=None) -> List[Dict]

def save_trained_model(path, theta_hat, Sigma_hat, feature_params, extra=None) -> None
def load_trained_model(path, map_location="cpu") -> Dict[str, Any]
    # {"theta_hat", "Sigma_hat", "feature_params", "extra"}
```

### `src/data/dataset.py`
```python
SETUPS = (1, 2, 3); HAN_TEST_SIZE = 30; FACTSCORE_PROMPT_TEMPLATE = "Tell me a bio of {entity}."

def prepare_factscore_bio(save_dir="data/raw/factscore_bio", source_path=None) -> Path
def prepare_longfact_objects(save_dir="data/raw/longfact", source_dir=None) -> Path
def load_factscore_bio_entities(save_dir) -> list[dict]
def load_han_test_entities(save_dir) -> list[dict]
def load_longfact_topics(save_dir) -> list[str]
def load_longfact_prompts(save_dir) -> list[dict]
def create_split(dataset, setup, factscore_dir, longfact_dir, save_path=None, seed=42) -> dict
def prepare_all_and_split(setup, factscore_dir, longfact_dir, splits_dir, seed=42, force_redownload=False) -> dict
def split_save_filename(setup: int) -> str
def summarise_split(split) -> str
```

### `src/data/generation.py`
```python
DEFAULT_TARGET_LAYERS = 8

def auto_select_layers(num_hidden_layers, target_count=8) -> list[int]
def resolve_selected_layers(num_hidden_layers, selected_layers) -> list[int]
def load_model(model_name, device="cuda", dtype=torch.float16) -> tuple[model, tokenizer, model_info]
def make_prompt(item: dict) -> str
def apply_chat_template_if_available(tokenizer, user_prompt) -> str

@torch.no_grad()
def generate_with_hidden_states(
    model, tokenizer, prompt, selected_layers, max_new_tokens=512,
    temperature=0.7, top_p=1.0, do_sample=True, apply_chat_template=True,
    store_dtype=torch.float16,
) -> dict
    # {text, prompt, prompt_text, prompt_ids, token_ids, hidden_states (T,K,D fp16),
    #  logits (T,V fp16), selected_layers, finished}

def save_generation(record, out_path, model_info, selected_layers, dataset_tag, meta=None) -> Path
def batch_generate(items, model, tokenizer, model_info, selected_layers,
                   factscore_dir, longfact_dir, *,
                   max_new_tokens=512, temperature=0.7, top_p=1.0,
                   do_sample=True, skip_existing=True, progress=True) -> dict
def write_dataset_metadata(out_dir, model_info, selected_layers, generation_config,
                           dataset_tag, items) -> Path
```

### `src/data/sentence_split.py`
```python
def load_spacy_model(lang="en") -> spacy.Language
def split_into_sentences(text: str, nlp) -> list[dict]                       # text, char_start, char_end
def map_sentences_to_tokens(sentences, token_ids, tokenizer) -> list[tuple[int, int]]
def process_generation(generation_result: dict, tokenizer, nlp) -> dict
    # {"sentences": [{text, char_start, char_end, token_range}, ...]}
```

### `src/data/annotation.py`
```python
DEFAULT_TEMPERATURE = 0.0; DEFAULT_AUX_MODEL = "gpt-4o-mini"
MIN_SENTENCE_CHARS = 8; MIN_SENTENCE_WORDS = 3

class ApiClient(Protocol):
    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> str: ...

class OpenAIChatClient:
    def __init__(self, model="gpt-4o-mini", api_key=None, max_retries=5,
                 base_delay=1.5, request_timeout=60.0)
    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> str

class RateLimiter:
    def __init__(self, rps: float = 0.0); def wait(self) -> None

def is_meaningful_sentence(sentence: str) -> bool
def decompose_to_atomic_facts(sentence, entity_or_topic, api_client, *,
                               response_context=None, temperature=0.0,
                               revise=True, drop_subjective=True) -> list[str]
def judge_atomic_fact(fact, knowledge_context, api_client, *, temperature=0.0) -> int   # 0 | 1
def retrieve_knowledge(entity_or_topic, dataset_type, *, cache_dir=None,
                       max_chars=8000, timeout=20.0, extra_query=None) -> str
def annotate_sentence(sentence, entity_or_topic, dataset_type, api_client, *,
                      knowledge_context=None, response_context=None,
                      knowledge_cache_dir=None, rate_limiter=None,
                      temperature=0.0, extra_query=None) -> dict        # {sentence, m_j, K_j, claims}
def annotate_record(record, dataset_type, api_client, *, knowledge_cache_dir=None,
                    rate_limiter=None, temperature=0.0) -> dict
def annotate_batch(processed_data, dataset_type, api_client, *, out_dir,
                   knowledge_cache_dir=None, rate_limiter=None,
                   temperature=0.0, progress=True, write_combined=True) -> dict
```

### `src/evaluation/metrics.py`
```python
def compute_ratio_level_metrics(U_true, mu_hat, m_j=None) -> Dict[str, float]
    # {MAE, RMSE, Pearson_r, [binomial_NLL]}
def compute_strict_factuality_metrics(A_true, p_strict, uncertainty) -> Dict[str, float]
    # {AUROC, AUPRC, Brier, ECE}
def compute_calibration_metrics(y_true, p_pred, n_bins=10) -> Dict[str, float]
    # {Brier, ECE}
def compute_prr(y_true, uncertainty, num_thresholds=100) -> Dict[str, Any]
    # {rejection_rates, remaining_quality, prr_auc}
def compute_bootstrapped_ci(y_true, scores, metric_fn,
                            n_bootstrap=1000, alpha=0.05, seed=None) -> Dict[str, float]
def plot_reliability_diagram(y_true, p_pred, n_bins=10, save_path=None, title="") -> Figure
def compare_mc_vs_linear_epistemic(predictor, test_sentences,
                                   num_mc_samples=100, generator=None) -> Dict[str, Any]
def full_evaluation(predictions, K_true, m_true, uncertainties) -> pandas.DataFrame
    # columns ["metric", "tier", "value"]
```

### `src/baselines/`
```python
# token_entropy.py
def compute_token_entropy_baseline(entropy: Tensor, token_range) -> float

# semantic_entropy.py
DEFAULT_NLI_MODEL = "microsoft/deberta-large-mnli"
class NLIScorer:
    def __init__(self, model_name=DEFAULT_NLI_MODEL, device="cuda", dtype=torch.float16)
    def entailment_prob(self, premises, hypotheses) -> Tensor
    def predict_label(self, premises, hypotheses) -> List[int]
def generate_semantic_samples(prompt, model, tokenizer, *, num_samples=10,
                              temperature=1.0, top_p=1.0, max_new_tokens=256) -> List[str]
def cluster_by_entailment(samples, nli_scorer) -> List[int]
def compute_semantic_entropy_from_samples(samples, nli_scorer) -> float
def compute_semantic_entropy(prompt, model, tokenizer, nli_scorer, num_samples=10, ...) -> float

# luq.py
def generate_luq_samples(prompt, model, tokenizer, *, num_samples=10, ...) -> List[str]
def compute_luq_for_sentences(sentences, samples, nli_scorer) -> List[float]
def compute_luq(prompt, model, tokenizer, nli_model, num_samples=10, *,
                sentences=None, samples=None, ...) -> List[float]

# logistic_regression.py
def build_sentence_features(hidden_states, entropy, top1, token_range) -> Tensor    # (D+2,)
class LogisticRegressionBaseline:
    def __init__(self, target="strict", C=1.0, max_iter=1000, random_state=0)
    def fit(self, Z, K, m) -> "LogisticRegressionBaseline"
    def predict_proba(self, Z) -> Tensor
def collate_sentence_features(sentence_records) -> Dict[str, Tensor]      # {Z, K, m}

# factuality_probe.py
DEFAULT_TARGET_LAYER = 14
def pick_layer_index(target_layer, selected_layers) -> int
def extract_adapted_features(hidden_states, token_range, layer_index) -> Tensor
def extract_original_features(claim_texts, model, tokenizer,
                              target_layer=14, *, batch_size=8) -> Tensor
class FactualityProbeBaseline:
    def __init__(self, variant="original", target_layer=14, C=1.0, max_iter=1000, random_state=0)
    def build_adapted_dataset(self, sentence_records, selected_layers) -> Dict     # {H, A, m}
    def build_original_dataset(self, sentence_records, model, tokenizer) -> Dict   # {H, y, sentence_to_claims, sentence_records}
    def fit(self, H, y) -> "FactualityProbeBaseline"
    def predict_proba(self, H) -> Tensor
    def aggregate_sentence_scores(self, claim_probs, sentence_to_claims, agg="mean") -> Tensor
```

### `src/utils/debug.py`
```python
def check_gradient_flow(loss, params: SentenceUQParams) -> Dict[str, Optional[float]]
def visualize_feature_distribution(feature_params, sample_hidden_states, save_path=None) -> Figure
def diagnose_fisher_scoring(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv,
                             eps=1e-6, num_iters=15, lambda_init=1e-4) -> Dict[str, Any]
def sanity_check_boundary_fraction(...) -> Dict[str, Any]
def check_m_j_distribution(all_m: torch.Tensor) -> Dict[str, Any]
```

---

## 4. Inter-module import graph

Only the directed edges that cross module boundaries are shown
(`__init__.py` re-exports and third-party imports omitted).

```
src.features.extractor          (leaf — only torch)
src.features.cached_scalars     (leaf — only torch)

src.models.fisher_scoring       (leaf — only torch)

src.models.bayesian_main
    ← src.features.extractor          (SentenceUQParams)
    ← src.models.fisher_scoring       (_compute_grad_and_fisher,
                                       _compute_clipped_objective,
                                       fisher_scoring_map,
                                       fisher_scoring_map_detached)

src.models.bayesian_aux         (leaf — only torch; receives ζ_j from caller)

src.train.trainer
    ← src.features.extractor          (extract_sentence_token_features)
    ← src.models.bayesian_main        (BayesianSentenceUQ, verify_local_pd)
src.train.__init__              (re-exports SentenceUQTrainer)

src.inference.predict
    ← src.features.extractor          (SentenceUQParams,
                                       extract_sentence_token_features)

src.utils.debug
    ← src.features.extractor          (SentenceUQParams)
    ← src.models.fisher_scoring       (_compute_clipped_objective,
                                       _compute_grad_and_fisher)

src.baselines.token_entropy     (leaf)
src.baselines.semantic_entropy  (leaf — NLI model + LM sampling)
src.baselines.luq
    ← src.baselines.semantic_entropy  (NLIScorer, generate_semantic_samples)
src.baselines.logistic_regression  (leaf — sklearn + torch)
src.baselines.factuality_probe  (leaf — sklearn + torch)

src.data.dataset                (leaf)
src.data.generation             (leaf — transformers loaded lazily inside load_model)
src.data.sentence_split         (leaf — spaCy loaded lazily)
src.data.annotation             (leaf — optional openai + urllib for Wikipedia)

src.evaluation.metrics          (leaf — numpy, pandas, sklearn.metrics, torch)
```

Script-level imports (each script also `sys.path`s the project root):

```
scripts/00_prepare_dataset.py   ← src.data.dataset
scripts/01_generate_data.py     ← src.data.dataset, src.data.generation
scripts/01b_cache_scalars.py    ← src.features.cached_scalars
scripts/02_annotate_factuality.py
    ← src.data.annotation, src.data.sentence_split
scripts/03_train.py
    ← src.data.dataset, src.features.extractor, src.inference.predict,
      src.models.bayesian_main, src.train.trainer
scripts/04_train_aux.py
    ← src.data.dataset, src.features.extractor, src.inference.predict,
      src.models.bayesian_aux, src.train.trainer
scripts/04_evaluate.py
    ← src.data.dataset, src.evaluation.metrics, src.features.extractor,
      src.inference.predict, src.models.bayesian_main, src.train.trainer
scripts/05_baselines.py
    ← src.baselines.{factuality_probe, logistic_regression, token_entropy,
                      semantic_entropy?, luq?},
      src.data.dataset, src.train.trainer
```

There are no circular imports — `bayesian_main` depends on `features.extractor`
and `models.fisher_scoring`; `trainer` is the only place that pulls in
`bayesian_main`, and `predict` only depends on `features.extractor`.

---

## 5. Config / YAML structure

All configs share the same schema. `setup_{N}.yaml` inherit from `default.yaml`
*by convention* — the scripts merge user-supplied overrides on top of defaults
(`pilot.yaml` and the per-setup files only carry the overrides).

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct   # any HF AutoModelForCausalLM
  selected_layers: null                       # null → auto-pick ~8 evenly spaced
  dtype: bfloat16

dataset:
  setup: 1 | 2 | 3                            # selects split builder
  factscore_bio_dir: data/raw/factscore_bio
  longfact_dir: data/raw/longfact
  splits_dir: data/splits
  split_file: null                            # null → data/splits/setup_{N}.json
  seed: 42
  pilot_size: 5                               # (pilot.yaml only)

generation:
  factscore_bio_dir: data/generations/factscore_bio
  longfact_dir: data/generations/longfact
  max_new_tokens: 512                         # 256 in pilot
  temperature: 0.7
  top_p: 1.0
  batch_size: 1

cache:
  factscore_bio_dir: data/cache/factscore_bio
  longfact_dir: data/cache/longfact

processed:
  factscore_bio_dir: data/processed/factscore_bio
  longfact_dir: data/processed/longfact

# === Feature extractor (Phase 2-1) ===
projection_dim: 64                            # p; feature_dim = p + 2 = 66

# === Training (Phase 4-1) ===
num_epochs: 50
lr: 1.0e-3
num_fisher_iters: 10
prior_sigma_init: 1.0
batch_size: null                              # full-batch when null
eval_every: 1
pd_check_every: 5

results_dir: results/setup_2
```

Notes:
- `model.name`, `model.dtype`, `model.selected_layers` drive `load_model` /
  `resolve_selected_layers`. `hidden_size`, `num_hidden_layers`, `vocab_size`
  are *read from `model.config` at runtime* and propagated into both
  `SentenceUQParams(hidden_dim=..., num_layers=...)` and the saved per-prompt
  `.pt` `model_config` block (see `CLAUDE.md` § Model Compatibility).
- `dataset.setup` selects which subset/split logic runs (table in §6).
- Generation/cache/processed dirs are split by dataset tag so Setup 1 and
  Setup 2 share the FActScore-Bio generations on disk.

---

## 6. Data flow / pipeline order

Below `→` is "writes / produces", read top to bottom. Each phase has an
explicit on-disk artifact, so re-runs resume from the first missing one.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 1-0  scripts/00_prepare_dataset.py                                  │
│   prepare_factscore_bio()          → data/raw/factscore_bio/entities.json │
│                                       + test_entities_han.json (30)      │
│   prepare_longfact_objects()       → data/raw/longfact/{prompts,topics}.json │
│   create_split(setup, seed)        → data/splits/setup_{N}.json          │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 1-1 / 1-2  scripts/01_generate_data.py                              │
│   load_model(model.name)           → model + tokenizer + model_info       │
│   resolve_selected_layers(...)     → list[int]                            │
│   for item in split:                                                      │
│     generate_with_hidden_states()  → {text, token_ids,                   │
│                                       hidden_states (T,K,D fp16),         │
│                                       logits (T,V fp16), ...}             │
│     save_generation()              → data/generations/{ds}/{name}.pt      │
│   process_generation() (sentence_split.py, inside script or Phase 2):     │
│                                    → sentences[{text, char_start,         │
│                                                  char_end, token_range}]  │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 1-3  scripts/01b_cache_scalars.py                                   │
│   cache_scalars_for_directory()    → data/cache/{ds}/{idx:05d}.pt         │
│     payload: {entropy (T,) fp32, top1_prob (T,) fp32, token_ids, source}  │
│   (Indexing is the sorted-rglob position of the source .pt file.)         │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 1-4  scripts/02_annotate_factuality.py                              │
│   Sentence split (Phase 1-2) per generation .pt:                          │
│     load_spacy_model() + process_generation(record, tok, nlp)             │
│   annotate_batch(processed, dataset_type, OpenAIChatClient):              │
│     decompose → revise → subjectivity → retrieve_knowledge(Wikipedia)     │
│     → judge_atomic_fact for each surviving claim                          │
│     → {m_j, K_j, claims[{text, label}]}                                   │
│   Writes data/processed/{ds}/{record}.json + .../annotated.json           │
│   Knowledge cache:        data/processed/{ds}/knowledge/*.txt             │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 4-1  scripts/03_train.py                                            │
│   SentenceUQTrainer.prepare_data(split, gen_dirs, cache_dirs, processed)  │
│     → per-sentence records {dataset, source_id, token_range, K_j, m_j,    │
│        hidden_states, entropy, top1}  for train / val / test              │
│   feature_params = SentenceUQParams(hidden_dim, num_layers, p=64)         │
│   model = BayesianSentenceUQ(feature_params, num_fisher_iters)            │
│   trainer.fit(train, val, test):                                          │
│     outer (Adam over ψ) wraps inner Fisher-scoring MAP θ̂(ψ)               │
│     periodic verify_local_pd                                              │
│   save_trained_model() → results/{setup}/trained_model.pt                 │
│     {theta_hat (k,), Sigma_hat (k,k), feature_params_state_dict, cfg}     │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  (optional, opportunistic)
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 4-2  scripts/04_train_aux.py                                        │
│   Inputs:  trained_model.pt + data/processed/u_star*.{json,pt}            │
│   Build ζ_j via extract_sentence_aggregate_feature (R^{3k})               │
│   BayesianLogitRegression(feature_dim=3k).fit(Z, U_star)                  │
│     → results/{setup}/aux_model.pt                                        │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 5  scripts/05_baselines.py                                          │
│   Operates on the same split + cached tensors:                            │
│     - compute_token_entropy_baseline                                       │
│     - LogisticRegressionBaseline (ζ_j → A_j or U_j)                        │
│     - FactualityProbeBaseline (original re-encode + adapted)               │
│     - NLIScorer-backed semantic_entropy + luq (prompt-level samples)       │
│   Writes results/{setup}/baselines.json                                    │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 6  scripts/04_evaluate.py                                           │
│   load_trained_model → Predictor                                           │
│   For every test sentence:                                                 │
│     z_j = extract_sentence_token_features(hidden_states, entropy, top1)    │
│     Predictor.predict_sentence(z_j, m_j)                                   │
│       → mu_hat, epi_mu, total_U, aleatoric_U, p_strict_factual, token_*    │
│   Compute ratio + strict + ECE + PRR + bootstrap CIs + reliability diagrams│
│   Ablations: binom vs Bernoulli, Bayesian vs point, MC vs linear           │
│   Compare with baselines.json                                              │
│   Outputs (per setup):                                                     │
│     results/{setup}/metrics.{json,csv}                                     │
│     results/{setup}/plots/{prr,reliability,alpha,token_heatmap,...}        │
│     results/{setup}/summary.md (when --compare-all)                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Tensor schema reference

| Producer | Tensor / file | Shape | Dtype | Notes |
|---|---|---|---|---|
| `generate_with_hidden_states` | `hidden_states` | `(T, K, D)` | fp16 | `K=len(selected_layers)`, `D=hidden_size` |
| same | `logits` | `(T, V)` | fp16 | `V=vocab_size` |
| same | `token_ids` | `(T,)` | int64 | post-prompt tokens only |
| `compute_token_entropy_and_top1` | `entropy`, `top1_prob` | `(T,)` | fp32 | nats |
| `extract_token_features` | `z` | `(T, k)` | fp32 | `k = projection_dim + 2` (66 by default) |
| `extract_sentence_token_features` | `z_j` | `(L_j, k)` | fp32 | `L_j = end - start` |
| `extract_sentence_aggregate_feature` | `ζ_j` | `(3k,)` | fp32 | mean ⊕ std ⊕ last |
| `BayesianSentenceUQ.compute_map` | `θ̂`, `H_fisher` | `(k,)`, `(k,k)` | fp32 | differentiable w.r.t. ψ |
| `save_trained_model` payload | `theta_hat`, `Sigma_hat`, `feature_params_state_dict` | — | fp32 (CPU) | `Sigma_hat = H_fisher⁻¹` via `_safe_inverse` |
| `Predictor.predict_sentence` | `token_pi`, `token_attr`, `token_local_epi` | `(L_j,)` | fp32 | token-level attribution + local epistemic |

### `m_j = 0` policy
Sentences with `m_j = 0` survive Phase 1-4 (they may be boilerplate/opinion)
but are **skipped** by `BayesianSentenceUQ.compute_loss`, the Fisher
gradient/Hessian, all evaluation metrics (`full_evaluation` does the mask),
and the trainer's `_ratio_metrics`. The annotator does *not* delete them so the
sentence index → token range mapping stays stable across phases (CLAUDE.md
rule 8).
