#!/usr/bin/env bash
# setup_prompts.sh
#
# Claude Code에서 사용할 수 있도록 각 Phase별 prompt를 파일로 분리.
# 사용법:
#   chmod +x setup_prompts.sh
#   ./setup_prompts.sh
# 실행 후 prompts/ 디렉토리에 각 prompt가 파일로 저장됨.

set -e

PROMPTS_DIR="prompts"
mkdir -p "$PROMPTS_DIR"

# ============================================================
# CLAUDE.md (Claude Code가 자동으로 읽는 프로젝트 컨텍스트)
# ============================================================
cat > CLAUDE.md << 'EOF'
# Project: Sentence-Level Bayesian Uncertainty Quantification for LLMs

## 연구 목표
LLM(Llama-3-8B-Instruct)의 hidden state에서 Bayesian posterior를 구성하여
문장별 factuality uncertainty를 single forward pass로 추정한다.
기존 sampling-based 방법(LUQ, Semantic Entropy)은 10× 생성이 필요한데,
우리는 1×로 comparable한 성능을 목표로 한다.

## 핵심 수식

### Generative model
- Prior: θ ~ N(μ_0, Σ_0)
- Token factuality (latent): π_ℓ(θ) = σ(θ^T z_ℓ)
- Sentence aggregation: μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(θ)
- Observation: F_j | θ ~ Bernoulli(μ_j(θ))

### Learnable feature
- z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k
- h_ℓ^agg = Σ_l α_l · h_ℓ^(l), α_l = softmax(α)_l

### Posterior (Laplace, Fisher-type precision)
- Σ̂^{-1} = Σ_0^{-1} + Σ_j (1/(μ̂_j(1-μ̂_j))) ĝ_j ĝ_j^T
- ĝ_j = (1/L_j) Σ_ℓ π̂_ℓ(1-π̂_ℓ) z_ℓ

### Uncertainty decomposition (1차 근사)
- Total ≈ μ̂*(1-μ̂*)
- Epistemic ≈ ĝ*^T Σ̂ ĝ*
- Aleatoric ≈ max(0, Total - Epistemic)

## 구현 원칙

### 이론-구현 대응
- 이론: unclipped ℓ_j = F_j log μ_j + (1-F_j) log(1-μ_j)
- 구현: clipped μ̃_j = clamp(μ_j, ε, 1-ε), ε=1e-6
- Boundary 접촉 비율을 monitoring (>5%면 prior tightening)

### 알고리즘
- Inner loop: Damped Fisher scoring (IRLS-유사, canonical GLM은 아님)
- Outer loop: Adam on ψ = (W, α, μ_0, log σ_0)
- Unrolled differentiation으로 gradient가 ψ로 흐름
- Posterior precision은 Fisher-type (expected Fisher information)
- Local PD 확인은 clipped true Hessian으로 (autograd)

### 코드 규칙
- PyTorch 2.x, type hints 필수
- 모든 함수에 docstring (수식, 차원, 반환값 명시)
- 단위 테스트 필수 (tests/)
- Numerical safety: eps clipping, fp32 계산 후 fp16 저장

## 프로젝트 구조

sentence_uq/
├── CLAUDE.md (이 파일)
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── pilot.yaml
├── src/
│   ├── data/        # 데이터 생성, 문장 분리, annotation
│   ├── features/    # Feature extractor
│   ├── models/      # Main/Aux Bayesian 모델, Fisher scoring
│   ├── train/       # Bilevel trainer
│   ├── inference/   # Predictive + decomposition
│   ├── baselines/   # 비교 baseline
│   ├── evaluation/  # Metrics
│   └── utils/       # I/O, 로깅, 디버깅
├── scripts/         # 01~07 실행 스크립트
├── tests/           # 단위 테스트
├── data/            # 원본 + 생성 결과 (gitignored)
├── results/         # 최종 실험 결과
└── prompts/         # 각 Phase별 구현 지시서

## 구현 순서

Phase 0: 프로젝트 초기화 (prompts/phase_0_init.md)
Phase 1: 데이터 파이프라인 (phase_1_*.md)
  1-1: LLM 생성 + hidden state
  1-2: 문장 분리 + 토큰 매핑
  1-3: Entropy/top-1 offline cache
  1-4: Factuality annotation
Phase 2: Feature extractor (phase_2_1_features.md)
Phase 3: Main Bayesian 모델 (phase_3_*.md)
  3-1: Fisher scoring inner loop
  3-2: Main model class
  3-3: Predictive inference
Phase 4: 학습 루프 (phase_4_*.md)
  4-1: Trainer
  4-2: Aux regression 모델
Phase 5: Baselines (phase_5_1_baselines.md)
Phase 6: 평가 (phase_6_*.md)
Phase 7: 통합 및 디버깅 (phase_7_*.md)

## 현재 Phase

<!-- 매 세션 시작 시 업데이트 -->
현재: Phase 0 (프로젝트 초기화) 진행 예정

## 중요 주의사항

1. 새 Phase 시작 전에 이전 Phase의 테스트가 모두 통과하는지 확인
2. 비용이 드는 단계(LLM 생성, annotation)는 5 entity로 먼저 pilot
3. Fisher scoring 수렴 실패 시: prior tightening, lambda 증가, iteration 확인
4. OOM 발생 시: selected layers 줄이기, batch size, gradient checkpointing
EOF

echo "✓ Created CLAUDE.md"

# ============================================================
# Phase 0: 프로젝트 초기화
# ============================================================
cat > "$PROMPTS_DIR/phase_0_init.md" << 'EOF'
# Phase 0: 프로젝트 초기화

CLAUDE.md에 정의된 프로젝트 구조를 생성한다.

## 작업

1. 디렉토리 구조 생성:
   sentence_uq/
   ├── configs/ (default.yaml, pilot.yaml)
   ├── src/ (data, features, models, train, inference, baselines, evaluation, utils)
   ├── scripts/
   ├── tests/
   └── data/ (raw, generations, processed, cache)

2. 모든 Python 패키지 디렉토리에 빈 __init__.py 생성.

3. requirements.txt 작성:
   - torch>=2.1
   - transformers>=4.40
   - spacy>=3.7
   - scikit-learn
   - numpy
   - scipy
   - pyyaml
   - tqdm
   - datasets
   - matplotlib
   - pandas
   - pytest

4. README.md 작성 (간단히):
   - 프로젝트 개요 (2-3줄)
   - 설치 방법 (pip install -r requirements.txt, python -m spacy download en_core_web_sm)
   - 실행 방법 (bash scripts/run_pilot.sh)
   - 디렉토리 구조

5. configs/default.yaml 작성:
   ```yaml
   model:
     name: "meta-llama/Meta-Llama-3-8B-Instruct"
     device: "cuda"
     dtype: "float16"
     selected_layers: [0, 8, 12, 16, 20, 24, 28, 32]
     max_new_tokens: 512

   data:
     entity_list_path: "data/raw/entities.txt"
     generations_dir: "data/generations"
     cache_dir: "data/cache"
     processed_path: "data/processed/annotated.json"

   features:
     projection_dim: 64
     # k = projection_dim + 2 = 66

   training:
     num_epochs: 50
     lr: 1.0e-3
     num_fisher_iters: 10
     fisher_lambda_init: 1.0e-4
     eps: 1.0e-6
     prior_sigma_init: 1.0
     eval_every: 1
     pd_check_every: 5
     early_stopping_patience: 10

   splits:
     train_entities: 350
     val_entities: 50
     test_entities: 100
     seed: 42

   output:
     results_dir: "results"
     checkpoint_dir: "results/checkpoints"
   ```

6. configs/pilot.yaml 작성 (default.yaml을 상속하는 것처럼, 차이만):
   ```yaml
   defaults: "default.yaml"  # (실제로 상속 로직은 utils/io.py에서 구현)

   splits:
     train_entities: 35
     val_entities: 5
     test_entities: 10

   training:
     num_epochs: 20  # pilot은 빨리
   ```

7. .gitignore 작성:
   - data/generations/*
   - data/cache/*
   - data/processed/*
   - results/
   - *.pt
   - __pycache__/
   - *.egg-info/
   - .env

## 검증

작업 완료 후 다음을 실행하여 확인:
- `find sentence_uq -type f -name "*.py" | head -20` (빈 파일들이 있는지)
- `cat sentence_uq/requirements.txt`
- `cat sentence_uq/configs/default.yaml`

모든 파일이 제대로 만들어졌는지 요약 보고.
EOF

echo "✓ Created phase_0_init.md"

# ============================================================
# Phase 1-1: LLM 생성
# ============================================================
cat > "$PROMPTS_DIR/phase_1_1_generation.md" << 'EOF'
# Phase 1-1: LLM 응답 생성 + Hidden State 추출

## 목표
Llama-3-8B-Instruct로 prompt를 받아 응답을 생성하면서, 각 생성된 토큰의
hidden state와 logit을 저장한다.

## 구현 파일
- src/data/generation.py
- scripts/01_generate_data.py
- tests/test_generation.py

## 상세 요구사항

### src/data/generation.py

1. `load_model(model_name, device, dtype)`:
   - transformers로 모델과 tokenizer 로드
   - output_hidden_states=True
   - device_map="auto"
   - 반환: (model, tokenizer)

2. `generate_with_hidden_states(model, tokenizer, prompt, max_new_tokens, selected_layers)`:
   - **Greedy decoding** (temperature=0, 재현성 위해)
   - 매 step마다 KV cache (past_key_values) 활용하여 효율적 생성
   - 각 step에서:
     * 마지막 position의 hidden state를 selected_layers만 저장
     * 마지막 position의 logit 저장 (fp16)
   - EOS 토큰 만나면 중단
   - 반환 dict:
     * "text": 생성된 텍스트 (decode, skip_special_tokens=True)
     * "token_ids": (T,) LongTensor
     * "hidden_states": (T, len(selected_layers), hidden_dim) fp16
     * "logits": (T, vocab_size) fp16

3. `save_generation(result, save_path)`:
   - torch.save로 .pt 파일 저장
   - hidden_states와 logits은 이미 fp16

4. `batch_generate(model, tokenizer, prompts, save_dir, selected_layers, ...)`:
   - prompts 리스트 순회
   - 이미 save_dir에 있는 인덱스는 skip (resume)
   - tqdm progress
   - 각 결과를 save_dir/{idx:05d}.pt로 저장
   - metadata.json에 (idx, prompt, entity) 기록

### 중요 구현 세부사항

- `model.generate()`를 쓰지 말고 **수동 루프**로 구현. 그래야 중간 hidden state를 뽑을 수 있음.
- 수동 루프 예시 구조:
  ```python
  past_kv = None
  input_ids = initial_input_ids
  for step in range(max_new_tokens):
      with torch.no_grad():
          outputs = model(
              input_ids=input_ids if past_kv is None else next_token.unsqueeze(0),
              past_key_values=past_kv,
              output_hidden_states=True,
              use_cache=True,
          )
      past_kv = outputs.past_key_values
      next_token_logits = outputs.logits[:, -1, :]
      # save hidden states and logits
      next_token = torch.argmax(next_token_logits, dim=-1)
      if next_token.item() == tokenizer.eos_token_id:
          break
      input_ids = next_token
  ```
- GPU 메모리 관리: `outputs.hidden_states`는 큰 tensor이므로 매 step마다 CPU로 옮기거나
  (T, num_layers, hidden_dim)이 커질 수 있으니 list로 모아서 마지막에 stack.

### scripts/01_generate_data.py

- configs/default.yaml 로드 (argparse --config)
- data/raw/entities.txt에서 entity 리스트 읽기 (한 줄에 하나)
- Prompt format: "Tell me a bio of {entity}."
- batch_generate 호출

### tests/test_generation.py

- 단위 테스트:
  * 작은 GPT-2 등으로 loading 테스트
  * Mock tokenizer로 generate_with_hidden_states 동작 확인
  * Hidden state shape가 (T, len(selected_layers), hidden_dim)인지
  * token_ids 길이가 T인지

### 검증 방법

5개 entity로 작은 테스트:
```python
entities = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Galileo Galilei"]
```

실행 후:
- data/generations/에 5개 .pt 파일 생성됨
- 각 파일을 torch.load해서 shape 확인
- generated text가 말이 되는지 눈으로 확인

## Notes

- 현재 단계에서는 annotation은 하지 않음 (Phase 1-4에서)
- Entity list는 일단 10개 정도 하드코딩해서 시작 가능 (나중에 FActScore 원본 리스트로 교체)
EOF

echo "✓ Created phase_1_1_generation.md"

# Phase 1-2, 1-3, 1-4도 유사하게...
# (공간 절약을 위해 나머지는 유사한 구조로 간략히 포함)

cat > "$PROMPTS_DIR/phase_1_2_sentence_split.md" << 'EOF'
# Phase 1-2: 문장 분리 + 토큰 매핑

## 목표
생성된 텍스트를 문장 단위로 분리하고, 각 문장의 토큰 index 범위를 찾는다.

## 구현 파일
- src/data/sentence_split.py
- tests/test_sentence_split.py

## 요구사항

### src/data/sentence_split.py

1. `load_spacy_model(lang="en")`:
   - spacy.load("en_core_web_sm"); 없으면 subprocess로 자동 설치 후 로드

2. `split_into_sentences(text, nlp)`:
   - 반환: [{"text", "char_start", "char_end"}, ...]

3. `map_sentences_to_tokens(sentences, token_ids, tokenizer)`:
   - 각 문장에 해당하는 토큰 범위 (tok_start, tok_end) 계산
   - Tokenizer re-encoding + offset_mapping 사용
   - **주의**: re-encoding 결과 길이가 원래 token_ids와 다를 수 있음.
     이 경우 원래 token_ids를 하나씩 decode하면서 char position 추적하는
     fallback 전략 필요.

4. `process_generation(generation_result, tokenizer, nlp)`:
   - 위 함수들을 조합하여 문장별 정보 구조체 반환
   - Invalid 문장 (빈 토큰 범위, 너무 짧은 문장) filter

### 테스트

- "Hello world. This is a test." → 2개 문장
- 각 문장의 token range decode 결과가 원본과 일치
- Edge case: 빈 text, 단일 문장, 매우 긴 문장
EOF

echo "✓ Created phase_1_2_sentence_split.md"

cat > "$PROMPTS_DIR/phase_1_3_cached_scalars.md" << 'EOF'
# Phase 1-3: Entropy / Top-1 Offline Cache

## 목표
각 토큰의 predictive entropy와 top-1 probability를 offline에서 미리 계산해서 캐시한다.
ψ에 의존하지 않으므로 한 번만 계산하면 됨.

## 구현 파일
- src/features/cached_scalars.py
- scripts/01b_cache_scalars.py

## 요구사항

### src/features/cached_scalars.py

1. `compute_token_entropy_and_top1(logits)`:
   - logits: (T, V) Tensor
   - log_probs = F.log_softmax(logits.float(), dim=-1)
   - probs = log_probs.exp()
   - entropy = -(probs * log_probs).nansum(dim=-1)  (nansum으로 0*log0 처리)
   - top1_prob = probs.max(dim=-1).values
   - 반환: (entropy, top1_prob), 둘 다 (T,) fp32

2. `cache_scalars_for_directory(generations_dir, cache_dir)`:
   - generations_dir의 모든 .pt 파일 순회
   - 각 파일의 logits에서 계산
   - {cache_dir}/{idx:05d}.pt에 {"entropy", "top1_prob", "token_ids"} 저장
   - tqdm progress

3. `load_scalars(idx, cache_dir)`:
   - 캐시 파일 로드

### scripts/01b_cache_scalars.py

- Config에서 경로 읽기
- cache_scalars_for_directory 호출

## Notes

- Logits은 fp16 → 계산 시 float()으로 cast
- Entropy 계산에서 numerical stability: F.log_softmax가 logsumexp 기반이라 안전
EOF

echo "✓ Created phase_1_3_cached_scalars.md"

cat > "$PROMPTS_DIR/phase_1_4_annotation.md" << 'EOF'
# Phase 1-4: Factuality Annotation

## 목표
생성된 각 문장의 factuality를 0/1로 annotate.

## 구현 파일
- src/data/annotation.py
- scripts/02_annotate_factuality.py

## 방식

두 가지 옵션:
- (A) factscore 라이브러리 (OpenAI API 필요, 더 정확)
- (B) LLM-as-judge (Claude/GPT-4o, 간단)

**기본은 (B) 방식으로 구현**하되, (A)용 stub도 남겨둠.

## 요구사항

### src/data/annotation.py

1. `annotate_sentence_with_llm_judge(entity, sentence, api_client, wikipedia_context=None)`:
   - Anthropic Claude API 또는 OpenAI API 사용
   - Prompt:
     ```
     You are a fact-checker. Given an entity and a sentence from a biography,
     determine if the sentence is factually correct.
     
     Entity: {entity}
     Sentence: {sentence}
     {If context: Reference: {context}}
     
     Answer with a single word: "SUPPORTED" or "NOT_SUPPORTED".
     ```
   - temperature=0
   - 반환: 1 (SUPPORTED), 0 (NOT_SUPPORTED), None (파싱 실패)

2. `retrieve_wikipedia_context(entity, max_chars=3000)`:
   - wikipedia-api 라이브러리 사용
   - 에러 시 None

3. `annotate_batch(processed_sentences, api_client, use_wiki=True, resume=True)`:
   - 이미 annotated된 문장은 skip (resume)
   - Rate limit 고려 (간단한 sleep)
   - 중간 저장 (매 100 문장마다)

### scripts/02_annotate_factuality.py

- Config에서 API key 읽기 (환경변수)
- data/processed/에서 문장 로드
- annotate_batch 호출
- data/processed/annotated.json에 저장

## 비용 관리

- 500 entities × ~15 sentences = 7,500 calls
- Claude Sonnet: ~$15
- GPT-4o: ~$10
- Pilot (50 entities)은 ~$1 내외

## Notes

- 프롬프트 injection 방지: sentence를 user turn 내 변수로 처리
- 너무 짧은 문장 (< 5 words) 또는 non-factual structure (질문, 인사 등)는 filter
EOF

echo "✓ Created phase_1_4_annotation.md"

# ============================================================
# Phase 2-1: Feature Extractor
# ============================================================
cat > "$PROMPTS_DIR/phase_2_1_features.md" << 'EOF'
# Phase 2-1: Feature Extractor

## 수식

z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k

h_ℓ^agg = Σ_l α_l · h_ℓ^(l),  α_l = softmax(α)_l

k = p + 2 (p = projection_dim)

학습 파라미터:
- W ∈ R^{p × d}
- α ∈ R^{L_layers}
- μ_0 ∈ R^k
- log σ_0 ∈ R^k

## 구현 파일
- src/features/extractor.py
- tests/test_features.py

## src/features/extractor.py

### SentenceUQParams(nn.Module)

```python
class SentenceUQParams(nn.Module):
    def __init__(self, hidden_dim=4096, num_layers=8, projection_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.feature_dim = projection_dim + 2  # k
        
        self.W = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(num_layers))  # before softmax
        self.mu_0 = nn.Parameter(torch.zeros(self.feature_dim))
        self.log_sigma_0 = nn.Parameter(torch.zeros(self.feature_dim))
    
    def get_Sigma_0_inv(self):
        # Σ_0 = diag(exp(2 log σ_0))
        # Σ_0^{-1} = diag(exp(-2 log σ_0))
        return torch.diag(torch.exp(-2 * self.log_sigma_0))
    
    def get_Sigma_0(self):
        return torch.diag(torch.exp(2 * self.log_sigma_0))
```

### 함수들

1. `extract_token_features(hidden_states, entropy, top1_prob, params)`:
   - hidden_states: (T, num_layers, hidden_dim) fp32
   - entropy: (T,) fp32
   - top1_prob: (T,) fp32
   - 계산:
     ```python
     w = torch.softmax(params.alpha, dim=0)  # (num_layers,)
     h_agg = torch.einsum("l, tlh -> th", w, hidden_states)  # (T, hidden_dim)
     h_proj = params.W(h_agg)  # (T, projection_dim)
     z = torch.cat([h_proj, entropy.unsqueeze(1), top1_prob.unsqueeze(1)], dim=1)
     ```
   - 반환: (T, k)

2. `extract_sentence_token_features(hidden_states, entropy, top1_prob, token_range, params)`:
   - 문장 범위만 slice
   - token_range: (start, end)
   - 반환: (L_j, k)

3. `extract_sentence_aggregate_feature(z_tokens)`:
   - 보조 모델용 문장 레벨 feature
   - 입력: z_tokens (L_j, k)
   - mean, std, last의 concat: (3k,)
   - L_j=1 edge case: std = zeros

## 테스트 (tests/test_features.py)

- Mock hidden_states로 feature dim이 k = projection_dim + 2인지
- requires_grad 체크 (W, alpha에 gradient)
- num_layers=1 edge case
- Token range 빈 범위 시 에러
- extract_sentence_aggregate_feature의 차원 = 3k
EOF

echo "✓ Created phase_2_1_features.md"

# ============================================================
# Phase 3-1: Fisher Scoring
# ============================================================
cat > "$PROMPTS_DIR/phase_3_1_fisher_scoring.md" << 'EOF'
# Phase 3-1: Damped Fisher Scoring Inner Loop

## 수식

Clipped objective:
  L̃(θ) = Σ_j [F_j log μ̃_j + (1-F_j) log(1-μ̃_j)] - (1/2)(θ-μ_0)^T Σ_0^{-1} (θ-μ_0)
  μ̃_j = clamp(μ_j, ε, 1-ε)

Gradient:
  ∇L̃ = -Σ_0^{-1}(θ-μ_0) + Σ_j R_j g_j
  R_j = (F_j - μ̃_j)/(μ̃_j(1-μ̃_j))
  g_j = (1/L_j) Σ_ℓ π_ℓ(1-π_ℓ) z_ℓ

Fisher-type precision:
  H = Σ_0^{-1} + Σ_j (1/(μ̃_j(1-μ̃_j))) g_j g_j^T

Damped update:
  θ ← θ + (H + λI)^{-1} ∇L̃

## 구현 파일
- src/models/fisher_scoring.py
- tests/test_fisher_scoring.py

## src/models/fisher_scoring.py

### 함수 목록

1. `_compute_grad_and_fisher(theta, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps)`:
   - 입력:
     * theta: (k,)
     * all_z_tokens: List[(L_j, k)]
     * all_F: (N,)
     * mu_0: (k,)
     * Sigma_0_inv: (k, k)
   - Gradient와 Fisher-type Hessian 계산 (모두 differentiable tensor 연산)
   - 반환: grad (k,), H_fisher (k, k)

2. `_compute_clipped_objective(theta, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps)`:
   - L̃(θ) 스칼라 값

3. `fisher_scoring_map(all_z_tokens, all_F, mu_0, Sigma_0_inv,
                      num_iters=15, eps=1e-6, lambda_init=1e-4, verbose=False)`:
   - Damped Fisher scoring
   - Algorithm:
     ```
     theta = mu_0.clone()
     lam = lambda_init
     prev_obj = -inf  # 또는 초기 objective 계산
     
     for iter in range(num_iters):
         grad, H = _compute_grad_and_fisher(theta, ...)
         
         # Damped solve
         try:
             delta = torch.linalg.solve(H + lam*torch.eye(k), grad)
         except torch._C._LinAlgError:
             lam *= 10
             continue
         
         theta_new = theta + delta
         new_obj = _compute_clipped_objective(theta_new, ...)
         
         if new_obj > prev_obj + 1e-8:  # 명확한 개선
             theta = theta_new
             prev_obj = new_obj
             lam = max(lam / 2, 1e-8)  # damping 줄이기
         else:
             lam *= 10
             if lam > 1e10:
                 break  # 수렴 실패, 포기
     
     # 최종 θ에서 H 재계산
     _, H_final = _compute_grad_and_fisher(theta, ...)
     return theta, H_final
     ```
   - **중요**: 이 함수 내의 모든 연산이 differentiable (backward 가능)
   - detach() 사용 금지

4. `fisher_scoring_map_detached(...)`:
   - Inference용. torch.no_grad() context 안에서 실행.

## 테스트 (tests/test_fisher_scoring.py)

1. Synthetic data:
   ```python
   k = 5
   N = 20
   true_theta = torch.randn(k)
   z_list = [torch.randn(10, k) for _ in range(N)]
   F_list = torch.tensor([
       (torch.sigmoid(z @ true_theta).mean() > 0.5).float().item() 
       for z in z_list
   ])
   ```
   → Fisher scoring이 true_theta 근처로 수렴하는지

2. Gradient check:
   ```python
   # 작은 사이즈에서 torch.autograd.gradcheck
   ```

3. Numerical stability:
   - 모든 F_j = 0: θ가 -∞로 발산하지 않는지 (prior가 regularize)
   - 모든 F_j = 1: θ가 +∞로 발산하지 않는지

4. Fisher PD 확인:
   - 수렴 후 H의 최소 eigenvalue > 0

## Notes

- Unrolled Newton은 메모리 많이 씀 → num_iters=10~15 권장
- Convergence 실패 시 warning 출력
- 함수가 tensor operation으로만 구성되어야 외부에서 backward 가능
EOF

echo "✓ Created phase_3_1_fisher_scoring.md"

# ============================================================
# 나머지 phase들도 유사하게...
# 여기서는 간결하게 3-2, 3-3, 4-1, 4-2, 5, 6, 7 헤더만 생성
# ============================================================

for phase in "3_2_main_model" "3_3_predictive" "4_1_trainer" "4_2_aux_model" \
             "5_1_baselines" "6_1_metrics" "6_2_evaluate" "7_1_integration" "7_2_debug"; do
    cat > "$PROMPTS_DIR/phase_${phase}.md" << EOF
# Phase ${phase//_/-}

(TODO: 이 phase는 앞선 Phase 완료 후 작성)

이 파일은 placeholder이다. 해당 phase 진행 시 research_document_v6.md의
관련 Part를 참조하여 구현 지시를 작성한다.

참조:
- Phase 3-2: research_document_v6.md Part VII (Main Bayesian Model)
- Phase 3-3: research_document_v6.md Part V (Predictive Inference)
- Phase 4-1: research_document_v6.md Part VII §7.6 (Outer Loop)
- Phase 4-2: research_document_v6.md Part VIII (Auxiliary Model)
- Phase 5-1: claude_code_prompts.md의 Prompt 5-1
- Phase 6-1: claude_code_prompts.md의 Prompt 6-1
- Phase 6-2: claude_code_prompts.md의 Prompt 6-2
- Phase 7-1: claude_code_prompts.md의 Prompt 7-1
- Phase 7-2: claude_code_prompts.md의 Prompt 7-2
EOF
done

echo "✓ Created phase_3_2 ~ phase_7_2 placeholders"

# ============================================================
# 실행 헬퍼 스크립트
# ============================================================
cat > "run_phase.sh" << 'EOF'
#!/bin/bash
# Claude Code에 Phase 파일을 pipe로 주입하여 실행
# 사용법:
#   ./run_phase.sh 0_init
#   ./run_phase.sh 1_1_generation

set -e

PHASE=$1
FILE="prompts/phase_${PHASE}.md"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found"
    echo "Available phases:"
    ls prompts/ | grep -oP 'phase_\K[^.]+' | sort
    exit 1
fi

echo "=== Running Phase: $PHASE ==="
echo "File: $FILE"
echo ""

# Claude Code CLI 호출
# (실제 설치된 command 이름에 따라 'claude' 또는 'claude-code' 등으로 수정)
cat "$FILE" | claude
EOF

chmod +x run_phase.sh

echo "✓ Created run_phase.sh"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Created:"
echo "  - CLAUDE.md (project context)"
echo "  - prompts/phase_*.md (individual phase instructions)"
echo "  - run_phase.sh (helper script)"
echo ""
echo "Usage:"
echo ""
echo "Option 1 — Pipe to Claude Code:"
echo "  ./run_phase.sh 0_init"
echo "  ./run_phase.sh 1_1_generation"
echo ""
echo "Option 2 — Reference in interactive session:"
echo "  $ claude"
echo "  > @prompts/phase_0_init.md 실행해"
echo ""
echo "Option 3 — Manual copy-paste (if @ syntax unsupported):"
echo "  $ cat prompts/phase_0_init.md | pbcopy    # macOS"
echo "  $ cat prompts/phase_0_init.md | xclip     # Linux"
echo "  Then paste into Claude Code session."
echo ""
echo "CLAUDE.md is automatically loaded by Claude Code on startup,"
echo "so the project context is always available."
