# Bayesian Sentence-Level Uncertainty Quantification for LLMs
## Research Document v8

**v7 → v8 변경 요약**:
- 관측 모델을 Bernoulli($F_j$)에서 Binomial($K_j \mid m_j$)로 일반화
- Annotation pipeline(FActScore)의 natural output인 atomic fact count를 직접 모델링
- Uncertainty decomposition을 latent probability / ratio / count / strict factuality 네 층위로 확장
- Evaluation protocol을 ratio-level primary + strict factuality secondary 구조로 체계화
- 모델 약점(conditional independence, overdispersion, $m_j$ weighting) 명시 및 대응 전략 추가

**v8 errata (code review 반영)**:
- §6.1: $\text{ent}_\ell$, $\text{top1}_\ell$이 generation-time uncertainty ($x_\ell$을 sampling한 분포 기준)임을 명시. 
  Hidden state와의 conditioning 차이 ($x_{\leq\ell}$ vs $x_{<\ell}$) 주석 추가.
- §7.2: "Clipped objective의 gradient" → "epsilon-stabilized Fisher scoring"으로 명명 변경. 
  Clipped objective는 line-search/evaluation에만 사용하고, gradient/Fisher에서는 
  분모 stabilization만 적용함을 명확화. Boundary fraction monitoring 추가.

---

# Part I. 문제 정의

## 1.1 Notation

| 기호 | 정의 |
|------|------|
| $x$ | Input query |
| $r = (s_1, \dots, s_M)$ | LLM 응답 |
| $s_j$ | j번째 문장, 길이 $L_j$ |
| $\mathbf{h}_\ell^{(l)} \in \mathbb{R}^d$ | 토큰 $\ell$, layer $l$의 hidden state |
| $\mathbf{z}_\ell \in \mathbb{R}^k$ | 토큰 $\ell$의 extracted feature |
| $m_j$ | 문장 $s_j$에서 추출된 atomic fact 수 |
| $K_j \in \{0, \ldots, m_j\}$ | 문장 $s_j$의 supported atomic fact 수 |
| $U_j := K_j / m_j$ | 문장 $s_j$의 factuality ratio |
| $A_j := \mathbf{1}\{K_j = m_j\}$ | Strict factuality indicator |
| $\theta \in \mathbb{R}^k$ | Global parameter |
| $\pi_\ell(\theta) := \sigma(\theta^\top \mathbf{z}_\ell)$ | 토큰 $\ell$의 latent factuality probability |
| $\mu_j(\theta) := \frac{1}{L_j}\sum_{\ell \in s_j}\pi_\ell(\theta)$ | 문장 $s_j$의 latent factuality probability |
| $\psi$ | Feature extractor + prior 파라미터 (§6) |

## 1.2 목표

Single forward pass의 $\{\mathbf{h}_\ell\}$만으로 각 문장의 factuality probability와 
epistemic/aleatoric uncertainty를 추정한다.

## 1.3 핵심 아이디어

토큰별 factuality label $y_\ell$이 관측되지 않으므로 $y_\ell$을 모델에 넣지 않는다.
대신 문장 factuality를 토큰별 latent probability의 평균으로 직접 모델링하고,
FActScore pipeline이 제공하는 atomic fact count $(K_j, m_j)$를 binomial likelihood로 직접 관측한다.

## 1.4 Prior Art: Factuality Probes

Han et al. (2025, EMNLP Findings)은 LLM hidden state가 long-form generation의 factuality를
높은 정확도로 예측할 수 있음을 보였다. 이들의 접근은 다음과 같다:

1. LLM이 생성한 텍스트를 atomic claim으로 분해
2. 각 claim을 LLM (generator 또는 더 작은 모델)에 re-encode하여 마지막 토큰의 
   single-layer hidden state $\mathbf{h}_c \in \mathbb{R}^d$를 추출
3. Lightweight probe $f(\mathbf{h}_c) = \sigma(\mathbf{w}^\top \mathbf{h}_c)$
   (L1-regularized logistic regression)로 factuality를 예측

주요 결과: Llama-3.1-8B에서 in-domain AUROC 0.7357, 405B에서 0.7579.
Single forward pass로 sampling-based 방법 대비 100배 이상 적은 FLOPs로 comparable한 성능.
작은 모델에서 학습한 probe가 큰 모델 및 closed-source 모델의 generation에도 일반화.

**본 연구의 출발점과 차별점**:

Han et al.의 결과는 "hidden state에 factuality 정보가 인코딩되어 있다"는 것을 
empirically establish한다. 본 연구는 이 finding을 출발점으로 삼되, 
**다른 질문**에 답한다:

- Han et al.: *"Which claims are hallucinated?"* (claim-level binary prediction)
- 본 연구: *"How uncertain is the prediction, and why?"* (sentence-level uncertainty quantification with count-based observation model)

Han et al.은 개별 claim에 대한 binary prediction을 제공한다.
본 연구는 문장 내의 supported claim count를 binomial likelihood로 직접 모델링하여,
factuality ratio의 예측뿐 아니라 그 예측의 epistemic/aleatoric uncertainty를 분해한다.
구체적 차별점은 §9.4에서 수학적으로 비교한다.


---

# Part II. 메인 모델

## 2.1 Generative Model

$$
\begin{aligned}
\theta &\sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \\[4pt]
K_j \mid \theta, m_j &\sim \text{Binomial}(m_j, \mu_j(\theta)), \quad \mu_j(\theta) = \frac{1}{L_j}\sum_{\ell \in s_j} \sigma(\theta^\top \mathbf{z}_\ell)
\end{aligned}
$$

여기서 $m_j$는 문장 $s_j$에서 추출된 atomic fact 수이고, $K_j$는 그중 supported로 판정된 수이다.

## 2.2 모델의 해석

이 모델은 다음의 latent atomic fact model에서 유도된다.
각 atomic fact $a = 1, \ldots, m_j$에 대해

$$
B_{ja} \mid \theta \overset{\text{ind}}{\sim} \text{Bernoulli}(\mu_j(\theta)), \qquad K_j = \sum_{a=1}^{m_j} B_{ja}
$$

로 놓으면 $K_j \mid \theta, m_j \sim \text{Binomial}(m_j, \mu_j(\theta))$이 된다.

$\mu_j(\theta)$는 token-level hidden state에서 추출한 sentence-level latent factuality probability이고,
$(K_j, m_j)$는 FActScore pipeline이 직접 제공하는 관측량이다.
따라서 모델의 관측 변수와 annotation pipeline의 출력 사이에 임의적인 binarization 규칙이 개입하지 않는다.

**Atomic fact와 token의 관계**: 
Atomic fact는 문장의 의미론적 분해이고, $\mu_j(\theta)$는 토큰 단위 aggregation이다.
이 둘은 같은 층위의 변수가 아니다. 모델은 token-level hidden state로 sentence-level 
factuality probability를 예측하고, 그 probability가 atomic fact count를 생성한다고 본다.
토큰과 atomic fact를 일대일로 맞추지 않는다.


### 2.2.1 Homogeneous Probability 가정의 정당화

Binomial model은 문장 $s_j$ 내의 모든 atomic fact가 동일한 성공 확률 $\mu_j(\theta)$를 
공유한다고 가정한다. 실제로는 같은 문장 안에서도 fact마다 참일 확률이 다를 수 있다.
예를 들어 "Einstein은 독일에서 태어났고 1921년에 노벨 물리학상을 받았다"에서 
출생지에 관한 claim과 수상 연도에 관한 claim의 factuality probability는 서로 다를 수 있다.
그럼에도 homogeneous probability를 채택하는 이유를 세 가지 관점에서 설명한다.

**관점 1: Feature의 정보 해상도와 예측 target의 정합**

본 모델의 입력 feature $\mu_j(\theta) = \frac{1}{L_j}\sum_{\ell \in s_j}\sigma(\theta^\top \mathbf{z}_\ell)$는
문장 내 토큰들의 평균이다. 이 feature는 설계상 sentence-level summary이며,
개별 atomic fact의 identity나 내용에 대한 정보를 담고 있지 않다.
Atomic fact $a$의 고유한 확률 $\mu_{ja}(\theta)$를 모델링하려면
각 fact에 대응하는 별도의 representation이 필요하지만,
sentence-level aggregated feature로부터 이를 복원하는 것은 information-theoretically 불가능하다.

Heterogeneous model $B_{ja} \sim \text{Bernoulli}(\mu_{ja}(\theta))$을 적용하려면
Han et al. (2025)의 접근처럼 각 atomic fact를 LLM에 re-encode하여 claim-level hidden state를 
추출해야 한다. 이는 추가적인 forward pass를 요구하여 single forward pass의 효율성을 잃으며,
claim-level binary prediction이라는 다른 연구 목표에 해당한다.
본 연구의 목표는 generation-time hidden state만으로 sentence-level uncertainty를 추정하는 것이므로,
feature의 해상도와 prediction target이 모두 sentence level에서 일치하는 것이 자연스럽다.

**관점 2: $\mu_j(\theta)$는 평균 확률의 predictor이다**

실제 각 atomic fact의 참일 확률이 $p_{j1}, \ldots, p_{jm_j}$로 서로 다르다고 하자.
이때 문장의 평균 factuality probability $\bar{p}_j := \frac{1}{m_j}\sum_{a=1}^{m_j} p_{ja}$는
여전히 well-defined한 quantity이며, $K_j$의 기댓값은 $\mathbb{E}[K_j] = \sum_a p_{ja} = m_j \bar{p}_j$이다.

Binomial model에서 $\mu_j(\theta)$의 역할은 바로 이 $\bar{p}_j$를 예측하는 것이다.
$\mu_j(\theta) \approx \bar{p}_j$인 한, $K_j$의 1차 moment (기댓값)에 대한 specification은 올바르다.
이는 quasi-likelihood theory (Wedderburn, 1974)에서 **mean function만 올바르게 지정하면
parameter estimator의 consistency가 보장된다**는 결과와 부합한다.

차이가 발생하는 것은 2차 moment이다. 
실제 variance는

$$
\text{Var}(K_j) = \sum_{a=1}^{m_j} p_{ja}(1 - p_{ja}) \le m_j \bar{p}_j(1 - \bar{p}_j)
$$

이고, 등호는 $p_{ja}$가 모두 동일할 때 성립한다 (Jensen's inequality). 
즉 heterogeneous probability는 binomial variance보다 **작은** variance를 유도하며,
이는 underdispersion에 해당한다.
반면 §XV.1에서 논의하는 conditional dependence (하나의 잘못된 entity가 여러 fact를 동시에 틀리게 만드는 현상)는
**overdispersion**을 유도한다. 두 효과가 부분적으로 상쇄될 수 있으며,
어느 쪽이 지배적인지는 실험적으로 확인한다 (§XV.2 overdispersion diagnostic).

**관점 3: Maximum entropy 원리**

관측에서 알 수 있는 것이 $(K_j, m_j)$뿐일 때,
$m_j$번의 시행에서 성공 확률에 대한 constraint가 $\mathbb{E}[K_j] = m_j \mu_j$뿐이라면,
maximum entropy distribution은 $\text{Binomial}(m_j, \mu_j)$이다.
개별 fact의 확률에 대한 추가 정보 없이 가장 적은 가정을 도입하는 선택이
homogeneous probability에 해당한다.


**Conditional independence 가정**:
Binomial model은 같은 문장 내의 atomic facts가 $\theta$가 주어졌을 때 conditionally independent라고 가정한다.
실제로는 하나의 잘못된 entity가 여러 fact를 동시에 틀리게 만들 수 있으므로 이 가정은 approximate이다.
이로 인한 overdispersion 가능성을 §XV.1에서 논의한다.

**요약**: Homogeneous probability 가정은 세 가지 수준에서 정당화된다.
(i) Feature의 정보 해상도가 sentence level이므로 claim-level heterogeneity를 식별할 수 없다.
(ii) $\mu_j(\theta)$는 평균 factuality probability의 predictor로서 mean specification이 올바르다.
(iii) $(K_j, m_j)$만 관측될 때 maximum entropy 선택에 해당한다.
가정의 위배가 estimation과 uncertainty에 미치는 영향은 §XV.1–XV.2에서 분석한다.


## 2.3 Bernoulli model과의 관계

기존 Bernoulli sentence-label model은 binomial model의 특수한 경우이다.
$m_j = 1$, $K_j = F_j$로 두면

$$
K_j \mid \theta, m_j = 1 \sim \text{Binomial}(1, \mu_j(\theta)) = \text{Bernoulli}(\mu_j(\theta))
$$

따라서 이 변경은 기존 모델을 버리는 것이 아니라 일반화하는 것이다.

## 2.4 Binomial 확장의 이점

1. **FActScore annotation과의 정합**: annotation pipeline의 natural output인 $(K_j, m_j)$를 직접 사용한다.
2. **Partial factuality 표현**: 3개 중 2개만 맞는 문장의 정보를 보존한다.
3. **정보량 반영**: $m_j$가 큰 문장은 더 많은 관측 정보를 제공하므로 posterior precision에 더 크게 기여한다.
4. **문장 간 비교**: $K_j$가 아니라 $U_j = K_j / m_j$ 또는 $\mu_j(\theta)$ 기준으로 비교한다.


---

# Part III. Posterior 분석

## 3.1 Log-Posterior

$$
\mathcal{L}(\theta) = \sum_{j=1}^N \ell_j(\theta) - \frac{1}{2}(\theta - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1}(\theta - \boldsymbol{\mu}_0) + \text{const}
$$

$$
\ell_j(\theta) = K_j \log\mu_j(\theta) + (m_j - K_j)\log(1-\mu_j(\theta)) + \log\binom{m_j}{K_j}
$$

MAP estimation에서는 $\log\binom{m_j}{K_j}$가 $\theta$에 의존하지 않으므로 생략할 수 있다.
비율 $U_j = K_j / m_j$를 사용하면 $\theta$-dependent part는

$$
\ell_j(\theta) = m_j \left[ U_j \log \mu_j(\theta) + (1 - U_j)\log(1 - \mu_j(\theta)) \right] + \text{const}
$$

즉 binomial likelihood는 $U_j$에 대한 **$m_j$-weighted binary cross entropy**이다.

$m_j = 0$인 문장은 likelihood에 정보를 제공하지 않으므로 $\ell_j(\theta) = 0$으로 처리한다.

## 3.2 Gradient

**Proposition 1** (Gradient).

$$
\nabla_\theta \ell_j(\theta) = R_j^{\text{bin}}(\theta)\,\mathbf{g}_j(\theta), \quad
R_j^{\text{bin}} := \frac{K_j - m_j\mu_j}{\mu_j(1-\mu_j)}, \quad
\mathbf{g}_j := \frac{1}{L_j}\sum_{\ell \in s_j}\pi_\ell(1-\pi_\ell)\,\mathbf{z}_\ell
$$

$U_j = K_j / m_j$를 사용하면

$$
R_j^{\text{bin}} = m_j \cdot \frac{U_j - \mu_j}{\mu_j(1-\mu_j)}
$$

전체: $\nabla_\theta \mathcal{L} = \sum_{j=1}^N R_j^{\text{bin}} \mathbf{g}_j - \boldsymbol{\Sigma}_0^{-1}(\theta - \boldsymbol{\mu}_0)$.

**증명**. Chain rule + $\frac{\partial \mu_j}{\partial \theta} = \mathbf{g}_j$. $\blacksquare$

Bernoulli model과 비교하면 $F_j$가 $K_j$와 $m_j$로 바뀌고, residual term이 $m_j$만큼 weighted된다.


## 3.3 Fisher-type Posterior Precision

### 정확한 Hessian

$$
-\nabla^2_\theta \ell_j = w_j^{\text{obs}}(\theta) \mathbf{g}_j\mathbf{g}_j^\top - R_j^{\text{bin}} \cdot \mathbf{Q}_j(\theta)
$$

여기서

$$
w_j^{\text{obs}}(\theta) = \frac{K_j}{\mu_j(\theta)^2} + \frac{m_j - K_j}{(1-\mu_j(\theta))^2}
$$

$\mathbf{Q}_j$는 $\mu_j$의 2차 도함수 항.

### 근사 선택

Expected Fisher information을 사용한다. $\mathbb{E}[K_j \mid \theta, m_j] = m_j\mu_j$이므로
$\mathbb{E}[R_j^{\text{bin}}] = 0$이 되어 $R_j^{\text{bin}} \mathbf{Q}_j$ 항이 소거되고,

$$
\mathbb{E}[w_j^{\text{obs}} \mid \theta, m_j] = \frac{m_j}{\mu_j(\theta)(1-\mu_j(\theta))}
$$

**Definition** (Fisher-type posterior precision).

$$
\boxed{
\hat{\boldsymbol{\Sigma}}^{-1} := \boldsymbol{\Sigma}_0^{-1} + \sum_{j=1}^N \frac{m_j}{\hat{\mu}_j(1-\hat{\mu}_j)}\hat{\mathbf{g}}_j\hat{\mathbf{g}}_j^\top
}
$$

**해석**: 기존 Bernoulli precision에 $m_j$가 곱해진다. 
Atomic fact가 많은 문장은 더 많은 independent trial을 제공하므로 
posterior precision에 더 크게 기여한다.

**근거**:
1. Binomial model의 expected Fisher information에 기반한다.
2. Gauss-Newton / Fisher scoring 계열의 표준 approximation이다.
3. Structural choice이지 asymptotic argument에 기반하지 않는다.

Posterior: $p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\hat{\theta}, \hat{\boldsymbol{\Sigma}})$.


## 3.4 MAP Existence: Local 분석

**Proposition 2** (True Hessian Decomposition).

$\hat\theta$가 critical point일 때, true negative Hessian은 다음과 같이 분해된다:

$$
-\nabla^2\mathcal{L}(\hat\theta) = \underbrace{\boldsymbol{\Sigma}_0^{-1} + \sum_{j=1}^N \frac{m_j}{\hat\mu_j(1-\hat\mu_j)}\hat{\mathbf{g}}_j\hat{\mathbf{g}}_j^\top}_{=\hat{\boldsymbol{\Sigma}}^{-1} \text{ (Fisher-type)}} - \sum_{j=1}^N R_j^{\text{bin}}(\hat\theta)\hat{\mathbf{Q}}_j
$$

$-\nabla^2\mathcal{L}(\hat\theta) \succ 0$이면 $\hat\theta$는 local MAP이고 Laplace approximation이 well-defined이다.

**논의**:

- 이 Proposition은 **true Hessian의 분해를 보이는 것**이며, Fisher-type term과 
  residual correction term을 분리한다. "Fisher-type term이 correction term을 dominate한다"는 
  조건은 $\hat{\boldsymbol{\Sigma}}^{-1} \succ \sum_j R_j^{\text{bin}} \hat{\mathbf{Q}}_j$와 동치이지만, 
  이것은 true Hessian의 PD 조건을 재진술한 것이므로 별도의 "쉽게 체크 가능한 sufficient condition"이 아니다.

- **Global strict concavity는 보장하지 않는다**. $\sigma(a)$의 비단조 2차 도함수,
  $|R_j^{\text{bin}}|$의 boundary 발산 등으로 인해 prior만으로 global PD가 보장되지 않는다.

- **실용적 접근**: MAP를 numerically 찾은 뒤 $-\nabla^2 \mathcal{L}(\hat\theta)$의 
  최소 eigenvalue를 autograd로 직접 계산하여 PD 여부를 확인한다 (§7.4).

- **Prior의 역할**: proper prior $\boldsymbol{\Sigma}_0^{-1} \succ 0$은 regularization을 제공하지만
  이것만으로 global PD를 보장한다고 주장하지 않는다.


---

# Part IV. Uncertainty Decomposition

## 4.1 Predictive Target의 분리

Binomial observation model 하에서 predictive target이 여러 층위로 분리된다.
각 target에 대한 uncertainty의 의미가 다르다.

| Target | 정의 | 용도 |
|--------|------|------|
| Latent probability $\mu_*(\theta)$ | 문장의 평균 factuality probability | 문장 간 비교, 주요 score |
| Factuality ratio $U_* = K_*/m_*$ | 실제 fact-checking 시 관측될 비율 | Calibration, prediction interval |
| Count $K_*$ | Supported atomic fact 수 | Count prediction |
| Strict factuality $A_* = \mathbf{1}\{K_* = m_*\}$ | 모든 fact가 맞는 사건 | Error detection (secondary) |

## 4.2 Latent Probability에 대한 Decomposition

**Theorem 1** (Law of Total Variance — Ratio Level).

$$
\text{Var}[U_* \mid \mathcal{D}, m_*] = \underbrace{\mathbb{E}_{\theta|\mathcal{D}}\!\left[\frac{\mu_*(\theta)(1-\mu_*(\theta))}{m_*}\right]}_{\text{Aleatoric}_U} + \underbrace{\text{Var}_{\theta|\mathcal{D}}[\mu_*(\theta)]}_{\text{Epistemic}_U}
$$

**증명**. $U_* = K_*/m_*$이고 $K_* \mid \theta, m_* \sim \text{Binomial}(m_*, \mu_*(\theta))$이므로
$\text{Var}(U_* \mid \theta, m_*) = \mu_*(1-\mu_*)/m_*$이고 $\mathbb{E}[U_* \mid \theta] = \mu_*(\theta)$.
Law of Total Variance 적용. $\blacksquare$

이 decomposition은 exact이며 approximation이 아니다.

**핵심**: Aleatoric term에 $1/m_*$가 붙는다. Atomic fact가 많은 문장일수록 
ratio $U_*$의 sampling noise가 줄어든다. 이는 "관측을 더 많이 하면 noise가 줄어든다"는
통계학의 기본 원리가 모델에 내장된 것이다.

**Bernoulli 특수 경우**: $m_* = 1$이면 $U_* = F_* \in \{0,1\}$이고

$$
\text{Var}[F_* \mid \mathcal{D}] = \mathbb{E}_\theta[\mu_*(1-\mu_*)] + \text{Var}_\theta[\mu_*]
$$

이는 v7의 Theorem 1과 동일하다.

## 4.3 1차 근사 (Computable Form)

$\mu_*(\theta) \approx \hat\mu_* + \hat{\mathbf{g}}_*^\top(\theta - \hat\theta)$ 전개 하에서:

**Proposition 3** (Approximate Decomposition — Ratio Level).

$$
\boxed{
\begin{aligned}
\text{Epi}_\mu(s_*) &\approx \hat{\mathbf{g}}_*^\top \hat{\boldsymbol{\Sigma}}\,\hat{\mathbf{g}}_* \\[4pt]
\text{Aleatoric}_U(s_*) &\approx \frac{\hat\mu_*(1-\hat\mu_*) - \text{Epi}_\mu(s_*)}{m_*} \\[4pt]
\text{Total}_U(s_*) &\approx \text{Aleatoric}_U(s_*) + \text{Epi}_\mu(s_*)
\end{aligned}
}
$$

**증명**: $\text{Var}[\mu_*] \approx \hat{\mathbf{g}}_*^\top \hat{\boldsymbol{\Sigma}}\hat{\mathbf{g}}_*$;
$\mathbb{E}[\mu_*(1-\mu_*)] \approx \hat\mu_*(1-\hat\mu_*) - \text{Var}[\mu_*]$. $\blacksquare$

**Well-definedness Caveat**: 선형 근사에서 $\text{Aleatoric}_U < 0$이 나올 수 있다.
구현에서 clipping과 MC verification을 병행한다.

$$
\text{Aleatoric}_U(s_*) \leftarrow \max\!\left\{0,\; \frac{\hat\mu_*(1-\hat\mu_*) - \text{Epi}_\mu(s_*)}{m_*}\right\}
$$

## 4.4 Count 및 Strict Factuality에 대한 Decomposition

**Count** $K_*$에 대해:

$$
\text{Aleatoric}_K(s_*) \approx m_*[\hat\mu_*(1-\hat\mu_*) - \text{Epi}_\mu(s_*)], \qquad
\text{Epistemic}_K(s_*) \approx m_*^2 \,\text{Epi}_\mu(s_*)
$$

**Strict factuality** $A_* = \mathbf{1}\{K_* = m_*\}$에 대한 score:

$$
\hat{p}(A_* = 1) = \hat\mu_*^{m_*}
$$

또는 posterior averaging:

$$
\hat{p}(A_* = 1) = \mathbb{E}_{\theta \mid D}[\mu_*(\theta)^{m_*}]
$$

Strict factuality는 학습 target이 아니라 **evaluation target**으로만 사용한다.
$\mu_*^{m_*}$을 직접 optimize하면 $m_*$가 큰 문장에 지나치게 불리한 target이 된다.

**요약 표**:

| Target | Aleatoric | Epistemic |
|--------|-----------|-----------|
| Latent $\mu_*$ | — (observation noise 없음) | $\text{Var}_\theta[\mu_*]$ |
| Ratio $U_* = K_*/m_*$ | $\mathbb{E}[\mu_*(1-\mu_*)]/m_*$ | $\text{Var}_\theta[\mu_*]$ |
| Count $K_*$ | $m_*\mathbb{E}[\mu_*(1-\mu_*)]$ | $m_*^2\text{Var}_\theta[\mu_*]$ |
| Strict $A_* = \mathbf{1}\{K_*=m_*\}$ | Bernoulli noise of $\mu_*^{m_*}$ | Variance of $\mu_*^{m_*}$ |

## 4.5 토큰 레벨 Attribution

**Theorem 2** (Additive Signed Attribution).

$$
\text{Epi}_\mu(s_*) = \sum_{\ell \in s_*} \text{Attr}_\ell, \quad \text{Attr}_\ell := \frac{1}{L_*}\,\mathbf{g}_\ell^\top \hat{\boldsymbol{\Sigma}}\,\hat{\mathbf{g}}_*, \quad \mathbf{g}_\ell := \hat\pi_\ell(1-\hat\pi_\ell)\mathbf{z}_\ell
$$

**증명**. $\hat{\mathbf{g}}_* = \frac{1}{L_*}\sum_\ell \mathbf{g}_\ell$ 대입. $\blacksquare$

**주의**: $\text{Attr}_\ell$은 **signed contribution**이며 개별 값이 음수일 수 있다.

Token-level attribution은 $\mu_*$에 대한 것이므로 Binomial 확장과 독립적으로 동일한 형태를 유지한다.
Ratio $U_*$의 epistemic attribution은 $\mu_*$와 같고,
count $K_*$의 epistemic attribution은 $m_*^2 \text{Attr}_\ell$로 rescale된다.

**시각화를 위한 marginal uncertainty** (delta method):

$$
\text{LocalEpi}_\ell := [\hat\pi_\ell(1-\hat\pi_\ell)]^2 \cdot \mathbf{z}_\ell^\top \hat{\boldsymbol{\Sigma}}\,\mathbf{z}_\ell
$$

항상 양수. 다만 $\sum_\ell \text{LocalEpi}_\ell \neq \text{Epi}_\mu$.


---

# Part V. Predictive Inference

## 5.1 추론 절차

1. Feature: $\mathbf{z}_\ell = \phi_\psi(\mathbf{h}_\ell)$
2. 토큰 factuality: $\hat\pi_\ell = \sigma(\hat\theta^\top \mathbf{z}_\ell)$
3. 문장 factuality: $\hat\mu_* = \frac{1}{L_*}\sum_\ell \hat\pi_\ell$
4. Effective gradient: $\hat{\mathbf{g}}_* = \frac{1}{L_*}\sum_\ell \hat\pi_\ell(1-\hat\pi_\ell)\mathbf{z}_\ell$
5. Epistemic: $\text{Epi}_\mu = \hat{\mathbf{g}}_*^\top \hat{\boldsymbol{\Sigma}}\hat{\mathbf{g}}_*$
6. (Given $m_*$) Aleatoric ratio: $\text{Aleatoric}_U = \max(0, [\hat\mu_*(1-\hat\mu_*) - \text{Epi}_\mu] / m_*)$
7. (Given $m_*$) Strict factuality: $\hat{p}(A_* = 1) = \hat\mu_*^{m_*}$
8. (Optional) $\text{Attr}_\ell$ 및 $\text{LocalEpi}_\ell$

**출력 구조**: $m_*$를 모르면 latent probability level에서 $(\hat\mu_*, \text{Epi}_\mu)$만 보고한다.
$m_*$가 주어지면 ratio-level prediction interval과 strict factuality probability까지 제공한다.

## 5.2 Probit-Shrinkage Bayesian Predictive

$$
\tilde\pi_\ell \approx \sigma\!\left(\frac{\hat\theta^\top \mathbf{z}_\ell}{\sqrt{1 + (\pi/8)\mathbf{z}_\ell^\top\hat{\boldsymbol{\Sigma}}\mathbf{z}_\ell}}\right), \quad \tilde\mu_* = \frac{1}{L_*}\sum_\ell \tilde\pi_\ell
$$


---

# Part VI. Feature Extractor

## 6.1 구조

$$
\mathbf{z}_\ell = \begin{bmatrix} W \cdot \mathbf{h}_\ell^{\text{agg}} \\ \text{ent}_\ell \\ \text{top1}_\ell \end{bmatrix} \in \mathbb{R}^k, \quad \mathbf{h}_\ell^{\text{agg}} = \sum_{l} \alpha_l \mathbf{h}_\ell^{(l)}, \quad \alpha_l = \text{softmax}(\boldsymbol{\alpha})_l
$$

- $W \in \mathbb{R}^{p \times d}$ ($d = 4096 \to p = 64$), $k = p + 2 = 66$
- $\psi = (W, \boldsymbol{\alpha}, \boldsymbol{\mu}_0, \log\boldsymbol{\sigma}_0)$

**Scalar feature 정의 (generation-time uncertainty)**:

$\text{ent}_\ell$과 $\text{top1}_\ell$은 **토큰 $\ell$을 생성(sampling)할 때 사용된 분포**의 entropy 및 top-1 probability이다:

$$
\text{ent}_\ell = H(x_\ell \mid x_{<\ell}) = -\sum_v p_v \log p_v, \quad \text{top1}_\ell = p^{(1)}(x_\ell \mid x_{<\ell}) = \max_v p_v
$$

여기서 $p = \text{softmax}(\text{logits}_\ell)$이고, $\text{logits}_\ell$은 토큰 $x_\ell$을 sampling하는 데 사용된 
logit vector이다 (즉, $x_\ell$ 직전 step의 model output). 이는 "모델이 이 토큰을 생성할 때 
얼마나 확신했는가"를 측정하며, hallucination detection의 직접적 signal이 된다.

**주의**: $h_\ell$은 토큰 $x_\ell$을 input으로 처리한 **후**의 representation ($x_{\leq\ell}$ conditioning)이고,
$\text{ent}_\ell$, $\text{top1}_\ell$은 토큰 $x_\ell$을 sampling하기 **전**의 분포 ($x_{<\ell}$ conditioning)이다.
두 요소의 conditioning이 한 step 차이나지만, learnable projection $W$와 aggregation이 
이 차이를 흡수하며, 중요한 것은 각 scalar가 해당 토큰의 generation-time uncertainty를 
정확히 반영하는 것이다.

## 6.2 설계 근거 (실험적 검증)

| Component | 가설 | 검증 방법 |
|---|---|---|
| Layer weights $\boldsymbol\alpha$ | Layer별 factuality 기여 다름 | 학습된 분포 확인 + 고정 layer 비교 |
| Projection $W$ | 차원 축소로 Hessian 안정화 | $p \in \{16, 32, 64, 128\}$ ablation |
| Entropy, Top-1 | Token-level uncertainty 신호 | With vs without ablation |


---

# Part VII. 학습 알고리즘

## 7.1 Primary Objective

**MAP plug-in bilevel**:

$$
\min_\psi \mathcal{J}(\psi) := -\sum_{j=1}^N \ell_j(\hat\theta(\psi);\, \{\mathbf{z}_\ell(\psi)\}_{\ell \in s_j}), \quad \hat\theta(\psi) = \arg\max_\theta \mathcal{L}(\theta;\psi)
$$

여기서 $\ell_j$는 binomial log-likelihood이다.

**Note on terminology**: 이것은 MAP plug-in이며 true empirical Bayes (Type-II ML)와는 다르다.
True EB는 $\int p(\mathcal{D}\mid\theta)p(\theta\mid\psi)d\theta$를 최적화하며 plug-in을 하지 않는다.

## 7.2 Numerical Stabilization: Epsilon-Stabilized Fisher Scoring

### 7.2.1 Clipped Objective (evaluation 및 line-search용)

구현상 $\log \mu_j$와 $\log(1-\mu_j)$의 수치 안정성을 위해 objective 평가 시 
$\mu_j$를 $[\epsilon, 1-\epsilon]$로 clip한다 (typically $\epsilon = 10^{-6}$):

$$
\tilde\ell_j(\theta) = K_j \log\tilde\mu_j(\theta) + (m_j - K_j)\log(1-\tilde\mu_j(\theta)), \quad \tilde\mu_j = \text{clip}(\mu_j, \epsilon, 1-\epsilon)
$$

$$
\tilde{\mathcal{L}}(\theta) = \sum_j \tilde\ell_j(\theta) - \frac{1}{2}(\theta-\mu_0)^\top\Sigma_0^{-1}(\theta-\mu_0)
$$

이 clipped objective는 Fisher scoring의 **line-search 판단** (accept/reject)과 
최종 **evaluation metric** 계산에 사용된다.

### 7.2.2 Epsilon-Stabilized Gradient and Fisher Precision

Gradient와 Fisher-type precision은 clipped objective의 **true gradient가 아니라**,
분모의 수치 안정성만 보장하는 **epsilon-stabilized** 형태를 사용한다:

$$
R_j^{\epsilon} = \frac{K_j - m_j \mu_j}{\max(\mu_j(1-\mu_j),\; \epsilon)}
$$

즉, gradient의 분자에서는 raw $\mu_j$를 사용하되, 분모에서 $\mu_j(1-\mu_j) \to 0$일 때의 
발산만 방지한다. 이는 true clipped objective의 gradient와 다르다:

- **True clipped gradient**: $\mu_j$가 boundary에 도달하면 $\frac{\partial \tilde\mu_j}{\partial \mu_j} = 0$이므로 
  해당 문장의 likelihood gradient 기여가 0이 된다.
- **Epsilon-stabilized gradient (현재 구현)**: boundary 여부와 무관하게 모든 문장이 
  gradient에 기여하되, 분모의 수치 안정성만 보장한다.

두 방식의 차이는 $\mu_j \in (\epsilon, 1-\epsilon)$인 interior region에서는 없으며,
$\mu_j$가 boundary에 도달하는 극단적 상황에서만 나타난다.

**설계 근거**: epsilon-stabilized 방식을 채택한 이유는 (i) boundary 근처의 문장 정보를 
완전히 버리지 않고, (ii) 실용적으로 $\epsilon = 10^{-6}$에서 boundary에 도달하는 문장이 
거의 없으므로 차이가 무시 가능하기 때문이다.

### 7.2.3 Boundary Fraction Monitoring

Epsilon stabilization이 실제로 active되는 빈도를 추적한다:

$$
\text{boundary\_frac} = \frac{1}{N}\sum_{j=1}^{N} \mathbf{1}\{\mu_j < \epsilon \text{ or } \mu_j > 1-\epsilon\}
$$

이 비율이 5%를 초과하면 모델이 overfit/separation 상태일 가능성이 있으며,
prior scale 조정이 필요하다. Pilot 실험에서 이 비율을 반드시 보고한다.

**이후 모든 코드에서**: gradient/Hessian은 "epsilon-stabilized Fisher scoring"으로 명명하고,
"clipped objective의 gradient"라는 표현을 사용하지 않는다.

## 7.3 Exploratory: Laplace-EB Correction

Marginal likelihood의 Laplace approximation:

$$
\log p(\mathcal{D}\mid\psi) \approx \tilde{\mathcal{L}}(\hat\theta(\psi);\psi) - \frac{1}{2}\log\det(-\nabla^2\tilde{\mathcal{L}}(\hat\theta)) + \frac{k}{2}\log(2\pi)
$$

$$
\mathcal{J}^{\text{EB}}(\psi) := -\tilde{\mathcal{L}}(\hat\theta(\psi);\psi) + \frac{1}{2}\log\det(-\nabla^2\tilde{\mathcal{L}}(\hat\theta))
$$

**구현 status**: Exploratory. Log-det gradient 계산 방식
(implicit diff / Hutchinson trace estimator / analytic w/ Fisher-type approximation)은
실험 단계에서 결정. 최종 논문에서는 결과에 따라 main/ablation/drop.

## 7.4 Inner Loop: Fisher Scoring (Binomial version)

**명명**: 우리 모델은 canonical exponential family GLM이 아니라 (토큰 평균 aggregation 구조), 
정확히 IRLS는 아니지만 **유사한 Fisher scoring** 방식이다.
True Hessian 대신 Fisher-type precision을 사용한 iterative update.

### Update rule (damped Fisher scoring)

$$
\theta^{(t+1)} = \theta^{(t)} + \big(H^{(t)}_{\text{fisher}} + \lambda^{(t)} I\big)^{-1} \nabla\tilde{\mathcal{L}}(\theta^{(t)})
$$

여기서:
- $H^{(t)}_{\text{fisher}}$: 현재 $\theta^{(t)}$에서의 binomial Fisher-type precision
- $\lambda^{(t)}$: damping parameter (small $\lambda_0 = 10^{-4}$에서 시작, 
  update가 improve하지 않으면 증가시키는 simple adaptive scheme)

### Code

```python
def fisher_scoring_map_binomial(
    all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv,
    num_iters=15, eps=1e-6, lambda_init=1e-4
):
    """
    Damped Fisher scoring (binomial version).
    
    Line-search 판단: clipped objective L̃(θ)
    Gradient/Hessian: epsilon-stabilized (§7.2.2)
    
    Returns:
        theta_hat: (k,) approximate MAP
        H_fisher_final: (k, k) Fisher-type precision at theta_hat
    """
    theta = mu_0.clone()
    lam = lambda_init
    prev_obj = compute_clipped_objective_binomial(
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
    )
    
    for iteration in range(num_iters):
        grad, H_fisher = _compute_grad_and_fisher_binomial(
            theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )
        
        # Damped Fisher scoring step
        try:
            delta = solve(H_fisher + lam * eye(k), grad)
        except LinAlgError:
            lam *= 10
            continue
        
        theta_new = theta + delta
        new_obj = compute_clipped_objective_binomial(
            theta_new, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )
        
        if new_obj > prev_obj:
            theta = theta_new
            prev_obj = new_obj
            lam = max(lam / 2, 1e-8)   # decrease damping when succeeding
        else:
            lam *= 10                    # increase damping and retry
            if lam > 1e10:
                break                    # give up
    
    _, H_fisher_final = _compute_grad_and_fisher_binomial(
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
    )
    return theta, H_fisher_final


def _compute_grad_and_fisher_binomial(
    theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
):
    """Epsilon-stabilized gradient와 Binomial Fisher-type precision (§7.2.2)."""
    grad = -Sigma_0_inv @ (theta - mu_0)
    H_fisher = Sigma_0_inv.clone()
    
    for j in range(len(all_K)):
        if all_m[j] == 0:
            continue
        
        z_j = all_z_tokens[j]
        K_j = all_K[j]
        m_j = all_m[j]
        
        pi_j = sigmoid(z_j @ theta)
        mu_j_raw = pi_j.mean()
        mu_j = clamp(mu_j_raw, eps, 1 - eps)
        
        weights = pi_j * (1 - pi_j)
        g_j = (weights.unsqueeze(1) * z_j).mean(dim=0)
        
        R_j = (K_j - m_j * mu_j) / (mu_j * (1 - mu_j))
        
        grad += R_j * g_j
        H_fisher += (m_j / (mu_j * (1 - mu_j))) * outer(g_j, g_j)
    
    return grad, H_fisher


def compute_clipped_objective_binomial(
    theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
):
    """Clipped binomial objective L̃(θ) evaluation."""
    obj = -0.5 * (theta - mu_0) @ Sigma_0_inv @ (theta - mu_0)
    for j in range(len(all_K)):
        if all_m[j] == 0:
            continue
        pi_j = sigmoid(all_z_tokens[j] @ theta)
        mu_j = clamp(pi_j.mean(), eps, 1 - eps)
        obj += all_K[j] * log(mu_j) + (all_m[j] - all_K[j]) * log(1 - mu_j)
    return obj
```

## 7.5 Monitoring Protocol

**Important distinction**: 
- Laplace approximation에서 사용하는 precision: **Fisher-type** $H_{\text{fisher}}$
- MAP의 local PD 조건: **clipped true Hessian** $-\nabla^2\tilde{\mathcal{L}}(\hat\theta)$

Proposition 2는 true Hessian에 대한 조건이며, Fisher-type만 PD여도 충분하지 않다.
단, 실험에서 관심 있는 것은 Fisher-type $\hat{\boldsymbol{\Sigma}}$로 계산한 uncertainty이므로
두 가지 모두 monitor한다.

```python
def verify_local_pd(
    theta_hat, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6
):
    """
    두 개의 Hessian 모두 PD 확인.
    
    Returns:
        fisher_min_eig: Fisher-type precision의 최소 eigenvalue
        true_min_eig: Clipped objective의 true negative Hessian 최소 eigenvalue
        laplace_valid: true_min_eig > 0 (Laplace approximation의 local 근거)
    """
    # Fisher-type
    _, H_fisher = _compute_grad_and_fisher_binomial(
        theta_hat, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
    )
    fisher_min_eig = eigvals(H_fisher).real.min().item()
    
    # True Hessian of clipped objective (via autograd)
    theta_var = theta_hat.clone().detach().requires_grad_(True)
    
    def neg_obj(theta_):
        total = 0.5 * (theta_ - mu_0) @ Sigma_0_inv @ (theta_ - mu_0)
        for j in range(len(all_K)):
            if all_m[j] == 0:
                continue
            pi_j = sigmoid(all_z_tokens[j] @ theta_)
            mu_j = clamp(pi_j.mean(), eps, 1 - eps)
            total -= all_K[j] * log(mu_j) + (all_m[j] - all_K[j]) * log(1 - mu_j)
        return total
    
    H_true = torch.autograd.functional.hessian(neg_obj, theta_var)
    true_min_eig = eigvals(H_true).real.min().item()
    
    return {
        "fisher_min_eig": fisher_min_eig,
        "true_min_eig": true_min_eig,
        "fisher_pd": fisher_min_eig > 0,
        "true_pd": true_min_eig > 0,
        "laplace_valid_local": true_min_eig > 0,
    }
```

**주의**: `H_true`는 **clipped objective** $-\nabla^2\tilde{\mathcal{L}}(\hat\theta)$의 Hessian이지 
원래 unclipped $-\nabla^2\mathcal{L}(\hat\theta)$가 아니다. Boundary 근처에서 두 값이 다를 수 있다.
Unclipped version의 PD는 해당 문장들이 interior에 있을 때에만 의미가 있다.

## 7.6 Outer Loop

```python
def train(train_data, num_epochs, lr, eps=1e-6):
    psi = initialize_psi()
    optimizer = Adam(psi.parameters(), lr=lr)
    
    # Entropy/top-1는 ψ에 의존하지 않으므로 offline 캐시
    cached_scalars = precompute_entropy_top1(train_data)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        all_z_tokens, all_K, all_m = extract_all_features(train_data, psi, cached_scalars)
        theta_hat, H_fisher = fisher_scoring_map_binomial(
            all_z_tokens, all_K, all_m, psi.mu_0, psi.Sigma_0_inv, 
            num_iters=10, eps=eps
        )
        
        loss = compute_binomial_loss(theta_hat, all_z_tokens, all_K, all_m, eps)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            status = verify_local_pd(
                theta_hat.detach(), all_z_tokens, all_K, all_m, 
                psi.mu_0.detach(), psi.Sigma_0_inv.detach(), eps
            )
            print(f"epoch {epoch}: loss={loss.item():.4f}, "
                  f"laplace_valid={status['laplace_valid_local']}, "
                  f"fisher_eig={status['fisher_min_eig']:.2e}, "
                  f"true_eig={status['true_min_eig']:.2e}")
            if not status['laplace_valid_local']:
                print("WARNING: True Hessian not PD. Tighten prior.")
```

## 7.7 Computational Cost

**설정**: 500 entities × ~15 sentences × ~20 tokens ≈ 150,000 tokens total (또는 
$N \approx 7500$ sentences, $\bar L \approx 20$, 평균적으로).

### Hidden state 저장 용량

- Vocab size $V \approx 128{,}000$
- Hidden dim $d = 4096$
- Layers $L_{\text{layers}} = 33$ (Llama-3-8B)
- Total tokens $T = 150{,}000$

Full storage (all layers, fp16):
$$T \times L_{\text{layers}} \times d \times 2\text{B} = 1.5 \times 10^5 \times 33 \times 4096 \times 2 = 4.05 \times 10^{10} \text{ bytes} \approx 40.5 \text{ GB}$$

Fp32로 하면 $\approx 81$ GB, fp16로 하면 $\approx 40.5$ GB.

**실용적 선택**: 8개 selected layers만 저장 (예: 0, 8, 12, 16, 20, 24, 28, 32):
$$T \times 8 \times d \times 2\text{B} \approx 9.8 \text{ GB (fp16)}$$

### 연산 비용 (per epoch)

| 단계 | 비용 |
|---|---|
| Entropy/top-1 softmax (offline, 1회) | $O(T \cdot V) \approx 2 \times 10^{10}$ flops |
| Layer aggregation + projection (매 epoch) | $O(T \cdot (L_{\text{layers}} \cdot d + d \cdot p)) \approx 3 \times 10^{10}$ flops |
| Fisher scoring inner loop ($K=10$) | $O(K \cdot N \cdot \bar{L} \cdot k + K \cdot k^3) \approx 10^8$ flops |
| Backward through unrolled loop | Inner loop × depth factor |

**Feasibility**:
- Offline precomputation (hidden states + entropy + top-1): 수 시간
- Training: GPU에서 epoch당 수 분 ~ 수십 분 예상
- Pilot 실험 (50 entities)으로 실제 wall-clock 측정 필요


---

# Part VIII. 보조 모델 — Bayesian Regression on Uncertainty

## 8.1 Target Variable의 bounded 성질

Target $U_j^* \in [0, 1]$은 probability-valued이다. 
Binomial 확장 후 $U_j = K_j / m_j$가 main model의 관측과 직접적으로 대응하므로,
보조 모델의 target variable 선택이 자연스러워진다.

Gaussian regression은 unbounded support를 가정하므로 엄밀히는 부적절하다.

### 선택지

(A) **Logit-transformed Gaussian regression** (권장):

$$
\text{logit}(U_j^*) = \log\frac{U_j^*}{1 - U_j^*} \sim \mathcal{N}(\theta^\top \mathbf{z}_j, \sigma^2)
$$

Prediction: $\hat{U}_* = \sigma(\hat\theta^\top \mathbf{z}_*)$. 
Conjugate posterior 유지되고 예측이 $[0,1]$ 내.

(B) **Beta regression**:

$$
U_j^* \sim \text{Beta}(\alpha(\theta^\top \mathbf{z}_j), \beta(\theta^\top \mathbf{z}_j))
$$

더 적절한 error model이지만 conjugate하지 않아 closed-form posterior를 잃는다.

(C) **직접 Gaussian regression** (baseline용):

$$
U_j^* = \theta^\top \mathbf{z}_j + \epsilon_j, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

예측이 $[0,1]$를 벗어날 수 있다. 
실용적으로는 $U_j^* \in [0.05, 0.95]$ 범위에서 근사가 합리적이며, conjugate 이점이 크다.

### 본 연구에서의 선택

**Logit-transformed Gaussian (옵션 A)** 을 기본으로 사용. 이유:
- $[0,1]$ 제약을 만족.
- Transformation 후 conjugate Gaussian이므로 exact closed-form posterior 유지.
- Sufficient statistics 해석이 깔끔.

구체적으로 $V_j := \text{logit}(U_j^*)$, $V_j \sim \mathcal{N}(\theta^\top\mathbf{z}_j, \sigma^2)$.
$U_j^* \in \{0, 1\}$ 경계값인 경우 $\epsilon$-smoothing으로 처리: $U_j^* \leftarrow (1-2\epsilon)U_j^* + \epsilon$.

## 8.2 Exact Posterior

$\mathbf{V} \in \mathbb{R}^N$: logit-transformed targets.

$$
\hat{\boldsymbol{\Sigma}}_N^{-1} = \boldsymbol{\Sigma}_0^{-1} + \frac{1}{\sigma^2}\mathbf{Z}^\top\mathbf{Z}, \qquad
\hat\theta_N = \hat{\boldsymbol{\Sigma}}_N\!\left(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0 + \frac{1}{\sigma^2}\mathbf{Z}^\top\mathbf{V}\right)
$$

Sufficient statistics: $T_1 = \mathbf{Z}^\top\mathbf{Z}$, $T_2 = \mathbf{Z}^\top\mathbf{V}$.

## 8.3 Predictive

Logit space에서:

$$
V_* \mid \mathbf{z}_* \sim \mathcal{N}(\hat\theta_N^\top \mathbf{z}_*, \sigma^2 + \mathbf{z}_*^\top \hat{\boldsymbol{\Sigma}}_N \mathbf{z}_*)
$$

Probability space로 변환: $\hat{U}_* = \sigma(\hat\theta_N^\top \mathbf{z}_*)$; 
95% interval은 $[\sigma(\mu - 1.96s), \sigma(\mu + 1.96s)]$ where $\mu, s^2$는 logit space parameter.

## 8.4 $\mathbf{z}_j$ (문장 레벨 feature)

토큰 feature $\{\mathbf{z}_\ell\}_{\ell \in s_j}$의 결합. 예:

$$
\mathbf{z}_j = \text{concat}(\text{mean}_\ell, \text{std}_\ell, \mathbf{z}_{\ell_j^{\text{last}}})
$$

구체적 형태는 실험 단계에서 선택.


---

# Part IX. Connections to Existing Methods

## 9.1 Clarification Ensembling (논문 1)과의 Conceptual Analogy

**주의**: 이것은 conceptual analogy이며 수학적 대응이 아니다.
논문 1은 **entropy** decomposition을 사용하고 우리는 **variance** decomposition을 사용하는데, 
entropy와 variance는 본질적으로 다른 uncertainty 측도 (각각 정보이론적 vs 통계적 2차 moment)이다.

구조적 유사성:

- 논문 1: Random variable $C$ (clarification)에 대한 marginalization으로 decomposition
- 우리: Random variable $\theta$ (parameter)에 대한 marginalization으로 decomposition

| 논문 1 decomposition | 우리 decomposition |
|---|---|
| $H(q(Y\mid X)) = I(Y;C\mid X) + \mathbb{E}_C[H(q(Y\mid X \oplus C))]$ | $\text{Var}[U_*] = \text{Var}_\theta[\mu_*] + \mathbb{E}_\theta[\mu_*(1-\mu_*)/m_*]$ |
| Entropy-based | Variance-based |
| Input-space marginalization | Parameter-space marginalization |

**보유한 것**: 구조적 유사성과 marginalization idea의 공유.

**보유하지 않은 것**:
- 두 decomposition 간 정확한 수학적 equivalence 또는 bound
- Entropy와 variance decomposition을 하나로 통합하는 framework

두 접근이 서로 다른 uncertainty source (입력 ambiguity vs model ignorance)를 포착하며
**상호보완적**일 수 있다는 것은 직관적으로 suggest된다. 
통합 framework은 future work.

## 9.2 Spectral Approach (논문 2)와의 Conceptual Analogy

**주의**: 이것도 conceptual analogy이며 정량적 이론적 연결이 아니다.

두 접근의 공통 모티프는 "semantic similarity matrix의 spectrum이 uncertainty와 관련"이라는 
아이디어이지만, 구체적 formulation과 quantity는 다르다.

**보유한 것**: 두 방법이 비슷한 구조적 요소를 사용한다는 관찰.

**보유하지 않은 것**: 
- von Neumann entropy와 우리 epistemic 간 정량적 함수 관계
- 한 quantity를 다른 것으로 bound하는 수학적 진술

**실험적 관계**: FActScore 데이터에서 상관관계 측정으로 empirical connection 보고.
이론적 통합은 future work.

## 9.3 표준 Bayesian GLM

**Proposition 4** (Special Case).
$L_j = 1$이면 $\mu_j = \sigma(\theta^\top\mathbf{z}_j)$로 표준 Bayesian logistic regression과 일치.
추가로 $m_j = 1$이면 기존 Bernoulli model로 환원.

**Proposition 5** (Information Inequality).

$$
\|\hat{\mathbf{g}}_j\|^2 \leq \frac{1}{L_j}\sum_{\ell \in s_j}\hat\pi_\ell^2(1-\hat\pi_\ell)^2\|\mathbf{z}_\ell\|^2
$$

(Jensen). 토큰 feature가 다양할수록 $\|\hat{\mathbf{g}}_j\|$ 감소 → Fisher information 기여 감소.

## 9.4 Factuality Probes (Han et al., 2025)와의 관계

### 수학적 비교

Han et al.의 probe는 다음과 같다:

$$
\hat{p}_c = \sigma(\mathbf{w}^\top \mathbf{h}_c), \quad \mathbf{w} \in \mathbb{R}^d, \quad \mathbf{h}_c \in \mathbb{R}^d \text{ (single layer, last token of re-encoded claim)}
$$

이는 point estimate만 제공하며 $\hat{p}_c$의 uncertainty를 정량화하지 않는다.

우리 모델과의 관계: Proposition 4에서 $L_j = 1$, $m_j = 1$이면 
$\mu_j = \sigma(\theta^\top \mathbf{z}_j)$이고 관측은 Bernoulli이다.
만약 추가로 (i) layer aggregation 없이 single layer를 사용하고,
(ii) projection $W$를 identity로 두고, (iii) entropy/top-1 feature를 제거하면,
우리 모델의 MAP estimate는 Han et al.의 logistic regression probe와 동일한 함수 형태가 된다.
즉, **Han et al.의 probe는 우리 framework의 maximally simplified special case**이다.

### 핵심 차별점

| 차원 | Han et al. (2025) | 본 연구 |
|---|---|---|
| **목표** | Factuality prediction (point estimate) | Uncertainty quantification (posterior) |
| **출력** | $\hat{p}_c \in [0,1]$ | $\hat{\mu}_j, \text{Epi}_\mu, \text{Aleatoric}_U, \hat{p}(A_j=1)$ |
| **관측 모델** | Binary claim label | Binomial count $(K_j, m_j)$ |
| **Hidden state 소스** | Claim re-encoding (추가 forward pass) | Generation-time hidden state (추가 비용 없음) |
| **Granularity** | Atomic claim, single token ($\mathbf{h}_c$) | 문장 내 모든 토큰 ($\{\mathbf{z}_\ell\}_{\ell \in s_j}$) |
| **Layer 사용** | Single layer (layer 14 optimal) | Multi-layer learnable aggregation ($\boldsymbol{\alpha}$) |
| **Attribution** | Heuristic span mapping (auxiliary LM) | 수학적 분해 (Theorem 2: $\text{Attr}_\ell$) |
| **Calibration 메커니즘** | 없음 (L1 regularization만) | Probit shrinkage (§XI), posterior-based |

### Generation-time vs Re-encoding Hidden State

Han et al.은 생성된 claim 텍스트를 LLM에 다시 넣어 hidden state를 추출한다.
이는 generation 시점의 internal state를 반영하지 않으며, claim 분해 과정에서의
정보 손실이 발생한다. 본 연구는 generation 시점의 hidden state를 직접 사용하므로,
"모델이 생성하는 순간에 자신의 불확실성을 이미 encoding하고 있다"는 
stronger claim을 뒷받침한다. 또한 inference 시 추가 forward pass가 불필요하다.

### Bayesian Extension의 의의

Han et al.의 probe가 $\hat{p}_c = 0.6$을 출력했을 때, 이것이 
"데이터가 충분하여 확신 있는 0.6"인지 "데이터가 부족하여 불확실한 0.6"인지 구분할 수 없다.
본 연구의 epistemic uncertainty $\text{Epi}_\mu$은 이 구분을 제공하며,
training data가 적은 영역에서 예측의 신뢰도가 낮음을 정량적으로 보고한다.

### Binomial Extension의 추가 차별점

Han et al.은 개별 claim에 대한 binary prediction을 제공한다.
이를 sentence level로 집계하면 우리의 $U_j$와 비슷한 quantity가 되지만,
그들은 이 집계에 대한 uncertainty를 제공하지 못한다.
우리 모델은 $m_j$가 주어졌을 때 $K_j$의 분포 전체를 예측하므로,
prediction interval, strict factuality probability 등을 자연스럽게 제공한다.

### 실험적 비교 전략

Han et al.의 probe를 baseline으로 포함한다:
- 같은 Llama-3-8B, FActScore 세팅에서 apple-to-apple 비교
- **ECE (calibration)**에서 Bayesian 접근의 우위가 핵심 가설
- **Binomial NLL**에서 count-aware model의 우위 비교
- Rejection curve에서의 비교 (Han et al. Figure 3에 대응)
- 학습된 $\boldsymbol{\alpha}$의 layer 분포를 Han et al.의 layer 14 finding과 비교


---

# Part X. Asymptotic Behavior

**Remark 1** (Posterior Concentration의 heuristic).

Regularity conditions 하에서 BvM-style argument로 
epistemic이 $O(1/N)$로 감소함을 기대할 수 있다.

Binomial model에서 Fisher precision의 $m_j$ factor는 effective sample size에 영향을 미친다.
$m_j > 1$인 문장이 많으면 effective information이 커져서 
posterior가 Bernoulli model보다 빠르게 concentrate할 수 있다.
직관적으로 effective sample size가 $\sum_j m_j$에 더 가까워진다.

**Regularity gap**:

엄밀한 BvM의 gap:
- **i.i.d. 가정**: FActScore-Bio에서 같은 entity의 문장들은 명백히 종속적. 
  이것은 **근본적 제약**.
- **Conditional independence**: Binomial model의 atomic fact independence 가정 위배 시
  effective information이 과대 추정될 수 있다.
- **Identifiability**: $\{\mathbf{z}_\ell\}$의 rank 조건 필요.
- **Fixed design**: $\mathbf{z}_\ell$는 $\psi$에 의존. 
  Outer loop에서 $\psi$를 고정한 뒤의 conditional statement에서만 fixed design 성립.

실험에서 $N$에 따른 epistemic scaling을 empirically만 확인.


---

# Part XI. Calibration

**Remark 2** (Probit Shrinkage Mechanism).

$$
\mathbb{E}_{\theta \sim \mathcal{N}(\hat\theta, \hat{\boldsymbol{\Sigma}})}[\sigma(\theta^\top \mathbf{z})] \approx \sigma\!\left(\frac{\hat\theta^\top \mathbf{z}}{\sqrt{1 + (\pi/8)\mathbf{z}^\top\hat{\boldsymbol{\Sigma}}\mathbf{z}}}\right)
$$

Effective logit이 0에 가까워져 예측이 0.5 쪽으로 shrink. Overconfidence 완화.

**주의**: Mechanism-level 설명이지 formal calibration guarantee가 아니다.
Prior misspecification, approximation error, model misspecification으로 인해 
calibration이 오히려 나빠질 수도 있다. Empirical validation.


---

# Part XII. Weighted Aggregation 확장

$$
\mu_j(\theta, \mathbf{v}) = \sum_\ell w_\ell(\mathbf{v})\pi_\ell(\theta), \quad w_\ell(\mathbf{v}) = \text{softmax}_{\ell \in s_j}(\mathbf{v}^\top \mathbf{z}_\ell)
$$

**Proposition 6**.

(a) $\nabla_\theta\mu_j = \sum_\ell w_\ell \pi_\ell(1-\pi_\ell)\mathbf{z}_\ell$ ($w_\ell$이 $\theta$-independent).

(b) Weighted $\hat{\mathbf{g}}_j^{\text{weighted}}$ 및 대응 $\mathbf{Q}_j^{\text{weighted}}$로 Proposition 2 재진술 필요. 
Uniform case에서 자동 이월되지 않는다.

(c) $\mathbf{v}$에 대한 non-concavity로 joint unimodality 보장 없음. 
$\mathbf{v}$는 outer loop (ψ)에 포함, inner loop에서 고정.

Binomial likelihood와의 결합은 straightforward하다: 
$\hat{\boldsymbol{\Sigma}}^{-1}$의 Fisher term에서 $\hat{\mathbf{g}}_j$를 
weighted version으로 교체하고 $m_j$ factor를 유지한다.


---

# Part XIII. Dataset Construction

## 13.1 Preferred Construction

각 generated sentence $s_j$에 대해 atomic facts를 추출하고, 
각 atomic fact가 Wikipedia context 하에서 supported인지 판정한다 (FActScore pipeline).

$$
m_j = \#\{\text{atomic facts in } s_j\}, \qquad K_j = \#\{\text{supported atomic facts in } s_j\}
$$

학습 데이터:

$$
D = \{(s_j, \{\mathbf{z}_\ell\}_{\ell \in s_j}, K_j, m_j)\}_{j=1}^N
$$

## 13.2 Sentence-level Alignment

FActScore 원래 pipeline은 response-level로 atomic facts를 추출한다.
본 연구는 sentence-level model이므로 각 atomic fact가 어느 문장에 속하는지 align해야 한다.

선택지:

1. **Sentence-level atomic fact extraction** (권장): 문장별로 독립적으로 atomic facts 추출 → 
   각 fact를 Wikipedia context와 대조 → supported 판정.
2. Response-level extraction 후 sentence mapping heuristic.
3. Response-level model로 formulation 변경.

본 연구의 목적이 sentence-level uncertainty이므로 1번이 가장 타당하다.
Response-level count를 sentence-level label로 나누는 것은 권장하지 않는다.

## 13.3 $m_j = 0$ 처리

Atomic fact가 없는 문장 (예: 순수 transitional sentence)은 
count likelihood에 정보를 제공하지 않으므로 supervised likelihood에서 제외한다.

$$
m_j = 0 \quad \Rightarrow \quad \ell_j(\theta) = 0
$$

단, inference 시에는 $\mu_j$와 epistemic uncertainty를 계산할 수 있다. 
Label이 없어서 학습에는 쓰지 않지만, 예측은 가능하다.


---

# Part XIV. Evaluation Protocol

## 14.1 Primary Metrics (Ratio-level)

Main target은 factuality ratio $U_j = K_j / m_j$이다.

- **Binomial NLL**: $-\log p(K_j \mid m_j, \hat\mu_j)$
  (model comparison에서 $\log\binom{m_j}{K_j}$는 constant이므로 ranking 불변. 
  Reporting metric으로는 full NLL 권장.)
- **Ratio Brier score**: $(U_j - \hat\mu_j)^2$
- **Calibration curve** for $U_j$
- **Predictive interval coverage** for $U_j$
- **Spearman / Pearson correlation** between $U_j$ and $\hat\mu_j$

## 14.2 Strict Factuality Metrics (Secondary)

Strict factuality label: $A_j = \mathbf{1}\{K_j = m_j\}$.

Score: $1 - \hat\mu_j^{m_j}$ (higher = more likely to contain error).

- **AUROC** for detecting $A_j = 0$
- **AUPRC** for factual error detection
- **Rejection curve**
- **Selective prediction accuracy**

## 14.3 Uncertainty Quality Metrics

1. **Epistemic quality**:
   - Training size $N$에 따른 평균 epistemic 감소
   - Entity split에서 OOD entity에 대한 epistemic 증가 여부
   - Low $U_j$ 문장에서 epistemic이 커지는지

2. **Aleatoric quality**:
   - $m_j$가 작을수록 $U_j$의 observation noise가 큰지
   - Predictive interval coverage가 $m_j$별로 잘 맞는지

3. **Total uncertainty**:
   - $U_j$ prediction interval coverage
   - Negative log likelihood
   - Calibration by uncertainty bins

## 14.4 Baselines

| Method | 유형 | 비용 |
|---|---|---|
| Token Entropy | Single-pass, token-level | 1× forward |
| Factuality Probe (Han et al., 2025) | Single-pass, claim-level, point estimate | 1× forward + re-encode |
| Logistic Regression (sentence feature) | Single-pass, point estimate | 1× forward |
| P(True) / Verbalized Confidence | Single-pass, self-assessment | 1× forward |
| SelfCheckGPT | Multi-sample, NLI-based | $m$× forward |
| LUQ (Zhang et al., 2024) | Multi-sample, NLI-based | $m$× forward |
| Semantic Entropy (Kuhn et al., 2023) | Multi-sample, clustering | $m$× forward |
| SC+VC / Graph Uncertainty | Multi-sample, graph-based | $m$× forward |

Han et al.의 probe는 가장 직접적인 비교 대상이다: 같은 hidden state 기반 접근이되
point estimate만 제공하므로, Bayesian extension의 부가가치를 보이는 ablation이 된다.

## 14.5 Ablations

| 실험 | 목적 |
|---|---|
| Layer weights $\boldsymbol\alpha$ | 학습된 분포 |
| Projection dim $p$ | {16, 32, 64, 128} |
| Feature components | entropy/top-1 유무 |
| Prior $\sigma_0$ | Calibration 영향 |
| **Bayesian vs Point** | $\hat{\boldsymbol{\Sigma}}$ 유무 |
| Fisher-type vs True Hessian | Laplace precision 비교 |
| 1차 근사 vs MC epistemic | Laplace quality |
| Clipped vs Unclipped boundary | $\mu_j$의 boundary 이탈 frequency |
| Logit vs Gaussian aux regression | §8 보조 모델의 target transformation |
| Uniform vs Attention weight | §XII |
| Training size $N$ | Epistemic scaling |
| Laplace-EB correction | Exploratory |
| Fisher scoring iterations $K$ | 수렴성 |
| Damping schedule | Line search 대안들 |
| **Ours vs Factuality Probe** | Bayesian UQ의 부가가치 (ECE, rejection curve) |
| Layer $\boldsymbol\alpha$ vs single layer 14 | Han et al. finding 재현 및 multi-layer 이점 검증 |
| Generation-time vs re-encoded hidden state | Hidden state 소스의 영향 |
| **Binomial vs Bernoulli** | $m_j = 1$ fallback과 비교하여 count-aware model의 이점 |
| **$m_j$ weighting $\alpha$** | $\alpha \in \{0, 0.5, 1\}$로 Fisher precision의 $m_j^\alpha$ weighting 비교 |
| **Overdispersion diagnostic** | Binomial variance vs 관측 variance 비교 |


---

# Part XV. 모델 약점 및 대응

## XV.1 Atomic Facts의 Conditional Independence 가정

Binomial model은 같은 문장 안의 atomic facts가 conditionally independent라고 가정한다.

$$
B_{j1}, \ldots, B_{jm_j} \mid \theta \quad \text{independent}
$$

하지만 실제로는 한 문장 안의 facts가 서로 강하게 dependent할 수 있다.
하나의 잘못된 entity나 date가 여러 atomic facts를 동시에 틀리게 만들 수 있다.

**대응**:
- 본문에서 binomial model을 **working likelihood**로 명시한다.
- Overdispersion diagnostic을 수행한다 (§XV.2).
- 필요하면 beta-binomial extension을 ablation으로 둔다.

## XV.2 Overdispersion

Binomial variance는

$$
\text{Var}(K_j \mid \theta, m_j) = m_j \mu_j(1-\mu_j)
$$

실제 variance가 이보다 크면 overdispersion이다. 간단한 확장은 quasi-binomial이다.

$$
\text{Var}(K_j \mid \theta, m_j) = \phi \, m_j \mu_j(1-\mu_j), \qquad \phi \ge 1
$$

또는 beta-binomial model을 사용할 수 있다.
다만 beta-binomial은 수식과 구현이 복잡해지므로, 
main model은 binomial로 두고 overdispersion은 limitation 또는 ablation으로 처리한다.

**Overdispersion이 posterior에 미치는 영향**:
Fisher precision의 $m_j$ factor가 실제 정보량보다 과대 추정되어
posterior가 과도하게 tight해지고 epistemic uncertainty를 under-estimate할 수 있다.
$m_j^\alpha$ weighting ($\alpha < 1$)이 이를 부분적으로 완화한다.

## XV.3 $m_j$ Weighting의 Dominance

Binomial likelihood는 $m_j$가 큰 문장에 더 큰 weight를 준다.
이는 통계적으로 자연스럽지만, 긴 문장 또는 fact-dense sentence가 objective를 지배할 수 있다.

**대응**:

$$
\ell_j^{(\alpha)}(\theta) = m_j^\alpha \left[ U_j \log \mu_j(\theta) + (1 - U_j)\log(1-\mu_j(\theta)) \right], \qquad \alpha \in [0, 1]
$$

기본값은 $\alpha = 1$이고, ablation으로 $\alpha = 0$ 또는 $\alpha = 1/2$를 비교한다.

## XV.4 $m_j$와 문장 난이도의 Confounding

Atomic fact가 많은 문장은 더 복잡하고, 복잡한 문장은 hallucination probability도 높을 수 있다.
이때 $m_j$는 단순한 trial count가 아니라 sentence complexity signal이기도 하다.

**대응**:
- $m_j$를 likelihood의 trial count로 쓰되, feature에는 직접 넣지 않는 것을 기본으로 한다.
- 별도 ablation에서 $m_j$ 또는 $\log m_j$를 sentence-level feature로 추가해본다.
- Feature로 넣으면 length bias가 커질 수 있으므로 calibration을 반드시 확인한다.

## XV.5 기존 약점 (v7에서 유지)

| 약점 | 대응 |
|---|---|
| 평균 가정의 단순함 | Attention-weighted 확장 (§XII) |
| White-box 접근 필요 | Open-source LLM의 실용성 |
| Laplace 근사 quality | §7.5 monitoring: true Hessian PD 확인, MC 비교 |
| Global concavity/MAP 유일성 미보장 | Local numerical check로 보완 |
| Clipped objective와 이론의 gap | Boundary 빈도 monitoring (§7.2) |
| FActScore 단일 데이터셋 | 추가 domain 실험 |
| MAP plug-in vs 진짜 EB | Laplace-EB correction as exploratory ablation |
| **i.i.d. 가정 violation (근본적)** | Entity 내 문장 종속성은 근본적 제약. Entity split은 leakage 완화일 뿐이며 training 내부 dependence는 해결 안 됨. BvM은 heuristic으로만 원용. |
| 1차 근사 aleatoric 음수 | Clipping + MC 검증 |
| $\text{Attr}_\ell$ signed | $\text{LocalEpi}_\ell$ 병기 |
| Spectral/clarification 연결 약함 | Conceptual analogy로 명시, empirical correlation만 보고 |
| Computational cost | Offline caching (entropy/top-1), selected layer만 저장, pilot feasibility |
| Feature extractor 근거 부족 | 본 논문 ablation으로 검증 |
| Laplace-EB 구현 미완성 | Exploratory로 표기, 결과에 따라 결정 |
| Bounded target을 Gaussian으로 모델링 | §8.1: Logit-transformed Gaussian 사용 |
| Fallback update의 ad-hoc | Damped Fisher scoring (§7.4)로 명확화 |
| True vs Fisher Hessian monitoring 혼재 | 둘 다 보고, clipped objective임을 명시 |
| **Hidden state probe의 novelty가 Han et al.에 의해 선점됨** | Novelty를 probe 자체가 아닌 Bayesian UQ에 둔다. Han et al.을 출발점으로 명시하고, uncertainty decomposition + calibration + attribution + count-aware observation이 contribution임을 강조. |
| Han et al. 대비 AUROC 향상이 미미할 수 있음 | AUROC는 ranking metric이므로 point estimate로도 충분. 핵심 비교 축은 ECE (calibration), binomial NLL, 그리고 epistemic uncertainty의 정보량. |


---

# Part XVI. 결과 수준 정리

| 번호 | 수준 | 내용 |
|---|---|---|
| Prop 1 | Proposition | Binomial gradient closed-form |
| — | Definition | Binomial Fisher-type posterior precision |
| Prop 2 | Proposition | True Hessian decomposition (binomial) |
| Prop 3 | Proposition | Ratio-level 1차 근사 uncertainty decomposition |
| Prop 4 | Proposition | $L_j = 1$, $m_j = 1$ reduction |
| Prop 5 | Proposition | Information inequality |
| Prop 6 | Proposition | Weighted case structure |
| **Thm 1** | **Theorem** | Exact Law of Total Variance (ratio level) |
| **Thm 2** | **Theorem** | Additive signed attribution |
| Rmk 1 | Remark | Posterior concentration heuristic (with $m_j$ effective information) |
| Rmk 2 | Remark | Probit shrinkage mechanism |
