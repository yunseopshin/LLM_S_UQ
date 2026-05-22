# Sentence-Level Bayesian Uncertainty Quantification for LLMs

LLM(Llama-3-8B-Instruct)의 hidden state에서 Bayesian posterior를 구성하여 문장별 factuality uncertainty를 single forward pass로 추정한다.
기존 sampling-based 방법(LUQ, Semantic Entropy)은 10× 생성이 필요하지만, 본 방법은 1×로 comparable한 성능을 목표로 한다.

## 설치

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 실행

```bash
bash scripts/run_pilot.sh
```

## 디렉토리 구조

```
sentence_uq/
├── configs/          # default.yaml, pilot.yaml
├── src/
│   ├── data/         # 데이터 생성, 문장 분리, annotation
│   ├── features/     # Feature extractor
│   ├── models/       # Bayesian 모델, Fisher scoring
│   ├── train/        # Bilevel trainer
│   ├── inference/    # Predictive + decomposition
│   ├── baselines/    # 비교 baseline
│   ├── evaluation/   # Metrics
│   └── utils/        # I/O, 로깅, 디버깅
├── scripts/          # 실행 스크립트
├── tests/            # 단위 테스트
├── data/             # raw, generations, processed, cache
└── results/          # 실험 결과
```
