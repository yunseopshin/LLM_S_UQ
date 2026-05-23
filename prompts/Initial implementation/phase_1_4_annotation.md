# Phase 1-4 — Factuality Annotation (Updated)

Implement `src/data/annotation.py`.

**Purpose**:
For each sentence, extract atomic facts and judge supported/not-supported
to produce **binomial counts (K_j, m_j)** — not binary F_j.

**Annotation pipeline**: Following Han et al. (2025).
π_aux = GPT-4o-mini (claim decomposition, revision, subjectivity filtering).
Retrieval-based scoring (Wikipedia or knowledge source comparison).

**Reference**: Han et al. (2025) Stage 1 pipeline. 
Code at https://github.com/JThh/fact-probe.

**Requirements**:

1. Function `decompose_to_atomic_facts(sentence, entity_or_topic, api_client)`:
   - Input: sentence (str), entity or topic context (str), API client
   - Ask π_aux to decompose sentence into atomic facts
   - Include claim revision (pronoun resolution, etc.)
   - Include subjectivity filtering (remove subjective claims)
   - Returns: list of str (atomic facts)

2. Function `judge_atomic_fact(fact, knowledge_context, api_client)`:
   - Input: atomic fact (str), retrieved knowledge context (str)
   - Ask π_aux whether the fact is supported by the context
   - Returns: 1 (supported) or 0 (not supported)

3. Function `retrieve_knowledge(entity_or_topic, dataset_type)`:
   - **Retrieval strategy depends on dataset_type**:
     * `"factscore_bio"`: Fetch entity's Wikipedia article via Wikipedia API
     * `"longfact"`: Follow Jiang et al. (2024) pipeline —
       search Wikipedia for topic-related articles, or web search fallback
   - Returns: knowledge context (str)

4. Function `annotate_sentence(sentence, entity_or_topic, dataset_type, api_client)`:
   - Run full annotation for one sentence
   - Returns: dict `{"m_j": int, "K_j": int, "claims": [{"text": ..., "label": 0|1}, ...]}`
   
   **Example output**:
   ```json
   {
     "sentence": "Einstein developed the theory of relativity and won the Nobel Prize in 1921.",
     "m_j": 3,
     "K_j": 3,
     "claims": [
       {"text": "Einstein developed the theory of relativity.", "label": 1},
       {"text": "Einstein won the Nobel Prize in Physics.", "label": 1},
       {"text": "Einstein won the Nobel Prize in 1921.", "label": 1}
     ]
   }
   ```

5. Function `annotate_batch(processed_data, dataset_type, api_client)`:
   - Annotate all sentences, respecting rate limits
   - Resume support (skip already-annotated sentences)

**Script `scripts/02_annotate_factuality.py`**:

```
python scripts/02_annotate_factuality.py --setup 2 --config configs/default.yaml
```

- `--setup` determines which dataset's sentences to annotate:
  * Setup 1: FActScore (test) + LongFact (train) — both datasets
  * Setup 2: FActScore only
  * Setup 3: LongFact only
- Save results to `data/processed/{dataset}/annotated.json`
  with `m_j`, `K_j`, and `claims` fields per sentence

**Key change from original phase_1_4**:
- OLD: `F_j ∈ {0, 1}` (sentence-level binary label)
- NEW: `(K_j, m_j)` (sentence-level binomial counts)
- This aligns with the binomial observation model in research_document_v8 Part II §2.1

**Cost estimate**: 
- FActScore 183 entities × ~15 sentences × ~3 claims/sentence = ~8,200 claims ≈ $16
- LongFact 1,140 prompts × ~10 sentences × ~3 claims/sentence = ~34,200 claims ≈ $68
- Total (Setup 1, both datasets): ~$84

**Important**:
- Use temperature=0 for deterministic judgments
- Filter very short or meaningless sentences before annotation
- Guard against prompt injection from sentence content
