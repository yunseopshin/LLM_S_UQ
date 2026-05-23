# Phase 1-1 — LLM Generation + Hidden State Extraction (Updated)

Implement `src/data/generation.py`.

**Purpose**:
Generate responses from an LLM and save per-token hidden states and logits.
**Supports both datasets (FActScore-Bio and LongFact-Objects).**
**Model-agnostic**: works with any HuggingFace causal LM. Model name comes from config.

**Critical difference from Han et al. (2025)**: Han et al. re-encode extracted claims 
through the LLM and use only the last token's single-layer hidden state. 
We save generation-time hidden states for ALL tokens across multiple layers.
This is the fundamental architectural difference — we capture the model's internal state 
at the moment of generation, not a post-hoc re-encoding.

**Requirements**:

1. Function `load_model(model_name: str, device="cuda", dtype=torch.float16)`:
   - Load model and tokenizer via AutoModelForCausalLM / AutoTokenizer
   - output_hidden_states=True, device_map="auto"
   - **Auto-detect model properties** from model.config:
     * hidden_dim = model.config.hidden_size
     * num_hidden_layers = model.config.num_hidden_layers
     * vocab_size = model.config.vocab_size
   - Returns: model, tokenizer, model_info dict {"hidden_dim", "num_hidden_layers", "vocab_size", "model_name"}
   - **No default model_name** — must be provided from config

2. Function `auto_select_layers(num_hidden_layers: int, target_count: int = 8) -> list[int]`:
   - Generate evenly spaced layer indices including layer 0 and the last layer
   - Example: num_hidden_layers=32, target=8 → [0, 4, 8, 12, 16, 20, 24, 28, 32]
   - Example: num_hidden_layers=28, target=8 → [0, 4, 8, 12, 16, 20, 24, 28]
   - Used when config has `selected_layers: null`

3-5. Functions generate_with_hidden_states, save_generation, batch_generate — unchanged.

**Changed: save_generation must include model_config metadata**:
   - Each .pt file stores: hidden_states, logits, token_ids, text, **model_config** (name, hidden_dim, num_hidden_layers, selected_layers)
   - This ensures reproducibility when switching models

**Changed: Script `scripts/01_generate_data.py`**:

```
python scripts/01_generate_data.py --setup 2 --config configs/default.yaml
```

- `--setup` selects the experimental setup (1, 2, or 3)
- Load train/val/test prompt lists from `data/splits/setup_{N}.json`
- **Generate only the datasets needed for the given setup**:
  * Setup 1: LongFact prompts (train) + FActScore entities (test)
  * Setup 2: FActScore entities only
  * Setup 3: LongFact prompts only
- Skip already-generated .pt files (resume support)
- Output directories separated by dataset:
  * `data/generations/factscore_bio/{entity_name}.pt`
  * `data/generations/longfact/{topic}/{prompt_idx:03d}.pt`
- Save `metadata.json` per dataset

**Prompt construction logic**:

```python
def make_prompt(item):
    if item["dataset"] == "factscore_bio":
        # FActScore-Bio: entity -> biography prompt
        return f"Tell me a bio of {item['entity']}."
    elif item["dataset"] == "longfact":
        # LongFact: use prompt as-is
        return item["prompt"]
```

**Important**:
- Use manual generation loop with KV cache (past_key_values), NOT model.generate().
  model.generate() doesn't expose intermediate hidden states easily.
- Set output_hidden_states=True in each forward call.
- Watch GPU memory — move to CPU for long sequences if needed.
- Output .pt format is identical for both datasets (hidden_states, logits, token_ids, text).
- Setup 1 requires generation on both datasets, so it takes the longest.
- Setup 2 and Setup 1 share the same FActScore entities — skip duplicates if 
  `data/generations/factscore_bio/` already has them.
