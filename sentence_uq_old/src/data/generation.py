"""
Phase 1-1: LLM response generation with hidden state extraction.

Generates text from Llama-3-8B-Instruct using a manual token-by-token loop
so that per-token hidden states and logits can be captured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B-Instruct").
        device: Target device string; used only as a hint — device_map="auto" handles placement.
        dtype: Weight dtype string, one of {"float16", "bfloat16", "float32"}.

    Returns:
        (model, tokenizer) tuple.  model is in eval mode with output_hidden_states=True.
    """
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def generate_with_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    selected_layers: Optional[List[int]] = None,
) -> Dict:
    """Generate a response token-by-token, collecting hidden states and logits.

    Uses greedy decoding (argmax) and KV-cache for efficiency.  Stops at
    EOS or when max_new_tokens is reached.

    Math notation (matching CLAUDE.md):
        h_ℓ^(l) — hidden state at generated token position ℓ, layer l.

    Args:
        model: A loaded causal LM (output_hidden_states=True).
        tokenizer: Matching tokenizer.
        prompt: Input text to condition on.
        max_new_tokens: Maximum number of tokens to generate.
        selected_layers: Indices of transformer layers whose hidden states to save.
            Layer 0 is the embedding output; higher indices are transformer blocks.
            If None, saves all layers.

    Returns:
        dict with keys:
            "text"          : str — decoded generated text (skip_special_tokens=True).
            "token_ids"     : LongTensor  (T,)
            "hidden_states" : fp16 Tensor (T, len(selected_layers), hidden_dim)
            "logits"        : fp16 Tensor (T, vocab_size)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Place on the same device as the model's first parameter
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)

    if selected_layers is None:
        # number of layers = config.num_hidden_layers + 1 (embedding)
        num_layers = model.config.num_hidden_layers + 1
        selected_layers = list(range(num_layers))

    past_kv = None
    current_input = input_ids  # full prompt on first step, single token afterwards

    generated_token_ids: List[int] = []
    hidden_states_list: List[torch.Tensor] = []  # each: (len(selected_layers), hidden_dim)
    logits_list: List[torch.Tensor] = []          # each: (vocab_size,)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
            )

        past_kv = outputs.past_key_values

        # outputs.hidden_states: tuple of (batch, seq, hidden_dim) per layer
        # We want the last position of each selected layer.
        # Move to CPU immediately to keep GPU memory pressure low.
        hs = torch.stack(
            [outputs.hidden_states[l][:, -1, :].cpu().to(torch.float16) for l in selected_layers],
            dim=1,
        )  # (1, len(selected_layers), hidden_dim)
        hidden_states_list.append(hs.squeeze(0))  # (len(selected_layers), hidden_dim)

        next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        logits_list.append(next_token_logits.cpu().to(torch.float16).squeeze(0))  # (vocab_size,)

        next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_token_ids.append(token_id)
        current_input = next_token.unsqueeze(0)  # (1, 1) for next step

    if not generated_token_ids:
        # Edge case: model immediately produced EOS
        empty = torch.zeros(0, dtype=torch.long)
        return {
            "text": "",
            "token_ids": empty,
            "hidden_states": torch.zeros(0, len(selected_layers), model.config.hidden_size, dtype=torch.float16),
            "logits": torch.zeros(0, model.config.vocab_size, dtype=torch.float16),
        }

    token_ids_tensor = torch.tensor(generated_token_ids, dtype=torch.long)  # (T,)
    hidden_states_tensor = torch.stack(hidden_states_list, dim=0)           # (T, L, D)
    logits_tensor = torch.stack(logits_list, dim=0)                         # (T, vocab_size)

    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return {
        "text": text,
        "token_ids": token_ids_tensor,
        "hidden_states": hidden_states_tensor,
        "logits": logits_tensor,
    }


def save_generation(result: Dict, save_path: str | Path) -> None:
    """Save a generation result dict to a .pt file.

    hidden_states and logits are already fp16; token_ids and text are saved as-is.

    Args:
        result: Dict returned by generate_with_hidden_states.
        save_path: Destination file path (should end in .pt).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, save_path)


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    save_dir: str | Path,
    selected_layers: Optional[List[int]] = None,
    max_new_tokens: int = 512,
    entities: Optional[List[str]] = None,
) -> None:
    """Generate responses for a list of prompts, saving each to save_dir/{idx:05d}.pt.

    Supports resuming: indices whose .pt file already exists are skipped.
    Writes a metadata.json with per-entry {idx, prompt, entity}.

    Args:
        model: Loaded causal LM.
        tokenizer: Matching tokenizer.
        prompts: List of prompt strings.
        save_dir: Directory in which to write .pt files and metadata.json.
        selected_layers: Layer indices to save (passed through to generate_with_hidden_states).
        max_new_tokens: Max tokens per generation.
        entities: Optional list of entity names parallel to prompts, stored in metadata.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = save_dir / "metadata.json"
    # Load existing metadata if resuming
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata: List[Dict] = json.load(f)
        existing_ids = {entry["idx"] for entry in metadata}
    else:
        metadata = []
        existing_ids = set()

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        save_path = save_dir / f"{idx:05d}.pt"

        if idx in existing_ids and save_path.exists():
            continue  # Resume: skip already-generated entries

        result = generate_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            selected_layers=selected_layers,
        )
        save_generation(result, save_path)

        entry: Dict = {"idx": idx, "prompt": prompt}
        if entities is not None:
            entry["entity"] = entities[idx]
        metadata.append(entry)

        # Persist metadata after each successful generation (crash-safe)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
