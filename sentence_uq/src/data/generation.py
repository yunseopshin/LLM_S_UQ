"""LLM generation + hidden-state extraction for Bayesian sentence-level UQ.

Phase 1-1. Model-agnostic: works with any HuggingFace ``AutoModelForCausalLM``.
Generates responses while saving **per-token hidden states and logits across
multiple selected layers**, captured during generation (not via post-hoc
re-encoding as in Han et al., 2025).

Saved tensor layout for one prompt (single ``.pt`` file)::

    {
        "text":           str,              # decoded generated response
        "prompt":         str,              # original user prompt
        "prompt_text":    str,              # exact string fed to the tokenizer
                                            # (after any chat template)
        "prompt_ids":     LongTensor (P,),
        "token_ids":      LongTensor (T,),  # generated tokens only
        "hidden_states":  fp16 Tensor (T, len(selected_layers), D),
        "logits":         fp16 Tensor (T, V),
        "selected_layers": list[int],
        "model_config":   {"name", "hidden_dim", "num_hidden_layers",
                           "selected_layers", "vocab_size"},
        "dataset":        str,              # "factscore_bio" | "longfact"
        "meta":           dict,             # entity / topic / prompt_idx
        "finished":       bool,             # True if EOS hit, else max_new_tokens
    }

Notes
-----
- ``hidden_states[t, k, :]`` is the hidden state at layer
  ``selected_layers[k]`` produced when generated token ``t`` was the
  input — i.e. the state conditioned on ``x_{≤t}``.
- ``logits[t]`` is the distribution that **sampled** ``token_ids[t]``
  (Phase 7-3 fix 1). Equivalently, ``logits[t]`` is the model output
  at the previous position, conditioned on ``x_{<t}``. So
  ``entropy[t] = H(x_t | x_{<t})`` and
  ``top1_prob[t] = p^{(1)}(x_t | x_{<t})`` are the **generation-time**
  uncertainty of the current token, not next-token uncertainty.
- HuggingFace returns ``output.hidden_states`` as a tuple of length
  ``num_hidden_layers + 1``: index 0 is the embedding-layer output and
  index ``num_hidden_layers`` is the final transformer-block output. The
  selectors below index directly into that tuple.
- Hidden states and logits are stored in **fp16**; computation runs in the
  model's native dtype (typically bf16 or fp16) on GPU.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants & small utilities
# ---------------------------------------------------------------------------

#: Default target number of layers when ``selected_layers`` is ``null``.
DEFAULT_TARGET_LAYERS: int = 8

#: Filename-unsafe characters; replaced with ``_`` in ``_safe_filename``.
_UNSAFE_FNAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(s: str, max_len: int = 200) -> str:
    """Make a string safe for use as a filename across common filesystems."""
    cleaned = _UNSAFE_FNAME_RE.sub("_", s.strip()).strip("._")
    if not cleaned:
        cleaned = "unnamed"
    return cleaned[:max_len]


def _ensure_dir(path: str | os.PathLike) -> Path:
    """Create the directory at ``path`` (and parents) and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------


def auto_select_layers(
    num_hidden_layers: int, target_count: int = DEFAULT_TARGET_LAYERS
) -> list[int]:
    """Return evenly spaced layer indices that include 0 and ``num_hidden_layers``.

    HuggingFace ``output_hidden_states`` returns a tuple of length
    ``num_hidden_layers + 1``: index 0 is the embedding-layer output and
    index ``num_hidden_layers`` is the final transformer-block output. This
    helper picks roughly ``target_count`` indices from ``[0, num_hidden_layers]``.

    Algorithm
    ---------
    ``step = max(1, round(num_hidden_layers / target_count))``, then
    ``range(0, num_hidden_layers + 1, step)``, with ``num_hidden_layers`` appended
    if it isn't already the last element.

    Examples
    --------
    >>> auto_select_layers(32, 8)
    [0, 4, 8, 12, 16, 20, 24, 28, 32]
    >>> auto_select_layers(28, 8)
    [0, 4, 8, 12, 16, 20, 24, 28]

    Parameters
    ----------
    num_hidden_layers : int
        ``model.config.num_hidden_layers``.
    target_count : int
        Desired number of layers. The returned list may have ``target_count``
        or ``target_count + 1`` entries depending on divisibility.

    Returns
    -------
    list[int]
        Sorted layer indices in ``[0, num_hidden_layers]``.

    Raises
    ------
    ValueError
        If ``num_hidden_layers < 1`` or ``target_count < 2``.
    """
    if num_hidden_layers < 1:
        raise ValueError(
            f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
        )
    if target_count < 2:
        raise ValueError(f"target_count must be >= 2, got {target_count}")

    step = max(1, round(num_hidden_layers / target_count))
    layers = list(range(0, num_hidden_layers + 1, step))
    if layers[-1] != num_hidden_layers:
        layers.append(num_hidden_layers)
    return layers


def resolve_selected_layers(
    num_hidden_layers: int, selected_layers: Optional[Sequence[int]]
) -> list[int]:
    """Return a validated list of layer indices.

    If ``selected_layers`` is ``None``, calls :func:`auto_select_layers`.
    Otherwise validates each entry lies in ``[0, num_hidden_layers]``.

    Raises
    ------
    ValueError
        If any provided index is out of range or duplicated.
    """
    if selected_layers is None:
        return auto_select_layers(num_hidden_layers)

    out = [int(i) for i in selected_layers]
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate layer indices: {out}")
    for i in out:
        if i < 0 or i > num_hidden_layers:
            raise ValueError(
                f"Layer index {i} out of range [0, {num_hidden_layers}]"
            )
    return sorted(out)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a HuggingFace causal LM + tokenizer and report its config.

    Parameters
    ----------
    model_name : str
        HuggingFace hub name or local path. **Required** — no default.
    device : str
        Target device. With ``"cuda"`` we let ``device_map="auto"`` choose
        placement; with ``"cpu"`` we keep everything on CPU.
    dtype : torch.dtype
        Model weight dtype (e.g. ``torch.float16``, ``torch.bfloat16``).

    Returns
    -------
    model : transformers.PreTrainedModel
        In ``eval`` mode, with ``output_hidden_states=True`` set on the config.
    tokenizer : transformers.PreTrainedTokenizerBase
        ``pad_token`` is set to ``eos_token`` if missing.
    model_info : dict
        ``{"model_name", "hidden_dim", "num_hidden_layers", "vocab_size"}``.

    Notes
    -----
    Reads ``hidden_size``, ``num_hidden_layers``, ``vocab_size`` from
    ``model.config`` — never hard-coded. Per CLAUDE.md, this keeps the
    pipeline model-agnostic across Llama / Gemma / Mistral / Qwen, etc.
    """
    if not model_name:
        raise ValueError("model_name is required (no default).")

    # Local import so the module is importable without `transformers`
    # available (useful for purely-static tests).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "output_hidden_states": True,
    }
    if device == "cuda" and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.output_hidden_states = True
    model.eval()

    if device == "cpu" or not torch.cuda.is_available():
        model = model.to("cpu")

    model_info = {
        "model_name": model_name,
        "hidden_dim": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "vocab_size": int(model.config.vocab_size),
    }
    return model, tokenizer, model_info


# ---------------------------------------------------------------------------
# Prompt construction (dispatches on dataset tag)
# ---------------------------------------------------------------------------


def make_prompt(item: dict[str, Any]) -> str:
    """Build the user-facing prompt string for a split record.

    - ``factscore_bio`` entries become ``"Tell me a bio of {entity}."``
      (re-derived; the split also carries a pre-rendered ``prompt`` field).
    - ``longfact`` entries use ``item["prompt"]`` verbatim.

    Falls back to ``item["prompt"]`` if the dataset tag is missing/unknown.
    """
    ds = item.get("dataset")
    if ds == "factscore_bio":
        entity = item.get("entity")
        if entity:
            return f"Tell me a bio of {entity}."
        return item["prompt"]
    if ds == "longfact":
        return item["prompt"]
    return item["prompt"]


def apply_chat_template_if_available(tokenizer: Any, user_prompt: str) -> str:
    """Return the tokenizer-rendered chat string, or ``user_prompt`` unchanged.

    Instruction-tuned models (Llama-3-Instruct, Gemma-IT, Qwen-Instruct, ...)
    ship a chat template; we apply it so that the model sees the same prompt
    format it was trained on. For raw base models we fall back to the prompt.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if apply is None or not chat_template:
        return user_prompt
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


# ---------------------------------------------------------------------------
# Generation loop with hidden-state capture
# ---------------------------------------------------------------------------


def _collect_eos_ids(tokenizer: Any) -> list[int]:
    """Gather stop-token IDs: ``eos_token_id`` (+ Llama-3 ``<|eot_id|>`` if any)."""
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.append(eos)
    elif isinstance(eos, (list, tuple)):
        ids.extend(int(x) for x in eos if x is not None)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for tok in ("<|eot_id|>",):
            try:
                tid = convert(tok)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
    return ids


def _sample_token(
    logits: Tensor, temperature: float, top_p: float, do_sample: bool
) -> Tensor:
    """Sample a single token id from a ``(1, V)`` logits tensor.

    Numerics run in fp32 (CLAUDE.md: "always compute numerics in fp32").
    Returns a ``LongTensor`` of shape ``(1,)`` on the logits device.
    """
    if (not do_sample) or temperature <= 0:
        return torch.argmax(logits, dim=-1).view(1)

    scaled = logits.float() / float(temperature)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep the smallest set whose cumulative mass >= top_p (nucleus).
        mask = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(1)


@torch.no_grad()
def generate_with_hidden_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    selected_layers: Sequence[int],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    do_sample: bool = True,
    apply_chat_template: bool = True,
    store_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Generate a response while capturing per-token hidden states + logits.

    Implements a manual generation loop with KV cache (``past_key_values``)
    so that intermediate hidden states are accessible — ``model.generate``
    does not expose them per step.

    Algorithm
    ---------
    1. Tokenize ``prompt`` (after the chat template, when available).
    2. Prefill forward pass on the prompt; seed sampling from the final logits.
    3. Loop up to ``max_new_tokens``:

       - Forward the just-sampled token through the model with cached KV.
       - Record its hidden states (selected layers) and logits.
       - Sample the next token (temperature + nucleus); stop on EOS / ``<|eot_id|>``.

    Math reference
    --------------
    Each saved ``hidden_states[t, k, :]`` ∈ ℝ^D feeds the Phase 2-1 feature
    ``z_ℓ = [W · Σ_l α_l h_ℓ^(l), entropy_ℓ, top1_ℓ]`` (CLAUDE.md Core Math).
    ``logits[t, :]`` underwrites the cached entropy / top-1 scalars of Phase 1-3.

    Parameters
    ----------
    model, tokenizer
        Outputs of :func:`load_model`.
    prompt : str
        Raw user prompt (chat template applied internally if available).
    selected_layers : Sequence[int]
        Layer indices into the ``output.hidden_states`` tuple.
    max_new_tokens : int
        Hard cap on generated tokens.
    temperature, top_p : float
        Sampling parameters. ``temperature <= 0`` or ``do_sample=False``
        forces greedy decoding.
    do_sample : bool
        If ``False``, take ``argmax`` of logits each step.
    apply_chat_template : bool
        Run ``tokenizer.apply_chat_template`` when the tokenizer has one.
    store_dtype : torch.dtype
        Dtype used for the returned ``hidden_states`` / ``logits`` tensors.
        Defaults to fp16 (CLAUDE.md: "Store hidden states in fp16").

    Returns
    -------
    dict
        See module docstring for the schema. Tensors live on CPU.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    device = next(model.parameters()).device
    selected = list(selected_layers)
    num_layers = int(model.config.num_hidden_layers)
    for i in selected:
        if i < 0 or i > num_layers:
            raise ValueError(
                f"selected_layers index {i} out of range [0, {num_layers}]"
            )

    prompt_text = (
        apply_chat_template_if_available(tokenizer, prompt)
        if apply_chat_template
        else prompt
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids: Tensor = enc["input_ids"].to(device)

    # Prefill ------------------------------------------------------------
    prefill = model(
        input_ids=prompt_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = prefill.past_key_values
    # Logits at the last prompt position predict the first generated token.
    # Phase 7-3 fix 1: these are also the logits that *sample* the first
    # generated token, so they are what should be stored as ``logits[0]``.
    prev_logits = prefill.logits[:, -1, :]
    next_token = _sample_token(prev_logits, temperature, top_p, do_sample)

    eos_ids = set(_collect_eos_ids(tokenizer))

    gen_token_ids: list[int] = []
    gen_hidden_per_layer: list[list[Tensor]] = [[] for _ in selected]
    gen_logits: list[Tensor] = []
    finished = False

    for _ in range(max_new_tokens):
        token_id_int = int(next_token.item())
        if token_id_int in eos_ids:
            finished = True
            break

        # Phase 7-3 fix 1: store the logits that PRODUCED this token
        # (generation-time distribution conditioned on x_{<t}) — not the
        # logits the model emits *after* processing it.
        gen_logits.append(
            prev_logits[0].detach().to("cpu", dtype=store_dtype)
        )

        step = model(
            input_ids=next_token.view(1, 1),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = step.past_key_values

        # step.hidden_states is a tuple of (1, 1, D) tensors, one per layer.
        # These are the states *after* processing the current token
        # (conditioning x_{≤t}).
        for k, layer_idx in enumerate(selected):
            h = step.hidden_states[layer_idx][0, 0, :].detach()
            gen_hidden_per_layer[k].append(h.to("cpu", dtype=store_dtype))
        gen_token_ids.append(token_id_int)

        # step.logits[:, -1, :] now predicts the NEXT token; carry it
        # forward as ``prev_logits`` so the next iteration stores it as
        # ``gen_logits[t+1]``.
        prev_logits = step.logits[:, -1, :]
        next_token = _sample_token(prev_logits, temperature, top_p, do_sample)

    T = len(gen_token_ids)
    if T == 0:
        hidden_states = torch.empty(
            (0, len(selected), int(model.config.hidden_size)), dtype=store_dtype
        )
        logits_out = torch.empty(
            (0, int(model.config.vocab_size)), dtype=store_dtype
        )
    else:
        per_layer_stacked = [
            torch.stack(layer_list, dim=0) for layer_list in gen_hidden_per_layer
        ]
        hidden_states = torch.stack(per_layer_stacked, dim=1)  # (T, K, D)
        logits_out = torch.stack(gen_logits, dim=0)  # (T, V)

    token_ids_t = torch.tensor(gen_token_ids, dtype=torch.long)
    text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    return {
        "text": text,
        "prompt": prompt,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids[0].detach().to("cpu"),
        "token_ids": token_ids_t,
        "hidden_states": hidden_states,
        "logits": logits_out,
        "selected_layers": selected,
        "finished": finished,
    }


# ---------------------------------------------------------------------------
# Saving / batch generation
# ---------------------------------------------------------------------------


def save_generation(
    record: dict[str, Any],
    out_path: str | os.PathLike,
    model_info: dict[str, Any],
    selected_layers: Sequence[int],
    dataset_tag: str,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist a single generation record to ``out_path`` (``.pt``).

    The payload combines the generation outputs with a ``model_config`` block
    so that downstream code can re-derive dimensions without consulting the
    original YAML / model.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": record["text"],
        "prompt": record["prompt"],
        "prompt_text": record.get("prompt_text", record["prompt"]),
        "prompt_ids": record["prompt_ids"],
        "token_ids": record["token_ids"],
        "hidden_states": record["hidden_states"],
        "logits": record["logits"],
        "selected_layers": list(selected_layers),
        "model_config": {
            "name": model_info["model_name"],
            "hidden_dim": int(model_info["hidden_dim"]),
            "num_hidden_layers": int(model_info["num_hidden_layers"]),
            "vocab_size": int(model_info["vocab_size"]),
            "selected_layers": list(selected_layers),
        },
        "dataset": dataset_tag,
        "meta": dict(meta or {}),
        "finished": record.get("finished", False),
    }
    torch.save(payload, out)
    return out


def _output_path_for(
    item: dict[str, Any], factscore_dir: Path, longfact_dir: Path
) -> Path:
    """Compute the canonical ``.pt`` location for a split record."""
    ds = item["dataset"]
    if ds == "factscore_bio":
        name = _safe_filename(item["entity"])
        return factscore_dir / f"{name}.pt"
    if ds == "longfact":
        topic = _safe_filename(item["topic"])
        idx = int(item.get("prompt_idx", 0))
        return longfact_dir / topic / f"{idx:03d}.pt"
    raise ValueError(f"Unknown dataset tag: {ds!r}")


def batch_generate(
    items: Iterable[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    model_info: dict[str, Any],
    selected_layers: Sequence[int],
    factscore_dir: str | os.PathLike,
    longfact_dir: str | os.PathLike,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    do_sample: bool = True,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Generate + save one ``.pt`` per item, with resume support.

    Parameters
    ----------
    items
        Iterable of split records (FActScore / LongFact dicts).
    model, tokenizer, model_info
        From :func:`load_model`.
    selected_layers
        From :func:`resolve_selected_layers`.
    factscore_dir, longfact_dir
        Output roots.
    skip_existing
        If ``True`` (default), records whose target file already exists are
        not re-generated. This is what makes Setup 1 and Setup 2 share
        FActScore generations without recomputing them.

    Returns
    -------
    dict
        ``{"generated": int, "skipped": int, "errors": [(item, msg), ...]}``.
    """
    fs_dir = _ensure_dir(factscore_dir)
    lf_dir = _ensure_dir(longfact_dir)

    items_list = list(items)
    iterator: Iterable[dict[str, Any]] = items_list
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(items_list, desc="generate", unit="item")
        except ImportError:
            pass

    generated = 0
    skipped = 0
    errors: list[tuple[dict[str, Any], str]] = []

    for item in iterator:
        try:
            out_path = _output_path_for(item, fs_dir, lf_dir)
        except ValueError as exc:
            errors.append((item, str(exc)))
            continue

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            prompt = make_prompt(item)
            rec = generate_with_hidden_states(
                model,
                tokenizer,
                prompt,
                selected_layers=selected_layers,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            meta = {
                k: item[k]
                for k in ("entity", "topic", "prompt_idx")
                if k in item
            }
            save_generation(
                rec,
                out_path,
                model_info=model_info,
                selected_layers=selected_layers,
                dataset_tag=item["dataset"],
                meta=meta,
            )
            generated += 1
        except Exception as exc:  # noqa: BLE001 - log & continue, don't kill batch
            errors.append((item, repr(exc)))

    return {"generated": generated, "skipped": skipped, "errors": errors}


def write_dataset_metadata(
    out_dir: str | os.PathLike,
    model_info: dict[str, Any],
    selected_layers: Sequence[int],
    generation_config: dict[str, Any],
    dataset_tag: str,
    items: Iterable[dict[str, Any]],
) -> Path:
    """Write a ``metadata.json`` summarising the generation run for a dataset."""
    out = _ensure_dir(out_dir)
    items_list = list(items)
    payload = {
        "dataset": dataset_tag,
        "model": {
            "name": model_info["model_name"],
            "hidden_dim": int(model_info["hidden_dim"]),
            "num_hidden_layers": int(model_info["num_hidden_layers"]),
            "vocab_size": int(model_info["vocab_size"]),
        },
        "selected_layers": list(selected_layers),
        "generation": dict(generation_config),
        "num_items": len(items_list),
    }
    path = out / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


__all__ = [
    "DEFAULT_TARGET_LAYERS",
    "auto_select_layers",
    "resolve_selected_layers",
    "load_model",
    "make_prompt",
    "apply_chat_template_if_available",
    "generate_with_hidden_states",
    "save_generation",
    "batch_generate",
    "write_dataset_metadata",
]
