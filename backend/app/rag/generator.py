from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def load_llm(model_id: str):
    device = _best_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Some models need pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    # Use fp16 only on cuda
    if device == "cuda":
        kwargs.update(dict(torch_dtype=torch.float16, device_map="auto"))
    elif device == "mps":
        kwargs.update(dict(torch_dtype=torch.float16))

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _context_window(model: Any, tokenizer: Any) -> int:
    """Best-effort context window for a local HF model.

    Many tokenizers report a huge sentinel for model_max_length. We clamp in that case.
    """
    window = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
    if not window:
        window = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if not window:
        window = 2048
    if window > 32768:
        window = 4096
    return window


def _split_system_user(prompt: str) -> Tuple[Optional[str], str]:
    """Split a prompt built as: <system>\n\n<user>.

    This keeps backwards compatibility with build_prompt().
    """
    parts = prompt.split("\n\n", 1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return None, prompt


def _encode_with_chat_template(tokenizer, messages, max_length: int):
    # Newer transformers support truncation/max_length directly.
    try:
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        # Fallback: render to string and tokenize with truncation.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]


def build_prompt(question: str, context: str, system_prompt: str, style: str = "citations") -> str:
    if style == "citations":
        return (
            f"{system_prompt}\n\n"
            f"Контекст (используй только его, обязательно ссылайся на источники):\n{context}\n\n"
            f"Вопрос: {question}\n"
            f"Ответ (кратко, по делу, с цитатами [1], [2] ...):"
        )
    return (
        f"{system_prompt}\n\nКонтекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    )


def generate_text(
    model_id: str,
    prompt: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
) -> str:
    model, tokenizer, device = load_llm(model_id)

    # Respect the model context window (leave space for generated tokens + template overhead)
    max_input_tokens = max(256, _context_window(model, tokenizer) - int(max_new_tokens) - 64)

    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template") and "<|" not in prompt:
        sys, user = _split_system_user(prompt)
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": user})
        else:
            messages.append({"role": "user", "content": user})

        input_ids = _encode_with_chat_template(tokenizer, messages, max_length=max_input_tokens).to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.strip()


def generate_text_with_stats(
    model_id: str,
    prompt: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
) -> Dict[str, int | str]:
    """Generate text and return token counts.

    This is useful for UI metrics. Cost is not computed here (local HF models).
    """
    model, tokenizer, device = load_llm(model_id)

    max_input_tokens = max(256, _context_window(model, tokenizer) - int(max_new_tokens) - 64)

    # tokenize
    if hasattr(tokenizer, "apply_chat_template") and "<|" not in prompt:
        sys, user = _split_system_user(prompt)
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": user})
        else:
            messages.append({"role": "user", "content": user})

        input_ids = _encode_with_chat_template(tokenizer, messages, max_length=max_input_tokens).to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    input_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[-1] :]
    output_tokens = int(gen_ids.shape[-1])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
