# query_variations.py
"""
Query variation generator 

- Works with DeepSeek (default) or any OpenAI-compatible endpoint.
- Reads configuration from environment variables.
- Gracefully degrades to returning the original query on errors/missing deps.

Environment variables (set in .env):
  # Choose a provider (optional; default: deepseek)
  #   Options: deepseek | openai | custom
  QUERY_VARIANT_PROVIDER=deepseek

  # API key (required if you actually want to call a provider)
  LLM_API_KEY=sk-...

  # Model name (optional; sensible defaults if omitted)
  # DeepSeek: deepseek-chat
  # OpenAI:   gpt-4o-mini (example)
  # Custom:   whatever your server exposes
  LLM_MODEL_NAME=deepseek-chat

  # Base URL (optional; defaults per provider)
  # DeepSeek default: https://api.deepseek.com
  # OpenAI default:   https://api.openai.com/v1
  # Custom:           e.g., http://localhost:8000/v1
  LLM_BASE_URL=https://api.deepseek.com
"""

from __future__ import annotations
import os
from typing import List

# Lazy import so this module doesn’t require openai unless actually used
def _get_openai_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "The 'openai' package is not installed. Add 'openai>=1.0.0' to requirements "
            "or disable query variations."
        ) from e
    return OpenAI(api_key=api_key, base_url=base_url)


_SYSTEM_PROMPT = (
    "You are a medical search expert specializing in diabetes. Generate concise, "
    "keyword-rich dense retrieval queries optimized for bi-encoder embedding models "
    "(e.g., BGE encoder).\n"
    "Rules:\n"
    "1) Limit to essential keywords — no filler words (e.g., 'what is the').\n"
    "2) Use clinical terms (e.g., 'DPN' for 'diabetic neuropathy') AND layman synonyms.\n"
    "3) Include therapy types where relevant (e.g., 'pharmacological', 'lifestyle interventions').\n"
    "4) Prioritize brevity and specificity (<12 words).\n"
    "5) Return only the queries, one per line (no numbering or explanations)."
)


def _provider_defaults(provider: str) -> tuple[str, str]:
    """
    Return (base_url, model_name) defaults per provider if envs are not set.
    """
    provider = (provider or "deepseek").lower()
    if provider == "deepseek":
        return ("https://api.deepseek.com", "deepseek-chat")
    if provider == "openai":
        return ("https://api.openai.com/v1", "gpt-4o-mini")
    # custom
    return (os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"), os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"))


def generate_query_variations(
    user_query: str,
    n_variations: int = 3,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> List[str]:
    """
    Generate multiple dense-retrieval-friendly rephrasings of `user_query`.

    If API credentials or the client library are unavailable, returns [user_query].

    Args:
        user_query: The original user query.
        n_variations: Desired number of variations (clamped to 1..20).
        provider: 'deepseek' | 'openai' | 'custom' (default: env QUERY_VARIANT_PROVIDER or 'deepseek')
        api_key:   Override API key (default: env LLM_API_KEY)
        base_url:  Override base URL (default: provider-specific)
        model_name:Override model name (default: provider-specific)
    """
    # Early sanity
    user_query = (user_query or "").strip()
    if not user_query:
        return []

    # Clamp n
    n = max(1, min(int(n_variations or 1), 20))

    # Resolve config from env if not passed
    provider = (provider or os.getenv("QUERY_VARIANT_PROVIDER") or "deepseek").lower()
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    default_base, default_model = _provider_defaults(provider)
    base_url = base_url or os.getenv("LLM_BASE_URL", default_base)
    model_name = model_name or os.getenv("LLM_MODEL_NAME", default_model)

    # If no key provided, return original only (public-safe default)
    if not api_key:
        return [user_query]

    # Instantiate client (won’t import openai unless needed)
    try:
        client = _get_openai_client(base_url=base_url, api_key=api_key)
    except Exception as e:
        # Missing dependency or misconfiguration – fail gracefully
        print(f"[query_variations] Client init failed: {e}")
        return [user_query]

    # Compose messages
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Original query: '{user_query}'\nGenerate {n} rephrasings for dense retrieval.",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[query_variations] API call failed: {e}")
        return [user_query]

    # Parse one-query-per-line output; tolerate bullets or commas
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [p.strip() for p in lines[0].split(",") if p.strip()]

    # Clean bullet markers and empty lines
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("-•0123456789. ").strip()
        if ln:
            cleaned.append(ln)

    # Deduplicate while preserving order; always include original first
    unique = []
    seen = set()
    def add(x: str):
        if x not in seen:
            seen.add(x)
            unique.append(x)

    add(user_query)
    for q in cleaned:
        add(q)

    # Truncate to 1 + n (original + n variations)
    return unique[: 1 + n]


# ----------------------------------------------------------------------
# Backwards-compatible shim
# ----------------------------------------------------------------------
def generate_query_variations_with_deepseek(
    api_key: str,
    user_query: str,
    n_variations: int = 2,
    model_name: str = "deepseek-chat",
):
    """
    Backwards-compatible wrapper for older code.
    """
    return generate_query_variations(
        user_query=user_query,
        n_variations=n_variations,
        provider="deepseek",
        api_key=api_key or os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        model_name=model_name or os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
    )
