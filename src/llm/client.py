"""Provider-agnostic chat completion wrapper.

LLM_PROVIDER in .env picks the backend: "openai" (default), "anthropic",
"gemini", or "ollama" (local, no key, needs `ollama serve` running with a
model pulled, e.g. `ollama pull llama3.2`). Each SDK is imported inside its
own function so the app doesn't crash on import if other providers' packages
aren't installed.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    if PROVIDER == "openai":
        return _chat_openai(system_prompt, user_prompt, temperature)
    if PROVIDER == "anthropic":
        return _chat_anthropic(system_prompt, user_prompt, temperature)
    if PROVIDER == "gemini":
        return _chat_gemini(system_prompt, user_prompt, temperature)
    if PROVIDER == "ollama":
        return _chat_ollama(system_prompt, user_prompt, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def _chat_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def _chat_anthropic(system_prompt: str, user_prompt: str, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


def _chat_ollama(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Runs against a local Ollama server through its OpenAI-compatible
    endpoint. Needs `ollama serve` running and the model already pulled."""
    from openai import OpenAI

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    client = OpenAI(base_url=base_url, api_key="ollama")  # the SDK requires a key string even though Ollama ignores it
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def _chat_gemini(system_prompt: str, user_prompt: str, temperature: float) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt, generation_config={"temperature": temperature})
    return resp.text
