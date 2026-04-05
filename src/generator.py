"""
Generator: LLM integration with real-time streaming support.

Supported backends (configured via .env):
  - "ollama"       → local Mistral / LLaMA via Ollama (preferred, no API key)
  - "huggingface"  → HuggingFace Inference API (free tier, HF_API_TOKEN required)

Environment variables (see .env.example):
  LLM_BACKEND   = ollama | huggingface
  OLLAMA_MODEL  = mistral
  OLLAMA_HOST   = http://localhost:11434
  HF_MODEL      = mistralai/Mistral-7B-Instruct-v0.2
  HF_API_TOKEN  = hf_...
"""

import os
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_BACKEND   = os.getenv("LLM_BACKEND",  "ollama")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",  "http://localhost:11434")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant") # Using supported Llama 3.1 8B

HF_MODEL      = os.getenv("HF_MODEL",     "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN  = os.getenv("HF_API_TOKEN", "")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise, helpful document assistant. "
    "Answer the user's question STRICTLY using the provided context. "
    "Do NOT use outside knowledge or make up information.\n"
    "Rules:\n"
    "1. Answer ONLY from the context below.\n"
    "2. If the context does not contain sufficient information, "
    "   say exactly: \"I couldn't find that information in the provided document.\"\n"
    "3. Be concise, factual, and well-structured.\n"
    "4. When relevant, cite the source section (e.g., 'According to Section 4...')."
)

RAG_PROMPT_TEMPLATE = """\
Context extracted from the document:
{context}

User Question:
{query}

Answer (based strictly on the context above):"""


def build_prompt(context: str, query: str) -> str:
    """Inject retrieved context and user query into the RAG prompt template."""
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)


# ── Ollama Backend ────────────────────────────────────────────────────────────

def stream_ollama(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from a locally running Ollama instance.

    Requirements:
        pip install ollama
        ollama serve          # in a separate terminal
        ollama pull mistral   # one-time download
    """
    try:
        import ollama as _ollama

        stream = _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            options={
                "temperature":  TEMPERATURE,
                "top_p":        0.9,
                "num_predict":  MAX_NEW_TOKENS,
                "repeat_penalty": 1.1,
            },
        )

        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    except ImportError:
        yield (
            "[!] Ollama package not installed.\n\n"
            "Run: `pip install ollama`\n\n"
            "Then restart the app."
        )
    except Exception as exc:
        err = str(exc)
        if "connection" in err.lower() or "refused" in err.lower():
            yield (
                "[!] Cannot connect to Ollama.\n\n"
                "Please make sure Ollama is running:\n"
                "```\nollama serve\n```\n"
                "And the model is downloaded:\n"
                "```\nollama pull mistral\n```"
            )
        else:
            yield f"[!] Ollama error: {err}"


# ── HuggingFace Backend ───────────────────────────────────────────────────────

def stream_huggingface(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from the HuggingFace Inference API (free tier).

    Requirements:
        pip install huggingface_hub
        Set HF_API_TOKEN in .env (get a free token at huggingface.co/settings/tokens)
    """
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model=HF_MODEL,
            token=HF_API_TOKEN or None,
        )

        # Mistral instruct format: <s>[INST] system\n\nuser [/INST]
        full_prompt = (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]"
        )

        for token in client.text_generation(
            full_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=max(TEMPERATURE, 0.01),   # HF requires > 0
            do_sample=True,
            stream=True,
        ):
            yield token

    except ImportError:
        yield "[!] huggingface_hub not installed. Run: `pip install huggingface_hub`"
    except Exception as exc:
        err = str(exc)
        if "401" in err or "unauthorized" in err.lower():
            yield (
                "⚠️  **HuggingFace auth error.**\n\n"
                "Set a valid `HF_API_TOKEN` in your `.env` file.\n"
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        elif "429" in err or "rate" in err.lower():
            yield "⚠️  **HuggingFace rate limit hit.** Please wait a moment and try again."
        else:
            yield f"[!] HuggingFace error: {err}"


# ── Groq Backend ──────────────────────────────────────────────────────────────

def stream_groq(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from Groq Cloud (Free tier, ultra-fast).
    
    Get a key: https://console.groq.com/
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            top_p=1,
            stream=True,
        )
        
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except ImportError:
        yield "[!] groq package not installed. Run: `pip install groq`"
    except Exception as exc:
        err = str(exc)
        if "401" in err or "unauthorized" in err.lower():
            yield "⚠️  **Groq API Key** missing or invalid in `.env`."
        else:
            yield f"[!] Groq error: {err}"


# ── Unified Streaming Interface ───────────────────────────────────────────────

def stream_response(prompt: str, backend: str = None) -> Generator[str, None, None]:
    """
    Route to the appropriate LLM backend and yield tokens.
    """
    backend = (backend or LLM_BACKEND).lower().strip()

    if backend == "ollama":
        yield from stream_ollama(prompt)
    elif backend == "groq":
        yield from stream_groq(prompt)
    elif backend in ("huggingface", "hf"):
        yield from stream_huggingface(prompt)
    else:
        yield f"[!] Unknown LLM backend: `{backend}`."


def get_model_info() -> dict:
    """Return display metadata about the active LLM configuration."""
    if LLM_BACKEND == "ollama":
        model_name_capitalized = OLLAMA_MODEL.title() if OLLAMA_MODEL else "Mistral"
        return {
            "backend":      "Ollama (Local)",
            "model":        OLLAMA_MODEL,
            "display_name": f"{model_name_capitalized} (Ollama/Local)",
        }
    elif LLM_BACKEND == "groq":
        return {
            "backend":      "Groq Cloud",
            "model":        GROQ_MODEL,
            "display_name": f"{GROQ_MODEL} (Groq Cloud)",
        }
    else:
        short = HF_MODEL.split("/")[-1]
        return {
            "backend":      "HuggingFace API",
            "model":        HF_MODEL,
            "display_name": f"{short} (HuggingFace)",
        }
