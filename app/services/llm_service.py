"""
LLM Service — Provider abstraction layer.

Switch between Gemini, OpenAI, or Groq by changing LLM_PROVIDER in .env.
This is the ONLY file with provider-specific code.
"""

from langchain_core.language_models import BaseChatModel
from app.config import settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the full-power LLM for extraction tasks.

    Supported providers:
        LLM_PROVIDER=gemini  →  Google Gemini Flash
        LLM_PROVIDER=openai  →  OpenAI GPT-4o-mini
        LLM_PROVIDER=groq    →  Groq Llama 3.3 70B
    """
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    elif settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{settings.LLM_PROVIDER}'. "
            f"Set LLM_PROVIDER to 'gemini', 'openai', or 'groq' in your .env file."
        )


def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get a lightweight, fast LLM for classification/routing tasks.

    Uses a smaller model optimized for speed over deep reasoning.
    Falls back to the full-power model for providers without a fast tier.

    Supported:
        groq    →  Llama 3.1 8B Instant (~3x faster than 70B)
        others  →  Falls back to the standard model
    """
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_FAST_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )
    else:
        # Other providers don't have a fast tier — use the standard model
        return get_llm(temperature=temperature)
