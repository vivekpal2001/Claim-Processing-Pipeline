"""
LLM Service — Provider abstraction layer.

Switch between Gemini, OpenAI, or Groq by changing LLM_PROVIDER env var.
This is the ONLY file that contains provider-specific code.
All agents use the returned ChatModel's uniform LangChain interface.
"""

from langchain_core.language_models import BaseChatModel
from app.config import settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get LLM instance based on configured provider.

    Returns a LangChain ChatModel with identical interface
    regardless of the underlying provider.

    Supported providers:
        LLM_PROVIDER=gemini  →  Google Gemini Flash
        LLM_PROVIDER=openai  →  OpenAI GPT-4o-mini
        LLM_PROVIDER=groq    →  Groq Llama 3.3 70B (recommended — fast & free)
    """
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )
    elif settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{settings.LLM_PROVIDER}'. "
            f"Set LLM_PROVIDER to 'gemini', 'openai', or 'groq' in your .env file."
        )
