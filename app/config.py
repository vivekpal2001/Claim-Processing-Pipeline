"""
Application configuration.

To switch LLM providers, change LLM_PROVIDER in your .env file:
    LLM_PROVIDER=gemini  →  Uses Google Gemini Flash
    LLM_PROVIDER=openai  →  Uses OpenAI GPT-4o-mini
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """App settings loaded from environment variables."""

    # LLM Provider — change this ONE variable to switch providers
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # OpenAI (only needed if LLM_PROVIDER=openai)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Groq (only needed if LLM_PROVIDER=groq)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


settings = Settings()
