"""
Provides a unified interface for invoking different LLM backends
(OpenAI or Ollama) using a common text-based API.
"""
import os
import json
import requests
from openai import OpenAI

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


class LLMClient:
    """
    Unified client interface for interacting with different LLM providers.

    Supported providers:
    - OpenAI (via official SDK)
    - Ollama (via HTTP API)

    This class abstracts provider-specific logic so the rest
    of the application can call a single interface.
    """

    def __init__(self):
        """
        Initialize the LLM client based on the configured provider.

        Responsibilities:
        - Read provider configuration from environment variables
        - Validate required credentials
        - Initialize the appropriate client object

        Raises:
            ValueError: if required credentials are missing
        """
        self.provider = LLM_PROVIDER
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def call_text(self, prompt: str) -> str:
        """
        Returns a STRING.
        The prompt already enforces 'JSON only' in generation step.
        """
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Return exactly what the user asks. No extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content

        if self.provider == "ollama":
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
