"""
LLM Factory - Unified interface for multiple LLM providers.

Supports:
- Anthropic (Claude models)
- OpenAI (GPT models)
- OpenRouter (any model via OpenAI-compatible API)
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    **kwargs,
) -> BaseChatModel:
    """
    Create an LLM instance based on the specified provider.

    Args:
        provider: LLM provider ('anthropic', 'openai', 'openrouter').
                  Defaults to LLM_PROVIDER env var.
        model: Model name. Defaults to LLM_MODEL env var.
        temperature: Sampling temperature. Defaults to LLM_TEMPERATURE env var.
        **kwargs: Additional arguments passed to the LLM constructor.

    Returns:
        A configured LangChain chat model instance.

    Raises:
        ValueError: If provider is unknown or required API key is missing.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))

    logger.info(
        f"Creating LLM: provider={provider}, model={model}, temperature={temperature}"
    )

    if provider == "anthropic":
        return _create_anthropic_llm(model, temperature, **kwargs)
    elif provider == "openai":
        return _create_openai_llm(model, temperature, **kwargs)
    elif provider == "openrouter":
        return _create_openrouter_llm(model, temperature, **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'anthropic', 'openai', or 'openrouter'."
        )


def _create_anthropic_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create an Anthropic Claude model."""
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
        )

    return ChatAnthropic(
        model=model, temperature=temperature, api_key=api_key, max_tokens=4096, **kwargs
    )


def _create_openai_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create an OpenAI GPT model."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI provider"
        )

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, **kwargs)


def _create_openrouter_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create a model via OpenRouter (OpenAI-compatible API)."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required for OpenRouter provider"
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv(
                "OPENROUTER_REFERER", "https://github.com/manus-agent"
            ),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Manus Agent"),
        },
        **kwargs,
    )


def get_available_providers() -> list[str]:
    """
    Get list of available providers based on configured API keys.

    Returns:
        List of provider names that have API keys configured.
    """
    available = []

    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    if os.getenv("OPENROUTER_API_KEY"):
        available.append("openrouter")

    return available


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print(f"Available providers: {get_available_providers()}")

    try:
        llm = create_llm()
        response = llm.invoke("Say 'Hello, I am working!' in exactly those words.")
        print(f"LLM Response: {response.content}")
    except ValueError as e:
        print(f"Error: {e}")
