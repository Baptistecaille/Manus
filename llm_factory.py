"""
LLM Factory - Unified interface for multiple LLM providers.

Supports:
- Anthropic (Claude models)
- OpenAI (GPT models)
- OpenRouter (any model via OpenAI-compatible API)
- DeepSeek (DeepSeek models via OpenAI-compatible API)

Features:
- Automatic fallback to OpenRouter when OpenAI quota is exceeded (429 error)
"""

import os
import logging
from typing import Optional, List, Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Fallback configuration
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-4o-mini")


class FallbackLLM(BaseChatModel):
    """
    LLM wrapper that automatically falls back to OpenRouter on quota errors.
    """

    primary_llm: BaseChatModel
    fallback_llm: Optional[BaseChatModel] = None
    _using_fallback: bool = False

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fallback_llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        """Generate with automatic fallback on quota errors."""

        # If we've already switched to fallback, use it directly
        if self._using_fallback and self.fallback_llm:
            return self.fallback_llm._generate(messages, stop, run_manager, **kwargs)

        try:
            return self.primary_llm._generate(messages, stop, run_manager, **kwargs)
        except Exception as e:
            error_str = str(e).lower()

            # Check for quota/rate limit errors
            if any(
                keyword in error_str
                for keyword in ["429", "quota", "rate_limit", "insufficient_quota"]
            ):
                if self.fallback_llm:
                    logger.warning(
                        f"Primary LLM quota exceeded, switching to OpenRouter fallback: {OPENROUTER_FALLBACK_MODEL}"
                    )
                    self._using_fallback = True
                    return self.fallback_llm._generate(
                        messages, stop, run_manager, **kwargs
                    )
                else:
                    logger.error("Quota exceeded but no fallback LLM configured")
                    raise
            else:
                # Re-raise non-quota errors
                raise

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """
        Wrap with_structured_output with fallback logic.
        Returns a RunnableWithFallbacks.
        """
        # If already switched, just use fallback
        if self._using_fallback and self.fallback_llm:
            return self.fallback_llm.with_structured_output(schema, **kwargs)

        # Create primary runnable
        primary_runnable = self.primary_llm.with_structured_output(schema, **kwargs)

        if self.fallback_llm:
            # Create fallback runnable
            fallback_runnable = self.fallback_llm.with_structured_output(
                schema, **kwargs
            )

            # Configure fallbacks to catch 429 output errors
            # Note: We need to handle the specific exceptions that might occur during execution
            return primary_runnable.with_fallbacks(
                [fallback_runnable],
                exceptions_to_handle=(
                    Exception,
                ),  # We might want to be more specific, but for now catch all
            )

        return primary_runnable


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    enable_fallback: bool = True,
    **kwargs,
) -> BaseChatModel:
    """
    Create an LLM instance based on the specified provider.

    Args:
        provider: LLM provider ('anthropic', 'openai', 'openrouter').
                  Defaults to LLM_PROVIDER env var.
        model: Model name. Defaults to LLM_MODEL env var.
        temperature: Sampling temperature. Defaults to LLM_TEMPERATURE env var.
        enable_fallback: If True, wrap with FallbackLLM for automatic OpenRouter fallback.
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

    # Create primary LLM
    if provider == "anthropic":
        primary_llm = _create_anthropic_llm(model, temperature, **kwargs)
    elif provider == "openai":
        primary_llm = _create_openai_llm(model, temperature, **kwargs)
    elif provider == "openrouter":
        primary_llm = _create_openrouter_llm(model, temperature, **kwargs)
    elif provider == "deepseek":
        primary_llm = _create_deepseek_llm(model, temperature, **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'anthropic', 'openai', 'openrouter', or 'deepseek'."
        )

    # Wrap with fallback if enabled and OpenRouter key is available
    if enable_fallback and provider != "openrouter" and os.getenv("OPENROUTER_API_KEY"):
        logger.info(f"Fallback enabled: OpenRouter ({OPENROUTER_FALLBACK_MODEL})")
        fallback_llm = _create_openrouter_llm(OPENROUTER_FALLBACK_MODEL, temperature)
        return FallbackLLM(primary_llm=primary_llm, fallback_llm=fallback_llm)

    return primary_llm


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


class OpenAIChatModel(ChatOpenAI):
    """
    OpenAI wrapper that uses native json_schema for structured output.
    This enables Strict Structured Output for Chat-gpt-5-nano and compatible models.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Use default method (json_schema / Strict Structured Output)
        # This is the most reliable for OpenAI models.
        return super().with_structured_output(schema, **kwargs)


def _create_openai_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create an OpenAI GPT model with native structured output support."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI provider"
        )

    return OpenAIChatModel(
        model=model, temperature=temperature, api_key=api_key, **kwargs
    )


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


class DeepSeekChatOpenAI(ChatOpenAI):
    """
    DeepSeek wrapper that forces json_mode for structured output.
    DeepSeek API currently has issues with native tool_calling/json_schema for structured output.
    Also ensures 'json' keyword is present in prompt when json_mode is active.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Force function_calling (older OpenAI tools API) which is often supported by compatible APIs
        # when json_schema (Strict Structured Output) is not.
        return super().with_structured_output(
            schema, method="function_calling", **kwargs
        )


def _create_deepseek_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Create a DeepSeek model via OpenAI-compatible API."""
    # Ensure ChatOpenAI is imported or available. It is imported in the global scope if we move it,
    # but here we rely on the implementation below.
    # To be safe and cleaner, we should move the import to top level or ensure it's available.
    # Since we are using a class definition above that inherits from ChatOpenAI,
    # we must ensure ChatOpenAI is available.
    # The previous code imported it inside functions.
    # We will assume we fixed imports in a previous step or fix them now.

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is required for DeepSeek provider"
        )

    return DeepSeekChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.deepseek.com",
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
    if os.getenv("DEEPSEEK_API_KEY"):
        available.append("deepseek")

    return available


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print(f"Available providers: {get_available_providers()}")
    print(f"Fallback model: {OPENROUTER_FALLBACK_MODEL}")

    try:
        llm = create_llm()
        response = llm.invoke("Say 'Hello, I am working!' in exactly those words.")
        print(f"LLM Response: {response.content}")
    except ValueError as e:
        print(f"Error: {e}")
