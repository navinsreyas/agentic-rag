"""
Flexible provider configuration for LLM and embedding models.
"""

import os
from typing import Optional, Union
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import native Groq support (requires pydantic-ai[groq])
try:
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.models.groq import GroqModel
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


def _is_groq(model_choice: Optional[str] = None) -> bool:
    """Return True when the active provider is Groq."""
    if os.getenv('LLM_PROVIDER', '').lower() == 'groq':
        return True
    base_url = os.getenv('LLM_BASE_URL', '')
    if 'groq.com' in base_url:
        return True
    # Deliberately NOT inferring from the model name: names like 'llama' /
    # 'mixtral' / 'gemma' are also served by non-Groq providers, so only an
    # explicit LLM_PROVIDER=groq or a groq.com base URL counts.
    return False


def get_llm_model(model_choice: Optional[str] = None) -> Union[OpenAIModel, "GroqModel"]:
    """
    Get LLM model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured model (GroqModel when LLM_PROVIDER=groq, OpenAIModel otherwise)
    """
    # Chained `or` (rather than os.getenv's default arg) so the result is a plain
    # str: it also treats an empty LLM_CHOICE as unset, which is what we want.
    llm_choice: str = model_choice or os.getenv('LLM_CHOICE') or 'gpt-4-turbo-preview'
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'ollama')

    if _is_groq(model_choice) and _GROQ_AVAILABLE:
        groq_provider = GroqProvider(api_key=api_key)
        return GroqModel(llm_choice, provider=groq_provider)

    openai_provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(llm_choice, provider=openai_provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    
    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    base_url = os.getenv('EMBEDDING_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('EMBEDDING_API_KEY', 'ollama')
    
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def get_ingestion_model() -> Union[OpenAIModel, "GroqModel"]:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).

    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = os.getenv('INGESTION_LLM_CHOICE')

    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()

    return get_llm_model(model_choice=ingestion_choice)


def fallback_enabled() -> bool:
    """Return True if a fallback model is configured via environment variables."""
    return bool(
        os.getenv('FALLBACK_LLM_PROVIDER')
        and os.getenv('FALLBACK_LLM_API_KEY')
    )


def get_fallback_model() -> Optional[OpenAIModel]:
    """
    Build the fallback OpenAI model from FALLBACK_LLM_* env vars.

    Returns None if fallback is not configured (FALLBACK_LLM_PROVIDER or
    FALLBACK_LLM_API_KEY are unset), so callers should check fallback_enabled()
    or guard against None before using the result.
    """
    api_key = os.getenv('FALLBACK_LLM_API_KEY', '')
    model_choice = os.getenv('FALLBACK_LLM_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('FALLBACK_LLM_BASE_URL', 'https://api.openai.com/v1')

    if not api_key:
        return None

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(model_choice, provider=provider)


def is_fallback_error(exc: Exception) -> bool:
    """
    Return True for errors where retrying with a different model may succeed.

    Covers:
    - Tool-call validation failures (Groq/Llama malformed tool calls)
    - Rate-limit / quota errors (429, "Too Many Requests")
    - Context-length errors ("Request too large")
    """
    msg = str(exc).lower()
    return any(phrase in msg for phrase in [
        "failed to call a function",
        "attempted to call tool",
        "tool_call_validation",
        "tool call validation",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "request too large",
        "context length",
        " 429",
    ])


# Provider information functions
def get_llm_provider() -> str:
    """Get the LLM provider name."""
    return os.getenv('LLM_PROVIDER', 'openai')


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return os.getenv('EMBEDDING_PROVIDER', 'openai')


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if configuration is valid
    """
    required_vars = [
        'LLM_API_KEY',
        'LLM_CHOICE',
        'EMBEDDING_API_KEY',
        'EMBEDDING_MODEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": os.getenv('LLM_CHOICE'),
        "llm_base_url": os.getenv('LLM_BASE_URL'),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "embedding_base_url": os.getenv('EMBEDDING_BASE_URL'),
        "ingestion_model": os.getenv('INGESTION_LLM_CHOICE', 'same as main'),
    }