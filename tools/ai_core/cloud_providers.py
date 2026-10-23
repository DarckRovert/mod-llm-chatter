"""
Project Jaina - Sovereign Cloud LLM Providers V1.0 (Mythos Edition)
Conectores de alta velocidad compatibles con OpenAI para inferencia en la nube:
Groq, Nvidia NIM, Mistral, DeepSeek Cloud, Together AI, Fireworks AI, OpenRouter.
Utiliza urllib.request estándar con cabecera User-Agent canónica anti-Cloudflare 1010.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

logger = logging.getLogger("jaina_ai.providers")

_DEFAULT_USER_AGENT = "mod-llm-chatter/1.0 (Project Jaina Sovereign Engine)"


class BaseCloudProvider:
    """Clase base de proveedor OpenAI-compatible en la nube."""
    name: str = "BaseCloud"
    key_id: str = ""
    base_url: str = ""
    default_context: int = 131072
    available_models: List[str] = []

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def get_api_key(self, config: Optional[dict] = None) -> Optional[str]:
        if self.api_key:
            return self.api_key
        from ai_core.ai_keystore import default_keystore
        return default_keystore.get_key(self.key_id, config=config)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 350,
        temperature: float = 0.8,
        timeout: float = 15.0,
        config: Optional[dict] = None,
    ) -> str:
        """Ejecuta una petición de Chat Completion de forma síncrona."""
        resolved_model = model or (self.available_models[0] if self.available_models else "")
        if not resolved_model:
            raise ValueError(f"[{self.name}] No se especificó ningún modelo.")

        key = self.get_api_key(config)
        if not key:
            raise PermissionError(f"[{self.name}] API Key no configurada o no disponible.")

        # Si el modelo incluye razonamiento oculto (reasoning tokens), un presupuesto menor a 200 tokens
        # causa que el modelo agote el presupuesto en su fase de pensamiento y retorne content vacío.
        # Ajustamos el límite de tokens para garantizar que la respuesta final se emita.
        is_reasoning_model = any(k in resolved_model.lower() for k in ("oss", "r1", "reason", "o1", "o3"))
        effective_max_tokens = max(max_tokens, 250) if is_reasoning_model else max_tokens

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", _DEFAULT_USER_AGENT)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choices = resp_data.get("choices", [])
                if not choices:
                    return ""
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if msg.get("reasoning_content") or msg.get("reasoning"):
                    clean_r = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                    if clean_r:
                        return clean_r
                return ""
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            logger.warning(f"[{self.name}] HTTPError {e.code} en modelo '{resolved_model}': {e.reason} | {err_body[:200]}")
            raise e
        except Exception as e:
            logger.warning(f"[{self.name}] Fallo de red en modelo '{resolved_model}': {e}")
            raise e


class GroqProvider(BaseCloudProvider):
    name = "Groq"
    key_id = "groq"
    base_url = "https://api.groq.com/openai/v1"
    available_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]


class NvidiaProvider(BaseCloudProvider):
    name = "Nvidia NIM"
    key_id = "nvidia"
    base_url = "https://integrate.api.nvidia.com/v1"
    available_models = [
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        "google/diffusiongemma-26b-a4b-it",
    ]


class MistralProvider(BaseCloudProvider):
    name = "Mistral AI"
    key_id = "mistral"
    base_url = "https://api.mistral.ai/v1"
    available_models = [
        "mistral-small-3-1",
        "mistral-large-2",
        "codestral-latest",
        "mistral-nemo",
    ]


class DeepSeekCloudProvider(BaseCloudProvider):
    name = "DeepSeek Cloud"
    key_id = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    available_models = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]


class TogetherProvider(BaseCloudProvider):
    name = "Together AI"
    key_id = "together"
    base_url = "https://api.together.xyz/v1"
    available_models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1",
    ]


class FireworksProvider(BaseCloudProvider):
    name = "Fireworks AI"
    key_id = "fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"
    available_models = [
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
    ]


class OpenRouterProvider(BaseCloudProvider):
    name = "OpenRouter"
    key_id = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    available_models = [
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-r1",
    ]


CLOUD_PROVIDER_REGISTRY: Dict[str, Any] = {
    "groq": GroqProvider,
    "nvidia": NvidiaProvider,
    "nvidia_nim": NvidiaProvider,
    "mistral": MistralProvider,
    "deepseek": DeepSeekCloudProvider,
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
    "openrouter": OpenRouterProvider,
}


def get_cloud_provider(provider_id: str, api_key: Optional[str] = None) -> Optional[BaseCloudProvider]:
    p_cls = CLOUD_PROVIDER_REGISTRY.get((provider_id or "").strip().lower())
    if p_cls:
        return p_cls(api_key=api_key)
    return None
