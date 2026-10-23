"""
Project Jaina - Sovereign Intelligent Failover Router V1.0 (Mythos Edition)
Orquestador dinámico de inferencia en caliente (Hot-Failover).
Permite conmutar transparentemente entre modelos locales y enjambre en la nube
(Groq, Nvidia NIM, Mistral, Together, DeepSeek, Fireworks, OpenRouter) sin reiniciar jamás worldserver.exe.
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional

from ai_core.ai_keystore import default_keystore
from ai_core.cloud_providers import (
    CLOUD_PROVIDER_REGISTRY,
    get_cloud_provider,
    BaseCloudProvider,
)
from ai_core.endpoint_auditor import EndpointAuditor

logger = logging.getLogger("jaina_ai.router")

_router_lock = threading.RLock()


class FailoverRouter:
    """Enrutador de IA con tolerancia a fallos, auditoría empírica y conmutación en caliente."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.mode = self._resolve_mode()
        self.active_provider: str = ""
        self.active_model: str = ""
        self._provider_priority: List[str] = ["groq", "nvidia", "mistral", "together", "deepseek", "openrouter"]
        self._initialize_routing()

    def _resolve_mode(self) -> str:
        prov = str(self.config.get("LLMChatter.Provider", "local")).strip().lower()
        if prov in ("cloud", "enjambre", "swarm"):
            return "cloud"
        if prov in ("hybrid", "hibrido", "auto"):
            return "hybrid"
        if prov in CLOUD_PROVIDER_REGISTRY:
            return "cloud"
        return "local"

    def _initialize_routing(self):
        """Inicializa los objetivos primarios de enrutamiento basados en configuración y salud."""
        prov = str(self.config.get("LLMChatter.Provider", "local")).strip().lower()
        model = str(self.config.get("LLMChatter.Model", "auto")).strip()

        if self.mode == "cloud":
            # Si se especificó un proveedor directo (ej: groq o nvidia)
            if prov in CLOUD_PROVIDER_REGISTRY:
                self.active_provider = prov
                p_obj = get_cloud_provider(prov)
                if p_obj and p_obj.available_models:
                    self.active_model = model if (model and model.lower() != "auto") else p_obj.available_models[0]
            else:
                # Modo enjambre cloud: seleccionar el primer proveedor con API key válida
                self._pick_best_healthy_cloud_target()
        elif self.mode == "hybrid":
            self.active_provider = "hybrid"
            self.active_model = model or "auto"
        else:
            self.active_provider = "local"
            self.active_model = model or "auto"

        logger.info(
            f"[FailoverRouter] Modo: {self.mode.upper()} | Proveedor Activo: {self.active_provider} | Modelo: {self.active_model}"
        )

    def _pick_best_healthy_cloud_target(self) -> bool:
        """Selecciona el modelo en la nube más rápido y certificado como ALIVE."""
        healthy = EndpointAuditor.get_healthy_models()
        if healthy:
            best = healthy[0]
            self.active_provider = best["provider"].lower()
            self.active_model = best["model"]
            return True

        # Si aún no hay auditoría en caché, auditar en caliente los proveedores con llave
        configured_keys = default_keystore.list_configured(config=self.config)
        for p_id in self._provider_priority:
            if p_id in configured_keys:
                p_cls = CLOUD_PROVIDER_REGISTRY.get(p_id)
                if p_cls:
                    provider = p_cls()
                    key = provider.get_api_key(self.config)
                    for m in provider.available_models:
                        res = EndpointAuditor.audit_model(provider.name, provider.base_url, key, m, timeout=3.5)
                        if res.get("status") == "ALIVE":
                            self.active_provider = p_id
                            self.active_model = m
                            logger.info(
                                f"[FailoverRouter] Enlace primario en caliente establecido: {provider.name} -> {m} ({res.get('latency_ms')}ms)"
                            )
                            return True
        return False

    def dispatch(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 350,
        temperature: float = 0.8,
        timeout: float = 12.0,
        is_short_chatter: bool = False,
        local_client: Optional[Any] = None,
        local_model: Optional[str] = None,
    ) -> Optional[str]:
        """Despacha la petición aplicando tolerancia a fallos y conmutación automática en caliente."""
        with _router_lock:
            # 1. Modo Híbrido: derivar según naturaleza del evento
            if self.mode == "hybrid":
                # Si es un emote o diálogo corto, intentar primero inferencia local rápida si está disponible
                if is_short_chatter and local_client:
                    try:
                        res = self._call_local(local_client, local_model or "auto", messages, max_tokens, temperature, timeout=5.0)
                        if res:
                            return res
                    except Exception as e:
                        logger.debug(f"[FailoverRouter] Inferencia local lenta o fallida en híbrido ({e}); conmutando a nube...")

                # Si no es corto o la inferencia local falló, ejecutar a través de nube
                return self._dispatch_cloud_with_failover(messages, max_tokens, temperature, timeout)

            # 2. Modo Cloud exclusivo
            if self.mode == "cloud":
                return self._dispatch_cloud_with_failover(messages, max_tokens, temperature, timeout)

            # 3. Modo Local exclusivo
            if local_client:
                try:
                    return self._call_local(local_client, local_model or "auto", messages, max_tokens, temperature, timeout)
                except Exception as e:
                    # Conmutación de emergencia a la nube si el motor local cayó
                    logger.warning(f"[FailoverRouter] Motor local no responde ({e}). Activando respaldo en la nube en caliente...")
                    return self._dispatch_cloud_with_failover(messages, max_tokens, temperature, timeout)

            return None

    def _call_local(self, client: Any, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, timeout: float) -> Optional[str]:
        """Ejecuta petición a través del cliente OpenAI local (LM Studio / Ollama)."""
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if response and response.choices:
            content = response.choices[0].message.content
            return content.strip() if content else None
        return None

    def _dispatch_cloud_with_failover(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> Optional[str]:
        """Bucle de ejecución tolerante a fallos para el enjambre de nube."""
        attempts = 0
        max_attempts = 4

        while attempts < max_attempts:
            attempts += 1

            if not self.active_provider or not self.active_model:
                if not self._pick_best_healthy_cloud_target():
                    logger.error("[FailoverRouter] No se encontraron proveedores cloud saludables ni con claves válidas.")
                    return None

            provider_obj = get_cloud_provider(self.active_provider)
            if not provider_obj:
                self.active_provider = ""
                continue

            try:
                t0 = time.monotonic()
                result = provider_obj.chat_completion(
                    messages=messages,
                    model=self.active_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    config=self.config,
                )
                if result:
                    latency = int((time.monotonic() - t0) * 1000)
                    logger.debug(f"[FailoverRouter] Respuesta exitosa [{provider_obj.name} / {self.active_model}] en {latency}ms")
                    return result
            except Exception as exc:
                old_prov = provider_obj.name
                old_model = self.active_model
                
                # Extraer código de error HTTP si está disponible
                code = getattr(exc, "code", 500)
                reason = getattr(exc, "reason", str(exc))
                err_str = (str(exc) + " " + reason).lower()

                # Detectar si el fallo es a nivel de proveedor (clave, cuota, rate-limit o 401/403/429)
                is_provider_wide = code in (401, 403, 429) or any(
                    k in err_str for k in ("rate limit", "rate_limit", "quota", "forbidden", "unauthorized", "credit", "billing", "429")
                )

                # Registrar falla en el auditor para actualizar la tabla de salud en caliente
                EndpointAuditor.mark_model_failed(old_prov, old_model, code=code, reason=reason)

                logger.warning(
                    f"[FailoverRouter] [HOT-FAILOVER ACTIVADO] Falla en {old_prov} ({old_model}): {exc}. Conmutando en caliente..."
                )

                # Si es un fallo a nivel de proveedor o ya van varios fallos en el mismo, excluir todo el proveedor
                exclude_p = old_prov if (is_provider_wide or attempts >= 2) else None

                switched = self._pick_next_available_target(
                    exclude_provider=exclude_p,
                    exclude_model=old_model
                )
                if not switched:
                    # Intento desesperado: buscar cualquier otro proveedor configurado
                    switched = self._pick_next_available_target()
                    if not switched:
                        logger.error("[FailoverRouter] No quedan más candidatos vivos disponibles en el enjambre.")
                        break

                logger.info(
                    f"[FailoverRouter] Conmutación en caliente exitosa: {old_prov} ({old_model}) -> {self.active_provider} ({self.active_model})"
                )

        return None

    def _pick_next_available_target(self, exclude_provider: Optional[str] = None, exclude_model: Optional[str] = None) -> bool:
        """Busca el próximo modelo vivo excluyendo los que fallaron recientemente."""
        healthy = EndpointAuditor.get_healthy_models()
        for cand in healthy:
            prov = cand["provider"].lower()
            mod = cand["model"]
            if exclude_provider and prov == exclude_provider.lower():
                continue
            if exclude_model and mod == exclude_model:
                continue
            self.active_provider = prov
            self.active_model = mod
            return True

        # Si no hay en caché saludable, buscar en el siguiente proveedor configurado
        configured = default_keystore.list_configured(config=self.config)
        for p_id in self._provider_priority:
            if exclude_provider and p_id == exclude_provider.lower():
                continue
            if p_id in configured:
                p_cls = CLOUD_PROVIDER_REGISTRY.get(p_id)
                if p_cls:
                    provider = p_cls()
                    key = provider.get_api_key(self.config)
                    for m in provider.available_models:
                        if exclude_model and m == exclude_model:
                            continue
                        res = EndpointAuditor.audit_model(provider.name, provider.base_url, key, m, timeout=3.5)
                        if res.get("status") == "ALIVE":
                            self.active_provider = p_id
                            self.active_model = m
                            return True
        return False
