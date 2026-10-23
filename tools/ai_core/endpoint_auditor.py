"""
Project Jaina - Sovereign Endpoint Auditor V1.0 (Mythos Edition)
Auditoría empírica proactiva de endpoints LLM (Regla 8).
Verifica la salud de los modelos mediante pings ultra-ligeros (max_tokens: 1).
Detecta HTTP 404/410 (DEPRECATED), 429 (RATE_LIMITED), y mantiene la tabla de salud caliente.
"""

import time
import json
import logging
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

from ai_core.ai_keystore import default_keystore
from ai_core.cloud_providers import CLOUD_PROVIDER_REGISTRY, _DEFAULT_USER_AGENT

logger = logging.getLogger("jaina_ai.auditor")

_health_cache: Dict[str, Dict[str, Any]] = {}
_health_lock = threading.RLock()


class EndpointAuditor:
    """Auditor empírico de modelos LLM para conmutación transparente en caliente."""

    @classmethod
    def audit_model(
        cls,
        provider_name: str,
        base_url: str,
        key: str,
        model: str,
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        """Ejecuta una petición real con max_tokens=1 para verificar la vitalidad del modelo."""
        t0 = time.monotonic()
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }).encode("utf-8")

        url = f"{base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(url, data=payload)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", _DEFAULT_USER_AGENT)

        cache_key = f"{provider_name.lower()}:{model}"

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                result = {
                    "provider": provider_name,
                    "model": model,
                    "status": "ALIVE",
                    "code": resp.status,
                    "latency_ms": elapsed_ms,
                    "last_checked": time.time(),
                }
                with _health_lock:
                    _health_cache[cache_key] = result
                return result
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if e.code in (404, 410):
                status = "DEPRECATED"
                logger.warning(
                    f"[EndpointAuditor] [REGLA 8] Modelo descontinuado detectado en {provider_name}: {model} (HTTP {e.code})"
                )
            elif e.code == 429:
                status = "RATE_LIMITED"
                logger.info(f"[EndpointAuditor] Rate limit (429) en {provider_name} ({model})")
            else:
                status = "ERROR"
            
            result = {
                "provider": provider_name,
                "model": model,
                "status": status,
                "code": e.code,
                "reason": str(e.reason),
                "latency_ms": elapsed_ms,
                "last_checked": time.time(),
            }
            with _health_lock:
                _health_cache[cache_key] = result
            return result
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            result = {
                "provider": provider_name,
                "model": model,
                "status": "OFFLINE",
                "reason": str(e),
                "latency_ms": elapsed_ms,
                "last_checked": time.time(),
            }
            with _health_lock:
                _health_cache[cache_key] = result
            return result

    @classmethod
    def audit_provider(cls, provider_id: str, config: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Audita todos los modelos configurados para un proveedor específico."""
        p_cls = CLOUD_PROVIDER_REGISTRY.get(provider_id.lower())
        if not p_cls:
            return []
        
        provider = p_cls()
        key = provider.get_api_key(config=config)
        if not key:
            return [{"model": m, "status": "SKIP", "reason": "No Key"} for m in provider.available_models]

        results = []
        for m in provider.available_models:
            res = cls.audit_model(provider.name, provider.base_url, key, m)
            results.append(res)
        return results

    @classmethod
    def audit_all_configured(cls, config: Optional[dict] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Audita empíricamente todos los proveedores que tienen una API key válida."""
        configured = default_keystore.list_configured(config=config)
        all_results = {}
        for p_id in configured:
            if p_id in CLOUD_PROVIDER_REGISTRY:
                all_results[p_id] = cls.audit_provider(p_id, config=config)
        return all_results

    @classmethod
    def get_healthy_models(cls, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna lista de modelos ALIVE ordenados por menor latencia."""
        with _health_lock:
            candidates = []
            for k, info in _health_cache.items():
                if info.get("status") == "ALIVE":
                    if provider_id and info.get("provider", "").lower() != provider_id.lower():
                        continue
                    candidates.append(info)
            # Ordenar por menor latencia de respuesta
            return sorted(candidates, key=lambda x: x.get("latency_ms", 99999))

    @classmethod
    def mark_model_failed(cls, provider_name: str, model: str, code: int = 500, reason: str = ""):
        """Marca inmediatamente un modelo como degradado o deprecado al fallar en tiempo real."""
        cache_key = f"{provider_name.lower()}:{model}"
        status = "DEPRECATED" if code in (404, 410) else ("RATE_LIMITED" if code == 429 else "ERROR")
        with _health_lock:
            _health_cache[cache_key] = {
                "provider": provider_name,
                "model": model,
                "status": status,
                "code": code,
                "reason": reason,
                "last_checked": time.time(),
            }
        logger.info(f"[EndpointAuditor] Estado en caliente actualizado: {provider_name} ({model}) -> {status}")


_auditor_thread = None
_auditor_running = False


def start_background_auditor(interval_seconds: float = 1800.0, config: Optional[dict] = None):
    """Inicia daemon en segundo plano para auditoría periódica de modelos."""
    global _auditor_thread, _auditor_running
    if _auditor_running:
        return

    def _worker():
        global _auditor_running
        _auditor_running = True
        logger.info(f"[EndpointAuditor] Worker de auditoría proactiva iniciado (intervalo: {interval_seconds}s)")
        while _auditor_running:
            try:
                report = EndpointAuditor.audit_all_configured(config=config)
                alive_count = len(EndpointAuditor.get_healthy_models())
                logger.info(f"[EndpointAuditor] Ciclo completado. Modelos vivos y saludables en enjambre: {alive_count}")
            except Exception as e:
                logger.error(f"[EndpointAuditor] Error en ciclo de auditoría: {e}")
            time.sleep(interval_seconds)

    _auditor_thread = threading.Thread(target=_worker, daemon=True, name="JainaEndpointAuditor")
    _auditor_thread.start()
