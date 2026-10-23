r"""
Project Jaina - Sovereign AI KeyStore V1.0 (Mythos Edition)
Almacenamiento seguro y cifrado de API keys mediante Windows DPAPI (win32crypt).
Persistencia soberana en E:\JainaCore_Server\server\data\ai_keystore.bin.
Thread-safe y completamente desacoplado de dependencias externas.
"""

import os
import json
import base64
import hashlib
import platform
import threading
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger("jaina_ai.keystore")

# Ruta canónica por defecto en disco E:\
_DEFAULT_KEYSTORE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
                 "server", "data", "ai_keystore.bin")
)

_keystore_lock = threading.RLock()

# Mapeo de proveedores conocidos y variables de entorno estándar
PROVIDER_ENV_MAP: Dict[str, List[str]] = {
    "groq": ["GROQ_API_KEY", "LLMCHATTER_GROQ_APIKEY"],
    "nvidia": ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "LLMCHATTER_NVIDIA_APIKEY"],
    "nvidia_nim": ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "LLMCHATTER_NVIDIA_APIKEY"],
    "mistral": ["MISTRAL_API_KEY", "LLMCHATTER_MISTRAL_APIKEY"],
    "deepseek": ["DEEPSEEK_API_KEY", "LLMCHATTER_DEEPSEEK_APIKEY"],
    "together": ["TOGETHER_API_KEY", "LLMCHATTER_TOGETHER_APIKEY"],
    "fireworks": ["FIREWORKS_API_KEY", "LLMCHATTER_FIREWORKS_APIKEY"],
    "xai": ["XAI_API_KEY", "LLMCHATTER_XAI_APIKEY"],
    "openrouter": ["OPENROUTER_API_KEY", "LLMCHATTER_OPENROUTER_APIKEY"],
    "openai": ["OPENAI_API_KEY", "LLMCHATTER_OPENAI_APIKEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "LLMCHATTER_ANTHROPIC_APIKEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "LLMCHATTER_GOOGLE_APIKEY"],
}


class KeyStoreCipher:
    """Cifrado y descifrado de datos con DPAPI de Windows y fallback por salt de máquina."""

    @staticmethod
    def _get_machine_salt() -> bytes:
        seed = f"{platform.node()}:{platform.machine()}:JainaSovereignKeystoreSalt"
        return hashlib.sha256(seed.encode("utf-8")).digest()

    @classmethod
    def encrypt(cls, plaintext: str) -> bytes:
        data = plaintext.encode("utf-8")
        try:
            import win32crypt
            # DPAPI con entropy machine-salt
            return win32crypt.CryptProtectData(data, "JainaCore_AI_KeyStore", cls._get_machine_salt(), None, None, 0)
        except Exception:
            # Fallback XOR cipher con key derivada
            salt = cls._get_machine_salt()
            encrypted = bytes([b ^ salt[i % len(salt)] for i, b in enumerate(data)])
            return b"XOR:" + base64.b64encode(encrypted)

    @classmethod
    def decrypt(cls, cipher_bytes: bytes) -> str:
        if not cipher_bytes:
            return ""
        if cipher_bytes.startswith(b"XOR:"):
            salt = cls._get_machine_salt()
            raw = base64.b64decode(cipher_bytes[4:])
            decrypted = bytes([b ^ salt[i % len(salt)] for i, b in enumerate(raw)])
            return decrypted.decode("utf-8", errors="ignore")
        try:
            import win32crypt
            _, decrypted = win32crypt.CryptUnprotectData(cipher_bytes, cls._get_machine_salt(), None, None, 0)
            return decrypted.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"[KeyStoreCipher] Fallo al descifrar blob DPAPI: {e}")
            return ""


class AIKeyStore:
    """Gestor unificado y soberano de claves API para Project Jaina."""

    def __init__(self, keystore_path: Optional[str] = None):
        self.keystore_path = os.path.normpath(keystore_path or _DEFAULT_KEYSTORE_PATH)
        self._ensure_dir()

    def _ensure_dir(self):
        parent = os.path.dirname(self.keystore_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def _read_raw(self) -> Dict[str, str]:
        with _keystore_lock:
            if not os.path.exists(self.keystore_path):
                return {}
            try:
                with open(self.keystore_path, "rb") as f:
                    cipher_bytes = f.read()
                if not cipher_bytes:
                    return {}
                decrypted_json = KeyStoreCipher.decrypt(cipher_bytes)
                if not decrypted_json:
                    return {}
                return json.loads(decrypted_json)
            except Exception as e:
                logger.error(f"[AIKeyStore] Error leyendo keystore: {e}")
                return {}

    def _write_raw(self, store: Dict[str, str]):
        with _keystore_lock:
            self._ensure_dir()
            payload = json.dumps(store, indent=2)
            cipher_bytes = KeyStoreCipher.encrypt(payload)
            with open(self.keystore_path, "wb") as f:
                f.write(cipher_bytes)

    def get_key(self, provider_id: str, config: Optional[dict] = None) -> Optional[str]:
        """Obtiene la clave del proveedor buscando en: 1) Config de mod-llm-chatter, 2) Keystore DPAPI, 3) Variables de Entorno."""
        provider = (provider_id or "").strip().lower()
        if not provider:
            return None

        # 1. Buscar en diccionario de configuración provisto (mod_llm_chatter.conf)
        if config and isinstance(config, dict):
            conf_keys = [
                f"LLMChatter.{provider.capitalize()}.ApiKey",
                f"LLMChatter.{provider.upper()}.ApiKey",
                f"LLMChatter.Cloud.{provider.capitalize()}.ApiKey",
            ]
            for ck in conf_keys:
                val = config.get(ck)
                if val and str(val).strip() and not str(val).strip().endswith("xxxxx"):
                    return str(val).strip()

        # 2. Buscar en almacén seguro cifrado en disco E:\
        store = self._read_raw()
        if provider in store and store[provider]:
            return store[provider]
        # Equivalencias comunes (ej: nvidia <-> nvidia_nim)
        if provider == "nvidia_nim" and "nvidia" in store and store["nvidia"]:
            return store["nvidia"]
        if provider == "nvidia" and "nvidia_nim" in store and store["nvidia_nim"]:
            return store["nvidia_nim"]

        # 3. Buscar en variables de entorno del sistema
        env_vars = PROVIDER_ENV_MAP.get(provider, [])
        for var in env_vars:
            val = os.environ.get(var)
            if val and val.strip():
                return val.strip()

        return None

    def set_key(self, provider_id: str, key_value: str) -> bool:
        """Guarda o actualiza una clave cifrada de forma segura."""
        provider = (provider_id or "").strip().lower()
        val = (key_value or "").strip()
        if not provider or not val:
            return False

        with _keystore_lock:
            store = self._read_raw()
            store[provider] = val
            self._write_raw(store)
            logger.info(f"[AIKeyStore] Clave para '{provider}' actualizada y cifrada exitosamente en E:\\.")
            return True

    def delete_key(self, provider_id: str) -> bool:
        provider = (provider_id or "").strip().lower()
        with _keystore_lock:
            store = self._read_raw()
            if provider in store:
                del store[provider]
                self._write_raw(store)
                return True
        return False

    def list_configured(self, config: Optional[dict] = None) -> List[str]:
        """Lista todos los proveedores que tienen una clave válida disponible."""
        result = set()
        store = self._read_raw()
        for p, k in store.items():
            if k and str(k).strip():
                result.add(p)

        for p in PROVIDER_ENV_MAP.keys():
            if self.get_key(p, config=config):
                result.add(p)

        return sorted(list(result))

    def import_external_keystore(self, external_file: str) -> int:
        """Migración soberana: importa claves desde otro archivo cifrado con DPAPI sin intervención manual."""
        if not os.path.exists(external_file):
            return 0
        try:
            with open(external_file, "rb") as f:
                cipher = f.read()
            import win32crypt
            _, plain_bytes = win32crypt.CryptUnprotectData(cipher, None, None, None, 0)
            data = json.loads(plain_bytes.decode("utf-8", errors="ignore"))
            imported = 0
            for k, v in data.items():
                if v and isinstance(v, str) and not v.endswith("xxxxx"):
                    self.set_key(k, v)
                    imported += 1
            logger.info(f"[AIKeyStore] Migración completada: {imported} claves importadas hacia E:\\.")
            return imported
        except Exception as e:
            logger.error(f"[AIKeyStore] Error al importar keystore externo: {e}")
            return 0


# Instancia singleton predeterminada para el módulo
default_keystore = AIKeyStore()
