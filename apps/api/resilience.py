"""Resiliencia frente a Model Serving: timeout por llamada, circuit breaker y caché de respuestas."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any


class CircuitOpenError(RuntimeError):
    """El circuito está abierto: se responde de inmediato sin llamar al servicio."""


class CircuitBreaker:
    """Tras `failure_threshold` fallos seguidos deja de llamar al servicio durante `reset_seconds`.

    Estados: cerrado (normal), abierto (falla rápido) y semiabierto (deja pasar una
    llamada de prueba; si funciona se cierra, si falla se vuelve a abrir).
    """

    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 30.0, clock=time.monotonic) -> None:
        self.failure_threshold, self.reset_seconds, self._clock = failure_threshold, reset_seconds, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "cerrado"
            return "semiabierto" if self._clock() - self._opened_at >= self.reset_seconds else "abierto"

    def call(self, fn: Callable[[], Any]) -> Any:
        if self.state == "abierto":
            raise CircuitOpenError("Model Serving marcado como no disponible (circuit breaker abierto)")
        try:
            result = fn()
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._opened_at = self._clock()
            raise
        with self._lock:
            self._failures, self._opened_at = 0, None
        return result


class TimeoutCaller:
    """Ejecuta una llamada con tiempo máximo (el cliente HTTP del SDK no expone un timeout por llamada)."""

    def __init__(self, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="serving")

    def __call__(self, fn: Callable[[], Any], timeout_seconds: float) -> Any:
        future = self._pool.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(f"Model Serving no respondió en {timeout_seconds:.1f} s") from exc


class TTLCache:
    """Caché LRU con expiración: la misma solicitud devuelve la misma respuesta (idempotencia)."""

    def __init__(self, max_items: int = 5000, ttl_seconds: float = 900.0, clock=time.monotonic) -> None:
        self.max_items, self.ttl_seconds, self._clock = max_items, ttl_seconds, clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if self._clock() - stored_at > self.ttl_seconds:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
