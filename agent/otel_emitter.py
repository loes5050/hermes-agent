"""OpenTelemetry GenAI emitter stub (opt-in, default OFF).

Phase 1 of the ``otel_then_langfuse`` observability plan.  This module provides
a *minimal* OpenTelemetry emitter that, when enabled via
``observability.otel.enabled: true`` in ``config.yaml``, emits GenAI-style
spans for tool start/complete events.

Design constraints (see AGENTS.md):
- **No hard dependency.**  If the ``opentelemetry`` packages are not installed
  the module imports cleanly and every method is a no-op.  Nothing breaks.
- **Default OFF.**  The config flag defaults to ``false`` and the default
  exporter is ``none`` (no network traffic) unless the user explicitly
  configures a console or OTLP exporter.
- **Pure library + thin hook.**  The emitter is a singleton; callers use
  :func:`get_emitter` and call :meth:`OtelEmitter.on_tool_start` /
  :meth:`OtelEmitter.on_tool_complete`.  No callbacks, no plugin surface —
  just two call sites in ``tool_executor.py``.
- **Never sends network traffic without explicit user config.**

Usage in ``tool_executor.py``::

    from agent.otel_emitter import get_emitter
    _otel = get_emitter()
    _otel.on_tool_start(function_name, function_args, tool_call_id)
    # ... execute tool ...
    _otel.on_tool_complete(function_name, duration, is_error, tool_call_id)

Config keys (in ``config.yaml``)::

    observability:
      otel:
        enabled: false          # master switch — must be true to emit
        service_name: "hermes-agent"
        exporter: none          # none | console | otlp
        otlp_endpoint: ""       # OTLP gRPC/HTTP endpoint when exporter=otlp
        sample_rate: 1.0         # 0.0–1.0, future use

This is intentionally a *stub*: spans carry tool name, duration, and error
status only.  No prompt/response bodies are recorded in Phase 1.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports — all guarded.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised when opentelemetry absent
    _OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]

try:
    from opentelemetry.exporters.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    _OTLP_AVAILABLE = True
except ImportError:
    _OTLP_AVAILABLE = False


class _NoopEmitter:
    """Fallback used when OTel is unavailable or disabled.

    Every method is a no-op so callers don't need to branch on availability.
    """

    __slots__ = ()

    def on_tool_start(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_call_id: str = "",
    ) -> None:
        pass

    def on_tool_complete(
        self,
        tool_name: str,
        duration: float = 0.0,
        is_error: bool = False,
        tool_call_id: str = "",
    ) -> None:
        pass

    def shutdown(self) -> None:
        pass


class OtelEmitter:
    """Active OTel emitter backed by a real ``TracerProvider``.

    Created only when ``observability.otel.enabled`` is true AND the
    ``opentelemetry`` package is importable.  Emits GenAI-style spans for
    tool lifecycle events.
    """

    __slots__ = ("_tracer", "_processor", "_sample_rate", "_provider")

    def __init__(
        self,
        service_name: str,
        exporter: str,
        otlp_endpoint: str,
        sample_rate: float,
    ) -> None:
        self._sample_rate = sample_rate
        self._provider: Optional[TracerProvider] = None
        self._processor = None

        if not _OTEL_AVAILABLE:
            logger.debug("OTel emitter requested but opentelemetry not installed")
            return

        # Build a *dedicated* provider so we never mutate the global one.
        self._provider = TracerProvider()
        exporter_lower = (exporter or "none").lower()

        if exporter_lower == "console":
            self._processor = SimpleSpanProcessor(ConsoleSpanExporter())
        elif exporter_lower == "otlp":
            if not _OTLP_AVAILABLE:
                logger.warning(
                    "OTel exporter=otlp but opentelemetry-exporter-otlp "
                    "is not installed; falling back to no-op"
                )
                self._provider = None
                return
            if not otlp_endpoint:
                logger.warning(
                    "OTel exporter=otlp but no otlp_endpoint configured; "
                    "falling back to no-op"
                )
                self._provider = None
                return
            self._processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=otlp_endpoint)
            )
        else:
            # "none" — provider exists but no exporter attached, so spans
            # are created but never exported.  Safe default.
            self._processor = None

        if self._processor is not None:
            self._provider.add_span_processor(self._processor)

        self._tracer = self._provider.get_tracer(service_name or "hermes-agent")
        logger.debug(
            "OTel emitter initialized: service=%s exporter=%s",
            service_name, exporter_lower,
        )

    # -- sampling ----------------------------------------------------------

    def _should_sample(self) -> bool:
        """Deterministic sampler for future partial-trace support."""
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        return random.random() < self._sample_rate

    # -- public API --------------------------------------------------------

    def on_tool_start(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_call_id: str = "",
    ) -> None:
        if not _OTEL_AVAILABLE or self._provider is None:
            return
        if not self._should_sample():
            return
        try:
            span = self._tracer.start_span(
                f"tool.{tool_name}",
                attributes={
                    "gen_ai.system": "hermes-agent",
                    "gen_ai.tool.name": tool_name,
                    "hermes.tool_call_id": tool_call_id,
                },
            )
            # Stash the span on a thread-local-ish dict keyed by call id;
            # we use a simple module-level dict since tool calls are
            # short-lived and tool_call_id is unique per turn.
            _active_spans[tool_call_id or tool_name] = span
        except Exception:
            logger.debug("OTel on_tool_start failed", exc_info=True)

    def on_tool_complete(
        self,
        tool_name: str,
        duration: float = 0.0,
        is_error: bool = False,
        tool_call_id: str = "",
    ) -> None:
        if not _OTEL_AVAILABLE or self._provider is None:
            return
        key = tool_call_id or tool_name
        span = _active_spans.pop(key, None)
        if span is None:
            return
        try:
            span.set_attribute("hermes.tool.duration_ms", duration * 1000.0)
            if is_error:
                span.set_attribute("gen_ai.tool.error", True)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
            else:
                span.set_status(trace.Status(trace.StatusCode.OK))
            span.end()
        except Exception:
            logger.debug("OTel on_tool_complete failed", exc_info=True)

    def shutdown(self) -> None:
        """Flush pending spans and shut down the provider."""
        if self._processor is not None:
            try:
                self._processor.shutdown()
            except Exception:
                logger.debug("OTel processor shutdown failed", exc_info=True)
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception:
                logger.debug("OTel provider shutdown failed", exc_info=True)


# Module-level dict of in-flight spans keyed by tool_call_id.
_active_spans: Dict[str, Any] = {}

# Singleton — lazily initialized on first access via config.
_emitter: Optional[Any] = None
_initialized = False


def get_emitter() -> Any:
    """Return the process-wide emitter (singleton).

    On first call, reads ``observability.otel`` from the Hermes config.  If
    disabled or the package is missing, a :class:`_NoopEmitter` is returned
    so all callers can invoke ``.on_tool_start()`` / ``.on_tool_complete()``
    unconditionally without checking availability.
    """
    global _emitter, _initialized
    if _initialized:
        return _emitter

    _initialized = True
    try:
        from hermes_cli.config import load_config as _load_cfg
        cfg = _load_cfg() or {}
        otel_cfg = (cfg.get("observability") or {}).get("otel") or {}
        enabled = bool(otel_cfg.get("enabled", False))
        if not enabled:
            _emitter = _NoopEmitter()
            logger.debug("OTel emitter disabled by config")
            return _emitter
        if not _OTEL_AVAILABLE:
            logger.info(
                "OTel emitter enabled in config but opentelemetry package "
                "not installed — install with: pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )
            _emitter = _NoopEmitter()
            return _emitter
        _emitter = OtelEmitter(
            service_name=otel_cfg.get("service_name", "hermes-agent"),
            exporter=otel_cfg.get("exporter", "none"),
            otlp_endpoint=otel_cfg.get("otlp_endpoint", ""),
            sample_rate=float(otel_cfg.get("sample_rate", 1.0)),
        )
    except Exception:
        logger.debug("OTel emitter init failed; using no-op", exc_info=True)
        _emitter = _NoopEmitter()
    return _emitter


def reset_for_tests() -> None:
    """Reset the singleton — for test isolation only."""
    global _emitter, _initialized, _active_spans
    if _emitter is not None and hasattr(_emitter, "shutdown"):
        _emitter.shutdown()
    _emitter = None
    _initialized = False
    _active_spans.clear()