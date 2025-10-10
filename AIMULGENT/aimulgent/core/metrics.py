"""Prometheus metrics and observability module for AIMULGENT.

Provides comprehensive monitoring capabilities including:
- System metrics (CPU, memory, uptime)
- Workflow metrics (task counts, durations, success rates)
- Token usage tracking
- Energy consumption estimates
- Custom business metrics
"""

import time
import psutil
from typing import Dict, Optional, Any
from functools import wraps
import logging

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus-client not installed. Metrics collection disabled.")


logger = logging.getLogger(__name__)


class MetricsCollector:
    """Central metrics collection and management."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.

        Args:
            registry: Custom Prometheus registry (uses default if None)
        """
        self.registry = registry
        self._initialized = False

        if PROMETHEUS_AVAILABLE:
            self._setup_metrics()
            self._initialized = True
        else:
            logger.warning(
                "Metrics collection disabled - prometheus-client not available"
            )

    def _setup_metrics(self):
        """Setup all Prometheus metrics."""
        # System metrics
        self.system_cpu_usage = Gauge(
            "aimulgent_system_cpu_percent",
            "Current CPU usage percentage",
            registry=self.registry,
        )
        self.system_memory_usage = Gauge(
            "aimulgent_system_memory_bytes",
            "Current memory usage in bytes",
            registry=self.registry,
        )
        self.system_uptime = Gauge(
            "aimulgent_system_uptime_seconds",
            "System uptime in seconds",
            registry=self.registry,
        )

        # Workflow metrics
        self.tasks_total = Counter(
            "aimulgent_tasks_total",
            "Total number of tasks processed",
            ["status", "task_type"],
            registry=self.registry,
        )
        self.task_duration = Histogram(
            "aimulgent_task_duration_seconds",
            "Task execution duration in seconds",
            ["task_type"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )
        self.active_tasks = Gauge(
            "aimulgent_active_tasks",
            "Number of currently active tasks",
            ["task_type"],
            registry=self.registry,
        )

        # Token usage metrics
        self.tokens_processed = Counter(
            "aimulgent_tokens_processed_total",
            "Total number of tokens processed",
            ["model", "operation"],
            registry=self.registry,
        )
        self.token_rate = Gauge(
            "aimulgent_token_rate_per_second",
            "Current token processing rate",
            ["model"],
            registry=self.registry,
        )

        # Energy metrics
        self.energy_consumption = Counter(
            "aimulgent_energy_consumption_wh",
            "Estimated energy consumption in watt-hours",
            ["component"],
            registry=self.registry,
        )

        # Agent metrics
        self.agents_active = Gauge(
            "aimulgent_agents_active",
            "Number of active agents",
            ["agent_type"],
            registry=self.registry,
        )
        self.agent_requests = Counter(
            "aimulgent_agent_requests_total",
            "Total agent requests",
            ["agent_type", "status"],
            registry=self.registry,
        )

        # Code analysis metrics
        self.code_lines_analyzed = Counter(
            "aimulgent_code_lines_analyzed_total",
            "Total lines of code analyzed",
            ["language"],
            registry=self.registry,
        )
        self.issues_found = Counter(
            "aimulgent_issues_found_total",
            "Total issues found in code",
            ["severity", "category"],
            registry=self.registry,
        )

        # System info
        self.system_info = Info(
            "aimulgent_system",
            "System information",
            registry=self.registry,
        )

        self._start_time = time.time()

    def update_system_metrics(self):
        """Update system-level metrics."""
        if not self._initialized:
            return

        try:
            self.system_cpu_usage.set(psutil.cpu_percent(interval=0.1))
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            self.system_uptime.set(time.time() - self._start_time)
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def record_task(self, task_type: str, status: str, duration: float):
        """Record a completed task.

        Args:
            task_type: Type of task (e.g., 'analysis', 'coordination')
            status: Task status ('success', 'failure', 'timeout')
            duration: Task duration in seconds
        """
        if not self._initialized:
            return

        self.tasks_total.labels(status=status, task_type=task_type).inc()
        self.task_duration.labels(task_type=task_type).observe(duration)

    def record_tokens(self, model: str, operation: str, count: int):
        """Record token usage.

        Args:
            model: Model name
            operation: Operation type ('input', 'output')
            count: Number of tokens
        """
        if not self._initialized:
            return

        self.tokens_processed.labels(model=model, operation=operation).inc(count)

    def record_energy(self, component: str, watt_hours: float):
        """Record energy consumption.

        Args:
            component: Component name ('cpu', 'gpu', 'memory')
            watt_hours: Energy consumed in watt-hours
        """
        if not self._initialized:
            return

        self.energy_consumption.labels(component=component).inc(watt_hours)

    def record_code_analysis(
        self, language: str, lines: int, issues: Dict[str, Dict[str, int]]
    ):
        """Record code analysis results.

        Args:
            language: Programming language
            lines: Number of lines analyzed
            issues: Dictionary of {severity: {category: count}}
        """
        if not self._initialized:
            return

        self.code_lines_analyzed.labels(language=language).inc(lines)

        for severity, categories in issues.items():
            for category, count in categories.items():
                self.issues_found.labels(severity=severity, category=category).inc(
                    count
                )

    def set_system_info(self, info: Dict[str, str]):
        """Set system information.

        Args:
            info: Dictionary of system information
        """
        if not self._initialized:
            return

        self.system_info.info(info)

    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format.

        Returns:
            Metrics data in Prometheus text format
        """
        if not self._initialized:
            return b""

        return generate_latest(self.registry)


def track_execution_time(metrics_collector: MetricsCollector, task_type: str):
    """Decorator to track function execution time.

    Args:
        metrics_collector: MetricsCollector instance
        task_type: Type of task being tracked
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                metrics_collector.active_tasks.labels(task_type=task_type).inc()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_task(task_type, "success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_task(task_type, "failure", duration)
                raise
            finally:
                metrics_collector.active_tasks.labels(task_type=task_type).dec()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                metrics_collector.active_tasks.labels(task_type=task_type).inc()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_task(task_type, "success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_task(task_type, "failure", duration)
                raise
            finally:
                metrics_collector.active_tasks.labels(task_type=task_type).dec()

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
