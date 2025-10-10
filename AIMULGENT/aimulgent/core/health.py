"""Health check endpoints and system status monitoring for AIMULGENT.

Provides:
- System health status
- Component readiness checks
- Dependency verification
- Metrics exposure endpoint
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Health endpoints disabled.")

from .metrics import MetricsCollector


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Health check system for AIMULGENT."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize health check system.

        Args:
            metrics_collector: Optional metrics collector for exposing metrics
        """
        self.metrics_collector = metrics_collector
        self.start_time = time.time()
        self.component_checks: Dict[str, callable] = {}

    def register_check(self, component: str, check_func: callable):
        """Register a health check for a component.

        Args:
            component: Component name
            check_func: Async function that returns (status, message)
        """
        self.component_checks[component] = check_func
        logger.info(f"Registered health check for component: {component}")

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Dictionary with health status and component details
        """
        uptime = time.time() - self.start_time

        # Run all component checks
        component_results = {}
        overall_status = HealthStatus.HEALTHY

        for component, check_func in self.component_checks.items():
            try:
                status, message = await check_func()
                component_results[component] = {
                    "status": (
                        status.value if isinstance(status, HealthStatus) else status
                    ),
                    "message": message,
                }

                # Update overall status based on component status
                if status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    status == HealthStatus.DEGRADED
                    and overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                component_results[component] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Check failed: {str(e)}",
                }
                overall_status = HealthStatus.UNHEALTHY

        return {
            "status": overall_status.value,
            "uptime_seconds": uptime,
            "timestamp": time.time(),
            "components": component_results,
        }

    async def check_readiness(self) -> Dict[str, Any]:
        """Check if system is ready to serve requests.

        Returns:
            Dictionary with readiness status
        """
        health_status = await self.check_health()

        # System is ready if not unhealthy
        is_ready = health_status["status"] != HealthStatus.UNHEALTHY.value

        return {
            "ready": is_ready,
            "status": health_status["status"],
            "timestamp": time.time(),
        }

    async def check_liveness(self) -> Dict[str, Any]:
        """Check if system is alive (basic heartbeat).

        Returns:
            Dictionary with liveness status
        """
        return {
            "alive": True,
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": time.time(),
        }


def create_health_app(
    health_check: HealthCheck, metrics_collector: Optional[MetricsCollector] = None
) -> Optional[FastAPI]:
    """Create FastAPI application with health endpoints.

    Args:
        health_check: HealthCheck instance
        metrics_collector: Optional metrics collector for /metrics endpoint

    Returns:
        FastAPI application or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available. Cannot create health app.")
        return None

    app = FastAPI(
        title="AIMULGENT Health & Metrics",
        description="Health check and metrics endpoints for AIMULGENT system",
        version="1.0.0",
    )

    @app.get("/health", response_model=Dict[str, Any])
    async def health_endpoint():
        """Comprehensive health check endpoint."""
        return await health_check.check_health()

    @app.get("/health/live", response_model=Dict[str, Any])
    async def liveness_endpoint():
        """Liveness probe endpoint for Kubernetes."""
        return await health_check.check_liveness()

    @app.get("/health/ready", response_model=Dict[str, Any])
    async def readiness_endpoint():
        """Readiness probe endpoint for Kubernetes."""
        return await health_check.check_readiness()

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        if metrics_collector is None:
            return PlainTextResponse("Metrics collection not enabled", status_code=503)

        try:
            metrics_data = metrics_collector.get_metrics()
            # Update system metrics before returning
            metrics_collector.update_system_metrics()
            return Response(
                content=metrics_data, media_type="text/plain; version=0.0.4"
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return PlainTextResponse(
                f"Error generating metrics: {str(e)}", status_code=500
            )

    @app.get("/status")
    async def status_endpoint():
        """General system status endpoint."""
        health = await health_check.check_health()

        status_info = {
            "service": "AIMULGENT",
            "version": "1.0.0",
            "status": health["status"],
            "uptime_seconds": health["uptime_seconds"],
            "timestamp": health["timestamp"],
            "components_count": len(health.get("components", {})),
        }

        return JSONResponse(content=status_info)

    return app


# Example component check functions
async def example_database_check() -> tuple[HealthStatus, str]:
    """Example database health check."""
    try:
        # Add actual database check logic here
        await asyncio.sleep(0.01)  # Simulate check
        return HealthStatus.HEALTHY, "Database connection OK"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Database error: {str(e)}"


async def example_agent_check() -> tuple[HealthStatus, str]:
    """Example agent system health check."""
    try:
        # Add actual agent system check logic here
        await asyncio.sleep(0.01)  # Simulate check
        return HealthStatus.HEALTHY, "All agents operational"
    except Exception as e:
        return HealthStatus.DEGRADED, f"Some agents unavailable: {str(e)}"
