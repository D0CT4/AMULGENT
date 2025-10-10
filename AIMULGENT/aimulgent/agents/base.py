"""
Base Agent Implementation
Provides the foundation for all AIMULGENT agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .hrm_reasoning import HRMReasoning

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all AIMULGENT agents.

    Follows KISS principle with minimal required interface.
    """

    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        self.agent_id = agent_id or self.__class__.__name__.lower()
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self.capabilities: List[str] = []
        self.running = False
        self.config = config or {}

        # Initialize HRM reasoning if enabled
        if self.config.get("hrm", {}).get("enable", False):
            self.hrm = HRMReasoning(self.config)
        else:
            self.hrm = None

    async def start(self) -> None:
        """Start the agent."""
        if self.running:
            return

        self.running = True
        self.logger.info(f"Agent {self.agent_id} started")

    async def stop(self) -> None:
        """Stop the agent."""
        if not self.running:
            return

        self.running = False
        self.logger.info(f"Agent {self.agent_id} stopped")

    @abstractmethod
    async def process_task(self, task_type: str, input_data: Dict[str, Any]) -> Any:
        """
        Process a task of the given type.

        Args:
            task_type: Type of task to process
            input_data: Input data for the task

        Returns:
            Task result
        """
        pass

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return self.capabilities.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = {
            "agent_id": self.agent_id,
            "running": self.running,
            "capabilities": self.capabilities,
        }

        # Add HRM status if available
        if self.hrm:
            status["hrm"] = self.hrm.get_status()

        return status

    def perform_reasoning(self, goal: str = None) -> Dict[str, Any]:
        """Integrate HRM reasoning into agent decision-making."""
        if not self.hrm:
            return {"error": "HRM reasoning not enabled"}

        if goal:
            self.hrm.add_goal(goal)

        result = self.hrm.reason_hierarchically()
        self.logger.info(f"HRM Reasoning Result for {self.agent_id}: {result}")
        return result
