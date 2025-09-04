"""
AIMULGENT Coordination System
Simplified coordinator following KISS principles.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(Enum):
    """Agent status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class Task:
    """Task definition for agent execution."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    agent_id: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class AgentInfo:
    """Information about registered agents."""
    
    agent_id: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    tasks_completed: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)


class Coordinator:
    """Simplified multi-agent coordinator."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, Any] = {}  # Store actual agent instances
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the coordination system."""
        if self.running:
            return
        
        self.running = True
        
        # Start background task processor
        task_processor = asyncio.create_task(self._process_tasks())
        self._background_tasks.append(task_processor)
        
        logger.info("Coordinator started")
    
    async def stop(self) -> None:
        """Stop the coordination system."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Coordinator stopped")
    
    async def register_agent(self, agent_id: str, capabilities: List[str], agent_instance: Any = None) -> None:
        """Register an agent with the coordinator."""
        agent_info = AgentInfo(agent_id=agent_id, capabilities=capabilities)
        self.agents[agent_id] = agent_info
        
        if agent_instance:
            self.agent_instances[agent_id] = agent_instance
        
        logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")
    
    async def submit_task(
        self, 
        task_type: str, 
        input_data: Dict[str, Any]
    ) -> str:
        """Submit a task for execution."""
        
        # Find suitable agent
        suitable_agents = [
            agent_id for agent_id, info in self.agents.items()
            if task_type in info.capabilities and info.status == AgentStatus.IDLE
        ]
        
        if not suitable_agents:
            raise ValueError(f"No available agents for task type: {task_type}")
        
        # Select first available agent (simple round-robin could be added)
        selected_agent = suitable_agents[0]
        
        # Create task
        task = Task(
            task_type=task_type,
            agent_id=selected_agent,
            input_data=input_data
        )
        
        self.tasks[task.task_id] = task
        await self.task_queue.put(task)
        
        logger.info(f"Task {task.task_id} submitted to {selected_agent}")
        return task.task_id
    
    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get task result, waiting for completion if necessary."""
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Wait for completion
        start_time = asyncio.get_event_loop().time()
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.FAILED:
            raise Exception(f"Task failed: {task.error}")
        
        return task.result
    
    async def _process_tasks(self) -> None:
        """Background task processor."""
        
        while self.running:
            try:
                # Get next task with timeout to allow periodic checks
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Update agent status
                agent_info = self.agents.get(task.agent_id)
                if agent_info:
                    agent_info.status = AgentStatus.BUSY
                    agent_info.current_task = task.task_id
                
                # Execute task
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.TimeoutError:
                # Normal timeout, continue processing
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Get the agent to execute the task
            agent_info = self.agents.get(task.agent_id)
            if not agent_info:
                raise ValueError(f"Agent {task.agent_id} not found")
            
            # Execute task using actual agent if available
            agent_instance = self.agent_instances.get(task.agent_id)
            if agent_instance and hasattr(agent_instance, 'process_task'):
                result = await agent_instance.process_task(task.task_type, task.input_data)
            else:
                # Fallback simulation for agents without instances
                await asyncio.sleep(0.1)
                result = {"status": "completed", "message": f"Task {task.task_type} executed"}
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Update agent status
            agent_info = self.agents.get(task.agent_id)
            if agent_info:
                agent_info.status = AgentStatus.IDLE
                agent_info.current_task = None
                agent_info.tasks_completed += 1
                agent_info.last_heartbeat = datetime.now()
    
    async def coordinate_agents(self, agents: List[Any]) -> Dict[str, Any]:
        """Coordinate agents with advanced HRM reasoning."""
        coordination_results = {
            "total_agents": len(agents),
            "hrm_enabled_agents": 0,
            "reasoning_cycles": 0,
            "total_actions": 0,
            "neural_updates": 0
        }
        
        for agent in agents:
            if hasattr(agent, 'hrm') and agent.hrm is not None:
                coordination_results["hrm_enabled_agents"] += 1
                
                # Get HRM status
                hrm_status = agent.hrm.get_status()
                
                # Trigger reasoning if goals exist
                if hrm_status["goals_count"] > 0:
                    reasoning_output = agent.perform_reasoning()
                    coordination_results["reasoning_cycles"] += 1
                    coordination_results["total_actions"] += len(reasoning_output.get("actions", []))
                    
                    # Check for neural learning
                    if "model_stats" in reasoning_output:
                        coordination_results["neural_updates"] += 1
                    
                    self.logger.info(f"Coordinated HRM for agent {agent.agent_id}: {len(reasoning_output.get('actions', []))} actions generated")
                else:
                    self.logger.debug(f"Agent {agent.agent_id} has no active goals")
        
        self.logger.info(f"Coordination complete: {coordination_results}")
        return coordination_results