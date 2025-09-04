"""
Tests for Advanced HRM Reasoning System
"""

import unittest
import torch
from unittest.mock import patch, MagicMock
from aimulgent.agents.hrm_reasoning import HRMReasoning, HierarchicalReasoningModel, AdaptiveComputationTime
from aimulgent.agents.base import BaseAgent


class TestHierarchicalReasoningModel(unittest.TestCase):
    """Test the neural HRM model components."""
    
    def setUp(self):
        self.config = {
            "hidden_size": 64,  # Smaller for testing
            "h_layers": 1,
            "l_layers": 2,
            "h_cycles": 1,
            "l_cycles": 2,
            "max_steps": 5,
            "learning_rate": 1e-3,
            "embedding_size": 32,
            "output_size": 32,
            "replay_buffer_size": 100,
            "max_execution_steps": 10,
            "exploration_prob": 0.1
        }
        self.model = HierarchicalReasoningModel(self.config)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, HierarchicalReasoningModel)
        self.assertEqual(self.model.hidden_size, 64)
        self.assertEqual(len(self.model.h_level), 1)
        self.assertEqual(len(self.model.l_level), 2)
    
    def test_forward_pass(self):
        """Test forward pass produces expected outputs."""
        batch_size = 2
        input_emb = torch.randn(batch_size, self.config["embedding_size"])
        
        output, q_values, should_halt = self.model(input_emb, steps=0)
        
        self.assertEqual(output.shape, (batch_size, self.config["output_size"]))
        self.assertEqual(q_values.shape, (batch_size, 2))  # halt vs continue
        self.assertIsInstance(should_halt, bool)
    
    def test_act_mechanism(self):
        """Test Adaptive Computation Time mechanism."""
        act = AdaptiveComputationTime(hidden_size=64, max_steps=5)
        hidden_state = torch.randn(1, 64)
        
        should_halt, q_logits = act(hidden_state, step=0, training=False)
        
        self.assertEqual(should_halt.shape, torch.Size([]))  # Scalar tensor
        self.assertEqual(q_logits.shape, torch.Size([2]))  # halt vs continue logits


class TestHRMReasoning(unittest.TestCase):
    """Test the main HRM reasoning system."""
    
    def setUp(self):
        self.config = {
            "hidden_size": 64,
            "h_layers": 1,
            "l_layers": 2,
            "h_cycles": 1,
            "l_cycles": 2,
            "max_steps": 5,
            "learning_rate": 1e-3,
            "embedding_size": 32,
            "output_size": 32,
            "replay_buffer_size": 100,
            "max_execution_steps": 10,
            "exploration_prob": 0.1
        }
        self.hrm = HRMReasoning(self.config)
    
    def test_initialization(self):
        """Test HRM initializes correctly."""
        self.assertIsInstance(self.hrm, HRMReasoning)
        self.assertIsInstance(self.hrm.model, HierarchicalReasoningModel)
        self.assertEqual(len(self.hrm.strategic_layer), 0)
        self.assertEqual(len(self.hrm.tactical_layer), 0)
        self.assertEqual(len(self.hrm.operational_layer), 0)
    
    def test_add_goal(self):
        """Test adding strategic goals."""
        goal = "Optimize system performance"
        self.hrm.add_goal(goal)
        
        self.assertEqual(len(self.hrm.strategic_layer), 1)
        self.assertEqual(self.hrm.strategic_layer[0], goal)
    
    @patch('aimulgent.agents.hrm_reasoning.torch.randn')
    def test_plan_tactical(self, mock_randn):
        """Test tactical planning generation."""
        # Mock random tensor for consistent testing
        mock_randn.return_value = torch.randn(1, 32)
        
        goal = "Test goal"
        plans = self.hrm.plan_tactical(goal)
        
        self.assertEqual(len(plans), 3)  # Should generate 3 plans
        self.assertEqual(len(self.hrm.tactical_layer), 3)
        
        for plan in plans:
            self.assertIn("plan_id", plan)
            self.assertEqual(plan["goal"], goal)
            self.assertIn("confidence", plan)
    
    @patch('aimulgent.agents.hrm_reasoning.torch.randn')
    def test_execute_operational(self, mock_randn):
        """Test operational action execution."""
        # Mock random tensor
        mock_randn.return_value = torch.randn(1, 32)
        
        plan = {
            "plan_id": "test_plan",
            "goal": "Test goal",
            "description": "Test plan",
            "embedding": torch.randn(1, 32),
            "priority": 1,
            "confidence": 0.8
        }
        
        actions = self.hrm.execute_operational(plan)
        
        self.assertGreater(len(actions), 0)
        self.assertEqual(len(self.hrm.operational_layer), len(actions))
        
        for action in actions:
            self.assertIn("action_id", action)
            self.assertEqual(action["plan_id"], plan["plan_id"])
            self.assertIn("q_values", action)
    
    @patch('aimulgent.agents.hrm_reasoning.torch.randn')
    def test_reason_hierarchically(self, mock_randn):
        """Test full hierarchical reasoning cycle."""
        # Mock random tensors
        mock_randn.return_value = torch.randn(1, 32)
        
        # Add a goal and run reasoning
        self.hrm.add_goal("Test optimization goal")
        result = self.hrm.reason_hierarchically()
        
        self.assertIn("goals", result)
        self.assertIn("plans", result)
        self.assertIn("actions", result)
        self.assertIn("model_stats", result)
        
        self.assertEqual(len(result["goals"]), 1)
        self.assertGreater(len(result["plans"]), 0)
        self.assertGreater(len(result["actions"]), 0)
    
    def test_get_status(self):
        """Test status reporting."""
        status = self.hrm.get_status()
        
        required_keys = [
            "goals_count", "plans_count", "actions_count",
            "replay_buffer_size", "model_stats", "active_goals", "pending_plans"
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
    
    def test_get_model_stats(self):
        """Test model statistics."""
        stats = self.hrm.get_model_stats()
        
        required_keys = [
            "total_parameters", "trainable_parameters", "device",
            "h_layers", "l_layers", "max_steps"
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertGreater(stats["total_parameters"], 0)
        self.assertGreater(stats["trainable_parameters"], 0)


class TestHRMIntegration(unittest.TestCase):
    """Test HRM integration with other components."""
    
    def setUp(self):
        self.config = {
            "hidden_size": 64,
            "h_layers": 1,
            "l_layers": 2,
            "h_cycles": 1,
            "l_cycles": 2,
            "max_steps": 5,
            "learning_rate": 1e-3,
            "embedding_size": 32,
            "output_size": 32,
            "replay_buffer_size": 100,
            "max_execution_steps": 10,
            "exploration_prob": 0.1
        }
    
    def test_experience_replay(self):
        """Test experience replay buffer functionality."""
        hrm = HRMReasoning(self.config)
        
        # Add some experiences
        for i in range(5):
            experience = {
                "state": torch.randn(1, 32),
                "action": torch.randn(1, 32),
                "q_values": torch.randn(1, 2),
                "reward": 1.0,
                "next_state": torch.randn(1, 32),
                "done": i == 4
            }
            hrm.replay_buffer.append(experience)
        
        self.assertEqual(len(hrm.replay_buffer), 5)
        
        # Test buffer size limit
        for i in range(100):
            experience = {
                "state": torch.randn(1, 32),
                "action": torch.randn(1, 32),
                "q_values": torch.randn(1, 2),
                "reward": 1.0,
                "next_state": torch.randn(1, 32),
                "done": False
            }
            hrm.replay_buffer.append(experience)
            if len(hrm.replay_buffer) > hrm.max_buffer_size:
                hrm.replay_buffer.pop(0)
        
        self.assertLessEqual(len(hrm.replay_buffer), hrm.max_buffer_size)


class TestBaseAgentWithHRM(unittest.TestCase):
    """Test cases for BaseAgent with HRM integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_with_hrm = {
            "hrm": {
                "enable": True,
                "max_goals": 5,
                "reasoning_depth": 3,
                "hidden_size": 64,
                "h_layers": 1,
                "l_layers": 2,
                "h_cycles": 1,
                "l_cycles": 2,
                "max_steps": 5,
                "learning_rate": 1e-3,
                "embedding_size": 32,
                "output_size": 32,
                "replay_buffer_size": 100,
                "max_execution_steps": 10,
                "exploration_prob": 0.1
            }
        }
        self.config_without_hrm = {
            "hrm": {
                "enable": False
            }
        }

    def test_agent_with_hrm_enabled(self):
        """Test agent initialization with HRM enabled."""
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def process_task(self, task_type: str, input_data):
                return {"result": "test"}

        agent = TestAgent("test_agent", self.config_with_hrm)
        self.assertIsNotNone(agent.hrm)
        self.assertIsInstance(agent.hrm, HRMReasoning)

    def test_agent_with_hrm_disabled(self):
        """Test agent initialization with HRM disabled."""
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def process_task(self, task_type: str, input_data):
                return {"result": "test"}

        agent = TestAgent("test_agent", self.config_without_hrm)
        self.assertIsNone(agent.hrm)

    @patch('aimulgent.agents.hrm_reasoning.torch.randn')
    def test_perform_reasoning_with_hrm(self, mock_randn):
        """Test performing reasoning with HRM enabled."""
        mock_randn.return_value = torch.randn(1, 32)
        
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def process_task(self, task_type: str, input_data):
                return {"result": "test"}

        agent = TestAgent("test_agent", self.config_with_hrm)
        result = agent.perform_reasoning("Test Goal")
        self.assertIn("goals", result)
        self.assertIn("plans", result)
        self.assertIn("actions", result)

    def test_perform_reasoning_without_hrm(self):
        """Test performing reasoning with HRM disabled."""
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def process_task(self, task_type: str, input_data):
                return {"result": "test"}

        agent = TestAgent("test_agent", self.config_without_hrm)
        result = agent.perform_reasoning("Test Goal")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "HRM reasoning not enabled")

    def test_get_status_with_hrm(self):
        """Test getting agent status with HRM."""
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def process_task(self, task_type: str, input_data):
                return {"result": "test"}

        agent = TestAgent("test_agent", self.config_with_hrm)
        status = agent.get_status()
        self.assertIn("hrm", status)
        self.assertIn("goals_count", status["hrm"])


if __name__ == '__main__':
    unittest.main()
