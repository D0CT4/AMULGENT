"""
Tests for Advanced HRM Reasoning System
"""

import unittest
import torch
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
    
    def test_plan_tactical(self):
        """Test tactical planning generation."""
        goal = "Test goal"
        plans = self.hrm.plan_tactical(goal)
        
        self.assertEqual(len(plans), 3)  # Should generate 3 plans
        self.assertEqual(len(self.hrm.tactical_layer), 3)
        
        for plan in plans:
            self.assertIn("plan_id", plan)
            self.assertEqual(plan["goal"], goal)
            self.assertIn("confidence", plan)
    
    def test_execute_operational(self):
        """Test operational action execution."""
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
    
    def test_reason_hierarchically(self):
        """Test full hierarchical reasoning cycle."""
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

    def test_perform_reasoning_with_hrm(self):
        """Test performing reasoning with HRM enabled."""
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


class TestEnhancedHRMFeatures(unittest.TestCase):
    """Test enhanced HRM features from external research integration."""
    
    def setUp(self):
        self.base_config = {
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
    
    def test_rmsnorm_feature(self):
        """Test RMSNorm enhancement."""
        config = self.base_config.copy()
        config["use_rmsnorm"] = True
        
        hrm = HRMReasoning(config)
        
        # Check that model uses RMSNorm
        self.assertTrue(hrm.model.use_rmsnorm)
        
        # Test forward pass works
        result = hrm.reason_hierarchically()
        self.assertIn("model_stats", result)
    
    def test_swiglu_feature(self):
        """Test SwiGLU activation enhancement."""
        config = self.base_config.copy()
        config["use_swiglu"] = True
        
        hrm = HRMReasoning(config)
        
        # Check that model uses SwiGLU
        self.assertTrue(hrm.model.use_swiglu)
        
        # Test forward pass works
        result = hrm.reason_hierarchically()
        self.assertIn("model_stats", result)
    
    def test_iterative_refinement(self):
        """Test iterative refinement feature."""
        config = self.base_config.copy()
        config["use_iterative_refinement"] = True
        config["refinement_segments"] = 3
        
        hrm = HRMReasoning(config)
        
        # Check that model uses iterative refinement
        self.assertTrue(hrm.model.use_iterative_refinement)
        self.assertEqual(hrm.model.refinement_segments, 3)
        
        # Test reasoning with refinement
        hrm.add_goal("Test iterative refinement")
        result = hrm.reason_hierarchically()
        
        self.assertIn("model_stats", result)
        self.assertGreater(len(result["actions"]), 0)
    
    def test_enhanced_act_mechanism(self):
        """Test enhanced ACT with separate Q-values."""
        config = self.base_config.copy()
        hrm = HRMReasoning(config)
        
        # Test that ACT returns separate halt/continue Q-values
        hidden_state = torch.randn(1, 64)
        should_halt, q_values = hrm.model.act(hidden_state, step=0, training=False)
        
        # Check output shapes and types
        self.assertEqual(q_values.shape, torch.Size([2]))  # [halt, continue]
        self.assertIsInstance(should_halt.item(), float)
    
    def test_enhanced_status_reporting(self):
        """Test enhanced status reporting with new metrics."""
        config = self.base_config.copy()
        config["use_rmsnorm"] = True
        config["use_swiglu"] = True
        config["use_iterative_refinement"] = True
        
        hrm = HRMReasoning(config)
        
        # Add some activity
        hrm.add_goal("Test enhanced status")
        
        status = hrm.get_status()
        
        # Check enhanced fields are present
        self.assertIn("architecture", status)
        self.assertTrue(status["architecture"]["use_rmsnorm"])
        self.assertTrue(status["architecture"]["use_swiglu"])
        self.assertTrue(status["architecture"]["use_iterative_refinement"])
    
    def test_enhanced_learning(self):
        """Test enhanced learning with separate Q-values."""
        config = self.base_config.copy()
        hrm = HRMReasoning(config)
        
        # Add some experiences to trigger learning
        for i in range(35):  # Enough to trigger learning
            experience = {
                "state": torch.randn(1, 32),
                "action": torch.randn(1, 32),
                "q_values": torch.randn(2),  # Now expects [halt, continue]
                "reward": 1.0 if i % 5 == 0 else 0.1,
                "next_state": torch.randn(1, 32),
                "done": i % 10 == 9
            }
            hrm.replay_buffer.append(experience)
        
        # Trigger learning
        initial_params = list(hrm.model.parameters())[0].clone()
        hrm._learn_from_experience()
        
        # Check that parameters have changed (learning occurred)
        final_params = list(hrm.model.parameters())[0]
        self.assertFalse(torch.equal(initial_params, final_params))
    
    def test_combined_enhancements(self):
        """Test all enhancements working together."""
        config = self.base_config.copy()
        config.update({
            "use_rmsnorm": True,
            "use_swiglu": True,
            "use_iterative_refinement": True,
            "refinement_segments": 2
        })
        
        hrm = HRMReasoning(config)
        
        # Test complete reasoning cycle with all enhancements
        hrm.add_goal("Test all enhancements together")
        result = hrm.reason_hierarchically()
        
        # Verify all features are active and working
        status = hrm.get_status()
        model_stats = hrm.get_model_stats()
        
        self.assertIn("enhanced_features", model_stats)
        self.assertTrue(model_stats["enhanced_features"]["rmsnorm"])
        self.assertTrue(model_stats["enhanced_features"]["swiglu"])
        self.assertTrue(model_stats["enhanced_features"]["iterative_refinement"])
        
        # Verify reasoning worked
        self.assertGreater(len(result["plans"]), 0)
        self.assertGreater(len(result["actions"]), 0)


if __name__ == '__main__':
    unittest.main()
