#!/usr/bin/env python3
"""
Demonstration of Advanced HRM Reasoning System in AMULGENT
Shows the hierarchical reasoning capabilities with neural networks and ACT.
"""

import asyncio
import json
import logging
from aimulgent.agents.hrm_reasoning import HRMReasoning
from aimulgent.agents.base import BaseAgent
from aimulgent.core.coordinator import Coordinator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_hrm_reasoning():
    """Demonstrate the advanced HRM reasoning system."""
    print("🚀 AMULGENT Advanced HRM Reasoning Demonstration")
    print("=" * 60)

    # Configuration for HRM
    hrm_config = {
        "hidden_size": 128,
        "h_layers": 2,
        "l_layers": 4,
        "h_cycles": 2,
        "l_cycles": 4,
        "max_steps": 10,
        "learning_rate": 0.0001,
        "embedding_size": 64,
        "output_size": 64,
        "replay_buffer_size": 1000,
        "max_execution_steps": 20,
        "exploration_prob": 0.1
    }

    # Create HRM reasoning system
    print("\n1. Initializing Advanced HRM Reasoning System...")
    hrm = HRMReasoning(hrm_config)
    print(f"✓ HRM initialized with {hrm.get_model_stats()['total_parameters']} parameters")

    # Create agent with HRM
    print("\n2. Creating Agent with HRM Integration...")
    agent_config = {"hrm": hrm_config}
    agent = BaseAgent("demo_agent", agent_config)
    print("✓ Agent created with HRM reasoning capabilities")

    # Add strategic goals
    print("\n3. Adding Strategic Goals...")
    goals = [
        "Optimize system performance and resource utilization",
        "Enhance decision-making accuracy through hierarchical reasoning",
        "Improve multi-agent coordination and task distribution",
        "Implement adaptive learning mechanisms for continuous improvement"
    ]

    for goal in goals:
        hrm.add_goal(goal)
        print(f"✓ Added goal: {goal}")

    # Perform hierarchical reasoning
    print("\n4. Executing Hierarchical Reasoning...")
    print("   Processing goals through H-level (strategic) and L-level (operational) modules...")

    reasoning_result = hrm.reason_hierarchically()

    print("✓ Reasoning completed!")
    print(f"   - Goals processed: {len(reasoning_result['goals'])}")
    print(f"   - Plans generated: {len(reasoning_result['plans'])}")
    print(f"   - Actions executed: {len(reasoning_result['actions'])}")

    # Display detailed results
    print("\n5. Reasoning Results:")
    print("-" * 40)

    print("\n📋 Strategic Goals:")
    for i, goal in enumerate(reasoning_result['goals'], 1):
        print(f"   {i}. {goal}")

    print("\n🎯 Tactical Plans:")
    for i, plan in enumerate(reasoning_result['plans'][:5], 1):  # Show first 5
        confidence = plan.get('confidence', 0) * 100
        print(".1f")

    print("\n⚡ Operational Actions:")
    for i, action in enumerate(reasoning_result['actions'][:10], 1):  # Show first 10
        print(f"   {i}. {action['description']} (Step {action['step']})")

    # Show model statistics
    print("\n6. Model Statistics:")
    print("-" * 40)
    stats = reasoning_result['model_stats']
    print(f"   Total Parameters: {stats['total_parameters']:,}")
    print(f"   Trainable Parameters: {stats['trainable_parameters']:,}")
    print(f"   H-level Layers: {stats['h_layers']}")
    print(f"   L-level Layers: {stats['l_layers']}")
    print(f"   Max Reasoning Steps: {stats['max_steps']}")
    print(f"   Device: {stats['device']}")

    # Demonstrate coordinator integration
    print("\n7. Multi-Agent Coordination Demo:")
    print("-" * 40)

    coordinator = Coordinator()
    await coordinator.register_agent("demo_agent", ["analysis", "reasoning"], agent)

    # Add goals to multiple agents
    agents = []
    for i in range(3):
        agent_i = BaseAgent(f"agent_{i}", agent_config)
        agent_i.hrm.add_goal(f"Collaborative goal {i+1} for multi-agent system")
        agents.append(agent_i)
        await coordinator.register_agent(f"agent_{i}", ["analysis", "reasoning"], agent_i)

    # Coordinate reasoning across agents
    coordination_result = await coordinator.coordinate_agents(agents)

    print("✓ Multi-agent coordination completed!")
    print(f"   - Agents with HRM: {coordination_result['hrm_enabled_agents']}")
    print(f"   - Reasoning cycles: {coordination_result['reasoning_cycles']}")
    print(f"   - Total actions: {coordination_result['total_actions']}")
    print(f"   - Neural updates: {coordination_result['neural_updates']}")

    # Show HRM status
    print("\n8. Final HRM Status:")
    print("-" * 40)
    status = hrm.get_status()
    print(f"   Active Goals: {status['goals_count']}")
    print(f"   Generated Plans: {status['plans_count']}")
    print(f"   Executed Actions: {status['actions_count']}")
    print(f"   Experience Buffer: {status['replay_buffer_size']}")

    print("\n🎉 Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("   ✓ Hierarchical Reasoning (H-level + L-level)")
    print("   ✓ Adaptive Computation Time (ACT)")
    print("   ✓ Neural Network-based Planning")
    print("   ✓ Experience Replay Learning")
    print("   ✓ Multi-agent Coordination")
    print("   ✓ Dynamic Goal Management")
    print("   ✓ Real-time Status Monitoring")


async def demonstrate_learning():
    """Demonstrate the learning capabilities of HRM."""
    print("\n🧠 HRM Learning Demonstration")
    print("=" * 50)

    hrm_config = {
        "hidden_size": 64,
        "h_layers": 1,
        "l_layers": 2,
        "h_cycles": 1,
        "l_cycles": 2,
        "max_steps": 5,
        "learning_rate": 0.001,
        "embedding_size": 32,
        "output_size": 32,
        "replay_buffer_size": 50,
        "max_execution_steps": 10,
        "exploration_prob": 0.2
    }

    hrm = HRMReasoning(hrm_config)

    print("Training HRM on reasoning tasks...")

    # Simulate multiple reasoning cycles to demonstrate learning
    for cycle in range(5):
        print(f"\nCycle {cycle + 1}:")
        hrm.add_goal(f"Learning goal {cycle + 1}")
        result = hrm.reason_hierarchically()

        print(f"   Goals: {len(result['goals'])}, Plans: {len(result['plans'])}, Actions: {len(result['actions'])}")
        print(f"   Experience buffer: {len(hrm.replay_buffer)}")

        # Show learning progress
        if len(hrm.replay_buffer) >= 10:
            print("   📈 Learning from experience...")
            hrm._learn_from_experience()

    print("\n✓ Learning demonstration completed!")


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demonstrate_hrm_reasoning())
    asyncio.run(demonstrate_learning())
