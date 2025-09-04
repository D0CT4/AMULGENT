"""
Hierarchical Reasoning Model (HRM) Implementation
Advanced hierarchical decision-making layers with Adaptive Computation Time (ACT).
"""

import logging
import math
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalReasoningLayer(nn.Module):
    """Single hierarchical reasoning layer with attention and MLP."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, expansion: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        inter_size = int(hidden_size * expansion)
        self.gate_up_proj = nn.Linear(hidden_size, inter_size * 2)
        self.down_proj = nn.Linear(inter_size, hidden_size)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Residual + norm
        x = self.norm1(x + attn_output)
        
        # MLP
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        mlp_output = self.down_proj(F.silu(gate) * up)
        
        # Residual + norm
        x = self.norm2(x + mlp_output)
        return x


class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time (ACT) mechanism for dynamic halting."""
    
    def __init__(self, hidden_size: int, max_steps: int = 10, exploration_prob: float = 0.1):
        super().__init__()
        self.max_steps = max_steps
        self.exploration_prob = exploration_prob
        
        # Q-network for halting decisions
        self.q_head = nn.Linear(hidden_size, 2)  # halt vs continue
        
    def forward(self, hidden_state: torch.Tensor, step: int, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute halting probabilities.
        Returns:
        - should_halt: scalar tensor (0.0 or 1.0)
        - q_logits: per-sample logits, shape [batch, 2] when batch>1, else [2]
        """
        # Per-sample logits
        q_logits = self.q_head(hidden_state)  # [batch, 2] or [1, 2]
        
        # Aggregate decision into a scalar for simplicity/compat with tests
        agg_logits = q_logits.mean(dim=0, keepdim=True)  # [1, 2]
        q_halt, q_continue = agg_logits.chunk(2, dim=-1)
        
        if training and torch.rand(1).item() < self.exploration_prob:
            # Exploration: random halting
            min_steps = torch.randint(2, self.max_steps + 1, (1,), dtype=torch.float)
            should_halt = torch.tensor(step >= min_steps.item(), dtype=torch.float)
        else:
            # Exploitation: use aggregated Q-values
            should_halt = (q_halt > q_continue).float()
        
        # Always halt at max steps
        max_steps_reached = torch.tensor(step >= self.max_steps, dtype=torch.float)
        should_halt = should_halt + max_steps_reached
        should_halt = torch.clamp(should_halt, 0, 1)
        
        return should_halt.squeeze(), (q_logits.squeeze(0) if q_logits.shape[0] == 1 else q_logits)


class HierarchicalReasoningModel(nn.Module):
    """Advanced Hierarchical Reasoning Model with H-level and L-level modules."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 256)
        self.embedding_size = config.get("embedding_size", self.hidden_size)
        self.h_layers = config.get("h_layers", 2)
        self.l_layers = config.get("l_layers", 4)
        self.h_cycles = config.get("h_cycles", 2)
        self.l_cycles = config.get("l_cycles", 4)
        self.max_steps = config.get("max_steps", 10)
        
        # H-level (high-level planning)
        self.h_level = nn.ModuleList([
            HierarchicalReasoningLayer(self.hidden_size) for _ in range(self.h_layers)
        ])
        
        # L-level (low-level execution)
        self.l_level = nn.ModuleList([
            HierarchicalReasoningLayer(self.hidden_size) for _ in range(self.l_layers)
        ])
        
        # Initial states
        self.h_init = nn.Parameter(torch.empty(self.hidden_size))
        self.l_init = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.h_init, mean=0.0, std=1.0)
        nn.init.normal_(self.l_init, mean=0.0, std=1.0)
        
        # ACT mechanism
        self.act = AdaptiveComputationTime(self.hidden_size, self.max_steps)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, config.get("output_size", 128))
        
        # Optional input projection if embedding size differs
        self.input_proj = None
        if self.embedding_size != self.hidden_size:
            self.input_proj = nn.Linear(self.embedding_size, self.hidden_size)
        
    def forward(self, input_embedding: torch.Tensor, steps: int = 0) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Forward pass with hierarchical reasoning."""
        # Move to model device if needed
        model_device = next(self.parameters()).device
        input_embedding = input_embedding.to(model_device)
        # Ensure 2D [batch, dim]
        if input_embedding.dim() == 3 and input_embedding.size(1) == 1:
            input_embedding = input_embedding.squeeze(1)
        elif input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
        batch_size = input_embedding.size(0)
        
        # Initialize states - ensure they match input dimensions
        h_state = self.h_init.unsqueeze(0).expand(batch_size, -1)
        l_state = self.l_init.unsqueeze(0).expand(batch_size, -1)
        
        # Ensure input embedding matches hidden size via persistent projection
        if self.input_proj is not None and input_embedding.size(-1) != self.hidden_size:
            input_embedding = self.input_proj(input_embedding)

        # Hierarchical reasoning cycles
        for h_cycle in range(self.h_cycles):
            # H-level processing
            for h_layer in self.h_level:
                h_state = h_layer(h_state.unsqueeze(1)).squeeze(1)

            # L-level processing
            for l_cycle in range(self.l_cycles):
                for l_layer in self.l_level:
                    # Combine L-state with input
                    combined_input = l_state + input_embedding
                    l_state = l_layer(combined_input.unsqueeze(1)).squeeze(1)

                # Update H-level with L-level information
                h_state = h_state + l_state

        # ACT decision
        should_halt, q_values = self.act(h_state, step=steps, training=self.training)

        # Output
        output = self.output_proj(h_state)

        return output, q_values, bool(should_halt.item() > 0.5)


class HRMReasoning:
    """
    Advanced Hierarchical Reasoning Model for agents.
    Features: H-level/L-level architecture, ACT, Q-learning for halting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Accept both flat and nested {'hrm': {...}} configurations
        self.config = config
        self.hrm_cfg = config.get("hrm", config)
        self.logger = logging.getLogger(__name__)
        self.strategic_layer: List[str] = []  # High-level goals
        self.tactical_layer: List[Dict[str, Any]] = []   # Mid-level plans
        self.operational_layer: List[Dict[str, Any]] = []  # Low-level actions
        
        # Neural components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HierarchicalReasoningModel(self.hrm_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hrm_cfg.get("learning_rate", 1e-4))
        
        # Copy config attributes for convenience
        self.h_layers = self.hrm_cfg.get("h_layers", 2)
        self.l_layers = self.hrm_cfg.get("l_layers", 4)
        self.max_steps = self.hrm_cfg.get("max_steps", 10)
        
        # Experience replay for Q-learning
        self.replay_buffer = []
        self.max_buffer_size = self.hrm_cfg.get("replay_buffer_size", 1000)
        
    def add_goal(self, goal: str) -> None:
        """Add a strategic goal."""
        self.strategic_layer.append(goal)
        self.logger.info(f"Added strategic goal: {goal}")
    
    def plan_tactical(self, goal: str) -> List[Dict[str, Any]]:
        """Generate tactical plans using neural reasoning."""
        # Convert goal to embedding (simplified)
        goal_embedding = self._text_to_embedding(goal)
        
        plans = []
        for i in range(3):  # Generate multiple plans
            plan_output, _, _ = self.model(goal_embedding)
            plan = {
                "plan_id": f"plan_{len(self.tactical_layer) + i}",
                "goal": goal,
                "description": f"Neural-generated plan for {goal}",
                "embedding": plan_output.detach(),
                "priority": 1,
                "confidence": torch.sigmoid(plan_output.mean()).item()
            }
            plans.append(plan)
        
        self.tactical_layer.extend(plans)
        return plans
    
    def execute_operational(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute operational actions with ACT."""
        actions = []
        steps = 0
        halted = False
        
        while not halted and steps < self.config.get("max_execution_steps", 20):
            action_output, q_values, should_halt = self.model(plan["embedding"], steps)
            
            action = {
                "action_id": f"action_{len(self.operational_layer)}",
                "plan_id": plan["plan_id"],
                "description": f"Neural action at step {steps}",
                "output": action_output.detach(),
                "q_values": q_values.detach(),
                "step": steps,
                "status": "completed" if should_halt else "in_progress"
            }
            actions.append(action)
            
            # Store experience for learning
            self.replay_buffer.append({
                "state": plan["embedding"],
                "action": action_output,
                "q_values": q_values,
                "reward": 1.0 if should_halt else 0.1,  # Simple reward
                "next_state": action_output,
                "done": should_halt
            })
            
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer.pop(0)
            
            steps += 1
            halted = should_halt
        
        self.operational_layer.extend(actions)
        self.logger.info(f"Executed {len(actions)} actions with ACT")
        return actions
    
    def reason_hierarchically(self) -> Dict[str, Any]:
        """Full HRM reasoning cycle with learning."""
        for goal in self.strategic_layer:
            plans = self.plan_tactical(goal)
            for plan in plans:
                self.execute_operational(plan)
        
        # Learning step
        if len(self.replay_buffer) >= 32:  # Minibatch size
            self._learn_from_experience()
        
        return {
            "goals": self.strategic_layer,
            "plans": self.tactical_layer,
            "actions": self.operational_layer,
            "model_stats": self.get_model_stats()
        }
    
    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """Simple text to embedding conversion (placeholder for actual embedding model)."""
        # This would be replaced with actual text embedding model
        embedding_size = self.hrm_cfg.get("embedding_size", 128)
        # Create a proper 2D tensor [batch_size=1, embedding_size]
        embedding = torch.randn(1, embedding_size).to(self.device)
        return embedding
    
    def _learn_from_experience(self) -> None:
        """Learn from experience replay buffer."""
        batch_size = min(32, len(self.replay_buffer))
        batch = self.replay_buffer[-batch_size:]

        # Each state is [1, emb], concatenate along batch to get [B, emb]
        states = torch.cat([exp["state"] for exp in batch], dim=0).to(self.device)
        q_targets = []

        for exp in batch:
            if exp["done"]:
                target = exp["reward"]
            else:
                with torch.no_grad():
                    _, next_q, _ = self.model(exp["next_state"].to(self.device))
                    # Bootstrap with max over actions
                    next_max = next_q.max(dim=-1).values
                    target = float(exp["reward"]) + 0.99 * float(next_max.mean().item())
            q_targets.append(target)

        q_targets = torch.tensor(q_targets, dtype=torch.float, device=self.device)

        # Update model
        self.optimizer.zero_grad()
        current_q = self.model(states)[1]  # [B, 2]
        # Regress against the best action value as a simple target
        current_max = current_q.max(dim=-1).values  # [B]
        loss = F.mse_loss(current_max, q_targets)
        loss.backward()
        self.optimizer.step()
        
        self.logger.debug(f"Learning step completed, loss: {loss.item()}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current HRM status."""
        return {
            "goals_count": len(self.strategic_layer),
            "plans_count": len(self.tactical_layer),
            "actions_count": len(self.operational_layer),
            "replay_buffer_size": len(self.replay_buffer),
            "model_stats": self.get_model_stats(),
            "active_goals": self.strategic_layer,
            "pending_plans": [p for p in self.tactical_layer if not any(a["plan_id"] == p["plan_id"] for a in self.operational_layer)]
        }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "max_steps": self.max_steps
        }