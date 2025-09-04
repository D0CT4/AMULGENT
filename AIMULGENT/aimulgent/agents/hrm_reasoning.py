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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for improved stability."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stability
        original_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(original_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function for improved performance."""
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8/3)  # Standard SwiGLU expansion
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class HierarchicalReasoningLayer(nn.Module):
    """Enhanced hierarchical reasoning layer with modern components."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, expansion: float = 2.0, 
                 use_rmsnorm: bool = False, use_swiglu: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        
        # Self-attention
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP or SwiGLU
        if use_swiglu:
            self.mlp = SwiGLU(hidden_size)
        else:
            inter_size = int(hidden_size * expansion)
            self.gate_up_proj = nn.Linear(hidden_size, inter_size * 2)
            self.down_proj = nn.Linear(inter_size, hidden_size)
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
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
        if self.use_swiglu:
            mlp_output = self.mlp(x)
        else:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            mlp_output = self.down_proj(F.silu(gate) * up)
        
        # Residual + norm
        x = self.norm2(x + mlp_output)
        return x


class AdaptiveComputationTime(nn.Module):
    """Enhanced ACT mechanism with separate halt/continue Q-values."""
    
    def __init__(self, hidden_size: int, max_steps: int = 10, exploration_prob: float = 0.1):
        super().__init__()
        self.max_steps = max_steps
        self.exploration_prob = exploration_prob
        
        # Enhanced Q-network with separate halt/continue heads
        self.q_halt_head = nn.Linear(hidden_size, 1)
        self.q_continue_head = nn.Linear(hidden_size, 1)
        
        # Initialize halt head with slight bias towards continuing (exploration)
        nn.init.constant_(self.q_halt_head.bias, -0.1)
        nn.init.constant_(self.q_continue_head.bias, 0.1)
        
    def forward(self, hidden_state: torch.Tensor, step: int, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute enhanced halting probabilities with separate Q-values.
        Returns:
        - should_halt: scalar tensor (0.0 or 1.0)
        - q_values: tensor with shape [batch, 2] containing [q_halt, q_continue]
        """
        # Separate Q-values for halt and continue
        q_halt = self.q_halt_head(hidden_state)  # [batch, 1]
        q_continue = self.q_continue_head(hidden_state)  # [batch, 1]
        
        # Combine into q_values tensor [batch, 2]
        q_values = torch.cat([q_halt, q_continue], dim=-1)  # [batch, 2]
        
        # Aggregate decision for compatibility
        agg_q_halt = q_halt.mean(dim=0, keepdim=True)  # [1, 1]
        agg_q_continue = q_continue.mean(dim=0, keepdim=True)  # [1, 1]
        
        if training and torch.rand(1).item() < self.exploration_prob:
            # Exploration: random halting with curriculum (shorter early, longer later)
            min_steps = max(2, min(self.max_steps, step + torch.randint(1, 4, (1,)).item()))
            should_halt = torch.tensor(step >= min_steps, dtype=torch.float)
        else:
            # Exploitation: use Q-values
            should_halt = (agg_q_halt > agg_q_continue).float()
        
        # Always halt at max steps
        max_steps_reached = torch.tensor(step >= self.max_steps, dtype=torch.float)
        should_halt = torch.clamp(should_halt + max_steps_reached, 0, 1)
        
        return should_halt.squeeze(), (q_values.squeeze(0) if q_values.shape[0] == 1 else q_values)


class HierarchicalReasoningModel(nn.Module):
    """Enhanced Hierarchical Reasoning Model with modern components."""
    
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
        
        # Enhanced architecture options
        self.use_rmsnorm = config.get("use_rmsnorm", False)
        self.use_swiglu = config.get("use_swiglu", False)
        self.use_iterative_refinement = config.get("use_iterative_refinement", False)
        self.refinement_segments = config.get("refinement_segments", 2)
        
        # H-level (high-level planning) with enhancements
        self.h_level = nn.ModuleList([
            HierarchicalReasoningLayer(
                self.hidden_size, 
                use_rmsnorm=self.use_rmsnorm,
                use_swiglu=self.use_swiglu
            ) for _ in range(self.h_layers)
        ])
        
        # L-level (low-level execution) with enhancements
        self.l_level = nn.ModuleList([
            HierarchicalReasoningLayer(
                self.hidden_size,
                use_rmsnorm=self.use_rmsnorm,
                use_swiglu=self.use_swiglu
            ) for _ in range(self.l_layers)
        ])
        
        # Enhanced initial states with better initialization
        self.h_init = nn.Parameter(torch.empty(self.hidden_size))
        self.l_init = nn.Parameter(torch.empty(self.hidden_size))
        self._init_states()
        
        # Enhanced ACT mechanism
        self.act = AdaptiveComputationTime(self.hidden_size, self.max_steps)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, config.get("output_size", 128))
        
        # Optional input projection if embedding size differs
        self.input_proj = None
        if self.embedding_size != self.hidden_size:
            self.input_proj = nn.Linear(self.embedding_size, self.hidden_size)
    
    def _init_states(self):
        """Enhanced initialization strategy."""
        # Use truncated normal initialization for better training stability
        std = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            self.h_init.normal_(mean=0.0, std=std)
            self.l_init.normal_(mean=0.0, std=std)
            # Clamp to reasonable range
            self.h_init.clamp_(-2*std, 2*std)
            self.l_init.clamp_(-2*std, 2*std)
        
    def forward(self, input_embedding: torch.Tensor, steps: int = 0) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Enhanced forward pass with optional iterative refinement."""
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
        if self.input_proj is not None:
            input_embedding = self.input_proj(input_embedding)
        elif input_embedding.size(-1) != self.hidden_size:
            # Create temporary projection if sizes don't match
            if not hasattr(self, '_temp_input_proj'):
                self._temp_input_proj = nn.Linear(
                    input_embedding.size(-1), 
                    self.hidden_size, 
                    bias=False
                ).to(input_embedding.device)
            input_embedding = self._temp_input_proj(input_embedding)
        
        # Enhanced hierarchical reasoning with optional iterative refinement
        if self.use_iterative_refinement:
            return self._forward_with_refinement(h_state, l_state, input_embedding, steps)
        else:
            return self._forward_standard(h_state, l_state, input_embedding, steps)
    
    def _forward_standard(self, h_state: torch.Tensor, l_state: torch.Tensor, 
                         input_embedding: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Standard hierarchical reasoning cycles."""
        for h_cycle in range(self.h_cycles):
            # H-level processing
            for h_layer in self.h_level:
                h_state = h_layer(h_state.unsqueeze(1)).squeeze(1)

            # L-level processing
            for l_cycle in range(self.l_cycles):
                for l_layer in self.l_level:
                    # Combine L-state with input and H-state
                    combined_input = l_state + input_embedding + h_state * 0.1  # Small H influence
                    l_state = l_layer(combined_input.unsqueeze(1)).squeeze(1)

                # Update H-level with L-level information
                h_state = h_state + l_state * 0.1  # Small L influence

        # ACT decision
        should_halt, q_values = self.act(h_state, step=steps, training=self.training)

        # Output
        output = self.output_proj(h_state)

        return output, q_values, bool(should_halt.item() > 0.5)
    
    def _forward_with_refinement(self, h_state: torch.Tensor, l_state: torch.Tensor,
                                input_embedding: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Enhanced forward with iterative refinement (inspired by external research)."""
        outputs = []
        
        for segment in range(self.refinement_segments):
            # Standard reasoning cycle
            h_working = h_state
            l_working = l_state
            
            for h_cycle in range(self.h_cycles):
                # H-level processing with refinement
                for h_layer in self.h_level:
                    h_working = h_layer(h_working.unsqueeze(1)).squeeze(1)

                # L-level processing with cross-connections
                for l_cycle in range(self.l_cycles):
                    for l_layer in self.l_level:
                        # Enhanced combination with previous outputs
                        combined_input = l_working + input_embedding + h_working * 0.1
                        if outputs:  # Add refinement from previous segment
                            # Project previous output back to hidden size for combination
                            prev_output_hidden = outputs[-1]
                            if prev_output_hidden.size(-1) != self.hidden_size:
                                # Create a temporary projection for refinement
                                if not hasattr(self, '_refinement_proj'):
                                    self._refinement_proj = nn.Linear(
                                        prev_output_hidden.size(-1), 
                                        self.hidden_size, 
                                        bias=False
                                    ).to(prev_output_hidden.device)
                                prev_output_hidden = self._refinement_proj(prev_output_hidden)
                            combined_input = combined_input + prev_output_hidden * 0.05
                        l_working = l_layer(combined_input.unsqueeze(1)).squeeze(1)

                    # Update H-level with refined L-level information
                    h_working = h_working + l_working * 0.1

            # Store intermediate output for refinement
            segment_output = self.output_proj(h_working)
            outputs.append(segment_output)
            
            # Detach for gradient stability (from research insights)
            if segment < self.refinement_segments - 1:
                h_state = h_working.detach()
                l_state = l_working.detach()
            else:
                h_state = h_working
                l_state = l_working

        # ACT decision on final state
        should_halt, q_values = self.act(h_state, step=steps, training=self.training)

        # Final output is the last refined output
        final_output = outputs[-1]

        return final_output, q_values, bool(should_halt.item() > 0.5)


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
        """Enhanced learning from experience replay buffer with separate Q-values."""
        batch_size = min(32, len(self.replay_buffer))
        batch = self.replay_buffer[-batch_size:]

        # Each state is [1, emb], concatenate along batch to get [B, emb]
        states = torch.cat([exp["state"] for exp in batch], dim=0).to(self.device)
        q_halt_targets = []
        q_continue_targets = []

        for exp in batch:
            if exp["done"]:
                # Terminal state: halt should be high, continue should be low
                halt_target = float(exp["reward"])
                continue_target = 0.0
            else:
                with torch.no_grad():
                    _, next_q, _ = self.model(exp["next_state"].to(self.device))
                    # next_q is [2] containing [q_halt, q_continue]
                    if next_q.dim() == 1 and next_q.size(0) == 2:
                        next_q_halt, next_q_continue = next_q[0], next_q[1]
                    else:
                        next_q_halt = next_q[:, 0].mean()
                        next_q_continue = next_q[:, 1].mean()
                    
                    # Use appropriate Q-value for target
                    max_next_q = max(float(next_q_halt), float(next_q_continue))
                    halt_target = float(exp["reward"]) + 0.99 * max_next_q * 0.8  # Slight discount for halting
                    continue_target = float(exp["reward"]) + 0.99 * max_next_q
            
            q_halt_targets.append(halt_target)
            q_continue_targets.append(continue_target)

        q_halt_targets = torch.tensor(q_halt_targets, dtype=torch.float, device=self.device)
        q_continue_targets = torch.tensor(q_continue_targets, dtype=torch.float, device=self.device)

        # Update model with enhanced loss
        self.optimizer.zero_grad()
        current_q = self.model(states)[1]  # [B, 2]
        
        if current_q.dim() == 1:
            current_q = current_q.unsqueeze(0)
        
        current_q_halt = current_q[:, 0]
        current_q_continue = current_q[:, 1]
        
        # Separate losses for halt and continue Q-values
        halt_loss = F.mse_loss(current_q_halt, q_halt_targets)
        continue_loss = F.mse_loss(current_q_continue, q_continue_targets)
        
        # Combined loss with slight preference for learning to continue (exploration)
        total_loss = halt_loss + 1.1 * continue_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        self.logger.debug(f"Enhanced learning: halt_loss={halt_loss.item():.4f}, continue_loss={continue_loss.item():.4f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Enhanced status reporting with new metrics."""
        base_status = {
            "goals_count": len(self.strategic_layer),
            "plans_count": len(self.tactical_layer),
            "actions_count": len(self.operational_layer),
            "replay_buffer_size": len(self.replay_buffer),
            "model_stats": self.get_model_stats(),
            "active_goals": self.strategic_layer,
            "pending_plans": [p for p in self.tactical_layer if not any(a["plan_id"] == p["plan_id"] for a in self.operational_layer)]
        }
        
        # Add enhanced metrics
        if hasattr(self.model, 'use_rmsnorm'):
            base_status["architecture"] = {
                "use_rmsnorm": self.model.use_rmsnorm,
                "use_swiglu": self.model.use_swiglu,
                "use_iterative_refinement": self.model.use_iterative_refinement,
                "refinement_segments": self.model.refinement_segments if hasattr(self.model, 'refinement_segments') else None
            }
        
        # Add learning metrics
        if self.replay_buffer:
            avg_reward = sum(exp["reward"] for exp in self.replay_buffer[-10:]) / min(10, len(self.replay_buffer))
            completion_rate = sum(1 for exp in self.replay_buffer[-20:] if exp["done"]) / min(20, len(self.replay_buffer))
            base_status["learning_metrics"] = {
                "avg_recent_reward": avg_reward,
                "completion_rate": completion_rate,
                "buffer_utilization": len(self.replay_buffer) / self.max_buffer_size
            }
        
        return base_status
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Enhanced model statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        base_stats = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "max_steps": self.max_steps
        }
        
        # Add enhanced architecture stats
        if hasattr(self.model, 'use_rmsnorm'):
            base_stats.update({
                "enhanced_features": {
                    "rmsnorm": self.model.use_rmsnorm,
                    "swiglu": self.model.use_swiglu,
                    "iterative_refinement": self.model.use_iterative_refinement
                }
            })
        
        return base_stats