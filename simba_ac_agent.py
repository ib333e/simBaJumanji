import functools
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale
from jumanji.types import TimeStep, Spec
from jumanji.training.agents.a2c import A2CAgent, A2CConfig

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.critics import ValueCritic

@dataclass(frozen=True)
class SimbaA2CConfig(A2CConfig):
    """Configuration for SimbaA2CAgent that extends Jumanji's A2CConfig."""
    # Network architecture configs
    actor_block_type: str = "residual"  # Options: "mlp", "residual", "transformer"
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 256
    actor_num_heads: int = 4  # For transformer
    actor_num_layers: int = 2  # For transformer
    
    critic_block_type: str = "residual"  # Options: "mlp", "residual", "transformer"
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_heads: int = 4  # For transformer
    critic_num_layers: int = 2  # For transformer
    
    mixed_precision: bool = False

    def __post_init__(self):
        # Validate network configurations
        if self.actor_block_type not in ["mlp", "residual", "transformer"]:
            raise ValueError(f"Invalid actor_block_type: {self.actor_block_type}")
        if self.critic_block_type not in ["mlp", "residual", "transformer"]:
            raise ValueError(f"Invalid critic_block_type: {self.critic_block_type}")

class SimbaA2CAgent(A2CAgent):
    """Actor-Critic agent using Simba's network architectures."""
    
    def __init__(
        self,
        observation_space: Spec,
        action_space: Spec,
        cfg: SimbaA2CConfig,
    ):
        super().__init__(observation_space, action_space, cfg)
        
        # Convert Jumanji specs to dimensions
        self._observation_dim = np.prod(observation_space.shape)
        self._action_dim = np.prod(action_space.shape)
        
        # Initialize Simba networks
        self._actor = self._init_actor_network()
        self._critic = self._init_critic_network()
        
    def _init_actor_network(self) -> Trainer:
        """Initialize Simba's actor network with appropriate architecture."""
        compute_dtype = jnp.float16 if self._cfg.mixed_precision else jnp.float32
        
        if self._cfg.actor_block_type == "mlp":
            network_def = NormalTanhPolicy(
                action_dim=self._action_dim,
                block_type="mlp",
                num_blocks=self._cfg.actor_num_blocks,
                hidden_dim=self._cfg.actor_hidden_dim,
                dtype=compute_dtype,
            )
        elif self._cfg.actor_block_type == "residual":
            network_def = NormalTanhPolicy(
                action_dim=self._action_dim,
                block_type="residual",
                num_blocks=self._cfg.actor_num_blocks,
                hidden_dim=self._cfg.actor_hidden_dim,
                dtype=compute_dtype,
            )
        else:  # transformer
            network_def = NormalTanhPolicy(
                action_dim=self._action_dim,
                block_type="transformer",
                num_blocks=self._cfg.actor_num_blocks,
                hidden_dim=self._cfg.actor_hidden_dim,
                num_heads=self._cfg.actor_num_heads,
                num_layers=self._cfg.actor_num_layers,
                dtype=compute_dtype,
            )
        
        return Trainer.create(
            network_def=network_def,
            network_inputs={
                "rngs": self._rng,
                "inputs": jnp.zeros((1, self._observation_dim)),
            },
            tx=optax.chain(
                optax.clip_by_global_norm(self._cfg.max_grad_norm),
                optax.adamw(
                    learning_rate=self._cfg.learning_rate,
                    weight_decay=self._cfg.weight_decay,
                ),
            ),
            dynamic_scale=dynamic_scale.DynamicScale() if self._cfg.mixed_precision else None,
        )
    
    def _init_critic_network(self) -> Trainer:
        """Initialize Simba's critic network with appropriate architecture."""
        compute_dtype = jnp.float16 if self._cfg.mixed_precision else jnp.float32
        
        if self._cfg.critic_block_type == "mlp":
            network_def = ValueCritic(
                block_type="mlp",
                num_blocks=self._cfg.critic_num_blocks,
                hidden_dim=self._cfg.critic_hidden_dim,
                dtype=compute_dtype,
            )
        elif self._cfg.critic_block_type == "residual":
            network_def = ValueCritic(
                block_type="residual",
                num_blocks=self._cfg.critic_num_blocks,
                hidden_dim=self._cfg.critic_hidden_dim,
                dtype=compute_dtype,
            )
        else:  # transformer
            network_def = ValueCritic(
                block_type="transformer",
                num_blocks=self._cfg.critic_num_blocks,
                hidden_dim=self._cfg.critic_hidden_dim,
                num_heads=self._cfg.critic_num_heads,
                num_layers=self._cfg.critic_num_layers,
                dtype=compute_dtype,
            )
        
        return Trainer.create(
            network_def=network_def,
            network_inputs={
                "rngs": self._rng,
                "observations": jnp.zeros((1, self._observation_dim)),
            },
            tx=optax.chain(
                optax.clip_by_global_norm(self._cfg.max_grad_norm),
                optax.adamw(
                    learning_rate=self._cfg.learning_rate,
                    weight_decay=self._cfg.weight_decay,
                ),
            ),
            dynamic_scale=dynamic_scale.DynamicScale() if self._cfg.mixed_precision else None,
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_actor_loss(
        self,
        params: Any,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Compute actor loss using Simba's policy network."""
        dist = self._actor.apply_fn({"params": params}, inputs=observations)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -self._cfg.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        info = {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
        }
        
        return total_loss, info
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_critic_loss(
        self,
        params: Any,
        observations: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Compute critic loss using Simba's value network."""
        values = self._critic.apply_fn({"params": params}, observations=observations)
        value_loss = ((values - returns) ** 2).mean()
        
        info = {
            "value_loss": value_loss,
        }
        
        return value_loss, info
    
    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, Any],
        training: bool,
    ) -> np.ndarray:
        """Sample actions using Simba's policy network."""
        observations = prev_timestep["observations"]
        # Flatten observations if they're not already flat
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
        
        self._rng, key = jax.random.split(self._rng)
        dist = self._actor(inputs=observations)
        actions = dist.sample(seed=key)
        
        return np.array(actions)
    
    def update(
        self,
        update_step: int,
        batch: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Update networks using Simba's training infrastructure."""
        # Convert numpy arrays to jax arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items()}
        
        # Compute advantages
        values = self._critic(observations=jax_batch["observations"])
        next_values = self._critic(observations=jax_batch["next_observations"])
        
        advantages = self._compute_gae(
            rewards=jax_batch["rewards"],
            values=values,
            next_values=next_values,
            dones=jax_batch["dones"],
        )
        
        # Update actor
        actor_grad_fn = jax.grad(self._compute_actor_loss, has_aux=True)
        actor_grad, actor_info = actor_grad_fn(
            self._actor.params,
            jax_batch["observations"],
            jax_batch["actions"],
            advantages,
        )
        self._actor = self._actor.apply_gradients(grads=actor_grad)
        
        # Update critic
        critic_grad_fn = jax.grad(self._compute_critic_loss, has_aux=True)
        critic_grad, critic_info = critic_grad_fn(
            self._critic.params,
            jax_batch["observations"],
            advantages + values,  # Returns = advantages + values
        )
        self._critic = self._critic.apply_gradients(grads=critic_grad)
        
        return {**actor_info, **critic_info}

class SimbaACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: Spec,
        action_space: Spec,
        cfg: SimbaACConfig,
    ):
        super().__init__(observation_space, action_space, cfg)
        
        # Convert Jumanji specs to dimensions
        self._observation_dim = np.prod(observation_space.shape)
        self._action_dim = np.prod(action_space.shape)
        
        self._rng, self._actor, self._critic = _init_simba_ac_networks(
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
            cfg=cfg,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, Any],
        training: bool,
    ) -> np.ndarray:
        observations = prev_timestep["observations"]
        # Flatten observations if they're not already flat
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
            
        self._rng, actions = _sample_actions(
            rng=self._rng,
            actor=self._actor,
            observations=observations,
            temperature=1.0 if training else 0.0,  # No exploration during evaluation
        )
        return np.array(actions)

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        # Convert numpy arrays to jax arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items()}
        
        self._rng, self._actor, self._critic, info = _update_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            batch=jax_batch,
            gamma=self._cfg.gamma,
            gae_lambda=self._cfg.gae_lambda,
            clip_ratio=self._cfg.clip_ratio,
            entropy_coef=self._cfg.entropy_coef,
            value_loss_coef=self._cfg.value_loss_coef,
        )
        return info 