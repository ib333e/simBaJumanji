"""
PPO version of Jumanji's A2CAgent
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, NamedTuple, Tuple

from jumanji.env import Environment
from jumanji.training.agents.base import Agent
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.types import TrainingState

class PPOAgent(Agent):
    """ PPO adaptation of Jumanji's vanilla actor critic agent"""

    def __init__(
            self,
            env: Environment,
            n_steps: int,
            total_batch_size: int,
            actor_critic_networks: ActorCriticNetworks,
            optimizer: optax.GradientTransformation,
            normalize_advantage: bool,
            discount_factor: float,
            l_td: float,
            l_en: float,
    ) -> None:
        super().__init__(total_batch_size=total_batch_size)
        self.env = env
        self.observation_spec = env.observation_spec
        self.n_steps = n_steps