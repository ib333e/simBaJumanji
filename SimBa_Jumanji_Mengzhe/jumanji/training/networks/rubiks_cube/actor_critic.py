# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.env import Observation, RubiksCube
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)

class RSNorm(hk.Module):
    def __init__(self, name=None, eps=1e-6):
        super().__init__(name)
        self.eps = eps

    def __call__(self, x, timestep, is_training=True):
        shape = x.shape
        
        mu_t = hk.get_state("mu_t", x.shape, init=jnp.zeros)
        var_t = hk.get_state("var_t", x.shape, init=jnp.ones)

        def norm(o, mu, var):
            return jnp.divide(o - mu, jnp.sqrt((var**2) + self.eps))

        def train_fn(_):
            timestep_scalar = timestep[0] if isinstance(timestep, jnp.ndarray) and timestep.size > 0 else timestep

            def first(_):
                return x
            
            def rest(_):
                delta = x - mu_t
                new_mu = mu_t + delta / timestep_scalar
                new_var = ((timestep_scalar-1) / timestep_scalar) * ((var_t**2) + (delta**2)/timestep_scalar) 
                hk.set_state("mu_t", new_mu)
                hk.set_state("var_t", new_var)
                return norm(x, new_mu, new_var)

            normalized_input = jax.lax.cond(timestep_scalar==0, first, rest, operand=None)
            return normalized_input
         
        def eval_fn(_):
            norm_input = norm(x, mu_t, var_t)
            return norm_input
            
        new_obs = jax.lax.cond(
            is_training,
            train_fn, 
            eval_fn,
            operand=None
        )
        return new_obs

class ResidualMLP(hk.Module):
    def __init__(self, hidden_dim: int, name=None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(self, x: chex.Array) -> chex.Array:
        he_normal = hk.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        h = hk.Linear(self.hidden_dim * 4, w_init=he_normal)(h)
        h = jax.nn.relu(h)
        h = hk.Linear(self.hidden_dim, w_init=he_normal)(h)
        return x + h

def make_actor_critic_networks_rubiks_cube_simba(
    rubiks_cube: RubiksCube,
    cube_embed_dim: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `RubiksCube` environment with SimBa architecture."""
    action_spec_num_values = np.asarray(rubiks_cube.action_spec.num_values)
    num_actions = int(np.prod(action_spec_num_values))
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=action_spec_num_values
    )
    time_limit = rubiks_cube.time_limit
    policy_network = make_network_simba(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=num_actions,
        critic=False,
    )
    value_network = make_network_simba(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=None,  # Not needed for critic
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def make_network_simba(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
    num_actions: int = None,
    critic: bool = False,
) -> FeedForwardNetwork:
    """Create a SimBa network for the Rubik's cube environment."""
    
    def net(obs) -> chex.Array:
        # Get timestep for normalization
        timestep = obs.step_count
        
        # Normalize and embed cube
        cube_norm = RSNorm(name="cube_norm")
        cube = obs.cube.astype(jnp.float32)
        cube_flattened = jnp.reshape(cube, (cube.shape[0], -1))
        cube_normalized = cube_norm(cube_flattened, timestep)
        
        # Cube embedding
        cube_embedder = hk.Embed(vocab_size=len(Face), embed_dim=cube_embed_dim)
        cube_embedding = cube_embedder(obs.cube).reshape(*obs.cube.shape[:-3], -1)
        
        # Apply residual blocks to cube embedding
        x = hk.Linear(dense_layer_dims[0])(cube_embedding)
        
        # Apply residual MLP blocks
        for i, dim in enumerate(dense_layer_dims):
            x = ResidualMLP(hidden_dim=dim, name=f"residual_{i}")(x)
            
        # Step count embedding with normalization
        step_count = (obs.step_count / time_limit)[:, None]
        step_norm = RSNorm(name="step_norm")
        step_normalized = step_norm(step_count, timestep)
        step_count_embedder = hk.Linear(step_count_embed_dim)
        step_count_embedding = step_count_embedder(step_normalized)
        
        # Combine embeddings
        embedding = jnp.concatenate([x, step_count_embedding], axis=-1)
        
        # Post-layer normalization
        embedding = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(embedding)
        
        if critic:
            # Value head
            value = hk.nets.MLP((*dense_layer_dims, 1))(embedding)
            return jnp.squeeze(value, axis=-1)
        else:
            # Policy head
            logits = hk.nets.MLP((*dense_layer_dims, num_actions))(embedding)
            return logits

    # Transform with state for tracking statistics
    init_raw, apply_raw = hk.transform_with_state(net)

    def init_fn(rng, dummy_obs):
        params, state = init_raw(rng, dummy_obs)
        return {"params": params, "state": state}

    def apply_fn(pytree, obs):
        params, state = pytree["params"], pytree["state"]
        out, _ = apply_raw(params, state, None, obs)  # Ignore new state
        return out

    return FeedForwardNetwork(init=init_fn, apply=apply_fn)

def make_actor_critic_networks_rubiks_cube(
    rubiks_cube: RubiksCube,
    cube_embed_dim: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `RubiksCube` environment."""
    action_spec_num_values = np.asarray(rubiks_cube.action_spec.num_values)
    num_actions = int(np.prod(action_spec_num_values))
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=action_spec_num_values
    )
    time_limit = rubiks_cube.time_limit
    policy_network = make_actor_network(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=num_actions,
    )
    value_network = make_critic_network(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_torso_network_fn(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
) -> Callable[[Observation], chex.Array]:
    def torso_network_fn(observation: Observation) -> chex.Array:
        # Cube embedding
        cube_embedder = hk.Embed(vocab_size=len(Face), embed_dim=cube_embed_dim)
        cube_embedding = cube_embedder(observation.cube).reshape(*observation.cube.shape[:-3], -1)

        # Step count embedding
        step_count_embedder = hk.Linear(step_count_embed_dim)
        step_count_embedding = step_count_embedder(observation.step_count[:, None] / time_limit)

        embedding = jnp.concatenate([cube_embedding, step_count_embedding], axis=-1)
        return embedding

    return torso_network_fn


def make_actor_network(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
    num_actions: int,
) -> FeedForwardNetwork:
    torso_network_fn = make_torso_network_fn(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> chex.Array:
        embedding = torso_network_fn(observation)
        logits = hk.nets.MLP((*dense_layer_dims, num_actions))(embedding)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> FeedForwardNetwork:
    torso_network_fn = make_torso_network_fn(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> chex.Array:
        embedding = torso_network_fn(observation)
        value = hk.nets.MLP((*dense_layer_dims, 1))(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
