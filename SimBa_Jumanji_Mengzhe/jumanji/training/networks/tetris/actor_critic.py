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

from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.packing.tetris.env import Observation, Tetris
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)


def make_actor_critic_networks_tetris(
    tetris: Tetris,
    conv_num_channels: int,
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Tetris` environment."""

    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(tetris.action_spec.num_values)
    )
    policy_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
        time_limit=tetris.time_limit,
        critic=False,
    )
    value_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
        time_limit=tetris.time_limit,
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_cnn(
    conv_num_channels: int,
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
    time_limit: int,
    critic: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        grid_net = hk.Sequential(
            [
                hk.Conv2D(conv_num_channels, (3, 5), (1, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 5), (2, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 5), (2, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 3), (2, 1)),
                jax.nn.relu,
            ]
        )
        grid_embeddings = grid_net(observation.grid.astype(float)[..., None])  # [B, 2, 10, 64]
        grid_embeddings = jnp.transpose(grid_embeddings, [0, 2, 1, 3])  # [B, 10, 2, 64]
        grid_embeddings = jnp.reshape(
            grid_embeddings, [*grid_embeddings.shape[:2], -1]
        )  # [B, 10, 128]

        tetromino_net = hk.Sequential(
            [
                hk.Flatten(),
                hk.nets.MLP(tetromino_layers, activate_final=True),
            ]
        )
        tetromino_embeddings = tetromino_net(observation.tetromino.astype(float))
        tetromino_embeddings = jnp.tile(
            tetromino_embeddings[:, None], (grid_embeddings.shape[1], 1)
        )
        norm_step_count = observation.step_count / time_limit
        norm_step_count = jnp.tile(norm_step_count[:, None, None], (grid_embeddings.shape[1], 1))

        embedding = jnp.concatenate(
            [grid_embeddings, tetromino_embeddings, norm_step_count], axis=-1
        )  # [B, 10, 145]

        if critic:
            embedding = jnp.sum(embedding, axis=-2)  # [B, 145]
            value = hk.nets.MLP((*head_layers, 1))(embedding)  # [B, 1]
            return jnp.squeeze(value, axis=-1)  # [B]
        else:
            num_rotations = observation.action_mask.shape[-2]
            logits = hk.nets.MLP((*head_layers, num_rotations))(embedding)  # [B, 10, 4]
            logits = jnp.transpose(logits, [0, 2, 1])  # [B, 4, 10]
            masked_logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            ).reshape(observation.action_mask.shape[0], -1)
            return masked_logits  # [B, 40]

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)

def make_actor_critic_tetris_simba(tetris: Tetris,
    conv_num_channels: int,
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Tetris` environment."""
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(tetris.action_spec.num_values)
    )
    policy_network = make_network_cnn_simba(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
        time_limit=tetris.time_limit,
        critic=False,
    )
    value_network = make_network_cnn_simba(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
        time_limit=tetris.time_limit,
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

class ResidualMLP(hk.Module):
    def __init__(self, hidden_dim: int, name=None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(self, x: chex.Array) -> chex.Array:
        #Assume feed once now
        he_normal = hk.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        h = hk.Linear(self.hidden_dim * 4, w_init=he_normal)(h)
        h = jax.nn.relu(h)
        h = hk.Linear(self.hidden_dim, w_init=he_normal)(h)
        return x + h
        

def make_network_cnn_simba(
    conv_num_channels: int,
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
    time_limit: int,
    critic: bool,
) -> FeedForwardNetwork:
    
    class RSNorm(hk.Module):
        def __init__(self, name = None, eps= 1e-6):
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
                return normalized_input# don't forget to reshape it
             
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

    def net(obs) -> chex.Array:
        timestep = obs.step_count
        grid = obs.grid.astype(jnp.float32)[..., None]
        grid_norm = RSNorm(name="grid_norm")
        g = grid_norm(grid, timestep)
        g = hk.Conv2D(conv_num_channels, (3,5), (1,1), padding="SAME")(g); g = jax.nn.relu(g)
        g = hk.Conv2D(conv_num_channels, (3,5), (2,1), padding="SAME")(g); g = jax.nn.relu(g)
        g = hk.Conv2D(conv_num_channels, (3,5), (2,1), padding="SAME")(g); g = jax.nn.relu(g)
        g = hk.Conv2D(conv_num_channels, (3,3), (2,1), padding="SAME")(g); g = jax.nn.relu(g)
        # shape now [B,2,10,C]
        g = jnp.transpose(g, (0,2,1,3)).reshape(g.shape[0], 10, -1)   # [B,10,128]
        g = ResidualMLP(hidden_dim=g.shape[-1])(g)
        g = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(g)
        #g = ResidualMLP(hidden_dim=g.shape[-1])(g)

        t_norm = RSNorm(name="tetri_norm")
        t = t_norm(grid, timestep)
        t = hk.Flatten()(obs.tetromino.astype(jnp.float32))

        for w in tetromino_layers:
            #t = ResidualMLP(w)(t)
            t = hk.nets.MLP([w], activate_final=True)(t)
            t = ResidualMLP(w)(t)
        
        t = jnp.repeat(t[:, None, :], 10, axis=1)                     # [B,10,*]
        t = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(t)                       # (B, 16)

        s = (obs.step_count / time_limit)[:, None, None]              # [B,1,1]
        s = jnp.repeat(s, 10, axis=1)                                 # [B,10,1]

        emb = jnp.concatenate([g, t, s], axis=-1)

        if critic:
            embedding = jnp.sum(emb, axis=-2)  # [B, 145]
            value = hk.nets.MLP((*head_layers, 1))(embedding)  # [B, 1]
            return jnp.squeeze(value, axis=-1)  # [B]
        else:
            num_rotations = obs.action_mask.shape[-2]
            logits = hk.nets.MLP((*head_layers, num_rotations))(emb)  # [B, 10, 4]
            logits = jnp.transpose(logits, [0, 2, 1])  # [B, 4, 10]2
            masked_logits = jnp.where(
                obs.action_mask, logits, jnp.finfo(jnp.float32).min
            ).reshape(obs.action_mask.shape[0], -1)
            return masked_logits  # [B, 40]

    init_raw, apply_raw = hk.transform_with_state(net)

    def init_fn(rng, dummy_obs):
        params, state = init_raw(rng, dummy_obs)
        return {"params": params, "state": state}          # ONE pytree

    def apply_fn(pytree, obs):
        params, state = pytree["params"], pytree["state"]
        out, _ = apply_raw(params, state, None, obs)       # ignore new_state
        return out

    return FeedForwardNetwork(init=init_fn, apply=apply_fn)

