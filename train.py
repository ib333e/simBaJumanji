import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools
import chex
from omegaconf import OmegaConf, DictConfig
from jumanji.training.agents.base import Agent
from jumanji.training.agents.a2c import A2CAgent
from agents.ppo_agent import PPOAgent
from jumanji.env import Environment
from jumanji.environments.packing.tetris.env import Observation, Tetris
from jumanji.training import utils
from jumanji.training.setup_train import (
    setup_env, setup_evaluators, setup_training_state
)
from jumanji.training.networks.tetris.actor_critic import make_actor_critic_networks_tetris
from jumanji.training.networks.parametric_distribution import FactorisedActionSpaceParametricDistribution
import haiku as hk
from jumanji.training.networks.actor_critic import ActorCriticNetworks, FeedForwardNetwork
from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState
from typing import Dict, Tuple, Sequence
import json
import os
import matplotlib.pyplot as plt
from tqdm import trange

cfg_a2c = OmegaConf.create({
    "seed": 0,
    "env": {
        "name": "tetris",
        "registered_version": "Tetris-v0",
        "network": {
            "conv_num_channels": 64,
            "tetromino_layers": [16, 16],
            "head_layers": [128],
        },
        "training": {
            "num_epochs": 500,
            "num_learner_steps_per_epoch": 150,
            "n_steps": 30,
            "total_batch_size": 128,
        },
        "evaluation": {
            "eval_total_batch_size": 1024,
            "greedy_eval_total_batch_size": 1024,
        },
        "a2c": {
            "normalize_advantage": False,
            "discount_factor": 0.9,
            "bootstrapping_factor": 0.9,
            "l_pg": 1.0,
            "l_td": 1.0,
            "l_en": 0.01,
            "learning_rate": 3e-4,
        },
    },
    "agent": "a2c"
})

cfg = OmegaConf.create({
    "seed": 0,
    "env": {
        "name": "tetris",
        "registered_version": "Tetris-v0",
        "network": {
            "conv_num_channels": 64,
            "tetromino_layers": [16, 16],
            "head_layers": [128],
        },
        "training": {
            "num_epochs": 500,
            "num_learner_steps_per_epoch": 150,
            "n_steps": 30,
            "total_batch_size": 128,
        },
        "evaluation": {
            "eval_total_batch_size": 1024,
            "greedy_eval_total_batch_size": 1024,
        },
        "ppo": {
            "normalize_advantage": False,
            "num_minibatches": 4,
            "ppo_epochs": 8,
            "discount_factor": 0.9,
            "gae_lambda": 0.95,
            "l_td": 1.0,
            "l_en": 0.01,
            "max_grad_norm": 0.5,
            "learning_rate": 3e-4,
            "clip_epsilon": 0.1,
        },
    },
    "agent": "ppo"
})


# Everything following is adapted from Jumanji's training/train.py
key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
env = setup_env(cfg=cfg)
# to make it work for now
IS_TRAINING = False
def make_network_cnn(
    conv_num_channels: int,
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
    time_limit: int,
    critic: bool,
    num_residual_blocks: int = 2,
) -> FeedForwardNetwork:
    """ Adapted from jumanji.training.networks.tetris.actor_critic with following changes
        1. Define a residual feedforward block identical to the one in the paper
        2. maintains 
    """
    class ResidualFF(hk.Module):
        def __init__(self, hidden_dim: int, name=None):
            super().__init__(name=name)
            self.hidden_dim = hidden_dim

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            # pre layer norm
            y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            # inverted bottle neck MLP
            y = hk.Linear(4* self.hidden_dim)(y)
            y = jax.nn.relu(y)
            y = hk.Linear(self.hidden_dim)(y)
            return x + y
        
    class RSNorm(hk.Module):
        def __init__(self, name = None, eps= 1e-6):
            super().__init__(name)
            self.eps = eps

        def __call__(self, x, timestep, is_training):
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
        
    def network_fn(observation: Observation, rng=None) -> chex.Array:
        # Taking the RSNorm of input observations, only for tetris right now
        timestep = observation.step_count
        # the grid
        grid = observation.grid.astype(jnp.float32)[..., None]
        grid_norm = RSNorm(name="grid_norm")
        normed_grid = grid_norm(grid, timestep, IS_TRAINING)

        # the tetromino
        tet = observation.tetromino.astype(jnp.float32)[..., None]
        tet_norm = RSNorm(name="tet_norm")
        normed_tet = tet_norm(tet, timestep, IS_TRAINING)

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
        grid_embeddings = grid_net(normed_grid)  # [B, 2, 10, 64]
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
        tetromino_embeddings = tetromino_net(normed_tet)
        tetromino_embeddings = jnp.tile(
            tetromino_embeddings[:, None], (grid_embeddings.shape[1], 1)
        )
        norm_step_count = observation.step_count / time_limit
        norm_step_count = jnp.tile(norm_step_count[:, None, None], (grid_embeddings.shape[1], 1))

        embedding = jnp.concatenate(
            [grid_embeddings, tetromino_embeddings, norm_step_count], axis=-1
        )  # [B, 10, 145]

        # apply ResidualFF blocks
        D = embedding.shape[-1]
        block = lambda x: ResidualFF(D)(x)
        embedding = jax.vmap(
            lambda x_t: functools.reduce(lambda h, _: block(h), range(num_residual_blocks), x_t), in_axes=1, out_axes=1
        )(embedding)
        
        
        # post layer normalization
        embedding = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(embedding)

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

    init, apply = hk.without_apply_rng(hk.transform_with_state(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)

def setup_actor_critic_networks(cfg: DictConfig, env: Environment) -> ActorCriticNetworks:
    if cfg.env.name == "tetris":
        assert isinstance(env.unwrapped, Tetris)
        tetris = env.unwrapped
        conv_num_channels=cfg.env.network.conv_num_channels
        tetromino_layers=cfg.env.network.tetromino_layers
        head_layers=cfg.env.network.head_layers

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
        
        # COMMENT THIS OUT RETURN ac_networks IF YOU WANT TO RUN A TRAINING/EVAL LOOP WITHOUT SIMBA
        # ac_networks = make_actor_critic_networks_tetris(
        #     tetris=env.unwrapped,
        #     conv_num_channels=conv_num_channels,
        #     tetromino_layers=tetromino_layers,
        #     head_layers=head_layers
        # )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution
    )



def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent == "a2c":
        actor_critic_networks = setup_actor_critic_networks(cfg, env)
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        agent = A2CAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.env.a2c.normalize_advantage,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
        )
    elif cfg.agent == "ppo":
        actor_critic_networks = setup_actor_critic_networks(cfg, env)
        optimizer = optax.chain(optax.clip_by_global_norm(cfg.env.ppo.max_grad_norm),
                                optax.adam(cfg.env.ppo.learning_rate))
        agent = PPOAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            normalize_advantage=cfg.env.ppo.normalize_advantage,
            discount_factor=cfg.env.ppo.discount_factor,
            l_pg=1.0,
            l_td=cfg.env.ppo.l_td,
            l_en=cfg.env.ppo.l_en,
            clip_epsilon=cfg.env.ppo.clip_epsilon,
            num_minibatches=cfg.env.ppo.num_minibatches,
            ppo_epochs=cfg.env.ppo.ppo_epochs,
            bootstrapping_factor=cfg.env.ppo.gae_lambda,
            max_grad_norm=cfg.env.ppo.max_grad_norm,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Expected agent name to be in ['random', 'a2c'], got {cfg.agent}.")
    return agent


agent = setup_agent(cfg=cfg, env=env)
stochastic_eval, _ = setup_evaluators(cfg=cfg, agent=agent)
training_state = setup_training_state(env=env, agent=agent, key=init_key)
num_steps_per_epoch = (
    cfg.env.training.n_steps
    * cfg.env.training.total_batch_size
    * cfg.env.training.num_learner_steps_per_epoch
)
eval_timer = Timer(out_var_name="metrics")
train_timer = Timer(out_var_name="metrics", num_steps_per_timing=num_steps_per_epoch)

def epoch_function(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
    training_state, metrics = jax.lax.scan(
        lambda training_state, _: agent.run_epoch(training_state),
        training_state,
        None,
        cfg.env.training.num_learner_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, metrics
epoch_fn = jax.pmap(epoch_function, axis_name="devices")

def track_and_dump_metrics(epoch, eval_metrics, train_metrics, filename="metrics.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {
            "epochs": [],
            "returns": [],
            "total_steps": [],
            "total_time": [],
        }
    
    # Extract metrics
    returns = float(eval_metrics["episode_return"])
    eval_time = float(eval_metrics["time"])
    train_time = float(train_metrics["time"])
    
    # Calculate step count for this epoch
    steps_this_epoch = cfg.env.training.n_steps * cfg.env.training.total_batch_size * cfg.env.training.num_learner_steps_per_epoch
    
    # Update running totals
    total_steps = steps_this_epoch if epoch == 0 else (data["total_steps"][-1] + steps_this_epoch if data["total_steps"] else steps_this_epoch)
    total_time = eval_time + train_time if epoch == 0 else (data["total_time"][-1] + eval_time + train_time if data["total_time"] else eval_time + train_time)
    
    # Add to data
    data["epochs"].append(epoch)
    data["returns"].append(returns)
    data["total_steps"].append(total_steps)
    data["total_time"].append(total_time)
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    return data

# in memory could speed things up
data = {"epochs": [], "returns": [], "total_steps": [], "total_time": []}
eval_every = 5
save_every = 10
for epoch in trange(cfg.env.training.num_epochs):
    # Eval
    key, eval_key = jax.random.split(key)
    
    IS_TRAINING = False
    with eval_timer:
        eval = stochastic_eval.run_evaluation(training_state.params_state, eval_key)
        jax.block_until_ready(eval)
        metrics = utils.first_from_device(eval)
    eval_metrics = metrics

    # Train
    IS_TRAINING = True
    with train_timer:
        state, train_metrics = epoch_fn(training_state)
        jax.block_until_ready((state, train_metrics))
        metrics = utils.first_from_device(train_metrics)
    train_metrics = metrics

    data["epochs"].append(epoch)
    data["returns"].append()
    track_and_dump_metrics(epoch=epoch, eval_metrics=eval_metrics, train_metrics=train_metrics, filename=f"{cfg.agent}_{cfg.env.name}.json")
    training_state = state

def plot_metrics(metrics_file="metrics.json", save_dir="./plots", window_size=5):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metrics data
    with open(metrics_file, "r") as f:
        data = json.load(f)
    
    # Extract data
    epochs = data["epochs"]
    returns = data["returns"]
    total_steps = data["total_steps"]
    total_time = data["total_time"]
    
    # Calculate moving average for smoother curves
    # Using a rolling window to calculate average returns
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data  # Return original if not enough data points
        
        averages = []
        for i in range(len(data) - window_size + 1):
            window_avg = np.mean(data[i:i+window_size])
            averages.append(window_avg)
        
        # Pad the beginning to match original length
        padding = [averages[0]] * (window_size - 1)
        return padding + averages
        
    avg_returns = moving_average(returns, window_size)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot average returns vs steps
    ax1.plot(total_steps, avg_returns, marker='o', label='Average Return')
    ax1.plot(total_steps, returns, alpha=0.3, linestyle='--', label='Raw Return')
    ax1.set_xlabel('Total Steps')
    ax1.set_ylabel('Return')
    ax1.set_title(f'Average Return (Window={window_size}) vs Training Steps')
    ax1.grid(True)
    ax1.legend()

    # Plot average returns vs time
    ax2.plot(total_time, avg_returns, marker='o', color='orange', label='Average Return')
    ax2.plot(total_time, returns, alpha=0.3, linestyle='--', color='coral', label='Raw Return')
    ax2.set_xlabel('Total Time (seconds)')
    ax2.set_ylabel('Return')
    ax2.set_title(f'Average Return (Window={window_size}) vs Training Time')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "performance_plots.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to {plot_path}")

    return fig

plot_metrics(metrics_file=f"{cfg.agent}_{cfg.env.name}.json", save_dir="./plots")