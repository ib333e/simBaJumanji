import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf, DictConfig
from jumanji.training.agents.base import Agent
from jumanji.training.agents.a2c import A2CAgent
from agents.ppo_agent import PPOAgent
from jumanji.env import Environment
from jumanji.environments import Tetris
from jumanji.training import utils
from jumanji.training.setup_train import (
    setup_env, setup_evaluators, setup_training_state
)
import jumanji.training.networks as networks
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState
from typing import Dict, Tuple

import json
import os
import matplotlib.pyplot as plt

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
            "num_learner_steps_per_epoch": 30,
            "n_steps": 150,
            "total_batch_size": 128,
        },
        "evaluation": {
            "eval_total_batch_size": 1024,
            "greedy_eval_total_batch_size": 1024,
        },
        "ppo": {
            "normalize_advantage": True,
            "num_minibatches": 4,
            "ppo_epochs": 5,
            "discount_factor": 0.9,
            "gae_lambda": 0.95,
            "l_td": 0.5,
            "l_en": 0.01,
            "max_grad_norm": 0.5,
            "learning_rate": 3e-4,
            "clip_epsilon": 0.2,
        },
    },
    "agent": "ppo"
})


# Everything following is adapted from Jumanji's training/train.py
key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
env = setup_env(cfg=cfg)

def setup_actor_critic_networks(cfg: DictConfig, env: Environment) -> ActorCriticNetworks:
    if cfg.env.name == "tetris":
        assert isinstance(env.unwrapped, Tetris)
        actor_critic_networks = networks.make_actor_critic_networks_tetris(
            tetris=env.unwrapped,
            conv_num_channels=cfg.env.network.conv_num_channels,
            tetromino_layers=cfg.env.network.tetromino_layers,
            head_layers=cfg.env.network.head_layers,
        )
    
    return actor_critic_networks



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

for epoch in range(cfg.env.training.num_epochs):
    # Eval
    key, eval_key = jax.random.split(key)
    
    with eval_timer:
        eval = stochastic_eval.run_evaluation(training_state.params_state, eval_key)
        jax.block_until_ready(eval)
        metrics = utils.first_from_device(eval)
    print("Epoch", epoch, "Evaluation:", metrics)
    eval_metrics = metrics
    # probably dump to json here

    # Train
    with train_timer:
        state, train_metrics = epoch_fn(training_state)
        jax.block_until_ready((state, train_metrics))
        metrics = utils.first_from_device(train_metrics)
    print("Epoch", epoch, "Train:", metrics)
    train_metrics = metrics

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