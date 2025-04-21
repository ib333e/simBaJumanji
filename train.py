import jax
import jax.numpy as jnp
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

for epoch in range(cfg.env.training.num_epochs):
    # Eval
    key, eval_key = jax.random.split(key)
    
    with eval_timer:
        eval = stochastic_eval.run_evaluation(training_state.params_state, eval_key)
        jax.block_until_ready(eval)
        metrics = utils.first_from_device(eval)
    print("Epoch", epoch, "Evaluation:", metrics)

    # Train
    with train_timer:
        state, train_metrics = epoch_fn(training_state)
        jax.block_until_ready((state, train_metrics))
        metrics = utils.first_from_device(train_metrics)
    print("Epoch", epoch, "Train:", metrics)