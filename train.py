import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from jumanji.training import utils
from jumanji.training.setup_train import (
    setup_env, setup_agent, setup_evaluators, setup_training_state
)
from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState
from typing import Dict, Tuple

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
        # Tiny values to just to set up 
        "training": {
            "num_epochs": 10,
            "num_learner_steps_per_epoch": 5,
            "n_steps": 3,
            "total_batch_size": 1,
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


# Everything following is adapted from Jumanji's training/train.py
key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
env = setup_env(cfg=cfg)
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
    # probably dump to json here

    # Train
    with train_timer:
        state, train_metrics = epoch_fn(training_state)
        jax.block_until_ready((state, train_metrics))
        metrics = utils.first_from_device(train_metrics)
    print("Epoch", epoch, "Train:", metrics)