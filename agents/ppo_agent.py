"""
PPO version of Jumanji's A2CAgent.
"""

import jax
import jax.numpy as jnp
import optax
import rlax
import chex
import functools
import haiku as hk
from typing import Any, Dict, NamedTuple, Tuple, Callable

from jumanji.env import Environment
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.types import (
    ActingState,
    ActorCriticParams,
    ParamsState,
    TrainingState,
    Transition,
)

class PPOAgent(A2CAgent):
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
            l_pg: float,
            l_td: float,
            l_en: float,
            clip_epsilon: float,
            num_minibatches: int,
            ppo_epochs: int,
            bootstrapping_factor: float, # gae_lambda
            max_grad_norm: float,
    ) -> None:
        super().__init__(
            env=env,
            n_steps=n_steps,
            total_batch_size=total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=normalize_advantage,
            discount_factor=discount_factor,
            bootstrapping_factor=bootstrapping_factor,
            l_pg=l_pg,
            l_td=l_td,
            l_en=l_en,
            )
        self.observation_spec = env.observation_spec
        self.clip_epsilon = clip_epsilon
        self.num_minibatches = num_minibatches
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm

        # Compile rollout once (self is static)
       #self.rollout = jax.jit(self.rollout, static_argnums=(0,))

        # Compile run_epoch end-to-end
        #self.run_epoch = jax.jit(self.run_epoch, static_argnums=(0,))

    # Same as vanilla
    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        actor_key, critic_key = jax.random.split(key)
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, ...], self.observation_spec.generate_value()
        )
        params = ActorCriticParams(
            actor=self.actor_critic_networks.policy_network.init(actor_key, dummy_obs),
            critic=self.actor_critic_networks.value_network.init(critic_key, dummy_obs),
        )
        opt_state = self.optimizer.init(params)
        params_state = ParamsState(
            params=params,
            opt_state=opt_state,
            update_count=jnp.array(0, float),
        )
        return params_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, ParamsState):
            raise TypeError(
                "Expected params_state to be of type ParamsState, got "
                f"type {type(training_state.params_state)}."
            )
        
        acting_state, data = self.rollout(
            training_state.params_state.params.actor,
            training_state.acting_state,
        )
        
        advantages, returns = self.compute_gae(
            training_state.params_state.params,
            data
        )

        def run_ppo_epoch(carry, _):
            current_params, current_opt_state, metrics_acc = carry

            grad, metrics = jax.grad(self.ppo_loss, has_aux=True)(
                current_params,
                training_state.params_state.params, # the old policy
                data,
                advantages,
                returns,
                training_state.acting_state.key,
            )
            grad, metrics = jax.lax.pmean((grad, metrics), axis_name="devices")
            updates, new_opt_state = self.optimizer.update(grad, current_opt_state)
            new_params = optax.apply_updates(current_params, updates)
            new_metrics_acc = jax.tree_util.tree_map(lambda a, b: a+b, metrics_acc, metrics)

            return (new_params, new_opt_state, new_metrics_acc), None
        
        metrics_shape = self.get_metrics_shape(data, advantages)
        empty_metrics = jax.tree_util.tree_map(lambda _: jnp.zeros([]), metrics_shape)

        init_carry = (training_state.params_state.params, training_state.params_state.opt_state, empty_metrics)
        (final_params, final_opt_state, accumulated_metrics), _ = jax.lax.scan(
            run_ppo_epoch,
            init_carry,
            None,
            length=self.ppo_epochs
        )

        metrics = jax.tree_util.tree_map(lambda x: x / self.ppo_epochs, accumulated_metrics)
        if data.extras:
            metrics.update(data.extras)
        
        training_state = TrainingState(
            params_state=ParamsState(
                params=final_params,
                opt_state=final_opt_state,
                update_count=training_state.params_state.update_count + 1,
            ),
            acting_state=acting_state,
        )

        return training_state, metrics
    
    def get_metrics_shape(self, data, advantages):
        """Helper to create empty metrics structure with correct shapes."""
        return {
            "total_loss": jnp.array(0.0),
            "policy_loss": jnp.array(0.0),
            "value_loss": jnp.array(0.0),
            "entropy_loss": jnp.array(0.0),
            "entropy": jnp.array(0.0),
            "advantage": jnp.array(0.0),
            "clip_fraction": jnp.array(0.0),
        }
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _jitted_compute_gae(self, params: ActorCriticParams, data: Transition) -> Tuple[chex.Array, chex.Array]:
        """Compute gae and returns."""
        value_apply = self.actor_critic_networks.value_network.apply
        
        last_observation = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        observation = jax.tree_util.tree_map(
            lambda obs_0_tm1, obs_t: jnp.concatenate([obs_0_tm1, obs_t[None]], axis=0),
            data.observation,
            last_observation,
        )
        
        def apply_value_fn(obs):   
            if isinstance(params.critic, tuple) and len(params.critic) == 2:
                critic_params, critic_state = params.critic
                value, _ = value_apply(critic_params, critic_state, obs)
                return value            
            else:
                value = value_apply(params.critic, obs)
        
        value = jax.vmap(apply_value_fn)(observation)
        discounts = jnp.asarray(self.discount_factor * data.discount, float)
        value_tm1 = value[:-1]
        value_t = value[1:]
        
        advantages = jax.vmap(
            functools.partial(
                rlax.td_lambda,
                lambda_=self.bootstrapping_factor,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )(
            value_tm1,
            data.reward,
            discounts,
            value_t,
        )
        
        returns = advantages + value_tm1
        
        return advantages, returns
    
    def compute_gae(self,params,data):
        return self._jitted_compute_gae(params, data)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _jitted_ppo_loss(
            self,
            params: ActorCriticParams,
            old_params: ActorCriticParams,
            data: Transition,
            advantages: chex.Array,
            returns: chex.Array,
            key: chex.PRNGKey,
    ) -> Tuple[float, Dict]:
        """ PPO loss function """
        parametric_action_distribution = self.actor_critic_networks.parametric_action_distribution
        policy_network = self.actor_critic_networks.policy_network
        value_network = self.actor_critic_networks.value_network

        # need to flatten the actions to match logits
        # this is only for tetris: flattened_index = column * num_rotations + rotation
        num_rotations = 4  # Based on your environment
        flattened_actions = data.action[..., 0] * num_rotations + data.action[..., 1]
        
        # Get current logits
        def apply_policy_fn(obs):
            if isinstance(params.actor, tuple) and len(params.actor) == 2:
                actor_params, actor_state = params.actor
                current_logits, _ = policy_network.apply(actor_params, actor_state, obs)
                return current_logits
            else:
                current_logits = policy_network.apply(params.actor, obs)
        
        current_logits = jax.vmap(apply_policy_fn)(data.observation)
        # Calculate log probs using flattened actions
        current_log_probs = jax.vmap(self.actor_critic_networks.parametric_action_distribution.log_prob)(
            current_logits, flattened_actions
        )

        old_log_probs = data.log_prob
        ratios = jnp.exp(current_log_probs - old_log_probs)

        surrogate1 = jnp.multiply(ratios, advantages)
        surrogate2 = jnp.multiply(jnp.clip(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon), advantages)
        policy_loss = -jnp.mean(jnp.minimum(surrogate1,surrogate2))

        def apply_value_fn(obs):
            if isinstance(params.critic, tuple) and len(params.critic) == 2:
                critic_params, critic_state = params.critic
                value, _ = value_network.apply(critic_params, critic_state, obs)
                return value
            else:
                current_values = value_network.apply(params.critic, obs)
                return current_values
        
        current_values = jax.vmap(apply_value_fn)(data.observation)
        value_loss = jnp.mean(jnp.square(returns - current_values))

        entropy = jnp.mean(
            parametric_action_distribution.entropy(current_logits, key)
        )
        entropy_loss = -entropy

        total_loss = policy_loss + self.l_td * value_loss + self.l_en * entropy_loss

        clip_fraction = jnp.mean(
            jnp.logical_or(
                ratios < 1.0 - self.clip_epsilon,
                ratios > 1.0 + self.clip_epsilon
            ).astype(jnp.float32)
        )

        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "advantage": jnp.mean(advantages),
            "clip_fraction": clip_fraction,
        }

        return total_loss, metrics
    
    def ppo_loss(self, params, old_params, data, advantages, returns, key):
        return self._jitted_ppo_loss(params,old_params,data,advantages,returns,key)
    
    def make_policy(
            self,
            policy_params: hk.Params,
            stochastic:bool = True,
    ) -> Callable[[Any, chex.PRNGKey], Tuple[chex.Array, Tuple[chex.Array, chex.Array]]]:
        policy_network = self.actor_critic_networks.policy_network
        parametric_action_distribution = self.actor_critic_networks.parametric_action_distribution

        @jax.jit
        def policy(
                observation: Any, key: chex.PRNGKey
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
            if isinstance(policy_params, tuple) and len(policy_params) == 2:
                params, state = policy_params
                logits, _ = policy_network.apply(params, state, observation)
            else:
                logits = policy_network.apply(policy_params, observation)

            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(logits,key)
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                del key
                raw_action = parametric_action_distribution.mode_no_postprocessing(logits)
                # log_prob is log(1), i.e. 0, for a greedy policy (deterministic distribution)
                # don't think we'll need to do a greedy eval, but why not leave it in
                log_prob = jnp.zeros_like(
                    parametric_action_distribution.log_prob(logits, raw_action)
                )
            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits)
        
        return policy
    @functools.partial(jax.jit, static_argnums=(0,))
    def rollout(
            self,
            policy_params: hk.Params,
            acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        policy = self.make_policy(policy_params=policy_params, stochastic=True)

        def run_one_step(
                acting_state: ActingState, key:chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, (log_prob, logits) = policy(timestep.observation, key)
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count
                + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count
                + jax.lax.psum(self.batch_size_per_device, "devices"), 
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition
        
        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape((self.n_steps, -1))
        acting_state, data = jax.lax.scan(run_one_step, acting_state, acting_keys)
        return acting_state, data