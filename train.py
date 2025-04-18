import jax
import jumanji
from omegaconf import OmegaConf

env = jumanji.make('Tetris-v0')

# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)

state, timestep = jax.jit(env.reset)(key)

jit_step = jax.jit(env.step)
# (Optional) Render the env state

action = env.action_spec.generate_value()          # Action selection (dummy value here)
state, timestep = jit_step(state, action)   # Take a step and observe the next state and time step
print(timestep['observation'])

action = env.action_spec.generate_value()
state, timestep = jit_step(state, action)
print(timestep['observation'])