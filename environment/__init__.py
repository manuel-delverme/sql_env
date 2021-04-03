import gym.envs

import environment.sql_env

gym.envs.register(
    id='SQL-v0',
    entry_point='environment.sql_env:SQLEnv',
    max_episode_steps=10,
    kwargs={'html': False}
)
