import gym.envs

import environment.sql_env
import environment.sql_env2

gym.envs.register(
    id='SQL-v0',
    entry_point='environment.sql_env:SQLEnv',
    max_episode_steps=10,
    kwargs={'html': False}
)

gym.envs.register(
    id='SQL-v1',
    entry_point='environment.sql_env2:SQLEnv',
    max_episode_steps=10,
    kwargs={'html': False}
)
