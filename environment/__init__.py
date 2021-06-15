import gym.envs

import environment.sql_env
import environment.structured_sql_env

gym.envs.register(
    id='SQL-v1',
    entry_point='environment.sql_env:SQLEnv',
    max_episode_steps=10,
)

gym.envs.register(
    id='SQLstruct-v1',
    entry_point='environment.structured_sql_env:SQLEnvStructured',
    max_episode_steps=10,
)
