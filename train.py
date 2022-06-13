import experiment_buddy
import numpy as np
import stable_baselines3
import stable_baselines3.common.buffers
import stable_baselines3.common.callbacks
import stable_baselines3.common.vec_env.stacked_observations
import torch

import config
import environment  # noqa
import environment.sql_env
import environment.wrappers

# Set the random seed manually for reproducibility.
np.random.seed(config.seed)
torch.manual_seed(config.seed)


def train(tb):
    env = stable_baselines3.common.vec_env.DummyVecEnv([environment.sql_env.SQLEnv])
    env = stable_baselines3.common.vec_env.VecFrameStack(env, 3)
    model = stable_baselines3.DQN(
        "MlpPolicy",
        env,
        verbose=2,
        learning_starts=100,
        gradient_steps=5,
        batch_size=2048,
    )
    model.learn(
        total_timesteps=1_000_000,
        log_interval=100,
    )
    # callback = stable_baselines3.common.callbacks.BaseCallback()

    #     tb.add_scalar('train/epsilon', agent.epsilon, env_steps)
    #     tb.add_scalar('train/episode_reward', episode_reward, env_steps)
    #     tb.add_scalar('train/episode_length', episode_length, env_steps)
    #     tb.add_scalar('train/episode_loss', episode_loss / episode_length, env_steps)
    #     tb.add_scalar('train/avg_episode_length', np.mean(avg_task_performance), env_steps)

    #     tb.add_scalar(f'train/task_{sql_env.hidden_parameter}_{sql_env.selected_columns}', np.mean(task_performance[sql_env.hidden_parameter, sql_env.selected_columns]), env_steps)


if __name__ == '__main__':
    experiment_buddy.register_defaults(vars(config))
    PROC_NUM = 1  # 10
    # HOST = "mila" if config.user == "esac" else ""
    HOST = ""
    RUN_SWEEP = False
    # tb = experiment_buddy.deploy(host=HOST, sweep_definition="sweep.yml" if RUN_SWEEP else "", proc_num=PROC_NUM,
    #                              wandb_kwargs={"mode": "disabled" if config.DEBUG else "online", "entity": "rl-sql"})
    tb = experiment_buddy.deploy(host=HOST, sweep_definition="sweep.yml" if RUN_SWEEP else "", proc_num=PROC_NUM,
                                 wandb_kwargs={"mode": "disabled" if config.DEBUG else "online", "entity": "rl-sql"})
    train(tb)
