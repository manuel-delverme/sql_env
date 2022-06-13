import faulthandler

import gym.wrappers
import numpy as np
import stable_baselines3
import stable_baselines3.common.buffers
import stable_baselines3.common.callbacks
import stable_baselines3.common.vec_env.stacked_observations
import agents.dqn
import torch

import callbacks
import config
import environment  # noqa
import environment.sql_env
import environment.wrappers
import experiment_buddy

# Set the random seed manually for reproducibility.
np.random.seed(config.seed)
torch.manual_seed(config.seed)

faulthandler.enable()


def train(tb):
    # Logs will be saved in log_dir/monitor.csv
    env = stable_baselines3.common.vec_env.DummyVecEnv([
        lambda: gym.wrappers.RecordEpisodeStatistics(environment.sql_env.SQLEnv()) for _ in range(1_000)
    ])
    env = stable_baselines3.common.vec_env.VecFrameStack(env, 5)
    ts = 5_000_000

    agent = stable_baselines3.DQN(
        agents.dqn.DQNPolicy,
        env,
        verbose=2,
        device="cpu",
        learning_starts=100,
        gradient_steps=1,
        batch_size=2048,
        target_update_interval=1_000,
    )
    agent.set_logger(tb)
    cb = stable_baselines3.common.callbacks.CallbackList([
        # wandb.integration.sb3.WandbCallback(gradient_save_freq=hyper.slow_log_iterate_every),
        # option_baselines.common.callbacks.OptionRollout(envs, eval_freq=1 if hyper.DEBUG else hyper.video_every, n_eval_episodes=hyper.num_envs),
        callbacks.CallBack(),
    ])
    agent.learn(
        total_timesteps=ts,
        log_interval=min(ts // 10_000, 1),
        callback=cb,
    )

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
