import numpy as np
import torch
import tqdm

import config
import environment  # noqa
import environment.sql_env
import environment.wrappers
import experiment_buddy
import ppo.envs
import ppo.model
from lstmDQN.custom_agent import FixedLengthAgent

# Set the random seed manually for reproducibility.
np.random.seed(config.general.random_seed)
torch.manual_seed(config.general.random_seed)


def train(tb):
    env = ppo.envs.make_vec_envs(config.env_name, config.seed, config.num_processes, config.gamma, config.log_dir, config.device, False)

    # This is a hack but the experiment defines it's own action space
    env.action_space = environment.sql_env.TextSpace(ppo.model.get_output_vocab(), env.action_space.sequence_length, (1, env.action_space.sequence_length))
    obs_len = env.observation_space.sequence_length + env.action_space.sequence_length * config.action_history_len
    env.observation_space = environment.sql_env.TextSpace(env.observation_space.vocab + env.action_space.vocab, obs_len, (1, obs_len))

    agent = FixedLengthAgent(env.observation_space, env.action_space, config.device)
    agent.model.train()
    processed_env = environment.wrappers.WordLevelPreprocessing(env, config.action_history_len, config.device)

    env_steps = 0

    pbar = tqdm.tqdm(total=config.num_env_steps)
    num_episodes = 0

    while env_steps < config.num_env_steps:
        pbar.update(1)

        obs = processed_env.reset()

        done = False
        episode_length = 0
        episode_reward = 0
        episode_loss = 0

        while not done:
            actions = agent.eps_greedy(obs.unsqueeze(-1))

            episode_loss += agent.update(config.gamma)

            # queries = processed_env.action_decode(actions)
            next_obs, rewards, dones, infos = processed_env.step(actions)
            done, = dones

            env_steps += 1
            episode_length += 1
            episode_reward += rewards

            agent.replay_memory.add(obs, next_obs, actions, rewards, dones, infos)
            obs = next_obs

        if num_episodes < agent.epsilon_anneal_episodes:
            agent.epsilon -= (agent.epsilon_anneal_from - agent.epsilon_anneal_to) / float(agent.epsilon_anneal_episodes)

        tb.add_scalar('train/epsilon', agent.epsilon, env_steps)
        tb.add_scalar('train/episode_reward', episode_reward, env_steps)
        tb.add_scalar('train/episode_length', episode_length, env_steps)
        tb.add_scalar('train/episode_loss', episode_loss / episode_length, env_steps)


if __name__ == '__main__':
    experiment_buddy.register_defaults(vars(config))
    PROC_NUM = 1
    # HOST = "mila" if config.user == "esac" else ""
    HOST = ""
    YAML_FILE = ""  # "env_suite.yml"
    tb = experiment_buddy.deploy(host=HOST, sweep_yaml=YAML_FILE, proc_num=PROC_NUM, wandb_kwargs={"mode": "disabled" if config.DEBUG else "online", "entity": "rl-sql"})
    train(tb)
