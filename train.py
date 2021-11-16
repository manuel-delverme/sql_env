import collections

import torch
import tqdm

import config
import environment  # noqa
import environment.sql_env
import experiment_buddy
import ppo.envs
import ppo.model
from lstmDQN.custom_agent import CustomAgent


def train():
    config_file_name = "lstmDQN/config.yaml"
    env = ppo.envs.make_vec_envs(config.env_name, config.seed, config.num_processes, config.gamma, config.log_dir, config.device, False)

    # This is a hack but the experiment defines it's own action space
    env.action_space = environment.sql_env.TextSpace(ppo.model.get_output_vocab(), env.action_space.sequence_length, (1, env.action_space.sequence_length))
    obs_len = env.observation_space.sequence_length + env.action_space.sequence_length * config.action_hist
    env.observation_space = environment.sql_env.TextSpace(env.observation_space.vocab + env.action_space.vocab, obs_len, (1, obs_len))

    agent = CustomAgent(config_file_name, env.observation_space, env.action_space)
    agent.train()

    total_steps = 0
    steps_ = 0
    prev_action = collections.deque([torch.zeros(env.action_space.shape, dtype=torch.int) for _ in range(config.action_hist)], maxlen=config.action_hist)

    pbar = tqdm.tqdm(total=config.num_steps)
    while total_steps < config.num_env_steps:
        pbar.update(1)
        obs = env.reset()
        obs_token = agent.model.env_encode(obs)
        done = False
        episode_length = 0
        episode_reward = 0
        agent.current_step = 0

        hist_token = torch.cat((obs_token, *prev_action), dim=1)
        while not done:
            actions = agent.act(hist_token.unsqueeze(-1))
            queries = idx_to_str(agent, actions)

            loss = agent.update()
            if loss is not None:
                agent.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                agent.optimizer.step()

            agent.current_step += 1
            total_steps += 1

            next_obs, rewards, dones, infos = env.step(queries)
            done, = dones

            episode_length += 1
            episode_reward += rewards

            next_obs_token = agent.model.env_encode(next_obs)
            del next_obs
            prev_action.append(actions)
            next_hist_token = torch.cat((next_obs_token, *prev_action), dim=1)

            agent.replay_memory.add(hist_token, next_hist_token, actions, rewards, dones, infos)
            hist_token = next_hist_token

        agent.current_episode += 1
        if agent.current_episode < agent.epsilon_anneal_episodes:
            agent.epsilon -= (agent.epsilon_anneal_from - agent.epsilon_anneal_to) / float(agent.epsilon_anneal_episodes)
        if steps_ is None:
            steps_ = episode_length
        steps_ = 0.9 * steps_ + 0.1 * episode_length

        pbar.set_description(f"{episode_reward.item():2.1f} pts | {episode_length:4.1f}({steps_:.1f}) steps")


def idx_to_str(agent, actions):
    chosen_strings = []
    for query_idx in actions:
        query_tokens = [agent.action_vocab[idx] for idx in query_idx]
        chosen_strings.append(query_tokens)

    queries = ["".join(query) for query in chosen_strings]
    return queries


if __name__ == '__main__':
    experiment_buddy.register_defaults(vars(config))
    PROC_NUM = 1
    # HOST = "mila" if config.user == "esac" else ""
    HOST = ""
    YAML_FILE = ""  # "env_suite.yml"
    tb = experiment_buddy.deploy(host=HOST, sweep_yaml=YAML_FILE, proc_num=PROC_NUM, wandb_kwargs={"mode": "disabled" if config.DEBUG else "online", "entity": "rl-sql"})
    train()
