import pickle

import torch
import yaml

import config
import environment  # noqa
import environment.sql_env
import ppo.envs
import ppo.model
from lstmDQN.custom_agent import CustomAgent


# from textworld import EnvInfos

def train():
    config_file_name = "lstmDQN/config.yaml"
    env = ppo.envs.make_vec_envs(
        config.env_name, config.seed, config.num_processes, config.gamma, config.log_dir, config.device, False)
    env.action_space = environment.sql_env.TextSpace(
        ppo.model.get_output_vocab(), env.action_space.sequence_length)  # This is a hack but the experiment defines it's own action space

    agent = CustomAgent(config_file_name, env.observation_space, env.action_space)
    reward_ = 0

    with open(config_file_name) as reader:
        config_ = yaml.safe_load(reader)
    full_stats = {}
    for epoch_no in range(1, agent.nb_epochs + 1):
        stats = {
            "rewards": [],
            "steps": [],
        }
        obs_token = agent.model.env_encode(env.reset())
        infos = {}
        agent.train()
        rewards = [0] * len(obs_token)
        dones = [False] * len(obs_token)
        steps = [0] * len(obs_token)

        while not all(dones):
            actions = agent.act(obs_token, rewards, dones, infos)
            queries = idx_to_str(agent, actions)

            if agent.current_step > 0 and agent.current_step % agent.update_per_k_game_steps == 0:
                loss = agent.update()
                if loss is not None:
                    agent.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                    agent.optimizer.step()
                    print(loss, reward_)

            agent.current_step += 1
            next_obs, rewards, dones, infos = env.step(queries)
            reward_ = reward_ * 0.9 + rewards * 0.1

            next_obs_token = agent.model.env_encode(next_obs)
            del next_obs

            if agent.current_step > 0:
                agent.replay_memory.add(obs_token, next_obs_token, actions, rewards, dones, infos)

            obs_token = next_obs_token

        agent.act(obs_token, rewards, dones, infos)
        agent.current_episode += 1
        if agent.current_episode < agent.epsilon_anneal_episodes:
            agent.epsilon -= (agent.epsilon_anneal_from - agent.epsilon_anneal_to) / float(agent.epsilon_anneal_episodes)

        stats["rewards"].extend(rewards)
        stats["steps"].extend(steps)


        score = sum(stats["rewards"])
        steps = sum(stats["steps"])
        print(f"Epoch: {epoch_no:3d} | {score.item():2.1f} pts | {steps:4.1f} steps")


def idx_to_str(agent, actions):
    chosen_strings = []
    for query_idx in actions:
        query_tokens = [agent.action_vocab[idx] for idx in query_idx]
        chosen_strings.append(query_tokens)

    queries = ["".join(query) for query in chosen_strings]
    return queries


if __name__ == '__main__':
    train()
