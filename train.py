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
            # Increase step counts.
            actions = agent.act(obs_token, rewards, dones, infos)

            chosen_strings = []
            for query_idx in actions:
                query_tokens = [agent.action_vocab[idx] for idx in query_idx]
                chosen_strings.append(query_tokens)

            # update neural model by replaying snapshots in replay memory
            if agent.current_step > 0 and agent.current_step % agent.update_per_k_game_steps == 0:
                loss = agent.update()
                if loss is not None:
                    # Backpropagate
                    agent.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                    agent.optimizer.step()  # apply gradients

            agent.current_step += 1

            queries = ["".join(query) for query in chosen_strings]
            next_obs, rewards, dones, infos = env.step(queries)
            next_obs_token = agent.model.env_encode(next_obs)
            del next_obs

            if agent.current_step > 0:
                agent.replay_memory.add(obs_token, next_obs_token, actions, rewards, dones, infos)

            obs_token = next_obs_token
            # append next step
            # history = [x + [obs[i]] for i, x in enumerate(history)]
        # Let the agent knows the game is done.
        agent.act(obs_token, rewards, dones, infos)
        agent._end_episode(obs_token, rewards, infos)

        stats["rewards"].extend(rewards)
        stats["steps"].extend(steps)
        full_stats[epoch_no] = stats

        score = sum(stats["rewards"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print(f"Epoch: {epoch_no:3d} | {score.item():2.1f} pts | {steps:4.1f} steps")
    stats_file_name = config_file_name.split("/")
    stats_file_name[-2] = "stats"
    stats_file_name[-1] = stats_file_name[-1].strip("yaml") + "pickle"
    stats_file_name = "/".join(stats_file_name)
    with open(stats_file_name, "wb") as f:
        pickle.dump(full_stats, f)


if __name__ == '__main__':
    train()
