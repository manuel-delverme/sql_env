import os
import time
from collections import deque

import numpy as np
import torch
import tqdm

import config
import environment  # noqa
import ppo.model
import wandb
from ppo import utils
from ppo.envs import make_vec_envs
from ppo.storage import RolloutStorage


def main():
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    log_dir = os.path.expanduser(config.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(config.env_name, config.seed, config.num_processes, config.gamma, config.log_dir, config.device, False)

    actor_critic = ppo.model.Policy(envs.observation_space.shape, envs.action_space.vocab,
                                    envs.action_space.sequence_length, eps=config.eps).to(config.device)

    agent = ppo.PPO(
        actor_critic, config.clip_param, config.ppo_epoch, config.num_mini_batch, config.value_loss_coef,
        config.entropy_coef, lr=config.lr, eps=config.eps,
        max_grad_norm=config.max_grad_norm)

    rollouts = RolloutStorage(config.num_steps, config.num_processes, envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0] = obs.copy()
    rollouts.to(config.device)

    episode_rewards = deque(maxlen=10)
    episode_distances = deque()

    start = time.time()
    num_updates = int(config.num_env_steps) // config.num_steps // config.num_processes

    data = []

    for network_updates in tqdm.trange(num_updates):
        episode_distances.clear()
        running_logprobs = torch.zeros(envs.action_space.sequence_length, len(ppo.model.Policy.output_vocab))

        for rollout_step in range(config.num_steps):
            with torch.no_grad():
                value, batch_queries, action_log_prob = actor_critic.act(rollouts.obs[rollout_step])
            running_logprobs += action_log_prob[0]

            queries = ["".join(query) for query in batch_queries]
            obs, reward, done, infos = envs.step(queries)

            if network_updates % config.log_query_interval == 0 and network_updates:
                data.extend([[network_updates, rollout_step, q, float(r), str(o)] for q, r, o in zip(queries, reward, obs)])

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

                episode_distances.append(info['similarity'])

            # If done then clean the history of observations.
            masks = torch.tensor(1 - done, dtype=torch.float32)

            rollouts.insert(obs, batch_queries, action_log_prob, value, reward, masks)

        if network_updates % config.log_query_interval == 0 and network_updates:
            config.tb.run.log({"train_queries": wandb.Table(columns=["network_update", "rollout_step", "query", "reward", "observation"], data=data)})
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        next_value = torch.zeros_like(next_value)
        rollouts.compute_returns(next_value, config.use_gae, config.gamma, config.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if network_updates % config.save_interval == 0 or network_updates == num_updates - 1:
            config.tb.add_object("model", actor_critic, global_step=network_updates)

        if network_updates % config.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (network_updates + 1) * config.num_processes * config.num_steps

            end = time.time()
            action_logprob = (running_logprobs / config.num_steps).mean(0)
            config.tb.add_histogram("train/log_prob", action_logprob, global_step=network_updates)
            config.tb.add_histogram('train/log_prob_per_action', np.histogram(np.arange(action_logprob.shape[0]), weights=action_logprob), global_step=network_updates)
            config.tb.add_scalar("train/fps", int(total_num_steps / (end - start)), global_step=network_updates)
            config.tb.add_scalar("tran/avg_rw", np.mean(episode_rewards), global_step=network_updates)
            config.tb.add_scalar("tran/max_return", np.max(episode_rewards), global_step=network_updates)
            config.tb.add_scalar("train/entropy", dist_entropy, global_step=network_updates)
            config.tb.add_scalar("train/mean_distance", np.mean(episode_distances), global_step=network_updates)
            config.tb.add_scalar("train/value_loss", value_loss, global_step=network_updates)
            config.tb.add_scalar("train/action_loss", action_loss, global_step=network_updates)

            if np.mean(episode_rewards) == 1:
                return


if __name__ == "__main__":
    main()
