import os
import time
from collections import deque

import numpy as np
import torch

import config
import environment  # noqa
import ppo.model
from ppo import utils
from ppo.envs import make_vec_envs
from ppo.evaluation import evaluate
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

    actor_critic = ppo.model.Policy(envs.observation_space.shape, envs.action_space.vocab).to(config.device)

    agent = ppo.PPO(
        actor_critic, config.clip_param, config.ppo_epoch, config.num_mini_batch, config.value_loss_coef, config.entropy_coef, lr=config.lr, eps=config.eps,
        max_grad_norm=config.max_grad_norm)

    rollouts = RolloutStorage(config.num_steps, config.num_processes, envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0] = obs.copy()
    rollouts.to(config.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(config.num_env_steps) // config.num_steps // config.num_processes

    for network_updates in range(num_updates):
        for step in range(config.num_steps):
            with torch.no_grad():
                value, batch_queries, action_log_prob = actor_critic.act(rollouts.obs[step])

            obs, reward, done, infos = envs.step([" ".join(query) for query in batch_queries])

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.tensor(1 - done, dtype=torch.float32)

            rollouts.insert(obs, batch_queries, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, config.use_gae, config.gamma, config.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if network_updates % config.save_interval == 0 or network_updates == num_updates - 1:
            save_path = config.save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)], save_path + ".pt")

        if network_updates % config.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (network_updates + 1) * config.num_processes * config.num_steps
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                  .format(network_updates, total_num_steps, int(total_num_steps / (end - start)), len(episode_rewards), np.mean(episode_rewards),
                          np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards), dist_entropy, value_loss, action_loss))

        if config.eval_interval is not None and len(episode_rewards) > 1 and network_updates % config.eval_interval == 0:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, config.env_name, config.seed, config.num_processes, eval_log_dir, config.device)


if __name__ == "__main__":
    main()
