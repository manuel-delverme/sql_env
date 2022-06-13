import numpy as np
import stable_baselines3.common
import torch
import config


class CallBack(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super(CallBack, self).__init__()
        # self.last_rollout_end = 0
        # self.last_on_step = 0

    def _on_step(self):
        # if (self.num_timesteps - self.last_on_step) <= config.log_iterate_every:
        #     return
        # self.last_on_step = self.num_timesteps

        # option_net = self.locals["self"].policy
        # obs_tensor = self.locals["obs_tensor"]

        # features = option_net.terminations.extract_features(obs_tensor)
        # termination_probs = torch.concat([
        #     ot(features).cpu().detach() for ot in option_net.terminations.option_terminations[:self.model.policy.meta_policy.initialization.available_options]
        # ], dim=1)
        # task_idx_mask = self.task_idx_mask.squeeze()
        # for task_idx in task_idx_mask.unique():
        #     self.logger.add_scalar(f"term_probs_mean/{task.names[int(task_idx)]}", termination_probs[task_idx_mask == task_idx].mean(), self.num_timesteps)

        super(CallBack, self)._on_step()

    def _on_rollout_end(self):
        super(CallBack, self)._on_rollout_end()
        # if (self.num_timesteps - self.last_rollout_end) <= config.log_iterate_every:
        #     return
        # self.last_rollout_end = self.num_timesteps

        # rollout_steps = self.locals["self"].n_steps
        # task_ids = self.task_idx_mask.unique()

        # executed_options = rollout_buffer.current_options
        # previous_options = rollout_buffer.previous_options

        # switches = executed_options != previous_options
        # switches[rollout_buffer.episode_starts.astype(bool)] = False

        # env_returns = rollout_buffer.returns.mean(0)
        # meta_log_probs = rollout_buffer.option_log_probs
        # task_returns = env_returns.reshape(len(task_ids), -1).mean(1)

        # task_idx_mask = self.task_idx_mask.repeat(rollout_steps, 1)

        # if self.model.ep_success_buffer:
        #     self.logger.add_scalar("rollout/sum_returns", sum(self.model.ep_success_buffer), self.num_timesteps)

        # self.logger.add_scalar("rollout_/switches", switches.sum(), self.num_timesteps)

        # self.logger.add_scalar("debug/mean_rewards", rollout_buffer.rewards.mean(), self.num_timesteps)
        # self.logger.add_scalar("debug/max_rewards", rollout_buffer.rewards.sum(0).max(), self.num_timesteps)
        # self.logger.add_scalar("rollout/change_points", switches.mean(), self.num_timesteps)
        # self.logger.add_scalar("rollout/action_gap", np.inf, self.num_timesteps)

        # if hasattr(self.model, "current_weights"):
        #     weights = torch.zeros(config.num_tasks)
        #     weighted_return = 0.
        #     for task_idx in range(config.num_tasks):
        #         task_name = task.names[int(task_idx)]
        #         task_mask = task_idx == self.task_idx_mask
        #         weights[task_idx] = self.model.current_weights[task_mask[0]].mean()
        #         self.logger.add_scalar(f"task_weight/task{task_name}", weights[task_idx], self.num_timesteps)
        #         weighted_return += task_returns[task_idx] * weights[task_idx]

        #     self.logger.add_scalar("rollout/weighted_mean_returns", weighted_return, self.num_timesteps)
        #     if config.num_tasks > 2:
        #         self.logger.add_histogram("boosting_/weights", weights, self.num_timesteps)

        # # self.logger.add_histogram("rollout/executed_options", executed_options.flatten(), self.num_timesteps) # Wandb is too slow

        # self.logger.add_scalar("boosting/num_bad_epochs", self.model.scheduler.num_bad_epochs, self.num_timesteps)
        # self.logger.add_scalar("boosting/best", self.model.scheduler.best, self.num_timesteps)
        # self.logger.add_scalar("boosting/cooldown", self.model.scheduler.cooldown_counter, self.num_timesteps)

        # for task_return, task_idx in zip(task_returns, task_ids):
        #     task_name = task.names[int(task_idx)]
        #     task_executed_option = executed_options[task_idx_mask == task_idx]
        #     task_meta_log_prob = meta_log_probs[task_idx_mask == task_idx]

        #     if len(task_executed_option.shape) > 1 and task_executed_option.shape[1] != 1:
        #         task_executed_option = task_executed_option.mean(0)

        #     if len(task_executed_option) < 3:
        #         task_executed_option = np.tile(task_executed_option, 3)[:3]
        #     self.logger.add_histogram(f"task{task_name}/task_executed_options", task_executed_option, self.num_timesteps)

        #     for i in range(self.model.policy.meta_policy.initialization.available_options):
        #         uniform_prob = 1 / self.model.policy.meta_policy.initialization.available_options
        #         option_prob = np.exp(task_meta_log_prob[task_executed_option == i]).mean()
        #         if option_prob != option_prob:
        #             option_prob = 0
        #         self.logger.add_scalar(f"option_{i}/task{task_name}_mean_prob", option_prob - uniform_prob, self.num_timesteps)

        #     if hasattr(rollout_buffer, "priority"):
        #         self.logger.add_scalar(f"task{task_name}/priority", rollout_buffer.priority[task_idx], self.num_timesteps)

        #     self.logger.add_scalar(f"task{task_name}/mean_return", task_return, self.num_timesteps)
