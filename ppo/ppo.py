import torch
import torch.optim as optim


class PPO:
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr=None, eps=None, max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1]  # - rollouts.value_preds[:-1]
        # advantages = (advantages - advantages.min()) / (advantages.max() - advantages.min())
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample
                old_action_log_probs_batch = torch.tensor(old_action_log_probs_batch, device=value_preds_batch.device)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, parsed_actions = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                entropy =  - torch.einsum('btx,btx->bt',torch.exp(action_log_probs),  action_log_probs).mean()
                action_log_probs = torch.einsum("btx,btx->bt", action_log_probs, parsed_actions)
                old_action_log_probs_batch = torch.einsum("btx,btx->bt", old_action_log_probs_batch, parsed_actions)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                action_loss = - (ratio * adv_targ).mean()
                # surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                # action_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (action_loss - entropy).backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                dist_entropy_epoch += entropy.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
