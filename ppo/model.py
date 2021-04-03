import numpy as np
import torch
import torch.nn as nn

import ppo.distributions
from ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()
        self.base = ppo.model.MLPBase(obs_shape[-1])

        num_outputs = action_space.nvec
        self.dist = ppo.distributions.Multinomial(self.base.output_size, num_outputs, )

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # action = action.squeeze()
        # dist_entropy = dist.entropy().mean()
        action = action.argmax(-1)
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action = action.unsqueeze(0)
        shifted_idx = list(range(action.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        action = action.permute(*shifted_idx)
        sample_shape = dist._batch_shape + dist._event_shape
        one_hot_actions = action.new(sample_shape).zero_()
        one_hot_actions.scatter_add_(-1, action, torch.ones_like(action))

        action_log_probs = dist.log_probs(one_hot_actions)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        self.gru = nn.GRU(recurrent_input_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = inputs
        assert x.shape[1:] == (4, 10)
        _x, rnn_hxs = self.gru(x.transpose(0, 1), None)
        rnn_hxs = rnn_hxs.squeeze(0)

        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension
        hidden_critic = self.critic(rnn_hxs)
        hidden_actor = self.actor(rnn_hxs)

        return self.critic_linear(hidden_critic), hidden_actor
