import typing

import torch
import torch.nn as nn
from torch.nn import functional as F


class Transition(typing.NamedTuple):
    state: torch.tensor
    action: torch.tensor
    next_state: torch.tensor
    reward: torch.tensor
    done: torch.tensor


class Output(typing.NamedTuple):
    action: torch.tensor
    policy_logits: torch.tensor
    value: torch.tensor


class Critic(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )


class MLPBase(nn.Module):
    def __init__(self, num_inputs, dictionary_size, hidden_size=64):
        super().__init__()
        self.body = Body(num_inputs, dictionary_size)
        self.actor = AutoregressiveActor(hidden_size, hidden_size, dictionary_size)
        self.critic = Critic(hidden_size)
        self.train()

    def forward(self, inputs):
        x = inputs.state
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        one_hot_last_action = F.one_hot(inputs.action.view(T * B), self.num_actions).float()
        core_output = torch.cat([x, inputs.reward, one_hot_last_action], dim=-1)

        hidden = self.body(core_output)
        action, policy_logits = self.actor(hidden)
        value = self.value(hidden).view(T, B)

        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1).view(T, B)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        return Output(action, policy_logits, value)


class Body(nn.Module):
    def __init__(self, num_inputs, dictionary_size):
        super(Body, self).__init__()
        self.gru = nn.GRU(num_inputs, dictionary_size)
        self.end_of_line = dictionary_size

    def forward(self, inputs):
        collect = []
        for x in inputs:
            assert x.ndim == 2
            _, rnn_hxs = self.gru(x.unsqueeze(1), None)  # (seq_len, batch, input_size)
            collect.append(rnn_hxs.squeeze(0))
        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension
        # TODO fix this with logprop
        rnn_hxs = torch.cat(collect)
        return rnn_hxs


class AutoregressiveActor(nn.Module):
    def __init__(self, num_inputs, hidden_size, dictionary_size):
        super().__init__()
        self.hidden_to_hidden = nn.Linear(num_inputs, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, dictionary_size)

    def forward(self, hidden, seq_len: int):
        query = []
        query_logprobs = []

        for _ in range(seq_len):
            word_logprobs, rnn_hxs = self._one_step(hidden)
            word = torch.distributions.Categorical(logits=word_logprobs).sample()
            query.append(word)
            query_logprobs.append(word_logprobs)
        # b x t x k
        query_logprobs = torch.stack(query_logprobs, dim=1)
        query = torch.stack(query, dim=1).unsqueeze(-1)
        assert query.shape[:2] == query_logprobs.shape[:2]
        return query, query_logprobs

    def one_step(self, hidden):
        next_hidden = self.hidden_to_hidden(hidden)
        output = self.hidden_to_output(hidden)
        output_prob = torch.log_softmax(output, 1)

        return output_prob, next_hidden
