import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

import config
from lstmDQN.model import LSTMDQN


class FixedLengthAgent:
    def __init__(self, observation_space, action_space, device):
        self.device = device

        self.action_vocab_size = len(action_space.vocab)

        self.model = LSTMDQN(len(observation_space.vocab), len(action_space.vocab), action_space.sequence_length, config.embedding_size, config.encoder_rnn_hidden_size,
                             config.action_scorer_hidden_dim, )

        # obs_vocab_size, action_vocab_size, device, output_length: int,
        # embedding_size, encoder_rnn_hidden_size, action_scorer_hidden_dim, ):
        self.model.to(self.device)

        self.replay_batch_size = config.replay_batch_size
        self.replay_memory = ReplayBuffer(
            config.buffer_size,
            gym.spaces.MultiDiscrete(np.ones(observation_space.sequence_length) * observation_space.vocab_length),
            gym.spaces.MultiDiscrete(np.ones(action_space.sequence_length) * action_space.vocab_length),
            device=self.device,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.epsilon_anneal_episodes = config.epsilon_anneal_episodes
        self.epsilon_anneal_from = config.epsilon_anneal_from
        self.epsilon_anneal_to = config.epsilon_anneal_to
        self.epsilon = self.epsilon_anneal_from

        self.clip_grad_norm = config.clip_grad_norm

    # def get_Q(self, token_idx):
    #     raise NotImplemented
    #     state_representation = self.model.representation_generator(token_idx)
    #     return self.model.get_Q(state_representation)  # each element in list has batch x n_vocab size

    def eps_greedy(self, token_batch: torch.Tensor) -> torch.Tensor:
        state_representation = self.model.representation_generator(token_batch)
        sequence_Q = self.model.output_qvalues(state_representation)  # list of batch x vocab

        # random number for epsilon greedy
        actions = sequence_Q.max(dim=2).indices
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(token_batch.shape[0]))
        rand_idx = torch.randint_like(actions, 0, self.action_vocab_size)
        actions[rand_num < self.epsilon, :] = rand_idx[rand_num < self.epsilon, :]
        return actions

    def update(self, discount):
        if not self.replay_memory.full and self.replay_memory.pos < self.replay_batch_size:
            return float("nan")

        batch = self.replay_memory.sample(self.replay_batch_size)

        current_repr = self.model.representation_generator(batch.observations.unsqueeze(-1))  # list of batch x input_length x num_vocab
        q_values = self.model.output_qvalues(current_repr)

        word_qvalues = []
        for ith_word_Q, idx in zip(q_values, batch.actions):
            Qw = ith_word_Q.gather(1, idx.unsqueeze(1))
            word_qvalues.append(Qw)

        q_value = torch.stack(word_qvalues, 0).mean(1)  # Mean across actions.

        with torch.no_grad():
            next_repr = self.model.representation_generator(batch.next_observations.unsqueeze(-1))
            next_server_reponse_Q = self.model.output_qvalues(next_repr)

            next_word_qvalues = next_server_reponse_Q.max(dim=-1).values
            next_q_value = torch.mean(next_word_qvalues, 1)
            next_q_value = next_q_value.detach().unsqueeze(-1)

        not_done = 1. - batch.dones
        values = batch.rewards + not_done * next_q_value * discount

        mask = torch.ones_like(values)  # Not used yet.
        loss = F.smooth_l1_loss(q_value * mask, values * mask)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return loss.item()
