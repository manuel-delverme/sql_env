import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from stable_baselines3.common.buffers import ReplayBuffer

import config
from lstmDQN.model import LSTM_DQN


class CustomAgent:
    def __init__(self, config_file_name, observation_space, action_space):
        self.mode = "train"
        self.env_vocab = observation_space.vocab
        self.action_vocab = action_space.vocab

        with open(config_file_name) as reader:
            self.config = yaml.safe_load(reader)

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']

        # Set the random seed manually for reproducibility.
        np.random.seed(self.config['general']['random_seed'])
        torch.manual_seed(self.config['general']['random_seed'])
        self.device = config.device
        self.model = LSTM_DQN(self.action_vocab, self.env_vocab, self.device, action_space.sequence_length)

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.model_checkpoint_path = self.config['checkpoint']['model_checkpoint_path']
        self.save_frequency = self.config['checkpoint']['save_frequency']

        self.model.to(config.device)

        self.replay_batch_size = self.config['general']['replay_batch_size']
        self.replay_memory = ReplayBuffer(
            config.buffer_size,
            gym.spaces.MultiDiscrete(np.ones(observation_space.sequence_length) * observation_space.vocab_length),
            gym.spaces.MultiDiscrete(np.ones(action_space.sequence_length) * action_space.vocab_length)
        )

        # optimizer
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['training']['optimizer']['learning_rate'])

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['general']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['general']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['general']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.update_per_k_game_steps = self.config['general']['update_per_k_game_steps']
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']

        self.discount_gamma = self.config['general']['discount_gamma']
        self.current_episode = 0
        self._epsiode_has_started = False
        self.best_avg_score_so_far = 0.0

    def train(self):
        self.mode = "train"
        self.model.train()

    def eval(self):
        self.mode = "eval"
        self.model.eval()

    def get_Q(self, token_idx):
        state_representation = self.model.representation_generator(token_idx)
        return self.model.get_Q(state_representation)  # each element in list has batch x n_vocab size

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        sequence_Q = self.get_Q(obs)  # list of batch x vocab

        # random number for epsilon greedy
        actions = sequence_Q.max(dim=2).indices
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(obs.shape[0]))
        rand_idx = torch.randint(0, self.model.agent_vocab_size, size=actions.shape)
        actions[rand_num < self.epsilon, :] = rand_idx[rand_num < self.epsilon, :]
        return actions

    def update(self):
        if not self.replay_memory.full and self.replay_memory.pos < self.replay_batch_size:
            return None
        batch = self.replay_memory.sample(self.replay_batch_size)

        server_response_Q = self.get_Q(batch.observations.unsqueeze(-1))  # list of batch x input_length x num_vocab
        word_qvalues = []
        for ith_word_Q, idx in zip(server_response_Q, batch.actions):
            Qw = ith_word_Q.gather(1, idx.unsqueeze(1))
            word_qvalues.append(Qw)
        q_value = torch.stack(word_qvalues, 0).mean(1)  # Mean across actions.

        next_server_reponse_Q = self.get_Q(batch.next_observations.unsqueeze(-1))

        next_word_qvalues = next_server_reponse_Q.max(dim=-1).values
        next_q_value = torch.mean(next_word_qvalues, 1)

        next_q_value = next_q_value.detach().unsqueeze(-1)

        rewards = batch.rewards
        not_done = 1. - batch.dones
        values = rewards + not_done * next_q_value * self.discount_gamma  # batch

        mask = torch.ones_like(values)  # Not used yet.
        loss = F.smooth_l1_loss(q_value * mask, values * mask)
        return loss
