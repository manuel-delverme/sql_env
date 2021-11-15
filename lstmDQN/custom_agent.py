from collections import namedtuple
from typing import List, Dict, Any

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from stable_baselines3.common.buffers import ReplayBuffer

import config
from lstmDQN.generic import to_np, to_pt
from lstmDQN.model import LSTM_DQN

# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('obs', 'reward', 'mask', 'done', 'next_obs'))


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
        self.model = LSTM_DQN(self.config["model"], self.action_vocab, self.env_vocab, self.device, action_space.sequence_length)

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.model_checkpoint_path = self.config['checkpoint']['model_checkpoint_path']
        self.save_frequency = self.config['checkpoint']['save_frequency']





        self.model.to(config.device)

        self.replay_batch_size = self.config['general']['replay_batch_size']
        self.replay_memory = ReplayBuffer(
            config.buffer_size,
            gym.spaces.MultiDiscrete(np.ones(observation_space.shape) * observation_space.vocab_length),
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
        self.current_step = 0
        self._epsiode_has_started = False
        self.best_avg_score_so_far = 0.0

    def train(self):
        self.mode = "train"
        self.model.train()

    def eval(self):
        self.mode = "eval"
        self.model.eval()

    def _start_episode(self, obs: List[str]) -> None:
        self.current_step = 0

    def get_chosen_strings(self, chosen_indices):
        """
        Turns list of word indices into actual command strings.

        Arguments:
            chosen_indices: Word indices chosen by model.
        """
        chosen_indices_np = [to_np(item)[:, 0] for item in chosen_indices]
        res_str = []
        batch_size = chosen_indices_np[0].shape[0]
        for i in range(batch_size):
            verb, adj, noun, adj_2, noun_2 = (chosen_indices_np[0][i], chosen_indices_np[1][i], chosen_indices_np[2][i], chosen_indices_np[3][i], chosen_indices_np[4][i])
            res_str.append(self.word_ids_to_commands(verb, adj, noun, adj_2, noun_2))
        return res_str

    def choose_random_command(self, word_ranks, word_masks_np):
        """
        Generate a command randomly, for epsilon greedy.

        Arguments:
            word_ranks: Q values for each word by model.get_Q.
            word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
        """
        batch_size = word_ranks[0].size(0)
        word_ranks_np = [to_np(item) for item in word_ranks]  # list of batch x n_vocab
        word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab
        word_indices = []
        for i in range(len(word_ranks_np)):
            indices = []
            for j in range(batch_size):
                msk = word_masks_np[i][j]  # vocab
                indices.append(np.random.choice(len(msk), p=msk / np.sum(msk, -1)))
            word_indices.append(np.array(indices))
        # word_indices: list of batch
        word_qvalues = [[] for _ in word_masks_np]
        for i in range(batch_size):
            for j in range(len(word_qvalues)):
                word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
        word_qvalues = [torch.stack(item) for item in word_qvalues]
        word_indices = [to_pt(item, self.use_cuda) for item in word_indices]
        word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of batch x 1
        return word_qvalues, word_indices

    def get_Q(self, token_idx):
        state_representation = self.model.representation_generator(token_idx)
        return self.model.get_Q(state_representation)  # each element in list has batch x n_vocab size

    def act_eval(self, obs: List[str]) -> List[str]:
        input_description = obs
        word_ranks = self.get_Q(input_description)  # list of batch x vocab
        _, word_indices_maxq = self.choose_maxQ_command(word_ranks)

        chosen_indices = word_indices_maxq
        chosen_indices = [item.detach() for item in chosen_indices]
        chosen_strings = self.get_chosen_strings(chosen_indices)
        self.current_step += 1

        return chosen_strings

    def act(self, obs: List[str]) -> List[str]:
        if self.mode == "eval":
            return self.act_eval(obs)

        sequence_Q = self.get_Q(obs)  # list of batch x vocab

        # random number for epsilon greedy
        actions = sequence_Q.max(dim=2).indices
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(obs.shape[0]))
        rand_idx = torch.randint(0, self.model.agent_vocab_size, size=actions.shape)
        actions[rand_num < self.epsilon, :] = rand_idx[rand_num < self.epsilon, :]
        return actions

    def compute_reward(self):
        """
        Compute rewards by agent. Note this is different from what the training/evaluation
        scripts do. Agent keeps track of scores and other game information for training purpose.

        """
        # mask = 1 if game is not finished or just finished at current step
        if len(self.dones) == 1:
            # it's not possible to finish a game at 0th step
            mask = [1.0 for _ in self.dones[-1]]
        else:
            assert len(self.dones) > 1
            mask = [1.0 if not self.dones[-2][i] else 0.0 for i in range(len(self.dones[-1]))]
        mask = np.array(mask, dtype='float32')
        mask_pt = to_pt(mask, self.use_cuda, type='float')
        # rewards returned by game engine are always accumulated value the
        # agent have recieved. so the reward it gets in the current game step
        # is the new value minus values at previous step.
        rewards = np.array(self.rewards[-1], dtype='float32')  # batch
        if len(self.rewards) > 1:
            prev_rewards = np.array(self.rewards[-2], dtype='float32')
            rewards = rewards - prev_rewards
        rewards_pt = to_pt(rewards, self.use_cuda, type='float')

        return rewards, rewards_pt, mask, mask_pt

    def update(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.

        """
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
