import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Policy, self).__init__()
        EMBEDDING_DIM = 10
        OBSERVATION_LEN = 20

        query_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", ""}
        query_vocab = sorted(query_vocab)
        output_vocab = sorted(action_shape)

        self.query_word_to_idx = {word: idx for idx, word in enumerate(query_vocab)}
        self.output_word_to_idx = {word: idx for idx, word in enumerate(output_vocab)}

        # self.observation_space = gym.Space((OBSERVATION_LEN, EMBEDDING_DIM), )
        # self.action_space = gym.spaces.Discrete(len(query_vocab))

        self.embeddings_in = nn.Embedding(len(query_vocab), EMBEDDING_DIM)
        self.embeddings_out = nn.Embedding(len(output_vocab), EMBEDDING_DIM)
        self.embeddings_in.weight.requires_grad = False
        self.embeddings_out.weight.requires_grad = False

        self.query_vocab = query_vocab
        self.output_vocab = output_vocab

        self.base = MLPBase(EMBEDDING_DIM, len(action_shape))

    def _decode(self, action):
        return " ".join([self.query_vocab[w] for w in action])

    def _encode(self, state):
        # words = words[:self.observation_space.shape[0]]
        retr = torch.tensor([self.output_word_to_idx[w] for w in state], dtype=torch.long)
        return retr

    def act(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        value, query, query_logprobs = self.base(embeds)
        query = [self.output_vocab[int(q)] for q in query]
        return value, np.array(query).reshape(1, 1, -1), torch.tensor(query_logprobs).reshape(1, 1, -1)

    def html_to_embedd(self, batch_response):
        word_idxes = []
        for response in batch_response:
            word_idxes.append([self.query_word_to_idx[word] for word in response])
        word_idxes = torch.tensor(word_idxes)
        embeds = self.embeddings_in(word_idxes)
        return embeds

    def get_value(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        value, _, _ = self.base(embeds)
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
    def __init__(self, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self.gru = nn.GRU(recurrent_input_size, hidden_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, dictionary_size, hidden_size=64):
        super(MLPBase, self).__init__(num_inputs, hidden_size)

        self.end_of_line = dictionary_size
        self.actor = AutoregressiveActor(hidden_size, hidden_size, dictionary_size)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.train()

    def forward(self, inputs):
        x = inputs
        hist = x.transpose(0, 1)

        _x, rnn_hxs = self.gru(hist, None)  # (seq_len, batch, input_size)
        rnn_hxs = rnn_hxs.squeeze(0)
        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension

        value = self.critic(rnn_hxs)
        query = []
        query_logprobs = []

        for _ in range(50):
            word_logprobs, rnn_hxs = self.actor(rnn_hxs)
            word = torch.argmax(word_logprobs)
            if word == 0.:
                break
            else:
                query.append(word)
                query_logprobs.append(word_logprobs[:, word])
        return value, query, query_logprobs


class AutoregressiveActor(nn.Module):
    def __init__(self, num_inputs, hidden_size, dictionary_size):
        super().__init__()
        self.hidden_to_hidden = nn.Linear(num_inputs, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, dictionary_size)

    def forward(self, hidden):
        next_hidden = self.hidden_to_hidden(hidden)
        output = self.hidden_to_output(hidden)
        output_prob = torch.log_softmax(output, 1)

        return output_prob, next_hidden
