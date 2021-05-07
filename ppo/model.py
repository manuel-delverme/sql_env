import numpy as np
import torch
import torch.nn as nn
import re


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, output_vocab):
        super(Policy, self).__init__()
        EMBEDDING_DIM = 10
        OBSERVATION_LEN = 20

        base_query = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR"}
        query_vocab = sorted(output_vocab)
        output_vocab = sorted(base_query)

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

        self.base = MLPBase(EMBEDDING_DIM, len(output_vocab))

    def _decode(self, action):
        return " ".join([self.query_vocab[w] for w in action])

    def _encode(self, state):
        # words = words[:self.observation_space.shape[0]]
        retr = torch.tensor([self.output_word_to_idx[w] for w in state], dtype=torch.long)
        return retr

    def act(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        # extend in batch dimnesion (will go away with more envs)
        # embeds = embeds.unsqueeze(1)
        value, query, query_logprobs = self.base(embeds)
        b, t, _ = query.shape
        queries = []
        for n in range(b):
            q = list(map(lambda q: self.output_vocab[int(q)], query[n]))
            queries.append(q)
        ## query = [self.output_vocab[int(q)] for q in query]
        return value, np.array(queries), query_logprobs

    def html_to_embedd(self, batch_response):
        word_embeddings = []
        for response in batch_response:
            for content in response:
                if len(content) > 0:
                    content = content.strip().split()
                    sentence_idxs = []
                    for word in content:
                        sentence_idxs.append(self.query_word_to_idx[word])
                embeds = self.embeddings_in(torch.tensor(sentence_idxs))
                word_embeddings.append(embeds)
        assert len(word_embeddings) == len(batch_response)
        return word_embeddings

    def get_value(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        # exted in batch dimension as this is used to estimate the value at last state
        # remove me when multiple env
        # embeds = embeds.unsqueeze(1)
        value, _, _ = self.base(embeds)
        return value

    def evaluate_actions(self, batch_response, actions):
        embeds = self.html_to_embedd(batch_response)
        # here we extend in the time domain instead as we ware batching
        value, query, query_logprobs = self.base(embeds)
        parsed_actions = []
        for action in actions:
            for q in action:
                action_idx = self.output_word_to_idx[q]
                action_vector = torch.zeros(size=(query_logprobs.shape[-1],))
                action_vector[action_idx] = 1
                parsed_actions.append(action_vector)
        parsed_actions = torch.stack(parsed_actions, dim=0).reshape(query_logprobs.shape)

        return value, query_logprobs, parsed_actions


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
        # assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3
        # x = inputs
        collect = []
        for x in inputs:
            assert x.ndim == 2
            _, rnn_hxs = self.gru(x.unsqueeze(1), None)  # (seq_len, batch, input_size)
            collect.append(rnn_hxs.squeeze(0))
        rnn_hxs = torch.cat(collect)
        # rnn_hxs = rnn_hxs.squeeze(0)
        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension
        # TODO fix this with logprop
        value = self.critic(rnn_hxs)
        query = []
        query_logprobs = []

        for _ in range(4):
            word_logprobs, rnn_hxs = self.actor(rnn_hxs)
            # word = torch.argmax(word_logprobs)
            word = torch.distributions.Categorical(logits=word_logprobs).sample()
            query.append(word)
            query_logprobs.append(word_logprobs)
        # b x t x k
        query_logprobs = torch.stack(query_logprobs, dim=1)
        query = torch.stack(query, dim=1).unsqueeze(-1)
        assert query.shape[:2] == query_logprobs.shape[:2]
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
