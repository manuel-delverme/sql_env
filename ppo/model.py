import string

import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_shape, output_vocab):
        super(Policy, self).__init__()
        EMBEDDING_DIM = 10

        action_vocab = set(string.ascii_lowercase + " " + string.digits + "'\"").union({" UNION SELECT ", " NULL ", " FROM "})
        query_vocab = sorted(output_vocab)
        output_vocab = sorted(action_vocab)

        self.query_word_to_idx = {word: idx for idx, word in enumerate(query_vocab)}
        self.output_token_to_idx = {word: idx for idx, word in enumerate(output_vocab)}

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
        retr = torch.tensor([self.output_token_to_idx[w] for w in state], dtype=torch.long)
        return retr

    def act(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        value, batch_query, query_logprobs = self.base(embeds)
        batch_size, query_length, _ = batch_query.shape
        queries = []
        for query_idx in batch_query:
            query_tokens = [self.output_vocab[idx] for idx in query_idx]
            queries.append(query_tokens)
        return value, np.array(queries), query_logprobs

    def html_to_embedd(self, batch_response):
        word_embeddings = []
        for response in batch_response:
            assert len(response) == 1
            for content in response:
                if not content:
                    sentence_idxs = [self.query_word_to_idx[""]]
                else:
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
        for token_sequence in actions:
            for token in token_sequence:
                action_idx = self.output_token_to_idx[token]
                action_vector = torch.zeros(size=(query_logprobs.shape[-1],))
                action_vector[action_idx] = 1
                parsed_actions.append(action_vector)
        parsed_actions = torch.stack(parsed_actions, dim=0).reshape(query_logprobs.shape)

        return value, query_logprobs, parsed_actions


class MLPBase(nn.Module):
    def __init__(self, num_inputs, dictionary_size, hidden_size=64):
        super().__init__()
        self._hidden_size = hidden_size
        self.gru = nn.GRU(num_inputs, hidden_size)

        self.end_of_line = dictionary_size
        self.actor = AutoregressiveActor(hidden_size, hidden_size, dictionary_size)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.train()

    def forward(self, inputs):
        collect = []
        for x in inputs:
            assert x.ndim == 2
            _, rnn_hxs = self.gru(x.unsqueeze(1), None)  # (seq_len, batch, input_size)
            collect.append(rnn_hxs.squeeze(0))
        rnn_hxs = torch.cat(collect)
        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension
        # TODO fix this with logprop
        value = self.critic(rnn_hxs)
        query = []
        query_logprobs = []

        for _ in range(8):
            word_logprobs, rnn_hxs = self.actor(rnn_hxs)
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
